//! C ABI for the OpenRetopo engine.
//!
//! Design goals:
//! * Stable, versioned interface — one native library per platform works for
//!   every supported Blender (4.2+), no Python-ABI coupling.
//! * Errors travel in a fixed-size message buffer, never across the FFI
//!   as unwinding (panics are caught at the boundary).
//! * Memory returned to the caller is freed with the matching
//!   `openretopo_free_*` functions.
//!
//! The Python side (`addon/__init__.py`) maps these types 1:1 with ctypes.

use std::os::raw::c_char;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;

use openretopo_core::{remesh, RemeshOptions, RetopoError};

/// Bump when the ABI layout changes. Python asserts this on load.
pub const ABI_VERSION: u32 = 1;
pub const ERR_BUF_LEN: usize = 512;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct OpenRetopoParams {
    /// `sizeof(OpenRetopoParams)` as seen by the caller. Must be >= 40;
    /// larger values (newer addons) are tolerated and the tail is ignored.
    pub struct_size: u32,
    /// Target face count for the decimation pass. `0` disables it.
    pub target_faces: u32,
    /// Vertex welding epsilon. `<= 0.0` selects an automatic value.
    pub weld_epsilon: f32,
    /// Edges whose adjacent face normals exceed this angle are locked
    /// against collapse. `180.0` disables crease locking.
    pub lock_crease_degrees: f32,
    /// Reserved for symmetry (0 = none, 1 = X, 2 = Y, 3 = Z). Unused today.
    pub symmetry_axis: u32,
    /// Bitfield of reserved feature flags.
    pub flags: u32,
    pub reserved: [u64; 2],
}

impl Default for OpenRetopoParams {
    fn default() -> Self {
        OpenRetopoParams {
            struct_size: std::mem::size_of::<OpenRetopoParams>() as u32,
            target_faces: 0,
            weld_epsilon: 0.0,
            lock_crease_degrees: 180.0,
            symmetry_axis: 0,
            flags: 0,
            reserved: [0; 2],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct OpenRetopoStats {
    pub input_vertices: u64,
    pub input_faces: u64,
    pub welded_vertices_removed: u64,
    pub degenerate_faces_removed: u64,
    pub duplicate_faces_removed: u64,
    pub decimated_faces_removed: u64,
    pub collapses: u64,
    /// 1 when the requested face count was reached, 0 otherwise (e.g. locked edges).
    pub target_reached: u64,
    pub output_vertices: u64,
    pub output_faces: u64,
    pub duration_seconds: f64,
    pub reserved: [u64; 3],
}

pub const OK: i32 = 0;
pub const ERR_INVALID_ARG: i32 = 1;
pub const ERR_CANCELLED: i32 = 2;
pub const ERR_INTERNAL: i32 = 3;

#[no_mangle]
pub extern "C" fn openretopo_abi_version() -> u32 {
    ABI_VERSION
}

/// Retopologize a triangle mesh.
///
/// # Safety
/// * `positions` must point to `positions_len` contiguous f32s (3 per vertex).
/// * `indices` must point to `indices_len` contiguous u32s (3 per triangle).
/// * `params` must be readable for `struct_size` bytes; if null, defaults are
///   used.
/// * `progress`, if non-null, is written with values in `0.0..=1.0`.
/// * `cancel`, if non-null, is read; setting it to non-zero aborts the run.
/// * `out_positions`/`out_indices` (and their length out-params) must be
///   valid writable pointers.
/// * `err_buf`, if non-null, must point to at least `ERR_BUF_LEN` writable
///   bytes.
/// * On success the caller owns the returned buffers and must release them
///   with [`openretopo_free_positions`] / [`openretopo_free_indices`].
#[no_mangle]
pub unsafe extern "C" fn openretopo_remesh(
    positions: *const f32,
    positions_len: u64,
    indices: *const u32,
    indices_len: u64,
    params: *const OpenRetopoParams,
    progress: *mut f32,
    cancel: *mut i32,
    out_positions: *mut *mut f32,
    out_positions_len: *mut u64,
    out_indices: *mut *mut u32,
    out_indices_len: *mut u64,
    out_stats: *mut OpenRetopoStats,
    err_buf: *mut c_char,
    err_buf_len: u64,
) -> i32 {
    let result = catch_unwind(AssertUnwindSafe(|| {
        remesh_impl(
            positions,
            positions_len,
            indices,
            indices_len,
            params,
            progress,
            cancel,
            out_positions,
            out_positions_len,
            out_indices,
            out_indices_len,
            out_stats,
            err_buf,
            err_buf_len,
        )
    }));
    match result {
        Ok(code) => code,
        Err(_) => {
            write_err(
                err_buf,
                err_buf_len,
                "internal panic in the retopology engine",
            );
            ERR_INTERNAL
        }
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn remesh_impl(
    positions: *const f32,
    positions_len: u64,
    indices: *const u32,
    indices_len: u64,
    params: *const OpenRetopoParams,
    progress: *mut f32,
    cancel: *mut i32,
    out_positions: *mut *mut f32,
    out_positions_len: *mut u64,
    out_indices: *mut *mut u32,
    out_indices_len: *mut u64,
    out_stats: *mut OpenRetopoStats,
    err_buf: *mut c_char,
    err_buf_len: u64,
) -> i32 {
    if positions.is_null()
        || indices.is_null()
        || out_positions.is_null()
        || out_positions_len.is_null()
        || out_indices.is_null()
        || out_indices_len.is_null()
    {
        write_err(err_buf, err_buf_len, "null pointer argument");
        return ERR_INVALID_ARG;
    }
    if positions_len > u64::MAX / 4 || indices_len > u64::MAX / 4 {
        write_err(err_buf, err_buf_len, "buffer length out of range");
        return ERR_INVALID_ARG;
    }

    let pos_slice = unsafe { std::slice::from_raw_parts(positions, positions_len as usize) };
    let idx_slice = unsafe { std::slice::from_raw_parts(indices, indices_len as usize) };

    let params_value = if params.is_null() {
        OpenRetopoParams::default()
    } else {
        let declared = unsafe { (*params).struct_size } as u64;
        if declared < std::mem::size_of::<OpenRetopoParams>() as u64 {
            // Caller's struct is smaller than our v1 layout — reading the
            // remaining fields would be out-of-bounds.
            write_err(
                err_buf,
                err_buf_len,
                "params struct_size too small for ABI v1",
            );
            return ERR_INVALID_ARG;
        }
        // Read only the v1 fields we know about; the rest is ignored on
        // purpose so newer addons stay compatible with older libraries.
        unsafe { *params }
    };

    let options = RemeshOptions {
        target_faces: params_value.target_faces,
        weld_epsilon: params_value.weld_epsilon as f64,
        lock_crease_degrees: params_value.lock_crease_degrees as f64,
    };

    // Progress/cancel trampolines bridging raw pointers into the core.
    let progress_ptr = progress;
    let cancel_ptr = cancel;
    let mut progress_cb = |t: f32| -> bool {
        if !progress_ptr.is_null() {
            unsafe { progress_ptr.write_volatile(t) };
        }
        if !cancel_ptr.is_null() {
            let requested = unsafe { cancel_ptr.read_volatile() };
            requested != 0
        } else {
            false
        }
    };

    match remesh(pos_slice, idx_slice, &options, Some(&mut progress_cb)) {
        Ok(output) => {
            let (pos_ptr, pos_len) = leak_boxed_slice(output.positions);
            let (idx_ptr, idx_len) = leak_boxed_slice(output.indices);
            unsafe {
                *out_positions = pos_ptr;
                *out_positions_len = pos_len as u64;
                *out_indices = idx_ptr;
                *out_indices_len = idx_len as u64;
                if !out_stats.is_null() {
                    *out_stats = OpenRetopoStats {
                        input_vertices: output.stats.input_vertices,
                        input_faces: output.stats.input_faces,
                        welded_vertices_removed: output.stats.welded_vertices_removed,
                        degenerate_faces_removed: output.stats.degenerate_faces_removed,
                        duplicate_faces_removed: output.stats.duplicate_faces_removed,
                        decimated_faces_removed: output.stats.decimated_faces_removed,
                        collapses: output.stats.collapses,
                        target_reached: output.stats.target_reached as u64,
                        output_vertices: output.stats.output_vertices,
                        output_faces: output.stats.output_faces,
                        duration_seconds: output.stats.duration_seconds,
                        reserved: [0; 3],
                    };
                }
            }
            OK
        }
        Err(RetopoError::Cancelled) => {
            write_err(err_buf, err_buf_len, "cancelled");
            ERR_CANCELLED
        }
        Err(e) => {
            write_err(err_buf, err_buf_len, &e.to_string());
            ERR_INTERNAL
        }
    }
}

fn leak_boxed_slice<T>(v: Vec<T>) -> (*mut T, usize) {
    let b = v.into_boxed_slice();
    let len = b.len();
    (Box::into_raw(b) as *mut T, len)
}

/// # Safety
/// `ptr` and `len` must come from a previous call to [`openretopo_remesh`].
#[no_mangle]
pub unsafe extern "C" fn openretopo_free_positions(ptr: *mut f32, len: u64) {
    if ptr.is_null() {
        return;
    }
    let slice = std::slice::from_raw_parts_mut(ptr, len as usize);
    drop(Box::from_raw(slice));
}

/// # Safety
/// `ptr` and `len` must come from a previous call to [`openretopo_remesh`].
#[no_mangle]
pub unsafe extern "C" fn openretopo_free_indices(ptr: *mut u32, len: u64) {
    if ptr.is_null() {
        return;
    }
    let slice = std::slice::from_raw_parts_mut(ptr, len as usize);
    drop(Box::from_raw(slice));
}

fn write_err(err_buf: *mut c_char, err_buf_len: u64, msg: &str) {
    if err_buf.is_null() || err_buf_len == 0 {
        return;
    }
    let capacity = err_buf_len as usize;
    let bytes = msg.as_bytes();
    let copy_len = bytes.len().min(capacity.saturating_sub(1));
    unsafe {
        ptr::copy_nonoverlapping(bytes.as_ptr(), err_buf as *mut u8, copy_len);
        *err_buf.add(copy_len) = 0;
    }
}
