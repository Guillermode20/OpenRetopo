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

use openretopo_core::{remesh, RemeshMode, RemeshOptions, RemeshStats, RetopoError, Symmetry};

/// Bump when the ABI layout changes. Python asserts this on load.
///
/// v2: polygon output (`out_face_sizes`), field-aligned mode by default,
///     `flags` bit 0 selects the legacy decimator, stats carry
///     `struct_size` and topology/pole fields.
/// v3: `symmetry_axis` is honoured (0-4, 4 = auto), `FLAG_NO_CURVATURE`,
///     stats gain `extraction_attempts` and `symmetry_axis` (same size).
///     Bumped so a stale v2 library — which silently ignores symmetry —
///     is rejected instead of producing unsymmetric output.
/// v4: `adaptivity` (curvature-adaptive density, 0-1) takes the first
///     reserved slot of `OpenRetopoParams` (same size).
pub const ABI_VERSION: u32 = 4;

/// `OpenRetopoParams::flags` bit: run the legacy weld + decimate pipeline
/// instead of field-aligned remeshing.
pub const FLAG_DECIMATE_ONLY: u32 = 1;
/// `OpenRetopoParams::flags` bit: disable principal-curvature alignment
/// (smoothest field only).
pub const FLAG_NO_CURVATURE: u32 = 2;
pub const ERR_BUF_LEN: usize = 512;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct OpenRetopoParams {
    /// `sizeof(OpenRetopoParams)` as seen by the caller. Must be >= 40;
    /// larger values (newer addons) are tolerated and the tail is ignored.
    pub struct_size: u32,
    /// Target face count (field-aligned: lattice density, `0` = auto;
    /// decimate: triangle budget, `0` = no decimation).
    pub target_faces: u32,
    /// Vertex welding epsilon. `<= 0.0` selects an automatic value.
    pub weld_epsilon: f32,
    /// Edges whose adjacent face normals exceed this angle are locked
    /// against collapse. `180.0` disables crease locking.
    pub lock_crease_degrees: f32,
    /// Mirror symmetry (field-aligned): 0 = none, 1 = X, 2 = Y, 3 = Z
    /// (local axes, plane through the surface centroid), 4 = auto-detect.
    pub symmetry_axis: u32,
    /// Feature flags, see `FLAG_*`.
    pub flags: u32,
    /// Curvature-adaptive density, 0 (uniform) to 1 (strongest); values
    /// outside are clamped, non-finite ones rejected.
    pub adaptivity: f32,
    pub _pad: u32,
    pub reserved: [u64; 1],
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
            adaptivity: 0.0,
            _pad: 0,
            reserved: [0; 1],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct OpenRetopoStats {
    /// `sizeof(OpenRetopoStats)` as seen by the caller; the library writes
    /// at most this many bytes. Must be >= 8.
    pub struct_size: u32,
    pub _pad: u32,
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
    pub quads: u64,
    pub triangles: u64,
    pub ngons: u64,
    pub dropped_cycles: u64,
    pub euler_input: i64,
    pub euler_output: i64,
    pub poles_valence3: u64,
    pub poles_valence5: u64,
    pub poles_other: u64,
    pub subdivision_vertices_added: u64,
    pub symmetry_deviation: f64,
    pub edge_length: f64,
    pub duration_seconds: f64,
    pub extraction_attempts: u64,
    /// Mirror plane used: 0 = none, 1 = X, 2 = Y, 3 = Z.
    pub symmetry_axis: u64,
    pub reserved: [u64; 2],
}

impl OpenRetopoStats {
    fn from_core(s: &RemeshStats) -> Self {
        OpenRetopoStats {
            struct_size: std::mem::size_of::<OpenRetopoStats>() as u32,
            _pad: 0,
            input_vertices: s.input_vertices,
            input_faces: s.input_faces,
            welded_vertices_removed: s.welded_vertices_removed,
            degenerate_faces_removed: s.degenerate_faces_removed,
            duplicate_faces_removed: s.duplicate_faces_removed,
            decimated_faces_removed: s.decimated_faces_removed,
            collapses: s.collapses,
            target_reached: s.target_reached as u64,
            output_vertices: s.output_vertices,
            output_faces: s.output_faces,
            quads: s.quads,
            triangles: s.triangles,
            ngons: s.ngons,
            dropped_cycles: s.dropped_cycles,
            euler_input: s.euler_input,
            euler_output: s.euler_output,
            poles_valence3: s.poles_valence3,
            poles_valence5: s.poles_valence5,
            poles_other: s.poles_other,
            subdivision_vertices_added: s.subdivision_vertices_added,
            symmetry_deviation: s.symmetry_deviation,
            edge_length: s.edge_length,
            duration_seconds: s.duration_seconds,
            extraction_attempts: s.extraction_attempts,
            symmetry_axis: s.symmetry_axis as u64,
            reserved: [0; 2],
        }
    }
}

pub const OK: i32 = 0;
pub const ERR_INVALID_ARG: i32 = 1;
pub const ERR_CANCELLED: i32 = 2;
pub const ERR_INTERNAL: i32 = 3;
/// The input is valid but unsupported by the field-aligned path (open
/// boundary, non-manifold edges) or the requested density is too high.
pub const ERR_UNSUPPORTED_INPUT: i32 = 4;
/// The solve or extraction failed its own validation; nothing was output.
pub const ERR_REMESH_FAILED: i32 = 5;

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
/// * `out_positions`/`out_indices`/`out_face_sizes` (and their length
///   out-params) must be valid writable pointers. Faces are polygons:
///   `out_face_sizes[k]` consecutive entries of `out_indices` form face k.
/// * `out_stats`, if non-null, must be writable for its `struct_size` bytes.
/// * `err_buf`, if non-null, must point to at least `ERR_BUF_LEN` writable
///   bytes.
/// * On success the caller owns the returned buffers and must release them
///   with [`openretopo_free_positions`] / [`openretopo_free_indices`]
///   (the latter also frees `out_face_sizes`). On failure no buffers are
///   allocated and the out-params are left untouched.
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
    out_face_sizes: *mut *mut u32,
    out_face_sizes_len: *mut u64,
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
            out_face_sizes,
            out_face_sizes_len,
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
    out_face_sizes: *mut *mut u32,
    out_face_sizes_len: *mut u64,
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
        || out_face_sizes.is_null()
        || out_face_sizes_len.is_null()
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

    let stats_capacity = if out_stats.is_null() {
        0usize
    } else {
        let declared = unsafe { (*out_stats).struct_size } as usize;
        if declared < 8 {
            write_err(err_buf, err_buf_len, "stats struct_size not set");
            return ERR_INVALID_ARG;
        }
        declared.min(std::mem::size_of::<OpenRetopoStats>())
    };

    let options = RemeshOptions {
        mode: if params_value.flags & FLAG_DECIMATE_ONLY != 0 {
            RemeshMode::Decimate
        } else {
            RemeshMode::FieldAligned
        },
        target_faces: params_value.target_faces,
        weld_epsilon: params_value.weld_epsilon as f64,
        lock_crease_degrees: params_value.lock_crease_degrees as f64,
        symmetry: match params_value.symmetry_axis {
            0 => Symmetry::None,
            1 => Symmetry::X,
            2 => Symmetry::Y,
            3 => Symmetry::Z,
            4 => Symmetry::Auto,
            other => {
                write_err(
                    err_buf,
                    err_buf_len,
                    &format!("invalid symmetry_axis {other} (expected 0-4)"),
                );
                return ERR_INVALID_ARG;
            }
        },
        curvature_alignment: if params_value.flags & FLAG_NO_CURVATURE != 0 {
            0.0
        } else {
            1.0
        },
        adaptivity: if params_value.adaptivity.is_finite() {
            params_value.adaptivity.clamp(0.0, 1.0) as f64
        } else {
            write_err(err_buf, err_buf_len, "adaptivity must be finite");
            return ERR_INVALID_ARG;
        },
        ..RemeshOptions::default()
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
            let (fs_ptr, fs_len) = leak_boxed_slice(output.face_sizes);
            unsafe {
                *out_positions = pos_ptr;
                *out_positions_len = pos_len as u64;
                *out_indices = idx_ptr;
                *out_indices_len = idx_len as u64;
                *out_face_sizes = fs_ptr;
                *out_face_sizes_len = fs_len as u64;
                if stats_capacity > 0 {
                    let mut value = OpenRetopoStats::from_core(&output.stats);
                    value.struct_size = stats_capacity as u32;
                    ptr::copy_nonoverlapping(
                        &value as *const OpenRetopoStats as *const u8,
                        out_stats as *mut u8,
                        stats_capacity,
                    );
                }
            }
            OK
        }
        Err(RetopoError::Cancelled) => {
            write_err(err_buf, err_buf_len, "cancelled");
            ERR_CANCELLED
        }
        Err(
            e @ (RetopoError::UnsupportedBoundary
            | RetopoError::NonManifoldInput
            | RetopoError::TooDense { .. }),
        ) => {
            write_err(err_buf, err_buf_len, &e.to_string());
            ERR_UNSUPPORTED_INPUT
        }
        Err(e @ (RetopoError::FieldDiverged | RetopoError::ExtractionInvalid(_))) => {
            write_err(err_buf, err_buf_len, &e.to_string());
            ERR_REMESH_FAILED
        }
        Err(e) => {
            write_err(err_buf, err_buf_len, &e.to_string());
            ERR_INVALID_ARG
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
