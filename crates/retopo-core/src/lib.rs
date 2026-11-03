//! Top-level engine entry points.

use std::time::Instant;

use thiserror::Error;

use crate::simplify::{simplify_to_target, SimplificationReport};
use crate::weld::{weld_vertices, WeldReport};

mod hierarchy;
mod simplify;
mod weld;

pub mod field;
#[doc(hidden)]
pub mod obj;

#[derive(Debug, Error)]
pub enum RetopoError {
    #[error("input positions buffer has {0} floats, expected a multiple of 3")]
    BadPositionLength(usize),
    #[error("face index buffer has {0} indices, expected a multiple of 3")]
    BadIndexLength(usize),
    #[error("face index {index} out of range ({count} vertices)")]
    IndexOutOfRange { index: u32, count: usize },
    #[error("position {index} is not finite")]
    NonFinitePosition { index: usize },
    #[error("input mesh is empty")]
    EmptyInput,
    #[error("mesh contains no valid faces after cleaning")]
    AllFacesRemoved,
    #[error("cancelled by the user")]
    Cancelled,
}

#[derive(Debug, Clone, Copy)]
pub struct RemeshOptions {
    /// Target face count for the decimation pass. `0` disables decimation.
    /// Note: this is a decimator, not yet the field-aligned remesher — the
    /// quad extraction pass lands in M3.
    pub target_faces: u32,
    /// Vertex welding distance. `<= 0.0` selects an automatic value derived
    /// from the bounding-box diagonal.
    pub weld_epsilon: f64,
    /// Edges whose adjacent face normals differ by more than this angle
    /// (degrees) are locked against collapse. `180.0` disables crease locking.
    pub lock_crease_degrees: f64,
}

impl Default for RemeshOptions {
    fn default() -> Self {
        RemeshOptions {
            target_faces: 0,
            weld_epsilon: 0.0,
            lock_crease_degrees: 180.0,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct RemeshStats {
    pub input_vertices: u64,
    pub input_faces: u64,
    pub welded_vertices_removed: u64,
    pub degenerate_faces_removed: u64,
    pub duplicate_faces_removed: u64,
    /// Faces removed by the decimation pass (0 when disabled).
    pub decimated_faces_removed: u64,
    /// Number of successful collapse operations.
    pub collapses: u64,
    /// `false` when the decimation could not reach the requested face
    /// count (boundary/crease locked edges block further collapses).
    pub target_reached: bool,
    pub output_vertices: u64,
    pub output_faces: u64,
    pub duration_seconds: f64,
}

#[derive(Debug)]
pub struct RemeshOutput {
    /// Interleaved xyz triplets, `f32` for direct hand-off to Blender.
    pub positions: Vec<f32>,
    /// One triangle per face, `u32` indices.
    pub indices: Vec<u32>,
    pub stats: RemeshStats,
}

/// Validate raw input, clean it (weld + degenerate removal) and optionally
/// decimate towards `target_faces`.
///
/// `progress` receives values in `0.0..=1.0`; returning `true` from the
/// callback cancels the operation with [`RetopoError::Cancelled`].
pub fn remesh(
    positions: &[f32],
    indices: &[u32],
    options: &RemeshOptions,
    mut progress: Option<&mut dyn FnMut(f32) -> bool>,
) -> Result<RemeshOutput, RetopoError> {
    let start = Instant::now();

    if !positions.len().is_multiple_of(3) {
        return Err(RetopoError::BadPositionLength(positions.len()));
    }
    if !indices.len().is_multiple_of(3) {
        return Err(RetopoError::BadIndexLength(indices.len()));
    }
    let vertex_count = positions.len() / 3;
    if vertex_count == 0 || indices.is_empty() {
        return Err(RetopoError::EmptyInput);
    }
    for (i, v) in positions.iter().enumerate() {
        if !v.is_finite() {
            return Err(RetopoError::NonFinitePosition { index: i });
        }
    }
    for &idx in indices {
        if idx as usize >= vertex_count {
            return Err(RetopoError::IndexOutOfRange {
                index: idx,
                count: vertex_count,
            });
        }
    }

    let input_vertices = vertex_count as u64;
    let input_faces = (indices.len() / 3) as u64;

    let mut verts: Vec<[f64; 3]> = positions
        .as_chunks::<3>()
        .0
        .iter()
        .map(|&[x, y, z]| [x as f64, y as f64, z as f64])
        .collect();
    let mut tris: Vec<[u32; 3]> = indices.as_chunks::<3>().0.to_vec();

    if let Some(p) = progress.as_mut() {
        if p(0.02) {
            return Err(RetopoError::Cancelled);
        }
    }

    // ---- bounding box + automatic welding distance -----------------------
    let (mut mn, mut mx) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
    for p in &verts {
        for k in 0..3 {
            mn[k] = mn[k].min(p[k]);
            mx[k] = mx[k].max(p[k]);
        }
    }
    let diag = (0..3).map(|k| (mx[k] - mn[k]).powi(2)).sum::<f64>().sqrt();
    let eps = if options.weld_epsilon > 0.0 {
        options.weld_epsilon
    } else {
        (diag * 1e-6).max(1e-12)
    };

    // ---- weld + degenerate face removal ----------------------------------
    let WeldReport {
        merged,
        degenerate_removed,
        duplicates_removed,
    } = weld_vertices(&mut verts, &mut tris, eps, diag);
    if tris.is_empty() {
        return Err(RetopoError::AllFacesRemoved);
    }
    if let Some(p) = progress.as_mut() {
        if p(0.25) {
            return Err(RetopoError::Cancelled);
        }
    }

    // ---- optional decimation ----------------------------------------------
    let mut decimated_faces_removed = 0u64;
    let mut collapses = 0u64;
    let mut target_reached = true;
    if options.target_faces > 0 && (tris.len() as u64) > options.target_faces as u64 {
        let report: SimplificationReport = simplify_to_target(
            &mut verts,
            &mut tris,
            options.target_faces as usize,
            options.lock_crease_degrees,
            |t| {
                progress
                    .as_deref_mut()
                    .map(|p| p(0.25 + 0.70 * t))
                    .unwrap_or(false)
            },
        )?;
        decimated_faces_removed = report.faces_removed as u64;
        collapses = report.collapses as u64;
        target_reached = report.reached_target;
    }

    if let Some(p) = progress.as_mut() {
        let _ = p(1.0);
    }

    // ---- compact: drop vertices no longer referenced by any face ---------
    let mut remap = vec![u32::MAX; verts.len()];
    let mut next = 0u32;
    for t in &tris {
        for &c in t {
            if remap[c as usize] == u32::MAX {
                remap[c as usize] = next;
                next += 1;
            }
        }
    }
    if (next as usize) != verts.len() {
        let mut compacted = vec![[0.0f64; 3]; next as usize];
        for (i, &r) in remap.iter().enumerate() {
            if r != u32::MAX {
                compacted[r as usize] = verts[i];
            }
        }
        for t in tris.iter_mut() {
            for c in t.iter_mut() {
                *c = remap[*c as usize];
            }
        }
        verts = compacted;
    }

    let output_vertices = verts.len() as u64;
    let output_faces = tris.len() as u64;

    let mut out_positions = Vec::with_capacity(verts.len() * 3);
    for p in &verts {
        out_positions.push(p[0] as f32);
        out_positions.push(p[1] as f32);
        out_positions.push(p[2] as f32);
    }
    let mut out_indices = Vec::with_capacity(tris.len() * 3);
    for t in &tris {
        out_indices.extend_from_slice(t);
    }

    Ok(RemeshOutput {
        positions: out_positions,
        indices: out_indices,
        stats: RemeshStats {
            input_vertices,
            input_faces,
            welded_vertices_removed: merged as u64,
            degenerate_faces_removed: degenerate_removed,
            duplicate_faces_removed: duplicates_removed,
            decimated_faces_removed,
            collapses,
            target_reached,
            output_vertices,
            output_faces,
            duration_seconds: start.elapsed().as_secs_f64(),
        },
    })
}
