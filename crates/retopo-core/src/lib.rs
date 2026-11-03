//! Top-level engine entry points.

use std::time::Instant;

use thiserror::Error;

use crate::simplify::{simplify_to_target, SimplificationReport};
use crate::weld::{weld_vertices, WeldReport};

pub mod cleanup;
pub mod crease;
pub mod extract;
pub mod guidance;
mod hierarchy;
mod simplify;
pub mod subdivide;
pub mod symmetrize;
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
    #[error("the orientation/position field solve diverged (non-finite values); the input may be degenerate or contain opposing face orientations")]
    FieldDiverged,
    #[error("mesh has an open boundary; extraction currently supports closed meshes only")]
    UnsupportedBoundary,
    #[error("mesh contains non-manifold edges (more than two faces sharing an edge)")]
    NonManifoldInput,
    #[error("quad extraction produced an invalid mesh: {0}")]
    ExtractionInvalid(String),
    #[error("target face count is too high for this input: the refined working mesh would exceed {limit} vertices")]
    TooDense { limit: usize },
}

/// Which pipeline [`remesh`] runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RemeshMode {
    /// Field-aligned quad-dominant remeshing (orientation field → position
    /// field → extraction). Requires a closed, edge-manifold input.
    #[default]
    FieldAligned,
    /// Legacy M0 path: weld + quadric decimation, triangle output.
    Decimate,
}

#[derive(Debug, Clone, Copy)]
pub struct RemeshOptions {
    pub mode: RemeshMode,
    /// Target face count. Field-aligned: sets the lattice spacing
    /// `sqrt(area / target_faces)`; `0` picks `input_vertices / 16`.
    /// Decimate: target triangle count, `0` disables decimation.
    pub target_faces: u32,
    /// Vertex welding distance. `<= 0.0` selects an automatic value derived
    /// from the bounding-box diagonal.
    pub weld_epsilon: f64,
    /// Decimate only: edges whose adjacent face normals differ by more than
    /// this angle (degrees) are locked against collapse. `180.0` disables.
    pub lock_crease_degrees: f64,
    /// Safety cap on the refined working mesh (field-aligned mode).
    pub max_working_vertices: usize,
    /// Mirror symmetry of the field-aligned result.
    pub symmetry: Symmetry,
    /// Strength of principal-curvature alignment (0 = smoothest field only,
    /// 1 = default). Makes edge loops wrap around limbs and tubes.
    pub curvature_alignment: f64,
    /// Curvature-adaptive density (0 = uniform quads, 1 = strongest): flat
    /// regions get larger faces, tight features (fingers, ears, joints)
    /// smaller ones, at the same total face budget.
    pub adaptivity: f64,
    /// Input edges whose faces meet at more than this angle (degrees) are
    /// creases: edge loops follow them exactly (thin rims, hard edges).
    /// `0` or `>= 180` disables.
    pub crease_degrees: f64,
    /// All-quad output: extract at twice the edge length, then split every
    /// face into quads once (triangles become three quads around a
    /// valence-3 pole).
    pub quads_only: bool,
}

/// Mirror-symmetry request. Planes pass through the surface's
/// area-weighted centroid and are perpendicular to a local axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Symmetry {
    #[default]
    None,
    /// Use the axis the input is most symmetric about, if any.
    Auto,
    X,
    Y,
    Z,
}

impl Default for RemeshOptions {
    fn default() -> Self {
        RemeshOptions {
            mode: RemeshMode::FieldAligned,
            target_faces: 0,
            weld_epsilon: 0.0,
            lock_crease_degrees: 180.0,
            max_working_vertices: 4_000_000,
            symmetry: Symmetry::None,
            curvature_alignment: 1.0,
            adaptivity: 0.6,
            crease_degrees: 40.0,
            quads_only: false,
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
    /// Field-aligned output composition.
    pub quads: u64,
    pub triangles: u64,
    pub ngons: u64,
    /// Hole loops that could not be closed (always 0 on success).
    pub dropped_cycles: u64,
    /// Euler characteristic of the cleaned input and of the output.
    pub euler_input: i64,
    pub euler_output: i64,
    /// Output vertices by valence: 3, 5, and anything else ≠ 4.
    pub poles_valence3: u64,
    pub poles_valence5: u64,
    pub poles_other: u64,
    /// Mean distance from each mirrored (x → −x) output vertex to the
    /// nearest output vertex, in units of the target edge length. ~0 for a
    /// perfectly X-symmetric result; only meaningful for symmetric input.
    pub symmetry_deviation: f64,
    /// Mirror plane used: 0 = none, 1 = X, 2 = Y, 3 = Z.
    pub symmetry_axis: u32,
    /// Target edge length used by the field solve (0 in decimate mode).
    pub edge_length: f64,
    /// Vertices added by the pre-solve refinement.
    pub subdivision_vertices_added: u64,
    /// Position-field/extraction attempts used (1 = first try; retries use
    /// slightly different scales after a rejected extraction; 0 in decimate).
    pub extraction_attempts: u64,
    pub duration_seconds: f64,
}

#[derive(Debug)]
pub struct RemeshOutput {
    /// Interleaved xyz triplets, `f32` for direct hand-off to Blender.
    pub positions: Vec<f32>,
    /// Vertex count of each face (3 or 4 today).
    pub face_sizes: Vec<u32>,
    /// Concatenated face vertex indices, `face_sizes` gives the split.
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

    if options.mode == RemeshMode::FieldAligned {
        return field_aligned(
            verts,
            tris,
            options,
            progress,
            start,
            input_vertices,
            input_faces,
            merged,
            degenerate_removed,
            duplicates_removed,
        );
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
        face_sizes: vec![3; tris.len()],
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
            triangles: output_faces,
            euler_input: field::euler_characteristic(verts.len(), &tris),
            euler_output: field::euler_characteristic(verts.len(), &tris),
            duration_seconds: start.elapsed().as_secs_f64(),
            ..RemeshStats::default()
        },
    })
}

fn report(progress: &mut Option<&mut dyn FnMut(f32) -> bool>, t: f32) -> Result<(), RetopoError> {
    if let Some(p) = progress.as_mut() {
        if p(t) {
            return Err(RetopoError::Cancelled);
        }
    }
    Ok(())
}

/// Field-aligned pipeline on an already welded/cleaned mesh.
#[allow(clippy::too_many_arguments)]
fn field_aligned(
    mut verts: Vec<[f64; 3]>,
    mut tris: Vec<[u32; 3]>,
    options: &RemeshOptions,
    mut progress: Option<&mut dyn FnMut(f32) -> bool>,
    start: Instant,
    input_vertices: u64,
    input_faces: u64,
    merged: usize,
    degenerate_removed: u64,
    duplicates_removed: u64,
) -> Result<RemeshOutput, RetopoError> {
    // fail fast, before any expensive work, on inputs extraction rejects
    extract::check_closed_manifold(&tris)?;
    let euler_input = field::euler_characteristic(verts.len(), &tris);

    let area = field::surface_area(&verts, &tris);
    let target = if options.target_faces > 0 {
        options.target_faces
    } else {
        ((verts.len() / 16).max(1)) as u32
    };
    let scale = (area / target as f64).sqrt();
    if !scale.is_finite() || scale <= 0.0 {
        return Err(RetopoError::AllFacesRemoved);
    }

    // ---- refine so the input can carry the lattice -----------------------
    // symmetry and curvature are estimated on the cleaned input, before
    // refinement adds flat-facet vertices
    let base = hierarchy::level_from_mesh(&verts, &tris);
    let mirror = mirror_plane(&base, options.symmetry);
    let before = verts.len();
    let mut crease_edges = if options.crease_degrees > 0.0 && options.crease_degrees < 180.0 {
        crease::crease_edges(&verts, &tris, options.crease_degrees, 1.5 * scale)
    } else {
        Default::default()
    };
    let crease_corners = crease::corners(&verts, &crease_edges, scale, 40.0);
    let avg = subdivide::average_edge_length(&verts, &tris);
    let adaptivity = if options.adaptivity.is_finite() {
        options.adaptivity.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let use_curvature = options.curvature_alignment > 0.0;
    let mut tensors = (use_curvature || adaptivity > 0.0)
        .then(|| guidance::curvature_tensors(&base, guidance::smoothing_for(&base, scale)));
    // per-vertex target edge length (uniform when adaptivity is 0)
    let mut sizes = match &tensors {
        Some(t) if adaptivity > 0.0 => {
            guidance::adaptive_scales(&base, t, target as f64, adaptivity)
        }
        _ => vec![scale; verts.len()],
    };
    let mut limits: Vec<f64> = sizes.iter().map(|h| (h * 0.5).min(avg * 2.0)).collect();
    let mut parents = Vec::new();
    subdivide::subdivide_adaptive(
        &mut verts,
        &mut tris,
        &mut limits,
        options.max_working_vertices,
        &mut parents,
    )
    .ok_or(RetopoError::TooDense {
        limit: options.max_working_vertices,
    })?;
    drop(limits);
    // carry the coarse estimates onto refinement vertices (edge midpoints)
    for &[a, b] in &parents {
        sizes.push(0.5 * (sizes[a as usize] + sizes[b as usize]));
        if let Some(t) = tensors.as_mut() {
            let m = (t[a as usize] + t[b as usize]) * 0.5;
            t.push(m);
        }
    }
    crease::refine(&mut crease_edges, &parents, before);
    let crease_tangents = crease::tangents(&verts, &crease_edges);
    let creases = (!crease_edges.is_empty()).then_some(crease_tangents.as_slice());
    let subdivision_vertices_added = (verts.len() - before) as u64;
    report(&mut progress, 0.30)?;

    // ---- fields ----------------------------------------------------------
    let curvature = if use_curvature { tensors } else { None };
    drop(base);
    let fopts = field::FieldOptions {
        curvature_weight: options.curvature_alignment.max(0.0),
        guidance_scale: scale,
        mirror,
        ..field::FieldOptions::default()
    };
    let (orient, _) =
        field::solve_orientation_guided(&verts, &tris, &fopts, curvature.as_deref(), creases)?;
    report(&mut progress, 0.55)?;
    // The orientation field is scale-independent; the position field and
    // extraction are not. At some lattice phases extraction tunnels through
    // creases and the validator rejects the result (Euler/manifold check).
    // Retry a few nearby scales deterministically before failing loudly.
    const RELAX_ITERATIONS: usize = 6;
    const SCALE_STEPS: [f64; 5] = [1.0, 0.95, 1.05, 0.9, 1.1];
    let mut attempt_error = None;
    let mut solved = None;
    let mut fallback = None;
    let mut mirror_image: Option<Vec<u32>> = None;
    for (attempt, m) in SCALE_STEPS.iter().enumerate() {
        let coarse = if options.quads_only { 2.0 } else { 1.0 };
        let scaled: Vec<f64> = sizes.iter().map(|h| h * m * coarse).collect();
        let pf = field::solve_position_adaptive(&verts, &tris, &orient, &scaled, &fopts, creases)?;
        report(&mut progress, 0.80)?; // also lets the host cancel between retries
        match extract::extract_mesh_adaptive(
            &verts,
            &tris,
            &orient.directions,
            &orient.normals,
            &pf.anchors,
            &pf.scales,
        ) {
            Ok(mut mesh) => {
                cleanup::merge_triangles(&mesh.positions, &mut mesh.faces, None);
                // walk lone triangles along quad strips into pairs; the
                // mirror seam is left alone so it stays a clean loop
                let near_seam = |v: u32| {
                    mirror.is_some_and(|m| {
                        (mesh.positions[v as usize][m.axis] as f64 - m.offset).abs()
                            < 0.3 * pf.scale
                    })
                };
                let positions = mesh.positions.clone();
                cleanup::pair_triangles(&positions, &mut mesh.faces, &near_seam, 8, 55.0);
                if cleanup::dissolve_doublets(&mut mesh.faces, mesh.positions.len()) > 0 {
                    cleanup::compact(&mut mesh.positions, &mut mesh.faces);
                }
                mesh.stats.quads = mesh.faces.iter().filter(|f| f.len() == 4).count() as u64;
                mesh.stats.triangles = mesh.faces.iter().filter(|f| f.len() == 3).count() as u64;
                let Some(plane) = mirror else {
                    solved = Some((pf, mesh, attempt as u32 + 1));
                    break;
                };
                // exact symmetry: mirror one half across the seam loop;
                // otherwise keep the (nearly symmetric) mesh as a fallback
                // and try the next scale for a clean seam
                let poly = symmetrize::PolyMesh {
                    positions: std::mem::take(&mut mesh.positions),
                    faces: std::mem::take(&mut mesh.faces),
                };
                match symmetrize::symmetrize(&poly, plane, mesh.stats.euler_output) {
                    Some(sym) => {
                        mesh.positions = sym.mesh.positions;
                        mesh.faces = sym.mesh.faces;
                        mirror_image = Some(sym.image);
                        mesh.stats.output_vertices = mesh.positions.len() as u64;
                        mesh.stats.quads =
                            mesh.faces.iter().filter(|f| f.len() == 4).count() as u64;
                        mesh.stats.triangles =
                            mesh.faces.iter().filter(|f| f.len() == 3).count() as u64;
                        solved = Some((pf, mesh, attempt as u32 + 1));
                        break;
                    }
                    None => {
                        mesh.positions = poly.positions;
                        mesh.faces = poly.faces;
                        if fallback.is_none() {
                            fallback = Some((pf, mesh, attempt as u32 + 1));
                        }
                    }
                }
            }
            Err(e @ RetopoError::ExtractionInvalid(_)) => attempt_error = Some(e),
            Err(e) => return Err(e),
        }
    }
    let Some((pf, mut mesh, extraction_attempts)) = solved.or(fallback) else {
        return Err(attempt_error.expect("at least one attempt ran"));
    };
    if options.quads_only {
        cleanup::subdivide_quads(&mut mesh.positions, &mut mesh.faces, mirror_image.as_mut());
        mesh.stats.quads = mesh.faces.len() as u64;
        mesh.stats.triangles = 0;
    }
    // ---- polish: even out the spacing, staying on the input surface -----
    {
        let surface = cleanup::SurfaceIndex::new(&verts, &tris, 2.0 * pf.scale);
        cleanup::relax(
            &mut mesh.positions,
            &mesh.faces,
            &surface,
            &cleanup::CreaseIndex::new(crease::segments(&verts, &crease_edges), 2.0 * pf.scale),
            &crease_corners,
            mirror,
            RELAX_ITERATIONS,
        );
        if let (Some(plane), Some(image)) = (mirror, &mirror_image) {
            symmetrize::enforce(&mut mesh.positions, image, plane);
        }
        let topo = extract::validate_closed_manifold(mesh.positions.len(), &mesh.faces)
            .map_err(RetopoError::ExtractionInvalid)?;
        if topo.euler != mesh.stats.euler_output
            || mesh.positions.iter().flatten().any(|c| !c.is_finite())
        {
            return Err(RetopoError::ExtractionInvalid(
                "polish broke the mesh".into(),
            ));
        }
    }
    report(&mut progress, 0.97)?;

    let es = &mesh.stats;
    let mut valence = vec![0u32; mesh.positions.len()];
    let mut face_sizes = Vec::with_capacity(mesh.faces.len());
    let mut indices = Vec::with_capacity(mesh.faces.len() * 4);
    for f in &mesh.faces {
        face_sizes.push(f.len() as u32);
        for &v in f {
            valence[v as usize] += 1;
            indices.push(v);
        }
    }
    let ngons = mesh.faces.iter().filter(|f| f.len() > 4).count() as u64;
    let positions: Vec<f32> = mesh.positions.iter().flatten().copied().collect();
    // quads-only output is one subdivision finer than the lattice
    let out_scale = if options.quads_only {
        0.5 * pf.scale
    } else {
        pf.scale
    };
    let symmetry_deviation = symmetry_deviation(
        &mesh.positions,
        out_scale,
        mirror.unwrap_or(guidance::MirrorPlane {
            axis: 0,
            offset: 0.0,
        }),
    );

    if let Some(p) = progress.as_mut() {
        let _ = p(1.0);
    }
    Ok(RemeshOutput {
        stats: RemeshStats {
            input_vertices,
            input_faces,
            welded_vertices_removed: merged as u64,
            degenerate_faces_removed: degenerate_removed,
            duplicate_faces_removed: duplicates_removed,
            target_reached: true,
            output_vertices: mesh.positions.len() as u64,
            output_faces: mesh.faces.len() as u64,
            quads: es.quads,
            triangles: es.triangles,
            ngons,
            dropped_cycles: es.dropped_cycles,
            euler_input,
            euler_output: es.euler_output,
            poles_valence3: valence.iter().filter(|&&k| k == 3).count() as u64,
            poles_valence5: valence.iter().filter(|&&k| k == 5).count() as u64,
            poles_other: valence
                .iter()
                .filter(|&&k| k != 3 && k != 4 && k != 5)
                .count() as u64,
            symmetry_deviation,
            symmetry_axis: mirror.map_or(0, |m| m.axis as u32 + 1),
            edge_length: out_scale,
            subdivision_vertices_added,
            extraction_attempts: extraction_attempts as u64,
            duration_seconds: start.elapsed().as_secs_f64(),
            ..RemeshStats::default()
        },
        positions,
        face_sizes,
        indices,
    })
}

/// Mean nearest-neighbour distance between the mirrored (x → −x) vertex set
/// and the original set, divided by `scale`. Uses a uniform hash grid with
/// cell size `scale`, searching the 27 neighbouring cells (distances beyond
/// one cell are clamped to 1.5 cells).
/// Resolve the requested symmetry to a plane through the area centroid.
fn mirror_plane(
    base: &hierarchy::HierarchyLevel,
    symmetry: Symmetry,
) -> Option<guidance::MirrorPlane> {
    let axis = match symmetry {
        Symmetry::None => return None,
        Symmetry::Auto => return guidance::detect_mirror(base),
        Symmetry::X => 0,
        Symmetry::Y => 1,
        Symmetry::Z => 2,
    };
    let total: f64 = base.areas.iter().sum();
    let offset = base
        .positions
        .iter()
        .zip(&base.areas)
        .map(|(p, a)| p[axis] * a)
        .sum::<f64>()
        / total.max(1e-300);
    Some(guidance::MirrorPlane { axis, offset })
}

fn symmetry_deviation(points: &[[f32; 3]], scale: f64, plane: guidance::MirrorPlane) -> f64 {
    if points.is_empty() || scale <= 0.0 {
        return 0.0;
    }
    let cell = |p: [f64; 3]| -> (i64, i64, i64) {
        (
            (p[0] / scale).floor() as i64,
            (p[1] / scale).floor() as i64,
            (p[2] / scale).floor() as i64,
        )
    };
    let pts: Vec<[f64; 3]> = points
        .iter()
        .map(|p| [p[0] as f64, p[1] as f64, p[2] as f64])
        .collect();
    let mut grid: std::collections::HashMap<(i64, i64, i64), Vec<usize>> =
        std::collections::HashMap::new();
    for (i, &p) in pts.iter().enumerate() {
        grid.entry(cell(p)).or_default().push(i);
    }
    let mut total = 0.0;
    for p in &pts {
        let m = plane.point(*p);
        let c = cell(m);
        let mut best = 1.5 * scale;
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(list) = grid.get(&(c.0 + dx, c.1 + dy, c.2 + dz)) {
                        for &j in list {
                            let q = pts[j];
                            let d = ((q[0] - m[0]).powi(2)
                                + (q[1] - m[1]).powi(2)
                                + (q[2] - m[2]).powi(2))
                            .sqrt();
                            best = best.min(d);
                        }
                    }
                }
            }
        }
        total += best;
    }
    total / pts.len() as f64 / scale
}
