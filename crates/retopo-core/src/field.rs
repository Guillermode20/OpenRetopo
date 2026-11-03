//! Cross-field (N-RoSy) orientation solving on the primal mesh graph.
//!
//! This module reimplements the orientation-field optimisation of
//! Jakob et al., *Instant Field-Aligned Meshes* (SIGGRAPH Asia 2015) in
//! idiomatic Rust. The algorithm is:
//!
//! 1. Each input vertex carries a tangent (unit-length, lying in the
//!    vertex's tangent plane) field direction.
//! 2. Every iteration, each vertex re-averages its one-ring neighbours'
//!    directions after snapping each to the best symmetry position
//!    relative to the current running average (see
//!    [`compat_orientation_intrinsic_4`]). The result is projected back
//!    onto the tangent plane and normalised.
//! 3. Repeating drives the field towards a low-energy state where most of
//!    the mesh shares a consistent grid direction, with singularities
//!    emerging where a smooth alignment is impossible.
//!
//! Differences from the reference implementation:
//!   * Updates are performed sequentially in a fixed order, giving fully
//!     deterministic results (no thread-related jitter).
//!   * The random initial direction is derived from a position hash, so
//!     repeated runs on the same input always produce the same field.
//!   * The multi-resolution hierarchy (coarsened primal graph, coarse-to-fine
//!     solve with parent→child projection) is implemented in
//!     [`crate::hierarchy`]; without it the local solve stagnates at poor
//!     local minima.

use nalgebra::Vector3;

use crate::guidance::{self, LevelGuide, MirrorPlane};
use crate::hierarchy::{self, Hierarchy, HierarchyLevel};

/// Per-vertex cross-field result.
#[derive(Debug, Clone)]
pub struct OrientationField {
    /// One unit-length tangent direction per input vertex.
    pub directions: Vec<Vector3<f64>>,
    /// Per-vertex outward normal used to project the field (area weighted).
    pub normals: Vec<Vector3<f64>>,
}

/// Per-face singularity classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Singularity {
    /// Field turns +90° inside the face (positive singularity).
    Positive,
    /// Field turns −90° inside the face (negative singularity).
    Negative,
}

#[derive(Debug, Clone, Default)]
pub struct FieldStats {
    pub iterations: u32,
    /// Mean edge mismatch between neighbouring vertices, in degrees
    /// (measured with the best symmetry alignment, as in the paper's error).
    pub mean_edge_error_degrees: f64,
    /// Number of singular faces (absolute index 1 or 3 mod 4).
    pub singular_faces: usize,
    /// Number of positive singular faces.
    pub singular_positive: usize,
    /// Number of negative singular faces.
    pub singular_negative: usize,
    /// Euler characteristic of the input mesh (V − E + F).
    pub euler_characteristic: i64,
    /// Number of hierarchy levels built (0 = just the input mesh).
    pub hierarchy_levels: usize,
}

/// Options controlling the solve.
#[derive(Debug, Clone, Copy)]
pub struct FieldOptions {
    /// Maximum number of global iterations at level 0 (the fine mesh).
    pub max_iterations: u32,
    /// Iterations per coarser hierarchy level.
    pub iterations_per_level: u32,
    /// Stop early when the largest per-vertex angular change in an
    /// iteration drops below this many degrees (any level).
    pub convergence_degrees: f64,
    /// rotational symmetry: 4 for quad-dominant output.
    pub rosy: u32,
    /// Solve the position field coarse-to-fine on the same hierarchy as the
    /// orientation field (reference behaviour).
    pub hierarchical_positions: bool,
    /// Pull toward principal curvature directions on anisotropic regions
    /// (0 = pure smoothness). Needs `guidance_scale`.
    pub curvature_weight: f64,
    /// Target edge length the curvature anisotropy is measured against.
    pub guidance_scale: f64,
    /// Couple both fields to their mirror image across this plane.
    pub mirror: Option<MirrorPlane>,
}

impl Default for FieldOptions {
    fn default() -> Self {
        FieldOptions {
            // Few sweeps: the lattice dynamics are weakly unstable around
            // frustrated singular regions and *diverge* with more passes.
            // The coarse->fine hierarchy supplies the global alignment.
            max_iterations: 6,
            iterations_per_level: 6,
            convergence_degrees: 0.1,
            rosy: 4,
            hierarchical_positions: true,
            curvature_weight: 0.0,
            guidance_scale: 0.0,
            mirror: None,
        }
    }
}

/// Rotate `q` (tangent to plane `source_normal`) into plane `target_normal`
/// using the minimal rotation (axis = source × target). Mirrors
/// `rotate_vector_into_plane` from the reference implementation.
fn rotate_into_plane(
    q: Vector3<f64>,
    source_normal: Vector3<f64>,
    target_normal: Vector3<f64>,
) -> Vector3<f64> {
    let cos_theta = source_normal.dot(&target_normal);
    if cos_theta >= 0.9999 {
        return q;
    }
    let axis = source_normal.cross(&target_normal);
    let axis_dot_axis = axis.dot(&axis);
    if axis_dot_axis < 1e-12 {
        return q;
    }
    let factor = (1.0 - cos_theta) / axis_dot_axis;
    q * cos_theta + axis.cross(&q) + axis * (axis.dot(&q) * factor)
}

/// Compatibility pair for intrinsic 4-RoSy averaging (eq. from the paper's
/// reference implementation): returns `(q0, q1_snapped)` where `q1` is moved
/// into plane `n0` and snapped to the closest 90° symmetry position of `q0`.
pub(crate) fn compat_orientation_intrinsic_4(
    q0: Vector3<f64>,
    n0: Vector3<f64>,
    q1_raw: Vector3<f64>,
    n1: Vector3<f64>,
) -> (Vector3<f64>, Vector3<f64>) {
    let q1 = rotate_into_plane(q1_raw, n1, n0);
    let t1 = n0.cross(&q1);
    let dp0 = q1.dot(&q0);
    let dp1 = t1.dot(&q0);

    if dp0.abs() > dp1.abs() {
        (q0, q1 * dp0.signum())
    } else {
        (q0, t1 * dp1.signum())
    }
}

/// Discrete turning-angle index between two faces sharing a directed edge.
/// Returns a value in `{0, 1, 2, 3}` counting quarter turns.
fn compat_orientation_index_4(
    q0: Vector3<f64>,
    n0: Vector3<f64>,
    q1_raw: Vector3<f64>,
    n1: Vector3<f64>,
) -> i32 {
    let q1 = rotate_into_plane(q1_raw, n1, n0);
    let dp0 = q1.dot(&q0);
    let dp1 = n0.cross(&q1).dot(&q0);

    if dp0.abs() > dp1.abs() {
        if dp0 > 0.0 {
            0
        } else {
            2
        }
    } else if dp1 > 0.0 {
        1
    } else {
        3
    }
}

/// Deterministic pseudo-random tangent direction at a vertex: hash the
/// quantised position and project an arbitrary axis onto the tangent plane.
fn initial_direction(position: [f64; 3], normal: Vector3<f64>) -> Vector3<f64> {
    let mut hash: u64 = 0xcbf29ce484222325;
    for c in position {
        for b in c.to_bits().to_le_bytes() {
            hash ^= b as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    // Two independent-ish hashes give two candidate axis directions.
    let a = hash as f64 / u64::MAX as f64 * std::f64::consts::TAU;
    let b = (hash.rotate_left(32) as f64 / u64::MAX as f64) * std::f64::consts::TAU;
    let raw = Vector3::new(a.sin() * b.cos(), a.cos() * b.sin(), (a * b).sin() + 0.3);
    let tangent = raw - normal * normal.dot(&raw);
    if tangent.norm() < 1e-8 {
        // Degenerate case: pick any axis perpendicular to the normal.
        let arbitrary = if normal.x.abs() < 0.9 {
            Vector3::new(1.0, 0.0, 0.0)
        } else {
            Vector3::new(0.0, 1.0, 0.0)
        };
        let t = arbitrary - normal * normal.dot(&arbitrary);
        return t.normalize();
    }
    tangent.normalize()
}

/// Project `q` onto the plane with normal `n` and renormalize; falls back
/// to an arbitrary in-plane axis when the projection vanishes.
fn project_unit(q: Vector3<f64>, n: Vector3<f64>) -> Vector3<f64> {
    let projected = q - n * n.dot(&q);
    if projected.norm() < 1e-12 {
        let arbitrary = if n.x.abs() < 0.9 {
            Vector3::new(1.0, 0.0, 0.0)
        } else {
            Vector3::new(0.0, 1.0, 0.0)
        };
        return (arbitrary - n * n.dot(&arbitrary)).normalize();
    }
    projected.normalize()
}

/// Greedy deterministic graph coloring: same-colored vertices never share
/// an edge. Colored sweeps break the pathological lockstep oscillation of
/// plain Gauss-Seidel on highly symmetric graphs (the reference uses the
/// same partitioning for its parallel updates).
fn color_graph(adj: &[Vec<(u32, f64)>]) -> Vec<Vec<usize>> {
    let n = adj.len();
    let mut colors = vec![usize::MAX; n];
    let mut phases: Vec<Vec<usize>> = Vec::new();
    for i in 0..n {
        let mut used = std::collections::HashSet::new();
        for &(j, _) in &adj[i] {
            let c = colors[j as usize];
            if c != usize::MAX {
                used.insert(c);
            }
        }
        let mut c = 0;
        while used.contains(&c) {
            c += 1;
        }
        colors[i] = c;
        if phases.len() <= c {
            phases.resize(c + 1, Vec::new());
        }
        phases[c].push(i);
    }
    phases
}

/// One full optimization pass over a level until convergence or the
/// iteration cap. Returns the number of iterations performed.
fn solve_level(
    level: &HierarchyLevel,
    guide: &LevelGuide,
    dirs: &mut [Vector3<f64>],
    max_iterations: u32,
    convergence_degrees: f64,
) -> Result<u32, crate::RetopoError> {
    let phases = color_graph(&level.adj);
    let mut iteration = 0u32;
    loop {
        iteration += 1;
        let mut max_step = 0.0f64;
        for phase in &phases {
            for &i in phase {
                let n_i = level.normals[i];
                if level.adj[i].is_empty() {
                    continue;
                }
                let mut sum = dirs[i];
                let mut weight_sum = 0.0f64;
                for &(j, w) in &level.adj[i] {
                    let q_j = dirs[j as usize];
                    let (v0, v1) =
                        compat_orientation_intrinsic_4(sum, n_i, q_j, level.normals[j as usize]);
                    sum = v0 * weight_sum + v1 * w;
                    sum -= n_i * n_i.dot(&sum);
                    weight_sum += w;
                    let norm = sum.norm();
                    if norm > 1e-12 {
                        sum /= norm;
                    }
                }
                let neighbour_weight = weight_sum;
                let mut extra =
                    |sum: &mut Vector3<f64>, q: Vector3<f64>, nq: Vector3<f64>, w: f64| {
                        if w <= 0.0 {
                            return;
                        }
                        let (v0, v1) = compat_orientation_intrinsic_4(*sum, n_i, q, nq);
                        *sum = v0 * weight_sum + v1 * w;
                        *sum -= n_i * n_i.dot(sum);
                        weight_sum += w;
                        let norm = sum.norm();
                        if norm > 1e-12 {
                            *sum /= norm;
                        }
                    };
                if let (Some(plane), Some(&p)) = (guide.plane, guide.partner.get(i)) {
                    if p != hierarchy::INVALID {
                        let q = plane.vector(dirs[p as usize]);
                        let nq = plane.vector(level.normals[p as usize]);
                        extra(&mut sum, q, nq, neighbour_weight);
                    }
                }
                if let Some(&(t, tw)) = guide.target.get(i) {
                    extra(&mut sum, t, n_i, tw * neighbour_weight);
                }
                if weight_sum > 0.0 {
                    let change = fold_quarter_turns(dirs[i].angle(&sum).to_degrees());
                    max_step = if change.is_finite() {
                        max_step.max(change)
                    } else {
                        f64::INFINITY
                    };
                    dirs[i] = sum;
                }
            }
        }
        if !max_step.is_finite() {
            return Err(crate::RetopoError::FieldDiverged);
        }
        if max_step < convergence_degrees || iteration >= max_iterations {
            return Ok(iteration);
        }
    }
}

/// Euler characteristic V − E + F.
pub fn euler_characteristic(vertex_count: usize, tris: &[[u32; 3]]) -> i64 {
    let mut edges = std::collections::HashSet::with_capacity(tris.len() * 3);
    for t in tris {
        for (i, j) in [(0, 1), (1, 2), (2, 0)] {
            let (a, b) = (t[i].min(t[j]), t[i].max(t[j]));
            edges.insert((a, b));
        }
    }
    vertex_count as i64 - edges.len() as i64 + tris.len() as i64
}

/// Solve the orientation field. Returns per-vertex directions and stats.
pub fn solve_orientation(
    positions: &[[f64; 3]],
    tris: &[[u32; 3]],
    options: &FieldOptions,
) -> Result<(OrientationField, FieldStats), crate::RetopoError> {
    solve_orientation_guided(positions, tris, options, None)
}

/// [`solve_orientation`] with optional precomputed per-vertex curvature
/// tensors (see [`guidance::tensors_through_refinement`]).
pub fn solve_orientation_guided(
    positions: &[[f64; 3]],
    tris: &[[u32; 3]],
    options: &FieldOptions,
    curvature: Option<&[nalgebra::Matrix3<f64>]>,
) -> Result<(OrientationField, FieldStats), crate::RetopoError> {
    let n = positions.len();

    let mut stats = FieldStats {
        euler_characteristic: euler_characteristic(n, tris),
        ..FieldStats::default()
    };

    // ---- build hierarchy and solve coarse-to-fine -------------------------
    let hierarchy = Hierarchy::build(positions, tris, 25);
    let n_levels = hierarchy.levels.len();
    stats.hierarchy_levels = n_levels;
    let guides = guidance::build_guides(
        &hierarchy,
        options.curvature_weight,
        options.guidance_scale,
        options.mirror,
        curvature,
    );

    let mut dirs: Vec<Vec<Vector3<f64>>> = Vec::with_capacity(n_levels);
    dirs.resize(n_levels, Vec::new());

    for level in (0..n_levels).rev() {
        let lv = &hierarchy.levels[level];
        let mut d: Vec<Vector3<f64>> = if level == n_levels - 1 {
            // coarsest level: deterministic pseudo-random tangent init
            (0..lv.positions.len())
                .map(|i| initial_direction(lv.positions[i], lv.normals[i]))
                .collect()
        } else {
            // inherit the solved coarser field, projected per child
            let mut init = vec![Vector3::zeros(); lv.positions.len()];
            let up = &hierarchy.to_upper[level];
            let parent = &dirs[level + 1];
            for (j, &[a, b]) in up.iter().enumerate() {
                for &dest in [a, b].iter() {
                    if dest == hierarchy::INVALID {
                        continue;
                    }
                    let n = lv.normals[dest as usize];
                    init[dest as usize] = project_unit(parent[j], n);
                }
            }
            init
        };
        let max_iter = if level == 0 {
            options.max_iterations
        } else {
            options.iterations_per_level
        };
        let iters = solve_level(
            lv,
            &guides[level],
            &mut d,
            max_iter,
            options.convergence_degrees,
        )?;
        if level == 0 {
            stats.iterations = iters;
        }
        dirs[level] = d;
    }

    let normals = hierarchy.levels[0].normals.clone();
    let dirs = dirs
        .into_iter()
        .next()
        .expect("at least one hierarchy level");

    // ---- residual error: mean best-symmetry edge angle -------------------
    let mut error_sum = 0.0;
    let mut error_count = 0usize;
    for t in tris {
        for (i, j) in [(0, 1), (1, 2), (2, 0)] {
            let a = t[i] as usize;
            let b = t[j] as usize;
            let na = normals[a];
            let nb = normals[b];
            let (va, vb) = compat_orientation_intrinsic_4(dirs[a], na, dirs[b], nb);
            let angle = fold_quarter_turns(va.angle(&vb).to_degrees());
            error_sum += angle * angle;
            error_count += 1;
        }
    }
    stats.mean_edge_error_degrees = if error_count > 0 {
        (error_sum / error_count as f64).sqrt()
    } else {
        0.0
    };

    // ---- singular faces ----------------------------------------------------
    for t in tris {
        let mut index = 0i32;
        for (i, j) in [(0, 1), (1, 2), (2, 0)] {
            let a = t[i] as usize;
            let b = t[j] as usize;
            index += compat_orientation_index_4(dirs[a], normals[a], dirs[b], normals[b]);
        }
        let m = index.rem_euclid(4);
        if m == 1 {
            stats.singular_positive += 1;
            stats.singular_faces += 1;
        } else if m == 3 {
            stats.singular_negative += 1;
            stats.singular_faces += 1;
        }
    }

    Ok((
        OrientationField {
            directions: dirs,
            normals,
        },
        stats,
    ))
}

/// Fold an angle (in degrees) into the fundamental domain of the 4-RoSy
/// symmetry: the smallest rotation that maps one field direction onto another
/// modulo 90°, in `[0, 45]`.
fn fold_quarter_turns(degrees: f64) -> f64 {
    let d = degrees.rem_euclid(90.0);
    if d > 45.0 {
        90.0 - d
    } else {
        d
    }
}

/// Convert singular faces into 3D positions (face centroids) for display.
pub fn singular_face_positions(
    positions: &[[f64; 3]],
    tris: &[[u32; 3]],
    field: &OrientationField,
) -> Vec<(usize, Singularity, Vector3<f64>)> {
    let mut out = Vec::new();
    for (fi, t) in tris.iter().enumerate() {
        let mut index = 0i32;
        for (i, j) in [(0, 1), (1, 2), (2, 0)] {
            let a = t[i] as usize;
            let b = t[j] as usize;
            index += compat_orientation_index_4(
                field.directions[a],
                field.normals[a],
                field.directions[b],
                field.normals[b],
            );
        }
        let m = index.rem_euclid(4);
        let kind = match m {
            1 => Some(Singularity::Positive),
            3 => Some(Singularity::Negative),
            _ => None,
        };
        if let Some(kind) = kind {
            let c = (Vector3::from(positions[t[0] as usize])
                + Vector3::from(positions[t[1] as usize])
                + Vector3::from(positions[t[2] as usize]))
                / 3.0;
            out.push((fi, kind, c));
        }
    }
    out
}

/// Unit-length check helper for tests / FFI consumers.
pub fn is_unit(v: &Vector3<f64>, eps: f64) -> bool {
    (v.norm() - 1.0).abs() < eps
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::obj;
    use std::path::PathBuf;

    fn fixture(name: &str) -> (Vec<[f64; 3]>, Vec<[u32; 3]>) {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures")
            .join(name);
        let mesh = obj::load(&path).unwrap_or_else(|e| panic!("failed to load {path:?}: {e}"));
        let positions = mesh
            .positions
            .iter()
            .map(|p| [p[0] as f64, p[1] as f64, p[2] as f64])
            .collect();
        (positions, mesh.triangles)
    }

    fn solve_named(name: &str) -> FieldStats {
        let (pos, tris) = fixture(name);
        let (field, stats) = solve_orientation(&pos, &tris, &FieldOptions::default()).unwrap();
        assert_eq!(field.directions.len(), pos.len());
        assert!(
            field.directions.iter().all(|d| is_unit(d, 1e-6)),
            "all field directions must be unit length"
        );
        println!("{name}: {stats:?}");
        stats
    }

    #[test]
    fn icosphere_field_is_smooth_and_singular() {
        let stats = solve_named("icosphere.obj");
        // 12 icosahedral vertices host positive singularities; a good solve
        // concentrates almost all turning there.
        assert!(
            stats.mean_edge_error_degrees < 8.0,
            "error: {}",
            stats.mean_edge_error_degrees
        );
        // A closed sphere needs at least two "plus" singularities
        // (plus − minus ≡ 2χ mod 4, i.e. even). Minimal-energy solves may
        // legitimately beat the 12 valence-5 sites.
        assert!(stats.singular_positive >= 2);
        assert_eq!(
            (stats.singular_positive + 4 - stats.singular_negative) % 2,
            0
        );
        assert_eq!(stats.euler_characteristic, 2);
    }

    #[test]
    fn uv_sphere_field_aligns() {
        let stats = solve_named("uv_sphere.obj");
        assert!(stats.mean_edge_error_degrees < 8.0);
        assert_eq!(stats.euler_characteristic, 2);
    }

    #[test]
    fn torus_field_is_balanced() {
        let stats = solve_named("torus.obj");
        assert!(stats.mean_edge_error_degrees < 10.0);
        // genus 1 -> Euler characteristic 0; for a closed surface the field
        // turning must cancel out globally.
        assert_eq!(stats.euler_characteristic, 0);
        assert_eq!(stats.singular_positive, stats.singular_negative);
    }

    #[test]
    fn suzanne_field_converges() {
        let stats = solve_named("suzanne.obj");
        assert!(stats.mean_edge_error_degrees < 20.0);
        assert_eq!(stats.euler_characteristic, 2);
    }

    #[test]
    fn cube_runs_without_nan() {
        let (_pos, _tris) = fixture("cube.obj");
        let stats = solve_named("cube.obj");
        assert!(stats.mean_edge_error_degrees.is_finite());
    }
}

#[cfg(test)]
mod diagnostics {
    use super::*;
    use crate::hierarchy;
    use crate::obj;
    use std::path::PathBuf;

    /// Diagnostic: split the residual field error into flat edges
    /// (adjacent face normals nearly parallel) and crease edges.
    /// Not a pass/fail gate — used to validate the solver.
    #[test]
    fn hierarchy_trace() {
        for name in ["cube.obj", "icosphere.obj"] {
            let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../tests/fixtures")
                .join(name);
            let mesh = obj::load(&path).unwrap();
            let positions: Vec<[f64; 3]> = mesh
                .positions
                .iter()
                .map(|p| [p[0] as f64, p[1] as f64, p[2] as f64])
                .collect();
            let tris = mesh.triangles;
            let hier = hierarchy::Hierarchy::build(&positions, &tris, 25);
            println!("{name}: {} levels", hier.levels.len());
            let n_levels = hier.levels.len();
            let mut dirs: Vec<Vec<Vector3<f64>>> = Vec::with_capacity(n_levels);
            dirs.resize(n_levels, Vec::new());
            for level in (0..n_levels).rev() {
                let lv = &hier.levels[level];
                let mut d: Vec<Vector3<f64>> = if level == n_levels - 1 {
                    (0..lv.positions.len())
                        .map(|i| initial_direction(lv.positions[i], lv.normals[i]))
                        .collect()
                } else {
                    let mut init = vec![Vector3::zeros(); lv.positions.len()];
                    let up = &hier.to_upper[level];
                    let parent = &dirs[level + 1];
                    for (j, &[a, b]) in up.iter().enumerate() {
                        for &dest in [a, b].iter() {
                            if dest == hierarchy::INVALID {
                                continue;
                            }
                            let n = lv.normals[dest as usize];
                            init[dest as usize] = project_unit(parent[j], n);
                        }
                    }
                    init
                };
                let iters = solve_level(lv, &LevelGuide::default(), &mut d, 300, 0.01).unwrap();
                let mismatch = level_mismatch(lv, &d);
                println!(
                    "  level {level:2}: nodes={:5} iters={iters:3} mean_mismatch={mismatch:.3} deg",
                    lv.positions.len()
                );
                dirs[level] = d;
            }
        }
    }

    fn level_mismatch(lv: &HierarchyLevel, dirs: &[Vector3<f64>]) -> f64 {
        let mut e = 0.0;
        let mut c = 0u64;
        for (i, nbrs) in lv.adj.iter().enumerate() {
            for &(j, _) in nbrs {
                if (j as usize) <= i {
                    continue;
                }
                let (va, vb) = compat_orientation_intrinsic_4(
                    dirs[i],
                    lv.normals[i],
                    dirs[j as usize],
                    lv.normals[j as usize],
                );
                e += fold_quarter_turns(va.angle(&vb).to_degrees());
                c += 1;
            }
        }
        e / c.max(1) as f64
    }

    #[test]
    fn cube_residual_split() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/cube.obj");
        let mesh = obj::load(&path).unwrap();
        let positions: Vec<[f64; 3]> = mesh
            .positions
            .iter()
            .map(|p| [p[0] as f64, p[1] as f64, p[2] as f64])
            .collect();
        let tris = mesh.triangles;

        // face normals
        let fnorm: Vec<Vector3<f64>> = tris
            .iter()
            .map(|t| {
                let (a, b, c) = (
                    Vector3::from(positions[t[0] as usize]),
                    Vector3::from(positions[t[1] as usize]),
                    Vector3::from(positions[t[2] as usize]),
                );
                (b - a).cross(&(c - a)).normalize()
            })
            .collect();
        let mut edge_flat: std::collections::HashMap<(u32, u32), Vec<usize>> =
            std::collections::HashMap::new();
        for (fi, t) in tris.iter().enumerate() {
            for (i, j) in [(0, 1), (1, 2), (2, 0)] {
                let (a, b) = (t[i].min(t[j]), t[i].max(t[j]));
                edge_flat.entry((a, b)).or_default().push(fi);
            }
        }

        let (field, stats) =
            solve_orientation(&positions, &tris, &FieldOptions::default()).unwrap();
        let normals = &field.normals;

        let mut flat_sum = 0.0;
        let mut flat_cnt = 0;
        let mut crease_sum = 0.0;
        let mut crease_cnt = 0;
        let mut worst_flat = 0.0f64;
        for (&(a, b), faces) in &edge_flat {
            let flat = faces.len() == 2 && fnorm[faces[0]].dot(&fnorm[faces[1]]) > 0.9999;
            let (va, vb) = compat_orientation_intrinsic_4(
                field.directions[a as usize],
                normals[a as usize],
                field.directions[b as usize],
                normals[b as usize],
            );
            let deg = fold_quarter_turns(va.angle(&vb).to_degrees());
            if flat {
                flat_sum += deg * deg;
                flat_cnt += 1;
                worst_flat = worst_flat.max(deg);
            } else {
                crease_sum += deg * deg;
                crease_cnt += 1;
            }
        }
        println!(
            "cube residual: flat({} edges) rms={:.3} deg (worst {:.3}), crease({} edges) rms={:.3} deg, overall {:.3} deg",
            flat_cnt,
            (flat_sum / flat_cnt.max(1) as f64).sqrt(),
            worst_flat,
            crease_cnt,
            (crease_sum / crease_cnt.max(1) as f64).sqrt(),
            stats.mean_edge_error_degrees,
        );
        assert!(flat_cnt > 0);
    }
}

#[cfg(test)]
mod cube_debug {
    use super::*;
    use crate::obj;
    use std::path::PathBuf;

    fn d_dot(d: &Vector3<f64>, n: &Vector3<f64>) -> f64 {
        d.dot(n)
    }

    #[test]
    fn cube_face_directions() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/cube.obj");
        let mesh = obj::load(&path).unwrap();
        for (i, p) in mesh.positions.iter().enumerate() {
            println!("v{i}: ({}, {}, {})", p[0], p[1], p[2]);
        }
        let positions: Vec<[f64; 3]> = mesh
            .positions
            .iter()
            .map(|p| [p[0] as f64, p[1] as f64, p[2] as f64])
            .collect();
        let (field, _) =
            solve_orientation(&positions, &mesh.triangles, &FieldOptions::default()).unwrap();
        for (i, d) in field.directions.iter().enumerate() {
            println!("d{i}: ({:.4}, {:.4}, {:.4})", d.x, d.y, d.z);
        }
        for (i, n) in field.normals.iter().enumerate() {
            println!(
                "n{i}: ({:.4}, {:.4}, {:.4}) |d.n|={:.4}",
                n.x,
                n.y,
                n.z,
                d_dot(&field.directions[i], n)
            );
        }
        for t in &mesh.triangles {
            println!("tri {} {} {}", t[0], t[1], t[2]);
        }
    }
}

#[cfg(test)]
mod flat_grid {
    use super::*;

    fn grid(n: usize) -> (Vec<[f64; 3]>, Vec<[u32; 3]>) {
        let mut positions = Vec::new();
        for y in 0..n {
            for x in 0..n {
                positions.push([x as f64 * 0.1, y as f64 * 0.1, 0.0]);
            }
        }
        let mut tris = Vec::new();
        for y in 0..n - 1 {
            for x in 0..n - 1 {
                let i = (y * n + x) as u32;
                let s = n as u32;
                tris.push([i, i + 1, i + s + 1]);
                tris.push([i, i + s + 1, i + s]);
            }
        }
        (positions, tris)
    }

    #[test]
    fn flat_grid_solves_to_near_zero() {
        let (pos, tris) = grid(8);
        let (_field, stats) = solve_orientation(&pos, &tris, &FieldOptions::default()).unwrap();
        println!("flat grid: {stats:?}");
        assert!(
            stats.mean_edge_error_degrees < 2.0,
            "flat grid residual must be near zero, got {}",
            stats.mean_edge_error_degrees
        );
    }
}

// ---------------------------------------------------------------------------
// position field
// ---------------------------------------------------------------------------

/// Per-vertex position-field anchor: a point near the vertex lying in the
/// tangent plane. The set of anchors tiles the surface with a regular
/// grid congruent with the orientation field.
#[derive(Debug, Clone)]
pub struct PositionField {
    /// One anchor per input vertex (same indexing as the input mesh).
    pub anchors: Vec<Vector3<f64>>,
    /// Mean target edge length the solve was run with.
    pub scale: f64,
    /// Per-vertex target edge length (all equal to `scale` when uniform).
    pub scales: Vec<f64>,
}

/// Total surface area of a triangle mesh.
pub fn surface_area(positions: &[[f64; 3]], tris: &[[u32; 3]]) -> f64 {
    let mut area = 0.0;
    for t in tris {
        let (a, b, c) = (
            Vector3::from(positions[t[0] as usize]),
            Vector3::from(positions[t[1] as usize]),
            Vector3::from(positions[t[2] as usize]),
        );
        area += (b - a).cross(&(c - a)).norm() * 0.5;
    }
    area
}

/// Point on both tangent planes closest to the midpoint between the two
/// surface points — the pivot for the minimal arc rotating frame planes.
pub(crate) fn middle_point(
    p0: Vector3<f64>,
    n0: Vector3<f64>,
    p1: Vector3<f64>,
    n1: Vector3<f64>,
) -> Vector3<f64> {
    let n0p0 = n0.dot(&p0);
    let n0p1 = n0.dot(&p1);
    let n1p0 = n1.dot(&p0);
    let n1p1 = n1.dot(&p1);
    let n0n1 = n0.dot(&n1);
    let denom = 1.0 / (1.0 - n0n1 * n0n1 + 1e-4);
    let lambda_0 = 2.0 * (n0p1 - n0p0 - n0n1 * (n1p0 - n1p1)) * denom;
    let lambda_1 = 2.0 * (n1p0 - n1p1 - n0n1 * (n0p1 - n0p0)) * denom;

    0.5 * (p0 + p1) - 0.25 * (n0 * lambda_0 + n1 * lambda_1)
}

/// Snap `p` onto the lattice defined by frame `(o, q, n)` with `scale`
/// spacing, returning the snapped lattice point.
pub(crate) fn position_round_4(
    o: Vector3<f64>,
    q: Vector3<f64>,
    n: Vector3<f64>,
    p: Vector3<f64>,
    scale: f64,
    inv_scale: f64,
) -> Vector3<f64> {
    let t = n.cross(&q);
    let d = p - o;
    o + q * (q.dot(&d) * inv_scale).round() * scale + t * (t.dot(&d) * inv_scale).round() * scale
}

/// Frame data for one side of a position-field compatibility call.
#[derive(Clone, Copy)]
struct FrameProbe {
    p: Vector3<f64>,
    n: Vector3<f64>,
    q: Vector3<f64>,
    o: Vector3<f64>,
}

/// Rotate the neighbor's position frame into the plane of `(p0, n0)` using
/// the minimal arc, including the antiparallel-normal fix. Returns the
/// rotated `(q, o)`.
pub(crate) fn rotate_frame_into_plane(
    o1: Vector3<f64>,
    q1: Vector3<f64>,
    p0: Vector3<f64>,
    n0: Vector3<f64>,
    p1: Vector3<f64>,
    n1: Vector3<f64>,
) -> (Vector3<f64>, Vector3<f64>) {
    let cos_theta = n1.dot(&n0);
    let mut q1 = q1;
    let mut o1 = o1;
    if cos_theta < 0.9999 {
        let axis = n1.cross(&n0);
        let axis_len_sq = axis.dot(&axis);
        if axis_len_sq > 1e-12 {
            let factor = (1.0 - cos_theta) / axis_len_sq;
            let middle = middle_point(p0, n0, p1, n1);
            o1 -= middle;
            q1 = q1 * cos_theta + axis.cross(&q1) + axis * (axis.dot(&q1) * factor);
            o1 = o1 * cos_theta + axis.cross(&o1) + axis * (axis.dot(&o1) * factor) + middle;
        } else if cos_theta < 0.0 {
            // exactly antiparallel normals: the rotation axis is ambiguous;
            // use a 180° turn around any orthonormal perpendicular axis.
            let arbitrary = if n1.x.abs() < 0.9 {
                Vector3::new(1.0, 0.0, 0.0)
            } else {
                Vector3::new(0.0, 1.0, 0.0)
            };
            let k = n1.cross(&arbitrary).normalize();
            let middle = middle_point(p0, n0, p1, n1);
            o1 -= middle;
            q1 = q1 * (-1.0) + k * (k.dot(&q1) * 2.0);
            o1 = o1 * (-1.0) + k * (k.dot(&o1) * 2.0) + middle;
        }
        // else: nearly identical normals — nothing to rotate
    }
    (q1, o1)
}

/// Compatibility pair for intrinsic position averaging: rotate the
/// neighbour frame into plane `f0.n` and snap onto the lattice relative to
/// the current anchor `f0.o`.
fn compat_position_intrinsic_4(
    f0: FrameProbe,
    f1: FrameProbe,
    scale: f64,
    inv_scale: f64,
) -> (Vector3<f64>, Vector3<f64>) {
    let (p0, n0, o0) = (f0.p, f0.n, f0.o);
    let (p1, n1) = (f1.p, f1.n);
    let (q1, o1) = rotate_frame_into_plane(f1.o, f1.q, p0, n0, p1, n1);
    (o0, position_round_4(o1, q1, n0, o0, scale, inv_scale))
}

fn position_hash(position: [f64; 3], normal: Vector3<f64>, scale: f64) -> Vector3<f64> {
    let mut hash: u64 = 0xcbf29ce484222325;
    for c in position {
        for b in c.to_bits().to_le_bytes() {
            hash ^= b as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    let s = (hash as f64 / u64::MAX as f64) * 2.0 - 1.0;
    let t = ((hash.rotate_left(32)) as f64 / u64::MAX as f64) * 2.0 - 1.0;
    // orthonormal frame from the normal
    let arbitrary = if normal.x.abs() < 0.9 {
        Vector3::new(1.0, 0.0, 0.0)
    } else {
        Vector3::new(0.0, 1.0, 0.0)
    };
    let t_axis = (arbitrary - normal * normal.dot(&arbitrary)).normalize();
    let b_axis = normal.cross(&t_axis);
    Vector3::from(position) + (t_axis * s + b_axis * t) * scale
}

/// Solve the position field given a solved orientation field. `target_faces`
/// derives the edge length as `sqrt(surface_area / target_faces)`; pass `0`
/// to auto-pick one quarter of the input vertex count.
pub fn solve_position(
    positions: &[[f64; 3]],
    tris: &[[u32; 3]],
    orientation: &OrientationField,
    target_faces: u32,
    options: &FieldOptions,
) -> Result<PositionField, crate::RetopoError> {
    let n = positions.len();
    let area = surface_area(positions, tris);
    let target = if target_faces > 0 {
        target_faces as f64
    } else {
        (n as f64 / 16.0).max(1.0)
    };
    let scale = (area / target).sqrt();
    let inv_scale = 1.0 / scale;

    if options.hierarchical_positions {
        let scales = vec![scale; n];
        let o = solve_position_hierarchical(positions, tris, orientation, &scales, options)?;
        return Ok(PositionField {
            anchors: o,
            scale,
            scales,
        });
    }
    let level = hierarchy::level_from_mesh(positions, tris);
    let mut o: Vec<Vector3<f64>> = (0..level.positions.len())
        .map(|i| position_hash(level.positions[i], level.normals[i], scale))
        .collect();
    let _ = inv_scale;
    let scales = vec![scale; n];
    solve_position_level(
        &level,
        &LevelGuide::default(),
        &orientation.directions,
        &mut o,
        &scales,
        options.max_iterations,
        options.convergence_degrees,
    )?;

    Ok(PositionField {
        anchors: o,
        scale,
        scales,
    })
}

/// Position solve with a per-vertex target edge length (density field,
/// see [`crate::guidance::adaptive_scales`]). `scales` is indexed like
/// `positions`.
pub fn solve_position_adaptive(
    positions: &[[f64; 3]],
    tris: &[[u32; 3]],
    orientation: &OrientationField,
    scales: &[f64],
    options: &FieldOptions,
) -> Result<PositionField, crate::RetopoError> {
    assert_eq!(scales.len(), positions.len(), "one scale per vertex");
    let areas = hierarchy::dual_areas(positions, tris);
    let total: f64 = areas.iter().sum();
    let mean = areas.iter().zip(scales).map(|(a, s)| a * s).sum::<f64>() / total.max(1e-300);
    let o = solve_position_hierarchical(positions, tris, orientation, scales, options)?;
    Ok(PositionField {
        anchors: o,
        scale: mean,
        scales: scales.to_vec(),
    })
}

/// Restrict the fine orientation field down the hierarchy: each coarse node
/// takes the symmetry-aligned sum of its (up to two) children, projected
/// onto its own tangent plane. Coarse position solves must use frames that
/// agree with the fine field, otherwise their anchors are meaningless as an
/// initialization for the finer level.
fn restrict_orientation(hier: &Hierarchy, fine: &[Vector3<f64>]) -> Vec<Vec<Vector3<f64>>> {
    let mut out: Vec<Vec<Vector3<f64>>> = vec![fine.to_vec()];
    for l in 0..hier.levels.len() - 1 {
        let lv = &hier.levels[l];
        let next = &hier.levels[l + 1];
        let src = &out[l];
        let mut q_next = Vec::with_capacity(next.positions.len());
        for (i, &[a, b]) in hier.to_upper[l].iter().enumerate() {
            let q0 = src[a as usize];
            let mut q = q0;
            if b != hierarchy::INVALID {
                let (v0, v1) = compat_orientation_intrinsic_4(
                    q0,
                    lv.normals[a as usize],
                    src[b as usize],
                    lv.normals[b as usize],
                );
                q = v0 + v1;
            }
            q_next.push(project_unit(q, next.normals[i]));
        }
        out.push(q_next);
    }
    out
}

/// Coarse-to-fine position solve (reference behaviour): hash-initialise the
/// coarsest level, solve a few sweeps per level, project anchors onto each
/// child's tangent plane, repeat down to the input mesh.
fn solve_position_hierarchical(
    positions: &[[f64; 3]],
    tris: &[[u32; 3]],
    orientation: &OrientationField,
    fine_scales: &[f64],
    options: &FieldOptions,
) -> Result<Vec<Vector3<f64>>, crate::RetopoError> {
    let hier = Hierarchy::build(positions, tris, 25);
    let q = restrict_orientation(&hier, &orientation.directions);
    // per-level scales: area-weighted mean of the (up to two) children
    let mut level_scales: Vec<Vec<f64>> = vec![fine_scales.to_vec()];
    for l in 0..hier.levels.len() - 1 {
        let (src, lv) = (&level_scales[l], &hier.levels[l]);
        let next: Vec<f64> = hier.to_upper[l]
            .iter()
            .map(|&[a, b]| {
                let (wa, mut acc, mut w) = (lv.areas[a as usize], 0.0, 0.0);
                acc += src[a as usize] * wa;
                w += wa;
                if b != hierarchy::INVALID {
                    acc += src[b as usize] * lv.areas[b as usize];
                    w += lv.areas[b as usize];
                }
                if w > 0.0 {
                    acc / w
                } else {
                    src[a as usize]
                }
            })
            .collect();
        level_scales.push(next);
    }
    let guides = guidance::build_guides(&hier, 0.0, 0.0, options.mirror, None);
    let n_levels = hier.levels.len();
    let mut o: Vec<Vector3<f64>> = Vec::new();
    for level in (0..n_levels).rev() {
        let lv = &hier.levels[level];
        if level == n_levels - 1 {
            o = (0..lv.positions.len())
                .map(|i| position_hash(lv.positions[i], lv.normals[i], level_scales[level][i]))
                .collect();
        } else {
            let mut init = vec![Vector3::zeros(); lv.positions.len()];
            for (j, &[a, b]) in hier.to_upper[level].iter().enumerate() {
                for dest in [a, b] {
                    if dest == hierarchy::INVALID {
                        continue;
                    }
                    let v = Vector3::from(lv.positions[dest as usize]);
                    let n = lv.normals[dest as usize];
                    init[dest as usize] = o[j] - n * n.dot(&(o[j] - v));
                }
            }
            o = init;
        }
        let iters = if level == 0 {
            options.max_iterations
        } else {
            options.iterations_per_level
        };
        solve_position_level(
            lv,
            &guides[level],
            &q[level],
            &mut o,
            &level_scales[level],
            iters,
            options.convergence_degrees,
        )?;
    }
    Ok(o)
}

/// One level of the position solve.
#[allow(clippy::too_many_arguments)]
fn solve_position_level(
    level: &HierarchyLevel,
    guide: &LevelGuide,
    q_dirs: &[Vector3<f64>],
    anchors: &mut [Vector3<f64>],
    scales: &[f64],
    max_iterations: u32,
    convergence_degrees: f64,
) -> Result<u32, crate::RetopoError> {
    let phases = color_graph(&level.adj);
    let min_scale = scales.iter().copied().fold(f64::INFINITY, f64::min);
    let convergence_dist = min_scale * (convergence_degrees.to_radians()).tan();
    let mut iteration = 0u32;
    loop {
        iteration += 1;
        let mut max_step = 0.0f64;
        for phase in &phases {
            for &i in phase {
                if level.adj[i].is_empty() {
                    continue;
                }
                let n_i = level.normals[i];
                let v_i = Vector3::from(level.positions[i]);
                let q_i = q_dirs[i];
                let s_i = scales[i];
                let mut sum = anchors[i];
                let mut weight_sum = 0.0f64;
                for &(j, w) in &level.adj[i] {
                    // neighbouring lattices of different spacing meet halfway
                    let scale = 0.5 * (s_i + scales[j as usize]);
                    let inv_scale = 1.0 / scale;
                    let q_j = q_dirs[j as usize];
                    let o_j = anchors[j as usize];
                    let p_j = Vector3::from(level.positions[j as usize]);
                    let (v0, v1) = compat_position_intrinsic_4(
                        FrameProbe {
                            p: v_i,
                            n: n_i,
                            q: q_i,
                            o: sum,
                        },
                        FrameProbe {
                            p: p_j,
                            n: level.normals[j as usize],
                            q: q_j,
                            o: o_j,
                        },
                        scale,
                        inv_scale,
                    );
                    sum = v0 * weight_sum + v1 * w;
                    weight_sum += w;
                    if weight_sum > 1e-12 {
                        sum /= weight_sum;
                    }
                    sum -= n_i * n_i.dot(&(sum - v_i));
                }
                if let (Some(plane), Some(&p)) = (guide.plane, guide.partner.get(i)) {
                    if p != hierarchy::INVALID && weight_sum > 0.0 {
                        let pu = p as usize;
                        let w = weight_sum;
                        let scale = 0.5 * (s_i + scales[pu]);
                        let inv_scale = 1.0 / scale;
                        let (v0, v1) = compat_position_intrinsic_4(
                            FrameProbe {
                                p: v_i,
                                n: n_i,
                                q: q_i,
                                o: sum,
                            },
                            FrameProbe {
                                p: plane.point_v(Vector3::from(level.positions[pu])),
                                n: plane.vector(level.normals[pu]),
                                q: plane.vector(q_dirs[pu]),
                                o: plane.point_v(anchors[pu]),
                            },
                            scale,
                            inv_scale,
                        );
                        sum = (v0 * weight_sum + v1 * w) / (weight_sum + w);
                        sum -= n_i * n_i.dot(&(sum - v_i));
                    }
                }
                // Mirror seam: near the plane, slide the lattice origin along
                // the plane-crossing direction onto the plane, so one lattice
                // line (an edge loop) runs exactly along it. Without this the
                // lattice may sit half a cell off and no half can be mirrored.
                if let Some(plane) = guide.plane {
                    let ax = plane.axis;
                    if (v_i[ax] - plane.offset).abs() < s_i {
                        let t_i = n_i.cross(&q_i);
                        let c = if q_i[ax].abs() >= t_i[ax].abs() {
                            q_i
                        } else {
                            t_i
                        };
                        if c[ax].abs() > 0.85 {
                            sum += c * ((plane.offset - sum[ax]) / c[ax]);
                        }
                    }
                }
                // Re-anchor: pick the lattice point (with origin `sum`) closest
                // to the vertex itself. Without this the anchor drifts freely
                // along the lattice and every extra sweep diffuses the field.
                let sum = position_round_4(sum, q_i, n_i, v_i, s_i, 1.0 / s_i);
                let step = (anchors[i] - sum).norm();
                max_step = if step.is_finite() {
                    max_step.max(step)
                } else {
                    f64::INFINITY
                };
                anchors[i] = sum;
            }
        }
        if !max_step.is_finite() {
            return Err(crate::RetopoError::FieldDiverged);
        }
        if max_step < convergence_dist || iteration >= max_iterations {
            return Ok(iteration);
        }
    }
}

#[cfg(test)]
mod position_tests {
    use super::*;
    use crate::obj;
    use std::path::PathBuf;

    fn fixture(name: &str) -> (Vec<[f64; 3]>, Vec<[u32; 3]>) {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures")
            .join(name);
        let mesh = obj::load(&path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
        let positions = mesh
            .positions
            .iter()
            .map(|p| [p[0] as f64, p[1] as f64, p[2] as f64])
            .collect();
        (positions, mesh.triangles)
    }

    fn grid(n: usize) -> (Vec<[f64; 3]>, Vec<[u32; 3]>) {
        let mut positions = Vec::new();
        for y in 0..n {
            for x in 0..n {
                positions.push([x as f64 * 0.1, y as f64 * 0.1, 0.0]);
            }
        }
        let mut tris = Vec::new();
        let s = n as u32;
        for y in 0..n - 1 {
            for x in 0..n - 1 {
                let i = (y * n + x) as u32;
                tris.push([i, i + 1, i + s + 1]);
                tris.push([i, i + s + 1, i + s]);
            }
        }
        (positions, tris)
    }

    #[test]
    fn flat_grid_lattice_is_exact() {
        let (pos, tris) = grid(8);
        let (orient, _) = solve_orientation(&pos, &tris, &FieldOptions::default()).unwrap();
        let pf = solve_position(&pos, &tris, &orient, 49, &FieldOptions::default()).unwrap();
        assert!((pf.scale - 0.1).abs() < 1e-9);
        // anchors must stay on the z=0 plane
        assert!(pf.anchors.iter().all(|a| a.z.abs() < 1e-9));
        // the field is globally consistent (flat sheet admits a perfect
        // solution): anchor pairwise differences lie on the solved lattice
        let axis = Vector3::new(orient.directions[0].x, orient.directions[0].y, 0.0).normalize();
        let perp = Vector3::new(-axis.y, axis.x, 0.0);
        let k = |v: f64| v / 0.1;
        for i in 0..pf.anchors.len() {
            for j in i + 1..pf.anchors.len() {
                let d = pf.anchors[i] - pf.anchors[j];
                let du = k(d.dot(&axis));
                let dv = k(d.dot(&perp));
                assert!(
                    (du - du.round()).abs() < 1e-6,
                    "off-lattice u between {i} and {j}"
                );
                assert!(
                    (dv - dv.round()).abs() < 1e-6,
                    "off-lattice v between {i} and {j}"
                );
            }
        }
    }

    #[test]
    fn positions_are_finite_on_fixtures() {
        for name in [
            "uv_sphere.obj",
            "icosphere.obj",
            "suzanne.obj",
            "torus.obj",
            "cube.obj",
        ] {
            let (pos, tris) = fixture(name);
            let (orient, _) = solve_orientation(&pos, &tris, &FieldOptions::default()).unwrap();
            let pf = solve_position(&pos, &tris, &orient, 2000, &FieldOptions::default()).unwrap();
            let finite = pf.anchors.iter().all(|a| a.norm().is_finite());
            println!("{name}: scale={:.4} finite={finite}", pf.scale);
            assert!(finite);
        }
    }
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)] // temporary diagnostics
mod position_convergence {
    use super::*;
    use crate::obj;
    use std::path::PathBuf;

    #[test]
    fn icosphere_convergence_trace() {
        let path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/icosphere.obj");
        let mesh = obj::load(&path).unwrap();
        let positions: Vec<[f64; 3]> = mesh
            .positions
            .iter()
            .map(|p| [p[0] as f64, p[1] as f64, p[2] as f64])
            .collect();
        let tris = mesh.triangles;
        let opts = FieldOptions::default();
        let (orient, _) = solve_orientation(&positions, &tris, &opts).unwrap();
        let pf = solve_position(
            &positions,
            &tris,
            &orient,
            320,
            &FieldOptions {
                max_iterations: 3000,
                iterations_per_level: 10,
                convergence_degrees: 0.0,
                rosy: 4,
                hierarchical_positions: true,
                ..FieldOptions::default()
            },
        )
        .unwrap();

        // residual between adjacent anchors, in lattice cells
        let adj = {
            let mut a: Vec<Vec<u32>> = vec![Vec::new(); positions.len()];
            let mut seen: Vec<std::collections::HashSet<u32>> = (0..positions.len())
                .map(|_| std::collections::HashSet::new())
                .collect();
            for t in &tris {
                for i in 0..3 {
                    let v = t[i] as usize;
                    for k in 1..3 {
                        let n = t[(i + k) % 3];
                        if seen[v].insert(n) {
                            a[v].push(n);
                        }
                    }
                }
            }
            a
        };
        let inv_scale = 1.0 / pf.scale;
        let mut count = 0u64;
        let mut within_one_cell = 0u64;
        let mut total_dev = 0.0;
        for i in 0..positions.len() {
            let n_i = orient.normals[i];
            let u = orient.directions[i].normalize();
            let v = n_i.cross(&u);
            for &j in &adj[i] {
                if (j as usize) <= i {
                    continue;
                }
                let d = pf.anchors[i] - pf.anchors[j as usize];
                // project onto the lattice axes of node i
                let du = (d.dot(&u) * inv_scale).round();
                let dv = (d.dot(&v) * inv_scale).round();
                let on_lattice = u * du * pf.scale + v * dv * pf.scale + n_i * n_i.dot(&d);
                let dev = (d - on_lattice).norm();
                total_dev += dev;
                if dev < 0.25 * pf.scale {
                    within_one_cell += 1;
                }
                count += 1;
            }
        }
        println!("edges={count} within_quarter_cell={within_one_cell} ({:.1}%) mean_dev={:.4} scale={:.4}",
                 100.0 * within_one_cell as f64 / count as f64, total_dev / count as f64, pf.scale);
    }
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)] // temporary diagnostics
mod position_lattice_sweep {
    use super::*;
    use crate::obj;
    use std::path::PathBuf;

    fn lattice_consistency(name: &str, target_faces: u32) -> (f64, f64, f64) {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures")
            .join(name);
        let mesh = obj::load(&path).unwrap();
        let positions: Vec<[f64; 3]> = mesh
            .positions
            .iter()
            .map(|p| [p[0] as f64, p[1] as f64, p[2] as f64])
            .collect();
        let tris = mesh.triangles;
        let opts = FieldOptions::default();
        let (orient, _) = solve_orientation(&positions, &tris, &opts).unwrap();
        let pf = solve_position(&positions, &tris, &orient, target_faces, &opts).unwrap();

        let mut adj: Vec<Vec<u32>> = vec![Vec::new(); positions.len()];
        let mut seen: Vec<std::collections::HashSet<u32>> = (0..positions.len())
            .map(|_| std::collections::HashSet::new())
            .collect();
        for t in &tris {
            for i in 0..3 {
                let v = t[i] as usize;
                for k in 1..3 {
                    let n = t[(i + k) % 3];
                    if seen[v].insert(n) {
                        adj[v].push(n);
                    }
                }
            }
        }
        let inv_scale = 1.0 / pf.scale;
        let mut count = 0u64;
        let mut within = 0u64;
        let mut total_dev = 0.0f64;
        for i in 0..positions.len() {
            let n_i = orient.normals[i];
            let u = orient.directions[i].normalize();
            let v = n_i.cross(&u);
            for &j in &adj[i] {
                if (j as usize) <= i {
                    continue;
                }
                let d = pf.anchors[i] - pf.anchors[j as usize];
                let du = (d.dot(&u) * inv_scale).round();
                let dv = (d.dot(&v) * inv_scale).round();
                let on_lattice = u * du * pf.scale + v * dv * pf.scale + n_i * n_i.dot(&d);
                let dev = (d - on_lattice).norm();
                total_dev += dev;
                if dev < 0.25 * pf.scale {
                    within += 1;
                }
                count += 1;
            }
        }
        (
            100.0 * within as f64 / count as f64,
            total_dev / count as f64,
            pf.scale,
        )
    }

    #[test]
    fn sweep() {
        for (name, target) in [
            ("icosphere.obj", 80u32),
            ("icosphere.obj", 160),
            ("icosphere.obj", 320),
            ("uv_sphere.obj", 300),
            ("suzanne.obj", 500),
        ] {
            let (within, dev, scale) = lattice_consistency(name, target);
            println!("{name}: consistent={within:.1}% mean_dev={dev:.4} scale={scale:.4}");
        }
    }
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)] // temporary diagnostics
mod sweep_iters {
    use super::*;
    use crate::obj;
    use std::path::PathBuf;

    fn consistent(name: &str, target: u32, iters: u32, hier: bool) -> f64 {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures")
            .join(name);
        let mesh = obj::load(&path).unwrap();
        let positions: Vec<[f64; 3]> = mesh
            .positions
            .iter()
            .map(|p| [p[0] as f64, p[1] as f64, p[2] as f64])
            .collect();
        let tris = mesh.triangles;
        let opts = FieldOptions {
            max_iterations: iters,
            iterations_per_level: iters,
            convergence_degrees: 0.0,
            rosy: 4,
            hierarchical_positions: hier,
            ..FieldOptions::default()
        };
        let (orient, _) = solve_orientation(&positions, &tris, &FieldOptions::default()).unwrap();
        let pf = solve_position(&positions, &tris, &orient, target, &opts).unwrap();

        let mut adj: Vec<Vec<u32>> = vec![Vec::new(); positions.len()];
        let mut seen: Vec<std::collections::HashSet<u32>> = (0..positions.len())
            .map(|_| std::collections::HashSet::new())
            .collect();
        for t in &tris {
            for i in 0..3 {
                let v = t[i] as usize;
                for k in 1..3 {
                    let n = t[(i + k) % 3];
                    if seen[v].insert(n) {
                        adj[v].push(n);
                    }
                }
            }
        }
        let inv_scale = 1.0 / pf.scale;
        let mut count = 0u64;
        let mut within = 0u64;
        for i in 0..positions.len() {
            let n_i = orient.normals[i];
            let u = orient.directions[i].normalize();
            let v = n_i.cross(&u);
            for &j in &adj[i] {
                if (j as usize) <= i {
                    continue;
                }
                let d = pf.anchors[i] - pf.anchors[j as usize];
                let du = (d.dot(&u) * inv_scale).round();
                let dv = (d.dot(&v) * inv_scale).round();
                let on_lattice = u * du * pf.scale + v * dv * pf.scale + n_i * n_i.dot(&d);
                if (d - on_lattice).norm() < 0.25 * pf.scale {
                    within += 1;
                }
                count += 1;
            }
        }
        100.0 * within as f64 / count as f64
    }

    #[test]
    fn iters_sweep() {
        for hier in [false, true] {
            for iters in [2u32, 6, 12, 25, 100] {
                println!(
                    "hier={hier} iters={iters}: ico={:.1} uv={:.1} torus={:.1} suzanne={:.1}",
                    consistent("icosphere.obj", 320, iters, hier),
                    consistent("uv_sphere.obj", 300, iters, hier),
                    consistent("torus.obj", 300, iters, hier),
                    consistent("suzanne.obj", 500, iters, hier),
                );
            }
        }
    }
}
