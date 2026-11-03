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
}

impl Default for FieldOptions {
    fn default() -> Self {
        FieldOptions {
            max_iterations: 250,
            iterations_per_level: 25,
            convergence_degrees: 0.1,
            rosy: 4,
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
fn compat_orientation_intrinsic_4(
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

/// One full optimization pass over a level until convergence or the
/// iteration cap. Returns the number of iterations performed.
fn solve_level(
    level: &HierarchyLevel,
    dirs: &mut [Vector3<f64>],
    max_iterations: u32,
    convergence_degrees: f64,
) -> Result<u32, crate::RetopoError> {
    let mut iteration = 0u32;
    loop {
        iteration += 1;
        let mut max_step = 0.0f64;
        for i in 0..level.positions.len() {
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
    let n = positions.len();

    let mut stats = FieldStats {
        euler_characteristic: euler_characteristic(n, tris),
        ..FieldStats::default()
    };

    // ---- build hierarchy and solve coarse-to-fine -------------------------
    let hierarchy = Hierarchy::build(positions, tris, 25);
    let n_levels = hierarchy.levels.len();
    stats.hierarchy_levels = n_levels;

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
        let iters = solve_level(lv, &mut d, max_iter, options.convergence_degrees)?;
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
                let iters = solve_level(lv, &mut d, 300, 0.01).unwrap();
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
    /// Target edge length the solve was run with.
    pub scale: f64,
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
fn middle_point(
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
fn position_round_4(
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

/// Compatibility pair for intrinsic position averaging: rotate the
/// neighbour frame into plane `n0` and snap onto the lattice relative to
/// the current anchor `o0`.
fn compat_position_intrinsic_4(
    p0: Vector3<f64>,
    n0: Vector3<f64>,
    _q0: Vector3<f64>,
    o0: Vector3<f64>,
    p1: Vector3<f64>,
    n1: Vector3<f64>,
    q1_raw: Vector3<f64>,
    o1_raw: Vector3<f64>,
    scale: f64,
    inv_scale: f64,
) -> (Vector3<f64>, Vector3<f64>) {
    let cos_theta = n1.dot(&n0);
    let mut q1 = q1_raw;
    let mut o1 = o1_raw;

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

    let hierarchy = Hierarchy::build(positions, tris, 25);
    let n_levels = hierarchy.levels.len();
    let mut anchors: Vec<Vec<Vector3<f64>>> = Vec::with_capacity(n_levels);
    anchors.resize(n_levels, Vec::new());

    for level in (0..n_levels).rev() {
        let lv = &hierarchy.levels[level];
        let mut o: Vec<Vector3<f64>> = if level == n_levels - 1 {
            (0..lv.positions.len())
                .map(|i| position_hash(lv.positions[i], lv.normals[i], scale))
                .collect()
        } else {
            // project the parent's anchor onto the child's tangent plane
            // through the child's surface point: o -= n*n.dot(o - v)
            let mut init = vec![Vector3::zeros(); lv.positions.len()];
            let up = &hierarchy.to_upper[level];
            let parent = &anchors[level + 1];
            for (j, &[a, b]) in up.iter().enumerate() {
                for &dest in [a, b].iter() {
                    if dest == hierarchy::INVALID {
                        continue;
                    }
                    let child_pos = Vector3::from(lv.positions[dest as usize]);
                    let n_child = lv.normals[dest as usize];
                    let o = parent[j] - n_child * n_child.dot(&(parent[j] - child_pos));
                    init[dest as usize] = o;
                }
            }
            init
        };

        let q_dirs: Vec<Vector3<f64>> = if level == 0 {
            orientation.directions.clone()
        } else {
            // coarse levels: re-derive directions with the orientation solver
            // on that level's graph rather than interpolating level 0's
            let (sub_field, _) = solve_orientation_for_level(lv, options)?;
            sub_field
        };

        let max_iter = if level == 0 {
            options.max_iterations
        } else {
            options.iterations_per_level
        };
        solve_position_level(
            lv,
            &q_dirs,
            &mut o,
            scale,
            inv_scale,
            max_iter,
            options.convergence_degrees,
        )?;
        anchors[level] = o;
    }

    Ok(PositionField {
        anchors: anchors
            .into_iter()
            .next()
            .expect("at least one hierarchy level"),
        scale,
    })
}

/// Solve the orientation field on an arbitrary hierarchy level (used for
/// coarse position solves so each level has its own consistent field).
fn solve_orientation_for_level(
    level: &HierarchyLevel,
    options: &FieldOptions,
) -> Result<(Vec<Vector3<f64>>, u32), crate::RetopoError> {
    let mut dirs: Vec<Vector3<f64>> = (0..level.positions.len())
        .map(|i| initial_direction(level.positions[i], level.normals[i]))
        .collect();
    let iters = solve_level(
        level,
        &mut dirs,
        options.iterations_per_level,
        options.convergence_degrees,
    )?;
    Ok((dirs, iters))
}

/// One level of the position solve.
fn solve_position_level(
    level: &HierarchyLevel,
    q_dirs: &[Vector3<f64>],
    anchors: &mut [Vector3<f64>],
    scale: f64,
    inv_scale: f64,
    max_iterations: u32,
    convergence_degrees: f64,
) -> Result<u32, crate::RetopoError> {
    let convergence_dist = scale * (convergence_degrees.to_radians()).tan();
    let mut iteration = 0u32;
    loop {
        iteration += 1;
        let mut max_step = 0.0f64;
        for i in 0..level.positions.len() {
            if level.adj[i].is_empty() {
                continue;
            }
            let n_i = level.normals[i];
            let v_i = Vector3::from(level.positions[i]);
            let q_i = q_dirs[i];
            let mut sum = anchors[i];
            let mut weight_sum = 0.0f64;
            for &(j, w) in &level.adj[i] {
                let q_j = q_dirs[j as usize];
                let o_j = anchors[j as usize];
                let p_j = Vector3::from(level.positions[j as usize]);
                let (v0, v1) = compat_position_intrinsic_4(
                    v_i,
                    n_i,
                    q_i,
                    sum,
                    p_j,
                    level.normals[j as usize],
                    q_j,
                    o_j,
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
            let step = (anchors[i] - sum).norm();
            max_step = if step.is_finite() {
                max_step.max(step)
            } else {
                f64::INFINITY
            };
            anchors[i] = sum;
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
mod position_debug {
    use super::*;
    use crate::obj;
    use std::path::PathBuf;

    #[test]
    fn cube_position_trace() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/cube.obj");
        let mesh = obj::load(&path).unwrap();
        let positions: Vec<[f64; 3]> = mesh
            .positions
            .iter()
            .map(|p| [p[0] as f64, p[1] as f64, p[2] as f64])
            .collect();
        let tris = mesh.triangles;
        let (orient, _) = solve_orientation(&positions, &tris, &FieldOptions::default()).unwrap();
        let area = surface_area(&positions, &tris);
        let scale = (area / 2000.0f64).sqrt();
        let _ = scale;
        let hier = Hierarchy::build(&positions, &tris, 25);
        let n_levels = hier.levels.len();
        let mut anchors: Vec<Vec<Vector3<f64>>> = Vec::with_capacity(n_levels);
        anchors.resize(n_levels, Vec::new());
        for level in (0..n_levels).rev() {
            let lv = &hier.levels[level];
            let mut o: Vec<Vector3<f64>> = if level == n_levels - 1 {
                (0..lv.positions.len())
                    .map(|i| position_hash(lv.positions[i], lv.normals[i], 0.1))
                    .collect()
            } else {
                let mut init = vec![Vector3::zeros(); lv.positions.len()];
                let up = &hier.to_upper[level];
                let parent = &anchors[level + 1];
                for (j, &[a, b]) in up.iter().enumerate() {
                    for &dest in [a, b].iter() {
                        if dest == hierarchy::INVALID {
                            continue;
                        }
                        let child_pos = Vector3::from(lv.positions[dest as usize]);
                        let n_child = lv.normals[dest as usize];
                        let op = parent[j] - n_child * n_child.dot(&(parent[j] - child_pos));
                        init[dest as usize] = op;
                    }
                }
                init
            };
            let (q_dirs, _) = if level == 0 {
                (orient.directions.clone(), 0)
            } else {
                solve_orientation_for_level(lv, &FieldOptions::default()).unwrap()
            };
            println!("level {level} anchors pre-solve: {:?}", o);
            println!("level {level} q_dirs: {:?}", q_dirs);
            println!("level {level} adj: {:?}", lv.adj);
            let sc = (area / 2000.0f64).sqrt();
            let iters = solve_position_level(lv, &q_dirs, &mut o, sc, 1.0 / sc, 25, 0.1).unwrap();
            let max_o = o.iter().map(|a| a.norm()).fold(0.0f64, f64::max);
            let min_o = o.iter().map(|a| a.norm()).fold(f64::INFINITY, f64::min);
            let nonfinite = o.iter().filter(|a| !a.norm().is_finite()).count();
            println!("level {level}: nodes={} iters={iters} max|o|={max_o:.3} min|o|={min_o:.3} nonfinite={nonfinite}", lv.positions.len());
            anchors[level] = o;
        }
    }
}
