//! Soft guidance for the field solves: principal-curvature alignment and
//! mirror symmetry.
//!
//! * **Curvature.** A per-vertex shape tensor is fitted from the normal
//!   curvature along each one-ring edge, diffused over roughly one target
//!   edge length, and restricted down the hierarchy. Where the surface is
//!   clearly anisotropic (limbs, necks, ears — `|k1 − k2|` large relative to
//!   the target edge length) the orientation field is pulled toward the
//!   principal directions, so edge loops wrap around tubes instead of
//!   spiralling. Isotropic regions (spheres, flats) are left to smoothness.
//! * **Symmetry.** Every hierarchy node is paired with the node nearest its
//!   mirror image. Both solves treat the reflected partner as one more
//!   (heavily weighted) neighbour, which drives the orientation and position
//!   fields — and therefore the extracted quads — to be mirror-consistent.
//!
//! Both are pure functions of the input, so results stay deterministic.

use std::collections::HashMap;

use nalgebra::{Matrix2, Matrix3, Vector3};

use crate::hierarchy::{Hierarchy, HierarchyLevel, INVALID};

/// Mirror plane `x[axis] == offset` (axis 0 = X, 1 = Y, 2 = Z).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MirrorPlane {
    pub axis: usize,
    pub offset: f64,
}

impl MirrorPlane {
    pub fn point(&self, p: [f64; 3]) -> [f64; 3] {
        let mut r = p;
        r[self.axis] = 2.0 * self.offset - p[self.axis];
        r
    }
    pub fn point_v(&self, p: Vector3<f64>) -> Vector3<f64> {
        let mut r = p;
        r[self.axis] = 2.0 * self.offset - p[self.axis];
        r
    }
    pub fn vector(&self, v: Vector3<f64>) -> Vector3<f64> {
        let mut r = v;
        r[self.axis] = -v[self.axis];
        r
    }
}

/// Per-level guidance consumed by the solvers. Empty vectors mean "off".
#[derive(Debug, Clone, Default)]
pub struct LevelGuide {
    /// Target direction and weight (relative to the neighbour weight sum).
    pub target: Vec<(Vector3<f64>, f64)>,
    /// Mirror partner per node (`INVALID` = none).
    pub partner: Vec<u32>,
    pub plane: Option<MirrorPlane>,
}

fn mean_edge_length(level: &HierarchyLevel) -> f64 {
    let (mut sum, mut count) = (0.0, 0usize);
    for (i, nbrs) in level.adj.iter().enumerate() {
        for &(j, _) in nbrs {
            if (j as usize) > i {
                let (a, b) = (level.positions[i], level.positions[j as usize]);
                sum +=
                    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
                count += 1;
            }
        }
    }
    if count == 0 {
        0.0
    } else {
        sum / count as f64
    }
}

/// Uniform hash grid over points for radius queries.
struct Grid {
    cell: f64,
    map: HashMap<(i64, i64, i64), Vec<u32>>,
}

impl Grid {
    fn new(points: &[[f64; 3]], cell: f64) -> Grid {
        let mut map: HashMap<(i64, i64, i64), Vec<u32>> = HashMap::new();
        for (i, p) in points.iter().enumerate() {
            map.entry(Self::key(p, cell)).or_default().push(i as u32);
        }
        Grid { cell, map }
    }
    fn key(p: &[f64; 3], cell: f64) -> (i64, i64, i64) {
        (
            (p[0] / cell).floor() as i64,
            (p[1] / cell).floor() as i64,
            (p[2] / cell).floor() as i64,
        )
    }
    /// Nearest point within `radius` (ties → smaller index) passing `accept`.
    fn nearest(
        &self,
        points: &[[f64; 3]],
        q: [f64; 3],
        radius: f64,
        mut accept: impl FnMut(u32) -> bool,
    ) -> Option<u32> {
        let (kx, ky, kz) = Self::key(&q, self.cell);
        let reach = (radius / self.cell).ceil() as i64;
        let mut best: Option<(f64, u32)> = None;
        for dx in -reach..=reach {
            for dy in -reach..=reach {
                for dz in -reach..=reach {
                    let Some(list) = self.map.get(&(kx + dx, ky + dy, kz + dz)) else {
                        continue;
                    };
                    for &i in list {
                        let p = points[i as usize];
                        let d2 =
                            (p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2);
                        if d2 > radius * radius {
                            continue;
                        }
                        if best.is_none_or(|(bd, bi)| d2 < bd || (d2 == bd && i < bi)) && accept(i)
                        {
                            best = Some((d2, i));
                        }
                    }
                }
            }
        }
        best.map(|(_, i)| i)
    }
}

/// Mirror partner of every node of `level` (nearest node to the reflected
/// position with a mirror-consistent normal, within 1.5 edge lengths).
pub fn mirror_partners(level: &HierarchyLevel, plane: MirrorPlane) -> Vec<u32> {
    let h = mean_edge_length(level).max(1e-12);
    let radius = 1.5 * h;
    let grid = Grid::new(&level.positions, radius);
    (0..level.positions.len())
        .map(|i| {
            let q = plane.point(level.positions[i]);
            let n = level.normals[i];
            grid.nearest(&level.positions, q, radius, |j| {
                plane.vector(level.normals[j as usize]).dot(&n) > 0.3
            })
            .unwrap_or(INVALID)
        })
        .collect()
}

/// How symmetric the surface is about `plane`: mean distance from each
/// reflected vertex to the nearest vertex, in units of the mean edge length
/// (`f64::INFINITY` if most vertices have no partner nearby).
pub fn mirror_error(level: &HierarchyLevel, plane: MirrorPlane) -> f64 {
    let h = mean_edge_length(level).max(1e-12);
    let radius = 3.0 * h;
    let grid = Grid::new(&level.positions, radius);
    let (mut sum, mut found) = (0.0, 0usize);
    let n = level.positions.len();
    for i in 0..n {
        let q = plane.point(level.positions[i]);
        if let Some(j) = grid.nearest(&level.positions, q, radius, |_| true) {
            let p = level.positions[j as usize];
            sum += ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt();
            found += 1;
        }
    }
    if found * 10 < n * 9 {
        return f64::INFINITY;
    }
    sum / found as f64 / h
}

/// Pick the mirror plane (through the area-weighted centroid, along a local
/// axis) the surface is most symmetric about, if any is convincingly so.
pub fn detect_mirror(level: &HierarchyLevel) -> Option<MirrorPlane> {
    let total: f64 = level.areas.iter().sum();
    if total <= 0.0 {
        return None;
    }
    let mut c = [0.0; 3];
    for (p, a) in level.positions.iter().zip(&level.areas) {
        for k in 0..3 {
            c[k] += p[k] * a / total;
        }
    }
    let mut best: Option<(f64, MirrorPlane)> = None;
    for (axis, &offset) in c.iter().enumerate() {
        let plane = MirrorPlane { axis, offset };
        let e = mirror_error(level, plane);
        if e < 0.6 && best.is_none_or(|(be, _)| e < be) {
            best = Some((e, plane));
        }
    }
    best.map(|(_, p)| p)
}

/// Orthonormal tangent basis at normal `n`.
fn tangent_basis(n: Vector3<f64>) -> (Vector3<f64>, Vector3<f64>) {
    let a = if n.x.abs() < 0.9 {
        Vector3::new(1.0, 0.0, 0.0)
    } else {
        Vector3::new(0.0, 1.0, 0.0)
    };
    let u = (a - n * n.dot(&a)).normalize();
    (u, n.cross(&u))
}

/// Per-vertex shape tensors (3×3, tangent-plane embedded) of level 0,
/// least-squares fitted to the one-ring normal curvatures and diffused
/// `smoothing` times over the one-ring.
pub fn curvature_tensors(level: &HierarchyLevel, smoothing: usize) -> Vec<Matrix3<f64>> {
    let n = level.positions.len();
    let mut s: Vec<Matrix3<f64>> = (0..n)
        .map(|i| {
            let p = Vector3::from(level.positions[i]);
            let nrm = level.normals[i];
            let (u, v) = tangent_basis(nrm);
            // normal equations for κ(t) = a c² + 2b c s + d s²
            let mut ata = Matrix3::<f64>::zeros();
            let mut atb = Vector3::<f64>::zeros();
            for &(j, _) in &level.adj[i] {
                let e = Vector3::from(level.positions[j as usize]) - p;
                let len2 = e.norm_squared();
                let t = e - nrm * nrm.dot(&e);
                let tl = t.norm();
                if len2 < 1e-30 || tl < 1e-15 {
                    continue;
                }
                let kappa = 2.0 * nrm.dot(&e) / len2;
                let (c, sn) = (t.dot(&u) / tl, t.dot(&v) / tl);
                let row = Vector3::new(c * c, 2.0 * c * sn, sn * sn);
                ata += row * row.transpose();
                atb += row * kappa;
            }
            // ridge term keeps the fit well-posed on tiny rings
            ata += Matrix3::identity() * 1e-6 * (ata.trace() + 1e-12);
            let x = ata.lu().solve(&atb).unwrap_or_else(Vector3::zeros);
            if !x.iter().all(|c| c.is_finite()) {
                return Matrix3::zeros();
            }
            u * u.transpose() * x[0]
                + (u * v.transpose() + v * u.transpose()) * x[1]
                + v * v.transpose() * x[2]
        })
        .collect();
    for _ in 0..smoothing {
        let prev = s.clone();
        for i in 0..n {
            let mut acc = prev[i] * level.areas[i];
            let mut w = level.areas[i];
            for &(j, _) in &level.adj[i] {
                acc += prev[j as usize] * level.areas[j as usize];
                w += level.areas[j as usize];
            }
            if w > 0.0 {
                s[i] = acc / w;
            }
        }
    }
    s
}

/// Diffusion passes so the smoothing radius (~sqrt(passes) edges) reaches
/// about one target edge length.
pub fn smoothing_for(level: &HierarchyLevel, scale: f64) -> usize {
    let h = mean_edge_length(level).max(1e-12);
    ((scale / h).powi(2).round() as usize).clamp(2, 40)
}

/// Curvature tensors estimated on a mesh *before* refinement and carried
/// onto the refined vertices through `parents` (refined vertex `n0 + k` is
/// the midpoint of `parents[k]`). Refined vertices sit on flat facets, so
/// estimating on them would see the original triangulation, not the shape.
pub fn tensors_through_refinement(
    coarse: &HierarchyLevel,
    parents: &[[u32; 2]],
    scale: f64,
) -> Vec<Matrix3<f64>> {
    let mut t = curvature_tensors(coarse, smoothing_for(coarse, scale));
    t.reserve(parents.len());
    for &[a, b] in parents {
        let m = (t[a as usize] + t[b as usize]) * 0.5;
        t.push(m);
    }
    t
}

/// Principal direction (of the larger curvature) and alignment weight from
/// a shape tensor. `scale` is the target edge length: anisotropy is
/// measured as `|k1 − k2| · scale`, so a tube of radius `r` scores
/// `scale / r`.
pub fn principal_target(s: &Matrix3<f64>, n: Vector3<f64>, scale: f64) -> (Vector3<f64>, f64) {
    let (u, v) = tangent_basis(n);
    let a = u.dot(&(s * u));
    let b = u.dot(&(s * v));
    let d = v.dot(&(s * v));
    let m = Matrix2::new(a, b, b, d);
    let eig = m.symmetric_eigen();
    let (l0, l1) = (eig.eigenvalues[0], eig.eigenvalues[1]);
    let k = if l0.abs() >= l1.abs() { 0 } else { 1 };
    let e = eig.eigenvectors.column(k);
    let dir = (u * e[0] + v * e[1]).normalize();
    let aniso = (l0 - l1).abs() * scale;
    // ramp: none below 0.2, full from 0.5 (tube radius ≈ 2 edges); torsos
    // and heads (radius ≳ 5 edges) are left to the smoothness term
    let w = ((aniso - 0.2) / 0.3).clamp(0.0, 1.0);
    if !dir.iter().all(|c| c.is_finite()) {
        return (Vector3::new(1.0, 0.0, 0.0), 0.0);
    }
    (dir, w * w)
}

/// Principal curvatures `(k_a, k_b)` of a shape tensor in the tangent plane.
fn principal_curvatures(s: &Matrix3<f64>, n: Vector3<f64>) -> (f64, f64) {
    let (u, v) = tangent_basis(n);
    let a = u.dot(&(s * u));
    let b = u.dot(&(s * v));
    let d = v.dot(&(s * v));
    let eig = Matrix2::new(a, b, b, d).symmetric_eigen();
    (eig.eigenvalues[0], eig.eigenvalues[1])
}

/// Spatially varying target edge length ("density field").
///
/// Curved detail (ears, fingers, feet, faces, the creases at elbows and
/// knees) gets smaller quads; broad, flat areas get larger ones. The raw
/// size follows `(curvedness·s0 + u0)^-γ`, is clamped to
/// `[s0/max_ratio, s0·max_ratio]`, diffused in log space (so it changes
/// gradually — no abrupt seams), and renormalised so the expected face
/// count `Σ area / h²` still matches `target_faces`.
///
/// `adaptivity` ∈ [0, 1]: 0 = uniform (every value `s0`), 1 = strong.
pub fn adaptive_scales(
    level: &HierarchyLevel,
    tensors: &[Matrix3<f64>],
    target_faces: f64,
    adaptivity: f64,
) -> Vec<f64> {
    let n = level.positions.len();
    let area: f64 = level.areas.iter().sum();
    let s0 = (area / target_faces.max(1.0)).sqrt();
    if adaptivity <= 0.0 || n == 0 || !(s0.is_finite() && s0 > 0.0) {
        return vec![s0; n];
    }
    let adaptivity = adaptivity.min(1.0);
    let gamma = 0.75 * adaptivity;
    let max_ratio = 1.0 + 2.0 * adaptivity; // up to 3x either way
    const U0: f64 = 0.15;
    let mut log_h: Vec<f64> = (0..n)
        .map(|i| {
            let (ka, kb) = principal_curvatures(&tensors[i], level.normals[i]);
            let u = (ka * ka + kb * kb).sqrt() * s0;
            let u = if u.is_finite() { u } else { 0.0 };
            -gamma * (u + U0).ln()
        })
        .collect();
    // gradual transitions: diffuse over a few edge lengths
    let passes = (2 * smoothing_for(level, s0)).clamp(4, 60);
    for _ in 0..passes {
        let prev = log_h.clone();
        for i in 0..n {
            let mut acc = prev[i] * level.areas[i];
            let mut w = level.areas[i];
            for &(j, _) in &level.adj[i] {
                acc += prev[j as usize] * level.areas[j as usize];
                w += level.areas[j as usize];
            }
            if w > 0.0 {
                log_h[i] = acc / w;
            }
        }
    }
    // normalise to the face budget, clamping the ratio to s0
    let mut shift = 0.0;
    let (lo, hi) = ((s0 / max_ratio).ln(), (s0 * max_ratio).ln());
    for _ in 0..8 {
        let faces: f64 = (0..n)
            .map(|i| level.areas[i] / (2.0 * (log_h[i] + shift).clamp(lo, hi)).exp())
            .sum();
        // faces ∝ h^-2  →  multiply h by sqrt(faces / target)
        shift += 0.5 * (faces / target_faces).ln();
    }
    log_h
        .iter()
        .map(|&l| (l + shift).clamp(lo, hi).exp())
        .collect()
}

/// Build per-level guidance for a hierarchy.
///
/// `curvature_weight` scales the curvature pull (0 disables it); `scale`
/// is the target edge length.
pub fn build_guides(
    hier: &Hierarchy,
    curvature_weight: f64,
    scale: f64,
    mirror: Option<MirrorPlane>,
    fine_tensors: Option<&[Matrix3<f64>]>,
) -> Vec<LevelGuide> {
    let n_levels = hier.levels.len();
    let mut guides: Vec<LevelGuide> = vec![LevelGuide::default(); n_levels];

    if curvature_weight > 0.0 && scale > 0.0 {
        let l0 = &hier.levels[0];
        let mut tensors = match fine_tensors {
            Some(t) if t.len() == l0.positions.len() => t.to_vec(),
            _ => curvature_tensors(l0, smoothing_for(l0, scale)),
        };
        let mut areas = l0.areas.clone();
        for (l, guide) in guides.iter_mut().enumerate() {
            let lv = &hier.levels[l];
            guide.target = (0..lv.positions.len())
                .map(|i| {
                    let (d, w) = principal_target(&tensors[i], lv.normals[i], scale);
                    (d, w * curvature_weight)
                })
                .collect();
            if l + 1 < n_levels {
                // restrict: area-weighted mean of the (up to two) children
                let up = &hier.to_upper[l];
                let mut next_t = Vec::with_capacity(up.len());
                let mut next_a = Vec::with_capacity(up.len());
                for &[a, b] in up {
                    let mut t = tensors[a as usize] * areas[a as usize];
                    let mut w = areas[a as usize];
                    if b != INVALID {
                        t += tensors[b as usize] * areas[b as usize];
                        w += areas[b as usize];
                    }
                    next_t.push(if w > 0.0 { t / w } else { t });
                    next_a.push(w);
                }
                tensors = next_t;
                areas = next_a;
            }
        }
    }
    if let Some(plane) = mirror {
        for (l, g) in guides.iter_mut().enumerate() {
            let lv = &hier.levels[l];
            g.partner = mirror_partners(lv, plane);
            g.plane = Some(plane);
            seam_alignment(lv, plane, &mut g.target);
        }
    }
    guides
}

/// Weight of the "edges along the mirror plane" prior.
const SEAM_WEIGHT: f64 = 0.5;

/// A reflection maps a 4-RoSy cross onto itself both when it is aligned
/// with the mirror plane and when it is at 45° to it, so mirror coupling
/// alone leaves flat symmetric regions (a torso) free to settle on the
/// diagonal — and which one wins flips with tiny changes to the input.
/// Pin the cross near the plane to the plane (a centre edge loop, as an
/// artist would draw), where curvature does not already prescribe one.
fn seam_alignment(lv: &HierarchyLevel, plane: MirrorPlane, target: &mut Vec<(Vector3<f64>, f64)>) {
    let n = lv.positions.len();
    if target.len() != n {
        target.clear();
        target.resize(n, (Vector3::zeros(), 0.0));
    }
    let band = 1.5 * mean_edge_length(lv);
    if band.is_nan() || band <= 0.0 {
        return;
    }
    let mut axis = Vector3::zeros();
    axis[plane.axis] = 1.0;
    for (i, tgt) in target.iter_mut().enumerate() {
        let d = (lv.positions[i][plane.axis] - plane.offset).abs();
        if d >= band {
            continue;
        }
        let nrm = lv.normals[i];
        let t = axis - nrm * nrm.dot(&axis);
        let tl = t.norm();
        if tl < 0.5 {
            continue; // surface nearly parallel to the plane: no seam here
        }
        let w = SEAM_WEIGHT * (1.0 - d / band) * tl;
        if w > tgt.1 {
            *tgt = (t / tl, w);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hierarchy::level_from_mesh;

    /// Closed cylinder along Z (radius r, length len) with caps.
    fn tube(r: f64, len: f64, ring: usize, rows: usize) -> (Vec<[f64; 3]>, Vec<[u32; 3]>) {
        let mut p = Vec::new();
        for k in 0..=rows {
            for i in 0..ring {
                let a = i as f64 / ring as f64 * std::f64::consts::TAU;
                p.push([
                    r * a.cos(),
                    r * a.sin(),
                    len * k as f64 / rows as f64 - len / 2.0,
                ]);
            }
        }
        let mut t = Vec::new();
        let id = |k: usize, i: usize| (k * ring + i % ring) as u32;
        for k in 0..rows {
            for i in 0..ring {
                t.push([id(k, i), id(k, i + 1), id(k + 1, i + 1)]);
                t.push([id(k, i), id(k + 1, i + 1), id(k + 1, i)]);
            }
        }
        let bottom = p.len() as u32;
        p.push([0.0, 0.0, -len / 2.0]);
        let top = p.len() as u32;
        p.push([0.0, 0.0, len / 2.0]);
        for i in 0..ring {
            t.push([bottom, id(0, i + 1), id(0, i)]);
            t.push([top, id(rows, i), id(rows, i + 1)]);
        }
        (p, t)
    }

    #[test]
    fn tube_principal_direction_is_circumferential_or_axial() {
        let (p, t) = tube(0.5, 4.0, 32, 40);
        let lv = level_from_mesh(&p, &t);
        let s = curvature_tensors(&lv, 2);
        // a mid-tube vertex
        let i = 20 * 32 + 3;
        let (d, w) = principal_target(&s[i], lv.normals[i], 0.25);
        assert!(w > 0.5, "weight {w}");
        // 4-RoSy: aligned with axis or with circumference — both fine
        let axial = d.z.abs();
        assert!(axial > 0.95 || axial < 0.05, "direction {d:?}");
    }

    #[test]
    fn density_puts_more_faces_on_thin_parts() {
        // thick tube (r = 1) joined in series with a thin one (r = 0.25):
        // build both, measure mean scale per part
        let (p1, t1) = tube(1.0, 4.0, 48, 24);
        let (mut p2, t2) = tube(0.25, 4.0, 24, 48);
        for v in &mut p2 {
            v[0] += 5.0;
        }
        let off = p1.len() as u32;
        let mut p = p1.clone();
        p.extend(p2);
        let mut t = t1.clone();
        t.extend(t2.iter().map(|f| [f[0] + off, f[1] + off, f[2] + off]));
        let lv = level_from_mesh(&p, &t);
        let tens = curvature_tensors(&lv, 2);
        let h = adaptive_scales(&lv, &tens, 1500.0, 1.0);
        let mean =
            |r: std::ops::Range<usize>| r.clone().map(|i| h[i]).sum::<f64>() / r.len() as f64;
        let (thick, thin) = (mean(0..p1.len()), mean(p1.len()..p.len()));
        assert!(thin < thick * 0.75, "thin {thin} thick {thick}");
        // face budget preserved (within the clamp's slack)
        let faces: f64 = (0..p.len()).map(|i| lv.areas[i] / (h[i] * h[i])).sum();
        assert!((faces / 1500.0 - 1.0).abs() < 0.1, "faces {faces}");
        // adaptivity 0 is uniform
        let u = adaptive_scales(&lv, &tens, 1500.0, 0.0);
        assert!(u.iter().all(|&x| (x - u[0]).abs() < 1e-12));
    }

    #[test]
    fn detects_symmetry_plane_of_offset_tube() {
        let (mut p, t) = tube(0.5, 4.0, 32, 40);
        for v in &mut p {
            v[0] += 2.0;
        }
        let lv = level_from_mesh(&p, &t);
        let plane = detect_mirror(&lv).expect("symmetric");
        // all three axis planes are symmetries of a tube; whichever wins,
        // the offset must match the centre on that axis
        let centre = [2.0, 0.0, 0.0];
        assert!(
            (plane.offset - centre[plane.axis]).abs() < 1e-6,
            "{plane:?}"
        );
        let partners = mirror_partners(&lv, plane);
        assert!(partners.iter().all(|&j| j != INVALID));
    }
}
