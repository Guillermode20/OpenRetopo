//! Output polishing: merge triangle pairs into quads, then relax vertices
//! tangentially and project them back onto the input surface.
//!
//! Both steps leave topology valid by construction (merges are only of two
//! triangles sharing one edge into a convex, non-degenerate quad; the
//! relaxation never changes connectivity and reverts any vertex whose move
//! would fold a face). Callers still re-validate.

use std::collections::HashMap;

use nalgebra::Vector3;

use crate::guidance::MirrorPlane;

type V3 = Vector3<f64>;

fn v(p: [f32; 3]) -> V3 {
    V3::new(p[0] as f64, p[1] as f64, p[2] as f64)
}

fn face_normal(pts: &[V3]) -> V3 {
    // Newell normal (robust for quads)
    let mut n = V3::zeros();
    for k in 0..pts.len() {
        let (a, b) = (pts[k], pts[(k + 1) % pts.len()]);
        n += a.cross(&b);
    }
    n
}

/// Worst corner deviation from 90° (radians) of the quad `q`, or `None` if
/// it is non-convex / degenerate / folded relative to `ref_n`.
fn quad_quality(q: [V3; 4], ref_n: V3) -> Option<f64> {
    let mut worst = 0.0f64;
    for k in 0..4 {
        let (p, a, b) = (q[k], q[(k + 3) % 4], q[(k + 1) % 4]);
        let (e0, e1) = (a - p, b - p);
        let (l0, l1) = (e0.norm(), e1.norm());
        if l0 < 1e-12 || l1 < 1e-12 {
            return None;
        }
        // convex: every corner turns the same way as the face normal
        if e1.cross(&e0).dot(&ref_n) <= 0.0 {
            return None;
        }
        let ang = (e0.dot(&e1) / (l0 * l1)).clamp(-1.0, 1.0).acos();
        worst = worst.max((ang - std::f64::consts::FRAC_PI_2).abs());
    }
    Some(worst)
}

/// Merge pairs of adjacent triangles into quads, best-shaped first. A merge
/// is accepted only if the quad is convex and no corner deviates from 90°
/// by more than 50°. Pairs whose shared edge lies on `seam` (a mirror
/// plane) are left alone so the halves stay mirror images.
pub fn merge_triangles(
    positions: &[[f32; 3]],
    faces: &mut Vec<Vec<u32>>,
    seam: Option<MirrorPlane>,
) {
    let mut edge_tris: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    for (fi, f) in faces.iter().enumerate() {
        if f.len() != 3 {
            continue;
        }
        for k in 0..3 {
            let (a, b) = (f[k], f[(k + 1) % 3]);
            edge_tris.entry((a.min(b), a.max(b))).or_default().push(fi);
        }
    }
    let on_seam = |a: u32, b: u32| -> bool {
        seam.is_some_and(|s| {
            let (pa, pb) = (positions[a as usize], positions[b as usize]);
            (pa[s.axis] as f64 - s.offset).abs() < 1e-6
                && (pb[s.axis] as f64 - s.offset).abs() < 1e-6
        })
    };
    let mut candidates: Vec<(f64, usize, usize, [u32; 4])> = Vec::new();
    let mut keys: Vec<_> = edge_tris.keys().copied().collect();
    keys.sort_unstable(); // determinism
    for (a, b) in keys {
        let ts = &edge_tris[&(a, b)];
        if ts.len() != 2 || on_seam(a, b) {
            continue;
        }
        let (f0, f1) = (&faces[ts[0]], &faces[ts[1]]);
        // rotate f0 so it reads (x, y, c) with (x, y) the shared edge
        let k0 = (0..3).find(|&k| {
            let (x, y) = (f0[k], f0[(k + 1) % 3]);
            (x == a && y == b) || (x == b && y == a)
        });
        let Some(k0) = k0 else { continue };
        let (x, y, c) = (f0[k0], f0[(k0 + 1) % 3], f0[(k0 + 2) % 3]);
        // f1 must contain (y, x) and an opposite vertex d
        let Some(d) = f1.iter().copied().find(|&w| w != x && w != y) else {
            continue;
        };
        if d == c {
            continue;
        }
        let quad = [x, d, y, c];
        let pts = quad.map(|i| v(positions[i as usize]));
        let n = face_normal(&[
            v(positions[x as usize]),
            v(positions[y as usize]),
            v(positions[c as usize]),
        ]) + face_normal(&[
            v(positions[y as usize]),
            v(positions[x as usize]),
            v(positions[d as usize]),
        ]);
        if let Some(q) = quad_quality(pts, n) {
            if q < 50f64.to_radians() {
                candidates.push((q, ts[0], ts[1], quad));
            }
        }
    }
    candidates.sort_by(|p, q| p.0.total_cmp(&q.0).then(p.1.cmp(&q.1)));
    let mut used = vec![false; faces.len()];
    let mut replace: Vec<Option<Vec<u32>>> = vec![None; faces.len()];
    for (_, t0, t1, quad) in candidates {
        if used[t0] || used[t1] {
            continue;
        }
        used[t0] = true;
        used[t1] = true;
        replace[t0] = Some(quad.to_vec());
    }
    let old = std::mem::take(faces);
    for (fi, f) in old.into_iter().enumerate() {
        match (&replace[fi], used[fi]) {
            (Some(q), _) => faces.push(q.clone()),
            (None, true) => {}
            (None, false) => faces.push(f),
        }
    }
}

/// Spatial hash of the input triangles for closest-point projection.
pub struct SurfaceIndex<'a> {
    pos: &'a [[f64; 3]],
    tris: &'a [[u32; 3]],
    cell: f64,
    grid: HashMap<(i64, i64, i64), Vec<u32>>,
}

impl<'a> SurfaceIndex<'a> {
    pub fn new(pos: &'a [[f64; 3]], tris: &'a [[u32; 3]], cell: f64) -> Self {
        let mut grid: HashMap<(i64, i64, i64), Vec<u32>> = HashMap::new();
        let key = |x: f64| (x / cell).floor() as i64;
        for (ti, t) in tris.iter().enumerate() {
            let mut lo = [f64::INFINITY; 3];
            let mut hi = [f64::NEG_INFINITY; 3];
            for &k in t {
                for c in 0..3 {
                    lo[c] = lo[c].min(pos[k as usize][c]);
                    hi[c] = hi[c].max(pos[k as usize][c]);
                }
            }
            for x in key(lo[0])..=key(hi[0]) {
                for y in key(lo[1])..=key(hi[1]) {
                    for z in key(lo[2])..=key(hi[2]) {
                        grid.entry((x, y, z)).or_default().push(ti as u32);
                    }
                }
            }
        }
        SurfaceIndex {
            pos,
            tris,
            cell,
            grid,
        }
    }

    /// Closest surface point to `p` within about one cell, preferring
    /// triangles whose normal agrees with `n` (so a point is never pulled
    /// through a thin part onto its back side).
    pub fn project(&self, p: V3, n: V3) -> Option<V3> {
        let k = |x: f64| (x / self.cell).floor() as i64;
        let (cx, cy, cz) = (k(p.x), k(p.y), k(p.z));
        let mut best: Option<(f64, V3)> = None;
        for x in cx - 1..=cx + 1 {
            for y in cy - 1..=cy + 1 {
                for z in cz - 1..=cz + 1 {
                    let Some(list) = self.grid.get(&(x, y, z)) else {
                        continue;
                    };
                    for &ti in list {
                        let t = self.tris[ti as usize];
                        let a = V3::from(self.pos[t[0] as usize]);
                        let b = V3::from(self.pos[t[1] as usize]);
                        let c = V3::from(self.pos[t[2] as usize]);
                        if (b - a).cross(&(c - a)).dot(&n) <= 0.0 {
                            continue;
                        }
                        let q = closest_on_triangle(p, a, b, c);
                        let d = (q - p).norm_squared();
                        if best.is_none_or(|(bd, _)| d < bd) {
                            best = Some((d, q));
                        }
                    }
                }
            }
        }
        best.map(|(_, q)| q)
    }
}

/// Ericson, Real-Time Collision Detection §5.1.5.
fn closest_on_triangle(p: V3, a: V3, b: V3, c: V3) -> V3 {
    let (ab, ac, ap) = (b - a, c - a, p - a);
    let (d1, d2) = (ab.dot(&ap), ac.dot(&ap));
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bp = p - b;
    let (d3, d4) = (ab.dot(&bp), ac.dot(&bp));
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return a + ab * (d1 / (d1 - d3));
    }
    let cp = p - c;
    let (d5, d6) = (ab.dot(&cp), ac.dot(&cp));
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return a + ac * (d2 / (d2 - d6));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
    }
    let denom = 1.0 / (va + vb + vc);
    a + ab * (vb * denom) + ac * (vc * denom)
}

/// Tangential relaxation: each vertex moves toward the mean of its incident
/// face centroids (tangential part only), then onto the input surface.
/// Vertices on `seam` stay on the plane. A vertex whose move would turn an
/// incident face by more than ~70° or shrink it below 10% is reverted.
pub fn relax(
    positions: &mut [[f32; 3]],
    faces: &[Vec<u32>],
    surface: &SurfaceIndex,
    seam: Option<MirrorPlane>,
    iterations: usize,
) {
    let n = positions.len();
    let mut incident: Vec<Vec<u32>> = vec![Vec::new(); n];
    for (fi, f) in faces.iter().enumerate() {
        for &vtx in f {
            incident[vtx as usize].push(fi as u32);
        }
    }
    let on_seam: Vec<bool> = positions
        .iter()
        .map(|p| seam.is_some_and(|s| (p[s.axis] as f64 - s.offset).abs() < 1e-6))
        .collect();
    let normals_of = |pos: &[V3]| -> Vec<V3> {
        faces
            .iter()
            .map(|f| face_normal(&f.iter().map(|&i| pos[i as usize]).collect::<Vec<_>>()))
            .collect()
    };
    let mut pos: Vec<V3> = positions.iter().map(|&p| v(p)).collect();
    for _ in 0..iterations {
        let fnorm = normals_of(&pos);
        let centroids: Vec<V3> = faces
            .iter()
            .map(|f| f.iter().map(|&i| pos[i as usize]).sum::<V3>() / f.len() as f64)
            .collect();
        let mut next = pos.clone();
        for i in 0..n {
            if incident[i].is_empty() {
                continue;
            }
            let mut nrm = V3::zeros();
            let mut c = V3::zeros();
            for &fi in &incident[i] {
                nrm += fnorm[fi as usize];
                c += centroids[fi as usize];
            }
            let nl = nrm.norm();
            if nl < 1e-20 {
                continue;
            }
            let nrm = nrm / nl;
            c /= incident[i].len() as f64;
            let d = c - pos[i];
            let mut p = pos[i] + (d - nrm * nrm.dot(&d)) * 0.5;
            if let Some(q) = surface.project(p, nrm) {
                p = q;
            }
            if let Some(s) = seam {
                if on_seam[i] {
                    p[s.axis] = s.offset;
                }
            }
            if p.iter().all(|x| x.is_finite()) {
                next[i] = p;
            }
        }
        // revert vertices of faces that folded or collapsed
        let new_norm = normals_of(&next);
        let mut revert = vec![false; n];
        for (fi, f) in faces.iter().enumerate() {
            let (a, b) = (fnorm[fi], new_norm[fi]);
            let (la, lb) = (a.norm(), b.norm());
            if lb < 0.1 * la || a.dot(&b) < 0.34 * la * lb {
                for &vtx in f {
                    revert[vtx as usize] = true;
                }
            }
        }
        for i in 0..n {
            if !revert[i] {
                pos[i] = next[i];
            }
        }
    }
    for (dst, p) in positions.iter_mut().zip(&pos) {
        *dst = [p.x as f32, p.y as f32, p.z as f32];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merges_two_triangles_into_a_square() {
        let pos = [
            [0.0f32, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let mut faces = vec![vec![0, 1, 2], vec![0, 2, 3]];
        merge_triangles(&pos, &mut faces, None);
        assert_eq!(faces.len(), 1);
        assert_eq!(faces[0].len(), 4);
        // orientation preserved: Newell normal points +z
        let n = face_normal(
            &faces[0]
                .iter()
                .map(|&i| v(pos[i as usize]))
                .collect::<Vec<_>>(),
        );
        assert!(n.z > 0.0);
    }

    #[test]
    fn does_not_merge_into_a_sliver() {
        let pos = [
            [0.0f32, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.05, 0.0],
            [1.0, 1.0, 0.0],
        ];
        let mut faces = vec![vec![0, 1, 3], vec![1, 2, 3]];
        merge_triangles(&pos, &mut faces, None);
        assert_eq!(faces.len(), 2);
    }
}
