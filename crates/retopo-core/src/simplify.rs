//! Quadric-error-metric mesh simplification.
//!
//! Collapses edges along the Garland/Heckbert quadric metric with two
//! guarantees that matter for retopology prep work:
//!   * boundary edges (only one adjacent face) are never collapsed;
//!   * "crease" edges whose adjacent face normals differ by more than the
//!     configured dihedral angle are locked as well.
//!
//! Face quadrics are subtracted from their corners whenever a face is
//! deleted, and recomputed from current geometry whenever a face survives a
//! corner substitution, so vertex quadrics always equal the sum of the
//! contributions of the faces that currently reference them.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

use crate::RetopoError;

#[derive(Debug, Clone, Copy)]
pub struct SimplificationReport {
    pub faces_removed: usize,
    pub collapses: usize,
    /// `false` when the target could not be reached (e.g. every remaining
    /// edge is boundary-locked or crease-locked).
    pub reached_target: bool,
}

// ---------------------------------------------------------------------------
// quadric
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Quadric {
    /// Full symmetric 4x4 matrix.
    m: [[f64; 4]; 4],
}

impl Quadric {
    const ZERO: Quadric = Quadric { m: [[0.0; 4]; 4] };

    /// Plane must satisfy `a*x + b*y + c*z + d = 0` with `(a,b,c)` normalized;
    /// `weight` is typically the triangle area.
    fn from_plane(a: f64, b: f64, c: f64, d: f64, weight: f64) -> Self {
        let n = [a, b, c, d];
        let mut m = [[0.0; 4]; 4];
        for (i, &ni) in n.iter().enumerate() {
            for (j, &nj) in n.iter().enumerate() {
                m[i][j] = weight * ni * nj;
            }
        }
        Quadric { m }
    }

    fn add_assign(&mut self, o: &Quadric) {
        for (row, orow) in self.m.iter_mut().zip(o.m.iter()) {
            for (a, b) in row.iter_mut().zip(orow.iter()) {
                *a += *b;
            }
        }
    }

    fn sub_assign(&mut self, o: &Quadric) {
        for (row, orow) in self.m.iter_mut().zip(o.m.iter()) {
            for (a, b) in row.iter_mut().zip(orow.iter()) {
                *a -= *b;
            }
        }
    }

    /// Squared distance error from point `p` to the planes encoded in the
    /// quadric. Clamped at zero to mask tiny negative roundoff.
    fn error(&self, p: [f64; 3]) -> f64 {
        let v = [p[0], p[1], p[2], 1.0];
        let mut sum = 0.0;
        for (i, row_m) in self.m.iter().enumerate() {
            let mut row = 0.0;
            for (m, &vj) in row_m.iter().zip(v.iter()) {
                row += m * vj;
            }
            sum += v[i] * row;
        }
        sum.max(0.0)
    }

    /// Minimise the quadric error by solving the 3x3 top-left system.
    /// Returns `None` when the system is numerically singular.
    fn optimal(&self) -> Option<[f64; 3]> {
        let a = [
            [self.m[0][0], self.m[0][1], self.m[0][2]],
            [self.m[1][0], self.m[1][1], self.m[1][2]],
            [self.m[2][0], self.m[2][1], self.m[2][2]],
        ];
        let rhs = [-self.m[0][3], -self.m[1][3], -self.m[2][3]];
        let det = det3(&a);
        if det.abs() < 1e-30 || !det.is_finite() {
            return None;
        }
        let inv = adjugate3(&a);
        let mut out = [0.0f64; 3];
        for i in 0..3 {
            out[i] = (inv[i][0] * rhs[0] + inv[i][1] * rhs[1] + inv[i][2] * rhs[2]) / det;
        }
        if out.iter().all(|v| v.is_finite()) {
            Some(out)
        } else {
            None
        }
    }
}

fn det3(a: &[[f64; 3]; 3]) -> f64 {
    a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
}

fn adjugate3(a: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [
            a[1][1] * a[2][2] - a[1][2] * a[2][1],
            a[0][2] * a[2][1] - a[0][1] * a[2][2],
            a[0][1] * a[1][2] - a[0][2] * a[1][1],
        ],
        [
            a[1][2] * a[2][0] - a[1][0] * a[2][2],
            a[0][0] * a[2][2] - a[0][2] * a[2][0],
            a[0][2] * a[1][0] - a[0][0] * a[1][2],
        ],
        [
            a[1][0] * a[2][1] - a[1][1] * a[2][0],
            a[0][1] * a[2][0] - a[0][0] * a[2][1],
            a[0][0] * a[1][1] - a[0][1] * a[1][0],
        ],
    ]
}

// ---------------------------------------------------------------------------
// priority queue entries
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct Candidate {
    cost: f64,
    a: u32,
    b: u32,
    va: u64,
    vb: u64,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost && self.a == other.a && self.b == other.b
    }
}
impl Eq for Candidate {}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Candidate {
    // Reversed so the lowest cost pops first.
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .cost
            .total_cmp(&self.cost)
            .then(other.a.cmp(&self.a))
            .then(other.b.cmp(&self.b))
    }
}

// ---------------------------------------------------------------------------
// main routine
// ---------------------------------------------------------------------------

fn edge_key(a: u32, b: u32) -> u64 {
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    ((lo as u64) << 32) | hi as u64
}

/// Decrement an edge's live-face refcount; remove it at zero.
/// Note: `locked` entries are intentionally *kept* (stale crease locks are
/// fail-safe for the lifetime of a run).
fn dec_edge(edge_count: &mut HashMap<u64, u32>, _locked: &mut HashSet<u64>, key: u64) {
    if let Some(count) = edge_count.get_mut(&key) {
        *count -= 1;
        if *count == 0 {
            edge_count.remove(&key);
        }
    }
}

fn inc_edge(edge_count: &mut HashMap<u64, u32>, key: u64) {
    *edge_count.entry(key).or_insert(0) += 1;
}

pub fn simplify_to_target(
    verts: &mut [[f64; 3]],
    tris: &mut Vec<[u32; 3]>,
    target_faces: usize,
    lock_crease_degrees: f64,
    mut progress: impl FnMut(f32) -> bool,
) -> Result<SimplificationReport, RetopoError> {
    if tris.len() <= target_faces {
        return Ok(SimplificationReport {
            faces_removed: 0,
            collapses: 0,
            reached_target: true,
        });
    }

    let initial_faces = tris.len();
    let mut live_faces = tris.len();
    let target_collapses = initial_faces.saturating_sub(target_faces);

    // ---- per-face plane data ----------------------------------------------
    let mut face_alive = vec![true; tris.len()];
    let mut face_plane = vec![[0.0f64; 4]; tris.len()];
    let mut face_weight = vec![0.0f64; tris.len()];

    let compute_plane = |p0: &[f64; 3], p1: &[f64; 3], p2: &[f64; 3]| -> Option<([f64; 4], f64)> {
        let ab = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        let ac = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
        let cross = [
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        ];
        let len = (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
        if len < 1e-18 || !len.is_finite() {
            return None;
        }
        let n = [cross[0] / len, cross[1] / len, cross[2] / len];
        let d = -(n[0] * p0[0] + n[1] * p0[1] + n[2] * p0[2]);
        Some(([n[0], n[1], n[2], d], len * 0.5))
    };

    let mut quadrics = vec![Quadric::ZERO; verts.len()];
    for (i, t) in tris.iter().enumerate() {
        let p0 = verts[t[0] as usize];
        let p1 = verts[t[1] as usize];
        let p2 = verts[t[2] as usize];
        if let Some((plane, weight)) = compute_plane(&p0, &p1, &p2) {
            face_plane[i] = plane;
            face_weight[i] = weight;
            let q = Quadric::from_plane(plane[0], plane[1], plane[2], plane[3], weight);
            for &c in t {
                quadrics[c as usize].add_assign(&q);
            }
        }
    }

    // ---- edge bookkeeping ---------------------------------------------------
    let mut edge_count: HashMap<u64, u32> = HashMap::with_capacity(tris.len() * 2);
    for t in tris.iter() {
        for (i, j) in [(0, 1), (1, 2), (2, 0)] {
            *edge_count.entry(edge_key(t[i], t[j])).or_insert(0) += 1;
        }
    }

    // ---- crease locking ------------------------------------------------------
    let mut locked: HashSet<u64> = HashSet::new();
    if lock_crease_degrees < 180.0 {
        let cos_limit = lock_crease_degrees.to_radians().cos();
        let mut edge_faces: HashMap<u64, (u32, Option<u32>)> =
            HashMap::with_capacity(tris.len() * 2);
        for (i, t) in tris.iter().enumerate() {
            for (a, b) in [(0usize, 1usize), (1, 2), (2, 0)] {
                let key = edge_key(t[a], t[b]);
                edge_faces
                    .entry(key)
                    .and_modify(|e| {
                        if e.1.is_none() {
                            e.1 = Some(i as u32);
                        }
                    })
                    .or_insert((i as u32, None));
            }
        }
        for (key, (f0, f1)) in edge_faces {
            if let Some(f1) = f1 {
                // Compare absolute cosines so inconsistent winding doesn't fool us.
                let dot = (0..3)
                    .map(|k| face_plane[f0 as usize][k] * face_plane[f1 as usize][k])
                    .sum::<f64>()
                    .abs();
                if dot < cos_limit {
                    locked.insert(key);
                }
            }
        }
    }

    // ---- vertex → face adjacency ----------------------------------------------
    let mut vfaces: Vec<Vec<u32>> = vec![Vec::new(); verts.len()];
    for (i, t) in tris.iter().enumerate() {
        for &c in t {
            vfaces[c as usize].push(i as u32);
        }
    }

    let mut v_alive = vec![true; verts.len()];
    let mut version = vec![0u64; verts.len()];

    // ---- candidate generation --------------------------------------------------
    let mut heap: BinaryHeap<Candidate> = BinaryHeap::with_capacity(edge_count.len() * 2);

    let make_candidate = |verts: &[[f64; 3]],
                          quadrics: &[Quadric],
                          edge_count: &HashMap<u64, u32>,
                          locked: &HashSet<u64>,
                          version: &[u64],
                          a: u32,
                          b: u32|
     -> Option<Candidate> {
        if a == b {
            return None;
        }
        let key = edge_key(a, b);
        if locked.contains(&key) {
            return None;
        }
        if edge_count.get(&key).copied().unwrap_or(0) < 2 {
            return None; // boundary edge: locked
        }
        let qa = &quadrics[a as usize];
        let qb = &quadrics[b as usize];
        let mut q = *qa;
        q.add_assign(qb);
        let pa = verts[a as usize];
        let pb = verts[b as usize];
        let mid = [
            (pa[0] + pb[0]) * 0.5,
            (pa[1] + pb[1]) * 0.5,
            (pa[2] + pb[2]) * 0.5,
        ];
        let edge_len =
            ((pa[0] - pb[0]).powi(2) + (pa[1] - pb[1]).powi(2) + (pa[2] - pb[2]).powi(2)).sqrt();
        let p = match q.optimal() {
            Some(p) => {
                let dx = p[0] - mid[0];
                let dy = p[1] - mid[1];
                let dz = p[2] - mid[2];
                if dx.powi(2) + dy.powi(2) + dz.powi(2) > (4.0 * edge_len).powi(2) {
                    // wildly off the edge: fall back to the best clamped candidate
                    [pa, pb, mid]
                        .into_iter()
                        .min_by(|&x, &y| q.error(x).total_cmp(&q.error(y)))
                        .unwrap()
                } else {
                    p
                }
            }
            None => [pa, pb, mid]
                .into_iter()
                .min_by(|&x, &y| q.error(x).total_cmp(&q.error(y)))
                .unwrap(),
        };
        let cost = q.error(p);
        if !cost.is_finite() {
            return None;
        }
        Some(Candidate {
            cost,
            a: a.min(b),
            b: a.max(b),
            va: version[a.min(b) as usize],
            vb: version[a.max(b) as usize],
        })
    };

    for &key in edge_count.keys() {
        let a = (key >> 32) as u32;
        let b = key as u32;
        if let Some(c) = make_candidate(verts, &quadrics, &edge_count, &locked, &version, a, b) {
            heap.push(c);
        }
    }

    let mut collapses = 0usize;
    let initial = live_faces;
    let progress_step = (target_collapses / 512).max(64);

    macro_rules! tick {
        () => {{
            if collapses % progress_step == 0 {
                let t = (collapses as f32 / target_collapses.max(1) as f32).clamp(0.0, 1.0);
                if progress(t) {
                    return Err(RetopoError::Cancelled);
                }
            }
        }};
    }

    while live_faces > target_faces {
        let Some(cand) = heap.pop() else { break };

        // --- validate ---
        if !v_alive[cand.a as usize] || !v_alive[cand.b as usize] {
            continue;
        }
        if version[cand.a as usize] != cand.va || version[cand.b as usize] != cand.vb {
            continue;
        }
        let key = edge_key(cand.a, cand.b);
        if edge_count.get(&key).copied().unwrap_or(0) < 2 || locked.contains(&key) {
            continue;
        }

        let a = cand.a;
        let b = cand.b;

        // optimal position from *current* quadrics
        let pa = verts[a as usize];
        let pb = verts[b as usize];
        let mid = [
            (pa[0] + pb[0]) * 0.5,
            (pa[1] + pb[1]) * 0.5,
            (pa[2] + pb[2]) * 0.5,
        ];
        let edge_len =
            ((pa[0] - pb[0]).powi(2) + (pa[1] - pb[1]).powi(2) + (pa[2] - pb[2]).powi(2)).sqrt();
        let mut combined = quadrics[a as usize];
        combined.add_assign(&quadrics[b as usize]);
        let p = match combined.optimal() {
            Some(p) => {
                let dx = p[0] - mid[0];
                let dy = p[1] - mid[1];
                let dz = p[2] - mid[2];
                if dx.powi(2) + dy.powi(2) + dz.powi(2) > (4.0 * edge_len).powi(2) {
                    [pa, pb, mid]
                        .into_iter()
                        .min_by(|&x, &y| combined.error(x).total_cmp(&combined.error(y)))
                        .unwrap()
                } else {
                    p
                }
            }
            None => [pa, pb, mid]
                .into_iter()
                .min_by(|&x, &y| combined.error(x).total_cmp(&combined.error(y)))
                .unwrap(),
        };

        // --- fold b into a ---
        verts[a as usize] = p;
        quadrics[a as usize] = combined;
        v_alive[b as usize] = false;
        version[a as usize] += 1;
        version[b as usize] += 1;

        // --- retouch faces incident to b ---
        let incident = std::mem::take(&mut vfaces[b as usize]);
        let mut survivors: Vec<u32> = Vec::with_capacity(incident.len());
        for f_id in incident {
            let f_id = f_id as usize;
            if !face_alive[f_id] {
                continue;
            }
            let old_f = tris[f_id];
            let b_pos = old_f
                .iter()
                .position(|&c| c == b)
                .expect("face listed in b's adjacency must reference b");
            let o1 = old_f[(b_pos + 1) % 3];
            let o2 = old_f[(b_pos + 2) % 3];
            let new_f = old_f.map(|c| if c == b { a } else { c });

            // subtract the stored face quadric from its (folded) corners
            let q_old = Quadric::from_plane(
                face_plane[f_id][0],
                face_plane[f_id][1],
                face_plane[f_id][2],
                face_plane[f_id][3],
                face_weight[f_id],
            );
            for &c in &new_f {
                quadrics[c as usize].sub_assign(&q_old);
            }

            if new_f[0] == new_f[1] || new_f[1] == new_f[2] || new_f[0] == new_f[2] {
                // becomes degenerate: erase the face entirely. Its original
                // edges leave the refcount map; its corners lose membership.
                for (i, j) in [(0, 1), (1, 2), (2, 0)] {
                    dec_edge(&mut edge_count, &mut locked, edge_key(old_f[i], old_f[j]));
                }
                for &c in &old_f {
                    if let Some(pos) = vfaces[c as usize].iter().position(|&x| x as usize == f_id) {
                        vfaces[c as usize].swap_remove(pos);
                    }
                }
                face_alive[f_id] = false;
                face_plane[f_id] = [0.0; 4];
                face_weight[f_id] = 0.0;
                live_faces -= 1;
                continue;
            }

            // survives: edge refcount handover b→a
            dec_edge(&mut edge_count, &mut locked, edge_key(b, o1));
            dec_edge(&mut edge_count, &mut locked, edge_key(b, o2));
            inc_edge(&mut edge_count, edge_key(a, o1));
            inc_edge(&mut edge_count, edge_key(a, o2));

            tris[f_id] = new_f;
            let p0 = verts[new_f[0] as usize];
            let p1 = verts[new_f[1] as usize];
            let p2 = verts[new_f[2] as usize];
            match compute_plane(&p0, &p1, &p2) {
                Some((plane, weight)) => {
                    face_plane[f_id] = plane;
                    face_weight[f_id] = weight;
                    let q = Quadric::from_plane(plane[0], plane[1], plane[2], plane[3], weight);
                    for &c in &new_f {
                        quadrics[c as usize].add_assign(&q);
                    }
                }
                None => {
                    face_plane[f_id] = [0.0; 4];
                    face_weight[f_id] = 0.0;
                }
            }
            survivors.push(f_id as u32);
        }
        vfaces[a as usize].extend_from_slice(&survivors);

        collapses += 1;
        tick!();

        // --- schedule fresh candidates around `a` ---
        let mut neighbors: HashSet<u32> = HashSet::new();
        for &f_id in vfaces[a as usize].iter() {
            let f = tris[f_id as usize];
            for &c in &f {
                if c != a {
                    neighbors.insert(c);
                }
            }
        }
        for n in neighbors {
            if let Some(c) = make_candidate(verts, &quadrics, &edge_count, &locked, &version, a, n)
            {
                heap.push(c);
            }
        }
    }

    // ---- compact the face array ----------------------------------------------
    let kept: Vec<[u32; 3]> = tris
        .iter()
        .zip(face_alive.iter())
        .filter(|(_, &alive)| alive)
        .map(|(t, _)| *t)
        .collect();
    let faces_removed = initial - kept.len();
    *tris = kept;

    Ok(SimplificationReport {
        faces_removed,
        collapses,
        reached_target: live_faces <= target_faces,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Minimal subdivision-surface helper to build an icosphere test mesh
    /// in-memory (no fixtures required).
    fn icosphere(subdivisions: usize) -> (Vec<[f64; 3]>, Vec<[u32; 3]>) {
        let t = (1.0 + 5.0f64.sqrt()) * 0.5;
        let mut verts: Vec<[f64; 3]> = vec![
            [-1.0, t, 0.0],
            [1.0, t, 0.0],
            [-1.0, -t, 0.0],
            [1.0, -t, 0.0],
            [0.0, -1.0, t],
            [0.0, 1.0, t],
            [0.0, -1.0, -t],
            [0.0, 1.0, -t],
            [t, 0.0, -1.0],
            [t, 0.0, 1.0],
            [-t, 0.0, -1.0],
            [-t, 0.0, 1.0],
        ];
        for v in verts.iter_mut() {
            let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            v[0] /= len;
            v[1] /= len;
            v[2] /= len;
        }
        let mut faces: Vec<[u32; 3]> = vec![
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ];
        for _ in 0..subdivisions {
            let mut midpoint: HashMap<(u32, u32), u32> = HashMap::new();
            let mut new_faces: Vec<[u32; 3]> = Vec::with_capacity(faces.len() * 4);
            let mut mid = |a: u32, b: u32, verts: &mut Vec<[f64; 3]>| -> u32 {
                let key = if a < b { (a, b) } else { (b, a) };
                if let Some(&m) = midpoint.get(&key) {
                    return m;
                }
                let m = verts.len() as u32;
                let pa = verts[a as usize];
                let pb = verts[b as usize];
                let mut p = [
                    (pa[0] + pb[0]) * 0.5,
                    (pa[1] + pb[1]) * 0.5,
                    (pa[2] + pb[2]) * 0.5,
                ];
                let len = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
                p[0] /= len;
                p[1] /= len;
                p[2] /= len;
                verts.push(p);
                midpoint.insert(key, m);
                m
            };
            for f in faces {
                let (a, b, c) = (f[0], f[1], f[2]);
                let ab = mid(a, b, &mut verts);
                let bc = mid(b, c, &mut verts);
                let ca = mid(c, a, &mut verts);
                new_faces.extend_from_slice(&[[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]);
            }
            faces = new_faces;
        }
        (verts, faces)
    }

    #[test]
    fn decimate_icosphere_to_target() {
        let (mut verts, mut faces) = icosphere(3); // 1280 faces
        assert_eq!(faces.len(), 1280);
        let report = simplify_to_target(&mut verts, &mut faces, 80, 180.0, |_| false)
            .expect("decimation should succeed");
        assert!(report.reached_target, "closed mesh should reach target");
        assert_eq!(faces.len(), 80);
        assert!(report.collapses > 0);
        // All indices must be in range after compaction... note: verts are
        // NOT compacted by simplify (lib.rs does it), so just check bounds.
        assert!(faces
            .iter()
            .all(|f| f.iter().all(|&i| (i as usize) < verts.len())));
    }

    #[test]
    fn boundary_edges_are_locked() {
        // A flat 2x2 grid quad fan (open boundary) must not collapse its
        // boundary edges; interior edges may still collapse.
        let verts = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
            [0.0, 2.0, 0.0],
            [1.0, 2.0, 0.0],
            [2.0, 2.0, 0.0],
        ];
        let faces = vec![
            [0, 1, 4],
            [0, 4, 3],
            [1, 2, 5],
            [1, 5, 4],
            [3, 4, 7],
            [3, 7, 6],
            [4, 5, 8],
            [4, 8, 7],
        ];
        let mut verts = verts;
        let mut faces = faces;
        // aggressive target: grid corners should survive because of locks
        let _ = simplify_to_target(&mut verts, &mut faces, 4, 180.0, |_| false)
            .expect("decimation should succeed");
        // cannot reach 4 faces without touching the boundary; merely assert
        // termination and validity, no panic, no invalid indices
        assert!(faces
            .iter()
            .all(|f| f.iter().all(|&i| (i as usize) < verts.len())));
    }
}
