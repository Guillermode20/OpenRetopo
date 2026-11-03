//! Longest-edge bisection so the input is fine enough to carry the
//! position-field lattice (the reference implementation subdivides to
//! `min(scale / 2, 2 · average edge length)` before solving; extraction can
//! only cluster input vertices, so a coarse input cannot express a finer
//! output grid).
//!
//! Deterministic: edges are split in decreasing length order, ties broken
//! by vertex indices. Closed or open triangle meshes are supported; the
//! mesh is expected to be edge-manifold (≤ 2 faces per edge).

use std::collections::{BinaryHeap, HashMap};

/// Average edge length of a triangle mesh (each undirected edge once).
pub fn average_edge_length(positions: &[[f64; 3]], tris: &[[u32; 3]]) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for t in tris {
        for (i, j) in [(0, 1), (1, 2), (2, 0)] {
            // count each interior edge from one side only (a < b); boundary
            // edges with a > b are skipped too, a negligible bias
            if t[i] < t[j] {
                sum += dist(positions[t[i] as usize], positions[t[j] as usize]);
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

fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// Split every edge longer than `max_length` at its midpoint until none
/// remain. Returns the number of splits performed. `max_vertices` caps the
/// output size; returns `None` (mesh untouched past that point is not
/// guaranteed) if the cap would be exceeded, so callers can fail cleanly.
pub fn subdivide_long_edges(
    positions: &mut Vec<[f64; 3]>,
    tris: &mut Vec<[u32; 3]>,
    max_length: f64,
    max_vertices: usize,
) -> Option<usize> {
    subdivide_long_edges_with_parents(positions, tris, max_length, max_vertices, &mut Vec::new())
}

/// As [`subdivide_long_edges`], also appending the split edge `(a, b)` of
/// every new vertex to `parents` (new vertex `k` ↔ `parents[k - n0]`), so
/// per-vertex data can be carried onto the refined mesh.
pub fn subdivide_long_edges_with_parents(
    positions: &mut Vec<[f64; 3]>,
    tris: &mut Vec<[u32; 3]>,
    max_length: f64,
    max_vertices: usize,
    parents: &mut Vec<[u32; 2]>,
) -> Option<usize> {
    if max_length <= 0.0 || !max_length.is_finite() {
        return Some(0);
    }
    let mut limits = vec![max_length; positions.len()];
    subdivide_adaptive(positions, tris, &mut limits, max_vertices, parents)
}

/// Spatially varying refinement: edge `(a, b)` is split while longer than
/// `min(limits[a], limits[b])`. New vertices get the mean of their parents'
/// limits (appended to `limits`) and their split edge appended to
/// `parents`. Returns `None` if `max_vertices` would be exceeded.
pub fn subdivide_adaptive(
    positions: &mut Vec<[f64; 3]>,
    tris: &mut Vec<[u32; 3]>,
    limits: &mut Vec<f64>,
    max_vertices: usize,
    parents: &mut Vec<[u32; 2]>,
) -> Option<usize> {
    assert_eq!(limits.len(), positions.len(), "one limit per vertex");
    if limits.iter().any(|l| !(l.is_finite() && *l > 0.0)) {
        return Some(0);
    }
    // directed edge -> face
    let mut half: HashMap<(u32, u32), u32> = HashMap::with_capacity(tris.len() * 3);
    for (fi, t) in tris.iter().enumerate() {
        for k in 0..3 {
            half.insert((t[k], t[(k + 1) % 3]), fi as u32);
        }
    }
    // max-heap on (length bits, then smaller indices first)
    let mut heap: BinaryHeap<(u64, std::cmp::Reverse<(u32, u32)>)> = BinaryHeap::new();
    let push = |heap: &mut BinaryHeap<(u64, std::cmp::Reverse<(u32, u32)>)>,
                positions: &[[f64; 3]],
                limits: &[f64],
                a: u32,
                b: u32| {
        let len = dist(positions[a as usize], positions[b as usize]);
        if len > limits[a as usize].min(limits[b as usize]) {
            heap.push((len.to_bits(), std::cmp::Reverse((a.min(b), a.max(b)))));
        }
    };
    for t in tris.iter() {
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            if a < b || !half.contains_key(&(b, a)) {
                push(&mut heap, positions, limits, a, b);
            }
        }
    }

    let mut splits = 0usize;
    while let Some((bits, std::cmp::Reverse((a, b)))) = heap.pop() {
        // stale entry: edge no longer exists or changed
        let f_ab = half.get(&(a, b)).copied();
        let f_ba = half.get(&(b, a)).copied();
        if f_ab.is_none() && f_ba.is_none() {
            continue;
        }
        if dist(positions[a as usize], positions[b as usize]).to_bits() != bits {
            continue;
        }
        if positions.len() >= max_vertices {
            return None;
        }
        let pa = positions[a as usize];
        let pb = positions[b as usize];
        let vn = positions.len() as u32;
        parents.push([a, b]);
        limits.push(0.5 * (limits[a as usize] + limits[b as usize]));
        positions.push([
            (pa[0] + pb[0]) * 0.5,
            (pa[1] + pb[1]) * 0.5,
            (pa[2] + pb[2]) * 0.5,
        ]);
        half.remove(&(a, b));
        half.remove(&(b, a));
        // split each incident face (x -> y is the split edge in that face)
        for (x, y, face) in [(a, b, f_ab), (b, a, f_ba)] {
            let Some(f) = face else { continue };
            let t = tris[f as usize];
            let k = (0..3)
                .find(|&k| t[k] == x && t[(k + 1) % 3] == y)
                .expect("half-edge map out of sync");
            let c = t[(k + 2) % 3];
            let nf = tris.len() as u32;
            tris[f as usize] = [x, vn, c];
            tris.push([vn, y, c]);
            half.insert((x, vn), f);
            half.insert((vn, c), f);
            half.insert((c, x), f);
            half.insert((vn, y), nf);
            half.insert((y, c), nf);
            half.insert((c, vn), nf);
            push(&mut heap, positions, limits, vn, c);
        }
        push(&mut heap, positions, limits, a, vn);
        push(&mut heap, positions, limits, vn, b);
        splits += 1;
    }
    Some(splits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tetra_subdivides_closed() {
        let mut p = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let mut t = vec![[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]];
        let chi0 = crate::field::euler_characteristic(p.len(), &t);
        let n = subdivide_long_edges(&mut p, &mut t, 0.2, 1_000_000).unwrap();
        assert!(n > 0);
        assert_eq!(crate::field::euler_characteristic(p.len(), &t), chi0);
        for tri in &t {
            for k in 0..3 {
                assert!(dist(p[tri[k] as usize], p[tri[(k + 1) % 3] as usize]) <= 0.2 + 1e-12);
            }
        }
        // still closed: every directed edge has a twin
        let set: std::collections::HashSet<(u32, u32)> = t
            .iter()
            .flat_map(|f| (0..3).map(move |k| (f[k], f[(k + 1) % 3])))
            .collect();
        assert!(set.iter().all(|&(a, b)| set.contains(&(b, a))));
    }
}
