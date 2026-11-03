//! Sharp feature lines ("creases"): thin rims such as ear edges, fins and
//! hard edges, where the surface turns by a large angle within one input
//! edge. A lattice that merely crosses such a rim at an angle produces a
//! staircase (vertices alternate between the two sides), so the field is
//! aligned with the crease and a lattice line is pinned onto it.

use std::collections::{HashMap, HashSet};

use nalgebra::Vector3;

/// Undirected edge key.
fn key(a: u32, b: u32) -> (u32, u32) {
    (a.min(b), a.max(b))
}

/// Edges whose two faces meet at more than `angle_deg` (dihedral angle
/// between face normals) *and* that stand out from their surroundings:
/// at least twice the dihedral of the flattest other edge of their two
/// faces. On a coarse but smooth input every edge bends a lot, but evenly
/// (no crease); a rim bends sharply between flat sides. Crease lines
/// shorter than `min_length` in total are dropped as noise.
pub fn crease_edges(
    pos: &[[f64; 3]],
    tris: &[[u32; 3]],
    angle_deg: f64,
    min_length: f64,
) -> HashSet<(u32, u32)> {
    let normals: Vec<Vector3<f64>> = tris
        .iter()
        .map(|t| {
            let p = |k: usize| Vector3::from(pos[t[k] as usize]);
            let n = (p(1) - p(0)).cross(&(p(2) - p(0)));
            let l = n.norm();
            if l > 0.0 {
                n / l
            } else {
                n
            }
        })
        .collect();
    let mut faces_of: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    for (fi, t) in tris.iter().enumerate() {
        for k in 0..3 {
            faces_of
                .entry(key(t[k], t[(k + 1) % 3]))
                .or_default()
                .push(fi);
        }
    }
    let dihedral = |e: &(u32, u32)| -> f64 {
        match faces_of.get(e).map(|f| f.as_slice()) {
            Some([f0, f1]) => normals[*f0].dot(&normals[*f1]).clamp(-1.0, 1.0).acos(),
            _ => 0.0,
        }
    };
    let min_angle = angle_deg.to_radians();
    let mut edges: HashSet<(u32, u32)> = HashSet::new();
    for (e, f) in &faces_of {
        let d = dihedral(e);
        if f.len() != 2 || d <= min_angle {
            continue;
        }
        let flattest = f
            .iter()
            .flat_map(|&fi| {
                let t = tris[fi];
                (0..3).map(move |k| key(t[k], t[(k + 1) % 3]))
            })
            .filter(|o| o != e)
            .map(|o| dihedral(&o))
            .fold(f64::INFINITY, f64::min);
        if d >= 2.0 * flattest {
            edges.insert(*e);
        }
    }
    // drop short crease lines (connected components by total length)
    let mut nbrs: HashMap<u32, Vec<u32>> = HashMap::new();
    for &(a, b) in &edges {
        nbrs.entry(a).or_default().push(b);
        nbrs.entry(b).or_default().push(a);
    }
    let len =
        |a: u32, b: u32| (Vector3::from(pos[a as usize]) - Vector3::from(pos[b as usize])).norm();
    let mut comp: HashMap<u32, usize> = HashMap::new();
    let mut comp_len: Vec<f64> = Vec::new();
    let mut starts: Vec<u32> = nbrs.keys().copied().collect();
    starts.sort_unstable();
    for s in starts {
        if comp.contains_key(&s) {
            continue;
        }
        let id = comp_len.len();
        comp_len.push(0.0);
        let mut stack = vec![s];
        comp.insert(s, id);
        while let Some(v) = stack.pop() {
            for &w in &nbrs[&v] {
                if v < w {
                    comp_len[id] += len(v, w); // each edge once, from its smaller end
                }
                if let std::collections::hash_map::Entry::Vacant(slot) = comp.entry(w) {
                    slot.insert(id);
                    stack.push(w);
                }
            }
        }
    }
    edges.retain(|(a, _)| comp_len[comp[a]] >= min_length);
    edges
}

/// Carry crease edges through refinement: refined vertex `n_before + k` is
/// the midpoint of `parents[k]`; a split crease edge becomes two.
pub fn refine(edges: &mut HashSet<(u32, u32)>, parents: &[[u32; 2]], n_before: usize) {
    for (k, &[a, b]) in parents.iter().enumerate() {
        if edges.remove(&key(a, b)) {
            let m = (n_before + k) as u32;
            edges.insert(key(a, m));
            edges.insert(key(m, b));
        }
    }
}

/// Unit tangent of the crease line at every vertex on it with exactly one
/// or two crease edges (corners and junctions get `None`, like every vertex
/// off the crease).
pub fn tangents(pos: &[[f64; 3]], edges: &HashSet<(u32, u32)>) -> Vec<Option<Vector3<f64>>> {
    let mut nbrs: Vec<Vec<u32>> = vec![Vec::new(); pos.len()];
    let mut sorted: Vec<_> = edges.iter().copied().collect();
    sorted.sort_unstable(); // determinism
    for (a, b) in sorted {
        nbrs[a as usize].push(b);
        nbrs[b as usize].push(a);
    }
    nbrs.iter()
        .enumerate()
        .map(|(i, nb)| {
            let p = |v: u32| Vector3::from(pos[v as usize]);
            let t = match nb.as_slice() {
                [a] => p(*a) - p(i as u32),
                [a, b] => p(*b) - p(*a),
                _ => return None,
            };
            let l = t.norm();
            (l > 1e-12).then(|| t / l)
        })
        .collect()
}

/// Line segments of the crease edges, for projecting output vertices.
pub fn segments(
    pos: &[[f64; 3]],
    edges: &HashSet<(u32, u32)>,
) -> Vec<(Vector3<f64>, Vector3<f64>)> {
    let mut sorted: Vec<_> = edges.iter().copied().collect();
    sorted.sort_unstable();
    sorted
        .into_iter()
        .map(|(a, b)| {
            (
                Vector3::from(pos[a as usize]),
                Vector3::from(pos[b as usize]),
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unit cube, two triangles per side: the 12 cube edges are creases,
    /// the 6 face diagonals are not.
    fn cube() -> (Vec<[f64; 3]>, Vec<[u32; 3]>) {
        let pos = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ];
        let quads = [
            [0, 3, 2, 1],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [2, 3, 7, 6],
            [1, 2, 6, 5],
            [0, 4, 7, 3],
        ];
        let tris = quads
            .iter()
            .flat_map(|q| [[q[0], q[1], q[2]], [q[0], q[2], q[3]]])
            .collect();
        (pos, tris)
    }

    #[test]
    fn cube_edges_are_creases_diagonals_are_not() {
        let (pos, tris) = cube();
        let e = crease_edges(&pos, &tris, 40.0, 0.0);
        assert_eq!(e.len(), 12);
        for &(a, b) in &e {
            let d: f64 = (0..3)
                .map(|c| (pos[a as usize][c] - pos[b as usize][c]).abs())
                .sum();
            assert!((d - 1.0).abs() < 1e-12, "diagonal ({a},{b}) marked");
        }
        // cube corners are junctions (3 crease edges): no tangent
        assert!(tangents(&pos, &e).iter().all(|t| t.is_none()));
    }

    #[test]
    fn refinement_splits_crease_edges() {
        let mut e: HashSet<(u32, u32)> = [(0, 1), (1, 2)].into_iter().collect();
        refine(&mut e, &[[0, 1], [3, 1]], 3);
        // 0-1 split by 3, then 3-1 split by 4
        let want: HashSet<(u32, u32)> = [(0, 3), (3, 4), (1, 4), (1, 2)].into_iter().collect();
        assert_eq!(e, want);
    }
}
