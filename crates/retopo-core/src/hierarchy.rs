//! Multi-resolution hierarchy for the orientation solve.
//!
//! The optimizer works far better when it starts from a coarse
//! approximation and refines it, because local greedy optimisation gets
//! stuck in bad minima on large graphs. This module coarsens the primal
//! (vertex) graph by greedily collapsing the most normal-aligned,
//! area-matched pairs, builds an adjacency per level, and propagates a
//! solved coarser field down to finer levels (children inherit the
//! parent's direction projected onto their tangent plane).
//!
//! Determinism: ties in the collapse priority are broken by vertex index,
//! and initial directions are derived from position hashes.

use std::collections::HashMap;

use nalgebra::Vector3;

pub const INVALID: u32 = u32::MAX;

/// One level of the multi-resolution hierarchy (level 0 = input mesh).
#[derive(Debug, Clone)]
pub struct HierarchyLevel {
    pub positions: Vec<[f64; 3]>,
    pub normals: Vec<Vector3<f64>>,
    /// Dual cell areas (sum of incident face areas / 3).
    pub areas: Vec<f64>,
    /// Uniq one-ring neighbours as (index, weight) pairs. Weights are
    /// accumulated from finer levels when coarsening.
    pub adj: Vec<Vec<(u32, f64)>>,
}

/// The full hierarchy. `to_upper[level]` has one entry per node of
/// `levels[level + 1]`; each entry references up to two children in
/// `levels[level]` (`INVALID` for the second slot).
#[derive(Debug, Clone)]
pub struct Hierarchy {
    pub levels: Vec<HierarchyLevel>,
    pub to_upper: Vec<Vec<[u32; 2]>>,
}

fn area_vector(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> Vector3<f64> {
    let ab = Vector3::new(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
    let ac = Vector3::new(c[0] - a[0], c[1] - a[1], c[2] - a[2]);
    ab.cross(&ac)
}

/// Sum of incident face areas per vertex (divided by 3, the mean face
/// contribution to each corner).
pub fn dual_areas(positions: &[[f64; 3]], tris: &[[u32; 3]]) -> Vec<f64> {
    let mut areas = vec![0.0f64; positions.len()];
    for t in tris {
        let p = |i: usize| positions[t[i] as usize];
        let n = area_vector(p(0), p(1), p(2));
        let a = n.norm() * 0.5;
        for i in 0..3 {
            areas[t[i] as usize] += a / 3.0;
        }
    }
    areas
}

fn one_ring(vertex_count: usize, tris: &[[u32; 3]]) -> Vec<Vec<u32>> {
    let mut nbrs: Vec<Vec<u32>> = vec![Vec::new(); vertex_count];
    let mut seen: Vec<std::collections::HashSet<u32>> = (0..vertex_count)
        .map(|_| std::collections::HashSet::new())
        .collect();
    for t in tris {
        for i in 0..3 {
            let v = t[i] as usize;
            for k in 1..3 {
                let n = t[(i + k) % 3];
                if seen[v].insert(n) {
                    nbrs[v].push(n);
                }
            }
        }
    }
    nbrs
}

fn vertex_normals(positions: &[[f64; 3]], tris: &[[u32; 3]]) -> Vec<Vector3<f64>> {
    let mut acc = vec![Vector3::zeros(); positions.len()];
    for t in tris {
        let n = area_vector(
            positions[t[0] as usize],
            positions[t[1] as usize],
            positions[t[2] as usize],
        );
        acc[t[0] as usize] += n;
        acc[t[1] as usize] += n;
        acc[t[2] as usize] += n;
    }
    acc.into_iter()
        .map(|v| {
            if v.norm() < 1e-20 {
                Vector3::new(0.0, 0.0, 1.0)
            } else {
                v.normalize()
            }
        })
        .collect()
}

fn level_from_mesh(positions: &[[f64; 3]], tris: &[[u32; 3]]) -> HierarchyLevel {
    let adj = one_ring(positions.len(), tris)
        .into_iter()
        .map(|nbrs| nbrs.into_iter().map(|n| (n, 1.0)).collect())
        .collect();
    HierarchyLevel {
        positions: positions.to_vec(),
        normals: vertex_normals(positions, tris),
        areas: dual_areas(positions, tris),
        adj,
    }
}

/// Coarsen one level. Returns `(coarser, to_upper)` where `to_upper[j]`
/// lists the (up to two) children in the current level for node j of the
/// coarser level.
fn coarsen(level: &HierarchyLevel) -> Option<(HierarchyLevel, Vec<[u32; 2]>)> {
    let n = level.positions.len();
    if n <= 1 {
        return None;
    }

    // ---- collapse candidates: (u, v, priority), sorted best-first --------
    let mut candidates: Vec<(f64, u32, u32)> = Vec::new();
    for (u, nbrs) in level.adj.iter().enumerate() {
        for &(v, _) in nbrs {
            if (v as usize) <= u {
                continue;
            }
            let nu = level.normals[u];
            let nv = level.normals[v as usize];
            let dp = nu.dot(&nv);
            let (a0, a1) = (level.areas[u], level.areas[v as usize]);
            let ratio = if a0 > a1 {
                if a1 > 1e-12 {
                    a0 / a1
                } else {
                    1.0
                }
            } else if a0 > 1e-12 {
                a1 / a0
            } else {
                1.0
            };
            candidates.push((dp * ratio, u as u32, v));
        }
    }
    // Deterministic order: priority desc, then indices asc.
    candidates.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));

    let mut to_upper: Vec<[u32; 2]> = Vec::with_capacity(n);
    let mut to_lower = vec![INVALID; n];
    let mut merged = vec![false; n];
    for (_, u, v) in candidates {
        if merged[u as usize] || merged[v as usize] {
            continue;
        }
        merged[u as usize] = true;
        merged[v as usize] = true;
        let idx = to_upper.len() as u32;
        to_upper.push([u, v]);
        to_lower[u as usize] = idx;
        to_lower[v as usize] = idx;
    }
    for i in 0..n {
        if !merged[i] {
            let idx = to_upper.len() as u32;
            to_upper.push([i as u32, INVALID]);
            to_lower[i] = idx;
        }
    }

    let m = to_upper.len();
    if m == n {
        return None; // unable to collapse anything — avoid infinite coarsening
    }

    // ---- positions / normals / areas ---------------------------------------
    let mut positions = Vec::with_capacity(m);
    let mut normals = Vec::with_capacity(m);
    let mut areas = Vec::with_capacity(m);
    for &[a, b] in &to_upper {
        if b == INVALID {
            positions.push(level.positions[a as usize]);
            normals.push(level.normals[a as usize]);
            areas.push(level.areas[a as usize]);
        } else {
            let (aa, ab) = (level.areas[a as usize], level.areas[b as usize]);
            let sum = aa + ab;
            if sum > 1e-20 {
                let pa = level.positions[a as usize];
                let pb = level.positions[b as usize];
                positions.push([
                    (pa[0] * aa + pb[0] * ab) / sum,
                    (pa[1] * aa + pb[1] * ab) / sum,
                    (pa[2] * aa + pb[2] * ab) / sum,
                ]);
            } else {
                positions.push(level.positions[a as usize]);
            }
            let nv = level.normals[a as usize] * aa.max(1e-12)
                + level.normals[b as usize] * ab.max(1e-12);
            normals.push(if nv.norm() < 1e-20 {
                Vector3::new(0.0, 0.0, 1.0)
            } else {
                nv.normalize()
            });
            areas.push(sum);
        }
    }

    // ---- adjacency rebuild: merge children link lists ----------------------
    let mut acc: Vec<HashMap<u32, f64>> = (0..m).map(|_| HashMap::new()).collect();
    for i in 0..n {
        let ci = to_lower[i];
        for &(j, w) in &level.adj[i] {
            let cj = to_lower[j as usize];
            if cj == ci {
                continue; // internal edge of the merged pair
            }
            *acc[ci as usize].entry(cj).or_insert(0.0) += w;
        }
    }
    let adj: Vec<Vec<(u32, f64)>> = acc
        .into_iter()
        .map(|m| {
            let mut v: Vec<(u32, f64)> = m.into_iter().collect();
            v.sort_unstable_by_key(|(id, _)| *id);
            v
        })
        .collect();

    Some((
        HierarchyLevel {
            positions,
            normals,
            areas,
            adj,
        },
        to_upper,
    ))
}

impl Hierarchy {
    /// Build the hierarchy for a mesh. The finest level is level 0.
    pub fn build(positions: &[[f64; 3]], tris: &[[u32; 3]], max_levels: usize) -> Hierarchy {
        let mut levels = vec![level_from_mesh(positions, tris)];
        let mut to_upper = Vec::new();
        for _ in 0..max_levels.saturating_sub(1) {
            match coarsen(levels.last().unwrap()) {
                Some((level, up)) => {
                    let single = level.positions.len() == 1;
                    levels.push(level);
                    to_upper.push(up);
                    if single {
                        break;
                    }
                }
                None => break,
            }
        }
        Hierarchy { levels, to_upper }
    }
}
