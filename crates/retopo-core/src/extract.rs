//! Quad-mesh extraction from a solved orientation + position field.
//!
//! This is a port of the `extract_graph` / `extract_faces` stages of the
//! reference implementation (Jakob et al., *Instant Field-Aligned Meshes*,
//! SIGGRAPH Asia 2015, `wjakob/instant-meshes`, BSD-3-Clause; see
//! NOTICE.md). Step numbers follow the reference:
//!
//! 1. **Classify** every input edge by the integer lattice offset between
//!    its endpoints' anchors (extrinsic index rule, as the reference uses
//!    regardless of field mode): offset 0 → collapse candidate, offset of
//!    one cell along one axis → output edge, anything else is ignored.
//! 2. **Collapse** candidates in increasing error order with a disjoint-set
//!    union, rejecting merges of clusters already joined by an output edge.
//! 3. **Assign vertices**: surviving clusters with output edges.
//!    3a. optional spurious-vertex removal (off by default, see
//!    [`ExtractOptions::remove_spurious`]).
//! 4. **Positions**: Gaussian-weighted average of anchors per cluster.
//! 5. **Snap & clean**: merge/snap sliver triangles and remove the diagonal
//!    of every triangle pair (turning them into quads).
//! 6. **Orient** each vertex's adjacency clockwise around its normal.
//! 7. **Extract faces** by walking the rotation system, quads first, then
//!    triangles, then 5..8-gons (split into quads/triangles).
//! 8. **Fill holes** bounded by half-edges whose twin is part of a face.
//!
//! Additions beyond the reference (it tolerates holes and non-manifold
//! output; we do not):
//!   * holes of any degree are filled, splitting chords are checked against
//!     existing edges so they cannot create non-manifold edges;
//!   * the result is validated as a closed, oriented 2-manifold whose
//!     Euler characteristic equals the input's. Anything else is an error —
//!     garbage is never returned.
//!
//! Scope limits (explicit, never silent):
//!   * Open meshes are rejected with [`RetopoError::UnsupportedBoundary`].
//!   * Non-manifold edges are rejected with [`RetopoError::NonManifoldInput`].

use std::collections::{HashMap, HashSet};

use nalgebra::Vector3;

use crate::field::middle_point;
use crate::RetopoError;

#[derive(Debug, Clone, Copy)]
pub struct ExtractOptions {
    /// Reference Step 3a: drop output vertices that absorbed fewer than a
    /// tenth of the average number of collapses.
    pub remove_spurious: bool,
    /// Reference Step 5: snap sliver triangles, remove triangle-pair
    /// diagonals.
    pub remove_unnecessary_edges: bool,
    /// Validate the result (closed 2-manifold, Euler characteristic
    /// preserved). Disable only for diagnostics.
    pub validate: bool,
}

impl Default for ExtractOptions {
    fn default() -> Self {
        ExtractOptions {
            remove_spurious: false,
            remove_unnecessary_edges: true,
            validate: true,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ExtractionStats {
    pub output_vertices: u64,
    pub quads: u64,
    pub triangles: u64,
    /// Faces emitted during the regular extraction pass (Step 7).
    pub faces_traced: u64,
    /// Holes filled in Step 8.
    pub holes_filled: u64,
    /// Hole loops that could not be traced closed (never emitted). A valid
    /// extraction always has 0.
    pub dropped_cycles: u64,
    pub collapses_accepted: u64,
    pub collapses_rejected: u64,
    /// Output vertices removed as "spurious" (absorbed too few inputs).
    pub spurious_removed: u64,
    /// Step 5 counters.
    pub snapped: u64,
    pub diagonals_removed: u64,
    /// Euler characteristic of the input and the output.
    pub euler_input: i64,
    pub euler_output: i64,
    /// Output vertices whose valence is not 4.
    pub irregular_vertices: u64,
}

#[derive(Debug, Clone)]
pub struct ExtractedMesh {
    /// One triple per output vertex.
    pub positions: Vec<[f32; 3]>,
    /// Faces with 3 or 4 vertex indices, counter-clockwise about the
    /// outward normal.
    pub faces: Vec<Vec<u32>>,
    pub stats: ExtractionStats,
}

// ---------------------------------------------------------------------------
// disjoint set
// ---------------------------------------------------------------------------

struct DisjointSet {
    parent: Vec<u32>,
    size: Vec<u32>,
}

impl DisjointSet {
    fn new(n: usize) -> Self {
        DisjointSet {
            parent: (0..n as u32).collect(),
            size: vec![1; n],
        }
    }

    fn find(&mut self, mut x: u32) -> u32 {
        let mut root = x;
        while self.parent[root as usize] != root {
            root = self.parent[root as usize];
        }
        while self.parent[x as usize] != root {
            let next = self.parent[x as usize];
            self.parent[x as usize] = root;
            x = next;
        }
        root
    }

    /// Union by size; ties go to the smaller index for determinism.
    fn union(&mut self, ra: u32, rb: u32) -> u32 {
        let (sa, sb) = (self.size[ra as usize], self.size[rb as usize]);
        let (big, small) = if sa > sb || (sa == sb && ra < rb) {
            (ra, rb)
        } else {
            (rb, ra)
        };
        self.parent[small as usize] = big;
        self.size[big as usize] += self.size[small as usize];
        big
    }
}

// ---------------------------------------------------------------------------
// lattice helpers (extrinsic rules, as used by the reference extractor)
// ---------------------------------------------------------------------------

fn compat_orientation_extrinsic_4(
    q0: Vector3<f64>,
    n0: Vector3<f64>,
    q1: Vector3<f64>,
    n1: Vector3<f64>,
) -> (Vector3<f64>, Vector3<f64>) {
    let a = [q0, n0.cross(&q0)];
    let b = [q1, n1.cross(&q1)];
    let mut best = (f64::NEG_INFINITY, 0usize, 0usize);
    for (i, ai) in a.iter().enumerate() {
        for (j, bj) in b.iter().enumerate() {
            let score = ai.dot(bj).abs();
            if score > best.0 {
                best = (score, i, j);
            }
        }
    }
    let dp = a[best.1].dot(&b[best.2]);
    (a[best.1], b[best.2] * if dp < 0.0 { -1.0 } else { 1.0 })
}

fn position_floor_index_4(
    o: Vector3<f64>,
    q: Vector3<f64>,
    n: Vector3<f64>,
    p: Vector3<f64>,
    inv_scale: f64,
) -> [i64; 2] {
    let t = n.cross(&q);
    let d = p - o;
    [
        (q.dot(&d) * inv_scale).floor() as i64,
        (t.dot(&d) * inv_scale).floor() as i64,
    ]
}

/// Integer lattice coordinates of the two anchors' closest lattice points
/// around the edge midpoint, plus the squared distance between them.
#[allow(clippy::too_many_arguments)]
fn compat_position_extrinsic_index_4(
    p0: Vector3<f64>,
    n0: Vector3<f64>,
    q0: Vector3<f64>,
    o0: Vector3<f64>,
    p1: Vector3<f64>,
    n1: Vector3<f64>,
    q1: Vector3<f64>,
    o1: Vector3<f64>,
    scale: f64,
    inv_scale: f64,
) -> ([i64; 2], [i64; 2], f64) {
    let (t0, t1) = (n0.cross(&q0), n1.cross(&q1));
    let middle = middle_point(p0, n0, p1, n1);
    let o0p = position_floor_index_4(o0, q0, n0, middle, inv_scale);
    let o1p = position_floor_index_4(o1, q1, n1, middle, inv_scale);
    let corner = |k: usize| [(k & 1) as i64, ((k & 2) >> 1) as i64];
    let mut best = (f64::INFINITY, 0usize, 0usize);
    for i in 0..4 {
        let ci = corner(i);
        let o0t = o0 + (q0 * (ci[0] + o0p[0]) as f64 + t0 * (ci[1] + o0p[1]) as f64) * scale;
        for j in 0..4 {
            let cj = corner(j);
            let o1t = o1 + (q1 * (cj[0] + o1p[0]) as f64 + t1 * (cj[1] + o1p[1]) as f64) * scale;
            let cost = (o0t - o1t).norm_squared();
            if cost < best.0 {
                best = (cost, i, j);
            }
        }
    }
    let (ci, cj) = (corner(best.1), corner(best.2));
    (
        [ci[0] + o0p[0], ci[1] + o0p[1]],
        [cj[0] + o1p[0], cj[1] + o1p[1]],
        best.0,
    )
}

// ---------------------------------------------------------------------------
// mesh helpers
// ---------------------------------------------------------------------------

/// Unique neighbour list per vertex, in first-seen order (deterministic).
fn one_ring(vertex_count: usize, tris: &[[u32; 3]]) -> Vec<Vec<u32>> {
    let mut nbrs: Vec<Vec<u32>> = vec![Vec::new(); vertex_count];
    for t in tris {
        for i in 0..3 {
            let v = t[i] as usize;
            for k in 1..3 {
                let n = t[(i + k) % 3];
                if !nbrs[v].contains(&n) {
                    nbrs[v].push(n);
                }
            }
        }
    }
    nbrs
}

/// Closed manifold input required for extraction.
pub fn check_closed_manifold(tris: &[[u32; 3]]) -> Result<(), RetopoError> {
    let mut counts: HashMap<(u32, u32), u32> = HashMap::new();
    for t in tris {
        for (i, j) in [(0, 1), (1, 2), (2, 0)] {
            let (a, b) = (t[i].min(t[j]), t[i].max(t[j]));
            *counts.entry((a, b)).or_insert(0) += 1;
        }
    }
    if counts.values().any(|&c| c > 2) {
        return Err(RetopoError::NonManifoldInput);
    }
    if counts.values().any(|&c| c == 1) {
        return Err(RetopoError::UnsupportedBoundary);
    }
    Ok(())
}

fn orthonormal_basis(n: Vector3<f64>) -> (Vector3<f64>, Vector3<f64>) {
    let arbitrary = if n.x.abs() < 0.9 {
        Vector3::new(1.0, 0.0, 0.0)
    } else {
        Vector3::new(0.0, 1.0, 0.0)
    };
    let s = (arbitrary - n * n.dot(&arbitrary)).normalize();
    (s, n.cross(&s))
}

fn remove_link(list: &mut Vec<u32>, x: u32) {
    list.retain(|&y| y != x);
}

/// Topology summary of a polygon mesh, or a description of why it is not
/// a closed oriented 2-manifold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Topology {
    pub vertices: usize,
    pub edges: usize,
    pub faces: usize,
    pub euler: i64,
}

/// Validate that `faces` form a closed, consistently oriented 2-manifold
/// (every directed edge used exactly once and its reverse exactly once, no
/// degenerate faces, one fan per vertex) and return its Euler
/// characteristic. Unreferenced vertices are not counted.
pub fn validate_closed_manifold(
    vertex_count: usize,
    faces: &[Vec<u32>],
) -> Result<Topology, String> {
    let mut half: HashMap<(u32, u32), u32> = HashMap::new();
    for (fi, f) in faces.iter().enumerate() {
        if f.len() < 3 {
            return Err(format!("face {fi} has {} vertices", f.len()));
        }
        for (k, &v) in f.iter().enumerate() {
            if v as usize >= vertex_count {
                return Err(format!("face {fi} references vertex {v} out of range"));
            }
            if f[..k].contains(&v) {
                return Err(format!("face {fi} repeats vertex {v}"));
            }
        }
        for k in 0..f.len() {
            let e = (f[k], f[(k + 1) % f.len()]);
            if half.insert(e, fi as u32).is_some() {
                return Err(format!(
                    "directed edge {e:?} used twice (non-manifold or flipped face)"
                ));
            }
        }
    }
    for &(a, b) in half.keys() {
        if !half.contains_key(&(b, a)) {
            return Err(format!("edge ({a}, {b}) is a boundary edge (hole)"));
        }
    }
    // vertex manifoldness: the outgoing half-edges around each vertex form
    // a single cycle. next around v after (v→a): face containing (v→a) has
    // predecessor p (p→v); the twin (v→p) is the next outgoing edge.
    let mut outgoing: HashMap<u32, Vec<u32>> = HashMap::new();
    for &(a, b) in half.keys() {
        outgoing.entry(a).or_default().push(b);
    }
    let pred = |fi: u32, v: u32| -> u32 {
        let f = &faces[fi as usize];
        let k = f.iter().position(|&x| x == v).unwrap();
        f[(k + f.len() - 1) % f.len()]
    };
    for (&v, outs) in &outgoing {
        let start = outs[0];
        let mut cur = start;
        let mut count = 0usize;
        loop {
            let fi = half[&(v, cur)];
            cur = pred(fi, v);
            count += 1;
            if cur == start || count > outs.len() {
                break;
            }
        }
        if count != outs.len() {
            return Err(format!(
                "vertex {v} is non-manifold (fan of {count} of its {} edges)",
                outs.len()
            ));
        }
    }
    let vertices = outgoing.len();
    let edges = half.len() / 2;
    Ok(Topology {
        vertices,
        edges,
        faces: faces.len(),
        euler: vertices as i64 - edges as i64 + faces.len() as i64,
    })
}

// ---------------------------------------------------------------------------
// extraction
// ---------------------------------------------------------------------------

/// Extract a quad-dominant mesh from solved fields with default options.
pub fn extract_mesh(
    positions: &[[f64; 3]],
    tris: &[[u32; 3]],
    directions: &[Vector3<f64>],
    normals: &[Vector3<f64>],
    anchors: &[Vector3<f64>],
    scale: f64,
) -> Result<ExtractedMesh, RetopoError> {
    extract_mesh_with(
        positions,
        tris,
        directions,
        normals,
        anchors,
        scale,
        &ExtractOptions::default(),
    )
}

/// Extract a quad-dominant mesh from solved fields. Inputs are aligned by
/// index with the (welded, closed, manifold) mesh the fields were solved on.
#[allow(clippy::too_many_arguments)]
pub fn extract_mesh_with(
    positions: &[[f64; 3]],
    tris: &[[u32; 3]],
    directions: &[Vector3<f64>],
    normals: &[Vector3<f64>],
    anchors: &[Vector3<f64>],
    scale: f64,
    options: &ExtractOptions,
) -> Result<ExtractedMesh, RetopoError> {
    check_closed_manifold(tris)?;
    let n = positions.len();
    let inv_scale = 1.0 / scale;
    let mut stats = ExtractionStats {
        euler_input: crate::field::euler_characteristic(n, tris),
        ..ExtractionStats::default()
    };

    // ---- Step 1: classify edges -------------------------------------------
    let adj = one_ring(n, tris);
    let mut collapses: Vec<(f64, u32, u32)> = Vec::new();
    let mut graph: Vec<Vec<u32>> = vec![Vec::new(); n];
    for i in 0..n {
        for &j in &adj[i] {
            let j = j as usize;
            if j < i {
                continue;
            }
            let (qi, qj) = compat_orientation_extrinsic_4(
                directions[i],
                normals[i],
                directions[j],
                normals[j],
            );
            let (si, sj, err) = compat_position_extrinsic_index_4(
                Vector3::from(positions[i]),
                normals[i],
                qi,
                anchors[i],
                Vector3::from(positions[j]),
                normals[j],
                qj,
                anchors[j],
                scale,
                inv_scale,
            );
            let (du, dv) = ((si[0] - sj[0]).abs(), (si[1] - sj[1]).abs());
            if du.max(dv) > 1 || (du == 1 && dv == 1) {
                continue; // long-distance or diagonal link
            }
            if du + dv == 0 {
                collapses.push((err, i as u32, j as u32));
            } else {
                graph[i].push(j as u32);
                graph[j].push(i as u32);
            }
        }
    }
    // stable sort by error: ties keep the deterministic generation order
    collapses.sort_by(|a, b| a.0.total_cmp(&b.0));

    // ---- Step 2: collapse --------------------------------------------------
    let mut dset = DisjointSet::new(n);
    let mut n_collapses = vec![0u32; n];
    for &(_, i, j) in &collapses {
        let (ri, rj) = (dset.find(i), dset.find(j));
        if ri == rj {
            continue;
        }
        let contained = graph[ri as usize]
            .clone()
            .into_iter()
            .any(|x| dset.find(x) == rj);
        if contained {
            stats.collapses_rejected += 1;
            continue;
        }
        let mut merged: Vec<u32> = std::mem::take(&mut graph[ri as usize]);
        merged.append(&mut graph[rj as usize]);
        for x in merged.iter_mut() {
            *x = dset.find(*x);
        }
        merged.sort_unstable();
        merged.dedup();
        let target = dset.union(ri, rj);
        graph[target as usize] = merged;
        n_collapses[target as usize] = n_collapses[ri as usize] + n_collapses[rj as usize] + 1;
        stats.collapses_accepted += 1;
    }

    // ---- Step 3: assign vertices ------------------------------------------
    let mut vertex_map: HashMap<u32, u32> = HashMap::new();
    let mut roots: Vec<u32> = Vec::new();
    for i in 0..n as u32 {
        if !graph[i as usize].is_empty() {
            debug_assert_eq!(dset.find(i), i);
            vertex_map.insert(i, roots.len() as u32);
            roots.push(i);
        }
    }
    let mut out_adj: Vec<Vec<u32>> = Vec::with_capacity(roots.len());
    for (new_idx, &r) in roots.iter().enumerate() {
        let mut list: Vec<u32> = graph[r as usize]
            .clone()
            .into_iter()
            .map(|x| vertex_map[&dset.find(x)])
            .collect();
        list.sort_unstable();
        list.dedup();
        debug_assert!(!list.contains(&(new_idx as u32)));
        out_adj.push(list);
    }
    let out_collapses: Vec<u32> = roots.iter().map(|&r| n_collapses[r as usize]).collect();

    // ---- Step 3a: spurious vertices (optional) ----------------------------
    if options.remove_spurious && !roots.is_empty() {
        let avg = out_collapses.iter().map(|&c| c as f64).sum::<f64>() / roots.len() as f64;
        for i in 0..roots.len() {
            if out_collapses[i] as f64 > avg / 10.0 {
                continue;
            }
            for nb in std::mem::take(&mut out_adj[i]) {
                remove_link(&mut out_adj[nb as usize], i as u32);
            }
            stats.spurious_removed += 1;
        }
    }

    // ---- Step 4: positions & normals --------------------------------------
    let out_count = roots.len();
    let mut o_new = vec![Vector3::zeros(); out_count];
    let mut n_new = vec![Vector3::zeros(); out_count];
    let mut w_acc = vec![0.0f64; out_count];
    for j in 0..n {
        let Some(&oi) = vertex_map.get(&dset.find(j as u32)) else {
            continue;
        };
        let d = anchors[j] - Vector3::from(positions[j]);
        let w = (-9.0 * d.norm_squared() * inv_scale * inv_scale).exp();
        o_new[oi as usize] += anchors[j] * w;
        n_new[oi as usize] += normals[j] * w;
        w_acc[oi as usize] += w;
    }
    for i in 0..out_count {
        if w_acc[i] > 0.0 && (n_new[i].norm() > 1e-30) {
            o_new[i] /= w_acc[i];
            n_new[i] = n_new[i].normalize();
        } else {
            // every anchor is many cells away from its vertex: fall back to
            // an unweighted average rather than emit a NaN
            let mut acc = Vector3::zeros();
            let mut nacc = Vector3::zeros();
            let mut cnt = 0.0;
            for j in 0..n {
                if dset.find(j as u32) == roots[i] {
                    acc += anchors[j];
                    nacc += normals[j];
                    cnt += 1.0;
                }
            }
            o_new[i] = acc / cnt;
            n_new[i] = if nacc.norm() > 1e-30 {
                nacc.normalize()
            } else {
                Vector3::new(0.0, 0.0, 1.0)
            };
        }
    }

    // ---- Step 5: snapping and removing unnecessary edges ------------------
    if options.remove_unnecessary_edges {
        snap_and_clean(&mut out_adj, &mut o_new, &mut n_new, scale, &mut stats);
    }

    // ---- Step 6: orient edges (clockwise around the normal) ---------------
    for v in 0..out_count {
        let p = o_new[v];
        let (s, t) = orthonormal_basis(n_new[v]);
        let angle = |x: u32| {
            let d = o_new[x as usize] - p;
            d.dot(&t).atan2(d.dot(&s))
        };
        out_adj[v].sort_by(|&a, &b| angle(b).total_cmp(&angle(a)).then(a.cmp(&b)));
    }

    // ---- Steps 7 + 8: faces -------------------------------------------------
    let traced = extract_faces(&out_adj, &mut o_new, &mut n_new, &mut stats);

    // ---- compact: drop vertices not referenced by any face -----------------
    let mut remap = vec![u32::MAX; o_new.len()];
    let mut positions_out: Vec<[f32; 3]> = Vec::new();
    let mut faces = traced;
    for f in faces.iter_mut() {
        for v in f.iter_mut() {
            if remap[*v as usize] == u32::MAX {
                remap[*v as usize] = positions_out.len() as u32;
                let p = o_new[*v as usize];
                positions_out.push([p.x as f32, p.y as f32, p.z as f32]);
            }
            *v = remap[*v as usize];
        }
    }

    stats.output_vertices = positions_out.len() as u64;
    for f in &faces {
        match f.len() {
            3 => stats.triangles += 1,
            4 => stats.quads += 1,
            _ => {}
        }
    }
    let mut valence = vec![0u32; positions_out.len()];
    for f in &faces {
        for &v in f {
            valence[v as usize] += 1;
        }
    }
    stats.irregular_vertices = valence.iter().filter(|&&k| k != 4).count() as u64;

    if positions_out
        .iter()
        .any(|p| p.iter().any(|c| !c.is_finite()))
    {
        return Err(RetopoError::FieldDiverged);
    }
    match validate_closed_manifold(positions_out.len(), &faces) {
        Ok(topo) => stats.euler_output = topo.euler,
        Err(why) => {
            stats.euler_output = i64::MIN;
            if options.validate {
                return Err(RetopoError::ExtractionInvalid(why));
            }
        }
    }
    if options.validate && stats.euler_output != stats.euler_input {
        return Err(RetopoError::ExtractionInvalid(format!(
            "Euler characteristic changed from {} to {} (topology not preserved)",
            stats.euler_input, stats.euler_output
        )));
    }

    Ok(ExtractedMesh {
        positions: positions_out,
        faces,
        stats,
    })
}

/// Reference Step 5. Repeats until nothing changes:
///   * for every path i–j–k where j–k is the longest side of triangle
///     (i, j, k) and the height from i is below 0.3·scale: if i is close to
///     j or k, merge them; otherwise move i onto the j–k midpoint and
///     replace the edge j–k by i–k (the triangle collapses onto a line);
///   * for every edge with exactly two incident triangles (a quad split by
///     its diagonal), remove it, best-matching diagonal length first.
fn snap_and_clean(
    adj: &mut [Vec<u32>],
    o: &mut [Vector3<f64>],
    nrm: &mut [Vector3<f64>],
    scale: f64,
    stats: &mut ExtractionStats,
) {
    let thresh = 0.3 * scale;
    let height = |a: f64, b: f64, c: f64| {
        let s = 0.5 * (a + b + c);
        let area2 = (s * (s - a) * (s - b) * (s - c)).max(0.0);
        2.0 * area2.sqrt() / a
    };
    let has = |adj: &[Vec<u32>], a: u32, b: u32| adj[a as usize].contains(&b);
    let mut guard = 0;
    loop {
        guard += 1;
        let mut changed = false;
        loop {
            let mut changed_inner = false;
            let mut candidates: Vec<(f64, u32, u32, u32)> = Vec::new();
            for i in 0..adj.len() as u32 {
                let p_i = o[i as usize];
                for &j in &adj[i as usize] {
                    let p_j = o[j as usize];
                    for &k in &adj[j as usize] {
                        if k == i {
                            continue;
                        }
                        let p_k = o[k as usize];
                        let (a, b, c) =
                            ((p_j - p_k).norm(), (p_i - p_j).norm(), (p_i - p_k).norm());
                        if a > b.max(c) {
                            let h = height(a, b, c);
                            if h < thresh {
                                candidates.push((h, i, j, k));
                            }
                        }
                    }
                }
            }
            candidates.sort_by(|x, y| x.0.total_cmp(&y.0));
            for (h0, i, j, k) in candidates {
                if !has(adj, i, j) || !has(adj, j, k) {
                    continue;
                }
                let edge3 = has(adj, k, i);
                let (p_i, p_j, p_k) = (o[i as usize], o[j as usize], o[k as usize]);
                let (a, b, c) = ((p_j - p_k).norm(), (p_i - p_j).norm(), (p_i - p_k).norm());
                if height(a, b, c) != h0 {
                    continue;
                }
                if b < thresh || c < thresh {
                    let m = if b < thresh { j } else { k };
                    o[i as usize] = (p_i + o[m as usize]) * 0.5;
                    nrm[i as usize] = (nrm[i as usize] + nrm[m as usize]).normalize();
                    let m_list = std::mem::take(&mut adj[m as usize]);
                    for &x in &m_list {
                        if x == i {
                            continue;
                        }
                        let lx = &mut adj[x as usize];
                        for y in lx.iter_mut() {
                            if *y == m {
                                *y = i;
                            }
                        }
                        lx.sort_unstable();
                        lx.dedup();
                    }
                    let mut merged: Vec<u32> =
                        adj[i as usize].iter().copied().chain(m_list).collect();
                    merged.retain(|&x| x != i && x != m);
                    merged.sort_unstable();
                    merged.dedup();
                    adj[i as usize] = merged;
                } else {
                    o[i as usize] = (p_j + p_k) * 0.5;
                    nrm[i as usize] = (nrm[j as usize] + nrm[k as usize]).normalize();
                    remove_link(&mut adj[j as usize], k);
                    remove_link(&mut adj[k as usize], j);
                    if !edge3 {
                        adj[i as usize].push(k);
                        adj[k as usize].push(i);
                    }
                }
                changed = true;
                changed_inner = true;
                stats.snapped += 1;
            }
            if !changed_inner {
                break;
            }
        }

        // remove quad diagonals
        let mut candidates: Vec<(f64, u32, u32)> = Vec::new();
        for i in 0..adj.len() as u32 {
            let p_i = o[i as usize];
            for &j in &adj[i as usize] {
                let p_j = o[j as usize];
                let mut n_tris = 0;
                let mut length = 0.0;
                for &k in &adj[i as usize] {
                    if k == j || !has(adj, j, k) {
                        continue;
                    }
                    n_tris += 1;
                    let p_k = o[k as usize];
                    length += (p_k - p_i).norm() + (p_k - p_j).norm();
                }
                if n_tris == 2 {
                    let exp_diag = length / 4.0 * std::f64::consts::SQRT_2;
                    let diag = (p_i - p_j).norm();
                    let score = ((diag - exp_diag) / diag.min(exp_diag)).abs();
                    candidates.push((score, i, j));
                }
            }
        }
        candidates.sort_by(|x, y| x.0.total_cmp(&y.0));
        for (_, i, j) in candidates {
            if !has(adj, i, j) {
                continue;
            }
            let n_tris = adj[i as usize].iter().filter(|&&k| has(adj, j, k)).count();
            if n_tris == 2 {
                remove_link(&mut adj[i as usize], j);
                remove_link(&mut adj[j as usize], i);
                changed = true;
                stats.diagonals_removed += 1;
            }
        }
        if !changed || guard > 1000 {
            break;
        }
    }
}

#[derive(Clone, Copy)]
struct Link {
    id: u32,
    used: bool,
    keep: bool,
}

/// Steps 7 + 8: trace faces by degree (4, 3, 5..8), then fill holes.
fn extract_faces(
    adj_ids: &[Vec<u32>],
    o: &mut Vec<Vector3<f64>>,
    nrm: &mut Vec<Vector3<f64>>,
    stats: &mut ExtractionStats,
) -> Vec<Vec<u32>> {
    let mut adj: Vec<Vec<Link>> = adj_ids
        .iter()
        .map(|l| {
            l.iter()
                .map(|&id| Link {
                    id,
                    used: false,
                    keep: false,
                })
                .collect()
        })
        .collect();
    let nv = adj.len();
    // undirected edge set of the graph, used to veto splitting chords that
    // would duplicate an existing edge
    let mut edges: HashSet<(u32, u32)> = HashSet::new();
    for (i, l) in adj_ids.iter().enumerate() {
        for &j in l {
            edges.insert(((i as u32).min(j), (i as u32).max(j)));
        }
    }
    let mut faces: Vec<Vec<u32>> = Vec::new();

    // walk the rotation system from (cur, idx); target 0 = any length
    let walk = |adj: &mut Vec<Vec<Link>>,
                start: u32,
                start_idx: usize,
                target: usize|
     -> Option<Vec<u32>> {
        let mut result: Vec<(u32, usize)> = Vec::new();
        let (mut cur, mut idx) = (start, start_idx);
        loop {
            if adj[cur as usize][idx].used || (target > 0 && result.len() + 1 > target) {
                return None;
            }
            result.push((cur, idx));
            let next = adj[cur as usize][idx].id;
            let rank = adj[next as usize].len();
            let back = adj[next as usize].iter().position(|l| l.id == cur)?;
            if rank == 1 {
                return None;
            }
            cur = next;
            idx = (back + 1) % rank;
            if cur == start {
                if idx != start_idx || (target > 0 && result.len() != target) {
                    return None;
                }
                break;
            }
            if result.len() > 4 * nv {
                return None;
            }
        }
        for &(c, i) in &result {
            adj[c as usize][i].used = true;
        }
        Some(result.into_iter().map(|(c, _)| c).collect())
    };

    for deg in [4usize, 3, 5, 6, 7, 8] {
        for i in 0..nv {
            for j in 0..adj[i].len() {
                if let Some(cycle) = walk(&mut adj, i as u32, j, deg) {
                    stats.faces_traced += 1;
                    if !fill_polygon(cycle, o, nrm, &mut edges, &mut faces) {
                        stats.dropped_cycles += 1;
                    }
                }
            }
        }
    }

    // Step 8: keep only half-edges on hole boundaries (unused, twin used)
    for i in 0..nv {
        for j in 0..adj[i].len() {
            if adj[i][j].used {
                continue;
            }
            let id = adj[i][j].id as usize;
            if let Some(k) = adj[id].iter().position(|l| l.id == i as u32) {
                if adj[id][k].used {
                    adj[i][j].keep = true;
                    adj[id][k].keep = true;
                }
            }
        }
    }
    for list in adj.iter_mut() {
        list.retain(|l| l.keep);
    }
    for i in 0..nv {
        for j in 0..adj[i].len() {
            if adj[i][j].used {
                continue;
            }
            match walk(&mut adj, i as u32, j, 0) {
                Some(cycle) => {
                    stats.holes_filled += 1;
                    if !fill_polygon(cycle, o, nrm, &mut edges, &mut faces) {
                        stats.dropped_cycles += 1;
                    }
                }
                None => stats.dropped_cycles += 1,
            }
        }
    }
    faces
}

/// Emit a traced cycle as faces; returns `false` if it could not be
/// decomposed without breaking manifoldness (caller counts it as dropped).
///
/// Only chord cuts are used, so every corner (wedge) of the cycle stays
/// with exactly one output face — this matters for non-simple cycles that
/// visit a vertex twice, where re-pairing corners would pinch the vertex:
///   1. spurs `x v a v y` (an edge walked out and back) are removed;
///   2. a 3- or 4-cycle with distinct vertices is emitted;
///   3. otherwise the best quad ear `c[i..=i+3]` is peeled (distinct
///      vertices, chord `c[i]–c[i+3]` not yet an edge; corner angles
///      closest to 90°), else the best triangle ear;
///   4. a simple cycle with no valid ear is fanned around a new centroid.
fn fill_polygon(
    cycle: Vec<u32>,
    o: &mut Vec<Vector3<f64>>,
    nrm: &mut Vec<Vector3<f64>>,
    edges: &mut HashSet<(u32, u32)>,
    faces: &mut Vec<Vec<u32>>,
) -> bool {
    let key = |a: u32, b: u32| (a.min(b), a.max(b));
    let mut cycle = cycle;
    loop {
        remove_spurs(&mut cycle);
        let n = cycle.len();
        if n < 3 {
            return n == 0;
        }
        let simple = first_repeat(&cycle).is_none();
        if simple && (n == 3 || n == 4) {
            faces.push(cycle);
            return true;
        }
        let angle_at = |a: u32, b: u32, c: u32| -> f64 {
            let d0 = (o[a as usize] - o[b as usize]).normalize();
            let d1 = (o[c as usize] - o[b as usize]).normalize();
            d0.dot(&d1).clamp(-1.0, 1.0).acos().to_degrees()
        };
        let mut best: Option<(f64, usize, usize)> = None; // (score, start, size)
        for size in [4usize, 3] {
            if size >= n {
                continue;
            }
            for i in 0..n {
                let ear: Vec<u32> = (0..size).map(|k| cycle[(i + k) % n]).collect();
                if first_repeat(&ear).is_some() || edges.contains(&key(ear[0], ear[size - 1])) {
                    continue;
                }
                let mut score = 0.0;
                let target = if size == 4 { 90.0 } else { 60.0 };
                for k in 0..size {
                    let (a, b, c) = (ear[k], ear[(k + 1) % size], ear[(k + 2) % size]);
                    score += (angle_at(a, b, c) - target).abs();
                }
                if score.is_nan() {
                    continue;
                }
                if best.is_none_or(|(s, _, _)| score < s) {
                    best = Some((score, i, size));
                }
            }
            if best.is_some() {
                break; // prefer any valid quad ear over triangle ears
            }
        }
        match best {
            Some((_, i, size)) => {
                let ear: Vec<u32> = (0..size).map(|k| cycle[(i + k) % n]).collect();
                edges.insert(key(ear[0], ear[size - 1]));
                faces.push(ear);
                // drop the ear's interior vertices from the cycle
                let mut rm: Vec<usize> = (1..size - 1).map(|k| (i + k) % n).collect();
                rm.sort_unstable();
                for &r in rm.iter().rev() {
                    cycle.remove(r);
                }
            }
            None if simple => {
                let c = o.len() as u32;
                let centroid =
                    cycle.iter().map(|&v| o[v as usize]).sum::<Vector3<f64>>() / n as f64;
                let cn = cycle.iter().map(|&v| nrm[v as usize]).sum::<Vector3<f64>>();
                o.push(centroid);
                nrm.push(if cn.norm() > 1e-30 {
                    cn.normalize()
                } else {
                    Vector3::new(0.0, 0.0, 1.0)
                });
                for k in 0..n {
                    let (a, b) = (cycle[k], cycle[(k + 1) % n]);
                    edges.insert(key(a, c));
                    faces.push(vec![a, b, c]);
                }
                return true;
            }
            None => return false,
        }
    }
}

/// Remove out-and-back spurs: `… x v a v y …` → `… x v y …`.
fn remove_spurs(cycle: &mut Vec<u32>) {
    loop {
        let n = cycle.len();
        if n < 3 {
            // a 2-cycle (v a) is itself a spur
            if n == 2 {
                cycle.clear();
            }
            return;
        }
        let Some(i) = (0..n).find(|&i| cycle[(i + n - 1) % n] == cycle[(i + 1) % n]) else {
            return;
        };
        // cycle[i] is the spur tip; remove it and the repeated base after it
        let (a, b) = (i, (i + 1) % n);
        let (hi, lo) = (a.max(b), a.min(b));
        cycle.remove(hi);
        cycle.remove(lo);
    }
}

/// First pair (a, b), a < b, with cycle[a] == cycle[b] and minimal b.
fn first_repeat(cycle: &[u32]) -> Option<(usize, usize)> {
    let mut seen: HashMap<u32, usize> = HashMap::new();
    for (b, &v) in cycle.iter().enumerate() {
        if let Some(&a) = seen.get(&v) {
            return Some((a, b));
        }
        seen.insert(v, b);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::{solve_orientation, solve_position, FieldOptions};
    use crate::obj;
    use std::path::PathBuf;

    pub(crate) fn fixture(name: &str) -> (Vec<[f64; 3]>, Vec<[u32; 3]>) {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures")
            .join(name);
        let mesh = obj::load(&path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
        let mut positions: Vec<[f64; 3]> = mesh
            .positions
            .iter()
            .map(|p| [p[0] as f64, p[1] as f64, p[2] as f64])
            .collect();
        let mut tris = mesh.triangles;
        // weld coincident vertices, exactly like the engine pipeline does
        let (mn, mx) = positions.iter().fold(
            ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]),
            |(mut mn, mut mx), p| {
                for k in 0..3 {
                    mn[k] = mn[k].min(p[k]);
                    mx[k] = mx[k].max(p[k]);
                }
                (mn, mx)
            },
        );
        let diag = (0..3).map(|k| (mx[k] - mn[k]).powi(2)).sum::<f64>().sqrt();
        crate::weld::weld_vertices(&mut positions, &mut tris, (diag * 1e-6).max(1e-12), diag);
        (positions, tris)
    }

    /// Mirror of the engine's preparation: subdivide so the input can carry
    /// the lattice (edges ≤ min(scale/2, 2·average edge)).
    pub(crate) fn prepared(name: &str, target_faces: u32) -> (Vec<[f64; 3]>, Vec<[u32; 3]>) {
        let (mut pos, mut tris) = fixture(name);
        let scale = (crate::field::surface_area(&pos, &tris) / target_faces as f64).sqrt();
        let avg = crate::subdivide::average_edge_length(&pos, &tris);
        crate::subdivide::subdivide_long_edges(
            &mut pos,
            &mut tris,
            (scale * 0.5).min(avg * 2.0),
            5_000_000,
        )
        .unwrap();
        (pos, tris)
    }

    fn extract_named_with(
        name: &str,
        target_faces: u32,
        eo: &ExtractOptions,
    ) -> Result<ExtractedMesh, RetopoError> {
        let (pos, tris) = prepared(name, target_faces);
        let opts = FieldOptions::default();
        let (orient, _) = solve_orientation(&pos, &tris, &opts).unwrap();
        let pf = solve_position(&pos, &tris, &orient, target_faces, &opts).unwrap();
        extract_mesh_with(
            &pos,
            &tris,
            &orient.directions,
            &orient.normals,
            &pf.anchors,
            pf.scale,
            eo,
        )
    }

    fn extract_named(name: &str, target_faces: u32) -> ExtractedMesh {
        extract_named_with(name, target_faces, &ExtractOptions::default())
            .unwrap_or_else(|e| panic!("extraction failed for {name}: {e}"))
    }

    fn report(mesh: &ExtractedMesh) -> String {
        let s = &mesh.stats;
        format!(
            "verts={} faces={} quads={} tris={} traced={} holes={} dropped={} snapped={} diag={} chi {}→{} irregular={}",
            mesh.positions.len(),
            mesh.faces.len(),
            s.quads,
            s.triangles,
            s.faces_traced,
            s.holes_filled,
            s.dropped_cycles,
            s.snapped,
            s.diagonals_removed,
            s.euler_input,
            s.euler_output,
            s.irregular_vertices,
        )
    }

    fn assert_good(name: &str, mesh: &ExtractedMesh, chi: i64) {
        println!("{name}: {}", report(mesh));
        assert!(mesh
            .positions
            .iter()
            .all(|p| p.iter().all(|v| v.is_finite())));
        let topo =
            validate_closed_manifold(mesh.positions.len(), &mesh.faces).expect("closed manifold");
        assert_eq!(topo.euler, chi, "{name}: topology must be preserved");
        assert_eq!(mesh.stats.dropped_cycles, 0);
        let quads = mesh.stats.quads as f64 / mesh.faces.len() as f64;
        assert!(
            quads > 0.75,
            "{name}: quad-dominant output expected, got {:.0}% quads",
            quads * 100.0
        );
    }

    #[test]
    fn fill_polygon_keeps_corners_and_never_repeats_vertices() {
        let mut o: Vec<Vector3<f64>> = (0..8)
            .map(|k| {
                let a = k as f64 * std::f64::consts::TAU / 8.0;
                Vector3::new(a.cos(), a.sin(), 0.0)
            })
            .collect();
        let mut nrm = vec![Vector3::new(0.0, 0.0, 1.0); o.len()];
        // spur 1 -> 2 -> 1 collapses to the quad 0 1 3 4
        let (mut edges, mut faces) = (HashSet::new(), Vec::new());
        assert!(fill_polygon(
            vec![0, 1, 2, 1, 3, 4],
            &mut o,
            &mut nrm,
            &mut edges,
            &mut faces
        ));
        assert_eq!(faces, vec![vec![0, 1, 3, 4]]);
        // figure-eight through vertex 0: every face simple, every corner used once
        let (mut edges, mut faces) = (HashSet::new(), Vec::new());
        let cycle = vec![0, 1, 2, 3, 0, 5, 6, 7];
        assert!(fill_polygon(
            cycle.clone(),
            &mut o,
            &mut nrm,
            &mut edges,
            &mut faces
        ));
        for f in &faces {
            assert!(first_repeat(f).is_none(), "face {f:?} repeats a vertex");
        }
        for k in 0..cycle.len() {
            let n = cycle.len();
            let wedge = (cycle[(k + n - 1) % n], cycle[k], cycle[(k + 1) % n]);
            let at = |w: (u32, u32, u32)| {
                faces
                    .iter()
                    .any(|f| f.contains(&w.0) && f.contains(&w.1) && f.contains(&w.2))
            };
            if wedge.1 == 0 {
                assert!(at(wedge), "corner {wedge:?} lost");
            }
        }
    }

    #[test]
    fn validator_accepts_cube_and_rejects_holes() {
        let cube: Vec<Vec<u32>> = vec![
            vec![0, 3, 2, 1],
            vec![4, 5, 6, 7],
            vec![0, 1, 5, 4],
            vec![1, 2, 6, 5],
            vec![2, 3, 7, 6],
            vec![3, 0, 4, 7],
        ];
        let t = validate_closed_manifold(8, &cube).unwrap();
        assert_eq!(t.euler, 2);
        assert!(validate_closed_manifold(8, &cube[..5]).is_err());
        let mut flipped = cube.clone();
        flipped[0].reverse();
        assert!(validate_closed_manifold(8, &flipped).is_err());
    }

    #[test]
    fn open_mesh_returns_boundary_error() {
        let n = 4usize;
        let mut pos = Vec::new();
        for y in 0..n {
            for x in 0..n {
                pos.push([x as f64 * 0.1, y as f64 * 0.1, 0.0]);
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
        let opts = FieldOptions::default();
        let (orient, _) = solve_orientation(&pos, &tris, &opts).unwrap();
        let pf = solve_position(&pos, &tris, &orient, 18, &opts).unwrap();
        let err = extract_mesh(
            &pos,
            &tris,
            &orient.directions,
            &orient.normals,
            &pf.anchors,
            pf.scale,
        )
        .unwrap_err();
        assert!(matches!(err, RetopoError::UnsupportedBoundary));
    }

    #[test]
    fn icosphere_extracts_topologically_closed() {
        assert_good("icosphere", &extract_named("icosphere.obj", 320), 2);
    }

    #[test]
    fn uv_sphere_extracts() {
        assert_good("uv_sphere", &extract_named("uv_sphere.obj", 300), 2);
    }

    #[test]
    fn torus_preserves_genus() {
        assert_good("torus", &extract_named("torus.obj", 300), 0);
    }

    #[test]
    fn cube_extracts() {
        // 96 target faces → 16 per side, a 4×4 grid on each cube face
        assert_good("cube", &extract_named("cube.obj", 96), 2);
    }

    #[test]
    fn displaced_sphere_extracts() {
        assert_good(
            "displaced_sphere",
            &extract_named("displaced_sphere.obj", 1500),
            2,
        );
    }

    #[test]
    fn suzanne_is_rejected_as_open() {
        // Suzanne has open eye/mouth boundaries (and one non-manifold edge):
        // extraction must refuse instead of producing garbage.
        let err = extract_named_with("suzanne.obj", 500, &ExtractOptions::default()).unwrap_err();
        assert!(
            matches!(
                err,
                RetopoError::UnsupportedBoundary | RetopoError::NonManifoldInput
            ),
            "unexpected error {err}"
        );
    }

    /// Diagnostic table (run with --nocapture). Does not validate.
    #[test]
    fn extraction_report() {
        let eo = ExtractOptions {
            validate: false,
            ..ExtractOptions::default()
        };
        for (name, target) in [
            ("icosphere.obj", 320u32),
            ("uv_sphere.obj", 300),
            ("torus.obj", 300),
            ("cube.obj", 96),
            ("cylinder.obj", 300),
            ("displaced_sphere.obj", 1500),
        ] {
            match extract_named_with(name, target, &eo) {
                Ok(m) => println!("{name:22} {}", report(&m)),
                Err(e) => println!("{name:22} ERROR {e}"),
            }
        }
    }
}
