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
        let n0 = face_normal(&[
            v(positions[x as usize]),
            v(positions[y as usize]),
            v(positions[c as usize]),
        ]);
        let n1 = face_normal(&[
            v(positions[y as usize]),
            v(positions[x as usize]),
            v(positions[d as usize]),
        ]);
        // never fold a quad over a crease
        if n0.dot(&n1) < 40f64.to_radians().cos() * n0.norm() * n1.norm() {
            continue;
        }
        let n = n0 + n1;
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

/// A found triangle walk: (worst corner score, (face slot, new quad) per
/// step, partner triangle, last quad slot).
type Walk = (f64, Vec<(usize, Vec<u32>)>, usize, usize);

/// Pair up triangles that are a few quads apart. A triangle next to a quad
/// forms a pentagon that can be re-split into a quad and a triangle on the
/// quad's far side, so a triangle can walk along a quad strip; when it
/// reaches another triangle the two merge into a quad. Every quad created on
/// the way must be convex with no corner more than `max_corner_deg` off
/// 90°, and no new edge may already exist. Faces touching a vertex for
/// which `protect` is true are never changed. Euler characteristic is
/// preserved (two triangles and k quads become k + 1 quads). Returns the
/// number of triangle pairs removed.
pub fn pair_triangles(
    positions: &[[f32; 3]],
    faces: &mut Vec<Vec<u32>>,
    protect: &dyn Fn(u32) -> bool,
    max_steps: usize,
    max_corner_deg: f64,
) -> usize {
    let limit = max_corner_deg.to_radians();
    let p = |i: u32| v(positions[i as usize]);
    let good_quad = |q: [u32; 4]| -> Option<f64> {
        let pts = q.map(p);
        quad_quality(pts, face_normal(&pts)).filter(|&w| w < limit)
    };
    let mut total = 0;
    for _pass in 0..8 {
        let mut edge_faces: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
        for (fi, f) in faces.iter().enumerate() {
            for k in 0..f.len() {
                let (a, b) = (f[k], f[(k + 1) % f.len()]);
                edge_faces
                    .entry((a.min(b), a.max(b)))
                    .or_default()
                    .push(fi as u32);
            }
        }
        let across = |fi: usize, a: u32, b: u32| -> Option<usize> {
            let l = edge_faces.get(&(a.min(b), a.max(b)))?;
            match l.as_slice() {
                [x, y] if *x as usize == fi => Some(*y as usize),
                [x, y] if *y as usize == fi => Some(*x as usize),
                _ => None,
            }
        };
        let frozen: Vec<bool> = faces
            .iter()
            .map(|f| f.iter().any(|&x| protect(x)))
            .collect();
        let mut touched = vec![false; faces.len()];
        let mut replace: Vec<Option<Vec<u32>>> = vec![None; faces.len()];
        let mut found = 0;
        for t0 in 0..faces.len() {
            if faces[t0].len() != 3 || touched[t0] || frozen[t0] {
                continue;
            }
            // best (worst-corner score, path) over the three exits
            let mut best: Option<Walk> = None;
            for k in 0..3 {
                let f = &faces[t0];
                // current triangle (u, v, c) with exit edge u→v
                let (mut u, mut w, mut c) = (f[k], f[(k + 1) % 3], f[(k + 2) % 3]);
                let mut cur = t0;
                let mut path: Vec<(usize, Vec<u32>)> = Vec::new(); // (face, new quad)
                let mut new_edges: Vec<(u32, u32)> = Vec::new();
                let mut score = 0.0f64;
                for _step in 0..=max_steps {
                    let Some(g) = across(cur, u, w) else { break };
                    if touched[g] || frozen[g] || path.iter().any(|(x, _)| *x == g) || g == t0 {
                        break;
                    }
                    let gf = &faces[g];
                    if gf.len() == 3 {
                        if path.is_empty() {
                            break; // adjacent pair: merge_triangles' call
                        }
                        // merge the walked triangle (u, w, c) with g
                        let Some(d) = gf.iter().copied().find(|&x| x != u && x != w) else {
                            break;
                        };
                        if let Some(s) = good_quad([u, d, w, c]) {
                            let score = score.max(s);
                            if best.as_ref().is_none_or(|b| score < b.0) {
                                let mut pth = path.clone();
                                pth.push((g, vec![u, d, w, c]));
                                best = Some((score, pth, g, cur));
                            }
                        }
                        break;
                    }
                    if gf.len() != 4 {
                        break;
                    }
                    // quad reads w→u→x→y (it holds the edge reversed)
                    let Some(r) = (0..4).find(|&j| gf[j] == w && gf[(j + 1) % 4] == u) else {
                        break;
                    };
                    let (x, y) = (gf[(r + 2) % 4], gf[(r + 3) % 4]);
                    // pentagon u→x→y→w→c; new triangle on edge x→y:
                    // A: tri (u,x,y) + quad (u,y,w,c), new edge u–y
                    // B: tri (x,y,w) + quad (u,x,w,c), new edge x–w
                    let exists = |a: u32, b: u32| {
                        edge_faces.contains_key(&(a.min(b), a.max(b)))
                            || new_edges.contains(&(a.min(b), a.max(b)))
                    };
                    let qa = (!exists(u, y)).then(|| good_quad([u, y, w, c])).flatten();
                    let qb = (!exists(x, w)).then(|| good_quad([u, x, w, c])).flatten();
                    let pick = match (qa, qb) {
                        (Some(a), Some(b)) => Some(a <= b),
                        (Some(_), None) => Some(true),
                        (None, Some(_)) => Some(false),
                        (None, None) => None,
                    };
                    let Some(use_a) = pick else { break };
                    // the quad replaces the walked triangle's face; the
                    // triangle moves into g
                    if use_a {
                        score = score.max(qa.unwrap());
                        path.push((cur, vec![u, y, w, c]));
                        new_edges.push((u.min(y), u.max(y)));
                        (u, w, c) = (x, y, u);
                    } else {
                        score = score.max(qb.unwrap());
                        path.push((cur, vec![u, x, w, c]));
                        new_edges.push((x.min(w), x.max(w)));
                        (u, w, c) = (x, y, w);
                    }
                    cur = g;
                }
            }
            let Some((_, path, partner, last)) = best else {
                continue;
            };
            // slots: the start triangle and every quad but the last take
            // the quad left behind by each move, the partner's slot takes
            // the merged quad, the last quad's slot (where the walking
            // triangle ended up) is emptied
            for (fi, q) in path {
                touched[fi] = true;
                replace[fi] = Some(q);
            }
            touched[partner] = true;
            touched[last] = true;
            found += 1;
        }
        if found == 0 {
            break;
        }
        let old = std::mem::take(faces);
        for (fi, f) in old.into_iter().enumerate() {
            match replace[fi].take() {
                Some(q) => faces.push(q),
                None if touched[fi] => {}
                None => faces.push(f),
            }
        }
        total += found;
    }
    total
}

/// Remove valence-2 vertices ("doublets"): a vertex used by exactly two
/// faces that share both of its edges is a fold in the quad grid. The two
/// faces become one (two quads → a quad, a quad and a triangle → a
/// triangle). Vertex, edge and face counts drop by 1, 2 and 1, so the Euler
/// characteristic is unchanged. Dissolved vertices are left unreferenced
/// (see [`compact`]). Returns the number dissolved.
pub fn dissolve_doublets(faces: &mut Vec<Vec<u32>>, vertex_count: usize) -> usize {
    let mut removed = 0;
    loop {
        let mut incident: Vec<Vec<u32>> = vec![Vec::new(); vertex_count];
        for (fi, f) in faces.iter().enumerate() {
            for &v in f {
                incident[v as usize].push(fi as u32);
            }
        }
        let mut dead = vec![false; faces.len()];
        let mut replace: Vec<Option<Vec<u32>>> = vec![None; faces.len()];
        let mut any = false;
        for (vtx, inc) in incident.iter().enumerate() {
            let [f1, f2] = inc.as_slice() else { continue };
            let (f1, f2) = (*f1 as usize, *f2 as usize);
            if dead[f1] || dead[f2] || replace[f1].is_some() || replace[f2].is_some() {
                continue; // overlapping doublet: next round
            }
            let vtx = vtx as u32;
            // rotate each face so it starts just after vtx: [a, …, b]
            let open = |f: &[u32]| -> Option<Vec<u32>> {
                let k = f.iter().position(|&x| x == vtx)?;
                Some((1..f.len()).map(|j| f[(k + j) % f.len()]).collect())
            };
            let (Some(p), Some(q)) = (open(&faces[f1]), open(&faces[f2])) else {
                continue;
            };
            // f1 reads vtx→a … b→vtx, f2 must read vtx→b … a→vtx
            let (a, b) = (p[0], p[p.len() - 1]);
            if q[0] != b || q[q.len() - 1] != a {
                continue;
            }
            // merged boundary: a … b (from f1) then b … a (from f2), with
            // the shared corners once
            let mut merged: Vec<u32> = p.clone();
            merged.extend_from_slice(&q[1..q.len() - 1]);
            if merged.len() < 3 || merged.len() > 4 {
                continue;
            }
            let mut s = merged.clone();
            s.sort_unstable();
            s.dedup();
            if s.len() != merged.len() {
                continue;
            }
            replace[f1] = Some(merged);
            dead[f2] = true;
            removed += 1;
            any = true;
        }
        if !any {
            return removed;
        }
        let old = std::mem::take(faces);
        for (fi, f) in old.into_iter().enumerate() {
            if dead[fi] {
                continue;
            }
            faces.push(replace[fi].take().unwrap_or(f));
        }
    }
}

/// Drop unreferenced vertices, renumbering faces. Returns the old→new map
/// (`u32::MAX` for dropped vertices).
pub fn compact(positions: &mut Vec<[f32; 3]>, faces: &mut [Vec<u32>]) -> Vec<u32> {
    let mut remap = vec![u32::MAX; positions.len()];
    let mut out = Vec::with_capacity(positions.len());
    for f in faces.iter_mut() {
        for v in f.iter_mut() {
            if remap[*v as usize] == u32::MAX {
                remap[*v as usize] = out.len() as u32;
                out.push(positions[*v as usize]);
            }
            *v = remap[*v as usize];
        }
    }
    *positions = out;
    remap
}

/// One step of quad subdivision: every n-sided face becomes n quads around
/// a face point, every edge gets a midpoint. Output is all quads. New
/// points are placed linearly (the caller projects them onto the surface).
/// `image` (per-vertex mirror image) is carried over when present.
pub fn subdivide_quads(
    positions: &mut Vec<[f32; 3]>,
    faces: &mut Vec<Vec<u32>>,
    image: Option<&mut Vec<u32>>,
) {
    let mut edge_point: HashMap<(u32, u32), u32> = HashMap::new();
    let mut edges_in_order: Vec<(u32, u32)> = Vec::new();
    let mid = |a: [f32; 3], b: [f32; 3]| {
        [
            0.5 * (a[0] + b[0]),
            0.5 * (a[1] + b[1]),
            0.5 * (a[2] + b[2]),
        ]
    };
    for f in faces.iter() {
        for k in 0..f.len() {
            let (a, b) = (f[k], f[(k + 1) % f.len()]);
            let e = (a.min(b), a.max(b));
            if let std::collections::hash_map::Entry::Vacant(slot) = edge_point.entry(e) {
                slot.insert(positions.len() as u32);
                positions.push(mid(positions[a as usize], positions[b as usize]));
                edges_in_order.push(e);
            }
        }
    }
    let first_face_point = positions.len() as u32;
    let mut face_key: HashMap<Vec<u32>, u32> = HashMap::new();
    let mut new_faces = Vec::with_capacity(faces.len() * 4);
    for (fi, f) in faces.iter().enumerate() {
        let n = f.len();
        let c = f.iter().fold([0.0f32; 3], |s, &v| {
            let p = positions[v as usize];
            [s[0] + p[0], s[1] + p[1], s[2] + p[2]]
        });
        let fp = positions.len() as u32;
        positions.push([c[0] / n as f32, c[1] / n as f32, c[2] / n as f32]);
        let mut key = f.clone();
        key.sort_unstable();
        face_key.insert(key, fi as u32);
        let ep = |a: u32, b: u32| edge_point[&(a.min(b), a.max(b))];
        for k in 0..n {
            let (prev, cur, next) = (f[(k + n - 1) % n], f[k], f[(k + 1) % n]);
            new_faces.push(vec![cur, ep(cur, next), fp, ep(prev, cur)]);
        }
    }
    if let Some(image) = image {
        let old = image.len() as u32;
        image.resize(positions.len(), u32::MAX);
        for (k, &(a, b)) in edges_in_order.iter().enumerate() {
            let (ia, ib) = (image[a as usize], image[b as usize]);
            image[(old as usize) + k] = edge_point
                .get(&(ia.min(ib), ia.max(ib)))
                .copied()
                .unwrap_or(old + k as u32);
        }
        for (fi, f) in faces.iter().enumerate() {
            let mut key: Vec<u32> = f.iter().map(|&v| image[v as usize]).collect();
            key.sort_unstable();
            image[(first_face_point + fi as u32) as usize] = face_key
                .get(&key)
                .map_or(first_face_point + fi as u32, |&g| first_face_point + g);
        }
    }
    *faces = new_faces;
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

/// Spatial hash of crease segments.
pub struct CreaseIndex {
    segs: Vec<(V3, V3)>,
    cell: f64,
    grid: HashMap<(i64, i64, i64), Vec<u32>>,
}

impl CreaseIndex {
    pub fn new(segs: Vec<(V3, V3)>, cell: f64) -> Self {
        let mut grid: HashMap<(i64, i64, i64), Vec<u32>> = HashMap::new();
        let key = |x: f64| (x / cell).floor() as i64;
        for (si, (a, b)) in segs.iter().enumerate() {
            let (lo, hi) = (a.inf(b), a.sup(b));
            for x in key(lo.x)..=key(hi.x) {
                for y in key(lo.y)..=key(hi.y) {
                    for z in key(lo.z)..=key(hi.z) {
                        grid.entry((x, y, z)).or_default().push(si as u32);
                    }
                }
            }
        }
        CreaseIndex { segs, cell, grid }
    }

    /// Distance to and closest point on the nearest segment within about
    /// one cell.
    pub fn nearest(&self, p: V3) -> Option<(f64, V3)> {
        if self.segs.is_empty() {
            return None;
        }
        let k = |x: f64| (x / self.cell).floor() as i64;
        let (cx, cy, cz) = (k(p.x), k(p.y), k(p.z));
        let mut best: Option<(f64, V3)> = None;
        for x in cx - 1..=cx + 1 {
            for y in cy - 1..=cy + 1 {
                for z in cz - 1..=cz + 1 {
                    let Some(list) = self.grid.get(&(x, y, z)) else {
                        continue;
                    };
                    for &si in list {
                        let (a, b) = self.segs[si as usize];
                        let ab = b - a;
                        let t = ((p - a).dot(&ab) / ab.norm_squared().max(1e-30)).clamp(0.0, 1.0);
                        let q = a + ab * t;
                        let d = (q - p).norm();
                        if best.is_none_or(|(bd, _)| d < bd) {
                            best = Some((d, q));
                        }
                    }
                }
            }
        }
        best
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

/// Blend between the size-evening target (mean of face centroids) and the
/// shape target (square / equilateral corners) in [`relax`].
const SHAPE: f64 = 0.5;

/// Tangential relaxation: each vertex moves toward the mean of its incident
/// face centroids (tangential part only), then onto the input surface.
/// Vertices on `seam` stay on the plane. A vertex whose move would turn an
/// incident face by more than ~70° or shrink it below 10% is reverted.
pub fn relax(
    positions: &mut [[f32; 3]],
    faces: &[Vec<u32>],
    surface: &SurfaceIndex,
    creases: &CreaseIndex,
    corners: &[[f64; 3]],
    seam: Option<MirrorPlane>,
    iterations: usize,
) {
    let n = positions.len();
    // crease vertices: on (or very near) a crease line. They slide along it
    // (between their two crease neighbours) instead of over the surface.
    let h = {
        let (mut s, mut c) = (0.0, 0usize);
        for f in faces {
            for k in 0..f.len() {
                s += (v(positions[f[k] as usize]) - v(positions[f[(k + 1) % f.len()] as usize]))
                    .norm();
                c += 1;
            }
        }
        s / c.max(1) as f64
    };
    let on_crease: Vec<bool> = positions
        .iter()
        .map(|&p| creases.nearest(v(p)).is_some_and(|(d, _)| d < 0.3 * h))
        .collect();
    let mut crease_nbrs: Vec<Vec<u32>> = vec![Vec::new(); n];
    for f in faces {
        for k in 0..f.len() {
            let (a, b) = (f[k], f[(k + 1) % f.len()]);
            if on_crease[a as usize] && on_crease[b as usize] {
                crease_nbrs[a as usize].push(b);
                crease_nbrs[b as usize].push(a);
            }
        }
    }
    for list in &mut crease_nbrs {
        list.sort_unstable();
        list.dedup();
    }
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
    // crease corners: the nearest output vertex (within one edge) is
    // pulled onto each corner and then held there
    let mut pin: Vec<Option<V3>> = vec![None; n];
    {
        let key = |p: V3| {
            (
                (p.x / h).floor() as i64,
                (p.y / h).floor() as i64,
                (p.z / h).floor() as i64,
            )
        };
        let mut grid: HashMap<(i64, i64, i64), Vec<u32>> = HashMap::new();
        for (i, &p) in pos.iter().enumerate() {
            if !incident[i].is_empty() {
                grid.entry(key(p)).or_default().push(i as u32);
            }
        }
        for c in corners {
            let c = V3::from(*c);
            let (kx, ky, kz) = key(c);
            let mut best: Option<(f64, usize)> = None;
            for x in kx - 1..=kx + 1 {
                for y in ky - 1..=ky + 1 {
                    for z in kz - 1..=kz + 1 {
                        for &i in grid.get(&(x, y, z)).into_iter().flatten() {
                            let d = (pos[i as usize] - c).norm();
                            if d < h && best.is_none_or(|(bd, _)| d < bd) {
                                best = Some((d, i as usize));
                            }
                        }
                    }
                }
            }
            if let Some((_, i)) = best {
                if pin[i].is_none() {
                    pin[i] = Some(c);
                }
            }
        }
    }
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
            // how sharply the surface bends here: the widest angle between
            // the vertex normal and an incident face normal. On a ridge a
            // tangential step slides the vertex off the ridge and the
            // faces cut across it, so the step shrinks with the bend.
            let bend = incident[i]
                .iter()
                .map(|&fi| {
                    let f = fnorm[fi as usize];
                    nrm.dot(&f) / f.norm().max(1e-30)
                })
                .fold(1.0f64, f64::min);
            // cos 15° → full step, cos 45° → none
            let step = ((bend - 0.707) / (0.966 - 0.707)).clamp(0.0, 1.0) * 0.5;
            let mut p = if let Some(c) = pin[i] {
                c
            } else if on_crease[i] {
                let p = match crease_nbrs[i].as_slice() {
                    [a, b] => {
                        let (pa, pb) = (pos[*a as usize], pos[*b as usize]);
                        let (u, w) = (pos[i] - pa, pb - pos[i]);
                        // a crease that turns by more than ~30° here is a
                        // corner (e.g. a toe tip): hold it
                        if u.dot(&w) < 0.866 * u.norm() * w.norm() {
                            pos[i]
                        } else {
                            pos[i] + ((pa + pb) * 0.5 - pos[i]) * 0.5
                        }
                    }
                    _ => pos[i], // crease junction / end: hold
                };
                creases.nearest(p).map_or(p, |(_, q)| q)
            } else {
                c /= incident[i].len() as f64;
                // shape target: where this corner would sit if each
                // incident face were a square (or equilateral triangle)
                // built on its two neighbours along this vertex's edges
                let mut ideal = V3::zeros();
                for &fi in &incident[i] {
                    let f = &faces[fi as usize];
                    let m = f.len();
                    let Some(k) = f.iter().position(|&x| x as usize == i) else {
                        continue;
                    };
                    let (a, b) = (
                        pos[f[(k + 1) % m] as usize],
                        pos[f[(k + m - 1) % m] as usize],
                    );
                    let half = if m == 3 { 0.866 } else { 0.5 };
                    ideal += (a + b) * 0.5 + nrm.cross(&(a - b)) * half;
                }
                ideal /= incident[i].len() as f64;
                let tgt = c * (1.0 - SHAPE) + ideal * SHAPE;
                let d = tgt - pos[i];
                let p = pos[i] + (d - nrm * nrm.dot(&d)) * step;
                surface.project(p, nrm).unwrap_or(p)
            };
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
    use crate::extract::validate_closed_manifold;

    /// Cube [-1,1]³ with each side split into a 2×2 grid of quads.
    fn grid_cube() -> (Vec<[f32; 3]>, Vec<Vec<u32>>) {
        let mut pos: Vec<[f32; 3]> = Vec::new();
        let mut id: HashMap<(i32, i32, i32), u32> = HashMap::new();
        let mut vid = |p: (i32, i32, i32), pos: &mut Vec<[f32; 3]>| -> u32 {
            *id.entry(p).or_insert_with(|| {
                pos.push([p.0 as f32, p.1 as f32, p.2 as f32]);
                pos.len() as u32 - 1
            })
        };
        let mut faces = Vec::new();
        // each side: axis, sign; grid coordinates in -1..=1
        for axis in 0..3 {
            for sign in [-1, 1] {
                let (u, w) = ((axis + 1) % 3, (axis + 2) % 3);
                for i in -1..1 {
                    for j in -1..1 {
                        let corner = |a: i32, b: i32| {
                            let mut c = [0i32; 3];
                            c[axis] = sign;
                            c[u] = a;
                            c[w] = b;
                            (c[0], c[1], c[2])
                        };
                        let mut q = vec![
                            vid(corner(i, j), &mut pos),
                            vid(corner(i + 1, j), &mut pos),
                            vid(corner(i + 1, j + 1), &mut pos),
                            vid(corner(i, j + 1), &mut pos),
                        ];
                        if sign < 0 {
                            q.reverse();
                        }
                        faces.push(q);
                    }
                }
            }
        }
        (pos, faces)
    }

    #[test]
    fn grid_cube_is_closed() {
        let (pos, faces) = grid_cube();
        assert_eq!(
            validate_closed_manifold(pos.len(), &faces).unwrap().euler,
            2
        );
    }

    #[test]
    fn dissolves_a_doublet() {
        // true doublet: two quads (x,m,y,c) and (y,m,x,d) glued along x–m–y
        let pos: Vec<[f32; 3]> = vec![
            [0.0, 0.0, 0.0],  // x 0
            [1.0, 0.0, 0.0],  // m 1
            [2.0, 0.0, 0.0],  // y 2
            [1.0, 1.0, 0.0],  // c 3
            [1.0, -1.0, 0.0], // d 4
        ];
        let mut faces = vec![vec![0, 1, 2, 3], vec![2, 1, 0, 4]];
        assert_eq!(dissolve_doublets(&mut faces, pos.len()), 1);
        assert_eq!(faces.len(), 1);
        let mut q = faces[0].clone();
        q.sort_unstable();
        assert_eq!(q, vec![0, 2, 3, 4]);
    }

    #[test]
    fn subdivision_gives_quads_and_keeps_the_mirror_image() {
        let (mut pos, mut faces) = grid_cube();
        // mirror x → −x
        let mut image: Vec<u32> = (0..pos.len())
            .map(|i| {
                let p = pos[i];
                pos.iter().position(|q| *q == [-p[0], p[1], p[2]]).unwrap() as u32
            })
            .collect();
        subdivide_quads(&mut pos, &mut faces, Some(&mut image));
        assert!(faces.iter().all(|f| f.len() == 4));
        assert_eq!(faces.len(), 24 * 4);
        assert_eq!(
            validate_closed_manifold(pos.len(), &faces).unwrap().euler,
            2
        );
        for (i, &m) in image.iter().enumerate() {
            let (p, q) = (pos[i], pos[m as usize]);
            assert!((p[0] + q[0]).abs() < 1e-6 && p[1] == q[1] && p[2] == q[2]);
        }
    }

    #[test]
    fn two_triangles_a_strip_apart_pair_up() {
        // flat 4×1 strip of unit quads; cut the two end quads into
        // triangles so a lone triangle sits at each end of a quad strip
        let mut pos: Vec<[f32; 3]> = Vec::new();
        for x in 0..=4 {
            for y in 0..=1 {
                pos.push([x as f32, y as f32, 0.0]);
            }
        }
        let id = |x: u32, y: u32| x * 2 + y;
        // end quads split in two: the inner triangle at each end is two
        // quads away from the other end's inner triangle
        let mut faces: Vec<Vec<u32>> = vec![
            vec![id(0, 0), id(1, 0), id(1, 1)],
            vec![id(0, 0), id(1, 1), id(0, 1)],
            vec![id(1, 0), id(2, 0), id(2, 1), id(1, 1)],
            vec![id(2, 0), id(3, 0), id(3, 1), id(2, 1)],
            vec![id(3, 0), id(4, 0), id(4, 1)],
            vec![id(3, 0), id(4, 1), id(3, 1)],
        ];
        let before: usize = faces.iter().filter(|f| f.len() == 3).count();
        let n = pair_triangles(&pos, &mut faces, &|_| false, 6, 89.0);
        let after: usize = faces.iter().filter(|f| f.len() == 3).count();
        assert!(n >= 1, "no pair walked");
        assert_eq!(after, before - 2 * n);
        // edges stay manifold (each used at most twice, consistently)
        let mut half = std::collections::HashSet::new();
        for f in &faces {
            for k in 0..f.len() {
                assert!(
                    half.insert((f[k], f[(k + 1) % f.len()])),
                    "duplicate half-edge"
                );
            }
        }
    }

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
