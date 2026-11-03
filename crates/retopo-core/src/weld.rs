//! Spatial-hash vertex welding and degenerate-face removal.

use std::collections::HashMap;

#[derive(Debug, Clone, Copy)]
pub struct WeldReport {
    /// Number of input vertices that were merged into another vertex.
    pub merged: usize,
    /// Number of faces removed because they became degenerate.
    pub degenerate_removed: u64,
    /// Number of faces removed because an identical face already existed.
    pub duplicates_removed: u64,
}

/// Weld vertices closer than `eps`, then drop degenerate or zero-area faces.
///
/// On return, `verts` contains only the welded representatives and `tris` the
/// surviving faces with updated indices.
pub fn weld_vertices(
    verts: &mut Vec<[f64; 3]>,
    tris: &mut Vec<[u32; 3]>,
    eps: f64,
    bbox_diag: f64,
) -> WeldReport {
    let input_count = verts.len();
    let cell = eps.max(1e-12);
    let inv_cell = 1.0 / cell;
    let eps2 = eps * eps;

    let cell_key = |p: &[f64; 3]| -> (i64, i64, i64) {
        (
            (p[0] * inv_cell).floor() as i64,
            (p[1] * inv_cell).floor() as i64,
            (p[2] * inv_cell).floor() as i64,
        )
    };

    let mut grid: HashMap<(i64, i64, i64), Vec<u32>> = HashMap::with_capacity(verts.len() * 2);
    let mut remap: Vec<u32> = Vec::with_capacity(verts.len());
    let mut welded: Vec<[f64; 3]> = Vec::with_capacity(verts.len());

    for (i, p) in verts.iter().enumerate() {
        let _ = i;
        let key = cell_key(p);
        let mut found: Option<u32> = None;
        'cells: for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let k = (key.0 + dx, key.1 + dy, key.2 + dz);
                    if let Some(bucket) = grid.get(&k) {
                        for &rep in bucket {
                            let q = &welded[rep as usize];
                            let d2 = (0..3).map(|c| (p[c] - q[c]).powi(2)).sum::<f64>();
                            if d2 <= eps2 {
                                found = Some(rep);
                                break 'cells;
                            }
                        }
                    }
                }
            }
        }
        match found {
            Some(rep) => remap.push(rep),
            None => {
                let idx = welded.len() as u32;
                welded.push(*p);
                grid.entry(key).or_default().push(idx);
                remap.push(idx);
            }
        }
    }

    let merged = input_count - welded.len();

    // ---- remap triangles, drop degenerate / zero-area faces --------------
    let area_eps = (bbox_diag * 1e-12).max(1e-18);
    let area_eps2 = area_eps * area_eps;
    let mut kept: Vec<[u32; 3]> = Vec::with_capacity(tris.len());
    let mut seen_faces: std::collections::HashSet<(u32, u32, u32)> =
        std::collections::HashSet::with_capacity(tris.len());
    let mut degenerate_removed: u64 = 0;
    let mut duplicates_removed: u64 = 0;

    for t in tris.drain(..) {
        let a = remap[t[0] as usize];
        let b = remap[t[1] as usize];
        let c = remap[t[2] as usize];
        if a == b || b == c || a == c {
            degenerate_removed += 1;
            continue;
        }
        let mut key = [a, b, c];
        key.sort_unstable();
        if !seen_faces.insert((key[0], key[1], key[2])) {
            duplicates_removed += 1;
            continue;
        }
        let pa = &welded[a as usize];
        let pb = &welded[b as usize];
        let pc = &welded[c as usize];
        let ab = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
        let ac = [pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]];
        let cross = [
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        ];
        let cross2 = cross.iter().map(|v| v * v).sum::<f64>();
        if cross2 < area_eps2 {
            degenerate_removed += 1;
            continue;
        }
        kept.push([a, b, c]);
    }

    *verts = welded;
    *tris = kept;

    WeldReport {
        merged,
        degenerate_removed,
        duplicates_removed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weld_merges_coincident_and_drops_degenerate() {
        // Two coincident "seam" vertices that should weld, one triangle is
        // degenerate after welding (shares the welded pair).
        let mut verts = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0000001, 0.0, 0.0], // welds to #1
        ];
        let mut tris = vec![[0, 1, 2], [0, 3, 2]];
        let report = weld_vertices(&mut verts, &mut tris, 1e-4, 2.0);
        assert_eq!(report.merged, 1);
        assert_eq!(verts.len(), 3);
        assert_eq!(tris.len(), 1); // the remapped copy is an identical duplicate, dropped
        assert_eq!(report.duplicates_removed, 1);
    }

    #[test]
    fn weld_drops_zero_area() {
        let mut verts = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        let mut tris = vec![[0, 1, 2]];
        let report = weld_vertices(&mut verts, &mut tris, 1e-6, 2.0);
        assert_eq!(report.degenerate_removed, 1);
        assert!(tris.is_empty());
    }
}
