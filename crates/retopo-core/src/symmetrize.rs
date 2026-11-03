//! Exact topological mirror symmetry for an extracted mesh.
//!
//! A mirror-coupled field only makes the two halves *nearly* alike: poles
//! and triangles still land on one side but not the other. When the
//! extracted mesh has an edge loop on the mirror plane (the position solve
//! pins one there), we keep one half, reflect it, and weld the halves along
//! that loop. The result is checked with the same closed-manifold / Euler
//! validator as extraction; if any step fails the caller keeps the
//! unmirrored mesh.

use std::collections::HashMap;

use crate::extract::validate_closed_manifold;
use crate::guidance::MirrorPlane;

/// A polygon mesh: positions and faces (counter-clockwise, 3 or 4 sides).
pub struct PolyMesh {
    pub positions: Vec<[f32; 3]>,
    pub faces: Vec<Vec<u32>>,
}

/// A mirrored mesh and, per vertex, the index of its mirror image (itself
/// for seam vertices).
pub struct Mirrored {
    pub mesh: PolyMesh,
    pub image: Vec<u32>,
}

/// Restore exact symmetry of positions after they were moved independently
/// (e.g. by relaxation): each mirror pair is replaced by the average of one
/// and the reflection of the other, seam vertices are put on the plane.
pub fn enforce(positions: &mut [[f32; 3]], image: &[u32], plane: MirrorPlane) {
    let ax = plane.axis;
    let reflect = |p: [f32; 3]| {
        let mut r = p;
        r[ax] = (2.0 * plane.offset - p[ax] as f64) as f32;
        r
    };
    for i in 0..positions.len().min(image.len()) {
        let j = image[i] as usize;
        if j == i {
            positions[i][ax] = plane.offset as f32;
        } else if i < j && j < positions.len() {
            let q = reflect(positions[j]);
            let mut avg = positions[i];
            for c in 0..3 {
                avg[c] = 0.5 * (avg[c] + q[c]);
            }
            positions[i] = avg;
            positions[j] = reflect(avg);
        }
    }
}

/// Which side of the plane a vertex lies on.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Side {
    Neg,
    On,
    Pos,
}

fn mean_edge_length(positions: &[[f32; 3]], faces: &[Vec<u32>]) -> f64 {
    let (mut sum, mut n) = (0.0f64, 0usize);
    for f in faces {
        for k in 0..f.len() {
            let (a, b) = (
                positions[f[k] as usize],
                positions[f[(k + 1) % f.len()] as usize],
            );
            let d = (0..3)
                .map(|c| (a[c] - b[c]) as f64)
                .map(|x| x * x)
                .sum::<f64>();
            sum += d.sqrt();
            n += 1;
        }
    }
    if n == 0 {
        0.0
    } else {
        sum / n as f64
    }
}

/// Mirror one half of `mesh` across `plane`. `keep_positive` selects the
/// half with coordinate above the plane offset. Returns `None` when the mesh
/// has no clean seam on the plane (a face straddles it, a seam edge is not
/// on it, or the welded result is not a closed manifold with Euler
/// characteristic `euler`).
pub fn mirror_half(
    mesh: &PolyMesh,
    plane: MirrorPlane,
    keep_positive: bool,
    euler: i64,
) -> Option<Mirrored> {
    let h = mean_edge_length(&mesh.positions, &mesh.faces);
    if !(h.is_finite() && h > 0.0) {
        return None;
    }
    let tol = 0.05 * h;
    let ax = plane.axis;
    let sign = if keep_positive { 1.0 } else { -1.0 };
    let side: Vec<Side> = mesh
        .positions
        .iter()
        .map(|p| {
            let d = sign * (p[ax] as f64 - plane.offset);
            if d > tol {
                Side::Pos
            } else if d < -tol {
                Side::Neg
            } else {
                Side::On
            }
        })
        .collect();

    // keep the faces of the chosen half; none may cross the plane, except
    // a quad lying diagonally across it (two opposite corners on the
    // plane), which is cut along that diagonal into one triangle per side
    let mut kept: Vec<Vec<u32>> = Vec::new();
    for f in &mesh.faces {
        let pos = f.iter().any(|&v| side[v as usize] == Side::Pos);
        let neg = f.iter().any(|&v| side[v as usize] == Side::Neg);
        match (pos, neg) {
            (true, false) => kept.push(f.clone()),
            (false, true) => {}
            (true, true) if f.len() == 4 => {
                let s4 = |k: usize| side[f[k % 4] as usize];
                let k = (0..2).find(|&k| s4(k) == Side::On && s4(k + 2) == Side::On)?;
                // corners k+1 and k+3 are on opposite sides
                let tri = if s4(k + 1) == Side::Pos {
                    vec![f[k], f[(k + 1) % 4], f[(k + 2) % 4]]
                } else {
                    vec![f[(k + 2) % 4], f[(k + 3) % 4], f[k]]
                };
                kept.push(tri);
            }
            _ => return None,
        }
    }
    if kept.is_empty() {
        return None;
    }
    // the open border of the half must lie entirely on the plane, and no
    // interior edge may run along it (it would get four faces once welded)
    let mut edge_uses: HashMap<(u32, u32), u32> = HashMap::new();
    for f in &kept {
        for k in 0..f.len() {
            let (a, b) = (f[k], f[(k + 1) % f.len()]);
            *edge_uses.entry((a.min(b), a.max(b))).or_default() += 1;
        }
    }
    for (&(a, b), &uses) in &edge_uses {
        let on = side[a as usize] == Side::On && side[b as usize] == Side::On;
        if (uses == 1) != on {
            return None;
        }
    }

    // assemble: half vertices, seam vertices snapped onto the plane, and
    // reflected copies of the off-plane ones
    let mut remap: Vec<u32> = vec![u32::MAX; mesh.positions.len()];
    let mut positions: Vec<[f32; 3]> = Vec::new();
    for f in &kept {
        for &v in f.iter() {
            if remap[v as usize] == u32::MAX {
                let mut p = mesh.positions[v as usize];
                if side[v as usize] == Side::On {
                    p[ax] = plane.offset as f32;
                }
                remap[v as usize] = positions.len() as u32;
                positions.push(p);
            }
        }
    }
    let half_count = positions.len();
    let mut mirror: Vec<u32> = vec![u32::MAX; half_count];
    for v in 0..mesh.positions.len() {
        let r = remap[v];
        if r == u32::MAX {
            continue;
        }
        if side[v] == Side::On {
            mirror[r as usize] = r;
        } else {
            let mut p = positions[r as usize];
            p[ax] = (2.0 * plane.offset - p[ax] as f64) as f32;
            mirror[r as usize] = positions.len() as u32;
            positions.push(p);
        }
    }
    let mut faces: Vec<Vec<u32>> = Vec::with_capacity(kept.len() * 2);
    for f in &kept {
        faces.push(f.iter().map(|&v| remap[v as usize]).collect());
    }
    for f in &kept {
        // reflection flips orientation: reverse the winding
        faces.push(
            f.iter()
                .rev()
                .map(|&v| mirror[remap[v as usize] as usize])
                .collect(),
        );
    }
    let topo = validate_closed_manifold(positions.len(), &faces).ok()?;
    if topo.euler != euler || topo.vertices != positions.len() {
        return None;
    }
    if positions.iter().flatten().any(|c| !c.is_finite()) {
        return None;
    }
    let mut image: Vec<u32> = (0..positions.len() as u32).collect();
    for (r, &m) in mirror.iter().enumerate() {
        image[r] = m;
        image[m as usize] = r as u32;
    }
    Some(Mirrored {
        mesh: PolyMesh { positions, faces },
        image,
    })
}

/// Irregularity score used to pick the better half: poles plus triangles
/// (weighted) — fewer is cleaner.
fn irregularity(mesh: &PolyMesh) -> usize {
    let mut valence = vec![0u32; mesh.positions.len()];
    let mut tris = 0usize;
    for f in &mesh.faces {
        tris += (f.len() != 4) as usize;
        for &v in f {
            valence[v as usize] += 1;
        }
    }
    valence.iter().filter(|&&k| k != 4).count() + 2 * tris
}

/// Mirror whichever half gives the cleaner result. `None` if neither half
/// has a clean seam.
pub fn symmetrize(mesh: &PolyMesh, plane: MirrorPlane, euler: i64) -> Option<Mirrored> {
    let a = mirror_half(mesh, plane, true, euler);
    let b = mirror_half(mesh, plane, false, euler);
    match (a, b) {
        (Some(a), Some(b)) => Some(if irregularity(&b.mesh) < irregularity(&a.mesh) {
            b
        } else {
            a
        }),
        (a, b) => a.or(b),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Axis-aligned box [-1,1]×[-1,1]×[-1,1] split into a 2×1×1 grid of
    /// quads per face so the x = 0 plane carries an edge loop.
    fn split_box() -> PolyMesh {
        // vertices: x in {-1,0,1}, y in {-1,1}, z in {-1,1}
        let mut positions = Vec::new();
        for x in [-1.0f32, 0.0, 1.0] {
            for y in [-1.0f32, 1.0] {
                for z in [-1.0f32, 1.0] {
                    positions.push([x, y, z]);
                }
            }
        }
        let id = |xi: u32, yi: u32, zi: u32| xi * 4 + yi * 2 + zi;
        let mut faces = Vec::new();
        for xi in 0..2 {
            // bottom (y=-1), top (y=+1), back (z=-1), front (z=+1)
            faces.push(vec![
                id(xi, 0, 0),
                id(xi + 1, 0, 0),
                id(xi + 1, 0, 1),
                id(xi, 0, 1),
            ]);
            faces.push(vec![
                id(xi, 1, 0),
                id(xi, 1, 1),
                id(xi + 1, 1, 1),
                id(xi + 1, 1, 0),
            ]);
            faces.push(vec![
                id(xi, 0, 0),
                id(xi, 1, 0),
                id(xi + 1, 1, 0),
                id(xi + 1, 0, 0),
            ]);
            faces.push(vec![
                id(xi, 0, 1),
                id(xi + 1, 0, 1),
                id(xi + 1, 1, 1),
                id(xi, 1, 1),
            ]);
        }
        // caps x = -1 and x = +1
        faces.push(vec![id(0, 0, 0), id(0, 0, 1), id(0, 1, 1), id(0, 1, 0)]);
        faces.push(vec![id(2, 0, 0), id(2, 1, 0), id(2, 1, 1), id(2, 0, 1)]);
        PolyMesh { positions, faces }
    }

    #[test]
    fn mirrors_a_box_with_a_seam_loop() {
        let mut m = split_box();
        assert_eq!(
            validate_closed_manifold(m.positions.len(), &m.faces)
                .unwrap()
                .euler,
            2
        );
        // make the +x half lopsided; the mirrored result must be symmetric
        for p in &mut m.positions {
            if p[0] > 0.5 {
                p[1] *= 1.5;
            }
        }
        let plane = MirrorPlane {
            axis: 0,
            offset: 0.0,
        };
        let out = mirror_half(&m, plane, true, 2).expect("clean seam").mesh;
        assert_eq!(out.faces.len(), m.faces.len());
        for p in &out.positions {
            let q = [-p[0], p[1], p[2]];
            assert!(
                out.positions.iter().any(|r| r == &q),
                "{p:?} has no mirror image"
            );
        }
        assert!(out
            .positions
            .iter()
            .any(|p| p[0] < -0.5 && p[1].abs() > 1.2));
    }

    #[test]
    fn rejects_a_face_straddling_the_plane() {
        let m = split_box();
        // plane at x = 0.5 cuts the +x quads
        let plane = MirrorPlane {
            axis: 0,
            offset: 0.5,
        };
        assert!(symmetrize(&m, plane, 2).is_none());
    }
}
