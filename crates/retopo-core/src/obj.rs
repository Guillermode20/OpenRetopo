//! Minimal Wavefront OBJ loader used by tests and tooling.
//!
//! Only supports what our fixtures contain: `v x y z` lines and
//! `f a b c ...` lines (with optional `/normal/uv` suffixes). Faces with
//! more than three vertices are fan-triangulated.

use std::io::{BufRead, BufReader};
use std::path::Path;

#[derive(Debug, Clone, Default)]
pub struct ObjMesh {
    pub positions: Vec<[f32; 3]>,
    pub triangles: Vec<[u32; 3]>,
}

pub fn load(path: &Path) -> std::io::Result<ObjMesh> {
    let file = std::fs::File::open(path)?;
    let reader = BufReader::new(file);
    let mut mesh = ObjMesh::default();

    for line in reader.lines() {
        let line = line?;
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("v ") {
            let mut it = rest.split_whitespace();
            let mut p = [0.0f32; 3];
            for coord in p.iter_mut() {
                *coord = it.next().and_then(|s| s.parse().ok()).ok_or_else(|| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, "bad vertex line")
                })?;
            }
            mesh.positions.push(p);
        } else if let Some(rest) = line.strip_prefix("f ") {
            let mut idx: Vec<u32> = Vec::new();
            for tok in rest.split_whitespace() {
                let head = tok.split('/').next().unwrap_or("");
                let n: i64 = head.parse().map_err(|_| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, "bad face index")
                })?;
                // OBJ indices are 1-based; negative means relative to the end.
                let resolved = if n > 0 {
                    n - 1
                } else {
                    mesh.positions.len() as i64 + n
                };
                if resolved < 0 || resolved >= mesh.positions.len() as i64 {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "face index out of range",
                    ));
                }
                idx.push(resolved as u32);
            }
            if idx.len() >= 3 {
                for i in 1..idx.len() - 1 {
                    mesh.triangles.push([idx[0], idx[i], idx[i + 1]]);
                }
            }
        }
    }

    Ok(mesh)
}
