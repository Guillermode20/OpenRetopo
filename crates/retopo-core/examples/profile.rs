//! Stage timing for the field-aligned pipeline.
//!
//! cargo run --release -p openretopo-core --example profile -- mesh.obj [target_faces] [out.obj]
//!
//! Stages are delimited by the progress milestones `remesh()` reports
//! (weld 0.25, subdivide 0.30, orientation 0.55, position 0.80,
//! extraction 0.97). Env: OPENRETOPO_SYM=none|auto|x|y|z, OPENRETOPO_CURV=<weight>,
//! OPENRETOPO_ADAPT=<0..1>, OPENRETOPO_CREASE=<degrees, 0 = off>.

use std::time::Instant;

use openretopo_core::{obj, remesh, RemeshOptions, Symmetry};

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: profile <mesh.obj> [target_faces] [out.obj]");
    let target: u32 = args
        .next()
        .map(|s| s.parse().expect("target_faces"))
        .unwrap_or(12000);
    let out_path = args.next();
    let m = obj::load(std::path::Path::new(&path)).expect("load obj");
    let p: Vec<f32> = m.positions.iter().flatten().copied().collect();
    let i: Vec<u32> = m.triangles.iter().flatten().copied().collect();

    let t0 = Instant::now();
    let mut marks: Vec<(f32, f64)> = Vec::new();
    let mut cb = |t: f32| {
        marks.push((t, t0.elapsed().as_secs_f64()));
        false
    };
    let result = remesh(
        &p,
        &i,
        &RemeshOptions {
            target_faces: target,
            // env knobs for experiments: OPENRETOPO_SYM=none|auto|x|y|z,
            // OPENRETOPO_CURV=<weight>
            symmetry: match std::env::var("OPENRETOPO_SYM").as_deref() {
                Ok("auto") => Symmetry::Auto,
                Ok("x") => Symmetry::X,
                Ok("y") => Symmetry::Y,
                Ok("z") => Symmetry::Z,
                _ => Symmetry::None,
            },
            curvature_alignment: std::env::var("OPENRETOPO_CURV")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1.0),
            adaptivity: std::env::var("OPENRETOPO_ADAPT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(RemeshOptions::default().adaptivity),
            crease_degrees: std::env::var("OPENRETOPO_CREASE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(RemeshOptions::default().crease_degrees),
            ..RemeshOptions::default()
        },
        Some(&mut cb),
    );
    let total = t0.elapsed().as_secs_f64();
    let names = [
        (0.25, "validate+weld"),
        (0.30, "subdivide"),
        (0.55, "orientation"),
        (0.80, "position"),
        (0.97, "extraction"),
        (1.0, "stats"),
    ];
    let mut prev = 0.0;
    for (t, at) in &marks {
        if let Some((_, n)) = names.iter().find(|(v, _)| (v - t).abs() < 1e-6) {
            println!("  {n:14} {:8.3}s", at - prev);
            prev = *at;
        }
    }
    match result {
        Ok(o) => {
            if let Some(out) = &out_path {
                let mut text = String::new();
                for v in o.positions.chunks(3) {
                    text += &format!("v {} {} {}\n", v[0], v[1], v[2]);
                }
                let mut off = 0;
                for &n in &o.face_sizes {
                    text += "f";
                    for k in 0..n as usize {
                        text += &format!(" {}", o.indices[off + k] + 1);
                    }
                    text += "\n";
                    off += n as usize;
                }
                std::fs::write(out, text).expect("write output obj");
            }
            let s = &o.stats;
            println!(
                "{path}: {} tris in → {} faces ({:.1}% quads), χ {}→{}, poles v3/v5/other {}/{}/{}, +{} subdiv verts, {} attempt(s), sym axis {} dev {:.3}, total {total:.3}s",
                i.len() / 3,
                s.output_faces,
                100.0 * s.quads as f64 / s.output_faces as f64,
                s.euler_input,
                s.euler_output,
                s.poles_valence3,
                s.poles_valence5,
                s.poles_other,
                s.subdivision_vertices_added,
                s.extraction_attempts,
                s.symmetry_axis,
                s.symmetry_deviation
            );
        }
        Err(e) => println!("{path}: ERROR after {total:.3}s: {e}"),
    }
}
