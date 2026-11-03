//! Stage timing for the field-aligned pipeline.
//!
//! cargo run --release -p openretopo-core --example profile -- mesh.obj [target_faces]
//!
//! Stages are delimited by the progress milestones `remesh()` reports
//! (weld 0.25, subdivide 0.30, orientation 0.55, position 0.80,
//! extraction 0.97).

use std::time::Instant;

use openretopo_core::{obj, remesh, RemeshOptions};

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: profile <mesh.obj> [target_faces]");
    let target: u32 = args
        .next()
        .map(|s| s.parse().expect("target_faces"))
        .unwrap_or(12000);
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
            let s = &o.stats;
            println!(
                "{path}: {} tris in → {} faces ({:.1}% quads), χ {}→{}, poles v3/v5/other {}/{}/{}, +{} subdiv verts, {} attempt(s), total {total:.3}s",
                i.len() / 3,
                s.output_faces,
                100.0 * s.quads as f64 / s.output_faces as f64,
                s.euler_input,
                s.euler_output,
                s.poles_valence3,
                s.poles_valence5,
                s.poles_other,
                s.subdivision_vertices_added,
                s.extraction_attempts
            );
        }
        Err(e) => println!("{path}: ERROR after {total:.3}s: {e}"),
    }
}
