//! End-to-end tests of the public `remesh()` entry point in field-aligned
//! mode: validation, topology preservation, determinism, failure modes.

use std::collections::HashMap;
use std::path::PathBuf;

use openretopo_core::{
    obj, remesh, RemeshMode, RemeshOptions, RemeshOutput, RetopoError, Symmetry,
};

fn load(name: &str) -> (Vec<f32>, Vec<u32>) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .join(name);
    let m = obj::load(&path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
    (
        m.positions.iter().flatten().copied().collect(),
        m.triangles.iter().flatten().copied().collect(),
    )
}

fn run(name: &str, target: u32) -> Result<RemeshOutput, RetopoError> {
    let (p, i) = load(name);
    remesh(
        &p,
        &i,
        &RemeshOptions {
            target_faces: target,
            ..RemeshOptions::default()
        },
        None,
    )
}

/// Independent re-check of the output buffers: closed, oriented manifold.
fn check_closed_manifold(out: &RemeshOutput) -> i64 {
    let nv = out.positions.len() / 3;
    assert!(
        out.positions.iter().all(|v| v.is_finite()),
        "non-finite output position"
    );
    let mut half: HashMap<(u32, u32), usize> = HashMap::new();
    let mut off = 0usize;
    for (fi, &n) in out.face_sizes.iter().enumerate() {
        let f = &out.indices[off..off + n as usize];
        off += n as usize;
        assert!(n == 3 || n == 4, "face {fi} has {n} sides");
        for k in 0..f.len() {
            assert!((f[k] as usize) < nv);
            assert!(
                half.insert((f[k], f[(k + 1) % f.len()]), fi).is_none(),
                "duplicate half-edge"
            );
        }
    }
    assert_eq!(off, out.indices.len());
    for &(a, b) in half.keys() {
        assert!(half.contains_key(&(b, a)), "boundary edge ({a},{b})");
    }
    nv as i64 - (half.len() / 2) as i64 + out.face_sizes.len() as i64
}

#[test]
fn closed_fixtures_remesh_with_topology_preserved() {
    for (name, target, chi) in [
        ("icosphere.obj", 320u32, 2i64),
        ("uv_sphere.obj", 300, 2),
        ("torus.obj", 300, 0),
        ("cube.obj", 96, 2),
        ("cylinder.obj", 300, 2),
        ("displaced_sphere.obj", 1500, 2),
    ] {
        let out = run(name, target).unwrap_or_else(|e| panic!("{name}: {e}"));
        let s = &out.stats;
        assert_eq!(check_closed_manifold(&out), chi, "{name}");
        assert_eq!(s.euler_input, chi, "{name}");
        assert_eq!(s.euler_output, chi, "{name}");
        assert_eq!(s.dropped_cycles, 0, "{name}");
        assert_eq!(s.quads + s.triangles, s.output_faces, "{name}");
        assert!(
            s.quads as f64 / s.output_faces as f64 > 0.85,
            "{name}: {} quads of {}",
            s.quads,
            s.output_faces
        );
        // face count lands near the request
        let ratio = s.output_faces as f64 / target as f64;
        assert!(
            (0.6..1.6).contains(&ratio),
            "{name}: {} faces for target {target}",
            s.output_faces
        );
    }
}

#[test]
fn symmetry_constraint_mirrors_the_result() {
    // cylinder is mirror-symmetric; an off-centre copy checks the plane offset
    let (mut p, i) = load("cylinder.obj");
    for v in p.chunks_mut(3) {
        v[0] += 3.0;
    }
    let run_sym = |symmetry| {
        remesh(
            &p,
            &i,
            &RemeshOptions {
                target_faces: 300,
                symmetry,
                ..RemeshOptions::default()
            },
            None,
        )
        .unwrap()
    };
    let free = run_sym(Symmetry::None);
    let auto = run_sym(Symmetry::Auto);
    let x = run_sym(Symmetry::X);
    assert_eq!(free.stats.symmetry_axis, 0);
    assert_ne!(auto.stats.symmetry_axis, 0, "auto found no plane");
    assert_eq!(x.stats.symmetry_axis, 1);
    for out in [&auto, &x] {
        assert_eq!(check_closed_manifold(out), 2);
        assert!(
            out.stats.symmetry_deviation < 0.01,
            "mirror deviation {} cells",
            out.stats.symmetry_deviation
        );
    }
    // X is exact: every vertex has a mirror image (plane x = 3), and the
    // mirrored face set is the face set (same topology on both sides)
    let pos: Vec<[f32; 3]> = x.positions.chunks(3).map(|v| [v[0], v[1], v[2]]).collect();
    let tol = 1e-3 * x.stats.edge_length as f32;
    let image: Vec<usize> = pos
        .iter()
        .map(|p| {
            let q = [6.0 - p[0], p[1], p[2]];
            (0..pos.len())
                .min_by(|&a, &b| {
                    let d = |k: usize| (0..3).map(|c| (pos[k][c] - q[c]).powi(2)).sum::<f32>();
                    d(a).total_cmp(&d(b))
                })
                .filter(|&k| (0..3).all(|c| (pos[k][c] - q[c]).abs() < tol))
                .unwrap_or_else(|| panic!("{p:?} has no mirror image"))
        })
        .collect();
    let mut faces = std::collections::HashSet::new();
    let mut off = 0;
    for &n in &x.face_sizes {
        let mut f: Vec<u32> = x.indices[off..off + n as usize].to_vec();
        f.sort_unstable();
        faces.insert(f);
        off += n as usize;
    }
    for f in &faces {
        let mut m: Vec<u32> = f.iter().map(|&v| image[v as usize] as u32).collect();
        m.sort_unstable();
        assert!(faces.contains(&m), "face {f:?} has no mirror image");
    }
}

#[test]
fn output_is_deterministic() {
    let a = run("displaced_sphere.obj", 1500).unwrap();
    let b = run("displaced_sphere.obj", 1500).unwrap();
    assert_eq!(a.positions.len(), b.positions.len());
    assert!(a
        .positions
        .iter()
        .zip(&b.positions)
        .all(|(x, y)| x.to_bits() == y.to_bits()));
    assert_eq!(a.face_sizes, b.face_sizes);
    assert_eq!(a.indices, b.indices);
}

#[test]
fn open_and_non_manifold_inputs_are_rejected() {
    // Suzanne: open eye/mouth boundaries + one non-manifold edge
    let err = run("suzanne.obj", 500).unwrap_err();
    assert!(
        matches!(
            err,
            RetopoError::UnsupportedBoundary | RetopoError::NonManifoldInput
        ),
        "{err}"
    );
    // a single triangle is open
    let err = remesh(
        &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        &[0, 1, 2],
        &RemeshOptions::default(),
        None,
    )
    .unwrap_err();
    assert!(matches!(err, RetopoError::UnsupportedBoundary));
}

#[test]
fn absurd_density_fails_cleanly() {
    let (p, i) = load("cube.obj");
    let err = remesh(
        &p,
        &i,
        &RemeshOptions {
            target_faces: 4_000_000,
            max_working_vertices: 100_000,
            ..RemeshOptions::default()
        },
        None,
    )
    .unwrap_err();
    assert!(matches!(err, RetopoError::TooDense { .. }), "{err}");
}

#[test]
fn decimate_mode_still_available() {
    let (p, i) = load("icosphere.obj");
    let out = remesh(
        &p,
        &i,
        &RemeshOptions {
            mode: RemeshMode::Decimate,
            target_faces: 80,
            ..RemeshOptions::default()
        },
        None,
    )
    .unwrap();
    assert!(out.face_sizes.iter().all(|&n| n == 3));
    assert!(out.stats.output_faces <= 80);
}

/// Report table (run with `--nocapture --ignored` in release mode).
#[test]
#[ignore]
fn fixture_report() {
    println!(
        "| fixture | in verts | target | out faces | quads % | χ in→out | poles v3/v5/other | sym dev | time |"
    );
    println!("|---|---|---|---|---|---|---|---|---|");
    for (name, target) in [
        ("icosphere.obj", 320u32),
        ("uv_sphere.obj", 300),
        ("torus.obj", 300),
        ("cube.obj", 96),
        ("cylinder.obj", 300),
        ("displaced_sphere.obj", 1500),
        ("displaced_sphere.obj", 6000),
    ] {
        let (p, _) = load(name);
        match run(name, target) {
            Ok(o) => {
                let s = &o.stats;
                println!(
                    "| {name} | {} | {target} | {} | {:.1} | {}→{} | {}/{}/{} | {:.3} | {:.3}s |",
                    p.len() / 3,
                    s.output_faces,
                    100.0 * s.quads as f64 / s.output_faces as f64,
                    s.euler_input,
                    s.euler_output,
                    s.poles_valence3,
                    s.poles_valence5,
                    s.poles_other,
                    s.symmetry_deviation,
                    s.duration_seconds
                );
            }
            Err(e) => println!("| {name} | {} | {target} | ERROR: {e} |", p.len() / 3),
        }
    }
}

#[test]
fn adaptive_density_spends_faces_on_small_features() {
    // a big sphere and a small one (radius 0.3) side by side: uniform
    // density gives the small one ~area share (8%), adaptive more
    let (p, i) = load("icosphere.obj");
    let nv = (p.len() / 3) as u32;
    let mut pos = p.clone();
    pos.extend(
        p.chunks(3)
            .flat_map(|v| [v[0] * 0.3 + 3.0, v[1] * 0.3, v[2] * 0.3]),
    );
    let mut idx = i.clone();
    idx.extend(i.iter().map(|k| k + nv));
    let small_share = |adaptivity: f64| {
        let out = remesh(
            &pos,
            &idx,
            &RemeshOptions {
                target_faces: 800,
                adaptivity,
                ..RemeshOptions::default()
            },
            None,
        )
        .unwrap_or_else(|e| panic!("adaptivity {adaptivity}: {e}"));
        assert_eq!(check_closed_manifold(&out), 4);
        let mut off = 0usize;
        let mut small = 0usize;
        for &n in &out.face_sizes {
            let x: f32 = out.indices[off..off + n as usize]
                .iter()
                .map(|&v| out.positions[3 * v as usize])
                .sum::<f32>()
                / n as f32;
            small += (x > 2.0) as usize;
            off += n as usize;
        }
        small as f64 / out.face_sizes.len() as f64
    };
    let (uniform, adaptive) = (small_share(0.0), small_share(1.0));
    assert!(uniform < 0.13, "uniform share {uniform}");
    assert!(
        adaptive > uniform * 1.5,
        "adaptive {adaptive} vs uniform {uniform}"
    );
}
