//! Output quality report: surface fidelity and topology regularity.
//!
//! cargo run --release -p openretopo-core --example quality -- input.obj output.obj
//!
//! Distances are in units of the output's mean edge length `h`:
//! `in→out` (input samples to the output surface: missing features, notches)
//! and `out→in` (output vertices to the input: vertices off the surface).

use std::collections::HashMap;

use nalgebra::Vector3;

type V3 = Vector3<f64>;

fn load(path: &str) -> (Vec<V3>, Vec<Vec<u32>>) {
    let text = std::fs::read_to_string(path).expect("read obj");
    let (mut p, mut f) = (Vec::new(), Vec::new());
    for line in text.lines() {
        let mut it = line.split_whitespace();
        match it.next() {
            Some("v") => {
                let c: Vec<f64> = it.take(3).map(|s| s.parse().unwrap()).collect();
                p.push(V3::new(c[0], c[1], c[2]));
            }
            Some("f") => f.push(
                it.map(|s| s.split('/').next().unwrap().parse::<u32>().unwrap() - 1)
                    .collect(),
            ),
            _ => {}
        }
    }
    (p, f)
}

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

struct Surf {
    tris: Vec<[V3; 3]>,
    cell: f64,
    grid: HashMap<(i64, i64, i64), Vec<u32>>,
}

impl Surf {
    fn new(p: &[V3], faces: &[Vec<u32>], cell: f64) -> Surf {
        let mut tris = Vec::new();
        for f in faces {
            for i in 1..f.len() - 1 {
                tris.push([p[f[0] as usize], p[f[i] as usize], p[f[i + 1] as usize]]);
            }
        }
        let k = |x: f64| (x / cell).floor() as i64;
        let mut grid: HashMap<_, Vec<u32>> = HashMap::new();
        for (ti, t) in tris.iter().enumerate() {
            let lo = t[0].inf(&t[1]).inf(&t[2]);
            let hi = t[0].sup(&t[1]).sup(&t[2]);
            for x in k(lo.x)..=k(hi.x) {
                for y in k(lo.y)..=k(hi.y) {
                    for z in k(lo.z)..=k(hi.z) {
                        grid.entry((x, y, z)).or_default().push(ti as u32);
                    }
                }
            }
        }
        Surf { tris, cell, grid }
    }
    fn dist(&self, p: V3) -> f64 {
        let k = |x: f64| (x / self.cell).floor() as i64;
        let mut best = f64::INFINITY;
        for r in 1..6 {
            for x in k(p.x) - r..=k(p.x) + r {
                for y in k(p.y) - r..=k(p.y) + r {
                    for z in k(p.z) - r..=k(p.z) + r {
                        if let Some(l) = self.grid.get(&(x, y, z)) {
                            for &ti in l {
                                let t = self.tris[ti as usize];
                                best =
                                    best.min((closest_on_triangle(p, t[0], t[1], t[2]) - p).norm());
                            }
                        }
                    }
                }
            }
            if best <= (r as f64) * self.cell {
                break;
            }
        }
        best
    }
}

fn pct(v: &mut [f64], q: f64) -> f64 {
    v.sort_by(f64::total_cmp);
    v[((v.len() - 1) as f64 * q) as usize]
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (ip, ifc) = load(&a[0]);
    let (op, ofc) = load(&a[1]);
    let mut h = 0.0;
    let mut ne = 0;
    for f in &ofc {
        for k in 0..f.len() {
            h += (op[f[k] as usize] - op[f[(k + 1) % f.len()] as usize]).norm();
            ne += 1;
        }
    }
    h /= ne as f64;
    let osurf = Surf::new(&op, &ofc, h);
    let isurf = Surf::new(&ip, &ifc, h);
    // input samples: vertices and triangle centroids
    let mut samples: Vec<V3> = ip.clone();
    for f in &ifc {
        samples.push(f.iter().map(|&i| ip[i as usize]).sum::<V3>() / f.len() as f64);
    }
    let mut d_in: Vec<(f64, V3)> = samples.iter().map(|&s| (osurf.dist(s) / h, s)).collect();
    let mut d_out: Vec<f64> = op.iter().map(|&s| isurf.dist(s) / h).collect();
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let mut din: Vec<f64> = d_in.iter().map(|x| x.0).collect();
    println!(
        "h {h:.4}  in→out mean {:.3} p99 {:.3} max {:.3}  out→in mean {:.3} p99 {:.3} max {:.3}",
        mean(&din),
        pct(&mut din, 0.99),
        pct(&mut din, 1.0),
        mean(&d_out),
        pct(&mut d_out, 0.99),
        pct(&mut d_out, 1.0)
    );
    // worst spots, de-duplicated by distance
    d_in.sort_by(|x, y| y.0.total_cmp(&x.0));
    let mut spots: Vec<(f64, V3)> = Vec::new();
    for (d, p) in d_in {
        if d < 0.25 || spots.len() >= 8 {
            break;
        }
        if spots.iter().all(|(_, q)| (q - p).norm() > 3.0 * h) {
            spots.push((d, p));
        }
    }
    for (d, p) in &spots {
        println!("  gap {d:.2}h at ({:.3}, {:.3}, {:.3})", p.x, p.y, p.z);
    }
    // topology
    let mut val: HashMap<u32, u32> = HashMap::new();
    let mut sizes: HashMap<usize, usize> = HashMap::new();
    for f in &ofc {
        *sizes.entry(f.len()).or_default() += 1;
        for &v in f {
            *val.entry(v).or_default() += 1;
        }
    }
    let mut vh: Vec<(u32, usize)> = Vec::new();
    for &k in val.values() {
        match vh.iter_mut().find(|(x, _)| *x == k) {
            Some(e) => e.1 += 1,
            None => vh.push((k, 1)),
        }
    }
    vh.sort();
    let mut edge_faces: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    for (fi, f) in ofc.iter().enumerate() {
        for k in 0..f.len() {
            let (x, y) = (f[k], f[(k + 1) % f.len()]);
            edge_faces.entry((x.min(y), x.max(y))).or_default().push(fi);
        }
    }
    let tri_tri = ofc
        .iter()
        .enumerate()
        .filter(|(fi, f)| {
            f.len() == 3
                && (0..3).any(|k| {
                    let (x, y) = (f[k], f[(k + 1) % 3]);
                    edge_faces[&(x.min(y), x.max(y))]
                        .iter()
                        .any(|&g| g != *fi && ofc[g].len() == 3)
                })
        })
        .count();
    // quad shape
    let mut bad = 0;
    let mut nq = 0;
    for f in ofc.iter().filter(|f| f.len() == 4) {
        nq += 1;
        let mut worst = 0.0f64;
        for k in 0..4 {
            let p = op[f[k] as usize];
            let e0 = op[f[(k + 3) % 4] as usize] - p;
            let e1 = op[f[(k + 1) % 4] as usize] - p;
            let ang = (e0.dot(&e1) / (e0.norm() * e1.norm()))
                .clamp(-1.0, 1.0)
                .acos();
            worst = worst.max((ang.to_degrees() - 90.0).abs());
        }
        if worst > 40.0 {
            bad += 1;
        }
    }
    let mut sz: Vec<_> = sizes.into_iter().collect();
    sz.sort();
    println!(
        "faces {} sizes {:?}  tris touching tris {}  valence {:?}  quads >40° off {:.1}%",
        ofc.len(),
        sz,
        tri_tri,
        vh,
        100.0 * bad as f64 / nq.max(1) as f64
    );
}
