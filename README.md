# OpenRetopo

**Field-aligned automatic retopology for Blender's characters — written in pure Rust.**

OpenRetopo rebuilds heavy sculpt/scanned meshes into clean, animation-ready
topology: quad-dominant output, edge flow that follows principal curvature,
symmetry-aware solving, and density weighting for deformation regions
(shoulders, elbows, knees, hips, face). Built for riggers, not just for
pretty renders.

> **Status — expect active development (M0: scaffold + native bridge).**
> What works today: native engine end-to-end (mesh welding, degenerate/
> duplicate face removal, quadric decimation with boundary & crease
> locking), installable Blender 4.2+/5.x extension, headless test suite.
> What lands next: cross-field solve → quad extraction (Instant Field-
> Aligned Meshes) → relaxation/snapping → symmetry & vertex-group density
> painting. See [Roadmap](#roadmap).

## Why yet another remesher

Commercial auto-retopo tools are either closed, pricey, or locked to one
host. OpenRetopo aims to be:

* **Native & fast** — the engine is a Rust `cdylib` called over a tiny C
  ABI (no Python-ABI coupling; one binary per platform works on Blender
  4.2 LTS and 5.x).
* **Rigging-aware** — X/Y/Z symmetry as a hard constraint (not a hope),
  per-vertex density weighting painted as a Blender vertex group, pole
  placement kept out of deformation-critical regions as far as local
  optimization allows.
* **Honest** — automatic retopology can't guarantee textbook deformation
  loops. OpenRetopo gives you the best-effort optimum plus the controls
  (symmetry, density, creases, boundary preservation) that actually get a
  rig shippable, and reports exactly what it did (quads %, poles, max
  valence, symmetry deviation).

## Install (end users)

1. Download `open_retopo-<version>.zip` from
   [Releases](https://github.com/Guillermode20/OpenRetopo/releases).
2. Blender → *Edit → Preferences → Get Extensions* → dropdown →
   *Install from Disk…* → pick the zip.
3. In the 3D View sidebar, open the **OpenRetopo** tab, select your
   character mesh, set **Target Faces**, hit **Retopologize**.

The zip bundles native libraries for Linux x86_64, Windows x86_64 and
macOS (Apple Silicon + Intel, Universal2). The original object is never
modified; the output is a new `*_retopo` object.

## Build from source

Requirements: Blender 4.2+ or 5.x, a stable Rust toolchain (1.85+), and
(on macOS) Xcode command line tools for `lipo`/`codesign`.

```bash
git clone https://github.com/Guillermode20/OpenRetopo.git
cd OpenRetopo
./tools/build_native.sh          # cargo build --release + copy into open_retopo/native/<platform>/
blender -b --factory-startup -noaudio --python tools/headless_test.py
```

To use the addon without packaging: *Install from Disk* on a zip produced
by `blender --command extension build --source-dir open_retopo --output-dir dist`.

### Running the tests

```bash
cargo test --workspace
blender -b --factory-startup -noaudio --python tools/headless_test.py
```

The Rust suite covers welding, quadric decimation and mesh invariants.
The headless Blender script covers addon registration, the native bridge,
and the operator on synthetic and real meshes. CI runs both on
ubuntu-24.04, windows-2022 and macos-14, plus headless Blender 4.2 and
5.2 jobs.

## How it works (architecture)

```
Blender mesh ──foreach_get──> numpy float32/uint32
            │                        │
            │                 ctypes C ABI (versioned structs)
            │                        ▼
            │              openretopo (Rust cdylib)
            │                        │
            │   weld ─► clean ─► (M1) field solve (M2)
            │                        ─► quad extraction (M3)
            │                        ─► relax/snap/poles (M4)
            │                        ─► symmetry + density (M5)
            │                        ▼
            └──────── new object (`*_retopo`), smooth-shaded
```

* `crates/retopo-core` — pure engine, no Blender dependencies, unit tested.
* `crates/retopo-ffi` — `cdylib` exposing `openretopo_*` C functions.
* `open_retopo/` — the Blender extension (`blender_manifest.toml`, modal
  operator with worker thread + Esc-cancel, sidebar panel).
* `tools/` — build script, fixture tooling, headless test.
* `.github/workflows/` — CI (fmt/clippy/test, native builds, Blender
  4.2/5.2 headless integration) and tagged releases.

## Roadmap

| Milestone | Scope | Status |
|---|---|---|
| M0 | Scaffold, C ABI bridge, weld/degenerate/duplicate cleanup, quadric decimation, installable extension, CI/release | ✅ done |
| M1 | Geometry kernel: half-edge mesh, welding stats, curvature & crease analysis, BVH surface projection | ▶ next |
| M2 | 4-RoSy cross-field + position field with boundary/crease constraints; Poincaré–Hopf validation; debug visualisation | |
| M3 | Iso-line tracing extraction → first field-aligned quad output (plane → sphere → genus-N) | |
| M4 | Relaxation, surface snapping, pole reduction, boundary handling, stats | |
| M5 | **v1.0** — X/Y/Z symmetry (half-solve + mirror stitch), vertex-group density weighting, crease toggle, presets, packaging | |

## Acknowledgements

* Wenzel Jakob, Marco Tarini, Daniele Panozzo, Olga Sorkine-Hornung —
  *Instant Field-Aligned Meshes* (SIGGRAPH Asia 2015), and the reference
  implementation `wjakob/instant-meshes` (BSD-3-Clause). OpenRetopo
  reimplements the pipeline in idiomatic Rust; nothing of their code is
  copied. See [NOTICE.md](NOTICE.md).
* The Blender Foundation and the Exoside team — for making retopology a
  first-class problem worth solving well.

## License

GPL-3.0-or-later. See [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md).
