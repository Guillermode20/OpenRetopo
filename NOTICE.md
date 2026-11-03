# OpenRetopo — Third-Party Notices

OpenRetopo as a whole is licensed under GPL-3.0-or-later; see [LICENSE](LICENSE).

## Algorithmic reference

*Instant Field-Aligned Meshes* — Wenzel Jakob, Marco Tarini, Daniele
Panozzo, Olga Sorkine-Hornung (SIGGRAPH Asia 2015). The paper is used as
the algorithmic reference for the cross-field/position-field retopology
pipeline. The reference implementation, `wjakob/instant-meshes`, is
BSD-3-Clause (with an additional enhancement-grant clause). OpenRetopo
re-implements the algorithm in idiomatic Rust; **no code is copied from
instant-meshes**. BSD-3 copyright:

> Copyright (c) 2015 Wenzel Jakob, Daniele Panozzo, Marco Tarini, and
> Olga Sorkine-Hornung. All rights reserved.

## Rust dependencies

All Rust dependencies are used unmodified under their own licenses
(MIT/Apache-2.0 unless noted): nalgebra (MIT), rayon (MIT/Apache-2.0),
thiserror (MIT/Apache-2.0), and their transitive dependencies. See
`Cargo.lock` for the full list. The `approx` crate (MIT/Apache-2.0) is a
development-only dependency used by tests.

## Blender

OpenRetopo interoperates with Blender, which is GPL-3.0-or-later. The
addon uses Blender's Python API (`bpy`) and is loaded by Blender; it is
therefore also distributed under GPL-3.0-or-later.
