# OpenRetopo — Third-Party Notices

OpenRetopo as a whole is licensed under GPL-3.0-or-later; see [LICENSE](LICENSE).

## Instant Meshes (BSD-3-Clause) — ported code

*Instant Field-Aligned Meshes* — Wenzel Jakob, Marco Tarini, Daniele
Panozzo, Olga Sorkine-Hornung (SIGGRAPH Asia 2015) is the algorithmic
reference for the whole field-aligned pipeline. Parts of OpenRetopo are
**translations into Rust of the reference implementation**
[`wjakob/instant-meshes`](https://github.com/wjakob/instant-meshes) and
are therefore derivative works of it, distributed under its BSD-3-Clause
license (reproduced below) in addition to GPL-3.0-or-later:

| OpenRetopo file | Ported from (instant-meshes) | Nature |
|---|---|---|
| `crates/retopo-core/src/extract.rs` | `src/extract.cpp` (`extract_graph`, `extract_faces` Steps 1–8), `src/field.cpp` (`compat_orientation_extrinsic_4`, `compat_position_extrinsic_index_4`, `position_floor_index_4`) | Close port, step by step. Own additions: unlimited-degree hole filling with chord checks, closed-manifold/Euler validation, deterministic tie-breaking. |
| `crates/retopo-core/src/field.rs` | `src/field.cpp` (`rotate_vector_into_plane`, `compat_orientation_intrinsic_4`, `middle_point`, `position_round_4`, `compat_position_intrinsic_4`, the orientation/position update loops) | Port of the update rules. Own changes: sequential deterministic sweeps, hash-based initialisation, antiparallel-normal handling, fail-hard divergence detection. |
| `crates/retopo-core/src/hierarchy.rs` | `src/hierarchy.cpp` (graph coarsening, `toUpper` prolongation, fine→coarse restriction) | Re-implementation of the same scheme. |
| `crates/retopo-core/src/subdivide.rs` | `src/subdivide.cpp` (longest-edge bisection before solving) | Re-implementation of the same rule (`min(scale/2, 2·avg edge)`), single-threaded and deterministic. |

The BSD-3 license of instant-meshes:

```
Copyright (c) 2015 Wenzel Jakob, Daniele Panozzo, Marco Tarini,
and Olga Sorkine-Hornung. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

You are under no obligation whatsoever to provide any bug fixes, patches, or
upgrades to the features, functionality or performance of the source code
("Enhancements") to anyone; however, if you choose to make your Enhancements
available either publicly, or directly to the authors of this software, without
imposing a separate written license agreement for such Enhancements, then you
hereby grant the following license: a non-exclusive, royalty-free perpetual
license to install, use, modify, prepare derivative works, incorporate into
other computer software, distribute, and sublicense such enhancements or
derivative works thereof, in binary and source code form.
```

The "Enhancements" clause is a grant back to the instant-meshes authors
for improvements we publish; it places no restriction on OpenRetopo's
users and is compatible with GPL-3.0. Combining BSD-3 code into a
GPL-3.0 work is permitted; this notice satisfies the BSD retention and
binary-reproduction conditions (it ships inside the extension zip).

## Rust dependencies

All Rust dependencies are used unmodified under their own licenses
(MIT/Apache-2.0 unless noted): nalgebra (Apache-2.0), rayon (MIT/Apache-2.0),
thiserror (MIT/Apache-2.0), and their transitive dependencies. See
`Cargo.lock` for the full list. The `approx` crate (MIT/Apache-2.0) is a
development-only dependency used by tests.

## Blender

OpenRetopo interoperates with Blender, which is GPL-3.0-or-later. The
addon uses Blender's Python API (`bpy`) and is loaded by Blender; it is
therefore also distributed under GPL-3.0-or-later.
