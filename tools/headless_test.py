"""Headless end-to-end check for the OpenRetopo addon.

Run with Blender (not the system python):

    blender -b --factory-startup -noaudio --python tools/headless_test.py

Exits 0 on success, 1 on failure. Asserts:
  * addon registers and the native library loads (ABI v2);
  * field-aligned remesh of closed meshes (icosphere, torus) creates a new
    smooth-shaded `<name>_retopo` object that is quad-dominant, closed,
    manifold, has the input's Euler characteristic, and leaves the original
    untouched;
  * open input (Suzanne) is rejected in field mode with no object created;
  * legacy decimate mode still welds/decimates open meshes.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import bpy  # noqa: E402

FAILURES = []


def check(condition, label):
    status = "PASS" if condition else "FAIL"
    print(f"[OpenRetopo][{status}] {label}")
    if not condition:
        FAILURES.append(label)


def reset_scene():
    # Do NOT use read_factory_settings here: it reloads factory preferences
    # and would unregister addons we enabled for this session.
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in list(bpy.data.meshes):
        bpy.data.meshes.remove(block)


def make_object(name, verts, faces):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    activate(obj)
    return obj


def activate(obj):
    for o in bpy.context.selected_objects:
        o.select_set(False)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)


def snapshot(obj):
    me = obj.data
    return (len(me.vertices), len(me.polygons), tuple(round(c, 6) for v in me.vertices[:50] for c in v.co))


def topology(obj):
    """(euler, boundary_edges, nonmanifold_edges) of a mesh object."""
    me = obj.data
    edge_faces = {}
    for p in me.polygons:
        for k in p.edge_keys:
            edge_faces[k] = edge_faces.get(k, 0) + 1
    boundary = sum(1 for c in edge_faces.values() if c == 1)
    nonmanifold = sum(1 for c in edge_faces.values() if c > 2)
    return len(me.vertices) - len(me.edges) + len(me.polygons), boundary, nonmanifold


def field_case(label, obj, target, expected_chi, **kwargs):
    import open_retopo

    before = snapshot(obj)
    src_name = obj.name
    result = bpy.ops.open_retopo.remesh("EXEC_DEFAULT", target_faces=target, **kwargs)
    check(result == {"FINISHED"}, f"{label}: operator finished ({result})")
    out = bpy.data.objects.get(src_name + "_retopo")
    stats = open_retopo.OPENRETOPO_LAST_STATS
    check(out is not None and stats is not None, f"{label}: {src_name}_retopo created")
    if out is None or stats is None:
        return
    me = out.data
    nq = sum(1 for p in me.polygons if len(p.vertices) == 4)
    nt = sum(1 for p in me.polygons if len(p.vertices) == 3)
    other = len(me.polygons) - nq - nt
    chi, boundary, nonmanifold = topology(out)
    print(f"[OpenRetopo] {label}: {len(me.polygons)} faces ({nq} quads, {nt} tris), chi={chi}, "
          f"poles v3/v5/other={stats['poles_valence3']}/{stats['poles_valence5']}/{stats['poles_other']}, "
          f"{stats['duration_seconds']:.3f}s")
    check(other == 0, f"{label}: only tris/quads ({other} other)")
    check(nq / max(1, len(me.polygons)) > 0.85, f"{label}: quad-dominant ({nq}/{len(me.polygons)})")
    check(boundary == 0 and nonmanifold == 0, f"{label}: closed manifold (boundary {boundary}, non-manifold {nonmanifold})")
    check(chi == expected_chi and stats["euler_output"] == expected_chi, f"{label}: Euler {chi} == {expected_chi}")
    check(0.6 < len(me.polygons) / target < 1.6, f"{label}: face count {len(me.polygons)} near target {target}")
    check(all(p.use_smooth for p in me.polygons), f"{label}: smooth shaded")
    check(not me.validate(verbose=False), f"{label}: passes mesh.validate()")
    drift = max(abs(a - b) for ra, rb in zip(out.matrix_world, obj.matrix_world) for a, b in zip(ra, rb))
    check(drift < 1e-5, f"{label}: transform copied (max drift {drift:.1e})")
    original = bpy.data.objects.get(src_name)
    check(original is not None and snapshot(original) == before, f"{label}: original untouched")
    return out, stats


def run():
    import open_retopo

    open_retopo.register()
    check(open_retopo._load_library().openretopo_abi_version() == open_retopo.EXPECTED_ABI,
          "addon registered and native library loaded (ABI match)")

    # ---- field-aligned: closed sphere -------------------------------------
    reset_scene()
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=4, radius=1.0, location=(0.5, 0.0, 0.0))
    ico = bpy.context.active_object
    field_case("icosphere", ico, 400, 2)

    # ---- field-aligned: genus-1 torus -------------------------------------
    reset_scene()
    bpy.ops.mesh.primitive_torus_add(major_radius=1.5, minor_radius=0.5, major_segments=48, minor_segments=24)
    field_case("torus", bpy.context.active_object, 600, 0)

    # ---- field-aligned: rotated, non-uniformly scaled, mirror-symmetric ----
    reset_scene()
    bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=1.0)
    egg = bpy.context.active_object
    for v in egg.data.vertices:  # break Y/Z symmetry, keep local X symmetry
        v.co.z += 0.35 * v.co.y + 0.25 * v.co.y * v.co.y
        v.co.y += 0.3 * v.co.z * v.co.z
    egg.rotation_euler = (0.0, 1.5708, 0.0)
    egg.scale = (1.0, 0.6, 1.0)
    bpy.context.view_layer.update()
    out, stats = field_case("scaled egg (auto symmetry)", egg, 800, 2, symmetry="AUTO")
    if out is not None:
        check(stats["symmetry_axis"] == 1, f"scaled egg: mirrored on local X (axis {stats['symmetry_axis']})")
        check(stats["symmetry_deviation"] < 0.1, f"scaled egg: mirror deviation {stats['symmetry_deviation']:.3f}")
        # quads should be roughly square in world space despite scale (1, 0.6, 1)
        mw = out.matrix_world
        ratios = []
        for p in out.data.polygons:
            if len(p.vertices) == 4:
                c = [mw @ out.data.vertices[i].co for i in p.vertices]
                a, b = (c[1] - c[0]).length, (c[2] - c[1]).length
                ratios.append(max(a, b) / max(min(a, b), 1e-9))
        ratios.sort()
        median = ratios[len(ratios) // 2] if ratios else 99
        check(median < 1.35, f"scaled egg: world-space quads square-ish (median aspect {median:.2f})")

    # ---- field-aligned: open input is rejected cleanly --------------------
    reset_scene()
    bpy.ops.mesh.primitive_monkey_add(size=2.0)
    monkey = bpy.context.active_object
    before = snapshot(monkey)
    open_retopo.OPENRETOPO_LAST_STATS = None
    try:
        result = bpy.ops.open_retopo.remesh("EXEC_DEFAULT", target_faces=500)
        message = ""
    except RuntimeError as exc:  # operators reporting ERROR raise when called from Python
        result, message = {"CANCELLED"}, str(exc)
    check(result == {"CANCELLED"} and "boundary" in message,
          f"suzanne (open) rejected in field mode ({result}: {message.strip()})")
    check(not any(o.name.endswith("_retopo") for o in bpy.data.objects), "suzanne: no output object on failure")
    check(not any(m.name.endswith("_retopo") for m in bpy.data.meshes), "suzanne: no orphan mesh on failure")
    check(snapshot(monkey) == before, "suzanne: original untouched on failure")

    # ---- legacy decimate: Suzanne ------------------------------------------
    input_faces = len(monkey.data.polygons)
    result = bpy.ops.open_retopo.remesh("EXEC_DEFAULT", target_faces=300, mode="DECIMATE")
    stats = open_retopo.OPENRETOPO_LAST_STATS
    out = bpy.data.objects.get(monkey.name + "_retopo")
    check(result == {"FINISHED"} and out is not None, "decimate: Suzanne_retopo created")
    if stats:
        check(stats["output_faces"] < input_faces and stats["output_faces"] <= 400,
              f"decimate: faces reduced {input_faces} -> {stats['output_faces']}")

    # ---- legacy decimate: weld on a synthetic mesh -------------------------
    reset_scene()
    verts = [
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),  # exact duplicates -> weld
    ]
    make_object("WeldTest", verts, [[0, 1, 2], [4, 5, 3], [0, 1, 2], [0, 1, 1]])
    bpy.ops.open_retopo.remesh("EXEC_DEFAULT", target_faces=100, mode="DECIMATE")
    stats = open_retopo.OPENRETOPO_LAST_STATS
    check(stats is not None, "weld run produced stats")
    if stats:
        check(stats["welded_vertices_removed"] == 2, f"welded {stats['welded_vertices_removed']} (expected 2)")
        check(stats["degenerate_faces_removed"] >= 1, f"degenerate dropped {stats['degenerate_faces_removed']}")
        check(stats["duplicate_faces_removed"] >= 1, f"duplicates dropped {stats['duplicate_faces_removed']}")
        check(bpy.data.objects.get("WeldTest") is not None, "weld: original object preserved")


def main():
    try:
        run()
    except Exception as exc:  # surface the traceback and fail hard
        import traceback

        traceback.print_exc()
        FAILURES.append(f"unhandled exception: {exc}")

    print()
    if FAILURES:
        print(f"[OpenRetopo][RESULT] FAIL ({len(FAILURES)} failure(s))")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("[OpenRetopo][RESULT] PASS")
    sys.exit(0)


main()
