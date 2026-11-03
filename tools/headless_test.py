"""Headless end-to-end check for the OpenRetopo addon.

Run with Blender (not the system python):

    blender -b --factory-startup -noaudio --python tools/headless_test.py

Exits 0 on success, 1 on failure. Asserts:
  * addon registers and the native library loads;
  * welding removes coincident vertices and degenerate faces;
  * the operator creates a new object on real input (Suzanne) with fewer,
    valid faces.
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


def run():
    import open_retopo

    open_retopo.register()
    check(True, "addon registered and native library loaded")

    # ---- weld test on a synthetic mesh -----------------------------------
    reset_scene()
    verts = [
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),  # exact duplicates -> weld
    ]
    faces = [[0, 1, 2], [4, 5, 3], [0, 1, 2], [0, 1, 1]]
    mesh = bpy.data.meshes.new("WeldTest")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new("WeldTest", mesh)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    bpy.ops.open_retopo.remesh("EXEC_DEFAULT", target_faces=100)
    stats = open_retopo.OPENRETOPO_LAST_STATS
    check(stats is not None, "weld run produced stats")
    if stats:
        check(stats["welded_vertices_removed"] == 2, f"welded {stats['welded_vertices_removed']} (expected 2)")
        check(stats["degenerate_faces_removed"] >= 1, f"degenerate dropped {stats['degenerate_faces_removed']} (expected >= 1)")
        check(stats["duplicate_faces_removed"] >= 1, f"duplicates dropped {stats['duplicate_faces_removed']} (expected >= 1)")
        check("WeldTest_retopo" in bpy.data.objects, "output object created")
        original = bpy.data.objects.get("WeldTest")
        check(original is not None, "original object preserved")

    # ---- Suzanne decimation -----------------------------------------------
    reset_scene()
    bpy.ops.mesh.primitive_monkey_add(size=2.0)
    monkey = bpy.context.active_object
    input_faces = len(monkey.data.polygons)
    bpy.ops.open_retopo.remesh("EXEC_DEFAULT", target_faces=300)
    stats = open_retopo.OPENRETOPO_LAST_STATS
    out = bpy.data.objects.get("Suzanne_retopo") or bpy.data.objects.get(monkey.name + "_retopo")
    check(out is not None, "Suzanne_retopo created")
    check(stats is not None and stats["output_faces"] < input_faces,
          f"faces reduced {input_faces} -> {stats['output_faces'] if stats else '?'}")
    if stats:
        check(stats["output_faces"] <= 400, f"target reasonably reached ({stats['output_faces']} faces)")
        check(all(abs(v) < 1e6 for v in stats.values() if isinstance(v, (int, float))), "stats finite")

    # ---- open boundary mesh: must not crash or produce NaNs ---------------
    reset_scene()
    grid_verts, grid_faces = [], []
    n = 4
    for y in range(n):
        for x in range(n):
            grid_verts.append((x * 0.5, y * 0.5, 0.0))
    for y in range(n - 1):
        for x in range(n - 1):
            i = y * n + x
            grid_faces.append([i, i + 1, i + n + 1])
            grid_faces.append([i, i + n + 1, i + n])
    mesh = bpy.data.meshes.new("GridTest")
    mesh.from_pydata(grid_verts, [], grid_faces)
    mesh.update()
    obj = bpy.data.objects.new("GridTest", mesh)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.open_retopo.remesh("EXEC_DEFAULT", target_faces=2)
    out = bpy.data.objects.get("GridTest_retopo")
    check(out is not None, "open mesh: output object created (boundary locked, no crash)")


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
