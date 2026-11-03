"""Export test fixture meshes as triangulated .obj files into tests/fixtures/.

Run:  blender -b --factory-startup -noaudio --python tools/export_fixtures.py
"""

import os
import sys

import bpy

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "fixtures")
os.makedirs(OUT, exist_ok=True)


def reset_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for coll in (bpy.data.meshes, bpy.data.curves, bpy.data.materials):
        for block in list(coll):
            coll.remove(block)


def export(name, filepath):
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = bpy.data.objects[name]
    bpy.data.objects[name].select_set(True)
    bpy.ops.wm.obj_export(
        filepath=filepath,
        export_selected_objects=True,
        forward_axis="Y",
        up_axis="Z",
        global_scale=1.0,
        apply_modifiers=True,
        export_triangulated_mesh=True,
        export_materials=False,
        export_normals=False,
        export_uv=False,
    )
    reset_scene()


# 1) UV sphere — orientation field should align with meridians/parallels
reset_scene()
bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=24)
export("Sphere", os.path.join(OUT, "uv_sphere.obj"))

# 2) Icosphere — classic singularity test: 12 positive singularities
reset_scene()
bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=4, radius=1.0)
export("Icosphere", os.path.join(OUT, "icosphere.obj"))

# 3) Suzanne — irregular real-world topology
reset_scene()
bpy.ops.mesh.primitive_monkey_add(size=2.0)
export("Suzanne", os.path.join(OUT, "suzanne.obj"))

# 4) Torus — genus 1; Euler characteristic 0
reset_scene()
bpy.ops.mesh.primitive_torus_add(major_radius=1.5, minor_radius=0.5, major_segments=32, minor_segments=16)
export("Torus", os.path.join(OUT, "torus.obj"))

# 5) Cube (sharp edges) — crease behaviour
reset_scene()
bpy.ops.mesh.primitive_cube_add(size=2.0)
export("Cube", os.path.join(OUT, "cube.obj"))

print("[fixtures] exported to", OUT)
for f in sorted(os.listdir(OUT)):
    print(" ", f)
