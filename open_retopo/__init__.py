"""OpenRetopo — field-aligned automatic retopology for Blender.

The heavy lifting happens in a native Rust library (see crates/). The addon
is a thin shell that:
  1. extracts triangle data from the active mesh (fast foreach_get path),
  2. hands it to the native engine on a worker thread (or synchronously when
     `sync=True` is passed, used by headless tests),
  3. rebuilds the result (quad-dominant polygons) as a new `<name>_retopo`
     object, smooth shaded.

The original object is never modified. If the engine fails, or the result
does not pass Blender's own mesh validation, nothing is created.
"""

import ctypes
import os
import queue
import threading

import numpy as np

import bpy
from bpy.props import BoolProperty, EnumProperty, IntProperty
from bpy.types import Operator, Panel, PropertyGroup

EXT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# native library loading
# ---------------------------------------------------------------------------

_LIB_SPECS = {
    ("linux", "x86_64"): ("linux_x86_64", "libopenretopo.so"),
    ("windows", "AMD64"): ("win_x86_64", "openretopo.dll"),
    ("darwin", "x86_64"): ("macos_universal2", "libopenretopo.dylib"),
    ("darwin", "arm64"): ("macos_universal2", "libopenretopo.dylib"),
}

EXPECTED_ABI = 3
FLAG_DECIMATE_ONLY = 1
FLAG_NO_CURVATURE = 2
SYMMETRY_CODES = {"NONE": 0, "X": 1, "Y": 2, "Z": 3, "AUTO": 4}


class OpenRetopoParams(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("target_faces", ctypes.c_uint32),
        ("weld_epsilon", ctypes.c_float),
        ("lock_crease_degrees", ctypes.c_float),
        ("symmetry_axis", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("reserved", ctypes.c_uint64 * 2),
    ]


class OpenRetopoStats(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("_pad", ctypes.c_uint32),
        ("input_vertices", ctypes.c_uint64),
        ("input_faces", ctypes.c_uint64),
        ("welded_vertices_removed", ctypes.c_uint64),
        ("degenerate_faces_removed", ctypes.c_uint64),
        ("duplicate_faces_removed", ctypes.c_uint64),
        ("decimated_faces_removed", ctypes.c_uint64),
        ("collapses", ctypes.c_uint64),
        ("target_reached", ctypes.c_uint64),
        ("output_vertices", ctypes.c_uint64),
        ("output_faces", ctypes.c_uint64),
        ("quads", ctypes.c_uint64),
        ("triangles", ctypes.c_uint64),
        ("ngons", ctypes.c_uint64),
        ("dropped_cycles", ctypes.c_uint64),
        ("euler_input", ctypes.c_int64),
        ("euler_output", ctypes.c_int64),
        ("poles_valence3", ctypes.c_uint64),
        ("poles_valence5", ctypes.c_uint64),
        ("poles_other", ctypes.c_uint64),
        ("subdivision_vertices_added", ctypes.c_uint64),
        ("symmetry_deviation", ctypes.c_double),
        ("edge_length", ctypes.c_double),
        ("duration_seconds", ctypes.c_double),
        ("extraction_attempts", ctypes.c_uint64),
        ("symmetry_axis", ctypes.c_uint64),
        ("reserved", ctypes.c_uint64 * 2),
    ]

_STAT_FIELDS = [name for name, _ in OpenRetopoStats._fields_ if not name.startswith("_") and name not in ("struct_size", "reserved")]


class _NativeShared(ctypes.Structure):
    """Written by the worker thread; polled by the modal operator."""

    _fields_ = [
        ("progress", ctypes.c_float),
        ("cancel", ctypes.c_int32),
    ]


class NativeRetopologyError(RuntimeError):
    pass


_lib = None

# Set to a dict of the most recent native run's stats (or None).
OPENRETOPO_LAST_STATS = None


def _private_copy(path):
    """Copy the library to a content-addressed temp path and return it.

    The OS caches loaded libraries by path for the life of the process, so
    after an in-place extension update `CDLL(path)` would silently hand back
    the *old* engine. Loading a copy named after the file's hash makes an
    update take effect as soon as the extension is re-enabled. Falls back to
    the original path if the copy cannot be made.
    """
    import hashlib
    import shutil
    import tempfile

    try:
        with open(path, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()[:16]
        folder = os.path.join(tempfile.gettempdir(), f"openretopo-{digest}")
        dest = os.path.join(folder, os.path.basename(path))
        if not os.path.exists(dest):
            os.makedirs(folder, exist_ok=True)
            tmp = dest + f".{os.getpid()}.tmp"
            shutil.copy2(path, tmp)
            os.replace(tmp, dest)
        return dest
    except OSError:
        return path


def _load_library():
    global _lib
    if _lib is not None:
        return _lib
    import platform

    key = (platform.system().lower(), platform.machine())
    spec = _LIB_SPECS.get(key)
    if spec is None:
        raise NativeRetopologyError(f"OpenRetopo: unsupported platform {key[0]}/{key[1]}")
    folder, filename = spec
    path = os.path.join(EXT_DIR, "native", folder, filename)
    if not os.path.exists(path):
        raise NativeRetopologyError(
            f"OpenRetopo: native library not found at {path}.\n"
            "Build it with `cargo build --release -p openretopo-ffi` and copy "
            "the result into the addon's native/ folder, or install a release "
            "build from the releases page."
        )
    lib = ctypes.CDLL(_private_copy(path))
    lib.openretopo_abi_version.restype = ctypes.c_uint32
    lib.openretopo_abi_version.argtypes = []
    found = lib.openretopo_abi_version()
    if found != EXPECTED_ABI:
        raise NativeRetopologyError(
            f"OpenRetopo: native library ABI mismatch (addon expects {EXPECTED_ABI}, library is {found}). "
            "Restart Blender after updating the extension; if that does not help, reinstall it."
        )
    pf, pi = ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint32)
    lib.openretopo_remesh.restype = ctypes.c_int32
    lib.openretopo_remesh.argtypes = [
        pf,          # positions
        ctypes.c_uint64,
        pi,          # indices
        ctypes.c_uint64,
        ctypes.POINTER(OpenRetopoParams),
        pf,          # progress (nullable)
        ctypes.POINTER(ctypes.c_int32),  # cancel (nullable)
        ctypes.POINTER(pf),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(pi),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(pi),          # face sizes
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(OpenRetopoStats),
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_uint64,
    ]
    lib.openretopo_free_positions.restype = None
    lib.openretopo_free_positions.argtypes = [pf, ctypes.c_uint64]
    lib.openretopo_free_indices.restype = None
    lib.openretopo_free_indices.argtypes = [pi, ctypes.c_uint64]
    _lib = lib
    return lib


# ---------------------------------------------------------------------------
# mesh extraction helpers
# ---------------------------------------------------------------------------

def _object_scale(obj):
    """Per-axis world scale of the object (absolute values, never zero)."""
    sc = obj.matrix_world.to_scale()
    return np.array([max(abs(sc.x), 1e-12), max(abs(sc.y), 1e-12), max(abs(sc.z), 1e-12)], dtype=np.float64)


def _extract_triangles(mesh, scale=None):
    """Return (positions f32[n*3], indices u32[m*3]) from an evaluated mesh.

    `scale` (per-axis) is applied so the engine sees the object's true
    proportions: a non-uniformly scaled object would otherwise get quads
    that look stretched in the viewport. Axes stay local, so symmetry
    planes still match the object's own X/Y/Z.
    """
    mesh.calc_loop_triangles()

    vcount = len(mesh.vertices)
    positions = np.empty(vcount * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", positions)
    if scale is not None:
        positions = (positions.reshape(-1, 3) * scale).astype(np.float32).reshape(-1)

    tcount = len(mesh.loop_triangles)
    indices = np.empty(tcount * 3, dtype=np.int32)
    mesh.loop_triangles.foreach_get("vertices", indices)
    indices = indices.astype(np.uint32)  # bit-identical; indices are non-negative
    return positions, indices


def _native_remesh(positions, indices, target_faces, progress=None, decimate_only=False,
                   symmetry="NONE", curvature=True):
    """Call the engine.

    Returns (positions f32[:,3], face_sizes u32[F], indices u32[sum(sizes)], stats dict).
    """
    lib = _load_library()
    pf, pi = ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint32)

    params = OpenRetopoParams()
    params.struct_size = ctypes.sizeof(OpenRetopoParams)
    params.target_faces = int(target_faces)
    params.weld_epsilon = 0.0  # auto
    params.lock_crease_degrees = 180.0  # disabled for now (M1+ will expose)
    params.symmetry_axis = SYMMETRY_CODES[symmetry]
    params.flags = (FLAG_DECIMATE_ONLY if decimate_only else 0) | (0 if curvature else FLAG_NO_CURVATURE)
    params.reserved = (ctypes.c_uint64 * 2)(0, 0)

    out_pos = pf()
    out_idx = pi()
    out_pos_len = ctypes.c_uint64(0)
    out_idx_len = ctypes.c_uint64(0)
    out_fs = pi()
    out_fs_len = ctypes.c_uint64(0)
    stats = OpenRetopoStats()
    stats.struct_size = ctypes.sizeof(OpenRetopoStats)
    err_buf = ctypes.create_string_buffer(512)

    progress_arg = None
    cancel_arg = None
    if isinstance(progress, _NativeShared):
        progress_arg = ctypes.cast(ctypes.byref(progress), pf)
        cancel_arg = ctypes.cast(ctypes.byref(progress, 4), ctypes.POINTER(ctypes.c_int32))

    rc = lib.openretopo_remesh(
        positions.ctypes.data_as(pf),
        ctypes.c_uint64(positions.size),
        indices.ctypes.data_as(pi),
        ctypes.c_uint64(indices.size),
        ctypes.byref(params),
        progress_arg,
        cancel_arg,
        ctypes.byref(out_pos),
        ctypes.byref(out_pos_len),
        ctypes.byref(out_idx),
        ctypes.byref(out_idx_len),
        ctypes.byref(out_fs),
        ctypes.byref(out_fs_len),
        ctypes.byref(stats),
        err_buf,
        ctypes.c_uint64(len(err_buf)),
    )
    if rc != 0:
        msg = err_buf.value.decode("utf-8", errors="replace") if err_buf.value else f"error code {rc}"
        raise NativeRetopologyError(msg)

    n_pos = out_pos_len.value
    n_idx = out_idx_len.value
    n_fs = out_fs_len.value
    try:
        pos = np.ctypeslib.as_array(out_pos, shape=(n_pos,)).copy().reshape(-1, 3)
        idx = np.ctypeslib.as_array(out_idx, shape=(n_idx,)).copy()
        sizes = np.ctypeslib.as_array(out_fs, shape=(n_fs,)).copy() if n_fs else np.empty(0, np.uint32)
    finally:
        lib.openretopo_free_positions(out_pos, n_pos)
        lib.openretopo_free_indices(out_idx, n_idx)
        lib.openretopo_free_indices(out_fs, n_fs)

    if int(sizes.sum()) != idx.size or (idx.size and int(idx.max()) >= len(pos)) or not np.isfinite(pos).all():
        raise NativeRetopologyError("engine returned inconsistent buffers (refusing to build a mesh)")

    out = {name: getattr(stats, name) for name in _STAT_FIELDS}
    out["target_reached"] = bool(stats.target_reached)
    return pos, sizes, idx, out


# ---------------------------------------------------------------------------
# output object creation (main thread only)
# ---------------------------------------------------------------------------

def _create_output_object(context, source, positions, face_sizes, indices, suffix="_retopo", scale=None):
    """Build the result object. Raises NativeRetopologyError (and leaves no
    trace in bpy.data) if Blender's validator would have to repair it.
    `scale` undoes the per-axis scale applied by `_extract_triangles`."""
    if scale is not None:
        positions = positions / scale
    starts = np.concatenate(([0], np.cumsum(face_sizes)[:-1])).astype(np.int64)
    faces = [indices[s:s + n].tolist() for s, n in zip(starts.tolist(), face_sizes.tolist())]
    mesh = bpy.data.meshes.new(source.name + suffix)
    mesh.from_pydata(positions.astype(np.float32).tolist(), [], faces)
    if mesh.validate(verbose=False, clean_customdata=False):
        bpy.data.meshes.remove(mesh)
        raise NativeRetopologyError("result failed Blender mesh validation (nothing was created)")
    mesh.update()
    if hasattr(mesh, "shade_smooth"):  # Blender 4.1+
        mesh.shade_smooth()
    else:
        for poly in mesh.polygons:
            poly.use_smooth = True

    obj = bpy.data.objects.new(source.name + suffix, mesh)
    for coll in source.users_collection or (context.scene.collection,):
        coll.objects.link(obj)
    obj.matrix_world = source.matrix_world.copy()

    for o in context.selected_objects:
        o.select_set(False)
    obj.select_set(True)
    context.view_layer.objects.active = obj
    return obj


# ---------------------------------------------------------------------------
# operator
# ---------------------------------------------------------------------------

class OPENRETOPO_OT_remesh(Operator):
    bl_idname = "open_retopo.remesh"
    bl_label = "Retopologize"
    bl_options = {"REGISTER", "UNDO"}
    bl_description = (
        "Rebuild the active mesh with clean, field-aligned topology. "
        "The original object is preserved; a new *_retopo object is created."
    )

    # Operator-invoked with explicit keywords or from the UI button.
    target_faces: IntProperty(
        name="Target Faces",
        default=12000,
        min=100,
        max=1_000_000,
        description="Approximate face count of the output (hero-asset range recommended)",
    )
    mode: EnumProperty(
        name="Mode",
        items=(
            ("FIELD", "Field-Aligned Quads", "Quad-dominant, field-aligned remesh (closed meshes)"),
            ("DECIMATE", "Decimate (legacy)", "Weld + triangle decimation, works on open meshes"),
        ),
        default="FIELD",
    )
    symmetry: EnumProperty(
        name="Symmetry",
        items=(
            ("AUTO", "Auto", "Mirror across the object's most symmetric local axis, if it has one"),
            ("NONE", "Off", "No symmetry constraint"),
            ("X", "X", "Mirror across the object's local X axis"),
            ("Y", "Y", "Mirror across the object's local Y axis"),
            ("Z", "Z", "Mirror across the object's local Z axis"),
        ),
        default="AUTO",
    )
    curvature: BoolProperty(
        name="Follow Curvature",
        default=True,
        description="Align edge loops with limbs and other tube-like shapes",
    )
    sync: BoolProperty(
        name="Run Synchronously",
        default=False,
        options={"HIDDEN", "SKIP_SAVE"},
        description="Internal: run on the calling thread (used by headless tests)",
    )

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == "MESH"

    def invoke(self, context, event):
        obj = context.active_object
        try:
            self._scale = _object_scale(obj)
            positions, indices = _extract_triangles(obj.data, self._scale)
        except Exception as exc:  # extraction must never kill Blender
            self.report({"ERROR"}, f"OpenRetopo: could not read mesh: {exc}")
            return {"CANCELLED"}

        self._source_name = obj.name
        self._source = obj
        self._positions = positions
        self._indices = indices
        self._shared = _NativeShared(0.0, 0)
        self._result = None
        self._error = None

        if self.sync:
            return self._finish_sync(context)

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._queue = queue.Queue(maxsize=1)
        self._thread.start()
        self._timer = context.window_manager.event_timer_add(0.1, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def _worker(self):
        try:
            result = _native_remesh(
                self._positions,
                self._indices,
                self.target_faces,
                progress=self._shared,
                **self._engine_options(),
            )
            self._queue.put(("ok", result))
        except Exception as exc:
            self._queue.put(("err", str(exc)))

    def _finish_sync(self, context):
        try:
            result = _native_remesh(self._positions, self._indices, self.target_faces, **self._engine_options())
        except NativeRetopologyError as exc:
            self.report({"ERROR"}, f"OpenRetopo: {exc}")
            return {"CANCELLED"}
        return self._create(context, *result)

    def _engine_options(self):
        return {
            "decimate_only": self.mode == "DECIMATE",
            "symmetry": self.symmetry,
            "curvature": self.curvature,
        }

    def _create(self, context, pos, sizes, idx, stats):
        global OPENRETOPO_LAST_STATS
        OPENRETOPO_LAST_STATS = None
        try:
            obj = _create_output_object(context, self._source, pos, sizes, idx, scale=self._scale)
        except NativeRetopologyError as exc:
            self.report({"ERROR"}, f"OpenRetopo: {exc}")
            return {"CANCELLED"}
        if self.mode == "FIELD":
            msg = (
                f"OpenRetopo: {stats['output_faces']} faces "
                f"({stats['quads']} quads, {stats['triangles']} tris), "
                f"poles v3/v5/other {stats['poles_valence3']}/{stats['poles_valence5']}/{stats['poles_other']}"
            )
            if stats["symmetry_axis"]:
                msg += f", mirrored on {'XYZ'[stats['symmetry_axis'] - 1]}"
            msg += f" in {stats['duration_seconds']:.2f}s"
        else:
            msg = (
                f"OpenRetopo: {stats['output_faces']} faces "
                f"(welded {stats['welded_vertices_removed']}, "
                f"degenerate {stats['degenerate_faces_removed']}, "
                f"collapses {stats['collapses']}) in {stats['duration_seconds']:.2f}s"
            )
            if not stats["target_reached"]:
                msg += " — target not fully reached (locked boundary/crease edges)"
        self.report({"INFO"}, msg)
        print("[OpenRetopo]", msg, "->", obj.name)
        OPENRETOPO_LAST_STATS = dict(stats)
        OPENRETOPO_LAST_STATS["output_object"] = obj.name
        return {"FINISHED"}

    def modal(self, context, event):
        if event.type == "ESC":
            self._shared.cancel = 1
            self.report({"INFO"}, "OpenRetopo: cancelling…")
            return {"RUNNING_MODAL"}
        if event.type != "TIMER":
            return {"RUNNING_MODAL"}

        if self._thread.is_alive():
            try:
                context.workspace.status_text_set(f"OpenRetopo {self._shared.progress * 100:.0f}% — Esc to cancel")
            except Exception:
                pass
            return {"RUNNING_MODAL"}

        context.workspace.status_text_set(None)
        context.window_manager.event_timer_remove(self._timer)
        self._thread.join()
        kind, payload = self._queue.get_nowait()
        if kind == "err":
            self.report({"ERROR"}, f"OpenRetopo: {payload}")
            return {"CANCELLED"}
        return self._create(context, *payload)

    def execute(self, context):
        # Fallback entry point (e.g. from Python scripts); behaves like sync.
        try:
            obj = context.active_object
            self._scale = _object_scale(obj)
            positions, indices = _extract_triangles(obj.data, self._scale)
            self._source = obj
            self._source_name = obj.name
            result = _native_remesh(positions, indices, self.target_faces, **self._engine_options())
        except NativeRetopologyError as exc:
            self.report({"ERROR"}, f"OpenRetopo: {exc}")
            return {"CANCELLED"}
        return self._create(context, *result)


# ---------------------------------------------------------------------------
# panel
# ---------------------------------------------------------------------------

class OPENRETOPO_PT_main(Panel):
    bl_label = "OpenRetopo"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "OpenRetopo"

    def draw(self, context):
        layout = self.layout
        obj = context.active_object

        col = layout.column()
        if obj is None or obj.type != "MESH":
            col.label(text="Select a mesh object", icon="INFO")
            return

        col.label(text=f"Active: {obj.name}", icon="MESH_DATA")
        col.separator()
        op = col.operator("open_retopo.remesh", text="Retopologize", icon="MOD_REMESH")
        scene_props = context.scene.open_retopo
        op.target_faces = scene_props.target_faces
        op.symmetry = scene_props.symmetry
        op.curvature = scene_props.curvature

        col.prop(scene_props, "target_faces")
        col.prop(scene_props, "symmetry")
        col.prop(scene_props, "curvature")
        col.separator()
        col.label(text="Closed meshes only (no holes).", icon="INFO")


class OPENRETOPO_SceneProps(PropertyGroup):
    target_faces: IntProperty(
        name="Target Faces",
        default=12000,
        min=100,
        max=1_000_000,
        description="Approximate face count for the output mesh",
    )
    symmetry: EnumProperty(
        name="Symmetry",
        items=(
            ("AUTO", "Auto", "Mirror across the object's most symmetric local axis, if it has one"),
            ("NONE", "Off", "No symmetry constraint"),
            ("X", "X", "Mirror across the object's local X axis"),
            ("Y", "Y", "Mirror across the object's local Y axis"),
            ("Z", "Z", "Mirror across the object's local Z axis"),
        ),
        default="AUTO",
    )
    curvature: BoolProperty(
        name="Follow Curvature",
        default=True,
        description="Align edge loops with limbs and other tube-like shapes",
    )


CLASSES = (OPENRETOPO_SceneProps, OPENRETOPO_OT_remesh, OPENRETOPO_PT_main)


def register():
    _load_library()  # fail fast with a clear error at enable time
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.open_retopo = bpy.props.PointerProperty(type=OPENRETOPO_SceneProps)


def unregister():
    del bpy.types.Scene.open_retopo
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
