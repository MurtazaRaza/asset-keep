"""3D models: geometry counts, bounds, and whether there is a rig in there.

FBX outnumbers OBJ 56 to 1 in the calibration project - 223 files against 4 -
so FBX is the primary path here, not a degraded one. It needs assimp, which is a
system library rather than a wheel, and that shapes the whole module: two
backends behind one interface, chosen by extension, with the assimp side allowed
to be absent.

The loader chain is ``impasse``, then ``pyassimp``, then an explicit
``assimp_lib_path`` from config. ``pyassimp`` is not first because it is poorly
maintained and routinely fails to locate the native library; it is present at
all because when impasse is missing it is often already installed as somebody
else's dependency. Note that both *raise on import* when the library is missing,
rather than failing to import, so the chain catches exceptions rather than
:class:`ImportError` alone.

macOS gets one extra courtesy: ``/opt/homebrew/lib`` is added to the search path
before either binding is imported. Neither looks there, Apple Silicon Homebrew
installs there and nowhere else, and without it ``brew install assimp`` appears
to do nothing at all.

**A sibling image sharing the model's basename beats any render.** Asset packs
ship these constantly, and a preview drawn by the person who made the model
carries information a flat-shaded turntable cannot.
"""

from __future__ import annotations

import functools
import logging
import os
from pathlib import Path

import numpy as np

from . import CONTROL_FLOW, ProbeResult

log = logging.getLogger(__name__)

#: Formats where assimp is the only realistic option.
ASSIMP_EXTENSIONS = frozenset({".fbx", ".dae", ".blend"})

#: Formats trimesh reads in pure Python, so they work with no system dependency.
TRIMESH_EXTENSIONS = frozenset({".gltf", ".glb", ".obj", ".stl", ".ply"})

#: Directories to add to the library search path before importing a binding.
LIBRARY_HINTS = ("/opt/homebrew/lib", "/usr/local/lib", "/opt/local/lib")

#: Suffixes a pack's supplied preview tends to carry beyond the bare stem.
PREVIEW_SUFFIXES = ("", "_preview", "_thumb", "_thumbnail", "_icon", "-preview")

PREVIEW_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".tga", ".bmp")

#: At or below this a mesh is low-poly by any reading, including a game
#: artist's. Above it the term stops meaning anything and the tag is not applied.
LOW_POLY_MAX_TRIANGLES = 2000


def probe(path: Path, assimp_lib_path: Path | None = None, **_options) -> ProbeResult:
    """Geometry attributes and tags for one model file."""
    result = ProbeResult(attributes={"bytes": path.stat().st_size})
    result.preview = sibling_preview(path)

    extension = path.suffix.lower()
    if extension in TRIMESH_EXTENSIONS:
        _load_trimesh(path, result)
    else:
        _load_assimp(path, result, assimp_lib_path)

    _tag_from_geometry(result)
    return result


def capabilities(assimp_lib_path: Path | None = None) -> dict[str, object]:
    """What this module can currently read, for ``/api/capabilities``.

    The point of reporting this is that the UI can say "223 FBX files are
    indexed but unparsed, install assimp" instead of showing 223 blank tiles
    with no explanation.
    """
    backend = _assimp_backend(str(assimp_lib_path) if assimp_lib_path else None)
    return {
        "trimesh": _trimesh() is not None,
        "assimp": backend[0] if backend else None,
        "formats": sorted(
            TRIMESH_EXTENSIONS if _trimesh() else set()
        ) + sorted(ASSIMP_EXTENSIONS if backend else set()),
    }


def load_geometry(
    path: Path, assimp_lib_path: Path | None = None
) -> tuple[np.ndarray, np.ndarray] | None:
    """Vertices and triangle indices for rendering, or ``None`` if unreadable.

    Separate from :func:`probe` because the two want different things. The probe
    wants counts and is run once at scan time; the thumbnailer wants the actual
    arrays and runs later, in a background job, only for models that have no
    supplied preview. Loading the file twice is the cheaper trade: keeping a
    127,000-triangle mesh in memory between the two would not be.

    Meshes are concatenated into one buffer with their indices rebased, since a
    flat-shaded 256 px render has no use for the scene graph.
    """
    extension = path.suffix.lower()
    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    offset = 0

    try:
        for mesh_vertices, mesh_faces in _meshes(path, extension, assimp_lib_path):
            if len(mesh_vertices) == 0 or len(mesh_faces) == 0:
                continue
            vertices.append(np.asarray(mesh_vertices, dtype=np.float32))
            faces.append(np.asarray(mesh_faces, dtype=np.int64) + offset)
            offset += len(mesh_vertices)
    except CONTROL_FLOW:
        raise
    except BaseException as exc:  # noqa: BLE001 - AssimpError is not an Exception
        log.debug("geometry unavailable for %s: %s", path, exc)
        return None

    if not vertices:
        return None
    return np.concatenate(vertices), np.concatenate(faces)


def _meshes(path: Path, extension: str, assimp_lib_path: Path | None):
    """Yield ``(vertices, faces)`` per mesh, in world space."""
    if extension in TRIMESH_EXTENSIONS:
        trimesh = _trimesh()
        if trimesh is None:
            return
        scene = trimesh.load(path, force="scene", process=False)
        # dump() is what bakes the scene graph into the vertices; iterating
        # scene.geometry hands back untransformed local-space copies.
        for geometry in scene.dump():
            if hasattr(geometry, "faces"):
                yield geometry.vertices, geometry.faces
        return

    backend = _assimp_backend(str(assimp_lib_path) if assimp_lib_path else None)
    if backend is None:
        return

    name, module = backend
    scene = module.load(str(path))
    context = scene if name == "pyassimp" else _null_context(scene)
    with context as loaded:
        yield from _world_meshes(loaded)


def _world_meshes(scene):
    """Walk the node tree, yielding ``(vertices, faces)`` in world space.

    Skipping this is the mistake that looks like it works. Every single-mesh
    prop has an identity transform, so a bush and a crate render correctly from
    raw mesh vertices and nothing seems wrong. A rigged character is eleven
    meshes hung off eleven transformed nodes, and in local space every one of
    them sits at the origin: the render comes out as all the body parts stacked
    on top of each other. Bounds are wrong for the same reason, which is how a
    humanoid ends up recorded as three times wider than it is tall.
    """
    stack = [(scene.root_node, np.identity(4))]
    while stack:
        node, inherited = stack.pop()
        transform = inherited @ np.asarray(node.transformation, dtype=np.float64)

        for mesh in node.meshes:
            # pyassimp hands back mesh objects here, but has used indices in
            # past versions, so both are accepted.
            if isinstance(mesh, (int, np.integer)):
                mesh = scene.meshes[int(mesh)]

            faces = np.asarray(mesh.faces)
            # assimp triangulates on import, but a mesh of lines or points has
            # a ragged face array that cannot be rendered and must not crash.
            if faces.ndim != 2 or faces.shape[1] != 3:
                continue

            # Both arrays are copied out of the scene, and that is not
            # defensive tidying. impasse hands back numpy views onto the C
            # scene's own memory, which assimp frees when the scene is
            # released - so a view that outlives it reads whatever now lives at
            # that address. It presents as face indices that are wildly out of
            # range on one file in a thousand, non-deterministically, which is
            # about the worst way a bug can present.
            local = np.array(mesh.vertices, dtype=np.float64, copy=True)
            yield (
                local @ transform[:3, :3].T + transform[:3, 3],
                np.array(faces, dtype=np.int64, copy=True),
            )

        for child in node.children:
            stack.append((child, transform))


def sibling_preview(path: Path) -> Path | None:
    """An image beside the model that is clearly a picture of it.

    Matching is case-insensitive because packs are authored on Windows and
    ``Chest.fbx`` ships next to ``chest.png`` about as often as not, and a
    case-sensitive filesystem would otherwise miss half of them.

    >>> import tempfile, pathlib
    >>> with tempfile.TemporaryDirectory() as d:
    ...     _ = pathlib.Path(d, "Chest.fbx").write_text("")
    ...     _ = pathlib.Path(d, "chest.png").write_text("")
    ...     sibling_preview(pathlib.Path(d, "Chest.fbx")).name
    'chest.png'
    """
    stem = path.stem.lower()
    wanted = {
        stem + suffix + extension
        for suffix in PREVIEW_SUFFIXES
        for extension in PREVIEW_EXTENSIONS
    }

    try:
        entries = sorted(path.parent.iterdir())
    except OSError:
        return None

    for entry in entries:
        if entry.name.lower() in wanted and entry.is_file():
            return entry
    return None


# --- backends ---------------------------------------------------------------


def _load_trimesh(path: Path, result: ProbeResult) -> None:
    trimesh = _trimesh()
    if trimesh is None:
        result.error = "trimesh not installed"
        return

    # force="scene" keeps multi-mesh files as scenes so the material count is
    # real; loading them merged would report every pack prop as one material.
    scene = trimesh.load(path, force="scene", process=False)
    geometries = list(getattr(scene, "geometry", {}).values()) or [scene]

    triangles = sum(int(len(getattr(g, "faces", ()))) for g in geometries)
    vertices = sum(int(len(getattr(g, "vertices", ()))) for g in geometries)

    result.attributes.update(
        triangles=triangles,
        vertices=vertices,
        materials=len({id(getattr(g, "visual", None)) for g in geometries}),
        animation_count=0,
        bone_count=0,
        has_rig=False,
    )
    _set_bounds(result, getattr(scene, "bounds", None))


def _load_assimp(
    path: Path, result: ProbeResult, assimp_lib_path: Path | None
) -> None:
    backend = _assimp_backend(str(assimp_lib_path) if assimp_lib_path else None)
    if backend is None:
        # Not an error the user did anything wrong, and not a reason to skip the
        # file: it still gets indexed, hashed, tagged from its name and folder,
        # and found by search. Only the geometry numbers are missing.
        result.error = "assimp not available"
        return

    name, module = backend
    scene = module.load(str(path))
    context = scene if name == "pyassimp" else _null_context(scene)

    with context as loaded:
        meshes = list(loaded.meshes)
        triangles = sum(int(len(mesh.faces)) for mesh in meshes)
        vertices = sum(int(len(mesh.vertices)) for mesh in meshes)
        bones = sum(int(len(getattr(mesh, "bones", ()))) for mesh in meshes)
        animations = len(getattr(loaded, "animations", ()))

        result.attributes.update(
            triangles=triangles,
            vertices=vertices,
            materials=len(getattr(loaded, "materials", ())),
            animation_count=animations,
            bone_count=bones,
            has_rig=bones > 0,
        )
        # Bounds have to come from the node walk, not from the meshes directly.
        # Counts are per-file and correct either way; a bounding box is a
        # statement about where the geometry actually is.
        _set_bounds(result, _bounds_of(_world_meshes(loaded)))


def _bounds_of(meshes) -> np.ndarray | None:
    """Axis-aligned bounds over world-space meshes, or ``None`` if empty."""
    low: np.ndarray | None = None
    high: np.ndarray | None = None

    for vertices, _faces in meshes:
        if len(vertices) == 0:
            continue
        mesh_low, mesh_high = vertices.min(axis=0), vertices.max(axis=0)
        low = mesh_low if low is None else np.minimum(low, mesh_low)
        high = mesh_high if high is None else np.maximum(high, mesh_high)

    return None if low is None else np.array([low, high])


def _set_bounds(result: ProbeResult, bounds) -> None:
    if bounds is None:
        return
    extent = np.asarray(bounds)[1] - np.asarray(bounds)[0]
    result.attributes.update(
        bounds_x=round(float(extent[0]), 4),
        bounds_y=round(float(extent[1]), 4),
        bounds_z=round(float(extent[2]), 4),
    )


def _tag_from_geometry(result: ProbeResult) -> None:
    attributes = result.attributes
    if attributes.get("has_rig"):
        result.tags.append(("rigged", "structural", None))
    if attributes.get("animation_count", 0) > 0:
        result.tags.append(("animation", "structural", None))

    triangles = attributes.get("triangles")
    if isinstance(triangles, int) and 0 < triangles <= LOW_POLY_MAX_TRIANGLES:
        result.tags.append(("low-poly", "heuristic", 0.7))


# --- loader discovery -------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _trimesh():
    try:
        import trimesh
    except Exception as exc:  # noqa: BLE001 - optional extra
        log.debug("trimesh unavailable: %s", exc)
        return None
    return trimesh


@functools.lru_cache(maxsize=4)
def _assimp_backend(lib_path: str | None):
    """First working assimp binding as ``(name, module)``, or ``None``.

    Cached because the failure case is the expensive one: without the library
    installed, each attempt walks a list of directories looking for a file that
    is not there, and a scan would otherwise pay that for all 223 FBX files.
    """
    _extend_library_path(lib_path)

    for name in ("impasse", "pyassimp"):
        try:
            module = __import__(name)
        except CONTROL_FLOW:
            raise
        except BaseException as exc:  # noqa: BLE001 - AssimpError is not an Exception
            log.debug("%s unavailable: %s", name, exc)
            continue
        return name, module

    return None


def _extend_library_path(lib_path: str | None) -> None:
    """Point the bindings at Homebrew, and at whatever config says.

    Both bindings read ``LD_LIBRARY_PATH`` when building their search list, and
    neither looks in ``/opt/homebrew/lib``. Writing to the environment is
    unlovely, but it is the only hook either one offers, and it has to happen
    before the import that triggers the search.
    """
    hints = list(LIBRARY_HINTS)
    if lib_path:
        candidate = Path(lib_path)
        hints.insert(0, str(candidate if candidate.is_dir() else candidate.parent))

    existing = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
    merged = list(dict.fromkeys(hints + existing))
    os.environ["LD_LIBRARY_PATH"] = ":".join(merged)


class _null_context:
    """Adapter so impasse's plain return value reads like pyassimp's manager."""

    def __init__(self, value) -> None:
        self._value = value

    def __enter__(self):
        return self._value

    def __exit__(self, *_exc) -> bool:
        return False
