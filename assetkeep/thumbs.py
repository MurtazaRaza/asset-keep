"""Making every kind of asset into a 256 px tile.

Three rules shape this module.

**If the source already fits the box, no thumbnail is generated at all.** A
32x32 sprite is about 1 KB and any WebP of it would be larger, so the original
is served instead. 36% of images in the calibration project are already under
256 px, which makes this the single biggest saving available, and it costs one
comparison.

**Encoding is chosen by content, not by format.** 256 colours or fewer goes
lossless WebP, which is tiny on flat art and avoids the ringing that lossy
compression puts around hard pixel-art edges. Everything else goes lossy at q80.
65% of the calibration project takes the lossless path.

**Thumbnails are content-addressed and live outside SQLite.** Sharded two levels
by hash prefix so no directory holds ten thousand entries, shared automatically
between duplicates, and deletable at any time - the database has to stay
disposable, and a database with a hundred megabytes of images in it is not.

3D is rendered by a small numpy z-buffer rasteriser rather than an OpenGL stack,
because headless GL on macOS is a fight with no payoff at 256 px. Audio is drawn
as waveform peaks, because otherwise an audio file is an invisible row in a grid
built for images.
"""

from __future__ import annotations

import io
import logging
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from . import hashing
from .config import Config
from .probe import audio, model3d

log = logging.getLogger(__name__)

#: At or below this, lossless WebP wins on both size and quality.
LOSSLESS_MAX_COLORS = 256

#: Rendered at twice the output edge and downsampled, which antialiases the
#: silhouette. A flat-shaded model at 256 px without it looks like a jaggy mess,
#: and the cost lands on pixels rather than on the triangle loop that dominates.
SUPERSAMPLE = 2

#: Triangles rendered before the mesh is decimated by simple striding. A
#: 127,000-triangle city renders identically at 256 px to a 60,000-triangle one,
#: and the loop is linear in this number.
MAX_RENDER_TRIANGLES = 60_000

#: Three-quarter view, in radians. Straight-on orthographic renders of game
#: props are close to unreadable: a chest becomes a rectangle.
VIEW_YAW = math.radians(35.0)
VIEW_PITCH = math.radians(-22.0)

#: Fraction of the frame the model fills, leaving a margin so nothing clips.
VIEW_FILL = 0.88

#: Light direction in view space, and how much of the surface is lit regardless.
#: Pure directional shading leaves unlit faces pure black, which reads as a hole.
LIGHT = np.array([0.35, 0.6, 0.72], dtype=np.float64)
AMBIENT = 0.28

MODEL_COLOR = (196, 200, 208)
WAVEFORM_COLOR = (120, 190, 240)
WAVEFORM_BACKGROUND = (24, 26, 32)

#: Columns that reach full scale are drawn in this instead, so that a clipped
#: file announces itself in the grid.
CLIPPING_COLOR = (240, 140, 120)

#: Amplitude at which a column counts as touching full scale, as a fraction of
#: it. Derived from the same decibel threshold ``is:clipping`` uses rather than
#: chosen separately: a tile that disagrees with the filter about which of its
#: files are clipping is worse than one that says nothing, and the first version
#: of this did exactly that - a hard-coded 0.999 compared against the
#: square-rooted amplitude, which works out at -0.017 dB against the filter's
#: -0.1 dB, so every file between the two matched `is:clipping` and drew clean.
CLIPPING_MAGNITUDE = 10 ** (audio.CLIPPING_DB / 20)

#: Columns in a waveform thumbnail. One per pixel at 256 px wide.
WAVEFORM_COLUMNS = 256

FFMPEG_TIMEOUT = 30


def path_for(config: Config, content_hash: str) -> Path:
    """Where this asset's thumbnail lives, whether or not it exists yet."""
    first, second = hashing.shard(content_hash)
    return config.thumbs_path / first / second / f"{content_hash}.webp"


def generate(config: Config, kind: str, source: Path, content_hash: str) -> Path | None:
    """Render and store one thumbnail. ``None`` means "serve the original".

    Never raises. A thumbnail is a nicety; failing to make one must not fail the
    job queue, and the UI already has to handle assets that legitimately have no
    thumbnail file because they were too small to need one.
    """
    source = Path(source)
    destination = path_for(config, content_hash)
    if destination.exists():
        return destination

    try:
        image = _render(config, kind, source)
    except Exception as exc:  # noqa: BLE001 - see the docstring
        log.warning("thumbnail failed for %s: %s", source, exc)
        return None

    if image is None:
        return None

    destination.parent.mkdir(parents=True, exist_ok=True)
    _save(image, destination, config.thumbnails.quality)
    return destination


def store(config: Config, content_hash: str, data: bytes) -> Path | None:
    """Make a tile out of image bytes that never came from a file on disk.

    This is the reference path: a page's ``og:image``, fetched over HTTP and
    held in memory. ``None`` means the bytes were not a decodable image, which
    is a thing servers send.

    Unlike :func:`generate`, a small source is written out rather than skipped.
    The skip rule exists because the original can be served instead, and here
    there is no original - only a remote URL that may be gone tomorrow, which is
    most of the reason to have kept a copy of the picture at all.
    """
    destination = path_for(config, content_hash)
    if destination.exists():
        return destination

    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            fitted = _fit(image, config.thumbnails.max_edge)
    except Exception as exc:  # noqa: BLE001 - see the docstring
        log.warning("preview bytes are not a usable image: %s", exc)
        return None

    destination.parent.mkdir(parents=True, exist_ok=True)
    _save(fitted, destination, config.thumbnails.quality)
    return destination


def _render(config: Config, kind: str, source: Path) -> Image.Image | None:
    box = config.thumbnails.max_edge

    if kind == "image":
        return _image_thumb(config, source, box)
    if kind == "model3d":
        return _model_thumb(config, source, box)
    if kind == "audio":
        return _audio_thumb(source, box)
    return None


# --- images -----------------------------------------------------------------


def _image_thumb(config: Config, source: Path, box: int) -> Image.Image | None:
    if source.suffix.lower() == ".exr":
        image = _decode_with_ffmpeg(source)
    elif source.suffix.lower() in (".aseprite", ".ase"):
        # No decoder, and an Aseprite file almost always sits beside the PNG it
        # was exported to, which the scan indexes separately anyway.
        return None
    else:
        image = Image.open(source)
        image.load()

    if image is None:
        return None

    if config.thumbnails.skip_smaller and max(image.size) <= box:
        return None

    return _fit(image, box)


def _fit(image: Image.Image, box: int) -> Image.Image:
    """Scale to fit the box, preserving aspect and transparency.

    NEAREST when scaling down by a whole number, which is the pixel-art case:
    LANCZOS on a 4x-upscaled sprite reintroduces exactly the soft edges the art
    style exists to avoid.
    """
    image = image.convert("RGBA")
    scale = box / max(image.size)
    if scale >= 1:
        return image

    factor = 1 / scale
    resample = (
        Image.Resampling.NEAREST
        if abs(factor - round(factor)) < 0.01 and round(factor) > 1
        else Image.Resampling.LANCZOS
    )
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, resample)


def _save(image: Image.Image, destination: Path, quality: int) -> None:
    """Write WebP, lossless for flat art and lossy for everything else."""
    colors = image.getcolors(maxcolors=LOSSLESS_MAX_COLORS)
    if colors is not None:
        image.save(destination, "WEBP", lossless=True, method=4)
    else:
        image.save(destination, "WEBP", quality=quality, method=4)


# --- 3D ---------------------------------------------------------------------


def _model_thumb(config: Config, source: Path, box: int) -> Image.Image | None:
    """A supplied preview if the pack shipped one, otherwise a render."""
    preview = model3d.sibling_preview(source)
    if preview is not None:
        try:
            with Image.open(preview) as image:
                image.load()
                return _fit(image, box)
        except Exception as exc:  # noqa: BLE001
            log.debug("sibling preview unusable for %s: %s", source, exc)

    geometry = model3d.load_geometry(source, config.assimp_lib_path)
    if geometry is None:
        return None

    vertices, faces = geometry
    return render_mesh(vertices, faces, box)


def render_mesh(vertices: np.ndarray, faces: np.ndarray, size: int = 256) -> Image.Image:
    """Flat-shaded orthographic render on a transparent background.

    A z-buffer over triangles, in numpy, with the per-triangle loop in Python
    and everything inside it vectorised over that triangle's bounding box. That
    split is what makes it fast enough: a 60,000-triangle mesh at 512 px spends
    its time in numpy, not in the interpreter.

    **Y is assumed to be up**, which is the Unity and FBX convention, and which
    assimp has usually already normalised to. Which way the model *faces* is not
    knowable - a character exported from Blender may face any horizontal
    direction - so the camera instead turns to put the wider horizontal extent
    across the screen. A figure with outstretched arms is then seen from the
    front rather than edge on, which is the difference between a recognisable
    silhouette and a vertical smear.

    >>> corners = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    >>> image = render_mesh(corners, np.array([[0, 1, 2]]), 32)
    >>> image.size, image.mode
    ((32, 32), 'RGBA')
    >>> bool(np.asarray(image)[:, :, 3].max() > 0)  # something was drawn
    True
    """
    resolution = size * SUPERSAMPLE
    screen, depth, shade = _project(vertices, faces, resolution)

    zbuffer = np.full((resolution, resolution), np.inf)
    canvas = np.zeros((resolution, resolution, 4), dtype=np.uint8)

    # Over ``screen``, not over ``faces``: _project decimates dense meshes, and
    # its rebinding of the name is local to it. Looping on the original count
    # walks off the end of every array here the moment a mesh exceeds the cap.
    for index in range(len(screen)):
        _draw_triangle(
            canvas, zbuffer, screen[index], depth[index], shade[index], resolution
        )

    image = Image.fromarray(canvas, mode="RGBA")
    return image.resize((size, size), Image.Resampling.LANCZOS)


def _project(
    vertices: np.ndarray, faces: np.ndarray, resolution: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotate, fit and shade, returning per-triangle screen data."""
    if len(faces) > MAX_RENDER_TRIANGLES:
        # Striding rather than a real decimation: it keeps a uniform sample of
        # the surface, costs nothing, and at 256 px the difference is invisible.
        faces = faces[:: math.ceil(len(faces) / MAX_RENDER_TRIANGLES)]

    span = vertices.max(axis=0) - vertices.min(axis=0)
    # Quarter turn when the model is deeper than it is wide, so the camera looks
    # along its narrow axis and the broad side faces us.
    facing = math.pi / 2 if span[2] > span[0] else 0.0
    rotated = vertices.astype(np.float64) @ _view_rotation(facing).T

    low, high = rotated.min(axis=0), rotated.max(axis=0)
    extent = float(max(high[0] - low[0], high[1] - low[1])) or 1.0
    scale = resolution * VIEW_FILL / extent
    centre = (low + high) / 2

    x = (rotated[:, 0] - centre[0]) * scale + resolution / 2
    # Screen y grows downwards; model y grows up.
    y = resolution / 2 - (rotated[:, 1] - centre[1]) * scale
    z = rotated[:, 2]

    corners = np.stack([x, y], axis=1)[faces]  # (M, 3, 2)
    depth = z[faces]  # (M, 3)

    edge_a = rotated[faces[:, 1]] - rotated[faces[:, 0]]
    edge_b = rotated[faces[:, 2]] - rotated[faces[:, 0]]
    normals = np.cross(edge_a, edge_b)
    lengths = np.linalg.norm(normals, axis=1)
    lengths[lengths == 0] = 1.0
    normals /= lengths[:, None]

    # Absolute dot product, so a mesh with inconsistent winding - which is most
    # of them - is lit on both sides instead of showing black holes.
    lambert = np.abs(normals @ (LIGHT / np.linalg.norm(LIGHT)))
    shade = AMBIENT + (1.0 - AMBIENT) * lambert
    return corners, depth, shade


def _view_rotation(facing: float = 0.0) -> np.ndarray:
    """Yaw about the up axis, then pitch, giving a three-quarter view."""
    yaw_angle = VIEW_YAW + facing
    cos_yaw, sin_yaw = math.cos(yaw_angle), math.sin(yaw_angle)
    cos_pitch, sin_pitch = math.cos(VIEW_PITCH), math.sin(VIEW_PITCH)
    yaw = np.array(
        [[cos_yaw, 0.0, sin_yaw], [0.0, 1.0, 0.0], [-sin_yaw, 0.0, cos_yaw]]
    )
    pitch = np.array(
        [[1.0, 0.0, 0.0], [0.0, cos_pitch, -sin_pitch], [0.0, sin_pitch, cos_pitch]]
    )
    return pitch @ yaw


def _draw_triangle(
    canvas: np.ndarray,
    zbuffer: np.ndarray,
    corners: np.ndarray,
    depth: np.ndarray,
    shade: float,
    resolution: int,
) -> None:
    """Z-tested fill of one triangle over its own bounding box."""
    min_x = max(0, int(np.floor(corners[:, 0].min())))
    max_x = min(resolution - 1, int(np.ceil(corners[:, 0].max())))
    min_y = max(0, int(np.floor(corners[:, 1].min())))
    max_y = min(resolution - 1, int(np.ceil(corners[:, 1].max())))
    if min_x > max_x or min_y > max_y:
        return

    (x0, y0), (x1, y1), (x2, y2) = corners
    denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(denominator) < 1e-12:  # degenerate: an edge seen exactly side on
        return

    grid_x, grid_y = np.meshgrid(
        np.arange(min_x, max_x + 1), np.arange(min_y, max_y + 1)
    )
    bary_0 = ((y1 - y2) * (grid_x - x2) + (x2 - x1) * (grid_y - y2)) / denominator
    bary_1 = ((y2 - y0) * (grid_x - x2) + (x0 - x2) * (grid_y - y2)) / denominator
    bary_2 = 1.0 - bary_0 - bary_1

    inside = (bary_0 >= 0) & (bary_1 >= 0) & (bary_2 >= 0)
    if not inside.any():
        # Smaller than a pixel. Dense meshes are mostly these at 256 px, and
        # dropping them punches holes in the surface, so the centroid is
        # plotted instead.
        inside = np.zeros_like(inside)
        inside[(grid_y.shape[0] - 1) // 2, (grid_x.shape[1] - 1) // 2] = True
        pixel_depth = np.full(inside.shape, depth.mean())
    else:
        pixel_depth = bary_0 * depth[0] + bary_1 * depth[1] + bary_2 * depth[2]

    window = zbuffer[min_y : max_y + 1, min_x : max_x + 1]
    visible = inside & (pixel_depth < window)
    if not visible.any():
        return

    window[visible] = pixel_depth[visible]
    tone = np.clip(np.array(MODEL_COLOR) * shade, 0, 255).astype(np.uint8)
    target = canvas[min_y : max_y + 1, min_x : max_x + 1]
    target[visible] = (*tone, 255)


# --- audio ------------------------------------------------------------------


def _audio_thumb(source: Path, box: int) -> Image.Image | None:
    """Waveform peaks, so an audio file is visible in a grid built for images."""
    samples = audio.decode_pcm(source)
    if samples is None or samples.size == 0:
        return None
    return render_waveform(samples, box)


def render_waveform(samples: np.ndarray, size: int = 256) -> Image.Image:
    """Draw min/max peaks per column, at a scale that keeps loudness readable.

    Peaks rather than a decimated sample: taking every Nth sample of a waveform
    aliases badly, and a quiet passage next to a loud one comes out looking the
    same. The min and max over each window is what the eye expects to see.

    **Not peak-normalised**, which is the change M6 made and the one worth
    explaining. Scaling each tile so its own loudest moment fills the frame is
    what every audio editor does, and it is wrong here: an asset grid is a
    comparison between files, not a view of one. Measured across 215 real
    files, peak level spans 67 files within a decibel of full scale and 18 more
    than 24 dB below it - and normalising drew all 85 of them identically, so
    the one question a wall of waveforms should be able to answer at a glance,
    "why is that one so quiet", was the one it could not.

    Amplitude is square-rooted rather than drawn linearly. Linear is honest and
    useless: at -24 dB a file is 6% of full scale, which is three pixels of a
    256 px tile and reads as an empty box. The square root is the standard
    half-decibel compromise - full scale stays full height, -6 dB draws at 71%,
    -24 dB at 25% - so a quiet file still shows its envelope while still
    being visibly quiet.

    >>> import numpy as np
    >>> tone = np.sin(np.linspace(0, 400, 44100)) * 32768 * 0.9
    >>> render_waveform(tone, 64).size
    (64, 64)

    A quiet clip draws shorter than a loud one rather than the same height:

    >>> loud = drawn_rows(render_waveform(tone, 64))
    >>> quiet = drawn_rows(render_waveform(tone / 16, 64))
    >>> quiet < loud
    True
    """
    columns = max(1, min(WAVEFORM_COLUMNS, size, samples.size))
    lows, highs = _column_peaks(samples, columns)

    # Measured before the square root, because the threshold is a real
    # amplitude and the square root is only how it gets drawn.
    clipping = np.maximum(highs, -lows) >= CLIPPING_MAGNITUDE

    # Square-rooted magnitude, sign preserved: see the docstring.
    lows = -np.sqrt(np.abs(lows))
    highs = np.sqrt(highs)

    canvas = np.zeros((size, size, 4), dtype=np.uint8)
    canvas[:, :] = (*WAVEFORM_BACKGROUND, 255)

    middle = size / 2
    for column in range(columns):
        left = int(column * size / columns)
        right = max(left + 1, int((column + 1) * size / columns))
        top = int(middle - highs[column] * middle * 0.92)
        bottom = int(middle - lows[column] * middle * 0.92)
        # A column that reaches full scale is drawn in the warning colour, so a
        # clipped file is recognisable as one from the grid rather than only
        # from `is:clipping`. 18 of the 175 in the calibration library light up.
        canvas[min(top, bottom) : max(top, bottom) + 1, left:right] = (
            *(CLIPPING_COLOR if clipping[column] else WAVEFORM_COLOR),
            255,
        )

    return Image.fromarray(canvas, mode="RGBA")


def _column_peaks(samples: np.ndarray, columns: int) -> tuple[np.ndarray, np.ndarray]:
    """Min and max of each column's window, as a fraction of full scale.

    ``reduceat`` over computed edges rather than a reshape. The reshape it
    replaces truncated the tail to make the length divide evenly, which is
    harmless on a music track and destructive on a short one: a clip of 300
    samples has 256 columns to fill, so it lost a fifth of itself, and anything
    under 256 samples was dropped entirely and drawn as a flat line.

    >>> import numpy as np
    >>> lows, highs = _column_peaks(np.array([-32768.0, 32768.0]), 2)
    >>> [round(float(v), 2) for v in lows], [round(float(v), 2) for v in highs]
    ([-1.0, 0.0], [0.0, 1.0])
    """
    edges = np.linspace(0, samples.size, columns + 1).astype(np.intp)[:-1]
    scaled = samples / audio.FULL_SCALE
    lows = np.minimum.reduceat(scaled, edges)
    highs = np.maximum.reduceat(scaled, edges)
    return np.clip(lows, -1, 0), np.clip(highs, 0, 1)


def drawn_rows(image: Image.Image) -> int:
    """How many rows of a waveform tile have any wave drawn on them.

    A measurement helper rather than part of the pipeline: it is what the
    doctest above compares, and what the M6 calibration run counted to find the
    one file in 215 that draws as a single flat row.
    """
    array = np.asarray(image.convert("RGB"))
    drawn = (array != np.array(WAVEFORM_BACKGROUND, dtype=array.dtype)).any(axis=2)
    return int(drawn.any(axis=1).sum())


# --- EXR --------------------------------------------------------------------


def _decode_with_ffmpeg(source: Path) -> Image.Image | None:
    """One frame out of a format PIL cannot open, as PNG on a pipe."""
    if shutil.which("ffmpeg") is None:
        return None

    completed = subprocess.run(
        ["ffmpeg", "-v", "quiet", "-i", str(source),
         "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"],
        capture_output=True,
        timeout=FFMPEG_TIMEOUT,
        check=False,
    )
    if completed.returncode != 0 or not completed.stdout:
        return None

    image = Image.open(io.BytesIO(completed.stdout))
    image.load()
    return image


def available() -> bool:
    """Whether the ffmpeg-backed paths (audio waveforms, EXR) can run."""
    return shutil.which("ffmpeg") is not None
