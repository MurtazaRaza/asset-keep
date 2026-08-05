"""What can be read off an image with numpy, PIL and no model at all.

Six detections, and every one of them fails the same way: by over-triggering. A
detector that calls one image in twenty a normal map is worse than no detector,
because the wrong tag is indistinguishable from a right one once it is in the
sidebar and someone is filtering on it. So each test below is written to reject
first and accept reluctantly, and each has a near-miss negative in the tests -
the blue sky that is not a normal map, the greyscale photo that is not a mask.

Two formats do not go through PIL at all. ``.exr`` is a float format PIL cannot
decode, and ``.aseprite`` is not a format it knows; both carry their dimensions
in a fixed header, so a dozen bytes of :mod:`struct` gets the attributes that
matter and the thumbnail is left to ffmpeg (EXR) or skipped (Aseprite).
"""

from __future__ import annotations

import struct
import warnings
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile

from .. import pixelgrid, similarity
from . import ProbeResult

# Re-exported sprite sheets and interrupted exports are common enough that
# refusing to read a slightly short file would cost real assets.
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Palette PNGs with byte transparency are ordinary in game art and PIL warns on
# every single one during the RGBA conversion this module always does. The
# advice is aimed at someone editing an image, not at a scan of 500 of them.
warnings.filterwarnings(
    "ignore", message="Palette images with Transparency", category=UserWarning
)

#: Colour counts saturate here. Every decision made from this number is a
#: threshold far below it - 256 for lossless WebP, 64 for pixel art - so exact
#: counts above a few thousand buy nothing and cost a full-image unique pass.
MAX_COLOR_COUNT = 4096

#: Above this the statistical heuristics are skipped. A 16-megapixel image is a
#: baked lightmap or a giant atlas, where "is it tileable" is not a meaningful
#: question, and the arrays involved are hundreds of megabytes on a machine that
#: is also running a scan.
ANALYSIS_MAX_PIXELS = 16 * 1024 * 1024

#: Pixel art at native resolution has no detectable block period, so it is
#: identified by being small and flat instead. Both bounds are deliberately
#: tight: a 128 px image with 64 colours is drawn, not photographed.
NATIVE_PIXEL_ART_MAX_EDGE = 128
NATIVE_PIXEL_ART_MAX_COLORS = 64

#: Blocks needed along the shorter axis before a detected period is believed.
#: Measured against the real library, this is the guard that matters most: a
#: 48 px sprite was being reported as a 30 px grid, which is one and a half
#: blocks, which is not a grid.
MIN_BLOCKS_PER_AXIS = 8

#: Mean channel spread allowed inside a block before it stops counting as flat.
#: An untouched nearest-neighbour upscale gives exactly 0; the allowance is for
#: pixel art that has been through a lossy re-encode at some point, which is
#: most of what ships in a pack.
FLAT_BLOCK_TOLERANCE = 8.0

#: Border contrast as a fraction of the image's mean interior contrast, below
#: which a wrap match proves nothing. A 2048 px character atlas on a flat
#: background scores a perfect seam because both its edges are the same solid
#: colour - it does technically tile, and it is not a tileable texture.
MIN_EDGE_CONTRAST_RATIO = 0.25

#: Wrap discontinuity as a multiple of the image's own interior contrast. A
#: texture that genuinely tiles lands near 1.0 at any resolution; one that does
#: not scores 3 and up. See :func:`seam_score`.
SEAM_THRESHOLD = 1.5

#: Fraction of pixels that must be opaque before edge-wrap and flatness tests
#: mean anything. A sprite on a transparent background has identical (empty)
#: opposite edges and would otherwise be called tileable, which is true and
#: useless.
OPAQUE_COVERAGE = 0.95

EXR_MAGIC = b"\x76\x2f\x31\x01"
ASEPRITE_MAGIC = 0xA5E0


def probe(path: Path, **_options) -> ProbeResult:
    """Attributes, heuristic tags and perceptual data for one image."""
    suffix = path.suffix.lower()
    if suffix == ".exr":
        return _probe_exr(path)
    if suffix in (".aseprite", ".ase"):
        return _probe_aseprite(path)

    with Image.open(path) as img:
        img.load()
        return _probe_pil(img, path)


# --- the PIL path -----------------------------------------------------------


def _probe_pil(img: Image.Image, path: Path) -> ProbeResult:
    result = ProbeResult()
    width, height = img.size

    colors = img.convert("RGBA").getcolors(maxcolors=MAX_COLOR_COUNT)
    result.attributes = {
        "width": width,
        "height": height,
        "aspect": round(width / height, 4) if height else 0.0,
        "mode": img.mode,
        "color_count": len(colors) if colors else MAX_COLOR_COUNT,
        "frame_count": int(getattr(img, "n_frames", 1)),
        "bytes": path.stat().st_size,
    }

    if result.attributes["frame_count"] > 1:
        result.tags.append(("animation", "structural", None))

    if width * height > ANALYSIS_MAX_PIXELS:
        result.attributes["has_alpha"] = img.mode in ("RGBA", "LA", "PA")
        return result

    rgba = np.asarray(img.convert("RGBA"))
    rgb, alpha = rgba[:, :, :3], rgba[:, :, 3]

    has_alpha = bool((alpha < 255).any())
    result.attributes["has_alpha"] = has_alpha
    if has_alpha:
        result.tags.append(("has-alpha", "structural", None))

    result.dhash = similarity.dhash(img)
    result.palette = similarity.palette(img)

    _detect_pixel_art(img, rgb, result)
    _detect_spritesheet(alpha, result)
    _detect_tileable(rgb, alpha, result)
    _detect_normal_map(rgb, result)
    _detect_mask(rgb, result)

    return result


def _detect_pixel_art(img: Image.Image, rgb: np.ndarray, result: ProbeResult) -> None:
    """Tag pixel art, by its block period if upscaled and by its shape if not.

    The two cases genuinely are different measurements. An upscaled grid has a
    period the autocorrelation can find, and the period is worth keeping as
    ``block_size`` because it is what a re-import would need. Native-resolution
    pixel art has a period of 1, which is unmeasurable by definition, so it is
    inferred from being small and having few colours - weaker evidence, and it
    carries a lower confidence to say so.
    """
    # A centre crop is enough: the period is global, and cropping keeps the
    # autocorrelation off an 8192 px atlas.
    crop = _centre_crop(img, 512)
    block = pixelgrid.detect(crop)

    if block is not None and _plausible_upscale(rgb, block):
        result.attributes["block_size"] = block
        result.tags.append(("pixel-art", "heuristic", 0.9))
        return

    edge = max(img.size)
    if (
        edge <= NATIVE_PIXEL_ART_MAX_EDGE
        and result.attributes["color_count"] <= NATIVE_PIXEL_ART_MAX_COLORS
    ):
        result.attributes["block_size"] = 1
        result.tags.append(("pixel-art", "heuristic", 0.6))


def _plausible_upscale(rgb: np.ndarray, block: int) -> bool:
    """Whether a detected period really is an integer upscale.

    The autocorrelation happily reports periods no upscale could have produced,
    and on real files it does. Three tests rule those out, each of which caught
    a distinct false positive in the calibration library:

    * A grid needs many blocks. A 48 px sprite reported as a 30 px grid is one
      and a half blocks across.
    * An upscale tiles the canvas exactly, so the block divides one axis without
      remainder. A 512 px image "on a 6 px grid" is 85.3 blocks wide.
    * **Every block is one flat colour.** This is the one that decides it, and
      it is a measurement rather than an inference: nearest-neighbour upscaling
      copies pixels, so within-block variation is zero by construction. A
      photographic PBR texture has a real 8 px period in its detail and passes
      both arithmetic tests; it fails this one by a mile.

    The earlier version of this compared the colour count against the implied
    logical pixel count, which reads well and does not work: the count saturates
    at :data:`MAX_COLOR_COUNT`, and a 512 px texture on a claimed 8 px grid has
    exactly 4,096 logical pixels, so the comparison was 4096 <= 4096 for every
    photograph in the library.
    """
    height, width = rgb.shape[:2]
    if block < 2 or min(width, height) // block < MIN_BLOCKS_PER_AXIS:
        return False
    if width % block and height % block:
        return False

    rows, cols = height // block, width // block
    tiles = rgb[: rows * block, : cols * block].reshape(rows, block, cols, block, 3)
    spread = tiles.max(axis=(1, 3)).astype(np.int16) - tiles.min(axis=(1, 3))
    return float(spread.mean()) <= FLAT_BLOCK_TOLERANCE


def _detect_spritesheet(alpha: np.ndarray, result: ProbeResult) -> None:
    """Tag a sheet and record its grid, from transparent gutters.

    Cell *boundaries* are tested rather than cell contents, which is what makes
    this survive real sheets: the ink in each frame is a different width, so
    content runs are not uniform and looking for uniform runs finds nothing. The
    boundaries, on the other hand, are exactly where a sheet is regular.

    A sheet packed edge to edge with no transparent gutter at all is not
    detected, and that is the accepted gap. Guessing a grid from an image with
    no separators means guessing, and a wrong ``cols`` is worse than none.
    """
    opaque = alpha >= 16
    cols = _cell_count(opaque.any(axis=0))
    rows = _cell_count(opaque.any(axis=1))

    if cols * rows < 2:
        return

    # An all-transparent image satisfies every boundary test there is, so the
    # cells themselves have to be shown to hold something.
    height, width = opaque.shape
    cell_h, cell_w = height // rows, width // cols
    filled = sum(
        bool(opaque[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w].any())
        for r in range(rows)
        for c in range(cols)
    )
    # Two filled cells minimum, always. Half the false positives in the
    # calibration library were single sprites with one empty half - a lone idle
    # frame with headroom above it splits cleanly into 1x2, and one frame is
    # not a sheet.
    if filled < max(2, (cols * rows + 1) // 2):
        return

    result.attributes["cols"] = cols
    result.attributes["rows"] = rows
    # Only when the container had nothing to say: an animated GIF that also
    # happens to be laid out in a grid has a real frame count already.
    if result.attributes.get("frame_count", 1) == 1:
        result.attributes["frame_count"] = filled
    # A two-cell split is weaker evidence than a 4x4 grid, and says so. Real
    # two-frame sheets exist (a full and an empty heart), but so do logos with
    # a gap in the middle, and nothing in the pixels tells them apart.
    result.tags.append(("spritesheet", "heuristic", 0.8 if filled >= 4 else 0.6))


def _detect_tileable(rgb: np.ndarray, alpha: np.ndarray, result: ProbeResult) -> None:
    """Tag a texture whose opposite edges join without a visible seam.

    Two ways to score a perfect seam without being a tileable texture, and both
    are common enough in a real library to need shutting out. A sprite on
    transparency has empty edges; a character atlas on a flat background has
    constant ones. Neither has *matched* its edges, and requiring the border to
    carry a comparable share of the image's own contrast excludes both.
    """
    if (alpha >= 16).mean() < OPAQUE_COVERAGE:
        return

    score = seam_score(rgb)
    if score is None or score > SEAM_THRESHOLD:
        return
    if _edge_contrast_ratio(rgb) < MIN_EDGE_CONTRAST_RATIO:
        return

    result.attributes["seam_score"] = round(score, 3)
    result.tags.append(("tileable", "heuristic", 0.8))


def _edge_contrast_ratio(rgb: np.ndarray) -> float:
    """Detail along the four borders, relative to detail across the whole image.

    Around 1.0 for a texture whose border is as busy as its middle, which is
    what a genuine tile looks like. Near 0 for a flat frame around a sprite.
    """
    arr = rgb.astype(np.float64)
    interior = (
        _mean_abs(arr[1:] - arr[:-1]) + _mean_abs(arr[:, 1:] - arr[:, :-1])
    ) / 2
    if interior <= 1e-9:
        return 0.0

    borders = (arr[0], arr[-1], arr[:, 0], arr[:, -1])
    edge = float(np.mean([_mean_abs(b[1:] - b[:-1]) for b in borders]))
    return edge / interior


def _detect_normal_map(rgb: np.ndarray, result: ProbeResult) -> None:
    """Tag a tangent-space normal map by its characteristic blue cast.

    Three conditions rather than one. A mean near (128, 128, 255) alone also
    describes a pale blue UI panel, so blue is additionally required to dominate
    almost everywhere - a normal map's whole surface points outwards - and the
    red and green means are required to sit near the middle, which a sky or a
    blue button does not do.
    """
    means = rgb.reshape(-1, 3).mean(axis=0)
    if not (means[2] > 200 and 96 < means[0] < 160 and 96 < means[1] < 160):
        return

    blue = rgb[:, :, 2].astype(np.int16)
    dominant = ((blue >= rgb[:, :, 0]) & (blue >= rgb[:, :, 1])).mean()
    if dominant < 0.9:
        return

    result.tags.append(("normal-map", "heuristic", 0.9))


def _detect_mask(rgb: np.ndarray, result: ProbeResult) -> None:
    """Tag a two-tone greyscale image: an alpha, roughness or occlusion map.

    The spec calls for "single channel *or* fully desaturated with two dominant
    values"; this requires both halves. Single-channel on its own also describes
    a black and white photograph, and a mode is not evidence about content -
    plenty of masks ship as RGB. Requiring two levels to cover almost every
    pixel is what actually separates a mask from a photograph or a gradient, and
    a single-channel image passes the desaturation half for free.
    """
    saturation = rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2)
    if saturation.mean() > 2:
        return

    counts = np.bincount(rgb[:, :, 0].ravel(), minlength=256)
    top_two = int(np.sort(counts)[-2:].sum())
    if top_two / rgb[:, :, 0].size < 0.9:
        return

    result.tags.append(("mask", "heuristic", 0.7))


def seam_score(rgb: np.ndarray) -> float | None:
    """How badly an image fails to tile, as a multiple of its own contrast.

    The wrap edge is just another pair of adjacent pixels, so the honest
    question is not "how different are the edges" - that answer scales with how
    busy the texture is - but "how different are they compared with any other
    two neighbours". Dividing by the mean interior difference is what makes one
    threshold work for a noisy rock texture and a flat gradient alike.

    ``None`` for an image with no interior contrast, where the ratio is 0/0: a
    flat fill does tile, but calling it a tileable texture is not useful.

    >>> import numpy as np
    >>> ramp = np.tile(np.arange(64, dtype=np.uint8), (64, 1))[:, :, None]
    >>> round(seam_score(np.repeat(ramp, 3, axis=2)))  # 63 -> 0 at the wrap
    63
    """
    arr = rgb.astype(np.float64)
    if arr.shape[0] < 2 or arr.shape[1] < 2:
        return None

    wrap = _mean_abs(arr[0] - arr[-1]) + _mean_abs(arr[:, 0] - arr[:, -1])
    interior = _mean_abs(arr[1:] - arr[:-1]) + _mean_abs(arr[:, 1:] - arr[:, :-1])
    if interior <= 1e-9:
        return None
    return float(wrap / interior)


def _mean_abs(diff: np.ndarray) -> float:
    return float(np.abs(diff).mean())


def _cell_count(used: np.ndarray) -> int:
    """How many cells one axis divides into, or 1 if it does not.

    Both gutter conventions are tried: the empty column can belong to the cell
    before the boundary (trailing pad) or the one after it (leading pad). The
    largest count that works wins, because if a sheet splits into eight it also
    splits into four and two, and eight is the one that is true.

    >>> import numpy as np
    >>> used = np.array([1, 1, 0, 0, 1, 1, 0, 0], dtype=bool)  # 2 cells of 4
    >>> _cell_count(used)
    2
    >>> _cell_count(np.ones(8, dtype=bool))  # no gutters anywhere
    1
    """
    length = int(used.size)
    best = 1
    for count in range(2, min(64, length // 4) + 1):
        if length % count:
            continue
        cell = length // count
        for offset in (0, -1):
            if not any(used[k * cell + offset] for k in range(1, count)):
                best = count
                break
    return best


def _centre_crop(img: Image.Image, size: int) -> Image.Image:
    """Middle ``size`` x ``size`` of an image, or the whole thing if smaller."""
    width, height = img.size
    if width <= size and height <= size:
        return img
    left = max(0, (width - size) // 2)
    top = max(0, (height - size) // 2)
    return img.crop((left, top, min(width, left + size), min(height, top + size)))


# --- formats PIL cannot open ------------------------------------------------


def _probe_exr(path: Path) -> ProbeResult:
    """Dimensions from an OpenEXR header, without decoding a single scanline.

    EXR is a float format used for HDR bakes and light maps; PIL cannot read it
    and pulling in OpenImageIO to learn two integers is not a trade worth
    making. The header is a flat list of ``name\\0type\\0size,data`` records, and
    ``dataWindow`` holds the pixel bounds.
    """
    result = ProbeResult(attributes={"bytes": path.stat().st_size})

    with path.open("rb") as fh:
        head = fh.read(4096)

    if not head.startswith(EXR_MAGIC):
        result.error = "not an OpenEXR file"
        return result

    offset = 8  # magic plus version
    while offset < len(head):
        name, offset = _read_cstring(head, offset)
        if not name:  # empty name terminates the header
            break
        kind, offset = _read_cstring(head, offset)
        (size,) = struct.unpack_from("<i", head, offset)
        offset += 4
        if name == "dataWindow" and kind == "box2i" and size == 16:
            x_min, y_min, x_max, y_max = struct.unpack_from("<4i", head, offset)
            width, height = x_max - x_min + 1, y_max - y_min + 1
            result.attributes.update(
                width=width,
                height=height,
                aspect=round(width / height, 4) if height else 0.0,
                mode="F",
                has_alpha=False,
            )
            break
        offset += size

    if "width" not in result.attributes:
        result.error = "no dataWindow in EXR header"
    return result


def _probe_aseprite(path: Path) -> ProbeResult:
    """Dimensions and frame count from the 128-byte Aseprite header.

    No thumbnail: frames are stored as compressed cel chunks that would need a
    real decoder, and an Aseprite file almost always sits beside the PNG it was
    exported to, which the scan indexes anyway.
    """
    result = ProbeResult(attributes={"bytes": path.stat().st_size})

    with path.open("rb") as fh:
        head = fh.read(16)

    if len(head) < 16:
        result.error = "truncated Aseprite header"
        return result

    _, magic, frames, width, height, depth = struct.unpack_from("<IHHHHH", head, 0)
    if magic != ASEPRITE_MAGIC:
        result.error = "not an Aseprite file"
        return result

    result.attributes.update(
        width=width,
        height=height,
        aspect=round(width / height, 4) if height else 0.0,
        mode={8: "P", 16: "LA", 32: "RGBA"}.get(depth, "RGBA"),
        has_alpha=True,
        frame_count=frames,
    )
    # An Aseprite file is a drawing, and a multi-frame one is an animation.
    result.tags.append(("pixel-art", "structural", None))
    if frames > 1:
        result.tags.append(("animation", "structural", None))
    return result


def _read_cstring(data: bytes, offset: int) -> tuple[str, int]:
    end = data.index(b"\x00", offset)
    return data[offset:end].decode("utf-8", "replace"), end + 1
