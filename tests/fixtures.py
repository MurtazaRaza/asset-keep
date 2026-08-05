"""Synthetic images for the detection tests.

Generated rather than committed as binaries, for two reasons. A checked-in PNG
is a black box - when a test fails you cannot see what changed about the input -
and each of these has a deliberate near-miss twin whose only difference is the
one property under test, which is only expressible in code.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

RNG = np.random.default_rng(20240805)


def upscaled_pixel_art(block: int = 4, cells: int = 24) -> Image.Image:
    """Random flat blocks at an integer upscale: a 4x pixel grid."""
    small = RNG.integers(0, 256, (cells, cells, 3), dtype=np.uint8)
    return Image.fromarray(small).resize(
        (cells * block, cells * block), Image.Resampling.NEAREST
    )


def organic_texture(size: int = 96) -> Image.Image:
    """The near miss for pixel art: busy and colourful, with no period.

    Built from sinusoids at deliberately non-integer frequencies plus grain, so
    the colour-change signal the detector autocorrelates has energy everywhere
    and a peak nowhere. A bicubic upscale would *not* work as this fixture: it
    blurs a grid but does not remove it, and the detector is supposed to find
    blurred grids.
    """
    axis = np.linspace(0, 1, size)
    grid_y, grid_x = np.meshgrid(axis, axis, indexing="ij")
    field = (
        np.sin(grid_x * 11.3) * np.cos(grid_y * 7.7)
        + np.sin((grid_x + grid_y) * 5.1)
        + np.cos(grid_y * 13.9 - grid_x * 3.3)
    )
    scaled = (field - field.min()) / np.ptp(field) * 200 + 20
    grain = RNG.normal(0, 6, (size, size, 3))
    stacked = np.clip(scaled[:, :, None] * [1.0, 0.85, 0.6] + grain, 0, 255)
    return Image.fromarray(stacked.astype(np.uint8), mode="RGB")


def spritesheet(cols: int = 3, rows: int = 2, cell: int = 32) -> Image.Image:
    """A grid of blobs on transparency, each a different width.

    Frames deliberately differ in size. A sheet whose every frame filled its
    cell identically would pass a detector that looked for uniform content runs,
    which real sheets do not have.
    """
    sheet = np.zeros((rows * cell, cols * cell, 4), dtype=np.uint8)
    for row in range(rows):
        for col in range(cols):
            inset = 3 + (row * cols + col) % 5
            top, left = row * cell + inset, col * cell + inset
            sheet[top : (row + 1) * cell - inset, left : (col + 1) * cell - inset] = (
                200,
                80,
                60,
                255,
            )
    return Image.fromarray(sheet, mode="RGBA")


def single_sprite(size: int = 96) -> Image.Image:
    """The near miss for a sheet: one blob with transparent margins."""
    arr = np.zeros((size, size, 4), dtype=np.uint8)
    arr[20 : size - 18, 14 : size - 25] = (90, 140, 200, 255)
    return Image.fromarray(arr, mode="RGBA")


def tiling_texture(size: int = 128) -> Image.Image:
    """A texture that genuinely wraps, built from periodic functions."""
    axis = np.arange(size) * (2 * np.pi / size)
    grid_y, grid_x = np.meshgrid(axis, axis, indexing="ij")
    field = (
        np.sin(3 * grid_x) + np.cos(2 * grid_y) + np.sin(grid_x + 2 * grid_y)
    )
    scaled = ((field - field.min()) / np.ptp(field) * 255).astype(np.uint8)
    return Image.fromarray(np.stack([scaled] * 3, axis=2), mode="RGB")


def non_tiling_texture(size: int = 128) -> Image.Image:
    """The near miss: the same texture with one axis stretched off-period."""
    axis_x = np.linspace(0, 2 * np.pi * 3.4, size)  # not a whole number of cycles
    axis_y = np.arange(size) * (2 * np.pi / size)
    grid_y, grid_x = np.meshgrid(axis_y, axis_x, indexing="ij")
    field = np.sin(3 * grid_x) + np.cos(2 * grid_y) + np.sin(grid_x + 2 * grid_y)
    scaled = ((field - field.min()) / np.ptp(field) * 255).astype(np.uint8)
    return Image.fromarray(np.stack([scaled] * 3, axis=2), mode="RGB")


def normal_map(size: int = 64) -> Image.Image:
    """Perturbed tangent-space normals: r and g near 128, b near 255."""
    arr = np.empty((size, size, 3), dtype=np.uint8)
    arr[:, :, 0] = 128 + RNG.integers(-30, 30, (size, size))
    arr[:, :, 1] = 128 + RNG.integers(-30, 30, (size, size))
    arr[:, :, 2] = 240 + RNG.integers(0, 16, (size, size))
    return Image.fromarray(arr, mode="RGB")


def blue_sky(size: int = 64) -> Image.Image:
    """The near miss for a normal map: blue-dominant, but not centred on grey."""
    arr = np.empty((size, size, 3), dtype=np.uint8)
    arr[:, :, 0] = 70 + RNG.integers(0, 20, (size, size))
    arr[:, :, 1] = 130 + RNG.integers(0, 20, (size, size))
    arr[:, :, 2] = 225 + RNG.integers(0, 20, (size, size))
    return Image.fromarray(arr, mode="RGB")


def two_tone_mask(size: int = 64) -> Image.Image:
    """A hard-edged black and white mask."""
    arr = np.zeros((size, size), dtype=np.uint8)
    arr[10:50, 12:44] = 255
    return Image.fromarray(arr, mode="L")


def greyscale_photo(size: int = 64) -> Image.Image:
    """The near miss for a mask: greyscale, but with a full tonal range."""
    axis = np.linspace(0, 1, size)
    grid_y, grid_x = np.meshgrid(axis, axis, indexing="ij")
    field = (np.sin(grid_x * 9) * np.cos(grid_y * 7) + 1) / 2
    return Image.fromarray((field * 255).astype(np.uint8), mode="L")


def periodic_photo_texture(size: int = 256, period: int = 8) -> Image.Image:
    """A photographic texture with a real repeating period but no flat blocks.

    The near miss the calibration library actually produced: PBR wall and rock
    textures whose detail repeats every few pixels, which satisfies every
    arithmetic test for an upscale while looking nothing like pixel art. Only
    measuring within-block flatness separates it.
    """
    axis = np.arange(size)
    grid_y, grid_x = np.meshgrid(axis, axis, indexing="ij")
    base = np.sin(2 * np.pi * grid_x / period) * np.cos(2 * np.pi * grid_y / period)
    detail = RNG.normal(0, 40, (size, size))
    field = np.clip((base + 1) * 90 + detail, 0, 255)
    return Image.fromarray(np.stack([field.astype(np.uint8)] * 3, axis=2), mode="RGB")


def sprite_with_empty_half(size: int = 64) -> Image.Image:
    """One sprite with headroom above it, which splits cleanly into 1x2."""
    arr = np.zeros((size, size, 4), dtype=np.uint8)
    arr[size // 2 + 2 :, 8 : size - 8] = (180, 120, 60, 255)
    return Image.fromarray(arr, mode="RGBA")


def flat_bordered_atlas(size: int = 128) -> Image.Image:
    """Busy art inside a solid frame: a perfect seam that means nothing."""
    arr = np.full((size, size, 3), 30, dtype=np.uint8)
    inner = RNG.integers(0, 255, (size - 40, size - 40, 3), dtype=np.uint8)
    arr[20 : size - 20, 20 : size - 20] = inner
    return Image.fromarray(arr, mode="RGB")


def write(image: Image.Image, path) -> "Path":  # noqa: F821 - typing only
    """Save a fixture where a probe can open it, and return the path."""
    image.save(path)
    return path
