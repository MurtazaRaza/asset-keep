"""Perceptual hashing and dominant palette: "more like this" with no model.

Both are pure numpy, both cost microseconds at scan time, and together they give
near-duplicate detection and visual similarity from the first scan, on a base
install. That is why they live in the core probe rather than the optional tier:
CLIP later *upgrades* this from visual to semantic similarity, but a library
that can only find similar images once you have downloaded a model is a library
that cannot find similar images.

dHash rather than aHash or pHash. aHash is defeated by a brightness change,
which re-exporting a sprite sheet at a different gamma will do to you. pHash
needs a DCT and buys accuracy that matters for photographs, which these are not.
dHash compares each pixel with its neighbour, so it keys on structure - and
structure is what survives a rescale, a recompression or a palette swap.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

#: 8x9 samples give 8x8 = 64 horizontal comparisons.
DHASH_SIZE = 8

#: Enough to characterise flat art without turning a photograph's gradient into
#: eight indistinguishable browns.
PALETTE_COLORS = 8

_MASK64 = (1 << 64) - 1


def dhash(image: Image.Image) -> int:
    """64-bit difference hash: is each pixel brighter than the one to its right.

    Transparency is composited over black rather than ignored, which makes the
    silhouette the dominant signal. For a library that is mostly sprites on
    transparent backgrounds that is the right bias: two goblins in different
    palettes should read as similar, and a goblin and a barrel should not.

    >>> from PIL import Image
    >>> flat = Image.new("RGB", (32, 32), (120, 120, 120))
    >>> dhash(flat)  # nothing is brighter than its neighbour
    0
    >>> ramp = Image.fromarray(np.tile(np.arange(256, dtype=np.uint8), (256, 1)))
    >>> bin(dhash(ramp)).count("1")  # every pixel brighter than its left neighbour
    64
    """
    flat = _composite(image).convert("L")
    small = flat.resize((DHASH_SIZE + 1, DHASH_SIZE), Image.Resampling.LANCZOS)
    pixels = np.asarray(small, dtype=np.int16)

    bits = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return value


def palette(image: Image.Image) -> bytes:
    """The eight dominant colours as packed RGB, most frequent first.

    Fully transparent pixels are dropped before quantising. Their RGB is
    undefined - exporters write black, white or the last drawn colour into them
    interchangeably - so including them would make the palette a fact about the
    exporter rather than about the art.

    >>> from PIL import Image
    >>> len(palette(Image.new("RGB", (8, 8), (200, 30, 40))))
    24
    >>> palette(Image.new("RGB", (8, 8), (200, 30, 40)))[:3]
    b'\\xc8\\x1e('
    """
    rgba = image.convert("RGBA")
    pixels = np.asarray(rgba).reshape(-1, 4)
    visible = pixels[pixels[:, 3] >= 16][:, :3]

    if visible.size == 0:
        return bytes(PALETTE_COLORS * 3)

    # A 1 x N strip carries the same colour statistics as the image and lets
    # PIL's adaptive quantiser do the clustering without a resize first.
    strip = Image.fromarray(visible.reshape(1, -1, 3), mode="RGB")
    quantised = strip.quantize(colors=PALETTE_COLORS, method=Image.Quantize.FASTOCTREE)

    table = quantised.getpalette() or []
    counts = sorted(quantised.getcolors() or [], key=lambda pair: -pair[0])

    out = bytearray()
    for _, index in counts[:PALETTE_COLORS]:
        out += bytes(table[index * 3 : index * 3 + 3])
    # Short palettes repeat their most common colour rather than padding with
    # black, so a two-colour sprite does not read as "mostly black" to a
    # distance function that has no way to know the tail is filler.
    while len(out) < PALETTE_COLORS * 3:
        out += out[:3] if out else b"\x00\x00\x00"
    return bytes(out[: PALETTE_COLORS * 3])


def hamming(left: int, right: int) -> int:
    """Differing bits between two dHashes.

    Accepts either the unsigned value or the signed form SQLite stores.

    >>> hamming(0b1011, 0b1001)
    1
    >>> hamming(to_signed(2**63), 0)
    1
    """
    return bin((left ^ right) & _MASK64).count("1")


def palette_distance(left: bytes, right: bytes) -> float:
    """Mean nearest-colour distance between two palettes, in 0..1.

    Symmetric and order-insensitive: two images with the same colours in a
    different frequency order are the same palette, because which colour happens
    to cover the most pixels flips with a small crop.

    >>> red = bytes([200, 30, 40] * 8)
    >>> palette_distance(red, red)
    0.0
    >>> palette_distance(bytes([0, 0, 0] * 8), bytes([255, 255, 255] * 8))
    1.0
    """
    a = np.frombuffer(left, dtype=np.uint8).reshape(-1, 3).astype(np.float64)
    b = np.frombuffer(right, dtype=np.uint8).reshape(-1, 3).astype(np.float64)

    # Pairwise euclidean distance in RGB, normalised by the diagonal of the cube.
    gaps = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    worst = float(np.linalg.norm([255.0, 255.0, 255.0]))
    mean = (gaps.min(axis=1).mean() + gaps.min(axis=0).mean()) / 2
    return float(mean / worst)


def to_signed(value: int) -> int:
    """Reinterpret a 64-bit hash as the signed integer SQLite can store.

    SQLite's INTEGER is signed 64-bit, so a dHash with the top bit set cannot go
    in as-is. Wrapping is lossless and :func:`hamming` masks the sign away
    again, so nothing downstream needs to care which form it is holding.

    >>> to_signed(2**63) == -2**63
    True
    >>> from_signed(to_signed(2**64 - 1)) == 2**64 - 1
    True
    """
    value &= _MASK64
    return value - (1 << 64) if value >> 63 else value


def from_signed(value: int) -> int:
    """Inverse of :func:`to_signed`."""
    return value & _MASK64


def _composite(image: Image.Image) -> Image.Image:
    """Flatten transparency onto black, leaving opaque images untouched."""
    if image.mode not in ("RGBA", "LA", "PA") and "transparency" not in image.info:
        return image.convert("RGB")
    rgba = image.convert("RGBA")
    backdrop = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
    return Image.alpha_composite(backdrop, rgba).convert("RGB")
