"""Finding the pixel grid an image was drawn on.

Lifted from AssetGeneratorHelper's post-processing, where it exists to downscale
generated art onto the grid the model actually painted. The measurement is the
same one this tool needs for a different reason: an image whose colour changes
land on a regular period is pixel art displayed at an integer upscale, and the
period is worth recording as ``block_size``.

Block boundaries are where colour changes, so the column-wise mean absolute
difference of the image is a spike train whose spikes sit on the boundaries. Its
autocorrelation peaks at the block size. The same holds row-wise, and both are
averaged because a real grid is square and noise in one axis is unlikely to be
echoed in the other.

What is scored is the *sharpness* of each peak, not its height. Autocorrelation
alone cannot be thresholded: any smooth signal correlates strongly with itself
at short lags, so a soft blob on a noisy backdrop scores 0.7 at lag 2 and would
be read as a 2 px grid. A real grid instead produces a peak that stands above
its immediate neighbours, so each lag is scored against the average of the two
beside it.

Detection is deliberately conservative: a weak peak returns ``None``. A wrong
grid is worse than no grid, because it becomes an attribute someone filters on.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

#: Minimum peak sharpness for a lag to count as a real grid. Sits in the gap
#: between grid-free images (0.13 and below) and blurred grids (0.15 and up).
PEAK_THRESHOLD = 0.2

#: A period's multiples are peaks too, and score within about 1% of the
#: fundamental, so the smallest lag scoring within this fraction of the best
#: wins. Without it a 4 px grid is reported as 8, 16 or 24 on a coin toss.
FUNDAMENTAL_RATIO = 0.9


def detect(image: Image.Image, min_block: int = 2, max_block: int = 32) -> int | None:
    """Return the dominant block size in pixels, or ``None`` if there isn't one.

    ``None`` means the image has no repeating structure at these scales: a
    photo-like texture, a smooth gradient, a flat fill, or pixel art already at
    its native resolution, where the block size is 1.

    >>> import numpy as np
    >>> from PIL import Image
    >>> art = np.random.default_rng(0).integers(0, 255, (16, 16, 3), dtype=np.uint8)
    >>> upscaled = Image.fromarray(art).resize((128, 128), Image.Resampling.NEAREST)
    >>> detect(upscaled)
    8
    >>> smooth = Image.fromarray(art).resize((128, 128), Image.Resampling.BICUBIC)
    >>> detect(smooth) is None
    True
    """
    # Lag 1 is excluded on purpose: a "grid" of one pixel is just the image, and
    # scoring it would need a neighbour at lag 0, which is always 1.0.
    if min_block < 2 or max_block < min_block:
        raise ValueError(f"bad block range {min_block}..{max_block}")

    gray = np.asarray(image.convert("L"), dtype=np.float64)
    if gray.ndim != 2 or min(gray.shape) < 4:
        return None

    signals = [
        np.abs(np.diff(gray, axis=1)).mean(axis=0),  # vertical boundaries
        np.abs(np.diff(gray, axis=0)).mean(axis=1),  # horizontal boundaries
    ]

    scores = [_peak_sharpness(s, min_block, max_block) for s in signals]
    usable = [s for s in scores if s is not None]
    if not usable:
        return None

    combined = np.mean(usable, axis=0)
    best = float(combined.max())
    if best < PEAK_THRESHOLD:
        return None

    fundamental = int(np.argmax(combined >= best * FUNDAMENTAL_RATIO))
    return min_block + fundamental


def _peak_sharpness(
    signal: np.ndarray, min_block: int, max_block: int
) -> np.ndarray | None:
    """How far the autocorrelation at each lag rises above its neighbours.

    Returns ``None`` when the signal carries no information (a flat image, or a
    gradient whose every step is identical) or is too short to measure the
    requested lags with a neighbour on each side.
    """
    if signal.size <= max_block + 2:
        return None

    centred = signal - signal.mean()
    variance = float(np.mean(centred**2))
    if variance <= 1e-12:
        return None

    # One lag of margin either side, since each score needs both neighbours.
    lags = np.arange(min_block - 1, max_block + 2)
    correlation = np.array(
        [np.mean(centred[:-k] * centred[k:]) / variance for k in lags]
    )
    return correlation[1:-1] - (correlation[:-2] + correlation[2:]) / 2
