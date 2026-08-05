"""Structural extraction, one module per kind of file.

Every probe answers the same question - what can be learned from this file
without a model - and returns the same shape, so :mod:`assetkeep.scan` never
branches on kind beyond picking which one to call.

Probes never raise on a bad file. A truncated PNG, an FBX assimp chokes on, an
audio file ffprobe does not recognise: each of those produces a result with
fewer attributes, not a scan that stops. A library that refuses to index 900
files because one of them is corrupt is not a library.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

#: Everything else is caught, including :class:`BaseException` subclasses, which
#: is not paranoia: ``impasse.errors.AssimpError`` derives from
#: ``BaseException`` directly, so a missing assimp or a malformed FBX sails
#: straight through ``except Exception`` and kills the scan. Ctrl-C and
#: ``sys.exit`` still have to get out.
CONTROL_FLOW = (KeyboardInterrupt, SystemExit, GeneratorExit)


@dataclass
class ProbeResult:
    """Everything one file yielded, ready for the scan to write."""

    #: Structural facts. Numbers go to ``attribute.value_num`` and are what
    #: ``w:>=512`` and ``tris:<5000`` search against.
    attributes: dict[str, object] = field(default_factory=dict)
    #: ``(name, source, confidence)`` triples, all with automated sources.
    tags: list[tuple[str, str, float | None]] = field(default_factory=list)
    #: Perceptual hash and palette, produced by the probe because it already has
    #: the pixels decoded and decoding twice is the expensive part.
    dhash: int | None = None
    palette: bytes | None = None
    #: An existing image to use as the thumbnail instead of rendering one.
    preview: Path | None = None
    #: Set when the probe could not read the file at all, for the UI to explain
    #: a blank tile rather than leave it mysterious.
    error: str | None = None


def probe(path: Path, kind: str, **options) -> ProbeResult:
    """Run the probe for ``kind``, degrading to an empty result on failure."""
    from . import audio, image, model3d

    runners = {"image": image.probe, "model3d": model3d.probe, "audio": audio.probe}
    runner = runners.get(kind)
    if runner is None:
        return ProbeResult()

    try:
        return runner(path, **options)
    except CONTROL_FLOW:
        raise
    except BaseException as exc:  # noqa: BLE001 - see the module docstring
        log.warning("probe failed for %s: %s", path, exc)
        return ProbeResult(error=f"{type(exc).__name__}: {exc}")
