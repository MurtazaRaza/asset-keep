"""Structural metadata for audio. No waveform analysis, no fingerprinting.

Deliberately shallow. BPM detection and audio fingerprinting are real
techniques, and neither answers a question anyone asks of an asset library: you
look for "the short metallic hit" or "the town theme", not for something at 128
BPM. Duration, channels and sample rate cover the questions that do get asked,
and cost a header read.

WAV goes through the stdlib :mod:`wave` module, which matters more than it
sounds: WAV is the bulk of a game project's audio, and parsing it in-process
avoids one subprocess spawn per file. Everything else goes to ``ffprobe``, which
is already installed here for other reasons.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import wave
from pathlib import Path

from . import ProbeResult

#: Under this, a clip is a sound effect. A footstep, a UI click, a sword hit.
SFX_MAX_SECONDS = 2.0

#: Over this, it is music or ambience. Nothing in a game project is a 45-second
#: sound effect.
MUSIC_MIN_SECONDS = 30.0

#: How long to let ffprobe think before giving up on one file.
FFPROBE_TIMEOUT = 15


def probe(path: Path, **_options) -> ProbeResult:
    """Duration, format and a duration-derived tag for one audio file."""
    if path.suffix.lower() == ".wav":
        result = _probe_wave(path)
        # A WAV can hold compressed data the stdlib module refuses to read,
        # which is what the fallback is for rather than a hard failure.
        if result.error is None:
            _tag_by_duration(result)
            return result

    result = _probe_ffprobe(path)
    _tag_by_duration(result)
    return result


def available() -> bool:
    """Whether non-WAV audio can be read at all."""
    return shutil.which("ffprobe") is not None


def _probe_wave(path: Path) -> ProbeResult:
    result = ProbeResult(attributes={"bytes": path.stat().st_size})
    try:
        with wave.open(str(path), "rb") as handle:
            frames = handle.getnframes()
            rate = handle.getframerate()
            result.attributes.update(
                duration=round(frames / rate, 3) if rate else 0.0,
                sample_rate=rate,
                channels=handle.getnchannels(),
                bit_depth=handle.getsampwidth() * 8,
                codec="pcm",
            )
    except (wave.Error, EOFError, OSError) as exc:
        result.error = f"{type(exc).__name__}: {exc}"
    return result


def _probe_ffprobe(path: Path) -> ProbeResult:
    result = ProbeResult(attributes={"bytes": path.stat().st_size})

    if not available():
        result.error = "ffprobe not installed"
        return result

    try:
        completed = subprocess.run(
            [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_streams", "-show_format", str(path),
            ],
            capture_output=True,
            timeout=FFPROBE_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        return result

    if completed.returncode != 0:
        result.error = "ffprobe could not read the file"
        return result

    payload = json.loads(completed.stdout or "{}")
    stream = next(
        (s for s in payload.get("streams", []) if s.get("codec_type") == "audio"), None
    )
    if stream is None:
        result.error = "no audio stream"
        return result

    # Duration lives on the stream for some containers and only on the format
    # for others; OGG in particular routinely omits the stream-level value.
    duration = stream.get("duration") or payload.get("format", {}).get("duration")
    result.attributes.update(
        duration=round(float(duration), 3) if duration else None,
        sample_rate=int(stream["sample_rate"]) if stream.get("sample_rate") else None,
        channels=stream.get("channels"),
        bit_depth=int(stream["bits_per_raw_sample"])
        if stream.get("bits_per_raw_sample")
        else None,
        codec=stream.get("codec_name"),
    )
    return result


def _tag_by_duration(result: ProbeResult) -> None:
    """Split sound effects from music, and leave the middle alone.

    Duration is the one dimension that reliably separates categories in a game
    project. The gap between the two thresholds is left untagged on purpose:
    a 10-second clip is a stinger, a long ambience loop or a voice line, and
    guessing between them would put a wrong tag on the files hardest to find.
    """
    duration = result.attributes.get("duration")
    if not isinstance(duration, (int, float)):
        return
    if duration <= SFX_MAX_SECONDS:
        result.tags.append(("sfx", "heuristic", 0.7))
    elif duration >= MUSIC_MIN_SECONDS:
        result.tags.append(("music", "heuristic", 0.7))
