"""Captions, from a vision model running in ollama.

The fifth and last tier, and the only one that writes prose. Everything below it
produces facts or canonical tags; this produces a sentence, which is worth
having for exactly one reason: it puts words in the index that nobody thought to
type. A file called ``prop_04.fbx`` in a folder called ``Set2`` is unreachable
by search until something looks at it and says "a wooden barrel with metal
bands", and no amount of filename tokenisation gets there.

**Opt-in, per asset or per collection, never over the library.** A caption costs
a second or two of a 1.7 GB model's attention on a machine with 8 GB of memory,
and most of a library does not need one - a sprite called ``goblin_walk`` is
already findable. So nothing enqueues captions automatically; the scan does not,
and neither does an import. Somebody asks, for a selection.

**ollama rather than another runtime.** It is already installed here, it holds
the weights in its own store, it unloads them when idle, and it is one HTTP
call. Bringing a second inference stack into this project to run one optional
feature would cost more than the feature is worth. The consequence is that this
tier needs no Python dependency at all - ``httpx`` is already a core one - which
is why there is no ``vlm`` extra to install.

**What the model is shown is the thumbnail, not the file.** 256 px is more than
moondream's vision tower keeps anyway, base64 of a 4K PNG is 20 MB over a local
socket for no gain, and for a 3D model the thumbnail is the only thing there is
to look at. The one exception is a sprite small enough that no thumbnail was
generated, where the original *is* the thumbnail.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
import time
from pathlib import Path
from typing import Callable

from ..config import Config

log = logging.getLogger(__name__)

#: ~1.7 GB quantised, which is the realistic ceiling alongside everything else
#: on an 8 GB machine. The alternatives worth naming: llava:7b is 4.7 GB and
#: swaps, qwen2.5vl:3b is 3.2 GB and leaves nothing for ComfyUI, and neither is
#: enough better at "what is this game asset" to earn the memory.
DEFAULT_MODEL = "moondream"

DEFAULT_URL = "http://127.0.0.1:11434"

#: A capability check runs on every ``/api/capabilities`` call, so it may not
#: wait on a model load. Generation may: a cold moondream takes a few seconds to
#: come off disk, and every caption after that is fast.
PROBE_TIMEOUT = 2.0
GENERATE_TIMEOUT = 180.0

#: Four words, and that is not laziness - it is what the measurement said.
#:
#: The prompt written first was "This is a game art asset. Describe what it
#: shows in one short sentence, naming the subject and its style." It is the
#: better instruction and it produces garbage: against real assets moondream
#: answered ``!!!POLYGONAL TREE!!!``, ``***************`` and ``xtremely
#: detailed and colorful checkerboard``, while the plain request produced "a 3D
#: rendering of a tree, composed entirely of geometric shapes" for the same
#: file. moondream2 is a captioner wrapped in a Question/Answer template, not an
#: instruction-following model, and a long instruction lands in the question
#: slot and derails it. Shaping the answer is this module's job, in
#: :func:`tidy`, not the prompt's.
PROMPT = "Describe this image."

#: Long enough for a sentence, short enough that a model which has started
#: listing every pixel is truncated rather than stored.
MAX_CAPTION_CHARS = 300

#: Deterministic, and capped. A caption is metadata: the same asset captioned
#: twice should not produce two different sentences, and temperature is the only
#: reason it would.
OPTIONS = {"temperature": 0.0, "num_predict": 96}

#: Openers every small VLM produces and no reader wants. Stripped rather than
#: prompted away, because a prompt that forbids them costs tokens on every call
#: and is obeyed about half the time.
#: The article after it is deliberately left in place: "shows a wooden barrel"
#: becomes "A wooden barrel", not "Wooden barrel".
_PREAMBLE = re.compile(
    r"^(the |this )?(image|picture|photo|illustration|artwork|sprite|asset)?\s*"
    r"(shows|depicts|features|displays|is|contains|presents)\s+",
    re.IGNORECASE,
)


def endpoint(config: Config) -> str:
    return (config.vlm.url or DEFAULT_URL).rstrip("/")


def model_name(config: Config) -> str:
    return config.vlm.model or DEFAULT_MODEL


def installed_models(config: Config, timeout: float = PROBE_TIMEOUT) -> list[str] | None:
    """Models ollama is holding, or ``None`` when ollama is not answering.

    The two failures are distinct and the UI needs them to stay that way: a
    server that is not running wants ``ollama serve``, and a server that is
    running without the model wants ``assetkeep vlm pull``. Collapsing both into
    "unavailable" produces the advice that fixes the other one.
    """
    import httpx

    try:
        response = httpx.get(f"{endpoint(config)}/api/tags", timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:  # noqa: BLE001 - not running is an answer
        log.debug("ollama is not answering at %s: %s", endpoint(config), exc)
        return None

    return [str(entry.get("name", "")) for entry in payload.get("models", [])]


def has_model(config: Config, models: list[str] | None = None) -> bool:
    """Whether the configured model is one ollama holds.

    Matched on the tag-less name too, since ``moondream`` and
    ``moondream:latest`` are the same weights and which one ``ollama list``
    reports depends on how it was pulled.

    >>> from assetkeep.config import Config, VlmConfig
    >>> config = Config(vlm=VlmConfig(model="moondream"))
    >>> has_model(config, ["qwen2.5:3b-instruct", "moondream:latest"])
    True
    >>> has_model(config, ["gemma2:2b"])
    False
    >>> has_model(config, None)
    False
    """
    if models is None:
        return False
    wanted = model_name(config).split(":")[0]
    return any(name.split(":")[0] == wanted for name in models)


def available(config: Config) -> bool:
    """Whether a caption can actually be produced right now."""
    return has_model(config, installed_models(config))


def status(config: Config) -> dict:
    """What ``/api/capabilities`` and ``assetkeep capabilities`` report.

    ``installed`` and ``available`` are the same fact here and are both
    reported, because the two callers ask different questions of it: the CLI
    prints whether the model is there, and the UI decides whether to show a
    button. ``server`` is the one that has to stay separate - it is the
    difference between "run ollama" and "pull the model".
    """
    models = installed_models(config)
    ready = has_model(config, models)
    return {
        "model": model_name(config),
        "url": endpoint(config),
        "server": models is not None,
        "installed": ready,
        "available": ready,
        "models": models or [],
    }


#: What transparency is composited onto before the model sees it, and the second
#: place in this project where that question was settled by measuring rather
#: than by arguing - with the opposite answer to
#: :data:`assetkeep.tagging.clip.BACKDROP`.
#:
#: On black, moondream described a UI icon atlas and a selection box as "a black
#: and white photograph of an empty room with no visible objects", twice, in
#: exactly those words: a dark-outlined sprite on black is nothing, and the
#: model says so at length. On white the same two came back as "a grid of black
#: and white icons" and "a square-shaped window". CLIP wanted black for the
#: opposite reason - it scores rather than describes, and a glow sheet on white
#: scores as nothing. Two models, two backdrops, both measured.
BACKDROP = (255, 255, 255)

#: Longest edge sent to the model. moondream resizes to 378 px internally, so
#: anything above this is base64 for nothing - and the fallback path can hand
#: over an original file rather than a 256 px tile.
MAX_EDGE = 768


#: Seconds to wait before the one retry a refused request gets. See
#: :func:`caption` for what this is actually for.
RETRY_PAUSE = 2.0


def caption(config: Config, image: Path | bytes, prompt: str = PROMPT) -> str:
    """One sentence about one image. Raises if ollama cannot answer.

    Raising rather than returning ``""`` is deliberate: this runs inside the job
    queue, and a failure that looks like an empty answer is a job marked done
    with nothing written, which is indistinguishable afterwards from an asset
    the model had nothing to say about.

    **One retry, after a pause, and it is not defensive padding.** Measured on
    the calibration library: a run of 172 captions produced 50 failures, all of
    them ``400`` from ollama, and every one of the fifty succeeded when asked
    again on its own. The cause is two AssetKeep processes draining one queue -
    a server left running has a worker thread of its own, and the CLI's drain
    joins it - so two generate requests arrive at a 1.7 GB vision model on an
    8 GB machine at once and it refuses one. With nothing else talking to
    ollama the same 172 assets fail zero times.

    The queue's own three attempts do not cover this, because all three happen
    back to back inside one drain: an overload that lasts two seconds burns the
    lot. This waits instead.
    """
    import httpx

    data = encode(image)
    body = {
        "model": model_name(config),
        "prompt": prompt,
        "images": [base64.b64encode(data).decode("ascii")],
        "stream": False,
        "options": OPTIONS,
    }
    url = f"{endpoint(config)}/api/generate"
    timeout = config.vlm.timeout or GENERATE_TIMEOUT

    response = httpx.post(url, json=body, timeout=timeout)
    if response.status_code >= 400:
        log.info(
            "ollama refused a caption (%s); retrying once in %.0fs",
            response.status_code,
            RETRY_PAUSE,
        )
        time.sleep(RETRY_PAUSE)
        response = httpx.post(url, json=body, timeout=timeout)

    if response.status_code >= 400:
        # ollama puts the reason in the body and nothing useful in the status,
        # and this string is what lands in the job's error column. A queue
        # reporting "400 Bad Request" fifty times is a mystery; one reporting
        # "image is too large" is a fix.
        raise RuntimeError(
            f"ollama returned {response.status_code}: {response.text.strip()[:300]}"
        )

    text = tidy(str(response.json().get("response", "")))
    if not text:
        raise ValueError("the model returned an empty caption")
    return text


def encode(image: Path | bytes) -> bytes:
    """The PNG actually sent to ollama.

    Re-encoded rather than uploaded as-is, and that is required rather than
    tidy: **ollama cannot decode WebP**, which is every thumbnail this tool
    produces. It answers ``400 Failed to load image or audio file``, and since
    the thumbnail is deliberately what the model is shown, without this
    conversion the entire tier fails on every asset that has one.

    Transparency is composited onto :data:`BACKDROP` here rather than left to
    ``convert("RGB")``, which keeps whatever colours happen to sit under fully
    transparent pixels and hands the model a sprite surrounded by noise.
    """
    from PIL import Image

    if isinstance(image, bytes):
        source = Image.open(io.BytesIO(image))
    else:
        source = Image.open(Path(image))

    with source:
        source.load()
        rgba = source.convert("RGBA")
        if max(rgba.size) > MAX_EDGE:
            scale = MAX_EDGE / max(rgba.size)
            rgba = rgba.resize(
                (max(1, round(rgba.width * scale)), max(1, round(rgba.height * scale))),
                Image.Resampling.LANCZOS,
            )
        flat = Image.new("RGB", rgba.size, BACKDROP)
        flat.paste(rgba, mask=rgba.split()[3])

    buffer = io.BytesIO()
    flat.save(buffer, "PNG")
    return buffer.getvalue()


def tidy(text: str) -> str:
    """Normalise a model's answer into one storable sentence.

    >>> tidy("  The image shows a  wooden barrel\\nwith metal bands.  ")
    'A wooden barrel with metal bands.'
    >>> tidy("A pixel-art goblin. It is facing left. Third sentence here.")
    'A pixel-art goblin. It is facing left.'
    >>> tidy("")
    ''
    """
    collapsed = " ".join(text.split())
    if not collapsed:
        return ""

    collapsed = _PREAMBLE.sub("", collapsed)
    # Two sentences at most. Small models tend to keep going, and the third
    # sentence is reliably about the background or a repetition of the first.
    parts = re.split(r"(?<=[.!?])\s+", collapsed)
    kept = " ".join(parts[:2]).strip()[:MAX_CAPTION_CHARS]
    return kept[:1].upper() + kept[1:] if kept else ""


def pull(
    config: Config,
    model: str | None = None,
    progress: Callable[[str, int, int], None] | None = None,
) -> str:
    """Ask ollama to fetch the model, reporting progress as it goes.

    Wrapped rather than left to ``ollama pull`` on the command line for one
    reason: the model this needs is a detail of this tool's configuration, and
    somebody who changed ``vlm.model`` should not have to know what to type.
    """
    import httpx

    name = model or model_name(config)
    with httpx.Client(timeout=None) as client:
        with client.stream(
            "POST", f"{endpoint(config)}/api/pull", json={"model": name}
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.strip():
                    continue
                try:
                    frame = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if frame.get("error"):
                    raise RuntimeError(str(frame["error"]))
                if progress is not None:
                    progress(
                        str(frame.get("status", "")),
                        int(frame.get("completed", 0) or 0),
                        int(frame.get("total", 0) or 0),
                    )
    return name
