"""CLIP: semantic search, semantic similarity, and zero-shot vocabulary tags.

The optional tier. Everything here degrades to absent rather than to broken -
:func:`available` is false without the extra installed or without the weights
downloaded, and every caller has a working path for false. Search falls back to
FTS, ``similar:`` falls back to the perceptual hash, and tagging falls back to
the heuristics, all of which shipped in M1.

**onnxruntime rather than torch.** The exported ViT-B/32 towers are 580 MB
against roughly 2 GB for a torch install, and this runs on an 8 GB machine that
also runs ComfyUI. Nothing here needs autograd, a training loop or a GPU: it
needs two matrix multiplies and a tokeniser.

**Two towers, loaded separately.** The vision tower is only needed by the
embedding job and the text tower only by search, so a server that never indexes
anything loads 242 MB of weights rather than 578. Loading them together would be
one line shorter and would put a third of a gigabyte into a process that has no
use for it.

**Fixed vocabulary, not generated prose.** Zero-shot scoring against the labels
in :data:`LABELS` produces canonical, filterable tags that mean the same thing
on every asset. A captioning model produces sentences, and sentences do not
aggregate into a sidebar. It is also what a small model is genuinely good at:
CLIP is a scorer, and asking it to score is asking for its strength.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import math
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
from PIL import Image

from ..config import Config
from . import vocab

log = logging.getLogger(__name__)

# --- weights ----------------------------------------------------------------

#: transformers.js's export of ``openai/clip-vit-base-patch32``, which is the
#: same weights every CLIP tutorial uses, already split into two ONNX graphs
#: with the projection heads attached.
REPO = "Xenova/clip-vit-base-patch32"

#: Pinned to a commit, not to ``main``. A branch that moves under a downloader
#: turns "the same command on two machines" into a coin flip, and the sha256s
#: below would start failing for a reason nobody would guess from the message.
REVISION = "d15189d7028b43f1d3e65039190477f6af591c2a"

BASE_URL = f"https://huggingface.co/{REPO}/resolve/{REVISION}"

#: Everything for one model lives in one folder, so both variants can coexist
#: and share the tokeniser rather than downloading 2 MB of it twice.
FOLDER = "clip-vit-base-patch32"

CHUNK = 1 << 20


@dataclass(frozen=True)
class Weight:
    """One file to fetch, and how to know it arrived intact."""

    remote: str
    sha256: str
    size: int

    @property
    def name(self) -> str:
        return self.remote.rsplit("/", 1)[-1]

    @property
    def url(self) -> str:
        return f"{BASE_URL}/{self.remote}"


TOKENIZER = Weight(
    "tokenizer.json",
    "f7f3b7af117d467b58374797691a6438d3e6b9e9cef800dfd5dced7f697a90cd",
    2224119,
)


@dataclass(frozen=True)
class Variant:
    """A pair of towers, and what to call the vectors they produce."""

    id: str
    label: str
    vision: Weight
    text: Weight

    @property
    def files(self) -> tuple[Weight, ...]:
        return (self.vision, self.text, TOKENIZER)

    @property
    def bytes_total(self) -> int:
        return sum(weight.size for weight in self.files)


VARIANTS: dict[str, Variant] = {
    "clip-vit-b-32-int8": Variant(
        id="clip-vit-b-32-int8",
        label="CLIP ViT-B/32 (quantised)",
        vision=Weight(
            "onnx/vision_model_quantized.onnx",
            "583fd1110a514667812fee7d684952aaf82a99b959760c8d7dca7e0ab9839299",
            89117001,
        ),
        text=Weight(
            "onnx/text_model_quantized.onnx",
            "73baab855d406190da9faa498cfedf65f15cf309f4cc7385b7b032e6d08e5c3a",
            64504507,
        ),
    ),
    "clip-vit-b-32": Variant(
        id="clip-vit-b-32",
        label="CLIP ViT-B/32",
        vision=Weight(
            "onnx/vision_model.onnx",
            "fd6e1402a588279d1723c7534d4bcba5bc0b14b47dfab0e46f8c47b8270d7d40",
            351685709,
        ),
        text=Weight(
            "onnx/text_model.onnx",
            "3f6571f5bad13a97c469c1622e1cfc4d9aef78b79fdbfcff804ca357bfada8cc",
            254058553,
        ),
    ),
}

#: Full precision by default, which is the opposite of what this was written to
#: do. The quantised pair is a quarter of the download and looked like the
#: obvious choice on an 8 GB machine, right up until it was measured: on the
#: retrieval benchmark in ``IMPLEMENTATION.md`` it scores 0.260 P@10 and 0.402
#: MRR against 0.360 and 0.645, and int8 and fp32 embeddings of the *same image*
#: agree only to a cosine of 0.94 - which sounds close and is not, in a space
#: where two unrelated images already sit at 0.74.
#:
#: The saving was 430 MB of disk, on a machine with 91 GB free, in exchange for
#: a worse index that is expensive to rebuild. ``clip_model`` still selects the
#: quantised pair for a machine where that trade goes the other way.
DEFAULT_VARIANT = "clip-vit-b-32"


def variant_for(name: str | None) -> Variant:
    """Look up a variant, falling back to the default rather than raising.

    A typo in a config file should cost a log line, not the whole optional tier.

    >>> variant_for(None).id
    'clip-vit-b-32'
    >>> variant_for("clip-vit-b-32-int8").id
    'clip-vit-b-32-int8'
    >>> variant_for("nonsense").id
    'clip-vit-b-32'
    """
    if name and name in VARIANTS:
        return VARIANTS[name]
    if name:
        log.warning("unknown clip_model %r; using %s", name, DEFAULT_VARIANT)
    return VARIANTS[DEFAULT_VARIANT]


def directory(config: Config) -> Path:
    return Path(config.models_path).expanduser() / FOLDER


def path_for(config: Config, weight: Weight) -> Path:
    return directory(config) / weight.name


def missing(config: Config, variant: Variant) -> list[Weight]:
    """Which of a variant's files are absent or the wrong length.

    Length rather than a full hash, because this runs on every capability check
    and rehashing 146 MB to answer "is it installed" would be felt. The sha256
    is verified once, at download, which is where a corrupt file comes from.
    """
    absent = []
    for weight in variant.files:
        path = path_for(config, weight)
        if not path.exists() or path.stat().st_size != weight.size:
            absent.append(weight)
    return absent


def installed(config: Config, variant: Variant | None = None) -> bool:
    return not missing(config, variant or variant_for(config.tagging.clip_model))


def deps_available() -> bool:
    """Whether the ``clip`` extra is installed, ignoring the weights.

    Checked by :func:`importlib.util.find_spec` rather than by importing.
    Importing onnxruntime costs half a second and 200 MB of resident memory,
    and this is called once per asset during a scan to decide whether to queue
    an embedding job - on a machine that may well be about to answer no.
    """
    return all(
        importlib.util.find_spec(module) is not None
        for module in ("onnxruntime", "tokenizers")
    )


def available(config: Config, variant: Variant | None = None) -> bool:
    """Whether embeddings can actually be computed right now."""
    return deps_available() and installed(config, variant)


def status(config: Config) -> dict:
    """What ``/api/capabilities`` and ``assetkeep capabilities`` report."""
    variant = variant_for(config.tagging.clip_model)
    absent = missing(config, variant)
    return {
        "model": variant.id,
        "label": variant.label,
        "deps": deps_available(),
        "weights": not absent,
        "available": deps_available() and not absent,
        "download_bytes": sum(weight.size for weight in absent),
        "path": str(directory(config)),
    }


def download(
    config: Config,
    variant: Variant | None = None,
    progress: Callable[[str, int, int], None] | None = None,
) -> list[Path]:
    """Fetch whatever is missing, verifying each file. Returns what was written.

    Each file is streamed to ``<name>.part`` and renamed only once its digest
    matches, so an interrupted download leaves no file that a later run would
    mistake for a complete one. That matters more than usual here: a truncated
    ONNX graph loads far enough to produce numbers before it fails, and numbers
    from half a model are indistinguishable from numbers.
    """
    import httpx

    variant = variant or variant_for(config.tagging.clip_model)
    target = directory(config)
    target.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    with httpx.Client(follow_redirects=True, timeout=60.0) as client:
        for weight in missing(config, variant):
            destination = target / weight.name
            partial = destination.with_suffix(destination.suffix + ".part")
            digest = hashlib.sha256()
            done = 0

            with client.stream("GET", weight.url) as response:
                response.raise_for_status()
                with partial.open("wb") as handle:
                    for chunk in response.iter_bytes(CHUNK):
                        handle.write(chunk)
                        digest.update(chunk)
                        done += len(chunk)
                        if progress is not None:
                            progress(weight.name, done, weight.size)

            if digest.hexdigest() != weight.sha256:
                partial.unlink(missing_ok=True)
                raise ValueError(
                    f"{weight.name} downloaded corrupt: expected sha256 "
                    f"{weight.sha256}, got {digest.hexdigest()}"
                )
            os.replace(partial, destination)
            written.append(destination)

    return written


# --- preprocessing ----------------------------------------------------------

#: From the model's own ``preprocessor_config.json``. Wrong values here do not
#: fail, they quietly degrade every score in the library, which is the worst
#: available failure mode and the reason they are pinned next to the weights
#: they belong to.
IMAGE_SIZE = 224
IMAGE_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
IMAGE_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

#: What transparency is composited onto, and a reminder that this kind of
#: question does not answer to argument.
#:
#: The reasoning for mid grey was that black loses a dark-outlined sprite's
#: outline, which is most 2D game art, and white erases a glow or a spark sheet,
#: which is most VFX - so grey keeps both readable. It sounds right. Measured on
#: the retrieval benchmark in ``IMPLEMENTATION.md``, grey came last of the three
#: at full precision (0.330 P@10) behind white (0.350) and black (0.360), and
#: last again quantised. Black wins on both metrics at both precisions.
#:
#: It is also what :mod:`assetkeep.similarity` already composites dHash onto, so
#: the two similarity measures now at least agree about what a sprite looks like.
BACKDROP = (0, 0, 0)

#: The whole image is letterboxed into the square rather than centre-cropped.
#:
#: CLIP's own preprocessing resizes the short edge to 224 and crops the middle
#: out, which is right for photographs and wrong for this library: a 1024x256
#: sprite strip loses three quarters of its frames, and a tall UI atlas loses
#: its top and bottom rows. What is being classified here is the sheet, not a
#: subject within a scene, so the sheet has to survive.
LETTERBOX = True


def preprocess(image: Image.Image) -> np.ndarray:
    """One PIL image to the ``(3, 224, 224)`` float32 tensor the tower wants."""
    flat = _composite(image)
    square = _letterbox(flat) if LETTERBOX else _centre_crop(flat)

    pixels = np.asarray(square, dtype=np.float32) / 255.0
    pixels = (pixels - IMAGE_MEAN) / IMAGE_STD
    return np.transpose(pixels, (2, 0, 1))


def preprocess_all(images: Iterable[Image.Image]) -> np.ndarray:
    batch = [preprocess(image) for image in images]
    if not batch:
        return np.empty((0, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    return np.stack(batch).astype(np.float32)


def _composite(image: Image.Image) -> Image.Image:
    """Flatten onto :data:`BACKDROP`, and handle the modes Pillow hands back.

    >>> _composite(Image.new("RGBA", (2, 2), (255, 0, 0, 0))).getpixel((0, 0))
    (0, 0, 0)
    >>> _composite(Image.new("RGB", (2, 2), (10, 20, 30))).getpixel((0, 0))
    (10, 20, 30)
    """
    if image.mode not in ("RGBA", "LA", "PA") and "transparency" not in image.info:
        return image.convert("RGB")
    rgba = image.convert("RGBA")
    backdrop = Image.new("RGBA", rgba.size, (*BACKDROP, 255))
    return Image.alpha_composite(backdrop, rgba).convert("RGB")


def _letterbox(image: Image.Image) -> Image.Image:
    """Fit the whole image into a 224 square, padding with the backdrop.

    >>> _letterbox(Image.new("RGB", (400, 100))).size
    (224, 224)
    >>> _letterbox(Image.new("RGB", (8, 8))).size
    (224, 224)
    """
    width, height = image.size
    scale = IMAGE_SIZE / max(width, height)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized = image.resize(size, Image.Resampling.BICUBIC)

    canvas = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), BACKDROP)
    canvas.paste(resized, ((IMAGE_SIZE - size[0]) // 2, (IMAGE_SIZE - size[1]) // 2))
    return canvas


def _centre_crop(image: Image.Image) -> Image.Image:
    """CLIP's own preprocessing, kept for the comparison that chose the other.

    >>> _centre_crop(Image.new("RGB", (400, 100))).size
    (224, 224)
    """
    width, height = image.size
    scale = IMAGE_SIZE / min(width, height)
    resized = image.resize(
        (max(IMAGE_SIZE, round(width * scale)), max(IMAGE_SIZE, round(height * scale))),
        Image.Resampling.BICUBIC,
    )
    left = (resized.width - IMAGE_SIZE) // 2
    top = (resized.height - IMAGE_SIZE) // 2
    return resized.crop((left, top, left + IMAGE_SIZE, top + IMAGE_SIZE))


# --- the model ---------------------------------------------------------------

#: CLIP's tokeniser context. Padded to the full width rather than to the longest
#: string in the batch, because the exported graph pools at the end-of-text
#: position and a shorter window is not a shape it was traced with.
CONTEXT = 77

#: ``<|endoftext|>``, which CLIP uses as both its terminator and its padding.
PAD_TOKEN = 49407

#: CLIP's learned temperature, ``exp(logit_scale)``, rounded to the value the
#: released checkpoints converge on. Turns cosines into the logits a softmax
#: over the vocabulary expects.
LOGIT_SCALE = 100.0

#: Query strings whose embedding is remembered. Small on purpose: this exists
#: for backspacing over a word, not as a search cache.
QUERY_CACHE = 64


class Encoder:
    """Both towers, each loaded the first time something asks for it.

    Sessions are created lazily and then kept, because loading is the expensive
    part - about a second for the quantised vision tower - and a search box
    encoding one short string per keystroke cannot pay that twice.
    """

    def __init__(self, config: Config, variant: Variant | None = None) -> None:
        self.config = config
        self.variant = variant or variant_for(config.tagging.clip_model)
        self._vision = None
        self._text = None
        self._tokenizer = None
        self._queries: dict[str, np.ndarray] = {}

    @property
    def model_id(self) -> str:
        """What goes in ``embedding.model``, so a variant swap is visible."""
        return self.variant.id

    # -- sessions ------------------------------------------------------------

    def _session(self, weight: Weight):
        import onnxruntime

        path = path_for(self.config, weight)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing; run `assetkeep model download` first"
            )

        options = onnxruntime.SessionOptions()
        options.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        # Named explicitly so onnxruntime does not spend startup probing for
        # accelerators that are not here, and does not warn about it either.
        return onnxruntime.InferenceSession(
            str(path), options, providers=["CPUExecutionProvider"]
        )

    @property
    def vision(self):
        if self._vision is None:
            self._vision = self._session(self.variant.vision)
        return self._vision

    @property
    def text(self):
        if self._text is None:
            self._text = self._session(self.variant.text)
        return self._text

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from tokenizers import Tokenizer

            self._tokenizer = Tokenizer.from_file(
                str(path_for(self.config, TOKENIZER))
            )
        return self._tokenizer

    # -- encoding ------------------------------------------------------------

    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        """Embed a batch of images. Returns ``(n, 512)``, L2-normalised."""
        if not images:
            return np.empty((0, 512), dtype=np.float32)
        return self.encode_pixels(preprocess_all(images))

    def encode_pixels(self, pixels: np.ndarray) -> np.ndarray:
        """Embed an already-preprocessed ``(n, 3, 224, 224)`` batch.

        Separate from :meth:`encode_images` so a caller holding sixteen 4K PNGs
        can preprocess and release them one at a time. Decoded, those are a
        quarter of a gigabyte held at once for no reason; as tensors they are
        ten megabytes.
        """
        if pixels.size == 0:
            return np.empty((0, 512), dtype=np.float32)

        name = self.vision.get_inputs()[0].name
        output = self.vision.run(None, {name: np.ascontiguousarray(pixels, np.float32)})[0]
        return _unit(np.asarray(output, dtype=np.float32))

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Embed a batch of strings. Returns ``(n, 512)``, L2-normalised."""
        if not texts:
            return np.empty((0, 512), dtype=np.float32)

        ids, mask = self._tokenize(texts)
        feed = {"input_ids": ids, "attention_mask": mask}
        names = {entry.name for entry in self.text.get_inputs()}
        output = self.text.run(None, {k: v for k, v in feed.items() if k in names})[0]
        return _unit(np.asarray(output, dtype=np.float32))

    def encode_query(self, text: str) -> np.ndarray:
        """One search string, remembering the last few.

        The search box is debounced but still re-queries as you type, and
        backspacing over a word asks for a string that was just encoded. The
        cache is small because the useful hits are the immediate ones.
        """
        cached = self._queries.get(text)
        if cached is None:
            cached = self.encode_texts([text])[0]
            if len(self._queries) >= QUERY_CACHE:
                self._queries.clear()
            self._queries[text] = cached
        return cached

    def _tokenize(self, texts: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        ids = np.full((len(texts), CONTEXT), PAD_TOKEN, dtype=np.int64)
        mask = np.zeros((len(texts), CONTEXT), dtype=np.int64)

        for row, encoding in enumerate(self.tokenizer.encode_batch(list(texts))):
            tokens = encoding.ids[:CONTEXT]
            ids[row, : len(tokens)] = tokens
            mask[row, : len(tokens)] = 1
        return ids, mask


def _unit(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.where(norms == 0, 1.0, norms)


# --- zero-shot vocabulary ----------------------------------------------------

#: Descriptions, not tag names. ``ui-icon`` is a slug and means nothing to a
#: text encoder; "a small icon for a game user interface" means what the tag is
#: supposed to mean. Each entry is scored as a small ensemble and averaged,
#: which is worth roughly two points of accuracy over any single phrasing and
#: costs nothing at query time because the result is cached.
#:
#: Only visual concepts are here. ``sfx`` and ``music`` are in the vocabulary
#: and are not scoreable by a model that has never heard anything, and offering
#: them would guarantee that every waveform got one.
LABELS: dict[str, tuple[str, ...]] = {
    # type
    "tileset": (
        "a tileset of terrain tiles for a 2d game",
        "a sheet of square map tiles that join together",
    ),
    "spritesheet": (
        "a sprite sheet of animation frames laid out in a grid",
        "several frames of the same character in a row",
    ),
    "character-sprite": (
        "a 2d game character sprite on a transparent background",
        "a single cartoon character standing",
    ),
    "portrait": (
        "a portrait of a face, head and shoulders",
        "a character portrait for a dialogue box",
    ),
    "ui-icon": (
        "a small icon for a game user interface",
        "a simple symbol on a button",
    ),
    "weapon": (
        "a sword, axe, bow or gun",
        "a weapon item from a game inventory",
    ),
    "prop": (
        "a barrel, crate, chest or other small object",
        "a piece of scenery furniture",
    ),
    "vfx": (
        "a glowing magic effect, explosion or spark",
        "a particle effect sprite on a dark background",
    ),
    "background": (
        "a wide scenic background landscape for a game level",
        "a painted backdrop of sky and hills",
    ),
    "font-atlas": (
        "a bitmap font sheet of letters and numbers",
        "the alphabet laid out as glyphs",
    ),
    "normal-map": (
        "a normal map texture, mostly light purple and blue",
        "a bump map showing surface direction rather than colour",
    ),
    "texture": (
        "a seamless material texture of stone, wood, grass or metal",
        "a repeating surface pattern",
    ),
    "concept-art": (
        "a painted concept art illustration",
        "a detailed digital painting of a scene",
    ),
    "mask": (
        "a black and white alpha mask",
        "a silhouette in white on black",
    ),
    # style
    "pixel-art": (
        "pixel art with visible square pixels",
        "low resolution 8-bit game graphics",
    ),
    "hand-drawn": (
        "a hand drawn illustration with visible brush strokes",
        "a sketchy inked drawing",
    ),
    "low-poly": (
        "a low polygon 3d model with flat faceted surfaces",
        "a simple untextured 3d render",
    ),
    "flat": (
        "flat vector graphics with solid colours and no shading",
        "a minimal flat design illustration",
    ),
    "realistic": (
        "a photorealistic rendering with detailed lighting",
        "a realistic photograph",
    ),
    "isometric": (
        "an isometric view at a 45 degree angle",
        "isometric game art seen from above at an angle",
    ),
    "top-down": (
        "a top down view seen from directly above",
        "an overhead orthographic view",
    ),
    # subject
    "fantasy": (
        "a fantasy setting with wizards, elves and castles",
        "medieval fantasy adventure art",
    ),
    "sci-fi": (
        "a science fiction setting with spaceships and robots",
        "futuristic technology and neon",
    ),
    "medieval": (
        "a medieval setting with knights and stone castles",
        "swords and armour from the middle ages",
    ),
    "modern": (
        "a modern day setting with cars and city streets",
        "contemporary everyday objects",
    ),
    "nature": (
        "trees, grass, rocks and plants outdoors",
        "a natural forest or meadow",
    ),
    "dungeon": (
        "a dark stone dungeon interior with torches",
        "an underground crypt of brick and bone",
    ),
    "town": (
        "a town of houses, shops and streets",
        "a village with wooden buildings",
    ),
    "cave": (
        "a rocky cave interior",
        "an underground cavern of stone",
    ),
}

#: Wrapped around each description before encoding, then averaged. Kept short
#: because the descriptions are already sentences; the usual "a photo of a {}"
#: ensemble exists to give bare class names a context, and these have one.
TEMPLATES = ("{}", "a game asset: {}")

#: Scored independently, and the winner of each is a separate question. An asset
#: is one type, in one style, of one subject, and forcing those into a single
#: softmax makes ``pixel-art`` compete with ``tileset`` for the same slot when
#: the truthful answer is both.
GROUPS = ("type", "style", "subject")

#: Descriptions that stand for "none of the above". Scored alongside the real
#: labels and never emitted, so the softmax has somewhere to put its mass when
#: an asset is not any of them.
#:
#: This is the single change that made zero-shot tagging usable here, and the
#: numbers are not close. A softmax over eight subjects has to pick one, so it
#: picked: ``dungeon`` landed on 146 of 708 assets, including a selection box
#: and a roughness map, because "a dark stone dungeon" was the least wrong of
#: eight settings for a grey rectangle. With somewhere else to put the mass,
#: ``dungeon`` falls to 13 and the whole namespace only speaks when it has
#: something to say. It sharpens ``type`` as well: measured against the
#: blue-dominance normal-map detector, which uses no model at all, false
#: positives fall from 21 to 11 while agreement holds at 31 of 33.
BACKGROUND: dict[str, tuple[str, ...]] = {
    "type": (
        "a plain photograph of nothing in particular",
        "an abstract pattern of colour",
        "a screenshot of text",
    ),
    "style": (
        "a plain photograph",
        "an abstract pattern of colour",
    ),
    "subject": (
        "an abstract image with no setting",
        "a plain object on a blank background",
        "a user interface element",
        "a flat colour swatch",
    ),
}


def labels_in(namespace: str) -> tuple[str, ...]:
    """The scoreable tags of one namespace, in a stable order.

    >>> labels_in("style")[:2]
    ('pixel-art', 'hand-drawn')
    >>> "sfx" in labels_in("type")   # nothing to look at
    False
    """
    return tuple(
        tag for tag in LABELS if vocab.namespace_for(tag) == namespace
    )


def labels_fingerprint() -> str:
    """Identity of the current label set, so an edit invalidates the cache.

    >>> len(labels_fingerprint())
    16
    """
    payload = json.dumps([LABELS, BACKGROUND, TEMPLATES], sort_keys=True).encode()
    return hashlib.blake2b(payload, digest_size=8).hexdigest()


def label_names() -> tuple[str, ...]:
    """Every row of the label matrix, in order.

    Real tags first, then one ``!namespace/n`` sentinel per background prompt.
    The prefix cannot collide with a tag, because every tag has been through
    :func:`assetkeep.tagging.vocab.canonical`, which strips punctuation.

    >>> label_names()[:1]
    ('tileset',)
    >>> [n for n in label_names() if n.startswith("!")][:2]
    ['!type/0', '!type/1']
    """
    background = tuple(
        f"!{namespace}/{position}"
        for namespace in GROUPS
        for position in range(len(BACKGROUND.get(namespace, ())))
    )
    return tuple(LABELS) + background


def _descriptions_for(name: str) -> tuple[str, ...]:
    """The phrases one label row is the average of.

    >>> _descriptions_for("!style/1")
    ('an abstract pattern of colour',)
    """
    if name in LABELS:
        return LABELS[name]
    namespace, _, position = name[1:].partition("/")
    return (BACKGROUND[namespace][int(position)],)


def label_matrix(encoder: Encoder) -> tuple[tuple[str, ...], np.ndarray]:
    """Text embeddings for every scoreable tag, computed once and cached.

    Cached to disk next to the weights rather than recomputed per run: it is
    seventy-odd forward passes through the text tower, about a second, and it is
    the same second every single time the tagger starts. The filename carries
    both the variant and a fingerprint of the labels, so editing :data:`LABELS`
    or switching models produces a different file rather than a stale one.
    """
    names = label_names()
    cache = directory(encoder.config) / (
        f"labels-{encoder.model_id}-{labels_fingerprint()}.npy"
    )

    if cache.exists():
        matrix = np.load(cache)
        if matrix.shape[0] == len(names):
            return names, matrix

    rows = [_embed_descriptions(encoder, _descriptions_for(name)) for name in names]

    matrix = np.stack(rows).astype(np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, matrix)
    return names, matrix


def _embed_descriptions(encoder: Encoder, descriptions) -> np.ndarray:
    prompts = [
        template.format(description)
        for description in descriptions
        for template in TEMPLATES
    ]
    return _unit(encoder.encode_texts(prompts).mean(axis=0))


def classify(
    names: Sequence[str],
    matrix: np.ndarray,
    vectors: np.ndarray,
    threshold: float,
) -> list[list[tuple[str, float]]]:
    """Zero-shot tags for a batch of image vectors, one list per image.

    Returns ``(tag, confidence)`` where confidence is a probability *within its
    namespace*, and that is the whole point of doing it this way.

    The obvious implementation thresholds the raw cosine, and it does not work.
    Image-text cosines here run from about 0.20 to 0.30 with no gap anywhere,
    and where in that band a label sits is mostly a fact about the label: "a
    repeating surface pattern" scores higher against everything than "a
    silhouette in white on black" does, so one fixed cutoff either admits
    texture onto the whole library or admits mask onto none of it. A softmax
    within the namespace asks the comparable question - which of these did it
    prefer, and by how much - and that answer does transfer across labels.

    The background rows are in the softmax and never in the output. A namespace
    whose winner is one of them produces no tag at all, which is how an asset
    gets to be none of the available answers.

    >>> names = ("tileset", "vfx", "pixel-art", "!type/0")
    >>> matrix = np.eye(4, dtype=np.float32)
    >>> confident = np.array([[0.9, 0.1, 0.42, 0.1]], dtype=np.float32)
    >>> classify(names, matrix, confident, 0.5)
    [[('tileset', 1.0), ('pixel-art', 1.0)]]
    >>> torn = np.array([[0.5, 0.5, 0.42, 0.1]], dtype=np.float32)
    >>> [tag for tag, _ in classify(names, matrix, torn, 0.9)[0]]
    ['pixel-art']
    >>> neither = np.array([[0.1, 0.1, 0.42, 0.9]], dtype=np.float32)
    >>> [tag for tag, _ in classify(names, matrix, neither, 0.5)[0]]
    ['pixel-art']
    """
    if vectors.size == 0 or matrix.size == 0:
        return [[] for _ in range(len(vectors))]

    index = {name: position for position, name in enumerate(names)}
    logits = LOGIT_SCALE * (np.asarray(vectors, dtype=np.float32) @ matrix.T)

    # Resolved once rather than per image: which rows of each namespace this
    # matrix carries, and where their columns are. Background rows sit at the
    # end of each group, so a winning index past `real` means "none of these".
    groups = []
    for namespace in GROUPS:
        tags = [tag for tag in labels_in(namespace) if tag in index]
        if not tags:
            continue
        background = [
            name
            for name in names
            if name.startswith(f"!{namespace}/") and name in index
        ]
        groups.append((tags, [index[name] for name in tags + background]))

    out: list[list[tuple[str, float]]] = []
    for row in logits:
        found: list[tuple[str, float]] = []
        for tags, columns in groups:
            probabilities = _softmax(row[columns])
            best = int(np.argmax(probabilities))
            confidence = float(probabilities[best])
            if best < len(tags) and confidence >= threshold:
                found.append((tags[best], round(confidence, 4)))
        out.append(found)
    return out


def _softmax(values: np.ndarray) -> np.ndarray:
    """Softmax that cannot overflow.

    >>> _softmax(np.array([1.0, 1.0])).tolist()
    [0.5, 0.5]
    >>> float(_softmax(np.array([800.0, 0.0]))[0])
    1.0
    """
    shifted = np.exp(np.asarray(values, dtype=np.float64) - np.max(values))
    return shifted / shifted.sum()


def tags_for(
    encoder: Encoder,
    vectors: np.ndarray,
    threshold: float | None = None,
) -> list[list[tuple[str, str, float]]]:
    """Zero-shot tags as the ``(name, source, confidence)`` triples db wants."""
    if threshold is None:
        threshold = encoder.config.tagging.clip_threshold
    names, matrix = label_matrix(encoder)
    return [
        [(tag, "clip", confidence) for tag, confidence in row]
        for row in classify(names, matrix, vectors, threshold)
    ]


# --- the two things the rest of the program asks for -------------------------

_shared: tuple[str, Encoder] | None = None
_shared_lock = threading.Lock()


def shared_encoder(config: Config) -> Encoder:
    """One encoder per process, keyed on which variant it holds.

    Both callers want the same object for the same reason: constructing one is
    free, but the first call through either tower loads and optimises an ONNX
    graph, which is about a second. The job worker would otherwise pay that per
    batch and the search box per keystroke.
    """
    global _shared
    with _shared_lock:
        wanted = variant_for(config.tagging.clip_model).id
        if _shared is None or _shared[0] != wanted:
            _shared = (wanted, Encoder(config))
        return _shared[1]


def ranker(config: Config, conn) -> Callable[[str], list[tuple[int, float]]] | None:
    """The semantic pass :func:`assetkeep.search.compile_query` takes, or None.

    None whenever the answer would be an empty ranking anyway - no extra
    installed, no weights, or a library nothing has embedded yet - because
    "there is no semantic pass" and "the semantic pass found nothing" have to
    stay distinguishable. The first falls back to plain FTS, which works; the
    second would be a text search that silently returned fewer results than it
    used to.
    """
    from .. import search, vectors

    if not available(config):
        return None
    model = vectors.primary_model(conn)
    if model is None:
        return None

    encoder = shared_encoder(config)

    def rank(text: str) -> list[tuple[int, float]]:
        return vectors.rank(
            conn,
            model,
            encoder.encode_query(text),
            limit=search.SEMANTIC_LIMIT,
            floor=search.SEMANTIC_FLOOR,
        )

    return rank


def describe_download(variant: Variant, absent: Sequence[Weight]) -> str:
    """The one line a CLI prints before spending several minutes of somebody's
    bandwidth.

    >>> describe_download(VARIANTS["clip-vit-b-32-int8"],
    ...                   VARIANTS["clip-vit-b-32-int8"].files)
    'CLIP ViT-B/32 (quantised): 3 files, 149 MB'
    """
    total = sum(weight.size for weight in absent)
    return (
        f"{variant.label}: {len(absent)} file{'s' if len(absent) != 1 else ''}, "
        f"{math.ceil(total / 1024 / 1024)} MB"
    )
