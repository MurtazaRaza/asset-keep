"""Tags derived from where a file sits and what it is called.

No model, no cost, and in practice the tier that does most of the work: asset
packs are organised by people who already sorted them into ``Characters/Enemies``
and named the file ``goblin_walk_01.png``. That information is free and sitting
right there.

The filter list is what makes this usable rather than noisy. Without dropping
version markers, export debris and bare numbers, a library ends up with ``v2``
as one of its most frequent tags, which is worse than having no tags at all
because it displaces the real ones in a frequency-sorted sidebar.
"""

from __future__ import annotations

import re
from pathlib import Path

from . import vocab

#: Tokens that appear in filenames without saying anything about the asset.
NOISE = frozenset(
    {
        "final", "finished", "copy", "copie", "export", "exported", "new", "old",
        "untitled", "default", "render", "output", "out", "temp", "tmp", "test",
        "asset", "assets", "file", "image", "img", "pic", "picture", "the", "and",
        "with", "for", "of", "png", "jpg", "jpeg", "fbx", "wav", "ogg",
    }
)

#: Folder names that describe a project's plumbing rather than its content.
NOISE_FOLDERS = frozenset(
    {"assets", "art", "resources", "content", "src", "source", "data", "files", "out"}
)

_VERSION = re.compile(r"^v\d+$")
_TRAILING_DIGITS = re.compile(r"\d+$")
_SPLIT = re.compile(r"[^A-Za-z0-9]+|(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

#: Filename and folder tags are good guesses, not facts, and CLIP will later
#: score against the same vocabulary with real confidences. Keeping these below
#: 1.0 leaves room to sort by confidence without heuristics dominating.
FILENAME_CONFIDENCE = 0.5
FOLDER_CONFIDENCE = 0.6


def tokenize(name: str) -> list[str]:
    """Content-bearing tokens in a filename, in order, deduplicated.

    >>> tokenize("goblin_walk_01.png")
    ['goblin', 'walk']
    >>> tokenize("Tree_Pine_Snow_v2_final.fbx")
    ['tree', 'pine', 'snow']
    >>> tokenize("UI_ButtonHover.png")
    ['ui', 'button', 'hover']
    >>> tokenize("00123.png")
    []
    """
    return _tokens(Path(name).stem)


def title_tags(text: str, limit: int = 6) -> list[str]:
    """The same tokenisation over free text rather than a filename.

    This is what turns a fetched page title into tags. It is the only tagger a
    reference gets without a model - there is no file to measure, no folder to
    read - and page titles are written by people to be recognised, which makes
    them better material than a filename.

    Capped, because a title is a sentence and a filename is not: without a limit
    an article headline contributes nine tags and outweighs the pack it is
    about.

    >>> title_tags("Kenney - Platformer Pack Redux")
    ['kenney', 'platformer', 'pack', 'redux']
    >>> title_tags("Free Pixel Art Tileset | itch.io")
    ['free', 'pixel', 'art', 'tileset', 'itch', 'io']
    """
    return _tokens(text)[:limit]


def _tokens(text: str) -> list[str]:
    tokens: list[str] = []

    for raw in _SPLIT.split(text):
        if not raw:
            continue
        token = vocab.canonical(raw)
        # "tile01" is the same tag as "tile"; "01" on its own is nothing.
        token = _TRAILING_DIGITS.sub("", token)
        if len(token) < 2 or token in NOISE or _VERSION.match(token):
            continue
        if token not in tokens:
            tokens.append(token)

    return tokens


def folder_tags(path: Path, root: Path, depth: int = 2) -> list[str]:
    """Parent folder names between the file and its root, nearest first.

    Stops at the root rather than walking up to ``/``, so a root inside
    ``~/Documents/Work`` does not tag every asset in it with ``documents``.

    >>> folder_tags(Path("/r/Characters/Enemies/goblin.png"), Path("/r"))
    ['enemies', 'character']
    >>> folder_tags(Path("/r/Assets/Sprites/x.png"), Path("/r"), depth=3)
    ['sprites']
    """
    tags: list[str] = []
    for parent in path.parents:
        if len(tags) >= depth:
            break
        if parent == root or root not in parent.parents:
            break
        tag = vocab.canonical(parent.name)
        if tag and tag not in NOISE_FOLDERS and tag not in tags:
            tags.append(tag)
    return tags


def tags_for(
    path: Path, root_path: Path, root_name: str = "", depth: int = 2
) -> list[tuple[str, str, float | None]]:
    """Every heuristic tag for one file, as ``(name, source, confidence)``.

    The root's name goes on as a ``source:`` namespaced tag, which is what makes
    "everything from that pack" answerable after the pack has been copied into
    three different projects and no longer shares a folder.
    """
    tags: list[tuple[str, str, float | None]] = []

    for token in tokenize(path.name):
        tags.append((token, "heuristic", FILENAME_CONFIDENCE))
    for folder in folder_tags(path, root_path, depth):
        tags.append((folder, "heuristic", FOLDER_CONFIDENCE))
    if root_name:
        tags.append((f"source:{root_name}", "structural", None))

    return tags
