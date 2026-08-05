"""The canonical tag vocabulary, and the one function that normalises into it.

Every tag entering the database passes through :func:`resolve`, so ``Pixel Art``,
``pixelart``, ``pixel_art`` and ``PixelArt`` are one tag rather than four. This
is worth doing at write time rather than query time because the sidebar shows
tags by frequency: four spellings of one concept do not merely look untidy, they
push the real tag down out of view.

The vocabulary here is also what CLIP scores against in M4. Scoring an image
against a fixed list rather than asking a model to write prose is the deliberate
choice - it yields canonical, filterable tags, and a small model does it far
better than it writes captions - which is why the list lives in the always-on
tier rather than inside the optional one.
"""

from __future__ import annotations

import re

#: Namespaces are advisory grouping for the sidebar, not part of the tag name.
#: ``source`` is applied to root names by the heuristic tagger; the other three
#: are whatever :data:`VOCABULARY` says they are.
NAMESPACES = ("type", "style", "subject", "source")

VOCABULARY: dict[str, tuple[str, ...]] = {
    # ``ui``, ``character``, ``sfx``, ``music`` and ``animation`` are additions
    # to the starting list: the heuristic and audio tiers emit them from folder
    # names and duration, and a tag nothing can canonicalise is a tag that
    # arrives in four spellings.
    "type": (
        "tileset", "character-sprite", "character", "ui", "ui-icon", "weapon",
        "prop", "vfx", "portrait", "background", "font-atlas", "normal-map",
        "concept-art", "texture", "spritesheet", "mask", "animation", "sfx",
        "music",
    ),
    "style": (
        "pixel-art", "hand-drawn", "low-poly", "flat", "realistic", "isometric",
        "top-down",
    ),
    "subject": (
        "fantasy", "sci-fi", "medieval", "modern", "nature", "dungeon", "town",
        "cave",
    ),
}

#: Spellings seen in real filenames and pack folder names, mapped onto the
#: canonical form. Keys are already normalised, so only case and separator
#: variants need no entry - :func:`canonical` handles those.
#:
#: Kept to run-together spellings, plurals and unambiguous abbreviations. The
#: tempting entries are the semantic ones - ``tile`` to ``tileset``, ``icon`` to
#: ``ui-icon`` - and they are exactly the ones that misfire, because this table
#: is applied to every token of every filename. ``char_orc.fbx`` is a character;
#: it is not a character *sprite*, and a mapping that decided otherwise would be
#: unfixable from the outside.
ALIASES: dict[str, str] = {
    "pixelart": "pixel-art",
    "sprite-sheet": "spritesheet",
    "spritesheets": "spritesheet",
    "tilesets": "tileset",
    "tilemap": "tileset",
    "normalmap": "normal-map",
    "normalmaps": "normal-map",
    "scifi": "sci-fi",
    "lowpoly": "low-poly",
    "topdown": "top-down",
    "gui": "ui",
    "hud": "ui",
    "icons": "ui-icon",
    "char": "character",
    "chars": "character",
    "characters": "character",
    "props": "prop",
    "weapons": "weapon",
    "textures": "texture",
    "backgrounds": "background",
    "bg": "background",
    "portraits": "portrait",
    "masks": "mask",
    "animations": "animation",
    "anim": "animation",
    "anims": "animation",
    "fx": "vfx",
    "effects": "vfx",
    "transparent": "has-alpha",
}

_NAMESPACE_BY_TAG = {
    tag: namespace for namespace, tags in VOCABULARY.items() for tag in tags
}

#: Split ``PixelArt`` and ``UIIcon`` before lowercasing flattens the boundary.
#: Digits deliberately do not start a boundary: ``Loft3D`` is one word, and
#: splitting after the 3 produced the tag ``loft3-d`` in the calibration library.
_CAMEL = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_SEPARATORS = re.compile(r"[^a-z0-9]+")


def canonical(name: str) -> str:
    """Normalise one tag to kebab-case and resolve any alias.

    >>> canonical("Pixel Art"), canonical("pixel_art"), canonical("PixelArt")
    ('pixel-art', 'pixel-art', 'pixel-art')
    >>> canonical("  UIIcon "), canonical("normalMap")
    ('ui-icon', 'normal-map')
    >>> canonical("goblin")
    'goblin'
    """
    text = _CAMEL.sub("-", name.strip()).lower()
    text = _SEPARATORS.sub("-", text).strip("-")
    return ALIASES.get(text, text)


def resolve(name: str) -> tuple[str, str | None]:
    """Canonical tag plus its namespace, honouring an explicit ``ns:tag`` prefix.

    An unrecognised prefix is not stripped, because ``dark:elf`` is far more
    likely to be someone's tag than a typo for a namespace, and silently eating
    half of it would be worse than keeping an odd-looking name.

    >>> resolve("source:my-starter-utils")
    ('my-starter-utils', 'source')
    >>> resolve("Tileset")
    ('tileset', 'type')
    >>> resolve("goblin")
    ('goblin', None)
    >>> resolve("dark:elf")
    ('dark-elf', None)
    """
    head, sep, tail = name.partition(":")
    if sep and canonical(head) in NAMESPACES and tail.strip():
        return canonical(tail), canonical(head)

    tag = canonical(name)
    return tag, _NAMESPACE_BY_TAG.get(tag)


def namespace_for(name: str) -> str | None:
    """Namespace of an already-canonical tag, or ``None`` if it is free-form."""
    return _NAMESPACE_BY_TAG.get(name)


def known() -> tuple[str, ...]:
    """Every tag in the curated vocabulary, in namespace order."""
    return tuple(tag for tags in VOCABULARY.values() for tag in tags)
