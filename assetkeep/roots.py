"""Walking a root: which files count, and which folders are never entered.

Two rules do almost all the work here.

**Extensions are an allowlist, not a blocklist.** Anything unlisted is ignored.
In the calibration project that is what keeps 1,243 ``.cs`` files and roughly
2,800 Unity-native YAML files out of the index without a single exclude rule,
and it means a new kind of junk file appearing in a project cannot quietly
pollute the library.

**Symlinks are never followed.** Asset folders cross-link constantly - a shared
pack linked into three projects - and a cycle turns a scan into an infinite
walk. Skipping them also stops one file being counted as two locations, which
would be wrong rather than merely slow.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterator

IMAGE = frozenset(
    {
        ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tga",
        ".tif", ".tiff", ".psd", ".exr", ".aseprite", ".ase",
    }
)
MODEL = frozenset({".fbx", ".gltf", ".glb", ".obj", ".stl", ".ply", ".dae", ".blend"})
AUDIO = frozenset({".wav", ".ogg", ".mp3", ".flac", ".aiff", ".aif"})

KNOWN_EXTENSIONS = IMAGE | MODEL | AUDIO

_KIND_BY_EXTENSION = {
    **{ext: "image" for ext in IMAGE},
    **{ext: "model3d" for ext in MODEL},
    **{ext: "audio" for ext in AUDIO},
}


def kind_for(path: Path) -> str | None:
    """Which probe handles this file, or ``None`` if it is not ours.

    >>> kind_for(Path("goblin_walk_01.PNG"))
    'image'
    >>> kind_for(Path("chest.fbx")), kind_for(Path("hit.ogg"))
    ('model3d', 'audio')
    >>> kind_for(Path("PlayerController.cs")) is None
    True
    """
    return _KIND_BY_EXTENSION.get(path.suffix.lower())


class Excluder:
    """Glob exclusion for one root, compiled once and reused per file.

    A pattern containing no ``/`` matches basenames anywhere in the tree, which
    is what makes ``*.meta`` and ``.DS_Store`` behave the way anyone writing
    them expects. Anything else matches the path relative to the root.

    >>> ex = Excluder(["**/Library/**", "*.meta", "Editor/*.png"])
    >>> ex.excludes_file("Library/metadata/00/x.png")
    True
    >>> ex.excludes_file("Art/hero.png.meta"), ex.excludes_file("Art/hero.png")
    (True, False)
    >>> ex.excludes_file("Editor/icon.png"), ex.excludes_file("Art/Editor/icon.png")
    (True, False)
    """

    def __init__(self, patterns) -> None:
        self._names: list[re.Pattern[str]] = []
        self._paths: list[re.Pattern[str]] = []
        for pattern in patterns:
            target = self._paths if "/" in pattern else self._names
            target.append(_translate(pattern))

    def excludes_file(self, rel: str) -> bool:
        name = rel.rsplit("/", 1)[-1]
        return any(p.match(name) for p in self._names) or any(
            p.match(rel) for p in self._paths
        )

    def excludes_dir(self, rel: str) -> bool:
        """Whether to prune a directory outright rather than descend into it.

        The trailing slash is what makes ``**/Library/**`` prune ``Library``
        itself: the pattern needs something after the folder name to match, and
        an empty tail satisfies it. Pruning rather than filtering per file is
        the difference between skipping a Unity ``Library/`` and stat-ing six
        figures of cached intermediates to reject them one at a time.
        """
        name = rel.rsplit("/", 1)[-1]
        if any(p.match(name) for p in self._names):
            return True
        return any(p.match(rel) or p.match(rel + "/") for p in self._paths)


def walk(
    root: Path, recursive: bool = True, excluder: Excluder | None = None
) -> Iterator[Path]:
    """Yield indexable files under ``root``, deepest-first within each folder.

    Unreadable directories are skipped rather than raised on: a permissions
    problem in one corner of a tree should cost you that corner, not the scan.
    """
    root = Path(root)
    excluder = excluder or Excluder(())

    for dirpath, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        here = Path(dirpath)
        if not recursive and here != root:
            dirnames.clear()
            continue

        # In place, because os.walk reads this list back to decide where to go.
        dirnames[:] = [
            d
            for d in dirnames
            if not excluder.excludes_dir(_relative(here / d, root))
            and not (here / d).is_symlink()
        ]

        for name in filenames:
            path = here / name
            if kind_for(path) is None or path.is_symlink():
                continue
            if excluder.excludes_file(_relative(path, root)):
                continue
            yield path

        if not recursive:
            dirnames.clear()


def _relative(path: Path, root: Path) -> str:
    """POSIX-style path relative to the root, for matching patterns against."""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _translate(pattern: str) -> re.Pattern[str]:
    r"""Compile a glob to a regex, with ``**`` crossing separators and ``*`` not.

    :mod:`fnmatch` is no use here because its ``*`` matches ``/`` as well, which
    would make ``Editor/*.png`` match ``Editor/sub/deep/x.png``. The distinction
    between the two wildcards is the whole reason these patterns are written
    with ``**`` in them.

    >>> bool(_translate("**/Temp/**").match("a/b/Temp/c.png"))
    True
    >>> bool(_translate("*.meta").match("x.meta")), bool(_translate("*.meta").match("a/x.meta"))
    (True, False)
    """
    out: list[str] = []
    index = 0
    length = len(pattern)

    while index < length:
        char = pattern[index]
        if char == "*":
            if pattern[index : index + 3] == "**/":
                # Zero or more directories, so "**/x" also matches a bare "x".
                out.append(r"(?:[^/]+/)*")
                index += 3
            elif pattern[index : index + 2] == "**":
                out.append(r".*")
                index += 2
            else:
                out.append(r"[^/]*")
                index += 1
        elif char == "?":
            out.append(r"[^/]")
            index += 1
        elif char == "[":
            close = _closing_bracket(pattern, index)
            if close is None:
                out.append(re.escape(char))
                index += 1
            else:
                body = pattern[index + 1 : close]
                if body[:1] in ("!", "^"):
                    body = "^" + body[1:]
                out.append(f"[{body}]")
                index = close + 1
        else:
            out.append(re.escape(char))
            index += 1

    return re.compile("".join(out) + r"\Z")


def _closing_bracket(pattern: str, start: int) -> int | None:
    """Index of the ``]`` ending a character class opened at ``start``."""
    index = start + 1
    if index < len(pattern) and pattern[index] in "!^":
        index += 1
    if index < len(pattern) and pattern[index] == "]":  # a literal ] first
        index += 1
    while index < len(pattern) and pattern[index] != "]":
        index += 1
    return index if index < len(pattern) else None
