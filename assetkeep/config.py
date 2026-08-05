"""Configuration: where things live, and which folders get indexed.

Roots live here rather than only in SQLite, and that is a deliberate split. The
index is layer 3 - purely derived, deletable at any time, rebuilt by a rescan -
so anything that could not be reconstructed from the filesystem has no business
living only inside it. The list of folders to scan is exactly that. The ``root``
table is a copy that :mod:`assetkeep.scan` refreshes from this file on every run.

A missing config file is not an error. Every value here has a working default,
so a fresh install runs, and the file gets written the first time something
changes it.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path

import tomli_w

from .tagging import vocab

DEFAULT_CONFIG_PATH = Path("~/.config/assetkeep/config.toml").expanduser()

#: Merged into every root unless it sets ``exclude_defaults = false``. A single
#: Unity project's Library/ can hold six figures of cached intermediates, and
#: .meta files outnumber real ones, so these are load-bearing rather than tidy.
DEFAULT_EXCLUDES = (
    "**/Library/**",
    "**/Temp/**",
    "**/Obj/**",
    "**/Build/**",
    "**/Logs/**",
    "**/.git/**",
    "**/node_modules/**",
    ".DS_Store",
    "*.meta",
)

#: Folder names that say nothing about which root you are looking at. A root at
#: ``MyStarterUtils/Assets`` is "mystarterutils", not "assets", or every Unity
#: project in the library would answer to the same ``root:`` filter.
GENERIC_DIR_NAMES = frozenset(
    {"assets", "out", "output", "src", "content", "resources", "data", "files"}
)


@dataclass(frozen=True)
class RootConfig:
    """One folder to index, and how."""

    path: Path
    #: ``indexed`` leaves the folder untouched; ``managed`` is the vault, whose
    #: contents this tool owns and may move.
    mode: str = "indexed"
    recursive: bool = True
    excludes: tuple[str, ...] = ()
    exclude_defaults: bool = True
    #: Third-party pack content rather than your own work. Most indexed imagery
    #: realistically is, which is why this separation is a flag on the root and
    #: not a tag someone has to remember to apply.
    vendor: bool = False
    enabled: bool = True

    @property
    def effective_excludes(self) -> tuple[str, ...]:
        """Patterns actually applied, defaults first.

        >>> RootConfig(Path("/x"), excludes=("*.bak",)).effective_excludes[-1]
        '*.bak'
        >>> RootConfig(Path("/x"), exclude_defaults=False).effective_excludes
        ()
        """
        if not self.exclude_defaults:
            return self.excludes
        return DEFAULT_EXCLUDES + self.excludes

    @property
    def name(self) -> str:
        """Short label for ``root:`` filters and the ``source:`` tag.

        Run through the tag canonicaliser so the root name, the ``source:`` tag
        it produces and the ``root:`` filter that finds it are all spelled the
        same way. Without that, ``MyStarterUtils`` is a root, ``mystarterutils``
        is a filter and ``my-starter-utils`` is a tag, and only two of the three
        ever match.

        >>> RootConfig(Path("/u/_UnityProjects/MyStarterUtils/Assets")).name
        'my-starter-utils'
        >>> RootConfig(Path("/u/AssetGeneratorHelper/out")).name
        'asset-generator-helper'
        >>> RootConfig(Path("/u/packs/kenney-platformer")).name
        'kenney-platformer'
        """
        parts = [p for p in self.path.parts if p not in ("/", "\\")]
        for part in reversed(parts):
            if part.lower() not in GENERIC_DIR_NAMES:
                return vocab.canonical(part)
        return vocab.canonical(parts[-1]) if parts else "root"


@dataclass(frozen=True)
class ThumbnailConfig:
    max_edge: int = 256
    quality: int = 80
    #: Skip generation when the source already fits the box. 36% of images in
    #: the calibration project are under 256 px, and a WebP of a 32x32 sprite is
    #: larger than the sprite.
    skip_smaller: bool = True


@dataclass(frozen=True)
class TaggingConfig:
    #: Optional user vocabulary file overriding :mod:`assetkeep.tagging.vocab`.
    vocab_path: Path | None = None
    #: Which exported CLIP to use; see :data:`assetkeep.tagging.clip.VARIANTS`.
    #: The value is also what lands in ``embedding.model``, so changing it means
    #: the existing vectors stop being used and the library re-embeds.
    clip_model: str = "clip-vit-b-32"
    #: How confident the zero-shot tagger has to be before it writes a tag. A
    #: probability within one namespace rather than a raw cosine, which is a
    #: change of meaning explained in :func:`assetkeep.tagging.clip.classify`.
    #: Measured on the calibration library: 0.35 puts ``normal-map`` on a fader
    #: mask and ``pixel-art`` on a debug menu, 0.40 removes both and keeps
    #: everything that was right.
    clip_threshold: float = 0.40
    #: How many parent folder names become tags.
    folder_depth: int = 2


@dataclass(frozen=True)
class VlmConfig:
    """Where the captioning model lives, and which one it is.

    A URL rather than a flag, because ollama is a server: it may be on another
    port, or on the desktop machine rather than this laptop, and neither case
    wants a code change.
    """

    #: See :data:`assetkeep.tagging.vlm.DEFAULT_MODEL` for why moondream.
    model: str = "moondream"
    url: str = "http://127.0.0.1:11434"
    #: Seconds to wait for one caption. Generous, because the first call after
    #: an idle period pays for loading 1.7 GB off disk.
    timeout: float = 180.0


@dataclass(frozen=True)
class Config:
    db_path: Path = Path("~/AssetKeep/index.db").expanduser()
    vault_path: Path = Path("~/AssetKeep/vault").expanduser()
    thumbs_path: Path = Path("~/AssetKeep/thumbs").expanduser()
    models_path: Path = Path("~/AssetKeep/models").expanduser()
    #: Remembered "copy to" destination, normally the Unity Assets/ folder
    #: currently being worked in.
    copy_target: Path | None = None
    #: What happens to assets whose every location has vanished. ``keep`` marks
    #: them absent and leaves the metadata alone; nothing deletes without an
    #: explicit ``assetkeep prune``.
    missing_policy: str = "keep"
    #: Explicit path to libassimp, for when neither impasse nor pyassimp finds
    #: it on their own.
    assimp_lib_path: Path | None = None
    thumbnails: ThumbnailConfig = field(default_factory=ThumbnailConfig)
    tagging: TaggingConfig = field(default_factory=TaggingConfig)
    vlm: VlmConfig = field(default_factory=VlmConfig)
    roots: tuple[RootConfig, ...] = ()
    #: Where this was loaded from, so :func:`save` can write it back.
    source_path: Path = DEFAULT_CONFIG_PATH

    def root_for(self, path: Path) -> RootConfig | None:
        """The configured root containing ``path``, longest match first.

        Nested roots are legal - a Unity project and one pack inside it - and
        the more specific one owns the file, since that is the one whose vendor
        flag and name were set with those files in mind.
        """
        best: RootConfig | None = None
        for root in self.roots:
            if path == root.path or root.path in path.parents:
                if best is None or len(root.path.parts) > len(best.path.parts):
                    best = root
        return best


def load(path: Path | None = None) -> Config:
    """Read config.toml, falling back to defaults for anything unset."""
    path = (path or DEFAULT_CONFIG_PATH).expanduser()
    if not path.exists():
        return Config(source_path=path)

    with path.open("rb") as fh:
        raw = tomllib.load(fh)

    general = raw.get("general", {})
    thumbs = raw.get("thumbnails", {})
    tagging = raw.get("tagging", {})
    vlm = raw.get("vlm", {})

    defaults = Config()
    return Config(
        db_path=_path(general.get("db_path"), defaults.db_path),
        vault_path=_path(general.get("vault_path"), defaults.vault_path),
        thumbs_path=_path(general.get("thumbs_path"), defaults.thumbs_path),
        models_path=_path(general.get("models_path"), defaults.models_path),
        copy_target=_optional_path(general.get("copy_target")),
        missing_policy=general.get("missing_policy", defaults.missing_policy),
        assimp_lib_path=_optional_path(general.get("assimp_lib_path")),
        thumbnails=ThumbnailConfig(
            max_edge=int(thumbs.get("max_edge", 256)),
            quality=int(thumbs.get("quality", 80)),
            skip_smaller=bool(thumbs.get("skip_smaller", True)),
        ),
        tagging=TaggingConfig(
            vocab_path=_optional_path(tagging.get("vocab_path")),
            clip_model=str(tagging.get("clip_model", defaults.tagging.clip_model)),
            clip_threshold=float(
                tagging.get("clip_threshold", defaults.tagging.clip_threshold)
            ),
            folder_depth=int(tagging.get("folder_depth", 2)),
        ),
        vlm=VlmConfig(
            model=str(vlm.get("model", defaults.vlm.model)),
            url=str(vlm.get("url", defaults.vlm.url)),
            timeout=float(vlm.get("timeout", defaults.vlm.timeout)),
        ),
        roots=tuple(_root(entry) for entry in raw.get("root", [])),
        source_path=path,
    )


def save(config: Config, path: Path | None = None) -> Path:
    """Write the config back out, and return where it went.

    Home-relative paths are re-contracted to ``~`` so the file stays readable
    and survives being copied to another machine.
    """
    path = (path or config.source_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)

    general: dict[str, object] = {
        "db_path": _contract(config.db_path),
        "vault_path": _contract(config.vault_path),
        "thumbs_path": _contract(config.thumbs_path),
        "models_path": _contract(config.models_path),
        "missing_policy": config.missing_policy,
    }
    if config.copy_target is not None:
        general["copy_target"] = _contract(config.copy_target)
    if config.assimp_lib_path is not None:
        general["assimp_lib_path"] = _contract(config.assimp_lib_path)

    document: dict[str, object] = {
        "general": general,
        "thumbnails": {
            "max_edge": config.thumbnails.max_edge,
            "quality": config.thumbnails.quality,
            "skip_smaller": config.thumbnails.skip_smaller,
        },
        "tagging": {
            "vocab_path": _contract(config.tagging.vocab_path)
            if config.tagging.vocab_path
            else "",
            "clip_model": config.tagging.clip_model,
            "clip_threshold": config.tagging.clip_threshold,
            "folder_depth": config.tagging.folder_depth,
        },
        "vlm": {
            "model": config.vlm.model,
            "url": config.vlm.url,
            "timeout": config.vlm.timeout,
        },
    }
    if config.roots:
        document["root"] = [_root_table(root) for root in config.roots]

    # Explicit UTF-8: the platform default is cp1252 on Windows, and a root
    # path with a non-ASCII folder name in it would fail to write at all.
    path.write_text(tomli_w.dumps(document), encoding="utf-8")
    return path


def with_root(config: Config, root: RootConfig) -> Config:
    """Config with ``root`` added, or replacing an existing root at that path."""
    kept = tuple(r for r in config.roots if r.path != root.path)
    return replace(config, roots=kept + (root,))


def without_root(config: Config, path: Path) -> Config:
    """Config with the root at ``path`` removed."""
    path = path.expanduser().resolve()
    return replace(config, roots=tuple(r for r in config.roots if r.path != path))


def _root(entry: dict) -> RootConfig:
    path = entry.get("path")
    if not path:
        raise ValueError("a [[root]] entry has no path")
    return RootConfig(
        path=Path(path).expanduser(),
        mode=entry.get("mode", "indexed"),
        recursive=bool(entry.get("recursive", True)),
        excludes=tuple(entry.get("exclude", ())),
        exclude_defaults=bool(entry.get("exclude_defaults", True)),
        vendor=bool(entry.get("vendor", False)),
        enabled=bool(entry.get("enabled", True)),
    )


def _root_table(root: RootConfig) -> dict[str, object]:
    """Serialise a root, omitting anything left at its default.

    Written back sparsely on purpose: a config full of restated defaults is
    unreadable, and every key present in the file is one the next version has to
    keep honouring.
    """
    table: dict[str, object] = {"path": _contract(root.path)}
    if root.mode != "indexed":
        table["mode"] = root.mode
    if not root.recursive:
        table["recursive"] = False
    if root.excludes:
        table["exclude"] = list(root.excludes)
    if not root.exclude_defaults:
        table["exclude_defaults"] = False
    if root.vendor:
        table["vendor"] = True
    if not root.enabled:
        table["enabled"] = False
    return table


def _path(value: object, fallback: Path) -> Path:
    if not isinstance(value, str) or not value:
        return fallback
    return Path(value).expanduser()


def _optional_path(value: object) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    return Path(value).expanduser()


def _contract(path: Path) -> str:
    """Inverse of ``expanduser`` for display and write-back.

    The property asserted is the round trip rather than a literal string,
    because the separator in that string is ``/`` on one of the machines this
    runs on and ``\\`` on the other, and the round trip is what the function is
    actually for.

    >>> inside = Path.home() / "AssetKeep" / "index.db"
    >>> _contract(inside).startswith("~"), Path(_contract(inside)).expanduser() == inside
    (True, True)
    >>> outside = Path(Path.home().anchor) / "opt" / "lib" / "assimp"
    >>> _contract(outside) == str(outside)
    True
    """
    home = Path.home()
    if path == home or home in path.parents:
        return str(Path("~") / path.relative_to(home))
    return str(path)
