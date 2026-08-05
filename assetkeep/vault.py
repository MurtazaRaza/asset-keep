"""The managed store: getting loose content in without a project to put it in.

Most assets arrive already living somewhere - a Unity project, a pack extracted
into a folder - and for those, indexing in place is the whole design. The vault
is for the rest: the itch.io bundle still sitting in Downloads, the sprite sheet
someone sent you, the zip you will otherwise extract into a folder called ``new``
and lose. Without somewhere to put those, they never enter the library at all.

**The vault is an ordinary root.** It is registered in the config with
``mode = "managed"``, walked by the same scanner, probed by the same probes, and
searchable through the same grammar. The only thing "managed" changes is that
this tool is allowed to write inside it. That is deliberate: an import path with
its own indexing logic would be a second, quietly divergent definition of what
an asset is, and there is no feature worth that.

**Imports are deduplicated by content.** Dropping the same pack twice does not
double the library, because the hash decides identity everywhere else in this
tool and there is no reason for the vault to be the exception. What it does mean
is that the second import is reported as a duplicate rather than silently doing
nothing, since "I dropped 40 files and got 3" needs an explanation.

Archives are expanded rather than stored, on the grounds that a zip in an asset
index is a file you cannot see inside. Extraction is deliberately narrow - see
:func:`_archive_members` for what is refused and why.
"""

from __future__ import annotations

import logging
import re
import shutil
import sqlite3
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from . import config as config_module, db, hashing, roots as roots_module, scan
from .config import Config, RootConfig
from .tagging import vocab

log = logging.getLogger(__name__)

ARCHIVE_SUFFIXES = frozenset({".zip"})

#: A leading component Windows reads as a drive rather than as a folder name.
#: Only dangerous in first position, which is exactly where an archive has to
#: put it for the join to escape.
_DRIVE = re.compile(r"^[A-Za-z]:")

#: Where a drop with no name of its own lands. A dated folder was the other
#: candidate and reads worse as a tag, which is what a vault folder name becomes.
DEFAULT_BATCH = "dropped"

#: Refused outright. Both are far above any real game asset and far below what
#: it takes to fill a disk, which is the only job they have: a 40 KB zip that
#: declares 8 GB of contents is a bomb, not a pack.
MAX_MEMBER_BYTES = 512 * 1024**2
MAX_ARCHIVE_BYTES = 8 * 1024**3


@dataclass
class ImportResult:
    """What one import did, in the terms the UI and the CLI report."""

    #: Asset ids for content that was not already in the library.
    imported: list[int] = field(default_factory=list)
    #: Asset ids that were already indexed, from a file we therefore did not copy.
    duplicates: list[int] = field(default_factory=list)
    #: Names passed over: unknown extensions, and anything an archive refused.
    skipped: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.imported) + len(self.duplicates)

    def as_dict(self) -> dict:
        return {
            "imported": self.imported,
            "duplicates": self.duplicates,
            "skipped": self.skipped,
            "errors": self.errors,
        }


def ensure_root(config: Config) -> tuple[Config, RootConfig]:
    """Register the vault as a managed root, writing the config if it was not.

    An import has to do this before it can index anything, and doing it here
    rather than at install time means the config file stays free of a root
    pointing at an empty folder nobody has used yet. The write is idempotent.
    """
    existing = next(
        (root for root in config.roots if root.path == config.vault_path), None
    )
    if existing is not None:
        config.vault_path.mkdir(parents=True, exist_ok=True)
        return config, existing

    root = RootConfig(path=config.vault_path, mode="managed")
    config = config_module.with_root(config, root)
    config.vault_path.mkdir(parents=True, exist_ok=True)
    config_module.save(config)
    return config, root


def import_paths(
    conn: sqlite3.Connection,
    config: Config,
    sources,
    batch: str | None = None,
    move: bool = False,
) -> ImportResult:
    """Bring files or folders on disk into the vault and index them.

    ``move`` is for the case where the source is a temporary upload that would
    otherwise be copied and then deleted.
    """
    result = ImportResult()
    config, root = ensure_root(config)
    root_ids = db.sync_roots(conn, config.roots)
    root_id = root_ids[root.path]

    for source in sources:
        source = Path(source).expanduser()
        if not source.exists():
            result.errors.append(f"{source}: not found")
            continue

        folder = vocab.canonical(batch) if batch else _batch_for(source)
        if source.is_dir():
            for path in sorted(source.rglob("*")):
                if path.is_file() and not path.is_symlink():
                    _ingest(
                        conn, config, root, root_id, result, path,
                        folder, path.relative_to(source).as_posix(), move,
                    )
        else:
            _ingest(
                conn, config, root, root_id, result, source,
                folder, source.name, move,
            )

    return result


def import_upload(
    conn: sqlite3.Connection,
    config: Config,
    filename: str,
    temp_path: Path,
    batch: str | None = None,
) -> ImportResult:
    """Ingest one uploaded file, whose real name is not its temp path's name.

    A browser dropping a folder sends each file with its path inside that folder
    as the name, since multipart carries a filename and nothing else. That path
    is untrusted in exactly the way an archive member is, and is sanitised the
    same way.
    """
    result = ImportResult()
    config, root = ensure_root(config)
    root_ids = db.sync_roots(conn, config.roots)

    relative = _safe_relative(filename)
    if batch:
        folder = vocab.canonical(batch)
    elif "/" in relative:
        # A dropped folder names its own batch, and is not then repeated as a
        # directory inside it: vault/pack/Tiles/x.png, not vault/pack/Pack/...
        head, relative = relative.split("/", 1)
        folder = vocab.canonical(head) or DEFAULT_BATCH
    else:
        folder = _batch_for(Path(relative))

    _ingest(
        conn, config, root, root_ids[root.path], result, temp_path,
        folder, relative, move=True,
    )
    return result


def _components(name: str) -> tuple[str, ...]:
    r"""``name`` split into path components, parsed as POSIX on every platform.

    The local :class:`~pathlib.Path` is the wrong tool here, and quietly so.
    A member named ``C:/Windows/x.png`` splits into three ordinary-looking
    components on macOS and into a *drive* plus two on Windows - and joining a
    drive onto the vault path does not extend it, it replaces it. So an archive
    that is inert on the machine this was written on writes into ``C:\Windows``
    on the machine it is going to, having passed the same check. A leading
    backslash is the same trap: POSIX ``Path`` does not see it as a separator
    at all, so it survives the split and re-anchors the join on Windows.

    Parsing as POSIX everywhere makes the answer a property of the name rather
    than of the machine reading it, which is the only version of this that can
    be tested on one platform and relied on from another.

    >>> _components("Pack/Tiles/x.png")
    ('Pack', 'Tiles', 'x.png')
    >>> _components("C:/Windows/x.png")
    ('C:', 'Windows', 'x.png')
    >>> _components(r"\Windows\x.png")
    ('/', 'Windows', 'x.png')
    """
    return PurePosixPath(name.replace("\\", "/")).parts


def _escapes(parts) -> bool:
    r"""Whether joining ``parts`` onto a folder could land outside it.

    >>> _escapes(_components("Pack/Tiles/x.png"))
    False
    >>> _escapes(_components("../../.zshrc"))
    True
    >>> _escapes(_components("/etc/passwd")), _escapes(_components("C:/x.png"))
    (True, True)
    >>> _escapes(_components(r"\Windows\x.png"))
    True
    """
    if not parts:
        return False
    if any(part == ".." for part in parts):
        return True
    return parts[0] in ("/", "//") or bool(_DRIVE.match(parts[0]))


def _safe_relative(name: str) -> str:
    r"""Strip an untrusted upload name down to something safe to join onto a path.

    Sanitising rather than refusing, unlike :func:`_archive_members`, because a
    browser is what produced this name: a drop of a folder full of assets that
    lost one file to a rejected name would be worse than one that lost the odd
    leading dot.

    >>> _safe_relative("Pack/Tiles/x.png")
    'Pack/Tiles/x.png'
    >>> _safe_relative("../../.zshrc")
    'zshrc'
    >>> _safe_relative("/etc/passwd")
    'etc/passwd'
    >>> _safe_relative("C:/Windows/System32/x.png")
    'Windows/System32/x.png'
    >>> _safe_relative(r"\Windows\x.png")
    'Windows/x.png'
    """
    parts = [part for part in _components(name) if part not in ("..", ".")]
    # Anchors and drives are stripped from the front rather than filtered
    # throughout: one behind the other ("/C:/x") has to be peeled, and a colon
    # further in is a legal, if odd, filename that there is no cause to drop.
    while parts and (parts[0] in ("/", "//") or _DRIVE.match(parts[0])):
        parts.pop(0)

    cleaned = "/".join(part.lstrip(".") or "_" for part in parts)
    return cleaned or "upload"


def _batch_for(source: Path) -> str:
    """The vault folder a source lands in, which becomes one of its tags.

    A zip or a folder names its own batch, because that name is nearly always
    the pack name and is the single most useful tag the import can produce. A
    loose file has nothing to offer - its own stem would make one folder per
    file - so it goes to a shared one.
    """
    if source.suffix.lower() in ARCHIVE_SUFFIXES or source.is_dir():
        return vocab.canonical(source.stem) or DEFAULT_BATCH
    return DEFAULT_BATCH


def _ingest(
    conn: sqlite3.Connection,
    config: Config,
    root: RootConfig,
    root_id: int,
    result: ImportResult,
    source: Path,
    batch: str,
    relative: str,
    move: bool,
) -> None:
    """One file: expand it, skip it, or place and index it."""
    if Path(relative).suffix.lower() in ARCHIVE_SUFFIXES:
        _expand(conn, config, root, root_id, result, source, batch)
        if move:
            source.unlink(missing_ok=True)
        return

    if roots_module.kind_for(Path(relative)) is None:
        result.skipped.append(relative)
        return

    try:
        digest = hashing.hash_file(source)
    except OSError as exc:
        result.errors.append(f"{relative}: {exc}")
        return

    known = db.asset_by_hash(conn, digest)
    if known is not None and _has_present_file(conn, int(known["id"])):
        result.duplicates.append(int(known["id"]))
        if move:
            source.unlink(missing_ok=True)
        return

    target = _unique(config.vault_path / batch / relative)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        if move:
            shutil.move(str(source), target)
        else:
            shutil.copy2(source, target)
    except OSError as exc:
        result.errors.append(f"{relative}: {exc}")
        return

    _record(conn, config, root, root_id, result, target, batch, relative)


def _expand(
    conn: sqlite3.Connection,
    config: Config,
    root: RootConfig,
    root_id: int,
    result: ImportResult,
    archive_path: Path,
    batch: str,
) -> None:
    """Extract an archive member by member, indexing each as it lands."""
    try:
        archive = zipfile.ZipFile(archive_path)
    except (zipfile.BadZipFile, OSError) as exc:
        result.errors.append(f"{archive_path.name}: {exc}")
        return

    with archive:
        wanted = list(_archive_members(archive, result))
        prefix = _shared_folder([relative for _, relative in wanted])

        for member, relative in wanted:
            # A pack zipped as ``Pack.zip/Pack/Tiles/x.png`` would otherwise land
            # at ``vault/pack/Pack/Tiles/x.png``, and folder tags only look two
            # levels up: the pack's own name gets pushed out of range by a
            # directory that says nothing. Dropping it is also what anyone
            # extracting the zip by hand does.
            if prefix:
                relative = relative[len(prefix) + 1 :]

            target = _unique(config.vault_path / batch / relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                with archive.open(member) as inside, target.open("wb") as out:
                    shutil.copyfileobj(inside, out)
            except (OSError, zipfile.BadZipFile) as exc:
                result.errors.append(f"{relative}: {exc}")
                target.unlink(missing_ok=True)
                continue

            digest = hashing.hash_file(target)
            known = db.asset_by_hash(conn, digest)
            if known is not None and _has_present_file(conn, int(known["id"])):
                # Extracted before it could be hashed, because a zip member is
                # not seekable without writing it out first. Undo the write.
                target.unlink(missing_ok=True)
                result.duplicates.append(int(known["id"]))
                continue

            _record(conn, config, root, root_id, result, target, batch, relative)


def _record(
    conn: sqlite3.Connection,
    config: Config,
    root: RootConfig,
    root_id: int,
    result: ImportResult,
    target: Path,
    batch: str,
    relative: str,
) -> None:
    """Index a file now sitting in the vault, and tag it with its batch.

    The batch tag is written with source ``imported`` rather than ``structural``,
    and that is the difference between it surviving and it not. A rescan clears
    every automated source and re-derives them from the path - which would
    delete this tag and never bring it back, because the fact that these forty
    files arrived together as one download is not recoverable from where they
    ended up. ``imported`` is exactly the source that exists for metadata that
    came with the asset.
    """
    try:
        asset_id = scan.index_file(conn, config, root, root_id, target)
    except Exception as exc:  # noqa: BLE001 - one bad file is not the batch
        log.exception("indexing %s failed", target)
        result.errors.append(f"{relative}: {type(exc).__name__}: {exc}")
        return

    if batch:
        db.add_tags(conn, asset_id, [(batch, "imported", None)])
        db.index_fts(conn, asset_id)
    result.imported.append(asset_id)


def _shared_folder(names: list[str]) -> str:
    """The single top-level directory every member sits under, if there is one.

    >>> _shared_folder(["Pack/a.png", "Pack/Tiles/b.png"])
    'Pack'
    >>> _shared_folder(["Pack/a.png", "readme/b.png"])
    ''
    >>> _shared_folder(["a.png", "Pack/b.png"])
    ''
    """
    if not names:
        return ""
    heads = {name.split("/", 1)[0] for name in names if "/" in name}
    if len(heads) != 1 or any("/" not in name for name in names):
        return ""
    return heads.pop()


def _archive_members(archive: zipfile.ZipFile, result: ImportResult):
    """Members worth extracting, as ``(info, relative path)``.

    Everything refused here is refused on purpose:

    - **Absolute paths, ``..`` components and drive letters.** This is Zip Slip:
      a member named ``../../.zshrc`` writes outside the vault, and the archive
      is the one place in this tool where a filename arrives from somewhere
      untrusted.
    - **Nested archives.** A zip inside a zip is not recursed into. It buys very
      little - packs are not usually nested - and it is the shape a decompression
      bomb takes.
    - **Anything not on the extension allowlist**, and ``__MACOSX`` and dotfiles
      with it. A pack's ``license.txt`` is a real loss, and there is nothing in
      the index that could show it; recording it as skipped is the honest answer
      until references land in M5.
    - **Sizes above the caps**, per member and in total.
    """
    total = 0
    for info in archive.infolist():
        name = info.filename
        if info.is_dir() or not name:
            continue

        parts = _components(name)
        if _escapes(parts):
            result.skipped.append(f"{name} (unsafe path)")
            continue
        if any(part.startswith(".") or part == "__MACOSX" for part in parts):
            continue

        relative = "/".join(parts)
        suffix = Path(relative).suffix.lower()
        if suffix in ARCHIVE_SUFFIXES:
            result.skipped.append(f"{relative} (nested archive)")
            continue
        if roots_module.kind_for(Path(relative)) is None:
            result.skipped.append(relative)
            continue

        if info.file_size > MAX_MEMBER_BYTES:
            result.skipped.append(f"{relative} (too large)")
            continue
        total += info.file_size
        if total > MAX_ARCHIVE_BYTES:
            result.errors.append("archive exceeds the size limit; stopped early")
            return

        yield info, relative


def _has_present_file(conn: sqlite3.Connection, asset_id: int) -> bool:
    """Whether this asset still has a file on disk.

    Content known only from a location that has since vanished is not a
    duplicate: importing it is how you get it back.
    """
    row = conn.execute(
        "SELECT abs_path FROM location WHERE asset_id = ? AND present = 1",
        (asset_id,),
    ).fetchall()
    return any(Path(entry["abs_path"]).exists() for entry in row)


def _unique(target: Path) -> Path:
    """A free filename at ``target``, adding ``-1``, ``-2`` as needed.

    Content collisions are already handled by the hash, so anything reaching
    here is genuinely different content that happens to share a name - two packs
    each with a ``tile_01.png``. Overwriting would destroy one of them.
    """
    if not target.exists():
        return target
    for index in range(1, 10_000):
        candidate = target.with_name(f"{target.stem}-{index}{target.suffix}")
        if not candidate.exists():
            return candidate
    raise OSError(f"too many files named like {target.name}")
