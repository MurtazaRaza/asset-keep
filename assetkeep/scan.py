"""The incremental walk that keeps the index in step with the filesystem.

Two decisions carry this module.

**Nothing is deleted.** A file that has vanished has its location marked
``present = 0`` and that is all. An unplugged external drive, a folder moved
while the tool was closed, a pack temporarily extracted elsewhere: none of those
may destroy tags that took real effort to apply. Assets with no present location
surface under ``is:missing``, and removal is an explicit ``assetkeep prune``.

**A file is only hashed when ``(size, mtime)`` says it changed.** This is what
makes a rescan of a 5,000-file Unity tree take seconds rather than minutes, and
it is the reason the whole thing can be re-run casually rather than scheduled.
The pair is not a cryptographic guarantee - a file rewritten within the same
mtime tick, at the same length, is missed - but the alternative is reading
576 MB to discover nothing changed, and the recovery is ``--rehash``.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from . import (
    db,
    hashing,
    probe as probe_module,
    roots as roots_module,
    similarity,
    vectors,
)
from .config import Config, RootConfig
from .tagging import clip, heuristic

log = logging.getLogger(__name__)

#: Rows written between commits. Small enough that a scan interrupted halfway
#: keeps most of its work, large enough that per-transaction overhead is noise.
COMMIT_EVERY = 250


@dataclass
class ScanStats:
    """What one scan did, in the terms the CLI and the progress stream report."""

    roots: int = 0
    #: Files considered, after excludes and the extension allowlist.
    seen: int = 0
    #: Content never indexed before.
    added: int = 0
    #: Content already known, found at a path it was not known at. A moved file,
    #: or the same pack copied into a second project.
    relinked: int = 0
    #: A known path whose bytes changed.
    updated: int = 0
    #: Skipped by the (size, mtime) fast path.
    unchanged: int = 0
    #: Assets whose probe was re-run over content already indexed.
    reprobed: int = 0
    #: Locations whose file was not there this time.
    absent: int = 0
    errors: int = 0
    failures: list[str] = field(default_factory=list)

    @property
    def hashed(self) -> int:
        return self.added + self.relinked + self.updated


def scan(
    conn: sqlite3.Connection,
    config: Config,
    only: Iterable[Path] | None = None,
    rehash: bool = False,
    reprobe: bool = False,
    progress: Callable[[ScanStats, Path], None] | None = None,
) -> ScanStats:
    """Bring the index up to date with every enabled root.

    ``rehash`` bypasses the ``(size, mtime)`` fast path, for the case where a
    file was rewritten in place quickly enough to keep its timestamp.

    ``reprobe`` re-runs the probes over content already indexed. This is the
    answer to "I installed assimp, now what": the probe only runs on first sight
    of a hash, so 223 FBX files indexed without geometry stay that way forever
    otherwise, and the alternative - deleting the index and rescanning - throws
    away every manual tag in the library to fix a derived field. Attributes and
    automated tags are replaced; manual and imported tags are not touched.
    """
    stats = ScanStats()
    wanted = {Path(p).expanduser().resolve() for p in only} if only else None

    root_ids = db.sync_roots(conn, config.roots)
    started = db.now()
    # Assets whose automated tags this run has already cleared. Clearing per
    # file instead would be wrong for content in two places: walking the second
    # location would wipe the tags the first one had just written.
    cleared: set[int] = set()

    for root in config.roots:
        if not root.enabled:
            continue
        if wanted is not None and root.path.resolve() not in wanted:
            continue
        if not root.path.is_dir():
            stats.errors += 1
            stats.failures.append(f"{root.path}: not a directory")
            continue

        stats.roots += 1
        _scan_root(
            conn, config, root, root_ids[root.path], stats,
            rehash, reprobe, cleared, progress,
        )
        stats.absent += _mark_absent(conn, root_ids[root.path], started)
        conn.execute(
            "UPDATE root SET last_scan = ? WHERE id = ?", (db.now(), root_ids[root.path])
        )

    return stats


def _scan_root(
    conn: sqlite3.Connection,
    config: Config,
    root: RootConfig,
    root_id: int,
    stats: ScanStats,
    rehash: bool,
    reprobe: bool,
    cleared: set[int],
    progress: Callable[[ScanStats, Path], None] | None,
) -> None:
    excluder = roots_module.Excluder(root.effective_excludes)
    pending = 0
    conn.execute("BEGIN")

    try:
        for path in roots_module.walk(root.path, root.recursive, excluder):
            stats.seen += 1
            try:
                _scan_file(
                    conn, config, root, root_id, path, stats, rehash, reprobe, cleared
                )
            except OSError as exc:
                # A file that vanished mid-walk, or one we cannot read. Neither
                # is a reason to abandon the other 900.
                stats.errors += 1
                stats.failures.append(f"{path}: {exc}")

            pending += 1
            if pending >= COMMIT_EVERY:
                conn.commit()
                conn.execute("BEGIN")
                pending = 0
            if progress is not None:
                progress(stats, path)
    finally:
        conn.commit()


def index_file(
    conn: sqlite3.Connection,
    config: Config,
    root: RootConfig,
    root_id: int,
    path: Path,
    stats: ScanStats | None = None,
) -> int:
    """Index one file that has just appeared, outside a full walk.

    This is what :mod:`assetkeep.vault` calls after placing an imported file.
    It goes through exactly the same code a scan does rather than a simplified
    copy of it, so an imported asset is indistinguishable from a scanned one -
    same hash, same probe, same path tags, same FTS row. The alternative, an
    import path that writes its own subset of that, is how two code paths start
    disagreeing about what an asset is.
    """
    return _scan_file(
        conn, config, root, root_id, path, stats or ScanStats(),
        rehash=False, reprobe=False, cleared=set(),
    )


def _scan_file(
    conn: sqlite3.Connection,
    config: Config,
    root: RootConfig,
    root_id: int,
    path: Path,
    stats: ScanStats,
    rehash: bool,
    reprobe: bool,
    cleared: set[int],
) -> int:
    stat = path.stat()
    location = conn.execute(
        "SELECT * FROM location WHERE abs_path = ?", (str(path),)
    ).fetchone()

    if (
        location is not None
        and not rehash
        and location["size"] == stat.st_size
        and location["mtime"] == stat.st_mtime
    ):
        conn.execute(
            "UPDATE location SET last_seen = ?, present = 1 WHERE id = ?",
            (db.now(), location["id"]),
        )
        stats.unchanged += 1
        asset_id = int(location["asset_id"])
        if not reprobe:
            return asset_id
        # Fall through to re-derive everything path- and content-derived, but
        # without re-reading the file to hash it: the fast path already proved
        # the bytes are the same, so the hash cannot have changed.
        _probe_into(conn, config, asset_id, path, cleared)
        stats.reprobed += 1
        _apply_path_tags(conn, config, root, asset_id, path)
        db.index_fts(conn, asset_id)
        return asset_id

    digest = hashing.hash_file(path)
    asset = db.asset_by_hash(conn, digest)

    if asset is None:
        asset_id = _create(conn, config, root, path, digest, cleared)
        stats.added += 1
    else:
        asset_id = int(asset["id"])
        if location is None:
            stats.relinked += 1
        elif location["asset_id"] != asset_id:
            stats.updated += 1
        else:
            # Same content, new timestamp. A touch, or a lossless re-save.
            stats.unchanged += 1
        if reprobe:
            _probe_into(conn, config, asset_id, path, cleared)
            stats.reprobed += 1

    _upsert_location(conn, asset_id, root_id, path, stat)

    _apply_path_tags(conn, config, root, asset_id, path)
    db.index_fts(conn, asset_id)
    return asset_id


def _apply_path_tags(
    conn: sqlite3.Connection,
    config: Config,
    root: RootConfig,
    asset_id: int,
    path: Path,
) -> None:
    """Tags derived from where the file sits, not from what is inside it.

    Re-applied for every location, not just the first. The same content living
    in two folders legitimately earns both sets, and the (asset, tag, source)
    key makes re-adding an existing one a no-op.
    """
    db.add_tags(
        conn,
        asset_id,
        heuristic.tags_for(path, root.path, root.name, config.tagging.folder_depth),
    )
    if root.vendor:
        db.add_tags(conn, asset_id, [("vendor", "structural", None)])


def _create(
    conn: sqlite3.Connection,
    config: Config,
    root: RootConfig,
    path: Path,
    digest: str,
    cleared: set[int],
) -> int:
    """Register new content: probe it, record what it said, queue the rest."""
    kind = roots_module.kind_for(path) or "image"
    asset_id = db.create_asset(
        conn, kind, digest, path.stem, managed=root.mode == "managed"
    )
    _probe_into(conn, config, asset_id, path, cleared)
    return asset_id


def _probe_into(
    conn: sqlite3.Connection,
    config: Config,
    asset_id: int,
    path: Path,
    cleared: set[int],
) -> None:
    """Run the probe for one file and write what it found onto its asset.

    Shared by first sight and by ``--reprobe``, so an asset that gains geometry
    once assimp is installed goes through exactly the same code as one indexed
    with assimp already there. Automated tags are cleared once per asset per
    run, which is what keeps a re-derivation from also deleting the tags a
    second location contributed earlier in the same walk.
    """
    kind = roots_module.kind_for(path) or "image"
    result = probe_module.probe(path, kind, assimp_lib_path=config.assimp_lib_path)

    if asset_id not in cleared:
        db.clear_automated_tags(conn, asset_id)
        cleared.add(asset_id)

    db.set_attributes(conn, asset_id, result.attributes)
    db.add_tags(conn, asset_id, result.tags)

    if result.dhash is not None and result.palette is not None:
        conn.execute(
            "INSERT OR REPLACE INTO phash (asset_id, dhash, palette) VALUES (?, ?, ?)",
            (asset_id, similarity.to_signed(result.dhash), result.palette),
        )

    # Re-queued on a reprobe as well: a capability that just appeared may make a
    # thumbnail renderable that was not before, which is the usual reason for
    # reprobing in the first place.
    db.enqueue(conn, "thumbnail", asset_id)

    # The same argument applies twice over to embeddings. A reprobe has just
    # cleared every automated tag, and CLIP's are automated, so without this the
    # zero-shot tags would vanish and nothing would ever bring them back - the
    # vector still exists, so no later run would find anything to do. The job is
    # cheap when the vector is already there: it re-derives tags from arithmetic
    # rather than re-running the model.
    if kind in vectors.EMBEDDABLE_KINDS and clip.available(config):
        db.enqueue(conn, "embedding", asset_id)


def _upsert_location(
    conn: sqlite3.Connection, asset_id: int, root_id: int, path: Path, stat
) -> None:
    conn.execute(
        """
        INSERT INTO location (asset_id, root_id, abs_path, size, mtime, last_seen, present)
        VALUES (?, ?, ?, ?, ?, ?, 1)
        ON CONFLICT(abs_path) DO UPDATE SET
            asset_id = excluded.asset_id, root_id = excluded.root_id,
            size = excluded.size, mtime = excluded.mtime,
            last_seen = excluded.last_seen, present = 1
        """,
        (asset_id, root_id, str(path), stat.st_size, stat.st_mtime, db.now()),
    )


def _mark_absent(conn: sqlite3.Connection, root_id: int, started: str) -> int:
    """Flag locations under this root that the walk did not reach.

    Keyed on ``last_seen`` rather than a set of visited paths, so the memory
    cost is constant no matter how large the tree is.
    """
    cursor = conn.execute(
        """
        UPDATE location SET present = 0
        WHERE root_id = ? AND present = 1 AND last_seen < ?
        """,
        (root_id, started),
    )
    return cursor.rowcount or 0


def prune(conn: sqlite3.Connection, dry_run: bool = False) -> list[tuple[int, str]]:
    """Delete assets whose every location is gone. Returns what went.

    Explicit because it is the one destructive operation in the tool. Tags,
    notes, source and licence all live on the asset, and none of it is
    reconstructable from a file that no longer exists.

    References are exempt, and that is load-bearing rather than a nicety: a
    reference has no location and never will, so the query that means "every
    copy has vanished" reads as "all of them" for links. Without the exemption,
    the first ``prune`` after adding one would delete every reference in the
    library, which is the single worst thing this function could do.
    """
    rows = conn.execute(
        """
        SELECT a.id, a.title FROM asset a
        WHERE a.kind != 'reference' AND NOT EXISTS (
            SELECT 1 FROM location l WHERE l.asset_id = a.id AND l.present = 1
        )
        ORDER BY a.id
        """
    ).fetchall()
    doomed = [(int(row["id"]), row["title"]) for row in rows]

    if not dry_run and doomed:
        with conn:
            conn.execute("BEGIN")
            conn.executemany(
                "DELETE FROM asset WHERE id = ?", [(i,) for i, _ in doomed]
            )
            conn.executemany(
                "DELETE FROM asset_fts WHERE rowid = ?", [(i,) for i, _ in doomed]
            )

    return doomed
