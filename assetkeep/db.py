"""SQLite schema, migrations, and the handful of writes everything shares.

The database is disposable by design: every row in it is derived from files on
disk plus the config, so deleting it and rescanning is a supported recovery path
rather than a disaster. That is what lets migrations stay simple - a numbered
list of functions keyed off ``PRAGMA user_version``, with no rollback story,
because the fallback is always "throw it away and scan again".
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .tagging import vocab

#: Tag sources a rescan is allowed to replace wholesale. ``manual`` and
#: ``imported`` are absent on purpose: work a person did, or metadata that
#: arrived with the asset, must survive re-running an improved tagger.
AUTOMATED_SOURCES = ("structural", "heuristic", "clip", "vlm")

TAG_SOURCES = AUTOMATED_SOURCES + ("manual", "imported")

KINDS = ("image", "model3d", "audio", "reference")


def connect(path: Path, same_thread: bool = True) -> sqlite3.Connection:
    """Open (creating if needed) the index, migrated and ready to use.

    ``same_thread=False`` lifts sqlite3's own thread check. Only pass it for a
    connection owned by exactly one logical task that may nonetheless be resumed
    on different threads - which is precisely what a FastAPI request is, since a
    sync generator dependency has its setup, its endpoint and its teardown run
    on three different threadpool workers. It is not a licence to share one
    connection between concurrent requests.
    """
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(path, isolation_level=None, check_same_thread=same_thread)
    conn.row_factory = sqlite3.Row
    # WAL so a long scan writing in the background does not block the server
    # reading; foreign keys because every ON DELETE CASCADE here is load-bearing
    # and SQLite leaves them off by default.
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA synchronous = NORMAL")
    migrate(conn)
    return conn


def migrate(conn: sqlite3.Connection) -> int:
    """Bring the schema up to date, returning the version now in force."""
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    for index, step in enumerate(MIGRATIONS[version:], start=version + 1):
        with conn:
            conn.execute("BEGIN")
            step(conn)
            conn.execute(f"PRAGMA user_version = {index}")
    return len(MIGRATIONS)


def _v1(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE root (
            id        INTEGER PRIMARY KEY,
            path      TEXT NOT NULL UNIQUE,
            name      TEXT NOT NULL DEFAULT '',
            mode      TEXT NOT NULL DEFAULT 'indexed',
            recursive INTEGER NOT NULL DEFAULT 1,
            excludes  TEXT NOT NULL DEFAULT '[]',
            vendor    INTEGER NOT NULL DEFAULT 0,
            enabled   INTEGER NOT NULL DEFAULT 1,
            last_scan TEXT
        );

        CREATE TABLE asset (
            id           INTEGER PRIMARY KEY,
            kind         TEXT NOT NULL,
            content_hash TEXT UNIQUE,
            title        TEXT NOT NULL,
            notes        TEXT NOT NULL DEFAULT '',
            caption      TEXT,
            source_url   TEXT,
            source_name  TEXT,
            license      TEXT,
            managed      INTEGER NOT NULL DEFAULT 0,
            added_at     TEXT NOT NULL,
            updated_at   TEXT NOT NULL
        );
        CREATE INDEX asset_kind ON asset(kind);

        CREATE TABLE location (
            id        INTEGER PRIMARY KEY,
            asset_id  INTEGER NOT NULL REFERENCES asset(id) ON DELETE CASCADE,
            root_id   INTEGER REFERENCES root(id) ON DELETE SET NULL,
            abs_path  TEXT NOT NULL UNIQUE,
            size      INTEGER NOT NULL,
            mtime     REAL NOT NULL,
            last_seen TEXT NOT NULL,
            present   INTEGER NOT NULL DEFAULT 1
        );
        CREATE INDEX location_asset ON location(asset_id);
        CREATE INDEX location_root ON location(root_id, present);

        CREATE TABLE tag (
            id        INTEGER PRIMARY KEY,
            name      TEXT NOT NULL UNIQUE,
            namespace TEXT
        );

        CREATE TABLE tag_alias (
            alias  TEXT PRIMARY KEY,
            tag_id INTEGER NOT NULL REFERENCES tag(id) ON DELETE CASCADE
        );

        CREATE TABLE asset_tag (
            asset_id   INTEGER NOT NULL REFERENCES asset(id) ON DELETE CASCADE,
            tag_id     INTEGER NOT NULL REFERENCES tag(id)   ON DELETE CASCADE,
            source     TEXT NOT NULL,
            confidence REAL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (asset_id, tag_id, source)
        );
        CREATE INDEX asset_tag_tag ON asset_tag(tag_id);

        CREATE TABLE attribute (
            asset_id   INTEGER NOT NULL REFERENCES asset(id) ON DELETE CASCADE,
            key        TEXT NOT NULL,
            value_text TEXT,
            value_num  REAL,
            PRIMARY KEY (asset_id, key)
        );
        CREATE INDEX attribute_num ON attribute(key, value_num);

        CREATE TABLE phash (
            asset_id INTEGER PRIMARY KEY REFERENCES asset(id) ON DELETE CASCADE,
            dhash    INTEGER NOT NULL,
            palette  BLOB NOT NULL
        );

        CREATE TABLE embedding (
            asset_id INTEGER NOT NULL REFERENCES asset(id) ON DELETE CASCADE,
            model    TEXT NOT NULL,
            vector   BLOB NOT NULL,
            PRIMARY KEY (asset_id, model)
        );

        CREATE TABLE collection (
            id         INTEGER PRIMARY KEY,
            name       TEXT NOT NULL UNIQUE,
            notes      TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL
        );
        CREATE TABLE collection_asset (
            collection_id INTEGER NOT NULL REFERENCES collection(id) ON DELETE CASCADE,
            asset_id      INTEGER NOT NULL REFERENCES asset(id)      ON DELETE CASCADE,
            position      INTEGER NOT NULL,
            PRIMARY KEY (collection_id, asset_id)
        );

        CREATE TABLE job (
            id         INTEGER PRIMARY KEY,
            kind       TEXT NOT NULL,
            asset_id   INTEGER REFERENCES asset(id) ON DELETE CASCADE,
            state      TEXT NOT NULL DEFAULT 'pending',
            attempts   INTEGER NOT NULL DEFAULT 0,
            error      TEXT,
            created_at TEXT NOT NULL
        );
        CREATE INDEX job_pending ON job(state, kind);
        CREATE UNIQUE INDEX job_unique ON job(kind, asset_id);

        CREATE VIRTUAL TABLE asset_fts USING fts5(
            title, notes, filename, caption, tags,
            tokenize = 'unicode61 remove_diacritics 2'
        );
        """
    )


#: Ordered; the index of a function plus one is the ``user_version`` it produces.
MIGRATIONS = (_v1,)


def now() -> str:
    """UTC timestamp in the one format every column in here uses.

    Microseconds, not seconds, and that is load-bearing rather than fussy.
    :func:`assetkeep.scan.scan` decides a file has vanished by comparing its
    ``last_seen`` against the moment the scan began, and at second resolution a
    scan that finishes inside one second marks nothing absent - every timestamp
    is equal, and ``<`` is false. Forcing the field width also keeps the strings
    lexicographically ordered, which is what ``sort:added`` relies on.
    """
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


# --- roots ------------------------------------------------------------------


def sync_roots(conn: sqlite3.Connection, roots) -> dict[Path, int]:
    """Mirror the configured roots into the index, returning path -> row id.

    Roots that have disappeared from the config are disabled rather than
    deleted, so the locations that referenced them keep pointing somewhere and
    re-adding the folder later recovers the association instead of re-walking
    from nothing.
    """
    ids: dict[Path, int] = {}
    configured = {str(root.path) for root in roots}

    for root in roots:
        conn.execute(
            """
            INSERT INTO root (path, name, mode, recursive, excludes, vendor, enabled)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                name = excluded.name, mode = excluded.mode,
                recursive = excluded.recursive, excludes = excluded.excludes,
                vendor = excluded.vendor, enabled = excluded.enabled
            """,
            (
                str(root.path),
                root.name,
                root.mode,
                int(root.recursive),
                json.dumps(list(root.effective_excludes)),
                int(root.vendor),
                int(root.enabled),
            ),
        )
        row = conn.execute(
            "SELECT id FROM root WHERE path = ?", (str(root.path),)
        ).fetchone()
        ids[root.path] = row["id"]

    for row in conn.execute("SELECT id, path FROM root").fetchall():
        if row["path"] not in configured:
            conn.execute("UPDATE root SET enabled = 0 WHERE id = ?", (row["id"],))

    return ids


# --- assets -----------------------------------------------------------------


def create_asset(
    conn: sqlite3.Connection,
    kind: str,
    content_hash: str | None,
    title: str,
    managed: bool = False,
) -> int:
    stamp = now()
    cursor = conn.execute(
        """
        INSERT INTO asset (kind, content_hash, title, managed, added_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (kind, content_hash, title, int(managed), stamp, stamp),
    )
    return int(cursor.lastrowid)


def asset_by_hash(conn: sqlite3.Connection, content_hash: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM asset WHERE content_hash = ?", (content_hash,)
    ).fetchone()


def touch_asset(conn: sqlite3.Connection, asset_id: int) -> None:
    conn.execute("UPDATE asset SET updated_at = ? WHERE id = ?", (now(), asset_id))


# --- attributes and tags ----------------------------------------------------


def set_attributes(conn: sqlite3.Connection, asset_id: int, attrs: dict) -> None:
    """Replace an asset's structural attributes.

    Numbers land in ``value_num`` and everything else in ``value_text``, so
    ``w:>=512`` and ``tris:<5000`` are index-backed range scans rather than
    casts over TEXT. Booleans count as numbers; ``has:alpha`` is ``value_num = 1``.
    """
    conn.execute("DELETE FROM attribute WHERE asset_id = ?", (asset_id,))
    rows = []
    for key, value in attrs.items():
        if value is None:
            continue
        if isinstance(value, bool):
            rows.append((asset_id, key, None, float(value)))
        elif isinstance(value, (int, float)):
            rows.append((asset_id, key, None, float(value)))
        else:
            rows.append((asset_id, key, str(value), None))
    conn.executemany(
        "INSERT INTO attribute (asset_id, key, value_text, value_num) VALUES (?, ?, ?, ?)",
        rows,
    )


def tag_id(conn: sqlite3.Connection, name: str) -> int:
    """Row id for a tag name, resolving aliases and creating on first use.

    Two alias layers, in order: the curated ones compiled into
    :mod:`assetkeep.tagging.vocab`, then ``tag_alias``, which is where aliases
    learned at runtime - by the LLM canonicaliser, or added by hand - live. The
    table wins because it is the one a person can edit.
    """
    canonical, namespace = vocab.resolve(name)

    alias = conn.execute(
        "SELECT tag_id FROM tag_alias WHERE alias = ?", (canonical,)
    ).fetchone()
    if alias:
        return int(alias["tag_id"])

    row = conn.execute("SELECT id FROM tag WHERE name = ?", (canonical,)).fetchone()
    if row:
        return int(row["id"])

    cursor = conn.execute(
        "INSERT INTO tag (name, namespace) VALUES (?, ?)", (canonical, namespace)
    )
    return int(cursor.lastrowid)


def add_tags(conn: sqlite3.Connection, asset_id: int, tags) -> int:
    """Attach ``(name, source, confidence)`` triples. Returns rows written.

    Re-adding an existing (asset, tag, source) is a no-op rather than an error,
    since a rescan legitimately re-derives tags it already wrote.
    """
    stamp = now()
    written = 0
    for name, source, confidence in tags:
        if source not in TAG_SOURCES:
            raise ValueError(f"unknown tag source {source!r}")
        cursor = conn.execute(
            """
            INSERT INTO asset_tag (asset_id, tag_id, source, confidence, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT DO NOTHING
            """,
            (asset_id, tag_id(conn, name), source, confidence, stamp),
        )
        written += cursor.rowcount or 0
    return written


def remove_tags(
    conn: sqlite3.Connection, asset_id: int, names, source: str = "manual"
) -> int:
    """Detach tags an asset has by hand. Returns rows removed.

    Scoped to one source, and ``manual`` by default, because the UI's "remove
    this tag" is asking to undo a person's decision - not to overrule the
    tagger, which would grow the tag straight back on the next rescan and look
    like the button did nothing.
    """
    removed = 0
    for name in names:
        cursor = conn.execute(
            "DELETE FROM asset_tag WHERE asset_id = ? AND tag_id = ? AND source = ?",
            (asset_id, tag_id(conn, name), source),
        )
        removed += cursor.rowcount or 0
    return removed


#: Columns the inspector may write. Everything else on ``asset`` is derived from
#: the file or from a scan, and letting an editor overwrite those would put the
#: index and the filesystem into a disagreement nothing resolves.
EDITABLE_FIELDS = (
    "title", "notes", "caption", "source_url", "source_name", "license"
)


def update_asset(conn: sqlite3.Connection, asset_id: int, fields: dict) -> bool:
    """Write the editable metadata fields. Unknown keys are ignored.

    >>> sorted(EDITABLE_FIELDS)
    ['caption', 'license', 'notes', 'source_name', 'source_url', 'title']
    """
    changes = {
        key: value for key, value in fields.items() if key in EDITABLE_FIELDS
    }
    if not changes:
        return False

    assignments = ", ".join(f"{key} = ?" for key in changes)
    conn.execute(
        f"UPDATE asset SET {assignments}, updated_at = ? WHERE id = ?",
        [*changes.values(), now(), asset_id],
    )
    return True


def clear_automated_tags(conn: sqlite3.Connection, asset_id: int) -> None:
    """Drop every regenerable tag on an asset, leaving manual ones untouched.

    This is the operation that ``(asset_id, tag_id, source)`` as a primary key
    exists to make safe. Manual tags are excluded by the shape of the data
    rather than by a WHERE clause someone will eventually get wrong.
    """
    placeholders = ",".join("?" * len(AUTOMATED_SOURCES))
    conn.execute(
        f"DELETE FROM asset_tag WHERE asset_id = ? AND source IN ({placeholders})",
        (asset_id, *AUTOMATED_SOURCES),
    )


def tag_names(conn: sqlite3.Connection, asset_id: int) -> list[str]:
    rows = conn.execute(
        """
        SELECT DISTINCT t.name FROM asset_tag at
        JOIN tag t ON t.id = at.tag_id
        WHERE at.asset_id = ? ORDER BY t.name
        """,
        (asset_id,),
    ).fetchall()
    return [row["name"] for row in rows]


# --- full text --------------------------------------------------------------


def index_fts(conn: sqlite3.Connection, asset_id: int) -> None:
    """Rebuild one asset's FTS row from whatever is currently in the tables.

    fts5 has no upsert, so the row is deleted and reinserted. ``rowid`` is the
    asset id, which is what keeps the two in step without a mapping table.
    """
    asset = conn.execute("SELECT * FROM asset WHERE id = ?", (asset_id,)).fetchone()
    if asset is None:
        conn.execute("DELETE FROM asset_fts WHERE rowid = ?", (asset_id,))
        return

    filenames = [
        Path(row["abs_path"]).name
        for row in conn.execute(
            "SELECT abs_path FROM location WHERE asset_id = ?", (asset_id,)
        )
    ]

    conn.execute("DELETE FROM asset_fts WHERE rowid = ?", (asset_id,))
    conn.execute(
        """
        INSERT INTO asset_fts (rowid, title, notes, filename, caption, tags)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            asset_id,
            asset["title"],
            asset["notes"],
            " ".join(dict.fromkeys(filenames)),
            asset["caption"] or "",
            " ".join(tag_names(conn, asset_id)),
        ),
    )


# --- jobs -------------------------------------------------------------------


def enqueue(conn: sqlite3.Connection, kind: str, asset_id: int) -> None:
    """Queue background work, ignoring a duplicate that is already pending."""
    conn.execute(
        """
        INSERT INTO job (kind, asset_id, state, created_at) VALUES (?, ?, 'pending', ?)
        ON CONFLICT(kind, asset_id) DO UPDATE SET
            state = 'pending', attempts = 0, error = NULL
        """,
        (kind, asset_id, now()),
    )
