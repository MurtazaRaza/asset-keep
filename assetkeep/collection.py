"""Collections: hand-made sets that outlive the query that found them.

A collection is the answer to "these forty, out of the six hundred the search
returned". Everything else in this tool is derived - tags come back from a
rescan, attributes from a reprobe, the whole index from nothing - but a
collection is a judgement someone made, and there is no rule that reconstructs
it. That makes it the second thing in the database, after manual tags, that a
rebuild genuinely loses.

Names are canonicalised the same way tags and root names are. It looks like an
imposition on what is obviously a human label, and it is the only thing that
makes ``collection:jam-prototype`` work from the query bar: the filter matches
by exact name, the sidebar writes filters unquoted, and a collection called
``Jam Prototype`` would be reachable by typing but not by clicking. The prose
belongs in ``notes``, which nothing parses.
"""

from __future__ import annotations

import sqlite3

from . import db
from .tagging import vocab


def create(conn: sqlite3.Connection, name: str, notes: str = "") -> int:
    """Make a collection, or return the id of the one already using the name."""
    canonical = vocab.canonical(name)
    if not canonical:
        raise ValueError("a collection needs a name")

    existing = by_name(conn, canonical)
    if existing is not None:
        return int(existing["id"])

    cursor = conn.execute(
        "INSERT INTO collection (name, notes, created_at) VALUES (?, ?, ?)",
        (canonical, notes, db.now()),
    )
    return int(cursor.lastrowid)


def update(
    conn: sqlite3.Connection,
    collection_id: int,
    name: str | None = None,
    notes: str | None = None,
) -> sqlite3.Row | None:
    """Rename or re-annotate. Either field may be left alone by passing ``None``."""
    if name is not None:
        conn.execute(
            "UPDATE collection SET name = ? WHERE id = ?",
            (vocab.canonical(name), collection_id),
        )
    if notes is not None:
        conn.execute(
            "UPDATE collection SET notes = ? WHERE id = ?", (notes, collection_id)
        )
    return get(conn, collection_id)


def delete(conn: sqlite3.Connection, collection_id: int) -> bool:
    """Drop a collection. The assets in it are untouched - it is a set, not a folder."""
    cursor = conn.execute("DELETE FROM collection WHERE id = ?", (collection_id,))
    return bool(cursor.rowcount)


def get(conn: sqlite3.Connection, collection_id: int) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM collection WHERE id = ?", (collection_id,)
    ).fetchone()


def by_name(conn: sqlite3.Connection, name: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM collection WHERE name = ?", (vocab.canonical(name),)
    ).fetchone()


def listing(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Every collection with its size, for the sidebar."""
    return conn.execute(
        """
        SELECT c.id, c.name, c.notes, c.created_at,
               COUNT(ca.asset_id) AS count
        FROM collection c
        LEFT JOIN collection_asset ca ON ca.collection_id = c.id
        GROUP BY c.id ORDER BY c.name
        """
    ).fetchall()


def members(conn: sqlite3.Connection, collection_id: int) -> list[int]:
    return [
        int(row["asset_id"])
        for row in conn.execute(
            "SELECT asset_id FROM collection_asset WHERE collection_id = ? "
            "ORDER BY position",
            (collection_id,),
        )
    ]


def add(conn: sqlite3.Connection, collection_id: int, asset_ids) -> int:
    """Append assets, keeping the position of anything already a member.

    Re-adding is a no-op rather than a move to the end. Someone selecting a
    rectangle in the grid and hitting add twice means "make sure these are in
    it", not "reorder the collection", and the second reading is the one that
    quietly destroys a hand-made ordering.
    """
    row = conn.execute(
        "SELECT COALESCE(MAX(position), -1) + 1 FROM collection_asset "
        "WHERE collection_id = ?",
        (collection_id,),
    ).fetchone()
    position = int(row[0])

    written = 0
    for asset_id in asset_ids:
        cursor = conn.execute(
            "INSERT INTO collection_asset (collection_id, asset_id, position) "
            "VALUES (?, ?, ?) ON CONFLICT DO NOTHING",
            (collection_id, int(asset_id), position),
        )
        if cursor.rowcount:
            written += 1
            position += 1
    return written


def remove(conn: sqlite3.Connection, collection_id: int, asset_ids) -> int:
    ids = [int(asset_id) for asset_id in asset_ids]
    if not ids:
        return 0
    placeholders = ",".join("?" * len(ids))
    cursor = conn.execute(
        f"DELETE FROM collection_asset WHERE collection_id = ? "
        f"AND asset_id IN ({placeholders})",
        (collection_id, *ids),
    )
    return cursor.rowcount or 0


def containing(conn: sqlite3.Connection, asset_id: int) -> list[sqlite3.Row]:
    """Which collections an asset is in, for the inspector."""
    return conn.execute(
        "SELECT c.id, c.name FROM collection_asset ca "
        "JOIN collection c ON c.id = ca.collection_id "
        "WHERE ca.asset_id = ? ORDER BY c.name",
        (asset_id,),
    ).fetchall()
