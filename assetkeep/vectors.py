"""Where embeddings live, and the arithmetic over them. Pure numpy.

Deliberately separate from :mod:`assetkeep.tagging.clip`. That module needs
onnxruntime, 150 MB of weights and a working download; this one needs neither,
and everything that *reads* embeddings - ranking a search, finding neighbours,
answering "how many are done" - goes through here. So a machine that once ran
the embedding job keeps semantic ``similar:`` and semantic search working after
the optional extra is uninstalled, and the tests for the ranking never have to
load a model.

Vectors are stored L2-normalised, which turns cosine similarity into a plain dot
product and makes the whole comparison one matrix multiply. It also means the
normalisation happens once per asset at write time rather than once per asset
per query.
"""

from __future__ import annotations

import logging
import sqlite3

import numpy as np

log = logging.getLogger(__name__)

#: Assets loaded into the in-memory matrix before this module admits it is the
#: wrong data structure. ViT-B/32 is 512 float32s, so 2 KB an asset: the
#: calibration library's 929 assets are 1.9 MB and a 50,000-asset one is 100 MB,
#: which is where an approximate index (hnswlib, sqlite-vec) starts earning its
#: dependency. Below it, a brute-force dot product over the whole library takes
#: about a millisecond and needs no index to keep in step with the data.
MATRIX_WARN_AT = 50_000

#: Cosine below which a semantic neighbour is noise. Not 0, and that surprises
#: people: a CLIP image tower's outputs occupy a narrow cone, so two entirely
#: unrelated images still score highly. Measured over all 500,556 pairs in the
#: calibration library, the 1st percentile is 0.537 and the median 0.736 - the
#: useful signal is the tail, not the distance from zero.
#:
#: 0.80 rather than the tighter 0.85 that the distribution alone suggests,
#: because looking at what sits between them settled it: for a sword model, 0.85
#: returns exactly one neighbour, and the 0.82s it excludes are a plant, a
#: crate piece and two pillars - every other small low-poly prop in the library,
#: which is a perfectly good answer to "more like this". The band is content,
#: not noise, and :data:`assetkeep.search.SIMILAR_LIMIT` already caps how much
#: of it anybody sees. Two percent of assets still have no neighbour at all and
#: fall back to the perceptual hash, which is better at that case anyway.
NEIGHBOUR_FLOOR = 0.80

#: Kinds a vision tower has anything to say about. Audio is absent on purpose:
#: its thumbnail is a waveform, and a waveform embedded as a picture produces a
#: confident, entirely fictional opinion about what the sound is of.
EMBEDDABLE_KINDS = ("image", "model3d")


def pack(vector) -> bytes:
    """Normalise a vector and render it as the float32 blob the column holds.

    Native byte order, which is fine for a file that never leaves the machine
    that wrote it, and this database is derived and rebuildable in any case.

    >>> np.frombuffer(pack([3.0, 4.0]), dtype=np.float32).astype(float).round(3).tolist()
    [0.6, 0.8]
    >>> np.frombuffer(pack([0.0, 0.0]), dtype=np.float32).tolist()
    [0.0, 0.0]
    """
    array = np.asarray(vector, dtype=np.float32).ravel()
    return normalise(array).astype(np.float32).tobytes()


def unpack(blob: bytes) -> np.ndarray:
    """Inverse of :func:`pack`.

    >>> unpack(pack([1.0, 0.0, 0.0])).tolist()
    [1.0, 0.0, 0.0]
    """
    return np.frombuffer(blob, dtype=np.float32)


def normalise(array: np.ndarray) -> np.ndarray:
    """Scale each row to unit length, leaving all-zero rows alone.

    A zero row means an encoder returned nothing useful. Dividing by its norm
    would fill the vector with NaN, and one NaN in the matrix poisons every
    comparison in the library rather than just its own.

    >>> normalise(np.array([[3.0, 4.0], [0.0, 0.0]])).astype(float).round(3).tolist()
    [[0.6, 0.8], [0.0, 0.0]]
    """
    array = np.asarray(array, dtype=np.float32)
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.where(norms == 0, 1.0, norms)


# --- storage ----------------------------------------------------------------


def store(conn: sqlite3.Connection, asset_id: int, model: str, vector) -> None:
    """Write one asset's embedding, replacing any earlier one for this model."""
    conn.execute(
        "INSERT INTO embedding (asset_id, model, vector) VALUES (?, ?, ?) "
        "ON CONFLICT(asset_id, model) DO UPDATE SET vector = excluded.vector",
        (int(asset_id), model, pack(vector)),
    )


def store_many(conn: sqlite3.Connection, model: str, rows) -> int:
    """Write ``(asset_id, vector)`` pairs. Returns how many were written."""
    written = 0
    for asset_id, vector in rows:
        store(conn, asset_id, model, vector)
        written += 1
    return written


def vector_for(
    conn: sqlite3.Connection, asset_id: int, model: str
) -> np.ndarray | None:
    row = conn.execute(
        "SELECT vector FROM embedding WHERE asset_id = ? AND model = ?",
        (int(asset_id), model),
    ).fetchone()
    return None if row is None else unpack(row["vector"])


def primary_model(conn: sqlite3.Connection) -> str | None:
    """Whichever model most of the library was embedded with, or ``None``.

    Lets the ranking code answer "are there embeddings, and of what" without
    being handed a config. Only one model is in play in any normal install; the
    query exists for the run right after somebody changed ``clip_model``, where
    two are, and the one that has actually been applied to the library is the
    one worth comparing against.
    """
    row = conn.execute(
        "SELECT model FROM embedding GROUP BY model ORDER BY COUNT(*) DESC LIMIT 1"
    ).fetchone()
    return None if row is None else str(row["model"])


def count(conn: sqlite3.Connection, model: str | None = None) -> int:
    """How many assets have an embedding, for the queue status line."""
    if model is None:
        row = conn.execute("SELECT COUNT(*) FROM embedding").fetchone()
    else:
        row = conn.execute(
            "SELECT COUNT(*) FROM embedding WHERE model = ?", (model,)
        ).fetchone()
    return int(row[0])


def pending(
    conn: sqlite3.Connection,
    model: str,
    kinds=EMBEDDABLE_KINDS,
    limit: int | None = None,
) -> list[int]:
    """Assets of these kinds with no embedding yet, oldest first.

    Restricted to kinds the encoder can actually see. A CLIP vision tower takes
    pixels, so an audio file has nothing to embed - and enqueueing one produces
    a job that fails three times and then sits in the failed count forever,
    which reads as a broken install rather than as a file that was never a
    candidate.
    """
    placeholders = ",".join("?" * len(kinds))
    sql = (
        f"SELECT a.id FROM asset a WHERE a.kind IN ({placeholders}) "
        "AND NOT EXISTS (SELECT 1 FROM embedding e "
        "WHERE e.asset_id = a.id AND e.model = ?) "
        "AND EXISTS (SELECT 1 FROM location l "
        "WHERE l.asset_id = a.id AND l.present = 1) "
        "ORDER BY a.id"
    )
    params: list = [*kinds, model]
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    return [int(row["id"]) for row in conn.execute(sql, params)]


# --- comparison -------------------------------------------------------------


def matrix(conn: sqlite3.Connection, model: str) -> tuple[np.ndarray, np.ndarray]:
    """Every embedding for one model as ``(ids, vectors)``.

    Read fresh every time, and that is a decision rather than an oversight. The
    obvious optimisation is to cache this and invalidate on ``(count, max rowid)``,
    which is wrong in a way a test caught and a person would not have: replacing
    a vector in place changes neither, so the cache serves the old numbers and
    the symptom is "semantic search gives odd results" - indistinguishable, to
    anybody looking at it, from CLIP simply being like that.

    Every honest fix was worse. A per-connection counter cannot key a cache
    shared between the request connections and the worker's. A timestamp column
    is a migration in aid of a cache. Measured, the whole read is 2.4 ms for a
    thousand assets, against a search that is debounced to at most a few a
    second. There was nothing here worth being wrong about.
    """
    rows = conn.execute(
        "SELECT asset_id, vector FROM embedding WHERE model = ? ORDER BY asset_id",
        (model,),
    ).fetchall()

    if not rows:
        ids = np.empty(0, dtype=np.int64)
        vectors = np.empty((0, 0), dtype=np.float32)
    else:
        ids = np.fromiter((int(row["asset_id"]) for row in rows), dtype=np.int64)
        vectors = np.stack([unpack(row["vector"]) for row in rows])
        if ids.size > MATRIX_WARN_AT:
            log.warning(
                "%d embeddings held in memory; past %d this wants a real vector "
                "index rather than a brute-force scan",
                ids.size,
                MATRIX_WARN_AT,
            )

    return ids, vectors


def rank(
    conn: sqlite3.Connection,
    model: str,
    query,
    limit: int = 60,
    floor: float = 0.0,
    exclude: int | None = None,
) -> list[tuple[int, float]]:
    """Assets ranked by cosine against ``query``, best first.

    One dot product against the whole library. ``floor`` is applied after
    ranking rather than instead of it, so a caller that wants "the best twenty
    whatever they score" and one that wants "everything genuinely close" are the
    same code path with different numbers.
    """
    ids, vectors = matrix(conn, model)
    if ids.size == 0:
        return []

    probe = normalise(np.asarray(query, dtype=np.float32).ravel())
    if probe.shape[0] != vectors.shape[1]:
        # A model swapped under a database that still holds the old vectors.
        # Returning nothing is right: these numbers are not comparable, and
        # comparing them anyway produces confident nonsense.
        return []

    scores = vectors @ probe
    if exclude is not None:
        scores = np.where(ids == exclude, -np.inf, scores)

    take = min(limit, scores.shape[0])
    if take <= 0:
        return []
    # Partition first, sort only the survivors: the full sort is O(n log n) over
    # the library, the partition is O(n) and the sort then runs over `limit`.
    if take < scores.shape[0]:
        best = np.argpartition(-scores, take - 1)[:take]
    else:
        best = np.arange(scores.shape[0])
    best = best[np.argsort(-scores[best])]

    return [
        (int(ids[index]), float(scores[index]))
        for index in best
        if scores[index] >= floor
    ]


def neighbours(
    conn: sqlite3.Connection,
    model: str,
    asset_id: int,
    limit: int = 60,
    floor: float = NEIGHBOUR_FLOOR,
) -> list[tuple[int, float]]:
    """Nearest embeddings to one asset's own, excluding itself.

    Returns ``[]`` when that asset has no embedding, which is the signal
    :func:`assetkeep.search.similar_ids` uses to fall back to the perceptual
    hash rather than to report no neighbours.
    """
    probe = vector_for(conn, asset_id, model)
    if probe is None:
        return []
    return rank(conn, model, probe, limit=limit, floor=floor, exclude=int(asset_id))
