"""The background queue that turns 929 scanned files into 929 tiles.

Thumbnailing is deliberately not part of the scan. A scan should finish in the
time it takes to walk and hash, so it can be re-run casually; rendering a
13,000-triangle mesh takes half a second, and 226 of those would triple the wall
time of every scan for work that is pure presentation. So the scan enqueues and
returns, and this drains the queue afterwards, which also means the grid can
start showing results while its thumbnails are still arriving.

Each worker owns its SQLite connection. Sharing one across threads is the
classic way to earn ``ProgrammingError: SQLite objects created in a thread can
only be used in that same thread``, and WAL mode means a second connection costs
nothing anyway.

Two handler shapes, because the two kinds of work want different ones.
Thumbnails are independent and go one at a time; embeddings want the whole
group at once, since a neural network run on sixteen images costs far less than
sixteen runs on one. :data:`BATCH_HANDLERS` is checked first, and a batch that
fails as a unit fails every job in it, while an asset the handler simply could
not read fails alone.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from . import db, thumbs, vectors
from .config import Config
from .tagging import clip, vlm

log = logging.getLogger(__name__)

#: Kinds a vision model has anything to look at. The same list as
#: :data:`assetkeep.vectors.EMBEDDABLE_KINDS` and for the same reason, kept
#: separate because the two tiers are installed independently and one of them
#: may grow a kind the other cannot handle.
CAPTIONABLE_KINDS = ("image", "model3d")

#: Tags that mean "this image is data, not a picture of anything". A caption of
#: one is worse than no caption, so these are never queued.
#:
#: Measured: moondream describes normal maps as "a vibrant purple square", "a
#: hexagonal pattern in shades of blue and purple" and "a close-up view of a
#: circuit board". All three are accurate about the pixels and useless about
#: the asset, and worse than useless in the index - every normal map in the
#: library would answer to "purple". This is the same call the audio exclusion
#: makes: a file that is not a picture of anything gets a confident description
#: of the thing it is not.
UNCAPTIONABLE_TAGS = ("normal-map",)

#: Failures before a job is left alone. Retrying a corrupt PSD forever is how a
#: queue never empties and the UI shows "3 pending" until the end of time.
MAX_ATTEMPTS = 3

#: Jobs claimed per transaction. Large enough to amortise the commit, small
#: enough that stopping the worker is responsive.
BATCH = 16

#: How long an idle worker waits before looking again, in seconds. Only reached
#: when nothing wakes it, since :meth:`Worker.nudge` is called after a scan.
IDLE_SLEEP = 2.0


@dataclass
class QueueStatus:
    """A snapshot of the queue, for the status bar and the SSE stream."""

    pending: int = 0
    failed: int = 0
    done: int = 0
    running: bool = False
    current: str | None = None
    #: Pending count per kind. The status line needs it because the kinds cost
    #: wildly different amounts of time: "40 queued" means seconds if they are
    #: thumbnails and two minutes if they are captions.
    pending_kinds: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "pending": self.pending,
            "failed": self.failed,
            "done": self.done,
            "running": self.running,
            "current": self.current,
            "pending_kinds": self.pending_kinds,
        }


def status(conn: sqlite3.Connection) -> QueueStatus:
    counts = dict(
        conn.execute("SELECT state, COUNT(*) FROM job GROUP BY state").fetchall()
    )
    kinds = {
        str(row["kind"]): int(row["count"])
        for row in conn.execute(
            "SELECT kind, COUNT(*) AS count FROM job WHERE state = 'pending' "
            "GROUP BY kind"
        )
    }
    return QueueStatus(
        pending=counts.get("pending", 0),
        failed=counts.get("failed", 0),
        done=counts.get("done", 0),
        pending_kinds=kinds,
    )


def drain(
    conn: sqlite3.Connection,
    config: Config,
    limit: int | None = None,
    progress: Callable[[int, Path], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> QueueStatus:
    """Run pending jobs until the queue is empty, ``limit`` is hit, or stopped."""
    processed = 0

    while limit is None or processed < limit:
        if should_stop is not None and should_stop():
            break

        remaining = BATCH if limit is None else min(BATCH, limit - processed)
        jobs = _claim(conn, remaining)
        if not jobs:
            break

        for kind, group in _by_kind(jobs):
            batched = BATCH_HANDLERS.get(kind)
            sources = (
                _run_batch(conn, config, group, batched)
                if batched is not None
                else [_run(conn, config, job) for job in group]
            )
            for source in sources:
                processed += 1
                if progress is not None and source is not None:
                    progress(processed, source)

    return status(conn)


def _by_kind(jobs: list[sqlite3.Row]) -> list[tuple[str, list[sqlite3.Row]]]:
    """Group a claimed batch by kind, preserving first-seen order.

    Grouping rather than sorting, so a queue holding thumbnails and embeddings
    still drains in roughly the order the scan enqueued them. Sorting by kind
    would silently make one kind starve the other for the length of a rebuild.

    >>> rows = [{"kind": "thumbnail"}, {"kind": "embedding"}, {"kind": "thumbnail"}]
    >>> [(kind, len(group)) for kind, group in _by_kind(rows)]
    [('thumbnail', 2), ('embedding', 1)]
    """
    groups: dict[str, list[sqlite3.Row]] = {}
    for job in jobs:
        groups.setdefault(job["kind"], []).append(job)
    return list(groups.items())


def _claim(conn: sqlite3.Connection, count: int) -> list[sqlite3.Row]:
    """Take the next batch of pending jobs, marking them running.

    Marked in the same transaction as the select, so two workers cannot pick up
    the same job. There is only ever one worker today, but the alternative is a
    bug that appears the day there are two.
    """
    conn.execute("BEGIN IMMEDIATE")
    try:
        jobs = conn.execute(
            "SELECT * FROM job WHERE state = 'pending' ORDER BY id LIMIT ?", (count,)
        ).fetchall()
        if jobs:
            conn.executemany(
                "UPDATE job SET state = 'running' WHERE id = ?",
                [(job["id"],) for job in jobs],
            )
    finally:
        conn.commit()
    return jobs


def _run(conn: sqlite3.Connection, config: Config, job: sqlite3.Row) -> Path | None:
    handler = HANDLERS.get(job["kind"])

    try:
        if handler is None:
            raise ValueError(f"no handler for job kind {job['kind']!r}")
        source = handler(conn, config, int(job["asset_id"]))
    except Exception as exc:  # noqa: BLE001 - a failed tile is not a failed queue
        _fail(conn, job, exc)
        return None

    _done(conn, job)
    return source


def _run_batch(
    conn: sqlite3.Connection,
    config: Config,
    jobs: list[sqlite3.Row],
    handler: "BatchHandler",
) -> list[Path | None]:
    """Hand a whole group to a handler that would rather see it at once.

    The handler returns a mapping of asset id to source, and anything it leaves
    out has failed - which is what keeps one unreadable PSD in a batch of
    sixteen from failing the other fifteen, while a model that will not load at
    all still fails every job it was given rather than marking them done.
    """
    ids = [int(job["asset_id"]) for job in jobs]

    try:
        results = handler(conn, config, ids)
    except Exception as exc:  # noqa: BLE001 - the batch failed, the queue did not
        for job in jobs:
            _fail(conn, job, exc)
        # One entry per job, including the failures. A drain with a limit counts
        # what came back, so under-reporting here means the loop claims the same
        # jobs again inside the same call and burns all three attempts on the
        # spot, instead of leaving them pending for the next run.
        return [None] * len(jobs)

    sources: list[Path | None] = []
    for job in jobs:
        asset_id = int(job["asset_id"])
        if asset_id in results:
            _done(conn, job)
            sources.append(results[asset_id])
        else:
            _fail(conn, job, ValueError("the batch produced no result for this asset"))
            sources.append(None)
    return sources


def _done(conn: sqlite3.Connection, job: sqlite3.Row) -> None:
    conn.execute(
        "UPDATE job SET state = 'done', error = NULL WHERE id = ?", (job["id"],)
    )


def _fail(conn: sqlite3.Connection, job: sqlite3.Row, exc: BaseException) -> None:
    log.warning("job %s (%s) failed: %s", job["id"], job["kind"], exc)
    attempts = int(job["attempts"]) + 1
    conn.execute(
        "UPDATE job SET state = ?, attempts = ?, error = ? WHERE id = ?",
        (
            "failed" if attempts >= MAX_ATTEMPTS else "pending",
            attempts,
            f"{type(exc).__name__}: {exc}",
            job["id"],
        ),
    )


def _thumbnail(conn: sqlite3.Connection, config: Config, asset_id: int) -> Path | None:
    """Render one asset's tile, from whichever of its locations is present."""
    row = conn.execute(
        """
        SELECT a.kind, a.content_hash, l.abs_path
        FROM asset a JOIN location l ON l.asset_id = a.id
        WHERE a.id = ? AND l.present = 1
        ORDER BY l.id LIMIT 1
        """,
        (asset_id,),
    ).fetchone()

    if row is None:
        # Every copy has gone missing since the scan queued this. Not an error;
        # there is simply nothing to render.
        return None

    source = Path(row["abs_path"])
    thumbs.generate(config, row["kind"], source, row["content_hash"])
    return source


def _embedding(
    conn: sqlite3.Connection, config: Config, asset_ids: list[int]
) -> dict[int, Path | None]:
    """Embed a group of assets in one forward pass, and tag them from it.

    Batched because the alternative wastes most of the run: sixteen separate
    calls into the vision tower re-enter onnxruntime sixteen times and give its
    threading nothing to overlap, where one call of sixteen keeps the matrix
    multiplies busy. Decode stays one image at a time on purpose - sixteen 4K
    PNGs held decoded is a quarter of a gigabyte, and this machine has eight.

    An asset that already has a vector is not re-encoded, only re-tagged. That
    is what makes ``scan --reprobe`` cheap: it clears every automated tag,
    including CLIP's, and the tags come back from arithmetic on vectors that
    are already sitting in the database rather than from the model.
    """
    encoder = clip.shared_encoder(config)
    results: dict[int, Path | None] = {}

    reused: dict[int, np.ndarray] = {}
    fresh: dict[int, Path] = {}
    tensors: list[np.ndarray] = []

    for asset_id in asset_ids:
        existing = vectors.vector_for(conn, asset_id, encoder.model_id)
        if existing is not None:
            reused[asset_id] = existing
            results[asset_id] = None
            continue

        row = _present_location(conn, asset_id)
        if row is None:
            # Every copy has gone missing, or it is a kind with nothing to look
            # at. Neither is a failure; there is simply nothing to embed.
            results[asset_id] = None
            continue

        tensor, source = _pixels_for(config, row)
        if tensor is None:
            # Done, not failed - the same answer the thumbnail handler gives to
            # the same file. Measured on the calibration library, this is 53 of
            # 226 models: animation-only FBX exports carrying one clip and zero
            # triangles, which render nothing because there is nothing in them
            # to render. Calling that a failure would mean three attempts each
            # and a permanent count of 54 problems that are not problems.
            #
            # Nothing is lost by not retrying, because the queue is not what
            # remembers: `vectors.pending` is, and it asks which assets have no
            # vector. Install assimp, run `assetkeep embed`, and these come back
            # round on their own.
            log.info("nothing to embed for asset %s", asset_id)
            results[asset_id] = None
            continue

        tensors.append(tensor)
        fresh[asset_id] = source

    if tensors:
        encoded = encoder.encode_pixels(np.stack(tensors))
        for position, asset_id in enumerate(fresh):
            vectors.store(conn, asset_id, encoder.model_id, encoded[position])
            reused[asset_id] = encoded[position]
            results[asset_id] = fresh[asset_id]

    if reused:
        ordered = list(reused)
        tags = clip.tags_for(encoder, np.stack([reused[i] for i in ordered]))
        for asset_id, found in zip(ordered, tags):
            db.add_tags(conn, asset_id, found)
            db.index_fts(conn, asset_id)

    return results


def _present_location(
    conn: sqlite3.Connection, asset_id: int, kinds=vectors.EMBEDDABLE_KINDS
) -> sqlite3.Row | None:
    """One asset's kind, hash and a path that is actually there.

    ``kinds`` is what a caller can see: a vision model needs pixels, so an
    audio file and a reference are both "nothing to do" rather than an error.
    """
    row = conn.execute(
        """
        SELECT a.kind, a.content_hash, l.abs_path
        FROM asset a JOIN location l ON l.asset_id = a.id
        WHERE a.id = ? AND l.present = 1
        ORDER BY l.id LIMIT 1
        """,
        (asset_id,),
    ).fetchone()
    if row is None or row["kind"] not in kinds:
        return None
    return row


def _pixels_for(
    config: Config, row: sqlite3.Row
) -> tuple[np.ndarray | None, Path | None]:
    """Preprocessed pixels for one asset, from the file or from its thumbnail.

    An image is itself. A 3D model is its thumbnail, because CLIP takes pixels
    and an FBX is not pixels - and the flat-shaded render is what a person
    recognises the model by in the grid, so it is the honest thing to index.
    That is what puts 3D content into semantic search at all, rather than
    leaving a quarter of the library reachable only by filename.

    The thumbnail is also the fallback for an image Pillow will not open: an
    EXR, or a PSD saved without a composite. That pipeline already solved both
    through ffmpeg, and solving them a second time here would be solving them
    differently.
    """
    source = Path(row["abs_path"])

    if row["kind"] == "image":
        tensor = _preprocess(source)
        if tensor is not None:
            return tensor, source

    thumbnail = thumbs.path_for(config, row["content_hash"])
    if not thumbnail.exists():
        thumbs.generate(config, row["kind"], source, row["content_hash"])
    if thumbnail.exists():
        tensor = _preprocess(thumbnail)
        if tensor is not None:
            # The real file, not the tile, because this is what gets reported.
            return tensor, source

    return None, None


def _preprocess(source: Path) -> np.ndarray | None:
    try:
        with Image.open(source) as image:
            image.load()
            return clip.preprocess(image)
    except Exception as exc:  # noqa: BLE001 - one bad file, not a bad batch
        log.warning("cannot read %s for embedding: %s", source, exc)
        return None


def _caption(conn: sqlite3.Connection, config: Config, asset_id: int) -> Path | None:
    """Ask the vision model for one sentence, and put it in the index.

    Not batched, unlike embeddings, because ollama serves one request at a time
    anyway: a batch here would only mean holding sixteen images in memory to
    hand them over one by one. The queue is what makes this bearable regardless
    - captions run in the background while the grid stays answerable.
    """
    row = _present_location(conn, asset_id, kinds=CAPTIONABLE_KINDS)
    if row is None:
        # Audio, a reference, or an asset whose every copy has gone. None of
        # those is a failure: a waveform drawn as a picture would get a
        # confident and entirely fictional description of what the sound is of,
        # which is the same argument that keeps audio out of the embeddings.
        log.info("nothing to caption for asset %s", asset_id)
        return None

    source = _visible_file(config, row)
    if source is None:
        return None

    text = vlm.caption(config, source)
    db.update_asset(conn, asset_id, {"caption": text})
    db.index_fts(conn, asset_id)
    return Path(row["abs_path"])


def _visible_file(config: Config, row: sqlite3.Row) -> Path | None:
    """The image a vision model should be shown for this asset.

    The thumbnail, when there is one: 256 px is past what the vision tower
    keeps, base64 of a 4K PNG is 20 MB over a socket, and for a 3D model the
    render is the only thing there is to look at. A sprite too small to have
    earned a thumbnail is served as itself, which is the same rule
    ``/api/thumb`` follows.
    """
    thumbnail = thumbs.path_for(config, row["content_hash"])
    if not thumbnail.exists():
        thumbs.generate(config, row["kind"], Path(row["abs_path"]), row["content_hash"])
    if thumbnail.exists():
        return thumbnail

    source = Path(row["abs_path"])
    return source if row["kind"] == "image" and source.exists() else None


#: Job kind -> handler, one asset at a time.
HANDLERS: dict[str, Callable[[sqlite3.Connection, Config, int], Path | None]] = {
    "thumbnail": _thumbnail,
    "caption": _caption,
}

BatchHandler = Callable[
    [sqlite3.Connection, Config, list[int]], dict[int, Path | None]
]

#: Job kinds that would rather see a whole group. Checked before
#: :data:`HANDLERS`, so a kind in both is batched.
BATCH_HANDLERS: dict[str, BatchHandler] = {
    "embedding": _embedding,
}


class Worker:
    """A background thread draining the queue for as long as the server runs."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._current: str | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._loop, name="assetkeep-jobs", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def nudge(self) -> None:
        """Wake the worker now, rather than at the next idle poll."""
        self._wake.set()

    @property
    def current(self) -> str | None:
        with self._lock:
            return self._current

    def _loop(self) -> None:
        conn = db.connect(self._config.db_path)
        try:
            while not self._stop.is_set():
                self._wake.clear()
                try:
                    drain(
                        conn,
                        self._config,
                        progress=self._note,
                        should_stop=self._stop.is_set,
                    )
                except Exception as exc:  # noqa: BLE001 - the loop must survive
                    log.exception("job worker error: %s", exc)
                finally:
                    with self._lock:
                        self._current = None
                self._wake.wait(timeout=IDLE_SLEEP)
        finally:
            conn.close()

    def _note(self, _processed: int, source: Path) -> None:
        with self._lock:
            self._current = source.name


def enqueue_captions(
    conn: sqlite3.Connection,
    asset_ids,
    redo: bool = False,
    limit: int | None = None,
) -> list[int]:
    """Queue captions for the assets a vision model can actually see.

    Filtered here rather than at each caller, so the CLI and the UI queue the
    same work for the same selection. Assets that already have a caption are
    skipped unless ``redo``: the expensive mistake with this feature is spending
    four minutes re-describing a collection to change nothing.

    ``limit`` caps what is *queued*, not what is reported. Applying it after the
    fact is a bug that measurement found: ``caption --limit 6`` enqueued 161
    jobs, printed 6, and then spent six minutes on all of them - because the
    queue does not know it was only asked for a few.
    """
    queued: list[int] = []
    for asset_id in [int(value) for value in asset_ids]:
        if limit is not None and len(queued) >= limit:
            break
        row = conn.execute(
            "SELECT kind, caption FROM asset WHERE id = ?", (asset_id,)
        ).fetchone()
        if row is None or row["kind"] not in CAPTIONABLE_KINDS:
            continue
        if row["caption"] and not redo:
            continue
        if _present_location(conn, asset_id, kinds=CAPTIONABLE_KINDS) is None:
            continue
        if _tagged_any(conn, asset_id, UNCAPTIONABLE_TAGS):
            continue
        db.enqueue(conn, "caption", asset_id)
        queued.append(asset_id)
    return queued


def _tagged_any(conn: sqlite3.Connection, asset_id: int, names) -> bool:
    placeholders = ",".join("?" * len(names))
    row = conn.execute(
        f"SELECT 1 FROM asset_tag at JOIN tag t ON t.id = at.tag_id "
        f"WHERE at.asset_id = ? AND t.name IN ({placeholders}) LIMIT 1",
        (asset_id, *names),
    ).fetchone()
    return row is not None


def requeue_failed(conn: sqlite3.Connection) -> int:
    """Give up-on jobs another go, after installing a missing capability."""
    cursor = conn.execute(
        "UPDATE job SET state = 'pending', attempts = 0, error = NULL "
        "WHERE state IN ('failed', 'running')"
    )
    return cursor.rowcount or 0
