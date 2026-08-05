"""HTTP API and static host for the browser UI.

``/api/capabilities`` is load-bearing rather than decorative: the frontend hides
controls for anything missing instead of showing buttons that error. Without
assimp there is no "3D" filter chip; without ffmpeg there are no waveforms. The
same principle as the CLI's ``capabilities`` command, serving the same list.

One SQLite connection per request, opened with sqlite3's thread check lifted -
and the second half of that is not optional. A sync generator dependency in
FastAPI has its setup, its endpoint body and its teardown dispatched to *three
different* threadpool workers, so "a connection per request" still trips
``SQLite objects created in a thread can only be used in that same thread`` on
almost every call. The connection is never shared between concurrent requests,
which is the thing the check actually exists to catch.
"""

from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import shutil
import sqlite3
import tempfile
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

from fastapi import Body, Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import (
    __version__,
    collection as collection_module,
    config as config_module,
    db,
    export as export_module,
    job,
    reference as reference_module,
    scan as scan_module,
    search,
    thumbs,
    vault,
    vectors,
)
from .config import Config, RootConfig
from .probe import audio as audio_probe, model3d
from .tagging import clip, vlm

log = logging.getLogger(__name__)

WEB_ROOT = Path(__file__).resolve().parent / "web"

#: How often the SSE stream emits while work is in progress, in seconds.
STATUS_INTERVAL = 0.4


@dataclass
class ScanState:
    """Live progress of a scan running in a background thread."""

    running: bool = False
    seen: int = 0
    added: int = 0
    hashed: int = 0
    absent: int = 0
    current: str = ""
    finished: str | None = None
    error: str | None = None

    def as_dict(self) -> dict:
        return {
            "running": self.running,
            "seen": self.seen,
            "added": self.added,
            "hashed": self.hashed,
            "absent": self.absent,
            "current": self.current,
            "finished": self.finished,
            "error": self.error,
        }


class ScanRunner:
    """Runs a scan off the request thread so the UI stays answerable."""

    def __init__(self, config: Config, worker: job.Worker) -> None:
        self._config = config
        self._worker = worker
        self._lock = threading.Lock()
        self.state = ScanState()

    def start(self, roots: list[Path] | None = None, reprobe: bool = False) -> bool:
        """Begin a scan. ``False`` if one is already running."""
        with self._lock:
            if self.state.running:
                return False
            self.state = ScanState(running=True)

        threading.Thread(
            target=self._run, args=(roots, reprobe), daemon=True, name="assetkeep-scan"
        ).start()
        return True

    def _run(self, roots: list[Path] | None, reprobe: bool) -> None:
        conn = db.connect(self._config.db_path)
        try:
            stats = scan_module.scan(
                conn,
                config_module.load(self._config.source_path),
                only=roots,
                reprobe=reprobe,
                progress=self._note,
            )
            self.state.absent = stats.absent
            self.state.finished = (
                f"{stats.added} new, {stats.relinked} relinked, "
                f"{stats.unchanged} unchanged, {stats.absent} missing"
            )
        except Exception as exc:  # noqa: BLE001 - reported to the UI, not raised
            log.exception("scan failed")
            self.state.error = f"{type(exc).__name__}: {exc}"
        finally:
            self.state.running = False
            conn.close()
            # Thumbnails were queued as we went; start on them immediately
            # rather than at the worker's next idle poll.
            self._worker.nudge()

    def _note(self, stats, path: Path) -> None:
        self.state.seen = stats.seen
        self.state.added = stats.added
        self.state.hashed = stats.hashed
        self.state.current = path.name


@dataclass
class FetchState:
    """Live progress of a model download running in a background thread."""

    running: bool = False
    #: ``weights`` for the CLIP export, ``ollama`` for the captioning model.
    kind: str = ""
    #: Whatever the fetcher is on right now - a filename, or ollama's own
    #: "pulling manifest" / "verifying sha256" stages.
    label: str = ""
    #: Reported as a pair rather than a percentage, so the UI can say "312 of
    #: 605 MB" instead of a bare number that could mean anything.
    done: int = 0
    total: int = 0
    finished: str | None = None
    error: str | None = None

    def as_dict(self) -> dict:
        return {
            "running": self.running,
            "kind": self.kind,
            "label": self.label,
            "done": self.done,
            "total": self.total,
            "finished": self.finished,
            "error": self.error,
        }


class FetchRunner:
    """Downloads a model off the request thread.

    One runner for both models rather than one each, because they are the same
    shape of work - a slow fetch reporting ``(label, done, total)``, which is
    the signature :func:`assetkeep.tagging.clip.download` and
    :func:`assetkeep.tagging.vlm.pull` already share - and because running both
    at once on a laptop with one network link and 8 GB of memory helps nobody.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.state = FetchState()

    def start(self, kind: str, work: Callable[[Callable], str]) -> bool:
        """Begin a fetch. ``False`` if one is already running."""
        with self._lock:
            if self.state.running:
                return False
            self.state = FetchState(running=True, kind=kind)

        threading.Thread(
            target=self._run, args=(work,), daemon=True, name=f"assetkeep-{kind}"
        ).start()
        return True

    def _run(self, work) -> None:
        try:
            self.state.finished = work(self._note)
        except Exception as exc:  # noqa: BLE001 - reported to the UI, not raised
            log.exception("%s download failed", self.state.kind)
            self.state.error = f"{type(exc).__name__}: {exc}"
        finally:
            self.state.running = False

    def _note(self, label: str, done: int, total: int) -> None:
        self.state.label = label
        self.state.done = done
        self.state.total = total


def create_app(config: Config) -> FastAPI:
    worker = job.Worker(config)
    runner = ScanRunner(config, worker)
    fetcher = FetchRunner()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # Created and migrated once, here, before anything else can open it.
        # Two connections racing to migrate the same empty file both read
        # user_version 0 and both run migration 1, and the second one dies on
        # "table root already exists" - which is what `assetkeep serve` against
        # a database that does not exist yet used to do, since the worker
        # thread and the first request reach it at the same moment. Every other
        # entry point happens to connect once on the main thread first, which
        # is why this only ever showed up on a machine with no index yet.
        db.connect(config.db_path).close()

        # The queue is drained for as long as the server is up, so thumbnails
        # left over from a CLI scan finish without anyone asking.
        worker.start()
        yield
        worker.stop()

    app = FastAPI(title="AssetKeep", version=__version__, lifespan=lifespan)

    def get_conn():
        conn = db.connect(config.db_path, same_thread=False)
        try:
            yield conn
        finally:
            conn.close()

    def current_config() -> Config:
        """The config as it is on disk right now, or as injected if there is none.

        Roots can be added and removed through the UI, which writes the file, so
        anything reading the root list has to re-read rather than trust the
        snapshot taken at startup. Falling back to the injected object matters
        for tests and for a config that has never been saved - otherwise the
        roots that are plainly configured simply do not appear.
        """
        if config.source_path.exists():
            return config_module.load(config.source_path)
        return config

    # --- capabilities -------------------------------------------------------

    @app.get("/api/capabilities")
    def capabilities(conn: sqlite3.Connection = Depends(get_conn)) -> dict:
        model = model3d.capabilities(config.assimp_lib_path)
        semantic = clip.status(config)
        captions = vlm.status(config)
        return {
            "version": __version__,
            "trimesh": bool(model["trimesh"]),
            "assimp": model["assimp"],
            "ffprobe": audio_probe.available(),
            "ffmpeg": thumbs.available(),
            # Two separate facts, and the UI needs both: whether this machine
            # *could* answer a semantic query, and whether anything has actually
            # been embedded yet. A model installed over an unembedded library
            # searches exactly like no model at all.
            "clip": semantic["available"],
            "clip_model": semantic["model"],
            "clip_weights": semantic["weights"],
            "clip_deps": semantic["deps"],
            "embedded": vectors.count(conn),
            # Three facts again, and again because they want different things
            # done about them: no ollama at all, ollama without the model, and
            # a model that is ready. The button is hidden for the first two and
            # says which one it is.
            "vlm": captions["available"],
            "vlm_model": captions["model"],
            "vlm_server": captions["server"],
            "model_formats": model["formats"],
            "copy_target": str(config.copy_target) if config.copy_target else None,
            "thumb_max_edge": config.thumbnails.max_edge,
            "vault_path": str(config.vault_path),
        }

    # --- browsing -----------------------------------------------------------

    @app.get("/api/assets")
    def assets(
        q: str = "",
        limit: int = Query(200, le=1000),
        offset: int = 0,
        conn: sqlite3.Connection = Depends(get_conn),
    ) -> dict:
        rows = search.search(
            conn, q, limit=limit, offset=offset, semantic=clip.ranker(config, conn)
        )
        return {"assets": _decorate(conn, rows), "offset": offset, "limit": limit}

    @app.get("/api/assets/count")
    def asset_count(
        q: str = "", conn: sqlite3.Connection = Depends(get_conn)
    ) -> dict:
        # Counted separately from the page so the grid can size its scrollbar
        # before it has fetched anything.
        sql, params = search.compile_query(
            conn,
            search.parse(q),
            limit=1_000_000,
            semantic=clip.ranker(config, conn),
        )
        total = conn.execute(
            f"SELECT COUNT(*) FROM ({sql})", params
        ).fetchone()[0]
        return {"count": int(total)}

    # --- bulk edits ---------------------------------------------------------
    #
    # These sit above ``/api/assets/{asset_id}`` because FastAPI matches routes
    # in declaration order, and ``bulk`` is a perfectly good string to try to
    # parse as an int. Declared after, every one of these returns 422 from the
    # detail route instead of running - which reads as a malformed request body
    # and is nothing of the sort.

    @app.post("/api/assets/summary")
    def selection_summary(
        payload: dict = Body(...), conn: sqlite3.Connection = Depends(get_conn)
    ) -> dict:
        """What a selection has in common, for the inspector's bulk mode.

        Sent as one request for the whole selection rather than fetching each
        asset, because the interesting selections are the large ones - an
        imported pack, everything a search returned - and the panel has to say
        something true about 400 assets without 400 round trips.

        A field comes back with a value only when every asset agrees on it;
        otherwise ``null``, which the panel renders as "mixed" and leaves alone
        unless it is edited. Tags carry their count, so the difference between
        "all of them" and "nine of forty" is visible rather than implied.
        """
        ids = [int(value) for value in payload.get("ids", [])]
        if not ids:
            return {"count": 0, "tags": [], "kinds": [], "fields": {}}

        placeholders = ",".join("?" * len(ids))
        tags = conn.execute(
            f"""
            SELECT t.name, COUNT(DISTINCT at.asset_id) AS count,
                   MAX(at.source = 'manual') AS manual
            FROM asset_tag at JOIN tag t ON t.id = at.tag_id
            WHERE at.asset_id IN ({placeholders})
            GROUP BY t.id ORDER BY count DESC, t.name LIMIT 60
            """,
            ids,
        ).fetchall()
        kinds = conn.execute(
            f"SELECT kind AS name, COUNT(*) AS count FROM asset "
            f"WHERE id IN ({placeholders}) GROUP BY kind ORDER BY count DESC",
            ids,
        ).fetchall()

        shared = ", ".join(
            f"CASE WHEN COUNT(DISTINCT COALESCE({column}, '')) = 1 "
            f"THEN MAX(COALESCE({column}, '')) END AS {column}"
            for column in ("license", "source_name", "source_url", "notes")
        )
        row = conn.execute(
            f"SELECT {shared} FROM asset WHERE id IN ({placeholders})", ids
        ).fetchone()

        return {
            "count": len(ids),
            "tags": [dict(tag) for tag in tags],
            "kinds": [dict(kind) for kind in kinds],
            "fields": dict(row) if row else {},
        }

    @app.post("/api/assets/bulk/tags")
    def bulk_tags(
        payload: dict = Body(...), conn: sqlite3.Connection = Depends(get_conn)
    ) -> dict:
        """Add and remove manual tags across a selection, in one transaction.

        Both directions in one call because the common edit is exactly that:
        promote forty search results out of ``wip`` and into ``approved``, and
        doing it as two round trips leaves a window where the selection is in
        neither state.
        """
        ids = [int(value) for value in payload.get("ids", [])]
        add = [str(name) for name in payload.get("add", []) if str(name).strip()]
        drop = [str(name) for name in payload.get("remove", []) if str(name).strip()]
        if not ids:
            raise HTTPException(400, "no assets given")

        added = removed = 0
        with conn:
            conn.execute("BEGIN")
            for asset_id in ids:
                if add:
                    added += db.add_tags(
                        conn, asset_id, [(name, "manual", None) for name in add]
                    )
                if drop:
                    removed += db.remove_tags(conn, asset_id, drop)
                if add or drop:
                    db.touch_asset(conn, asset_id)
                    db.index_fts(conn, asset_id)

        return {"assets": len(ids), "added": added, "removed": removed}

    @app.patch("/api/assets/bulk")
    def bulk_metadata(
        payload: dict = Body(...), conn: sqlite3.Connection = Depends(get_conn)
    ) -> dict:
        """Set the same source, licence or notes across a selection.

        This is what an imported pack needs: forty files that share one licence
        and one attribution, which is tedious enough per asset that in practice
        it does not get recorded at all.
        """
        ids = [int(value) for value in payload.get("ids", [])]
        fields = {
            key: value
            for key, value in payload.items()
            if key in db.EDITABLE_FIELDS and key != "title"
        }
        if not ids:
            raise HTTPException(400, "no assets given")
        if not fields:
            raise HTTPException(400, "nothing to set")

        with conn:
            conn.execute("BEGIN")
            for asset_id in ids:
                db.update_asset(conn, asset_id, fields)
                db.index_fts(conn, asset_id)
        return {"assets": len(ids), "fields": sorted(fields)}

    @app.get("/api/assets/{asset_id}")
    def asset_detail(
        asset_id: int, conn: sqlite3.Connection = Depends(get_conn)
    ) -> dict:
        row = conn.execute("SELECT * FROM asset WHERE id = ?", (asset_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "no such asset")

        payload = _decorate(conn, [row])[0]
        payload["tags"] = [
            dict(tag)
            for tag in conn.execute(
                "SELECT t.name, t.namespace, at.source, at.confidence FROM asset_tag at "
                "JOIN tag t ON t.id = at.tag_id WHERE at.asset_id = ? "
                "ORDER BY at.source, t.name",
                (asset_id,),
            )
        ]
        payload["locations"] = [
            dict(loc)
            for loc in conn.execute(
                "SELECT abs_path, size, present FROM location WHERE asset_id = ? "
                "ORDER BY present DESC, id",
                (asset_id,),
            )
        ]
        payload["collections"] = [
            dict(row) for row in collection_module.containing(conn, asset_id)
        ]
        return payload

    @app.patch("/api/assets/{asset_id}")
    def edit_asset(
        asset_id: int,
        payload: dict = Body(...),
        conn: sqlite3.Connection = Depends(get_conn),
    ) -> dict:
        """Editable metadata: title, notes, caption, source and licence."""
        if conn.execute(
            "SELECT 1 FROM asset WHERE id = ?", (asset_id,)
        ).fetchone() is None:
            raise HTTPException(404, "no such asset")

        db.update_asset(conn, asset_id, payload)
        db.index_fts(conn, asset_id)
        return asset_detail(asset_id, conn)

    @app.get("/api/assets/{asset_id}/similar")
    def similar(
        asset_id: int,
        limit: int = Query(60, le=200),
        conn: sqlite3.Connection = Depends(get_conn),
    ) -> dict:
        ids = search.similar_ids(conn, asset_id)[:limit]
        if not ids:
            return {"assets": []}
        placeholders = ",".join("?" * len(ids))
        rows = conn.execute(
            f"SELECT * FROM asset WHERE id IN ({placeholders})", ids
        ).fetchall()
        order = {value: index for index, value in enumerate(ids)}
        rows.sort(key=lambda row: order[row["id"]])
        return {"assets": _decorate(conn, rows)}

    @app.get("/api/facets")
    def facets(
        q: str = "",
        limit: int = Query(60, le=300),
        conn: sqlite3.Connection = Depends(get_conn),
    ) -> dict:
        return {
            "tags": [
                dict(row)
                for row in search.facets(
                    conn, q, limit=limit, semantic=clip.ranker(config, conn)
                )
            ],
            "kinds": [
                dict(row)
                for row in conn.execute(
                    "SELECT kind AS name, COUNT(*) AS count FROM asset "
                    "GROUP BY kind ORDER BY count DESC"
                )
            ],
        }

    # --- media --------------------------------------------------------------

    @app.get("/api/thumb/{content_hash}")
    def thumb(content_hash: str, conn: sqlite3.Connection = Depends(get_conn)):
        """The generated tile, or the original when it was small enough to skip."""
        generated = thumbs.path_for(config, content_hash)
        if generated.exists():
            return FileResponse(generated, media_type="image/webp")

        row = conn.execute(
            "SELECT l.abs_path FROM asset a JOIN location l ON l.asset_id = a.id "
            "WHERE a.content_hash = ? AND l.present = 1 ORDER BY l.id LIMIT 1",
            (content_hash,),
        ).fetchone()
        if row is None or not Path(row["abs_path"]).exists():
            raise HTTPException(404, "no thumbnail and no source")

        source = Path(row["abs_path"])
        return FileResponse(source, media_type=_media_type(source))

    @app.get("/api/file/{asset_id}")
    def raw_file(asset_id: int, conn: sqlite3.Connection = Depends(get_conn)):
        """The asset's own bytes. Backs Quick Look playback and drag-out."""
        source = _present_path(conn, asset_id)
        if source is None:
            raise HTTPException(404, "no present file for this asset")
        return FileResponse(
            source, media_type=_media_type(source), filename=source.name
        )

    # --- retrieval ----------------------------------------------------------

    @app.post("/api/assets/{asset_id}/reveal")
    def reveal(asset_id: int, conn: sqlite3.Connection = Depends(get_conn)) -> dict:
        source = _present_path(conn, asset_id)
        if source is None:
            raise HTTPException(404, "no present file for this asset")
        if sys.platform != "darwin":
            raise HTTPException(501, "reveal is macOS only")
        subprocess.run(["open", "-R", str(source)], check=False)
        return {"revealed": str(source)}

    @app.post("/api/assets/copy")
    def copy_to(
        payload: dict = Body(...), conn: sqlite3.Connection = Depends(get_conn)
    ) -> dict:
        ids = [int(value) for value in payload.get("ids", [])]
        raw_destination = payload.get("destination") or config.copy_target
        if not raw_destination:
            raise HTTPException(400, "no destination, and no copy_target configured")

        destination = Path(str(raw_destination)).expanduser()
        if not destination.is_dir():
            raise HTTPException(400, f"not a directory: {destination}")

        copied, skipped = [], []
        for asset_id in ids:
            source = _present_path(conn, asset_id)
            if source is None:
                skipped.append(asset_id)
                continue
            target = _unique_name(destination / source.name)
            shutil.copy2(source, target)
            copied.append(str(target))

        # Remembering the destination is the whole point of the feature: the
        # common case is copying into the project you are working in right now.
        if payload.get("remember") and str(raw_destination) != str(config.copy_target):
            config_module.save(replace(current_config(), copy_target=destination))

        return {"copied": copied, "skipped": skipped}

    # --- tags ---------------------------------------------------------------

    @app.post("/api/assets/{asset_id}/tags")
    def add_tags(
        asset_id: int,
        payload: dict = Body(...),
        conn: sqlite3.Connection = Depends(get_conn),
    ) -> dict:
        names = [str(name) for name in payload.get("tags", []) if str(name).strip()]
        db.add_tags(conn, asset_id, [(name, "manual", None) for name in names])
        db.touch_asset(conn, asset_id)
        db.index_fts(conn, asset_id)
        return {"tags": db.tag_names(conn, asset_id)}

    @app.delete("/api/assets/{asset_id}/tags/{name}")
    def remove_tag(
        asset_id: int, name: str, conn: sqlite3.Connection = Depends(get_conn)
    ) -> dict:
        db.remove_tags(conn, asset_id, [name])
        db.index_fts(conn, asset_id)
        return {"tags": db.tag_names(conn, asset_id)}

    @app.get("/api/tags")
    def all_tags(
        prefix: str = "",
        limit: int = Query(20, le=200),
        conn: sqlite3.Connection = Depends(get_conn),
    ) -> dict:
        """Tag autocomplete: what already exists, most-used first.

        Ranked with prefix matches ahead of substring ones, then by how many
        assets carry the tag. Suggesting an existing tag is the whole mechanism
        that stops a library growing ``goblin``, ``goblins`` and ``Goblin`` as
        three separate things - the canonicaliser catches the third, and only
        seeing the first two written down catches the second.
        """
        needle = prefix.strip().lower()
        rows = conn.execute(
            """
            SELECT t.name, t.namespace, COUNT(at.asset_id) AS count
            FROM tag t LEFT JOIN asset_tag at ON at.tag_id = t.id
            WHERE ? = '' OR t.name LIKE ?
            GROUP BY t.id
            ORDER BY (t.name LIKE ?) DESC, count DESC, t.name
            LIMIT ?
            """,
            (needle, f"%{needle}%", f"{needle}%", limit),
        ).fetchall()
        return {"tags": [dict(row) for row in rows]}

    # --- collections --------------------------------------------------------

    @app.get("/api/collections")
    def collections(conn: sqlite3.Connection = Depends(get_conn)) -> dict:
        return {
            "collections": [dict(row) for row in collection_module.listing(conn)]
        }

    @app.post("/api/collections")
    def create_collection(
        payload: dict = Body(...), conn: sqlite3.Connection = Depends(get_conn)
    ) -> dict:
        name = str(payload.get("name", "")).strip()
        if not name:
            raise HTTPException(400, "a collection needs a name")
        try:
            collection_id = collection_module.create(
                conn, name, str(payload.get("notes", ""))
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

        ids = [int(value) for value in payload.get("ids", [])]
        if ids:
            collection_module.add(conn, collection_id, ids)
        return dict(collection_module.get(conn, collection_id))

    @app.patch("/api/collections/{collection_id}")
    def edit_collection(
        collection_id: int,
        payload: dict = Body(...),
        conn: sqlite3.Connection = Depends(get_conn),
    ) -> dict:
        row = collection_module.update(
            conn, collection_id, payload.get("name"), payload.get("notes")
        )
        if row is None:
            raise HTTPException(404, "no such collection")
        return dict(row)

    @app.delete("/api/collections/{collection_id}")
    def delete_collection(
        collection_id: int, conn: sqlite3.Connection = Depends(get_conn)
    ) -> dict:
        if not collection_module.delete(conn, collection_id):
            raise HTTPException(404, "no such collection")
        return {"deleted": collection_id}

    @app.post("/api/collections/{collection_id}/assets")
    def add_to_collection(
        collection_id: int,
        payload: dict = Body(...),
        conn: sqlite3.Connection = Depends(get_conn),
    ) -> dict:
        if collection_module.get(conn, collection_id) is None:
            raise HTTPException(404, "no such collection")
        ids = [int(value) for value in payload.get("ids", [])]
        return {"added": collection_module.add(conn, collection_id, ids)}

    @app.delete("/api/collections/{collection_id}/assets")
    def remove_from_collection(
        collection_id: int,
        payload: dict = Body(...),
        conn: sqlite3.Connection = Depends(get_conn),
    ) -> dict:
        ids = [int(value) for value in payload.get("ids", [])]
        return {"removed": collection_module.remove(conn, collection_id, ids)}

    # --- import -------------------------------------------------------------

    @app.post("/api/import")
    async def import_files(
        files: list[UploadFile] = File(...),
        batch: str = Form(""),
        collection: str = Form(""),
        conn: sqlite3.Connection = Depends(get_conn),
    ) -> dict:
        """Take uploaded files into the vault, expanding any archives.

        Each upload is spooled to a temp file before anything looks at it: an
        archive member cannot be read without a seekable source, and hashing for
        the duplicate check needs the bytes on disk anyway. The temp file is
        moved into place rather than copied, so nothing is written twice.
        """
        result = vault.ImportResult()
        current = current_config()

        for upload in files:
            # The name is carried separately and sanitised by the vault, because
            # a dropped folder sends "Pack/Tiles/x.png" as the filename and that
            # is both the structure worth keeping and untrusted input.
            name = upload.filename or "upload"
            with tempfile.TemporaryDirectory(prefix="assetkeep-import-") as staging:
                staged = Path(staging) / "upload"
                with staged.open("wb") as out:
                    while chunk := await upload.read(1024 * 1024):
                        out.write(chunk)

                one = await asyncio.to_thread(
                    vault.import_upload, conn, current, name, staged, batch or None
                )
            result.imported.extend(one.imported)
            result.duplicates.extend(one.duplicates)
            result.skipped.extend(one.skipped)
            result.errors.extend(one.errors)

        if collection.strip() and result.imported:
            collection_id = collection_module.create(conn, collection.strip())
            collection_module.add(conn, collection_id, result.imported)

        # Thumbnails for everything that arrived, without waiting for a scan.
        worker.nudge()
        return result.as_dict()

    # --- references ---------------------------------------------------------

    @app.post("/api/references")
    def add_reference(
        payload: dict = Body(...), conn: sqlite3.Connection = Depends(get_conn)
    ) -> dict:
        """Index a URL, fetching what the page says about itself.

        Synchronous, unlike an import: one page and one preview image is a
        second or two, and a link that appears in the grid only after a
        background job has run is a link somebody adds twice.
        """
        try:
            result = reference_module.add(
                conn,
                current_config(),
                str(payload.get("url", "")),
                title=payload.get("title"),
                notes=payload.get("notes"),
                license=payload.get("license"),
                tags=[str(tag) for tag in payload.get("tags", []) if str(tag).strip()],
                fetch=payload.get("fetch", True),
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

        name = str(payload.get("collection", "")).strip()
        if name:
            collection_id = collection_module.create(conn, name)
            collection_module.add(conn, collection_id, [result.asset_id])

        return {**result.as_dict(), "asset": asset_detail(result.asset_id, conn)}

    @app.post("/api/references/{asset_id}/refresh")
    def refresh_reference(
        asset_id: int,
        payload: dict = Body(default={}),
        conn: sqlite3.Connection = Depends(get_conn),
    ) -> dict:
        result = reference_module.refresh(
            conn, current_config(), asset_id, overwrite=bool(payload.get("overwrite"))
        )
        if result is None:
            raise HTTPException(404, "no such reference")
        return {**result.as_dict(), "asset": asset_detail(asset_id, conn)}

    # --- export -------------------------------------------------------------

    @app.post("/api/collections/{collection_id}/export")
    def export_collection(
        collection_id: int,
        payload: dict = Body(...),
        conn: sqlite3.Connection = Depends(get_conn),
    ) -> dict:
        row = collection_module.get(conn, collection_id)
        if row is None:
            raise HTTPException(404, "no such collection")

        raw_destination = payload.get("destination") or config.copy_target
        if not raw_destination:
            raise HTTPException(400, "no destination, and no copy_target configured")

        try:
            result = export_module.export_collection(
                conn,
                config,
                row["name"],
                Path(str(raw_destination)).expanduser(),
                folder=payload.get("folder"),
                layout=str(payload.get("layout", "flat")),
                manifest=bool(payload.get("manifest", True)),
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

        if payload.get("remember") and str(raw_destination) != str(config.copy_target):
            config_module.save(
                replace(
                    current_config(),
                    copy_target=Path(str(raw_destination)).expanduser(),
                )
            )
        return result.as_dict()

    # --- captions -----------------------------------------------------------

    @app.post("/api/captions")
    def request_captions(
        payload: dict = Body(...), conn: sqlite3.Connection = Depends(get_conn)
    ) -> dict:
        """Queue captions for a selection, and let the worker get on with it.

        Queued rather than run here, which is the opposite of the reference
        endpoint above and for the opposite reason: a caption is seconds of a
        1.7 GB model's time, a selection is often forty of them, and the queue
        and its progress stream already exist to make exactly that bearable.
        """
        ids = [int(value) for value in payload.get("ids", [])]
        if not ids:
            raise HTTPException(400, "no assets given")
        if not vlm.available(config):
            state = vlm.status(config)
            raise HTTPException(
                409,
                f"ollama is not answering at {state['url']}"
                if not state["server"]
                else f"{state['model']} is not installed: assetkeep vlm pull",
            )

        queued = job.enqueue_captions(conn, ids, redo=bool(payload.get("redo")))
        worker.nudge()
        return {"queued": len(queued), "assets": queued}

    # --- models and maintenance ---------------------------------------------
    #
    # The CLI half of this tool could do all of the below and the UI could not,
    # which made "install the optional extras" the one workflow that assumed a
    # terminal. The split was never deliberate: `/api/capabilities` already
    # reported precisely which piece was missing, and then offered no way to go
    # and get it.
    #
    # Two shapes here, chosen by what the work actually is. Fetching a model is
    # a single slow download with byte progress, so it gets a runner and a
    # thread. Embedding and thumbnailing are per-asset work that the job queue
    # and its SSE stream were built for, so those only enqueue and nudge.

    @app.get("/api/maintenance")
    def maintenance(conn: sqlite3.Connection = Depends(get_conn)) -> dict:
        """What the settings panel needs: counts, and what is outstanding.

        Separate from ``/api/capabilities`` because that is cached for the life
        of the page - it answers "what can this machine do", which cannot change
        while the tab is open. These numbers change every time the worker
        finishes something.
        """
        semantic = clip.status(config)
        embedded = vectors.count(conn, semantic["model"])
        outstanding = len(vectors.pending(conn, semantic["model"]))
        captioned = conn.execute(
            "SELECT COUNT(*) FROM asset WHERE caption IS NOT NULL AND caption != ''"
        ).fetchone()[0]
        missing_files = conn.execute(
            "SELECT COUNT(*) FROM asset a WHERE a.kind != 'reference' AND NOT EXISTS "
            "(SELECT 1 FROM location l WHERE l.asset_id = a.id AND l.present = 1)"
        ).fetchone()[0]

        return {
            "assets": conn.execute("SELECT COUNT(*) FROM asset").fetchone()[0],
            "clip_model": semantic["model"],
            "clip_label": semantic["label"],
            "clip_weights": semantic["weights"],
            "clip_deps": semantic["deps"],
            "clip_download_bytes": semantic["download_bytes"],
            "embedded": embedded,
            "outstanding": outstanding,
            "captioned": captioned,
            "missing_files": int(missing_files),
            "queue": job.status(conn).as_dict(),
            "fetch": fetcher.state.as_dict(),
        }

    @app.post("/api/model/download")
    def download_model(payload: dict = Body(default={})) -> dict:
        """Fetch the CLIP weights - the last step before semantic search works."""
        variant = clip.variant_for(
            str(payload.get("model") or "") or config.tagging.clip_model
        )
        absent = clip.missing(config, variant)
        if not absent:
            return {"started": False, "reason": f"{variant.label} is already installed"}

        def work(progress) -> str:
            written = clip.download(config, variant, progress=progress)
            return f"{len(written)} file(s) written"

        if not fetcher.start("weights", work):
            raise HTTPException(409, "a model download is already running")
        return {
            "started": True,
            "label": variant.label,
            "bytes": sum(weight.size for weight in absent),
        }

    @app.post("/api/vlm/pull")
    def pull_vlm(payload: dict = Body(default={})) -> dict:
        """Ask ollama to fetch the captioning model."""
        name = str(payload.get("model") or "").strip() or vlm.model_name(config)
        if vlm.installed_models(config) is None:
            raise HTTPException(
                409,
                f"ollama is not answering at {vlm.endpoint(config)}; "
                "start it with: ollama serve",
            )

        def work(progress) -> str:
            vlm.pull(config, name, progress=progress)
            return f"{name} is installed"

        if not fetcher.start("ollama", work):
            raise HTTPException(409, "a model download is already running")
        return {"started": True, "model": name}

    @app.post("/api/embed")
    def start_embed(
        payload: dict = Body(default={}), conn: sqlite3.Connection = Depends(get_conn)
    ) -> dict:
        """Queue embeddings for whatever has none.

        Only enqueues, where ``assetkeep embed`` also drains: the server has a
        worker running already, and the SSE stream reports it. Doing the work
        here would block a request for the length of a library.
        """
        semantic = clip.status(config)
        if not semantic["deps"]:
            raise HTTPException(409, "the clip runtime is missing: uv sync --extra clip")
        if not semantic["weights"]:
            raise HTTPException(409, "no weights yet - download the model first")

        model = semantic["model"]
        if payload.get("redo"):
            # Everything, rather than only what has no vector. The reason to ask
            # for this is a changed backdrop, preprocessing or model, and none of
            # those show up as a missing row.
            conn.execute("DELETE FROM embedding WHERE model = ?", (model,))

        limit = payload.get("limit")
        outstanding = vectors.pending(
            conn, model, limit=int(limit) if limit else None
        )
        for asset_id in outstanding:
            db.enqueue(conn, "embedding", asset_id)

        worker.nudge()
        return {"queued": len(outstanding), "model": model}

    @app.post("/api/thumbs")
    def start_thumbs(
        payload: dict = Body(default={}), conn: sqlite3.Connection = Depends(get_conn)
    ) -> dict:
        """Queue tiles for anything that should have one and does not.

        Assets that are *meant* to have no tile are skipped rather than queued,
        which is the whole difficulty here: with ``skip_smaller`` on, a 32x32
        sprite legitimately has no file, and "everything without a thumbnail"
        would re-queue every small sprite in the library on every run, report a
        large number, and then do nothing at all with it.
        """
        requeued = job.requeue_failed(conn) if payload.get("retry") else 0
        redo = bool(payload.get("redo"))
        box = config.thumbnails.max_edge

        rows = conn.execute(
            """
            SELECT a.id, a.content_hash, a.kind,
                   MAX(CASE WHEN t.key = 'width' THEN t.value_num END) AS width,
                   MAX(CASE WHEN t.key = 'height' THEN t.value_num END) AS height
            FROM asset a
            LEFT JOIN attribute t ON t.asset_id = a.id AND t.key IN ('width', 'height')
            WHERE a.kind != 'reference'
              AND EXISTS (SELECT 1 FROM location l
                          WHERE l.asset_id = a.id AND l.present = 1)
            GROUP BY a.id
            """
        ).fetchall()

        queued = 0
        for row in rows:
            if not redo and thumbs.path_for(config, row["content_hash"]).exists():
                continue
            if not redo and _skips_thumbnail(config, row, box):
                continue
            db.enqueue(conn, "thumbnail", int(row["id"]))
            queued += 1

        worker.nudge()
        return {"queued": queued, "requeued": requeued}

    @app.post("/api/prune")
    def prune_missing(
        payload: dict = Body(default={}), conn: sqlite3.Connection = Depends(get_conn)
    ) -> dict:
        """Drop assets whose every file has gone.

        Defaults to a dry run, and the UI asks before the real one. This is the
        only destructive operation in the tool - tags, notes, source and licence
        live on the asset and none of it comes back - so the confirmation is
        deliberate rather than ceremonial.
        """
        doomed = scan_module.prune(conn, dry_run=True)
        preview = [{"id": asset_id, "title": title} for asset_id, title in doomed[:20]]
        if not payload.get("confirm"):
            return {"deleted": 0, "count": len(doomed), "preview": preview}

        scan_module.prune(conn)
        return {"deleted": len(doomed), "count": len(doomed), "preview": preview}

    # --- roots and scanning -------------------------------------------------

    @app.get("/api/roots")
    def roots(conn: sqlite3.Connection = Depends(get_conn)) -> dict:
        counts = dict(
            conn.execute(
                "SELECT r.path, COUNT(DISTINCT l.asset_id) FROM root r "
                "LEFT JOIN location l ON l.root_id = r.id AND l.present = 1 "
                "GROUP BY r.id"
            ).fetchall()
        )
        current = current_config()
        return {
            "roots": [
                {
                    "name": root.name,
                    "path": str(root.path),
                    "mode": root.mode,
                    "vendor": root.vendor,
                    "enabled": root.enabled,
                    "count": counts.get(str(root.path), 0),
                }
                for root in current.roots
            ]
        }

    @app.post("/api/roots")
    def add_root(payload: dict = Body(...)) -> dict:
        """Register a folder to index.

        Takes the same options as ``assetkeep root add`` rather than a subset,
        because the point of having this in the UI at all is that adding a root
        should not be the one thing that sends someone to the terminal.
        """
        path = Path(str(payload.get("path", ""))).expanduser().resolve()
        if not path.is_dir():
            raise HTTPException(400, f"not a directory: {path}")

        current = current_config()
        root = RootConfig(
            path=path,
            mode="managed" if payload.get("managed") else "indexed",
            recursive=bool(payload.get("recursive", True)),
            excludes=tuple(str(g) for g in payload.get("exclude", []) if str(g).strip()),
            exclude_defaults=bool(payload.get("exclude_defaults", True)),
            vendor=bool(payload.get("vendor", False)),
        )
        config_module.save(config_module.with_root(current, root))
        return {"added": str(path), "name": root.name}

    @app.delete("/api/roots")
    def remove_root(path: str) -> dict:
        """Stop indexing a folder. Indexed assets are kept, as with the CLI.

        Dropping the rows here as well would be the surprising reading: the
        tags, notes and licences on those assets are the part that cannot be
        rebuilt by rescanning, and ``/api/prune`` is where deleting lives.
        """
        target = Path(path).expanduser().resolve()
        current = current_config()
        if not any(root.path == target for root in current.roots):
            raise HTTPException(404, f"not a configured root: {target}")

        config_module.save(config_module.without_root(current, target))
        return {"removed": str(target)}

    @app.post("/api/scan")
    def start_scan(payload: dict = Body(default={})) -> dict:
        roots = [Path(p) for p in payload.get("roots", [])] or None
        started = runner.start(roots, reprobe=bool(payload.get("reprobe")))
        if not started:
            raise HTTPException(409, "a scan is already running")
        return {"started": True}

    @app.get("/api/scan/status")
    async def scan_status() -> StreamingResponse:
        """Server-sent events carrying scan and queue progress.

        Emits on a timer rather than on every file: at 900 files a second an
        event per file is thousands of DOM updates for a number the eye cannot
        read anyway.
        """

        async def stream():
            # Same reason as get_conn: an async generator is resumed on
            # whichever loop worker is free, not necessarily the one that
            # opened this.
            conn = db.connect(config.db_path, same_thread=False)
            try:
                last: str | None = None
                while True:
                    queue = job.status(conn)
                    queue.running = runner.state.running
                    queue.current = worker.current

                    payload = {
                        "scan": runner.state.as_dict(),
                        "queue": queue.as_dict(),
                        "fetch": fetcher.state.as_dict(),
                    }
                    busy = (
                        runner.state.running
                        or queue.pending > 0
                        or fetcher.state.running
                    )

                    # Emit whenever anything changed, rather than only while
                    # busy. The previous rule - one frame on going idle - looks
                    # equivalent and is not: an idle stream polls every three
                    # seconds, so a scan that starts *and* finishes between two
                    # polls is never once observed as busy, no frame is ever
                    # sent, and the grid sits empty until somebody reloads the
                    # page. That is the ordinary case for a first small root,
                    # which is the worst possible moment for it.
                    body = json.dumps(payload)
                    if body != last:
                        yield f"data: {body}\n\n"
                        last = body
                    else:
                        yield ": heartbeat\n\n"

                    await asyncio.sleep(STATUS_INTERVAL if busy else 3.0)
            finally:
                conn.close()

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/jobs/retry")
    def retry_jobs(conn: sqlite3.Connection = Depends(get_conn)) -> dict:
        count = job.requeue_failed(conn)
        worker.nudge()
        return {"requeued": count}

    if WEB_ROOT.is_dir():
        app.mount("/", StaticFiles(directory=WEB_ROOT, html=True), name="web")

    return app


# --- shared shaping ---------------------------------------------------------

#: Attributes the grid and Quick Look actually read. Fetching all of them for
#: 200 assets is one query; fetching them per asset is 200.
GRID_ATTRIBUTES = (
    "width", "height", "aspect", "has_alpha", "block_size", "cols", "rows",
    "frame_count", "duration", "triangles", "bytes",
    # A reference's shape line is its host, which is the one thing about a link
    # worth reading at thumbnail size.
    "host",
)


def _decorate(conn: sqlite3.Connection, rows) -> list[dict]:
    """Turn asset rows into what the frontend needs, in three queries total."""
    assets = [dict(row) for row in rows]
    if not assets:
        return []

    ids = [asset["id"] for asset in assets]
    placeholders = ",".join("?" * len(ids))

    attributes: dict[int, dict] = {asset_id: {} for asset_id in ids}
    keys = ",".join("?" * len(GRID_ATTRIBUTES))
    for row in conn.execute(
        f"SELECT asset_id, key, value_text, value_num FROM attribute "
        f"WHERE asset_id IN ({placeholders}) AND key IN ({keys})",
        [*ids, *GRID_ATTRIBUTES],
    ):
        value = row["value_text"] if row["value_num"] is None else row["value_num"]
        attributes[row["asset_id"]][row["key"]] = value

    paths: dict[int, str] = {}
    present: dict[int, bool] = {}
    for row in conn.execute(
        f"SELECT asset_id, abs_path, present FROM location "
        f"WHERE asset_id IN ({placeholders}) ORDER BY present DESC, id",
        ids,
    ):
        paths.setdefault(row["asset_id"], row["abs_path"])
        present.setdefault(row["asset_id"], bool(row["present"]))

    for asset in assets:
        asset["attributes"] = attributes.get(asset["id"], {})
        asset["path"] = paths.get(asset["id"])
        # A reference has no location and is not therefore missing: its "file"
        # is a URL, and the grid's missing badge means "the bytes have gone".
        asset["present"] = present.get(
            asset["id"], asset["kind"] == "reference"
        )
    return assets


def _skips_thumbnail(config: Config, row: sqlite3.Row, box: int) -> bool:
    """Whether this asset is meant to have no tile, so a backfill leaves it be.

    Mirrors the rule in :func:`assetkeep.thumbs._image_thumb`: only an image
    skips, and only when it already fits the box. An asset whose dimensions were
    never probed is not assumed to skip - queueing one job too many costs a
    render, and skipping one too many leaves a permanently blank tile.
    """
    if row["kind"] != "image" or not config.thumbnails.skip_smaller:
        return False
    if row["width"] is None or row["height"] is None:
        return False
    return max(int(row["width"]), int(row["height"])) <= box


def _present_path(conn: sqlite3.Connection, asset_id: int) -> Path | None:
    row = conn.execute(
        "SELECT abs_path FROM location WHERE asset_id = ? AND present = 1 "
        "ORDER BY id LIMIT 1",
        (asset_id,),
    ).fetchone()
    if row is None:
        return None
    path = Path(row["abs_path"])
    return path if path.exists() else None


def _media_type(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(path.name)
    if guessed:
        return guessed
    # TGA and EXR are not in the stdlib's table, and a browser given
    # application/octet-stream for an image will download it instead of showing it.
    return {".tga": "image/x-tga", ".exr": "image/x-exr"}.get(
        path.suffix.lower(), "application/octet-stream"
    )


def _unique_name(target: Path) -> Path:
    """Never overwrite at the destination; add ``-1``, ``-2`` as needed."""
    if not target.exists():
        return target
    for index in range(1, 1000):
        candidate = target.with_name(f"{target.stem}-{index}{target.suffix}")
        if not candidate.exists():
            return candidate
    raise HTTPException(409, f"too many files named like {target.name}")
