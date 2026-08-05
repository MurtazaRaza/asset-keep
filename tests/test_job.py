"""The queue, and the batched embedding handler that runs on it.

The encoder is a stub throughout. What is being tested is the plumbing around
the model - that a batch reaches the handler as a batch, that one unreadable
file does not fail the other fifteen, that a vector already in the database is
not recomputed - and every one of those is a property of the queue rather than
of CLIP.
"""

from __future__ import annotations

import numpy as np
import pytest

from assetkeep import db, job, vectors
from assetkeep.config import Config
from assetkeep.tagging import clip

from . import fixtures


@pytest.fixture
def cfg(tmp_path):
    return Config(
        db_path=tmp_path / "index.db",
        thumbs_path=tmp_path / "thumbs",
        models_path=tmp_path / "models",
        source_path=tmp_path / "config.toml",
    )


@pytest.fixture
def conn(cfg):
    connection = db.connect(cfg.db_path)
    yield connection
    connection.close()


class StubEncoder:
    """Returns a vector per image without looking at any of them."""

    model_id = "stub"

    def __init__(self):
        self.batches: list[int] = []

    def encode_pixels(self, pixels):
        self.batches.append(len(pixels))
        rows = np.zeros((len(pixels), 4), dtype=np.float32)
        rows[:, 0] = 1.0
        return rows


@pytest.fixture
def encoder(monkeypatch):
    stub = StubEncoder()
    monkeypatch.setattr(clip, "shared_encoder", lambda _cfg: stub)
    monkeypatch.setattr(clip, "tags_for", lambda _e, vectors: [
        [("tileset", "clip", 0.9)] for _ in vectors
    ])
    return stub


def add_image(conn, tmp_path, name: str, kind: str = "image"):
    path = tmp_path / f"{name}.png"
    fixtures.single_sprite(64).save(path)
    asset_id = db.create_asset(conn, kind, f"hash-{name}", name)
    conn.execute(
        "INSERT INTO location (asset_id, abs_path, size, mtime, last_seen, present) "
        "VALUES (?, ?, ?, 1.0, ?, 1)",
        (asset_id, str(path), path.stat().st_size, db.now()),
    )
    return asset_id, path


# --- dispatch ----------------------------------------------------------------


def test_a_batch_reaches_the_handler_as_one_call(conn, cfg, monkeypatch):
    seen = []

    def handler(_conn, _cfg, asset_ids):
        seen.append(list(asset_ids))
        return {asset_id: None for asset_id in asset_ids}

    monkeypatch.setitem(job.BATCH_HANDLERS, "embedding", handler)
    ids = [db.create_asset(conn, "image", f"h{i}", f"t{i}") for i in range(5)]
    for asset_id in ids:
        db.enqueue(conn, "embedding", asset_id)

    job.drain(conn, cfg)
    assert seen == [ids]


def test_an_asset_the_handler_omits_is_retried_not_marked_done(conn, cfg, monkeypatch):
    def handler(_conn, _cfg, asset_ids):
        return {asset_ids[0]: None}

    monkeypatch.setitem(job.BATCH_HANDLERS, "embedding", handler)
    ids = [db.create_asset(conn, "image", f"h{i}", f"t{i}") for i in range(2)]
    for asset_id in ids:
        db.enqueue(conn, "embedding", asset_id)

    job.drain(conn, cfg, limit=2)
    states = dict(
        conn.execute("SELECT asset_id, state FROM job WHERE kind = 'embedding'")
    )
    assert states[ids[0]] == "done"
    assert states[ids[1]] == "pending"


def test_a_handler_that_raises_fails_its_whole_group(conn, cfg, monkeypatch):
    def handler(_conn, _cfg, _asset_ids):
        raise RuntimeError("the model will not load")

    monkeypatch.setitem(job.BATCH_HANDLERS, "embedding", handler)
    for index in range(3):
        db.enqueue(conn, "embedding", db.create_asset(conn, "image", f"h{index}", "t"))

    job.drain(conn, cfg, limit=3)
    rows = conn.execute(
        "SELECT state, attempts, error FROM job WHERE kind = 'embedding'"
    ).fetchall()
    assert all(row["state"] == "pending" for row in rows), "first failure retries"
    assert all("will not load" in row["error"] for row in rows)


def test_a_repeatedly_failing_batch_gives_up(conn, cfg, monkeypatch):
    monkeypatch.setitem(
        job.BATCH_HANDLERS,
        "embedding",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("no")),
    )
    db.enqueue(conn, "embedding", db.create_asset(conn, "image", "h", "t"))

    for _ in range(job.MAX_ATTEMPTS):
        job.drain(conn, cfg, limit=1)

    state = conn.execute("SELECT state FROM job").fetchone()["state"]
    assert state == "failed"


def test_kinds_are_grouped_but_not_reordered(conn, cfg, monkeypatch):
    """A queue of both kinds must not let one starve the other."""
    order = []
    monkeypatch.setitem(
        job.HANDLERS, "thumbnail", lambda _c, _cf, asset_id: order.append("thumb")
    )
    monkeypatch.setitem(
        job.BATCH_HANDLERS,
        "embedding",
        lambda _c, _cf, ids: (order.append("embed"), {i: None for i in ids})[1],
    )
    first = db.create_asset(conn, "image", "h1", "t1")
    db.enqueue(conn, "thumbnail", first)
    db.enqueue(conn, "embedding", first)

    job.drain(conn, cfg)
    assert order == ["thumb", "embed"]


# --- the embedding handler ---------------------------------------------------


def test_embedding_stores_a_vector_and_the_tags_it_implies(conn, cfg, tmp_path, encoder):
    asset_id, _ = add_image(conn, tmp_path, "goblin")
    db.enqueue(conn, "embedding", asset_id)

    job.drain(conn, cfg)

    assert vectors.vector_for(conn, asset_id, "stub") is not None
    tags = conn.execute(
        "SELECT t.name, at.source, at.confidence FROM asset_tag at "
        "JOIN tag t ON t.id = at.tag_id WHERE at.asset_id = ?",
        (asset_id,),
    ).fetchall()
    assert [(row["name"], row["source"]) for row in tags] == [("tileset", "clip")]
    assert tags[0]["confidence"] == pytest.approx(0.9)


def test_the_whole_group_goes_through_the_model_once(conn, cfg, tmp_path, encoder):
    for index in range(4):
        asset_id, _ = add_image(conn, tmp_path, f"sprite{index}")
        db.enqueue(conn, "embedding", asset_id)

    job.drain(conn, cfg)
    assert encoder.batches == [4]


def test_an_existing_vector_is_reused_rather_than_recomputed(
    conn, cfg, tmp_path, encoder
):
    """What makes ``scan --reprobe`` cheap: tags come back, the model does not run."""
    asset_id, _ = add_image(conn, tmp_path, "goblin")
    vectors.store(conn, asset_id, "stub", [0.0, 1.0, 0.0, 0.0])
    db.clear_automated_tags(conn, asset_id)
    db.enqueue(conn, "embedding", asset_id)

    job.drain(conn, cfg)

    assert encoder.batches == [], "the vision tower should not have run"
    assert vectors.vector_for(conn, asset_id, "stub").tolist() == [0.0, 1.0, 0.0, 0.0]
    assert db.tag_names(conn, asset_id) == ["tileset"]


def test_an_unreadable_file_does_not_take_the_batch_with_it(
    conn, cfg, tmp_path, encoder
):
    """Done with no vector, which is what the thumbnail handler already says.

    Nothing is lost by not retrying: ``vectors.pending`` is what remembers, so
    the next ``assetkeep embed`` asks again.
    """
    good, _ = add_image(conn, tmp_path, "good")
    broken = db.create_asset(conn, "image", "hash-broken", "broken")
    corrupt = tmp_path / "broken.png"
    corrupt.write_bytes(b"this is not a png")
    conn.execute(
        "INSERT INTO location (asset_id, abs_path, size, mtime, last_seen, present) "
        "VALUES (?, ?, 17, 1.0, ?, 1)",
        (broken, str(corrupt), db.now()),
    )
    db.enqueue(conn, "embedding", good)
    db.enqueue(conn, "embedding", broken)

    job.drain(conn, cfg, limit=2)

    assert vectors.vector_for(conn, good, "stub") is not None
    assert vectors.vector_for(conn, broken, "stub") is None
    assert encoder.batches == [1], "the good one still went through the model"
    assert broken in vectors.pending(conn, "stub"), "still outstanding, not lost"


def test_an_asset_whose_every_copy_is_gone_is_done_not_failed(
    conn, cfg, tmp_path, encoder
):
    asset_id, path = add_image(conn, tmp_path, "vanished")
    conn.execute("UPDATE location SET present = 0 WHERE asset_id = ?", (asset_id,))
    db.enqueue(conn, "embedding", asset_id)

    job.drain(conn, cfg)

    assert conn.execute("SELECT state FROM job").fetchone()["state"] == "done"
    assert vectors.vector_for(conn, asset_id, "stub") is None


def test_audio_is_never_embedded(conn, cfg, tmp_path, encoder):
    """Its thumbnail is a waveform, and a waveform has no visual meaning."""
    asset_id, _ = add_image(conn, tmp_path, "song", kind="audio")
    db.enqueue(conn, "embedding", asset_id)

    job.drain(conn, cfg)

    assert encoder.batches == []
    assert conn.execute("SELECT state FROM job").fetchone()["state"] == "done"


def test_a_model_is_embedded_through_its_thumbnail(conn, cfg, tmp_path, encoder):
    """CLIP takes pixels and an FBX is not pixels; the render is what it sees."""
    from assetkeep import thumbs

    asset_id, _ = add_image(conn, tmp_path, "sword", kind="model3d")
    # Stand in for the render the thumbnail pipeline would have produced.
    tile = thumbs.path_for(cfg, "hash-sword")
    tile.parent.mkdir(parents=True, exist_ok=True)
    fixtures.single_sprite(64).save(tile, format="WEBP")

    db.enqueue(conn, "embedding", asset_id)
    job.drain(conn, cfg)

    assert encoder.batches == [1]
    assert vectors.vector_for(conn, asset_id, "stub") is not None
