"""Embedding storage and the arithmetic over it.

No model anywhere in this file, deliberately. Every failure these tests can
catch - a vector stored unnormalised, a stale cache, neighbours ranked the wrong
way up - is a failure that would otherwise be discovered as "semantic search
gives odd results", which is indistinguishable from "CLIP is like that".
"""

from __future__ import annotations

import numpy as np
import pytest

from assetkeep import db, vectors
from assetkeep.config import Config


@pytest.fixture
def conn(tmp_path):
    cfg = Config(db_path=tmp_path / "index.db", source_path=tmp_path / "config.toml")
    connection = db.connect(cfg.db_path)
    yield connection
    connection.close()


def add_asset(conn, title: str, kind: str = "image", present: bool = True) -> int:
    asset_id = db.create_asset(conn, kind, f"hash-{title}", title)
    conn.execute(
        "INSERT INTO location (asset_id, abs_path, size, mtime, last_seen, present) "
        "VALUES (?, ?, 1, 1.0, ?, ?)",
        (asset_id, f"/library/{title}.png", db.now(), int(present)),
    )
    return asset_id


def test_vectors_are_stored_normalised(conn):
    asset_id = add_asset(conn, "goblin")
    vectors.store(conn, asset_id, "test", [3.0, 4.0, 0.0])

    stored = vectors.vector_for(conn, asset_id, "test")
    assert pytest.approx(float(np.linalg.norm(stored)), abs=1e-6) == 1.0
    assert stored.tolist() == pytest.approx([0.6, 0.8, 0.0], abs=1e-6)


def test_storing_twice_replaces_rather_than_duplicating(conn):
    asset_id = add_asset(conn, "goblin")
    vectors.store(conn, asset_id, "test", [1.0, 0.0])
    vectors.store(conn, asset_id, "test", [0.0, 1.0])

    assert vectors.count(conn, "test") == 1
    assert vectors.vector_for(conn, asset_id, "test").tolist() == [0.0, 1.0]


def test_two_models_coexist_on_one_asset(conn):
    asset_id = add_asset(conn, "goblin")
    vectors.store(conn, asset_id, "old", [1.0, 0.0])
    vectors.store(conn, asset_id, "new", [0.0, 1.0])

    assert vectors.count(conn) == 2
    assert vectors.vector_for(conn, asset_id, "old").tolist() == [1.0, 0.0]


def test_pending_skips_what_is_already_embedded(conn):
    first = add_asset(conn, "a")
    second = add_asset(conn, "b")
    vectors.store(conn, first, "test", [1.0, 0.0])

    assert vectors.pending(conn, "test") == [second]


def test_pending_skips_kinds_with_nothing_to_look_at(conn):
    add_asset(conn, "song", kind="audio")
    model = add_asset(conn, "sword", kind="model3d")

    assert vectors.pending(conn, "test") == [model]


def test_pending_skips_assets_whose_every_copy_is_gone(conn):
    add_asset(conn, "vanished", present=False)
    here = add_asset(conn, "here")

    assert vectors.pending(conn, "test") == [here]


def test_rank_orders_by_cosine(conn):
    ids = [add_asset(conn, name) for name in ("east", "north", "diagonal")]
    vectors.store(conn, ids[0], "test", [1.0, 0.0])
    vectors.store(conn, ids[1], "test", [0.0, 1.0])
    vectors.store(conn, ids[2], "test", [1.0, 1.0])

    ranked = vectors.rank(conn, "test", [1.0, 0.0], floor=-1.0)
    assert [asset_id for asset_id, _ in ranked] == [ids[0], ids[2], ids[1]]
    assert ranked[0][1] == pytest.approx(1.0)
    assert ranked[2][1] == pytest.approx(0.0, abs=1e-6)


def test_rank_applies_the_floor(conn):
    ids = [add_asset(conn, name) for name in ("east", "north")]
    vectors.store(conn, ids[0], "test", [1.0, 0.0])
    vectors.store(conn, ids[1], "test", [0.0, 1.0])

    assert [i for i, _ in vectors.rank(conn, "test", [1.0, 0.0], floor=0.5)] == [ids[0]]


def test_rank_refuses_to_compare_across_dimensions(conn):
    """A model swapped under a database still holding the old vectors.

    Comparing a 512-vector against a 768-vector is not an error numpy would
    raise on every path, and the numbers it does produce look like scores.
    """
    asset_id = add_asset(conn, "goblin")
    vectors.store(conn, asset_id, "test", [1.0, 0.0, 0.0])

    assert vectors.rank(conn, "test", [1.0, 0.0], floor=-1.0) == []


def test_neighbours_exclude_the_asset_itself(conn):
    ids = [add_asset(conn, name) for name in ("a", "b")]
    vectors.store(conn, ids[0], "test", [1.0, 0.0])
    vectors.store(conn, ids[1], "test", [1.0, 0.0])

    found = vectors.neighbours(conn, "test", ids[0], floor=-1.0)
    assert [asset_id for asset_id, _ in found] == [ids[1]]


def test_neighbours_of_an_unembedded_asset_are_empty_not_wrong(conn):
    """The signal search uses to fall back to the perceptual hash."""
    embedded = add_asset(conn, "a")
    bare = add_asset(conn, "b")
    vectors.store(conn, embedded, "test", [1.0, 0.0])

    assert vectors.neighbours(conn, "test", bare) == []


def test_primary_model_is_the_one_most_of_the_library_uses(conn):
    ids = [add_asset(conn, name) for name in ("a", "b", "c")]
    vectors.store(conn, ids[0], "old", [1.0, 0.0])
    vectors.store(conn, ids[1], "new", [1.0, 0.0])
    vectors.store(conn, ids[2], "new", [0.0, 1.0])

    assert vectors.primary_model(conn) == "new"


def test_primary_model_is_none_before_anything_is_embedded(conn):
    add_asset(conn, "a")
    assert vectors.primary_model(conn) is None


def test_the_matrix_notices_a_new_row(conn):
    first = add_asset(conn, "a")
    vectors.store(conn, first, "test", [1.0, 0.0])
    assert vectors.matrix(conn, "test")[0].tolist() == [first]

    second = add_asset(conn, "b")
    vectors.store(conn, second, "test", [0.0, 1.0])
    assert vectors.matrix(conn, "test")[0].tolist() == [first, second]


def test_the_matrix_notices_a_replaced_vector(conn):
    """Replacing in place changes neither the row count nor the highest rowid.

    Which is why there is no cache here. The obvious one keyed on that pair
    served stale vectors, silently, and the symptom would have read as the
    model being unreliable rather than as a bug.
    """
    asset_id = add_asset(conn, "a")
    vectors.store(conn, asset_id, "test", [1.0, 0.0])
    assert vectors.matrix(conn, "test")[1][0].tolist() == [1.0, 0.0]

    vectors.store(conn, asset_id, "test", [0.0, 1.0])
    assert vectors.matrix(conn, "test")[1][0].tolist() == [0.0, 1.0]


def test_deleting_an_asset_takes_its_embedding_with_it(conn):
    asset_id = add_asset(conn, "goblin")
    vectors.store(conn, asset_id, "test", [1.0, 0.0])

    conn.execute("DELETE FROM asset WHERE id = ?", (asset_id,))
    assert vectors.count(conn, "test") == 0
