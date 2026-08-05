"""Collections: the one thing in the index a rebuild cannot re-derive."""

from __future__ import annotations

import pytest

from assetkeep import collection, db, search


@pytest.fixture
def conn(tmp_path):
    connection = db.connect(tmp_path / "index.db")
    for index in range(5):
        db.create_asset(connection, "image", f"hash{index}", f"asset{index}")
    yield connection
    connection.close()


def test_names_are_canonicalised_so_the_query_bar_can_reach_them(conn):
    """``collection:Jam Prototype`` cannot be written unquoted, and the sidebar
    writes filters unquoted. Canonical names are what keeps clicking and typing
    the same operation."""
    collection.create(conn, "Jam Prototype")
    assert collection.by_name(conn, "jam-prototype")["name"] == "jam-prototype"
    assert collection.by_name(conn, "Jam Prototype") is not None


def test_creating_an_existing_name_returns_it_rather_than_failing(conn):
    first = collection.create(conn, "hero-kit")
    assert collection.create(conn, "Hero Kit") == first


def test_an_empty_name_is_refused(conn):
    with pytest.raises(ValueError):
        collection.create(conn, "   ")


def test_membership_is_ordered_and_appends(conn):
    cid = collection.create(conn, "jam")
    collection.add(conn, cid, [3, 1])
    collection.add(conn, cid, [5])
    assert collection.members(conn, cid) == [3, 1, 5]


def test_re_adding_keeps_the_existing_order(conn):
    """Adding twice means "make sure these are in it", not "move them to the
    end" - the second reading silently destroys a hand-made ordering."""
    cid = collection.create(conn, "jam")
    collection.add(conn, cid, [3, 1, 5])

    assert collection.add(conn, cid, [3, 1]) == 0
    assert collection.members(conn, cid) == [3, 1, 5]


def test_removing_takes_assets_out_without_deleting_them(conn):
    cid = collection.create(conn, "jam")
    collection.add(conn, cid, [1, 2, 3])

    assert collection.remove(conn, cid, [2]) == 1
    assert collection.members(conn, cid) == [1, 3]
    assert conn.execute("SELECT COUNT(*) FROM asset").fetchone()[0] == 5


def test_deleting_a_collection_leaves_its_assets_alone(conn):
    cid = collection.create(conn, "jam")
    collection.add(conn, cid, [1, 2, 3])

    assert collection.delete(conn, cid) is True
    assert conn.execute("SELECT COUNT(*) FROM asset").fetchone()[0] == 5
    assert conn.execute("SELECT COUNT(*) FROM collection_asset").fetchone()[0] == 0


def test_deleting_an_asset_takes_it_out_of_its_collections(conn):
    """The ON DELETE CASCADE only fires with foreign keys on, which SQLite
    leaves off by default."""
    cid = collection.create(conn, "jam")
    collection.add(conn, cid, [1, 2])

    conn.execute("DELETE FROM asset WHERE id = 1")
    assert collection.members(conn, cid) == [2]


def test_listing_carries_sizes(conn):
    empty = collection.create(conn, "empty")
    full = collection.create(conn, "full")
    collection.add(conn, full, [1, 2, 3])

    sizes = {row["name"]: row["count"] for row in collection.listing(conn)}
    assert sizes == {"empty": 0, "full": 3}
    assert {empty, full}


def test_the_search_grammar_finds_members_by_either_spelling(conn):
    cid = collection.create(conn, "Jam Prototype")
    collection.add(conn, cid, [2, 4])

    for query in ("collection:jam-prototype", "collection:JamPrototype"):
        found = {int(row["id"]) for row in search.search(conn, query)}
        assert found == {2, 4}, query


def test_containing_reports_every_collection_an_asset_is_in(conn):
    first = collection.create(conn, "one")
    second = collection.create(conn, "two")
    collection.add(conn, first, [1])
    collection.add(conn, second, [1])

    assert [row["name"] for row in collection.containing(conn, 1)] == ["one", "two"]
