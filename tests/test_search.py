"""The query grammar, from parse through to rows out of SQLite."""

from __future__ import annotations

import pytest

from assetkeep import db, scan, search
from assetkeep.config import Config, RootConfig

from . import fixtures


@pytest.fixture
def indexed(tmp_path):
    """A small library with enough variety for every clause to bite."""
    root = tmp_path / "pack"
    (root / "Characters").mkdir(parents=True)
    (root / "Tiles").mkdir()

    fixtures.upscaled_pixel_art(block=4, cells=24).save(root / "Characters/goblin.png")
    fixtures.tiling_texture(128).save(root / "Tiles/stone_wall.png")
    fixtures.two_tone_mask(32).save(root / "Tiles/tiny_mask.png")

    cfg = Config(
        db_path=tmp_path / "index.db",
        roots=(RootConfig(path=root, vendor=True),),
        source_path=tmp_path / "config.toml",
    )
    conn = db.connect(cfg.db_path)
    scan.scan(conn, cfg)
    yield conn
    conn.close()


def titles(conn, query, **kwargs):
    return sorted(row["title"] for row in search.search(conn, query, **kwargs))


def test_bare_words_hit_the_full_text_index(indexed):
    assert titles(indexed, "goblin") == ["goblin"]
    assert titles(indexed, "stone") == ["stone_wall"]


def test_filenames_are_searchable_not_just_titles(indexed):
    assert titles(indexed, "wall") == ["stone_wall"]


def test_tag_include_and_exclude(indexed):
    assert "goblin" in titles(indexed, "tag:pixel-art")
    assert "goblin" not in titles(indexed, "-tag:pixel-art")


def test_numeric_comparison_against_attributes(indexed):
    assert titles(indexed, "w:>=96") == ["goblin", "stone_wall"]
    assert titles(indexed, "w:<64") == ["tiny_mask"]


def test_size_units_are_understood(indexed):
    assert titles(indexed, "size:<1mb") == ["goblin", "stone_wall", "tiny_mask"]
    assert titles(indexed, "size:>1gb") == []


def test_kind_and_is_vendor(indexed):
    assert len(titles(indexed, "kind:image")) == 3
    assert titles(indexed, "kind:audio") == []
    assert len(titles(indexed, "is:vendor")) == 3


def test_root_filter_uses_the_canonical_root_name(indexed):
    assert len(titles(indexed, "root:pack")) == 3


def test_is_missing_finds_nothing_while_the_files_are_there(indexed):
    assert titles(indexed, "is:missing") == []


def test_has_alpha(indexed):
    """The pixel-art fixture is opaque; only real transparency should match."""
    assert titles(indexed, "has:alpha") == []


def test_clauses_combine_without_duplicating_rows(indexed):
    """Two tag filters as JOINs would return each asset once per tag."""
    rows = search.search(indexed, "tag:pixel-art tag:character")
    assert len(rows) == len({row["id"] for row in rows})


def test_sorting(indexed):
    by_name = [row["title"] for row in search.search(indexed, "sort:name")]
    assert by_name == sorted(by_name, key=str.lower)


def test_an_fts_operator_typed_as_a_word_is_not_a_syntax_error(indexed):
    """``NOT``, ``*`` and ``(`` are things people type, not query syntax."""
    assert search.search(indexed, "goblin NOT") == []
    assert search.search(indexed, "AND OR *") == []


def test_unknown_prefixes_degrade_to_words(indexed):
    assert search.parse("colour:red").text == ("colour:red",)


def test_similar_finds_the_asset_itself_is_excluded(indexed):
    target = search.search(indexed, "goblin")[0]["id"]
    assert target not in search.similar_ids(indexed, int(target))


def test_facets_count_tags_within_the_result_set(indexed):
    counts = {row["name"]: row["count"] for row in search.facets(indexed, "kind:image")}
    assert counts["pack"] == 3, "every asset carries the root's source tag"
    assert counts.get("pixel-art", 0) >= 1


def test_serialise_round_trips(indexed):
    query = "dark tag:tileset -tag:wip w:>=512 sort:name"
    assert search.serialise(search.parse(query)) == query
