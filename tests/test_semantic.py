"""The semantic pass, with a fake model standing in for CLIP.

The pass is injected into :func:`assetkeep.search.compile_query` as a callable,
which is what lets these tests exist at all: the ranking is exercised with a
six-line stub that returns whatever the test says, and none of it depends on
having downloaded 149 MB or on a real model's opinion of a synthetic fixture.

What matters here is the merge. Two retrievals with different failure modes are
being combined, and every property worth having - the literal side stays
exhaustive, a filter still filters, an explicit sort still wins - is a property
somebody could break while the search box carried on looking fine.
"""

from __future__ import annotations

import pytest

from assetkeep import db, scan, search, vectors
from assetkeep.config import Config, RootConfig

from . import fixtures


@pytest.fixture
def indexed(tmp_path):
    """A library where the literal and semantic answers deliberately differ."""
    root = tmp_path / "pack"
    (root / "Characters").mkdir(parents=True)
    (root / "Tiles").mkdir()

    fixtures.upscaled_pixel_art(block=4, cells=24).save(root / "Characters/goblin.png")
    fixtures.tiling_texture(128).save(root / "Tiles/stone_wall.png")
    fixtures.two_tone_mask(32).save(root / "Tiles/tiny_mask.png")
    fixtures.single_sprite(64).save(root / "Characters/knight.png")

    cfg = Config(
        db_path=tmp_path / "index.db",
        roots=(RootConfig(path=root, vendor=True),),
        source_path=tmp_path / "config.toml",
    )
    conn = db.connect(cfg.db_path)
    scan.scan(conn, cfg)
    yield conn
    conn.close()


def ident(conn, title: str) -> int:
    return int(
        conn.execute("SELECT id FROM asset WHERE title = ?", (title,)).fetchone()["id"]
    )


def titles(conn, query, **kwargs):
    return [row["title"] for row in search.search(conn, query, **kwargs)]


def ranker(*ranked: tuple[int, float]):
    """A semantic pass that always returns the same ranking."""
    return lambda _text: list(ranked)


# --- what the merge is for ---------------------------------------------------


def test_a_semantic_hit_with_no_literal_match_still_arrives(indexed):
    """The whole point: "knight" finds a file that never says knight."""
    stone = ident(indexed, "stone_wall")
    assert "stone_wall" not in titles(indexed, "knight")

    found = titles(indexed, "goblin", semantic=ranker((stone, 0.31)))
    assert "stone_wall" in found


def test_the_literal_side_stays_exhaustive(indexed):
    """Adding a model must not lose results that worked without one."""
    literal = set(titles(indexed, "goblin"))
    assert literal

    merged = set(titles(indexed, "goblin", semantic=ranker((ident(indexed, "knight"), 0.3))))
    assert literal <= merged


def test_agreement_outranks_a_confident_single_list(indexed):
    """The two passes fail differently, so what they agree on is rarely wrong."""
    goblin = ident(indexed, "goblin")
    stone = ident(indexed, "stone_wall")

    # "goblin" matches goblin literally. The semantic pass puts stone first but
    # also has goblin; goblin appears in both lists and should win.
    found = titles(indexed, "goblin", semantic=ranker((stone, 0.4), (goblin, 0.3)))
    assert found == ["goblin", "stone_wall"]


def test_a_semantic_only_result_ranks_below_a_double_hit(indexed):
    goblin = ident(indexed, "goblin")
    mask = ident(indexed, "tiny_mask")

    found = titles(indexed, "goblin", semantic=ranker((mask, 0.9), (goblin, 0.2)))
    assert found.index("goblin") < found.index("tiny_mask")


def test_no_semantic_pass_leaves_the_old_path_exactly_as_it_was(indexed):
    with_none = titles(indexed, "goblin")
    assert with_none == titles(indexed, "goblin", semantic=None)


def test_an_empty_semantic_ranking_is_not_an_empty_result(indexed):
    """A model that recognises nothing must not delete the literal matches."""
    assert titles(indexed, "goblin", semantic=ranker()) == titles(indexed, "goblin")


def test_a_query_nothing_matches_either_way_is_empty(indexed):
    assert titles(indexed, "xyzzy", semantic=ranker()) == []


# --- filters still filter ----------------------------------------------------


def test_a_filter_still_excludes_a_semantic_hit(indexed):
    """Text ranks; filters filter. A semantic hit is not exempt from tag:."""
    stone = ident(indexed, "stone_wall")
    found = titles(indexed, "goblin tag:pixel-art", semantic=ranker((stone, 0.4)))
    assert "stone_wall" not in found


def test_a_negated_filter_applies_to_the_semantic_side_too(indexed):
    stone = ident(indexed, "stone_wall")
    found = titles(indexed, "goblin -tag:tileable", semantic=ranker((stone, 0.4)))
    assert "stone_wall" not in found


def test_kind_filters_survive_the_merge(indexed):
    stone = ident(indexed, "stone_wall")
    assert titles(indexed, "goblin kind:audio", semantic=ranker((stone, 0.4))) == []


# --- ordering ----------------------------------------------------------------


def test_an_explicit_sort_wins_over_the_fused_rank(indexed):
    stone = ident(indexed, "stone_wall")
    found = titles(indexed, "goblin sort:name", semantic=ranker((stone, 0.4)))
    assert found == sorted(found, key=str.lower)


def test_sort_relevance_is_the_fused_rank(indexed):
    goblin = ident(indexed, "goblin")
    stone = ident(indexed, "stone_wall")
    fused = titles(indexed, "goblin", semantic=ranker((stone, 0.4), (goblin, 0.3)))
    assert titles(indexed, "goblin sort:relevance", semantic=ranker((stone, 0.4), (goblin, 0.3))) == fused


def test_the_default_sort_is_not_treated_as_explicit(indexed):
    """`sort:added` typed by hand and the absence of any sort are different."""
    assert search.parse("goblin").sort_explicit is False
    assert search.parse("goblin sort:added").sort_explicit is True


def test_a_nonsense_sort_is_not_explicit_either(indexed):
    assert search.parse("goblin sort:sideways").sort_explicit is False


def test_paging_is_stable_across_two_identical_queries(indexed):
    stone = ident(indexed, "stone_wall")
    pass_one = titles(indexed, "goblin", semantic=ranker((stone, 0.4)), limit=2)
    pass_two = titles(indexed, "goblin", semantic=ranker((stone, 0.4)), limit=2)
    assert pass_one == pass_two


def test_paging_does_not_repeat_a_row(indexed):
    stone = ident(indexed, "stone_wall")
    first = titles(indexed, "goblin", semantic=ranker((stone, 0.4)), limit=1)
    second = titles(
        indexed, "goblin", semantic=ranker((stone, 0.4)), limit=1, offset=1
    )
    assert first != second


# --- fusion itself -----------------------------------------------------------


def test_fuse_prefers_what_both_lists_hold():
    assert search.fuse([1, 2, 3], [3, 4])[0] == 3


def test_fuse_keeps_a_single_list_in_order():
    assert search.fuse([5, 6, 7]) == [5, 6, 7]


def test_fuse_is_deterministic_under_ties():
    assert search.fuse([1], [2]) == search.fuse([1], [2]) == [1, 2]


def test_fuse_of_nothing_is_nothing():
    assert search.fuse([], []) == []


# --- similar: -----------------------------------------------------------------


def test_similar_uses_embeddings_when_they_exist(indexed):
    goblin = ident(indexed, "goblin")
    knight = ident(indexed, "knight")
    stone = ident(indexed, "stone_wall")

    # Deliberately contradicting the perceptual hash: knight is declared the
    # near neighbour of goblin, stone the opposite.
    vectors.store(indexed, goblin, "test", [1.0, 0.0])
    vectors.store(indexed, knight, "test", [0.98, 0.2])
    vectors.store(indexed, stone, "test", [-1.0, 0.0])

    found = search.similar_ids(indexed, goblin)
    assert found[0] == knight
    assert stone not in found


def test_similar_falls_back_to_the_hash_for_an_unembedded_asset(indexed):
    """Not a degraded mode: it is exactly the M1 behaviour, still working."""
    goblin = ident(indexed, "goblin")
    knight = ident(indexed, "knight")
    vectors.store(indexed, knight, "test", [1.0, 0.0])

    found = search.similar_ids(indexed, goblin)
    assert found and goblin not in found


def test_similar_still_excludes_the_asset_itself_with_embeddings(indexed):
    goblin = ident(indexed, "goblin")
    knight = ident(indexed, "knight")
    vectors.store(indexed, goblin, "test", [1.0, 0.0])
    vectors.store(indexed, knight, "test", [1.0, 0.0])

    assert goblin not in search.similar_ids(indexed, goblin)


def test_similar_ranks_before_a_fused_text_query(indexed):
    """``similar:412 dark`` is a question about 412, so 412's order wins."""
    goblin = ident(indexed, "goblin")
    knight = ident(indexed, "knight")
    stone = ident(indexed, "stone_wall")
    vectors.store(indexed, goblin, "test", [1.0, 0.0])
    vectors.store(indexed, knight, "test", [0.99, 0.1])
    vectors.store(indexed, stone, "test", [0.98, 0.2])

    found = search.search(
        indexed, f"similar:{goblin}", semantic=ranker((stone, 0.9))
    )
    assert [int(row["id"]) for row in found][:2] == [knight, stone]


# --- facets ------------------------------------------------------------------


def test_facets_count_the_same_set_the_grid_shows(indexed):
    stone = ident(indexed, "stone_wall")
    rows = search.facets(indexed, "goblin", semantic=ranker((stone, 0.4)))
    counts = {row["name"]: row["count"] for row in rows}

    shown = titles(indexed, "goblin", semantic=ranker((stone, 0.4)))
    assert counts["pack"] == len(shown)
