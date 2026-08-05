"""Scan correctness, on the cases that actually break.

Content-addressed identity only pays off if moves, duplicates and disappearances
all behave. Each of these is a case where a path-keyed index would silently do
the wrong thing: lose tags, double-count, or delete work.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from assetkeep import db, hashing, scan
from assetkeep.config import Config, RootConfig

from . import fixtures


@pytest.fixture
def library(tmp_path):
    """A database plus one indexed root, ready to scan."""
    root_dir = tmp_path / "pack"
    root_dir.mkdir()
    cfg = Config(
        db_path=tmp_path / "index.db",
        roots=(RootConfig(path=root_dir),),
        source_path=tmp_path / "config.toml",
    )
    conn = db.connect(cfg.db_path)
    yield conn, cfg, root_dir
    conn.close()


def put(directory: Path, name: str, image=None) -> Path:
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    (image or fixtures.single_sprite()).save(path)
    return path


def assets(conn) -> list:
    return conn.execute("SELECT * FROM asset ORDER BY id").fetchall()


def test_only_known_extensions_are_indexed(library):
    conn, cfg, root = library
    put(root, "hero.png")
    (root / "PlayerController.cs").write_text("class Player {}")
    (root / "hero.png.meta").write_text("guid: abc")

    stats = scan.scan(conn, cfg)

    assert stats.seen == 1
    assert [row["title"] for row in assets(conn)] == ["hero"]


def test_excluded_directories_are_never_entered(library):
    conn, cfg, root = library
    put(root, "Art/hero.png")
    put(root, "Library/metadata/00/cached.png")
    put(root, "Temp/staging.png")

    stats = scan.scan(conn, cfg)

    assert stats.seen == 1
    assert len(assets(conn)) == 1


def test_a_file_moved_between_roots_keeps_its_tags(library, tmp_path):
    conn, cfg, root = library
    second = tmp_path / "project"
    second.mkdir()
    original = put(root, "goblin.png")

    scan.scan(conn, cfg)
    asset_id = int(assets(conn)[0]["id"])
    db.add_tags(conn, asset_id, [("keeper", "manual", None)])

    original.rename(second / "goblin.png")
    cfg = replace(cfg, roots=cfg.roots + (RootConfig(path=second),))
    scan.scan(conn, cfg)

    assert len(assets(conn)) == 1, "the same bytes must not become a second asset"
    assert "keeper" in db.tag_names(conn, asset_id)
    present = conn.execute(
        "SELECT abs_path FROM location WHERE asset_id = ? AND present = 1", (asset_id,)
    ).fetchall()
    assert [row["abs_path"] for row in present] == [str(second / "goblin.png")]


def test_a_duplicate_in_two_locations_stays_one_asset(library):
    conn, cfg, root = library
    sprite = fixtures.single_sprite()
    put(root, "Characters/hero.png", sprite)
    put(root, "Backup/hero.png", sprite)

    scan.scan(conn, cfg)

    assert len(assets(conn)) == 1
    assert conn.execute("SELECT COUNT(*) FROM location").fetchone()[0] == 2


def test_an_unchanged_file_is_never_rehashed(library, monkeypatch):
    conn, cfg, root = library
    put(root, "hero.png")
    scan.scan(conn, cfg)

    calls = []
    monkeypatch.setattr(
        scan.hashing, "hash_file", lambda path: calls.append(path) or "x" * 32
    )
    stats = scan.scan(conn, cfg)

    assert calls == []
    assert stats.unchanged == 1


def test_a_changed_file_is_rehashed(library):
    conn, cfg, root = library
    path = put(root, "hero.png")
    scan.scan(conn, cfg)
    first = assets(conn)[0]["content_hash"]

    fixtures.two_tone_mask().save(path)
    scan.scan(conn, cfg)

    rows = assets(conn)
    assert len(rows) == 2, "new content is new identity, even at the same path"
    assert {row["content_hash"] for row in rows} != {first}


def test_rehash_bypasses_the_fast_path(library, monkeypatch):
    conn, cfg, root = library
    put(root, "hero.png")
    scan.scan(conn, cfg)

    calls = []
    original = hashing.hash_file
    monkeypatch.setattr(
        scan.hashing, "hash_file", lambda path: calls.append(path) or original(path)
    )
    scan.scan(conn, cfg, rehash=True)

    assert len(calls) == 1


def test_a_vanished_file_is_marked_absent_rather_than_deleted(library):
    conn, cfg, root = library
    path = put(root, "goblin.png")
    scan.scan(conn, cfg)
    asset_id = int(assets(conn)[0]["id"])
    db.add_tags(conn, asset_id, [("expensive", "manual", None)])

    path.unlink()
    stats = scan.scan(conn, cfg)

    assert stats.absent == 1
    assert len(assets(conn)) == 1
    assert "expensive" in db.tag_names(conn, asset_id)
    assert conn.execute("SELECT present FROM location").fetchone()["present"] == 0


def test_a_returning_file_is_present_again(library):
    """An unplugged drive is the case this protects."""
    conn, cfg, root = library
    path = put(root, "goblin.png")
    scan.scan(conn, cfg)
    data = path.read_bytes()

    path.unlink()
    scan.scan(conn, cfg)
    path.write_bytes(data)
    scan.scan(conn, cfg)

    assert conn.execute("SELECT present FROM location").fetchone()["present"] == 1


def test_prune_removes_only_what_has_no_file_left(library):
    conn, cfg, root = library
    kept = put(root, "kept.png")
    gone = put(root, "gone.png", fixtures.two_tone_mask())
    scan.scan(conn, cfg)

    gone.unlink()
    scan.scan(conn, cfg)
    removed = scan.prune(conn)

    assert [title for _, title in removed] == ["gone"]
    assert [row["title"] for row in assets(conn)] == ["kept"]
    assert kept.exists()


def test_heuristic_tags_come_from_path_and_root(library):
    conn, cfg, root = library
    put(root, "Characters/Enemies/goblin_walk_01.png")

    scan.scan(conn, cfg)
    names = db.tag_names(conn, int(assets(conn)[0]["id"]))

    assert "goblin" in names and "walk" in names
    assert "enemies" in names and "character" in names
    assert "pack" in names, "the root's own name, as a source: tag"


def test_vendor_roots_tag_their_contents(library):
    conn, cfg, root = library
    put(root, "hero.png")
    cfg = replace(cfg, roots=(replace(cfg.roots[0], vendor=True),))

    scan.scan(conn, cfg)

    assert "vendor" in db.tag_names(conn, int(assets(conn)[0]["id"]))


def test_automated_tags_regenerate_without_touching_manual_ones(library):
    conn, cfg, root = library
    put(root, "hero.png")
    scan.scan(conn, cfg)
    asset_id = int(assets(conn)[0]["id"])
    db.add_tags(conn, asset_id, [("mine", "manual", None)])

    db.clear_automated_tags(conn, asset_id)

    assert db.tag_names(conn, asset_id) == ["mine"]


class TestReprobe:
    """Re-deriving what a probe found, after a capability appears.

    The probe only runs on first sight of a hash, so installing assimp does
    nothing for the 223 FBX files already indexed without it. Deleting the
    index would fix that and throw away every manual tag in the library, which
    is not a trade anyone should be asked to make.
    """

    def test_attributes_are_re_extracted(self, library, monkeypatch):
        conn, cfg, root = library
        put(root, "hero.png")
        scan.scan(conn, cfg)
        conn.execute("DELETE FROM attribute")

        scan.scan(conn, cfg, reprobe=True)

        assert conn.execute(
            "SELECT value_num FROM attribute WHERE key = 'width'"
        ).fetchone()["value_num"] == 96

    def test_manual_tags_survive(self, library):
        conn, cfg, root = library
        put(root, "hero.png")
        scan.scan(conn, cfg)
        asset_id = int(assets(conn)[0]["id"])
        db.add_tags(conn, asset_id, [("expensive", "manual", None)])

        scan.scan(conn, cfg, reprobe=True)

        assert "expensive" in db.tag_names(conn, asset_id)

    def test_stale_automatic_tags_are_dropped(self, library):
        conn, cfg, root = library
        put(root, "hero.png")
        scan.scan(conn, cfg)
        asset_id = int(assets(conn)[0]["id"])
        db.add_tags(conn, asset_id, [("wrong-guess", "heuristic", 0.5)])

        scan.scan(conn, cfg, reprobe=True)

        assert "wrong-guess" not in db.tag_names(conn, asset_id)

    def test_content_in_two_places_keeps_both_sets_of_folder_tags(self, library):
        """Clearing per file rather than per run would lose the first set."""
        conn, cfg, root = library
        sprite = fixtures.single_sprite()
        put(root, "Characters/hero.png", sprite)
        put(root, "Dungeon/hero.png", sprite)
        scan.scan(conn, cfg)
        asset_id = int(assets(conn)[0]["id"])

        scan.scan(conn, cfg, reprobe=True)
        names = db.tag_names(conn, asset_id)

        assert "character" in names and "dungeon" in names

    def test_it_does_not_rehash(self, library, monkeypatch):
        """The fast path already proved the bytes are unchanged."""
        conn, cfg, root = library
        put(root, "hero.png")
        scan.scan(conn, cfg)

        calls = []
        monkeypatch.setattr(
            scan.hashing, "hash_file", lambda path: calls.append(path) or "x" * 32
        )
        stats = scan.scan(conn, cfg, reprobe=True)

        assert calls == []
        assert stats.reprobed == 1


def test_the_same_tag_can_be_both_automatic_and_manual(library):
    """The reason the primary key includes ``source``."""
    conn, cfg, root = library
    put(root, "hero.png")
    scan.scan(conn, cfg)
    asset_id = int(assets(conn)[0]["id"])

    db.add_tags(conn, asset_id, [("hero", "manual", None)])
    db.clear_automated_tags(conn, asset_id)

    assert "hero" in db.tag_names(conn, asset_id)
