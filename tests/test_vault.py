"""Vault import: what gets in, what is refused, and what is not written twice."""

from __future__ import annotations

import zipfile

import pytest

from assetkeep import db, scan, vault
from assetkeep.config import Config

from . import fixtures


@pytest.fixture
def setup(tmp_path):
    """A vault, an index, and a folder of loose files outside both."""
    config = Config(
        db_path=tmp_path / "index.db",
        vault_path=tmp_path / "vault",
        thumbs_path=tmp_path / "thumbs",
        source_path=tmp_path / "config.toml",
    )
    incoming = tmp_path / "incoming"
    (incoming / "TinyPack" / "Tiles").mkdir(parents=True)
    fixtures.upscaled_pixel_art(block=4, cells=16).save(
        incoming / "TinyPack/Tiles/grass.png"
    )
    fixtures.single_sprite(48).save(incoming / "TinyPack/hero.png")
    (incoming / "TinyPack" / "readme.txt").write_text("not an asset")

    conn = db.connect(config.db_path)
    yield config, incoming, conn
    conn.close()


def zip_of(folder, target, arc_root=""):
    with zipfile.ZipFile(target, "w") as archive:
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                inside = path.relative_to(folder)
                archive.write(path, str(arc_root / inside) if arc_root else str(inside))
    return target


# --- placement ---------------------------------------------------------------


def test_a_folder_import_keeps_its_structure(setup):
    config, incoming, conn = setup
    result = vault.import_paths(conn, config, [incoming / "TinyPack"])

    assert len(result.imported) == 2
    assert (config.vault_path / "tiny-pack/Tiles/grass.png").exists()
    assert (config.vault_path / "tiny-pack/hero.png").exists()


def test_only_indexable_extensions_come_across(setup):
    config, incoming, conn = setup
    result = vault.import_paths(conn, config, [incoming / "TinyPack"])

    assert not (config.vault_path / "tiny-pack/readme.txt").exists()
    assert any("readme.txt" in name for name in result.skipped)


def test_a_loose_file_lands_in_the_shared_batch(setup):
    config, incoming, conn = setup
    vault.import_paths(conn, config, [incoming / "TinyPack/hero.png"])
    assert (config.vault_path / vault.DEFAULT_BATCH / "hero.png").exists()


def test_the_import_registers_the_vault_as_a_managed_root(setup):
    config, incoming, conn = setup
    vault.import_paths(conn, config, [incoming / "TinyPack/hero.png"])

    from assetkeep import config as config_module

    saved = config_module.load(config.source_path)
    root = next(r for r in saved.roots if r.path == config.vault_path)
    assert root.mode == "managed"


def test_imported_content_is_searchable_immediately(setup):
    config, incoming, conn = setup
    from assetkeep import search

    vault.import_paths(conn, config, [incoming / "TinyPack"])
    assert len(search.search(conn, "hero")) == 1
    assert len(search.search(conn, "is:managed")) == 2


def test_different_content_with_the_same_name_does_not_overwrite(setup):
    config, incoming, conn = setup
    other = incoming / "Second"
    other.mkdir()
    fixtures.normal_map(32).save(other / "hero.png")

    vault.import_paths(conn, config, [incoming / "TinyPack/hero.png"])
    vault.import_paths(conn, config, [other / "hero.png"])

    landed = sorted(p.name for p in (config.vault_path / vault.DEFAULT_BATCH).iterdir())
    assert landed == ["hero-1.png", "hero.png"]


# --- deduplication -----------------------------------------------------------


def test_the_same_content_twice_is_one_asset_and_one_file(setup):
    config, incoming, conn = setup
    first = vault.import_paths(conn, config, [incoming / "TinyPack"])
    second = vault.import_paths(conn, config, [incoming / "TinyPack"])

    assert len(second.imported) == 0
    assert sorted(second.duplicates) == sorted(first.imported)
    assert len(list(config.vault_path.rglob("*.png"))) == 2


def test_content_whose_file_has_vanished_can_be_imported_again(setup):
    """A known hash with no file left is not a duplicate: re-importing is how
    you get it back."""
    config, incoming, conn = setup
    vault.import_paths(conn, config, [incoming / "TinyPack/hero.png"])

    (config.vault_path / vault.DEFAULT_BATCH / "hero.png").unlink()
    again = vault.import_paths(conn, config, [incoming / "TinyPack/hero.png"])

    assert len(again.imported) == 1
    assert not again.duplicates


# --- archives ----------------------------------------------------------------


def test_a_zip_is_expanded_rather_than_stored(setup, tmp_path):
    config, incoming, conn = setup
    archive = zip_of(incoming / "TinyPack", tmp_path / "TinyPack.zip")

    result = vault.import_paths(conn, config, [archive])

    assert len(result.imported) == 2
    assert not list(config.vault_path.rglob("*.zip"))
    assert (config.vault_path / "tiny-pack/Tiles/grass.png").exists()


def test_a_zips_redundant_top_folder_is_dropped(setup, tmp_path):
    """``Pack.zip/Pack/Tiles/x.png`` would otherwise land two levels deep, and
    folder tags only look two levels up - so the pack's own name falls out of
    range."""
    from pathlib import Path

    config, incoming, conn = setup
    archive = zip_of(
        incoming / "TinyPack", tmp_path / "TinyPack.zip", arc_root=Path("TinyPack")
    )

    vault.import_paths(conn, config, [archive])
    assert (config.vault_path / "tiny-pack/Tiles/grass.png").exists()
    assert not (config.vault_path / "tiny-pack/TinyPack").exists()


def test_the_batch_survives_a_rescan(setup, tmp_path):
    """The fact that these files arrived together is not recoverable from where
    they ended up, so the tag has to be written with a source a rescan keeps."""
    config, incoming, conn = setup
    archive = zip_of(incoming / "TinyPack", tmp_path / "KenneyPlatformer.zip")
    result = vault.import_paths(conn, config, [archive])

    from assetkeep import config as config_module

    scan.scan(conn, config_module.load(config.source_path), reprobe=True)

    tags = db.tag_names(conn, result.imported[0])
    assert "kenney-platformer" in tags


def test_a_zip_slip_member_is_refused(setup, tmp_path):
    """A member named ``../../.zshrc`` writes outside the vault. The archive is
    the one place a filename arrives from somewhere untrusted."""
    config, incoming, conn = setup
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as out:
        out.writestr("../../escaped.png", b"not really a png")
        out.writestr("/absolute.png", b"nor this")

    result = vault.import_paths(conn, config, [archive])

    assert not result.imported
    assert not (tmp_path.parent / "escaped.png").exists()
    assert any("unsafe path" in name for name in result.skipped)


def test_a_nested_archive_is_not_recursed_into(setup, tmp_path):
    config, incoming, conn = setup
    inner = zip_of(incoming / "TinyPack", tmp_path / "inner.zip")
    outer = tmp_path / "outer.zip"
    with zipfile.ZipFile(outer, "w") as out:
        out.write(inner, "inner.zip")

    result = vault.import_paths(conn, config, [outer])

    assert not result.imported
    assert any("nested archive" in name for name in result.skipped)


def test_an_oversized_member_is_skipped(setup, tmp_path, monkeypatch):
    config, incoming, conn = setup
    monkeypatch.setattr(vault, "MAX_MEMBER_BYTES", 10)
    archive = zip_of(incoming / "TinyPack", tmp_path / "TinyPack.zip")

    result = vault.import_paths(conn, config, [archive])

    assert not result.imported
    assert all("too large" in name for name in result.skipped if ".png" in name)


def test_macos_metadata_is_ignored_silently(setup, tmp_path):
    config, incoming, conn = setup
    archive = tmp_path / "TinyPack.zip"
    with zipfile.ZipFile(archive, "w") as out:
        out.write(incoming / "TinyPack/hero.png", "hero.png")
        out.writestr("__MACOSX/._hero.png", b"resource fork")

    result = vault.import_paths(conn, config, [archive])

    assert len(result.imported) == 1
    assert not result.skipped


def test_a_corrupt_archive_is_an_error_not_a_crash(setup, tmp_path):
    config, incoming, conn = setup
    archive = tmp_path / "broken.zip"
    archive.write_bytes(b"PK\x03\x04 and then nonsense")

    result = vault.import_paths(conn, config, [archive])

    assert not result.imported
    assert result.errors


# --- uploads -----------------------------------------------------------------


def test_an_uploaded_folder_path_names_its_own_batch(setup):
    config, incoming, conn = setup
    result = vault.import_upload(
        conn, config, "TinyPack/Tiles/grass.png", incoming / "TinyPack/Tiles/grass.png"
    )

    assert len(result.imported) == 1
    assert (config.vault_path / "tiny-pack/Tiles/grass.png").exists()


def test_an_upload_cannot_escape_the_vault(setup):
    config, incoming, conn = setup
    vault.import_upload(
        conn, config, "../../escaped.png", incoming / "TinyPack/hero.png"
    )

    assert not (config.vault_path.parent.parent / "escaped.png").exists()
    assert list(config.vault_path.rglob("escaped.png"))
