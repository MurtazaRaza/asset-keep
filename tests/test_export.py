"""Export: what lands in the project folder, and what is written beside it."""

from __future__ import annotations

import json

import pytest

from assetkeep import collection as collection_module, db, export, reference, scan
from assetkeep.config import Config, RootConfig
from assetkeep.probe import reference as reference_probe
from assetkeep.probe.reference import Page

from . import fixtures


@pytest.fixture
def library(tmp_path):
    """A scanned root, a collection over it, and somewhere to export to."""
    source = tmp_path / "pack"
    (source / "Tiles").mkdir(parents=True)
    fixtures.upscaled_pixel_art(block=4, cells=16).save(source / "Tiles/grass.png")
    fixtures.single_sprite(48).save(source / "hero.png")
    fixtures.normal_map(32).save(source / "Tiles/wall_normal.png")

    config = Config(
        db_path=tmp_path / "index.db",
        vault_path=tmp_path / "vault",
        thumbs_path=tmp_path / "thumbs",
        models_path=tmp_path / "models",
        source_path=tmp_path / "config.toml",
        roots=(RootConfig(path=source),),
    )
    destination = tmp_path / "Unity" / "Assets"
    destination.mkdir(parents=True)

    conn = db.connect(config.db_path)
    scan.scan(conn, config)
    ids = [int(row["id"]) for row in conn.execute("SELECT id FROM asset ORDER BY id")]
    collection_id = collection_module.create(conn, "Jam Prototype", "for the jam")
    collection_module.add(conn, collection_id, ids)

    yield config, conn, destination, ids
    conn.close()


# --- placement ----------------------------------------------------------------


def test_a_collection_lands_in_a_folder_named_after_it(library):
    config, conn, destination, ids = library
    result = export.export_collection(conn, config, "jam-prototype", destination)

    assert result.destination == destination / "jam-prototype"
    assert len(result.copied) == 3
    assert sorted(p.name for p in result.copied) == [
        "grass.png", "hero.png", "wall_normal.png",
    ]


def test_the_folder_can_be_turned_off(library):
    config, conn, destination, ids = library
    result = export.export_collection(conn, config, "jam-prototype", destination, folder="")

    assert result.destination == destination
    assert (destination / "hero.png").exists()


def test_the_kind_layout_splits_into_subfolders(library):
    config, conn, destination, ids = library
    result = export.export_collection(
        conn, config, "jam-prototype", destination, layout="kind"
    )

    assert (result.destination / "Images" / "hero.png").exists()


def test_an_unknown_collection_is_an_error(library):
    config, conn, destination, ids = library
    with pytest.raises(ValueError):
        export.export_collection(conn, config, "nope", destination)


def test_a_destination_that_is_not_a_directory_is_an_error(library, tmp_path):
    config, conn, destination, ids = library
    with pytest.raises(ValueError):
        export.export(conn, config, ids, tmp_path / "does-not-exist")


# --- running it twice ---------------------------------------------------------


def test_exporting_twice_copies_nothing_the_second_time(library):
    """The expensive mistake is a folder of tile-1.png, tile-2.png after three
    exports of a collection that gained one asset."""
    config, conn, destination, ids = library
    export.export_collection(conn, config, "jam-prototype", destination)
    again = export.export_collection(conn, config, "jam-prototype", destination)

    assert not again.copied
    assert len(again.unchanged) == 3
    assert len(list((destination / "jam-prototype").glob("*.png"))) == 3


def test_a_name_clash_with_different_content_keeps_both(library, tmp_path):
    config, conn, destination, ids = library
    target = destination / "jam-prototype"
    target.mkdir()
    fixtures.two_tone_mask(24).save(target / "hero.png")

    result = export.export_collection(conn, config, "jam-prototype", destination)

    assert (target / "hero.png").exists() and (target / "hero-1.png").exists()
    assert any(path.name == "hero-1.png" for path in result.copied)


def test_a_file_that_has_vanished_is_reported_not_fatal(library):
    config, conn, destination, ids = library
    path = conn.execute(
        "SELECT abs_path FROM location WHERE asset_id = ?", (ids[0],)
    ).fetchone()["abs_path"]
    __import__("pathlib").Path(path).unlink()

    result = export.export_collection(conn, config, "jam-prototype", destination)

    assert result.missing == [ids[0]]
    assert len(result.copied) == 2


# --- the written record -------------------------------------------------------


def test_the_manifest_carries_what_a_folder_of_files_loses(library):
    config, conn, destination, ids = library
    db.update_asset(
        conn, ids[0], {"license": "cc0", "source_name": "Kenney",
                       "source_url": "https://kenney.nl"}
    )
    result = export.export_collection(conn, config, "jam-prototype", destination)

    manifest = json.loads(result.manifest.read_text())
    assert manifest["collection"] == "jam-prototype"
    entry = next(item for item in manifest["assets"] if item["id"] == ids[0])
    assert entry["license"] == "cc0"
    assert entry["source_name"] == "Kenney"
    assert entry["file"] and entry["tags"]


def test_credits_group_by_source_and_licence(library):
    config, conn, destination, ids = library
    for asset_id in ids[:2]:
        db.update_asset(
            conn, asset_id, {"license": "cc0", "source_name": "Kenney"}
        )
    result = export.export_collection(conn, config, "jam-prototype", destination)

    text = result.credits.read_text()
    assert "## Kenney" in text
    assert "Licence: cc0" in text
    # The third asset has neither, and saying so is the point: an export with
    # unrecorded licensing is something to go and fix.
    assert "## No source or licence recorded" in text


def test_the_manifest_can_be_skipped(library):
    config, conn, destination, ids = library
    result = export.export_collection(
        conn, config, "jam-prototype", destination, manifest=False
    )

    assert result.manifest is None
    assert not (result.destination / export.MANIFEST_NAME).exists()


# --- references ---------------------------------------------------------------


def test_a_reference_is_written_down_rather_than_copied(library, monkeypatch):
    config, conn, destination, ids = library
    monkeypatch.setattr(
        reference_probe,
        "fetch",
        lambda url, timeout=None: Page(
            url=url, title="Platformer Pack Redux", site_name="Kenney"
        ),
    )
    monkeypatch.setattr(reference_probe, "fetch_image", lambda url, timeout=None: None)

    link = reference.add(conn, config, "https://kenney.nl/assets/platformer-pack-redux")
    collection_id = collection_module.by_name(conn, "jam-prototype")["id"]
    collection_module.add(conn, collection_id, [link.asset_id])

    result = export.export_collection(conn, config, "jam-prototype", destination)

    assert result.references == [link.asset_id]
    assert len(result.copied) == 3
    credits = result.credits.read_text()
    assert "## References" in credits
    assert "https://kenney.nl/assets/platformer-pack-redux" in credits
