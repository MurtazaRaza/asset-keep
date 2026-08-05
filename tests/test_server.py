"""HTTP surface: shapes the frontend depends on, and the retrieval paths."""

from __future__ import annotations

import pytest
from pathlib import Path

from fastapi.testclient import TestClient

from assetkeep import db, scan, thumbs
from assetkeep.config import Config, RootConfig
from assetkeep.server import create_app

from . import fixtures


@pytest.fixture
def client(tmp_path):
    root = tmp_path / "pack"
    (root / "Characters").mkdir(parents=True)
    fixtures.upscaled_pixel_art(block=4, cells=24).save(root / "Characters/goblin.png")
    fixtures.organic_texture(512).save(root / "Characters/wall.png")
    fixtures.two_tone_mask(32).save(root / "tiny.png")

    config = Config(
        db_path=tmp_path / "index.db",
        thumbs_path=tmp_path / "thumbs",
        vault_path=tmp_path / "vault",
        roots=(RootConfig(path=root, vendor=True),),
        source_path=tmp_path / "config.toml",
    )
    conn = db.connect(config.db_path)
    scan.scan(conn, config)
    conn.close()

    with TestClient(create_app(config)) as test_client:
        test_client.config = config
        yield test_client


def test_capabilities_reports_what_is_installed(client):
    payload = client.get("/api/capabilities").json()
    assert set(payload) >= {"trimesh", "assimp", "ffprobe", "ffmpeg", "clip"}
    assert isinstance(payload["clip"], bool)


def test_capabilities_separates_the_model_from_the_embeddings(client):
    """A model installed over a library nothing has embedded searches like no
    model at all, so the UI has to be able to tell those apart."""
    payload = client.get("/api/capabilities").json()
    assert payload["clip"] is False, "no weights are downloaded in a test"
    assert payload["clip_model"]
    assert payload["clip_weights"] is False
    assert payload["embedded"] == 0


def test_search_still_works_with_no_model_installed(client):
    """The whole optional tier absent is the ordinary case, not a broken one."""
    hits = client.get("/api/assets", params={"q": "goblin"}).json()["assets"]
    assert [asset["title"] for asset in hits] == ["goblin"]


def test_assets_carry_everything_the_grid_needs(client):
    payload = client.get("/api/assets").json()
    assert len(payload["assets"]) == 3

    asset = next(a for a in payload["assets"] if a["title"] == "goblin")
    assert asset["content_hash"]
    assert asset["path"].endswith("goblin.png")
    assert asset["present"] is True
    assert asset["attributes"]["width"] == 96


def test_the_query_grammar_reaches_the_endpoint(client):
    """``tiny`` is genuinely pixel art too: 32 px and two colours. ``wall`` is
    a 512 px organic texture and must not be."""
    hits = client.get("/api/assets", params={"q": "tag:pixel-art"}).json()["assets"]
    assert {asset["title"] for asset in hits} == {"goblin", "tiny"}

    narrowed = client.get(
        "/api/assets", params={"q": "tag:pixel-art w:>=64"}
    ).json()["assets"]
    assert {asset["title"] for asset in narrowed} == {"goblin"}


def test_count_matches_the_result_set(client):
    for query in ("", "kind:image", "tag:pixel-art", "w:>=512"):
        listed = client.get("/api/assets", params={"q": query}).json()["assets"]
        counted = client.get("/api/assets/count", params={"q": query}).json()["count"]
        assert counted == len(listed), query


def test_facets_count_within_the_current_query(client):
    payload = client.get("/api/facets", params={"q": "kind:image"}).json()
    names = {tag["name"]: tag["count"] for tag in payload["tags"]}
    assert names["pack"] == 3
    assert {kind["name"] for kind in payload["kinds"]} == {"image"}


def test_asset_detail_includes_tags_and_locations(client):
    asset_id = client.get("/api/assets").json()["assets"][0]["id"]
    payload = client.get(f"/api/assets/{asset_id}").json()

    assert payload["tags"] and {"name", "source"} <= set(payload["tags"][0])
    assert payload["locations"] and payload["locations"][0]["present"] == 1


def test_a_missing_asset_is_a_404(client):
    assert client.get("/api/assets/99999").status_code == 404


def test_thumb_falls_back_to_the_original_when_too_small_to_generate(client):
    """The skip-small rule means most sprites have no thumbnail file at all."""
    assets = client.get("/api/assets", params={"q": "tiny"}).json()["assets"]
    digest = assets[0]["content_hash"]
    assert not thumbs.path_for(client.config, digest).exists()

    response = client.get(f"/api/thumb/{digest}")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"


def test_thumb_serves_the_generated_webp_when_there_is_one(client):
    asset = next(
        a
        for a in client.get("/api/assets").json()["assets"]
        if a["attributes"].get("width") == 512
    )
    thumbs.generate(
        client.config, "image", Path(asset["path"]), asset["content_hash"]
    )

    response = client.get(f"/api/thumb/{asset['content_hash']}")
    assert response.headers["content-type"] == "image/webp"


def test_raw_file_serves_the_original_bytes(client):
    asset = client.get("/api/assets", params={"q": "goblin"}).json()["assets"][0]
    response = client.get(f"/api/file/{asset['id']}")

    assert response.status_code == 200
    assert response.content == open(asset["path"], "rb").read()


def test_tags_can_be_added_and_removed(client):
    asset_id = client.get("/api/assets").json()["assets"][0]["id"]

    added = client.post(f"/api/assets/{asset_id}/tags", json={"tags": ["favourite"]})
    assert "favourite" in added.json()["tags"]

    removed = client.delete(f"/api/assets/{asset_id}/tags/favourite")
    assert "favourite" not in removed.json()["tags"]


def test_manual_tags_are_searchable_immediately(client):
    """The FTS row has to be rebuilt on write, or the tag is invisible."""
    asset_id = client.get("/api/assets").json()["assets"][0]["id"]
    client.post(f"/api/assets/{asset_id}/tags", json={"tags": ["unmistakable"]})

    hits = client.get("/api/assets", params={"q": "tag:unmistakable"}).json()
    assert len(hits["assets"]) == 1


def test_copy_writes_the_files_and_never_overwrites(client, tmp_path):
    destination = tmp_path / "out"
    destination.mkdir()
    ids = [asset["id"] for asset in client.get("/api/assets").json()["assets"]]

    first = client.post(
        "/api/assets/copy", json={"ids": ids, "destination": str(destination)}
    ).json()
    second = client.post(
        "/api/assets/copy", json={"ids": ids, "destination": str(destination)}
    ).json()

    assert len(first["copied"]) == 3
    assert len(list(destination.iterdir())) == 6, "the second copy must not clobber"
    assert all("-1" in name for name in second["copied"])


def test_copy_to_a_nonexistent_folder_is_a_400(client):
    assert (
        client.post(
            "/api/assets/copy", json={"ids": [1], "destination": "/no/such/place"}
        ).status_code
        == 400
    )


def test_roots_are_listed_with_their_counts(client):
    roots = client.get("/api/roots").json()["roots"]
    assert len(roots) == 1
    assert roots[0]["count"] == 3
    assert roots[0]["vendor"] is True


def test_similar_excludes_the_asset_itself(client):
    asset_id = client.get("/api/assets").json()["assets"][0]["id"]
    payload = client.get(f"/api/assets/{asset_id}/similar").json()
    assert all(asset["id"] != asset_id for asset in payload["assets"])


def test_the_frontend_is_served(client):
    page = client.get("/")
    assert page.status_code == 200
    assert "AssetKeep" in page.text
    assert client.get("/app.js").status_code == 200
    assert client.get("/inspector.js").status_code == 200


# --- M3: curation ------------------------------------------------------------


def ids(client, query=""):
    return [
        asset["id"]
        for asset in client.get("/api/assets", params={"q": query}).json()["assets"]
    ]


def test_metadata_can_be_edited_and_comes_back(client):
    asset_id = ids(client)[0]
    updated = client.patch(
        f"/api/assets/{asset_id}",
        json={"license": "cc0", "source_name": "Kenney", "notes": "from the bundle"},
    ).json()

    assert updated["license"] == "cc0"
    assert client.get(f"/api/assets/{asset_id}").json()["source_name"] == "Kenney"


def test_editing_reindexes_for_search(client):
    """Notes are an FTS column; without the reindex the words are unfindable."""
    asset_id = ids(client)[0]
    client.patch(f"/api/assets/{asset_id}", json={"notes": "unmistakable phrase"})

    assert ids(client, "unmistakable") == [asset_id]


def test_derived_columns_cannot_be_written_through_the_editor(client):
    """``content_hash`` and ``managed`` come from the file and the scan. An
    editor that could overwrite them puts the index and the disk into a
    disagreement nothing resolves."""
    asset_id = ids(client)[0]
    before = client.get(f"/api/assets/{asset_id}").json()

    client.patch(
        f"/api/assets/{asset_id}", json={"content_hash": "deadbeef", "managed": 1}
    )
    after = client.get(f"/api/assets/{asset_id}").json()

    assert after["content_hash"] == before["content_hash"]
    assert after["managed"] == before["managed"]


def test_editing_a_missing_asset_is_a_404(client):
    assert client.patch("/api/assets/99999", json={"notes": "x"}).status_code == 404


def test_tag_autocomplete_ranks_prefix_matches_first(client):
    asset_id = ids(client)[0]
    client.post(f"/api/assets/{asset_id}/tags", json={"tags": ["pixelated-thing"]})

    names = [tag["name"] for tag in client.get("/api/tags", params={"prefix": "pix"}).json()["tags"]]
    assert names and all(name.startswith("pix") for name in names[:2])
    assert "pixel-art" in names


def test_tag_autocomplete_carries_usage_counts(client):
    tags = client.get("/api/tags", params={"prefix": "pixel"}).json()["tags"]
    assert next(tag for tag in tags if tag["name"] == "pixel-art")["count"] == 2


def test_bulk_tags_add_and_remove_in_one_call(client):
    everything = ids(client)
    client.post("/api/assets/bulk/tags", json={"ids": everything, "add": ["wip"]})
    assert set(ids(client, "tag:wip")) == set(everything)

    client.post(
        "/api/assets/bulk/tags",
        json={"ids": everything, "add": ["approved"], "remove": ["wip"]},
    )
    assert ids(client, "tag:wip") == []
    assert set(ids(client, "tag:approved")) == set(everything)


def test_bulk_tag_removal_leaves_automated_tags_alone(client):
    """Removing a tag the tagger applied would grow it back on the next rescan,
    so the button has to be honest about only undoing manual work."""
    tagged = ids(client, "tag:pixel-art")
    client.post("/api/assets/bulk/tags", json={"ids": tagged, "remove": ["pixel-art"]})
    assert set(ids(client, "tag:pixel-art")) == set(tagged)


def test_bulk_metadata_sets_one_licence_across_a_selection(client):
    everything = ids(client)
    result = client.patch(
        "/api/assets/bulk", json={"ids": everything, "license": "cc-by-4.0"}
    ).json()

    assert result["assets"] == len(everything)
    assert all(
        client.get(f"/api/assets/{asset_id}").json()["license"] == "cc-by-4.0"
        for asset_id in everything
    )


def test_bulk_metadata_will_not_overwrite_every_title_at_once(client):
    """Title is per-asset by definition; setting forty to one string destroys
    them, so the bulk endpoint refuses the field rather than trusting callers."""
    everything = ids(client)
    response = client.patch("/api/assets/bulk", json={"ids": everything, "title": "x"})

    assert response.status_code == 400
    assert len({client.get(f"/api/assets/{i}").json()["title"] for i in everything}) == 3


def test_a_bulk_edit_with_no_assets_is_a_400(client):
    assert client.post("/api/assets/bulk/tags", json={"ids": []}).status_code == 400
    assert client.patch("/api/assets/bulk", json={"ids": []}).status_code == 400


def test_the_selection_summary_says_what_is_shared(client):
    everything = ids(client)
    client.patch("/api/assets/bulk", json={"ids": everything, "license": "cc0"})
    client.post("/api/assets/bulk/tags", json={"ids": everything[:1], "add": ["rare"]})

    summary = client.post("/api/assets/summary", json={"ids": everything}).json()

    assert summary["count"] == 3
    assert summary["fields"]["license"] == "cc0"
    counts = {tag["name"]: tag["count"] for tag in summary["tags"]}
    assert counts["rare"] == 1, "a partial tag reports how many carry it"
    assert counts["pack"] == 3


def test_the_summary_reports_a_disagreeing_field_as_null(client):
    everything = ids(client)
    client.patch(f"/api/assets/{everything[0]}", json={"license": "cc0"})

    summary = client.post("/api/assets/summary", json={"ids": everything}).json()
    assert summary["fields"]["license"] is None


def test_collections_round_trip_through_the_api(client):
    everything = ids(client)
    created = client.post(
        "/api/collections", json={"name": "Jam Prototype", "ids": everything[:2]}
    ).json()

    listed = client.get("/api/collections").json()["collections"]
    assert listed[0]["name"] == "jam-prototype"
    assert listed[0]["count"] == 2

    assert set(ids(client, "collection:jam-prototype")) == set(everything[:2])

    client.request(
        "DELETE",
        f"/api/collections/{created['id']}/assets",
        json={"ids": everything[:1]},
    )
    assert ids(client, "collection:jam-prototype") == [everything[1]]

    client.delete(f"/api/collections/{created['id']}")
    assert client.get("/api/collections").json()["collections"] == []


def test_deleting_a_collection_keeps_its_assets(client):
    everything = ids(client)
    created = client.post(
        "/api/collections", json={"name": "temp", "ids": everything}
    ).json()
    client.delete(f"/api/collections/{created['id']}")

    assert len(ids(client)) == 3


def test_an_asset_reports_the_collections_it_is_in(client):
    asset_id = ids(client)[0]
    client.post("/api/collections", json={"name": "hero-kit", "ids": [asset_id]})

    detail = client.get(f"/api/assets/{asset_id}").json()
    assert [entry["name"] for entry in detail["collections"]] == ["hero-kit"]


def test_a_nameless_collection_is_a_400(client):
    assert client.post("/api/collections", json={"name": "  "}).status_code == 400


def test_import_takes_an_upload_into_the_vault(client, tmp_path):
    source = tmp_path / "gem.png"
    fixtures.normal_map(48).save(source)

    with source.open("rb") as handle:
        result = client.post(
            "/api/import",
            files={"files": ("Loot/gem.png", handle, "image/png")},
            data={"collection": "loot"},
        ).json()

    assert len(result["imported"]) == 1
    assert (client.config.vault_path / "loot/gem.png").exists()
    assert len(ids(client, "collection:loot")) == 1


def test_import_reports_a_duplicate_rather_than_doubling_the_library(client, tmp_path):
    source = tmp_path / "gem.png"
    fixtures.normal_map(48).save(source)

    for _ in range(2):
        with source.open("rb") as handle:
            result = client.post(
                "/api/import", files={"files": ("gem.png", handle, "image/png")}
            ).json()

    assert result["imported"] == []
    assert len(result["duplicates"]) == 1
    assert len(list(client.config.vault_path.rglob("gem*.png"))) == 1


# --- references, export and captions ------------------------------------------


@pytest.fixture
def page(monkeypatch):
    """A stubbed page fetch, so no test here touches the network."""
    from assetkeep.probe import reference as reference_probe
    from assetkeep.probe.reference import Page

    monkeypatch.setattr(
        reference_probe,
        "fetch",
        lambda url, timeout=None: Page(
            url=url, title="Platformer Pack Redux", site_name="Kenney",
            description="360 tiles.",
        ),
    )
    monkeypatch.setattr(reference_probe, "fetch_image", lambda url, timeout=None: None)


def test_a_posted_url_becomes_an_asset_the_grid_can_show(client, page):
    payload = client.post(
        "/api/references", json={"url": "https://kenney.nl/assets/platformer-pack-redux"}
    ).json()

    assert payload["created"] is True
    assert payload["asset"]["kind"] == "reference"
    # Present, because a reference is not a file that has gone missing - the
    # grid's missing badge would otherwise be on every link in the library.
    assert payload["asset"]["present"] is True
    assert ids(client, "kind:reference") == [payload["asset_id"]]


def test_a_url_that_is_not_one_is_a_400(client, page):
    assert client.post("/api/references", json={"url": "file:///etc/passwd"}).status_code == 400


def test_refreshing_something_that_is_not_a_reference_is_a_404(client, page):
    asset_id = ids(client, "goblin")[0]
    assert client.post(f"/api/references/{asset_id}/refresh").status_code == 404


def test_a_collection_exports_into_a_project_folder(client, tmp_path):
    created = client.post(
        "/api/collections", json={"name": "jam", "ids": ids(client, "")}
    ).json()
    destination = tmp_path / "Unity" / "Assets"
    destination.mkdir(parents=True)

    result = client.post(
        f"/api/collections/{created['id']}/export",
        json={"destination": str(destination)},
    ).json()

    assert len(result["copied"]) == 3
    assert (destination / "jam" / "CREDITS.md").exists()
    assert result["destination"].endswith("/jam")


def test_exporting_with_no_destination_is_a_400(client):
    created = client.post("/api/collections", json={"name": "jam"}).json()
    response = client.post(f"/api/collections/{created['id']}/export", json={})
    assert response.status_code == 400


def test_captions_are_refused_when_ollama_is_not_there(client):
    """A 409 rather than a queued job nothing will ever run, and the message is
    the command that fixes it."""
    response = client.post("/api/captions", json={"ids": ids(client, "")})
    assert response.status_code == 409
    assert "ollama" in response.json()["detail"] or "vlm pull" in response.json()["detail"]


def test_captions_are_queued_when_the_model_is_there(client, monkeypatch):
    from assetkeep.tagging import vlm

    monkeypatch.setattr(vlm, "available", lambda config: True)
    result = client.post("/api/captions", json={"ids": ids(client, "")}).json()

    assert result["queued"] == 3
