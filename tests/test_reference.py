"""References: what a URL becomes, and what the rest of the tool does with it.

No network. Every test here stubs the two functions in
:mod:`assetkeep.probe.reference` that touch it, which is the same argument the
CLIP tests make about the model: what is worth testing is what the library does
with a page's answer, not whether kenney.nl is up this morning. The parsing
itself is tested against strings, which is what it takes as input anyway.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image

from assetkeep import db, reference, scan, search, thumbs
from assetkeep.config import Config
from assetkeep.probe import reference as reference_probe
from assetkeep.probe.reference import Page

from . import fixtures


@pytest.fixture
def setup(tmp_path):
    config = Config(
        db_path=tmp_path / "index.db",
        vault_path=tmp_path / "vault",
        thumbs_path=tmp_path / "thumbs",
        models_path=tmp_path / "models",
        source_path=tmp_path / "config.toml",
    )
    conn = db.connect(config.db_path)
    yield config, conn
    conn.close()


@pytest.fixture
def page(monkeypatch):
    """A stubbed fetch whose answer each test sets."""

    answer = Page(
        url="https://kenney.nl/assets/platformer-pack-redux",
        title="Platformer Pack Redux",
        description="A pack of 360 tiles and sprites.",
        site_name="Kenney",
        image_url="https://kenney.nl/img/pack.png",
    )
    state = {"page": answer, "image": _png_bytes(), "fetches": 0, "images": 0}

    def fetch(url, timeout=None):
        state["fetches"] += 1
        return state["page"]

    def fetch_image(url, timeout=None):
        state["images"] += 1
        return state["image"]

    monkeypatch.setattr(reference_probe, "fetch", fetch)
    monkeypatch.setattr(reference_probe, "fetch_image", fetch_image)
    return state


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    fixtures.single_sprite(320).convert("RGB").save(buffer, "PNG")
    return buffer.getvalue()


# --- adding -------------------------------------------------------------------


def test_a_url_becomes_a_searchable_asset(setup, page):
    config, conn = setup
    result = reference.add(conn, config, "https://kenney.nl/assets/platformer-pack-redux")

    assert result.created
    row = conn.execute("SELECT * FROM asset WHERE id = ?", (result.asset_id,)).fetchone()
    assert row["kind"] == "reference"
    assert row["title"] == "Platformer Pack Redux"
    assert row["source_name"] == "Kenney"
    assert row["notes"] == "A pack of 360 tiles and sprites."
    assert [r["id"] for r in search.search(conn, "platformer")] == [result.asset_id]


def test_the_same_url_twice_is_one_asset(setup, page):
    config, conn = setup
    first = reference.add(conn, config, "https://kenney.nl/assets/platformer-pack-redux")
    again = reference.add(
        conn, config, "HTTPS://kenney.nl/assets/platformer-pack-redux#tiles"
    )

    assert again.asset_id == first.asset_id
    assert not again.created
    assert conn.execute("SELECT COUNT(*) FROM asset").fetchone()[0] == 1


def test_the_og_image_becomes_the_thumbnail(setup, page):
    config, conn = setup
    result = reference.add(conn, config, "https://kenney.nl/assets/platformer-pack-redux")

    row = conn.execute(
        "SELECT content_hash FROM asset WHERE id = ?", (result.asset_id,)
    ).fetchone()
    tile = thumbs.path_for(config, row["content_hash"])
    assert result.thumbnail and tile.exists()
    with Image.open(tile) as image:
        assert max(image.size) <= config.thumbnails.max_edge


def test_a_dead_link_still_becomes_a_reference(setup, page):
    """The URL is the asset. A page that will not answer costs a title, not
    the entry."""
    config, conn = setup
    page["page"] = Page(
        url="https://example.com/refs/castle_wall.jpg",
        title="castle wall",
        error="ConnectError: nothing there",
    )

    result = reference.add(conn, config, "https://example.com/refs/castle_wall.jpg")

    assert result.created and result.error
    row = conn.execute("SELECT * FROM asset WHERE id = ?", (result.asset_id,)).fetchone()
    assert row["title"] == "castle wall"
    assert not result.thumbnail


def test_an_explicit_title_wins_over_the_page(setup, page):
    config, conn = setup
    result = reference.add(
        conn, config, "https://kenney.nl/assets/platformer-pack-redux",
        title="tiles to steal from",
    )

    row = conn.execute("SELECT title FROM asset WHERE id = ?", (result.asset_id,))
    assert row.fetchone()["title"] == "tiles to steal from"


def test_no_fetch_takes_the_title_from_the_url(setup, page):
    config, conn = setup
    result = reference.add(
        conn, config, "https://kenney.nl/assets/platformer-pack-redux", fetch=False
    )

    assert page["fetches"] == 0
    assert result.title == "platformer pack redux"


def test_a_url_that_is_not_one_is_refused(setup, page):
    config, conn = setup
    with pytest.raises(ValueError):
        reference.add(conn, config, "file:///etc/passwd")


# --- tagging ------------------------------------------------------------------


def test_the_site_and_the_title_become_tags(setup, page):
    config, conn = setup
    result = reference.add(conn, config, "https://kenney.nl/assets/platformer-pack-redux")

    tags = db.tag_names(conn, result.asset_id)
    assert "kenney-nl" in tags  # the source: namespace, as a root name would be
    assert "platformer" in tags and "pack" in tags


def test_manual_tags_survive_a_refresh(setup, page):
    """The same guarantee a rescan gives a file: a refetch may replace what it
    derived and nothing a person applied."""
    config, conn = setup
    result = reference.add(
        conn, config, "https://kenney.nl/assets/platformer-pack-redux",
        tags=["to-buy"],
    )

    page["page"] = Page(
        url="https://kenney.nl/assets/platformer-pack-redux",
        title="Something Else Entirely",
        site_name="Kenney",
    )
    reference.refresh(conn, config, result.asset_id, overwrite=True)

    tags = db.tag_names(conn, result.asset_id)
    assert "to-buy" in tags
    assert "platformer" not in tags  # re-derived from the new title


# --- refreshing ---------------------------------------------------------------


def test_a_refresh_fills_gaps_without_overwriting_a_rename(setup, page):
    config, conn = setup
    result = reference.add(conn, config, "https://kenney.nl/assets/platformer-pack-redux")
    db.update_asset(conn, result.asset_id, {"title": "the good tiles", "notes": ""})

    page["page"] = Page(
        url="https://kenney.nl/assets/platformer-pack-redux",
        title="Platformer Pack Redux",
        description="Now with 400 tiles.",
        site_name="Kenney",
    )
    reference.refresh(conn, config, result.asset_id)

    row = conn.execute("SELECT * FROM asset WHERE id = ?", (result.asset_id,)).fetchone()
    assert row["title"] == "the good tiles"
    assert row["notes"] == "Now with 400 tiles."


def test_refreshing_something_that_is_not_a_reference_is_none(setup, page, tmp_path):
    config, conn = setup
    asset_id = db.create_asset(conn, "image", "abc123", "a sprite")
    assert reference.refresh(conn, config, asset_id) is None


# --- how the rest of the tool sees them ---------------------------------------


def test_a_reference_is_not_missing_and_is_not_pruned(setup, page):
    """Both of these read the same "no present location" condition, and both
    would be catastrophically wrong about a link."""
    config, conn = setup
    result = reference.add(conn, config, "https://kenney.nl/assets/platformer-pack-redux")

    assert search.search(conn, "is:missing") == []
    assert scan.prune(conn, dry_run=True) == []


def test_kind_and_source_filters_find_it(setup, page):
    config, conn = setup
    result = reference.add(conn, config, "https://kenney.nl/assets/platformer-pack-redux")

    assert [r["id"] for r in search.search(conn, "kind:reference")] == [result.asset_id]
    assert [r["id"] for r in search.search(conn, "has:source")] == [result.asset_id]
    assert [r["id"] for r in search.search(conn, "source:kenney")] == [result.asset_id]


# --- parsing ------------------------------------------------------------------


def test_og_tags_beat_the_document_title():
    found = reference_probe.parse_html(
        "<html><head><title>itch.io</title>"
        '<meta property="og:title" content="Free Pixel Tileset">'
        '<meta property="og:site_name" content="itch.io">'
        '<meta property="og:image" content="/uploads/cover.png">'
        "</head></html>"
    )
    page = Page(url="https://itch.io/things/tileset", title="tileset")
    reference_probe._apply(page, found)

    assert page.title == "Free Pixel Tileset"
    assert page.site_name == "itch.io"
    assert page.image_url == "https://itch.io/uploads/cover.png"


def test_a_relative_favicon_is_the_last_resort_preview():
    found = reference_probe.parse_html(
        '<html><head><link rel="shortcut icon" href="favicon.ico"></head></html>'
    )
    page = Page(url="https://example.com/docs/page", title="page")
    reference_probe._apply(page, found)

    assert page.image_url == "https://example.com/docs/favicon.ico"


def test_a_javascript_preview_url_is_refused():
    """The image URL comes off an untrusted page and is fetched by us."""
    found = {"og:image": "javascript:alert(1)"}
    page = Page(url="https://example.com/", title="x")
    reference_probe._apply(page, found)

    assert page.image_url is None


def test_a_host_has_to_look_like_one(setup, page):
    """Found by a headless run: a click handler passed a MouseEvent where a URL
    was expected, urlparse read ``https://{'isTrusted': true}`` as a host, and a
    reference to it was created."""
    config, conn = setup
    for nonsense in ("{'isTrusted': true}", "not a url", "[object Object]"):
        with pytest.raises(ValueError):
            reference.add(conn, config, nonsense, fetch=False)


def test_a_typed_scheme_allows_a_dotless_host(setup, page):
    """``nas:8080`` on a LAN is real; a bare word almost never is."""
    assert reference_probe.normalise("http://nas:8080/pack") == "http://nas:8080/pack"
    assert reference_probe.normalise("nas") is None
