"""VLM captions: the queue, the filtering, and the tidying. No model.

The same argument the CLIP tests make, one tier up. What a real moondream says
about a fixture is not a testable claim - it is an opinion, and one that changes
with the weights - so the model is stubbed and what is tested is the behaviour
around it: which assets are asked about, what happens to the answer, and that
one unreadable file does not stop the queue.
"""

from __future__ import annotations

import pytest

from assetkeep import db, job, scan
from assetkeep.config import Config, RootConfig, VlmConfig
from assetkeep.tagging import vlm

from . import fixtures


@pytest.fixture
def library(tmp_path, monkeypatch):
    """A scanned root with two images and one audio file.

    ``hero.png`` is deliberately larger than the thumbnail box and ``grass.png``
    smaller, because those take different paths to the model: one has a tile
    rendered for it and the other is shown as itself.
    """
    source = tmp_path / "pack"
    source.mkdir()
    fixtures.single_sprite(320).save(source / "hero.png")
    fixtures.upscaled_pixel_art(block=4, cells=16).save(source / "grass.png")
    # A WAV the audio probe can read, so the asset exists and is not captionable.
    _silence(source / "hit.wav")

    config = Config(
        db_path=tmp_path / "index.db",
        vault_path=tmp_path / "vault",
        thumbs_path=tmp_path / "thumbs",
        models_path=tmp_path / "models",
        source_path=tmp_path / "config.toml",
        roots=(RootConfig(path=source),),
        vlm=VlmConfig(model="moondream"),
    )

    conn = db.connect(config.db_path)
    scan.scan(conn, config)
    yield config, conn
    conn.close()


def _silence(path, seconds=0.2, rate=8000):
    import wave

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x00" * int(rate * seconds))


@pytest.fixture
def model(monkeypatch):
    """A stubbed vision model that answers with the filename it was shown."""
    seen = []

    def caption(config, image, prompt=vlm.PROMPT):
        from pathlib import Path

        seen.append(Path(image))
        return f"A pixel-art sprite of a {Path(image).stem}."

    monkeypatch.setattr(vlm, "caption", caption)
    return seen


def ids_by_kind(conn, kind):
    return [
        int(row["id"])
        for row in conn.execute("SELECT id FROM asset WHERE kind = ? ORDER BY id", (kind,))
    ]


# --- what gets queued ---------------------------------------------------------


def test_audio_is_never_queued(library, model):
    """A waveform drawn as a picture would get a confident and entirely
    fictional account of what the sound is of."""
    config, conn = library
    everything = [int(row["id"]) for row in conn.execute("SELECT id FROM asset")]

    queued = job.enqueue_captions(conn, everything)

    assert set(queued) == set(ids_by_kind(conn, "image"))
    assert not set(queued) & set(ids_by_kind(conn, "audio"))


def test_an_asset_that_already_has_one_is_skipped(library, model):
    config, conn = library
    images = ids_by_kind(conn, "image")
    db.update_asset(conn, images[0], {"caption": "already described"})

    assert job.enqueue_captions(conn, images) == images[1:]
    assert job.enqueue_captions(conn, images, redo=True) == images


def test_a_reference_has_nothing_to_look_at(library, model):
    config, conn = library
    asset_id = db.create_asset(conn, "reference", "deadbeef", "a link")
    assert job.enqueue_captions(conn, [asset_id]) == []


# --- what the answer does -----------------------------------------------------


def test_a_caption_is_written_and_becomes_searchable(library, model):
    from assetkeep import search

    config, conn = library
    images = ids_by_kind(conn, "image")
    job.enqueue_captions(conn, images)
    job.drain(conn, config)

    row = conn.execute("SELECT caption FROM asset WHERE id = ?", (images[0],)).fetchone()
    assert row["caption"].startswith("A pixel-art sprite")
    assert sorted(r["id"] for r in search.search(conn, "has:caption")) == images


def test_the_model_is_shown_the_thumbnail_when_there_is_one(library, model):
    """256 px is past what a vision tower keeps, and base64 of a 4K PNG is 20 MB
    over a socket for nothing. A sprite too small to have earned a thumbnail is
    shown as itself, which is the same rule ``/api/thumb`` follows."""
    config, conn = library
    job.drain(conn, config)  # thumbnails first
    job.enqueue_captions(conn, ids_by_kind(conn, "image"))
    job.drain(conn, config)

    # A tile is named by its content hash, so the store is what identifies it.
    tiles = [path for path in model if str(path).startswith(str(config.thumbs_path))]
    originals = [path for path in model if path not in tiles]

    assert [path.suffix for path in tiles] == [".webp"]  # hero.png, 320 px
    assert [path.name for path in originals] == ["grass.png"]  # 64 px, no tile


def test_one_failure_does_not_stop_the_queue(library, monkeypatch):
    """A file the model will not answer about burns its three attempts and
    lands in the failed count; everything else in the queue still runs."""
    config, conn = library

    def caption(config, image, prompt=vlm.PROMPT):
        from pathlib import Path

        # Keyed on the file rather than on a call count, because a failed job is
        # retried inside the same drain and a counter would let it through on
        # the second attempt.
        if Path(image).suffix == ".webp":
            raise RuntimeError("ollama fell over")
        return "A patch of grass."

    monkeypatch.setattr(vlm, "caption", caption)
    job.drain(conn, config)  # thumbnails, so one asset is shown a .webp
    job.enqueue_captions(conn, ids_by_kind(conn, "image"))
    status = job.drain(conn, config)

    captioned = conn.execute(
        "SELECT COUNT(*) FROM asset WHERE caption IS NOT NULL AND caption != ''"
    ).fetchone()[0]
    assert captioned == 1
    assert status.failed == 1
    assert status.pending == 0


def test_the_queue_reports_captions_separately_from_thumbnails(library, model):
    config, conn = library
    job.enqueue_captions(conn, ids_by_kind(conn, "image"))

    kinds = job.status(conn).pending_kinds
    assert kinds["caption"] == 2
    assert kinds["thumbnail"] == 3


# --- talking to ollama --------------------------------------------------------


def test_availability_distinguishes_no_server_from_no_model(monkeypatch):
    config = Config(vlm=VlmConfig(model="moondream"))

    monkeypatch.setattr(vlm, "installed_models", lambda config, timeout=2.0: None)
    assert vlm.status(config)["server"] is False
    assert vlm.available(config) is False

    monkeypatch.setattr(vlm, "installed_models", lambda config, timeout=2.0: ["gemma2:2b"])
    state = vlm.status(config)
    assert state["server"] is True and state["installed"] is False

    monkeypatch.setattr(
        vlm, "installed_models", lambda config, timeout=2.0: ["moondream:latest"]
    )
    assert vlm.available(config) is True


def test_a_caption_request_carries_the_image_and_the_model(monkeypatch, tmp_path):
    """The one thing worth asserting about the HTTP call: that what arrives is
    the model named in the config, and a PNG of the file asked about.

    A PNG rather than the file's own bytes because ollama cannot decode WebP,
    which is every thumbnail this tool makes - see :func:`assetkeep.tagging.vlm.encode`.
    """
    import base64
    import io

    import httpx
    from PIL import Image

    config = Config(vlm=VlmConfig(model="moondream", url="http://localhost:11434"))
    image = tmp_path / "hero.png"
    fixtures.single_sprite(32).save(image)
    captured = {}

    class Response:
        status_code = 200

        def json(self):
            return {"response": "The image shows a small blue sprite."}

    def post(url, json=None, timeout=None):
        captured.update({"url": url, "body": json})
        return Response()

    monkeypatch.setattr(httpx, "post", post)
    text = vlm.caption(config, image)

    assert captured["url"] == "http://localhost:11434/api/generate"
    assert captured["body"]["model"] == "moondream"
    sent = Image.open(io.BytesIO(base64.b64decode(captured["body"]["images"][0])))
    assert sent.format == "PNG"
    assert sent.size == Image.open(image).size
    assert text == "A small blue sprite."


def test_a_webp_thumbnail_is_re_encoded(monkeypatch, tmp_path):
    """ollama answers 400 to a WebP, and a WebP is what every thumbnail is."""
    import io

    from PIL import Image

    tile = tmp_path / "tile.webp"
    fixtures.single_sprite(64).save(tile, "WEBP")
    assert Image.open(io.BytesIO(vlm.encode(tile))).format == "PNG"


def test_transparency_is_composited_rather_than_dropped(tmp_path):
    """``convert("RGB")`` keeps whatever sits under transparent pixels, which
    hands the model a sprite surrounded by noise."""
    import io

    from PIL import Image

    source = tmp_path / "sprite.png"
    fixtures.single_sprite(64).save(source)

    encoded = Image.open(io.BytesIO(vlm.encode(source)))
    assert encoded.mode == "RGB"
    assert encoded.getpixel((0, 0)) == vlm.BACKDROP


def test_an_empty_answer_is_an_error_not_a_caption(monkeypatch, tmp_path):
    import httpx

    config = Config(vlm=VlmConfig())
    image = tmp_path / "hero.png"
    fixtures.single_sprite(32).save(image)

    class Response:
        status_code = 200

        def json(self):
            return {"response": "   "}

    monkeypatch.setattr(httpx, "post", lambda *a, **k: Response())
    with pytest.raises(ValueError):
        vlm.caption(config, image)


def test_a_refused_request_is_retried_once(monkeypatch, tmp_path):
    """Two AssetKeep processes draining one queue put two requests into a 1.7 GB
    model at once and it refuses one. Measured: 50 of 172 captions failed that
    way, and all fifty succeeded when asked again."""
    import httpx

    config = Config(vlm=VlmConfig())
    image = tmp_path / "hero.png"
    fixtures.single_sprite(32).save(image)
    calls = []

    class Response:
        def __init__(self, status):
            self.status_code = status
            self.text = "busy"

        def json(self):
            return {"response": "A small blue sprite."}

    def post(url, json=None, timeout=None):
        calls.append(url)
        return Response(400 if len(calls) == 1 else 200)

    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setattr(vlm, "RETRY_PAUSE", 0)

    assert vlm.caption(config, image) == "A small blue sprite."
    assert len(calls) == 2


def test_a_second_refusal_carries_ollamas_own_message(monkeypatch, tmp_path):
    """"400 Bad Request" fifty times is a mystery; the body says which."""
    import httpx

    config = Config(vlm=VlmConfig())
    image = tmp_path / "hero.png"
    fixtures.single_sprite(32).save(image)

    class Response:
        status_code = 400
        text = '{"error":"Failed to load image or audio file"}'

    monkeypatch.setattr(httpx, "post", lambda *a, **k: Response())
    monkeypatch.setattr(vlm, "RETRY_PAUSE", 0)

    with pytest.raises(RuntimeError, match="Failed to load image"):
        vlm.caption(config, image)


def test_the_limit_caps_the_work_not_the_report(library, model):
    """Measured the hard way: ``caption --limit 6`` queued 161 jobs, printed 6
    and then spent six minutes on all of them."""
    config, conn = library
    images = ids_by_kind(conn, "image")

    queued = job.enqueue_captions(conn, images, limit=1)

    assert len(queued) == 1
    assert job.status(conn).pending_kinds.get("caption") == 1


def test_a_normal_map_is_never_queued(library, model):
    """Measured: moondream calls one "a vibrant purple square". Accurate about
    the pixels, useless about the asset, and worse than useless in an index
    where every normal map would then answer to "purple"."""
    config, conn = library
    images = ids_by_kind(conn, "image")
    db.add_tags(conn, images[0], [("normal-map", "heuristic", 0.9)])

    assert job.enqueue_captions(conn, images) == images[1:]
