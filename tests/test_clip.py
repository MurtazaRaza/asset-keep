"""The CLIP tier, without downloading CLIP.

Everything here runs on a base install. The model is a stub whose text and image
"embeddings" are whatever the test says they are, which is the only way to test
a classifier's behaviour rather than its opinions: with a real model, a test that
asserts "this sprite is tagged pixel-art" fails when the model is right and the
fixture is ambiguous, and passes for the wrong reasons the rest of the time.

What the real model does is measured in ``IMPLEMENTATION.md`` against the
calibration library, which is where an opinion can be checked against 921 real
assets instead of one synthetic one.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from assetkeep.config import Config, TaggingConfig
from assetkeep.tagging import clip


@pytest.fixture
def cfg(tmp_path):
    return Config(
        db_path=tmp_path / "index.db",
        models_path=tmp_path / "models",
        source_path=tmp_path / "config.toml",
    )


class StubEncoder:
    """An encoder whose embeddings are dictated by the test.

    Each label gets its own basis vector, so an image vector can be written as
    "this is exactly tileset" or "these two, evenly" and the classifier's answer
    is a fact about the classifier.
    """

    def __init__(self, cfg, model_id="stub"):
        self.config = cfg
        self.model_id = model_id
        self.calls = 0

    def encode_texts(self, texts):
        self.calls += 1
        names = clip.label_names()
        rows = []
        for text in texts:
            vector = np.zeros(len(names), dtype=np.float32)
            for position, name in enumerate(names):
                if any(phrase in text for phrase in clip._descriptions_for(name)):
                    vector[position] = 1.0
            rows.append(vector)
        return np.stack(rows)


def basis(*tags: str, weight: float = 1.0) -> np.ndarray:
    """An image vector pointing straight at one or more label directions."""
    names = clip.label_names()
    vector = np.zeros(len(names), dtype=np.float32)
    for tag in tags:
        vector[names.index(tag)] = weight
    return vector


# --- variants and weights ----------------------------------------------------


def test_every_variant_pins_a_sha256_and_a_size():
    for variant in clip.VARIANTS.values():
        for weight in variant.files:
            assert len(weight.sha256) == 64
            assert weight.size > 0
            assert clip.REVISION in weight.url


def test_the_url_is_pinned_to_a_commit_not_a_branch():
    """A moving branch makes the pinned digests fail for an unguessable reason."""
    assert "/main/" not in clip.BASE_URL
    assert len(clip.REVISION) == 40


def test_missing_lists_everything_before_anything_is_downloaded(cfg):
    variant = clip.variant_for(None)
    assert clip.missing(cfg, variant) == list(variant.files)
    assert not clip.installed(cfg, variant)


def test_a_truncated_file_counts_as_missing(cfg):
    variant = clip.variant_for(None)
    clip.directory(cfg).mkdir(parents=True)
    for weight in variant.files:
        clip.path_for(cfg, weight).write_bytes(b"not really the model")

    assert clip.missing(cfg, variant) == list(variant.files)


def test_a_full_length_file_counts_as_present(cfg):
    """Length, not hash: this is checked on every capability call."""
    variant = clip.variant_for(None)
    clip.directory(cfg).mkdir(parents=True)
    for weight in variant.files:
        clip.path_for(cfg, weight).write_bytes(b"\0" * weight.size)

    assert clip.installed(cfg, variant)


def test_an_unknown_variant_falls_back_rather_than_raising():
    assert clip.variant_for("clip-vit-h-14-imaginary").id == clip.DEFAULT_VARIANT


def test_status_reports_what_is_missing_and_how_big(cfg):
    state = clip.status(cfg)
    assert state["model"] == clip.DEFAULT_VARIANT
    assert state["weights"] is False
    assert state["download_bytes"] == clip.VARIANTS[clip.DEFAULT_VARIANT].bytes_total


def test_available_is_false_without_weights_even_with_the_extra(cfg):
    assert clip.available(cfg) is False


# --- preprocessing -----------------------------------------------------------


def test_letterboxing_keeps_the_whole_sprite_strip():
    """A centre crop of a 1024x256 strip discards three quarters of the frames."""
    strip = Image.new("RGB", (1024, 256), (255, 255, 255))
    strip.paste(Image.new("RGB", (64, 64), (255, 0, 0)), (0, 96))
    strip.paste(Image.new("RGB", (64, 64), (0, 0, 255)), (960, 96))

    boxed = np.asarray(clip._letterbox(strip))
    cropped = np.asarray(clip._centre_crop(strip))

    reds = ((boxed[:, :, 0] > 200) & (boxed[:, :, 2] < 100)).sum()
    blues = ((boxed[:, :, 2] > 200) & (boxed[:, :, 0] < 100)).sum()
    assert reds > 0 and blues > 0

    cropped_reds = ((cropped[:, :, 0] > 200) & (cropped[:, :, 2] < 100)).sum()
    assert cropped_reds == 0


def test_transparency_is_flattened_onto_the_backdrop():
    """Not left as whatever RGB the exporter happened to write under alpha 0."""
    sprite = Image.new("RGBA", (64, 64), (200, 30, 40, 0))
    flat = np.asarray(clip._composite(sprite))
    assert flat[0, 0].tolist() == list(clip.BACKDROP)


def test_the_two_similarity_measures_agree_about_transparency():
    """dHash and CLIP composite onto the same colour, so "similar" means one
    thing whichever of them answered."""
    from assetkeep import similarity

    assert clip.BACKDROP == (0, 0, 0)
    sprite = Image.new("RGBA", (8, 8), (200, 30, 40, 0))
    assert (
        np.asarray(clip._composite(sprite))[0, 0].tolist()
        == np.asarray(similarity._composite(sprite))[0, 0].tolist()
    )


def test_preprocess_produces_the_shape_the_tower_declares():
    tensor = clip.preprocess(Image.new("RGB", (37, 512)))
    assert tensor.shape == (3, clip.IMAGE_SIZE, clip.IMAGE_SIZE)
    assert tensor.dtype == np.float32


def test_preprocess_normalises_with_the_published_constants():
    """A mid-grey image lands where mean and std say it should."""
    tensor = clip.preprocess(Image.new("RGB", (64, 64), (128, 128, 128)))
    expected = (128 / 255 - clip.IMAGE_MEAN) / clip.IMAGE_STD
    assert tensor[:, 112, 112] == pytest.approx(expected, abs=1e-5)


def test_an_empty_batch_is_still_the_right_shape():
    assert clip.preprocess_all([]).shape == (0, 3, 224, 224)


# --- zero-shot classification ------------------------------------------------


def test_labels_only_cover_tags_the_vocabulary_knows():
    from assetkeep.tagging import vocab

    for tag in clip.LABELS:
        assert vocab.namespace_for(tag) in clip.GROUPS, tag


def test_labels_leave_out_what_cannot_be_seen():
    assert "sfx" not in clip.LABELS
    assert "music" not in clip.LABELS


def test_one_tag_per_namespace_at_most(cfg):
    encoder = StubEncoder(cfg)
    names, matrix = clip.label_matrix(encoder)

    found = clip.classify(names, matrix, np.stack([basis("tileset")]), threshold=0.0)
    namespaces = [
        __import__("assetkeep.tagging.vocab", fromlist=["x"]).namespace_for(tag)
        for tag, _ in found[0]
    ]
    assert len(namespaces) == len(set(namespaces))


def test_a_confident_image_gets_its_tag(cfg):
    encoder = StubEncoder(cfg)
    names, matrix = clip.label_matrix(encoder)

    found = clip.classify(names, matrix, np.stack([basis("tileset")]), threshold=0.3)
    assert ("tileset", 1.0) in [(tag, round(score)) for tag, score in found[0]]


def test_a_torn_image_gets_nothing_from_that_namespace(cfg):
    """Two type labels equally: the honest answer is that it does not know."""
    encoder = StubEncoder(cfg)
    names, matrix = clip.label_matrix(encoder)

    found = clip.classify(
        names, matrix, np.stack([basis("tileset", "weapon")]), threshold=0.9
    )
    assert [tag for tag, _ in found[0] if tag in ("tileset", "weapon")] == []


def test_confidence_is_a_probability_within_the_namespace(cfg):
    encoder = StubEncoder(cfg)
    names, matrix = clip.label_matrix(encoder)

    found = clip.classify(
        names, matrix, np.stack([basis("tileset", "weapon")]), threshold=0.0
    )
    scores = dict(found[0])
    assert 0.0 < scores["tileset"] <= 1.0
    assert scores["tileset"] == pytest.approx(0.5, abs=0.01)


def test_the_threshold_is_what_decides(cfg):
    encoder = StubEncoder(cfg)
    names, matrix = clip.label_matrix(encoder)
    vectors = np.stack([basis("tileset", "weapon")])

    assert clip.classify(names, matrix, vectors, threshold=0.4)[0]
    assert not clip.classify(names, matrix, vectors, threshold=0.6)[0]


def test_a_whole_batch_is_classified_at_once(cfg):
    encoder = StubEncoder(cfg)
    names, matrix = clip.label_matrix(encoder)

    found = clip.classify(
        names,
        matrix,
        np.stack([basis("tileset"), basis("weapon"), basis("pixel-art")]),
        threshold=0.5,
    )
    assert [tag for tag, _ in found[0]][0] == "tileset"
    assert [tag for tag, _ in found[1]][0] == "weapon"
    assert [tag for tag, _ in found[2]] == ["pixel-art"]


def test_an_image_that_is_none_of_the_labels_gets_no_tag(cfg):
    """The background rows exist so "none of these" is an available answer.

    Without them a softmax over eight subjects has to pick one, and on the real
    library it picked ``dungeon`` for 146 assets including a selection box.
    """
    encoder = StubEncoder(cfg)
    names, matrix = clip.label_matrix(encoder)

    found = clip.classify(names, matrix, np.stack([basis("!subject/0")]), 0.3)
    assert [tag for tag, _ in found[0]] == []


def test_a_background_row_never_becomes_a_tag(cfg):
    encoder = StubEncoder(cfg)
    names, matrix = clip.label_matrix(encoder)

    every = np.stack([basis(name) for name in names])
    for row in clip.classify(names, matrix, every, threshold=0.0):
        assert not any(tag.startswith("!") for tag, _ in row)


def test_background_rows_are_scored_but_not_offered(cfg):
    names = clip.label_names()
    assert any(name.startswith("!") for name in names)
    assert not any(tag.startswith("!") for tag in clip.LABELS)


def test_tags_come_out_as_db_triples(cfg):
    encoder = StubEncoder(cfg)
    triples = clip.tags_for(encoder, np.stack([basis("tileset")]), threshold=0.3)

    from assetkeep import db

    for name, source, confidence in triples[0]:
        assert source in db.AUTOMATED_SOURCES
        assert source == "clip"
        assert 0.0 <= confidence <= 1.0


# --- the label cache ---------------------------------------------------------


def test_the_label_matrix_is_computed_once_and_reused(cfg):
    encoder = StubEncoder(cfg)
    clip.label_matrix(encoder)
    first = encoder.calls
    assert first > 0

    clip.label_matrix(StubEncoder(cfg))
    assert StubEncoder(cfg).calls == 0
    assert (clip.directory(cfg)).exists()


def test_editing_the_labels_invalidates_the_cache(cfg, monkeypatch):
    encoder = StubEncoder(cfg)
    clip.label_matrix(encoder)
    before = sorted(p.name for p in clip.directory(cfg).glob("labels-*.npy"))

    monkeypatch.setitem(clip.LABELS, "tileset", ("a completely different phrase",))
    clip.label_matrix(StubEncoder(cfg))
    after = sorted(p.name for p in clip.directory(cfg).glob("labels-*.npy"))

    assert len(after) == len(before) + 1


def test_two_variants_do_not_share_a_cache_file(cfg):
    clip.label_matrix(StubEncoder(cfg, model_id="clip-vit-b-32"))
    clip.label_matrix(StubEncoder(cfg, model_id="clip-vit-b-32-int8"))

    assert len(list(clip.directory(cfg).glob("labels-*.npy"))) == 2


# --- the ranker factory ------------------------------------------------------


def test_no_ranker_without_weights(cfg):
    """Distinct from a ranker that finds nothing, and the difference matters."""
    from assetkeep import db

    conn = db.connect(cfg.db_path)
    try:
        assert clip.ranker(cfg, conn) is None
    finally:
        conn.close()


def test_no_ranker_when_nothing_has_been_embedded(cfg, monkeypatch):
    from assetkeep import db

    monkeypatch.setattr(clip, "available", lambda *args, **kwargs: True)
    conn = db.connect(cfg.db_path)
    try:
        assert clip.ranker(cfg, conn) is None
    finally:
        conn.close()


def test_the_threshold_default_is_a_probability_not_a_cosine():
    """0.22 meant a raw cosine in the M1 spec; this one is a softmax share.

    Left here as a tripwire: the two numbers live in the same config key and
    look identical, so a copied config from before the change would quietly
    tag far more than it should.
    """
    assert TaggingConfig().clip_threshold >= 1.0 / len(clip.labels_in("type"))
