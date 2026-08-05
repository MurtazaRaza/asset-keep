"""Thumbnail generation, storage rules and the rasteriser."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from assetkeep import hashing, thumbs
from assetkeep.config import Config, ThumbnailConfig

from . import fixtures


@pytest.fixture
def config(tmp_path):
    return Config(
        db_path=tmp_path / "index.db",
        thumbs_path=tmp_path / "thumbs",
        source_path=tmp_path / "config.toml",
    )


def test_a_small_image_gets_no_thumbnail_at_all(config, tmp_path):
    """A WebP of a 32x32 sprite is larger than the sprite."""
    source = tmp_path / "tiny.png"
    fixtures.two_tone_mask(32).save(source)

    assert thumbs.generate(config, "image", source, "a" * 32) is None
    assert not thumbs.path_for(config, "a" * 32).exists()


def test_a_large_image_is_reduced_into_the_box(config, tmp_path):
    source = tmp_path / "big.png"
    fixtures.organic_texture(512).save(source)

    written = thumbs.generate(config, "image", source, "b" * 32)

    assert written is not None
    with Image.open(written) as thumb:
        assert max(thumb.size) == config.thumbnails.max_edge
        assert thumb.format == "WEBP"


def test_skip_small_can_be_turned_off(tmp_path):
    from dataclasses import replace

    config = Config(
        thumbs_path=tmp_path / "thumbs",
        thumbnails=ThumbnailConfig(skip_smaller=False),
        source_path=tmp_path / "config.toml",
    )
    source = tmp_path / "tiny.png"
    fixtures.two_tone_mask(32).save(source)

    assert thumbs.generate(config, "image", source, "c" * 32) is not None


def test_flat_art_is_stored_losslessly(config, tmp_path):
    """Lossy compression puts ringing around every hard pixel-art edge."""
    source = tmp_path / "art.png"
    fixtures.upscaled_pixel_art(block=8, cells=64).quantize(colors=16).convert(
        "RGB"
    ).save(source)

    written = thumbs.generate(config, "image", source, "d" * 32)
    original = np.asarray(Image.open(source).convert("RGB").resize((256, 256), Image.Resampling.NEAREST))
    stored = np.asarray(Image.open(written).convert("RGB"))

    assert np.array_equal(original, stored), "lossless means exactly the source pixels"


def test_the_store_is_sharded_by_hash(config, tmp_path):
    source = tmp_path / "big.png"
    fixtures.organic_texture(512).save(source)
    digest = hashing.hash_file(source)

    written = thumbs.generate(config, "image", source, digest)

    assert written.parent.name == digest[2:4]
    assert written.parent.parent.name == digest[:2]


def test_an_existing_thumbnail_is_not_regenerated(config, tmp_path):
    source = tmp_path / "big.png"
    fixtures.organic_texture(512).save(source)

    first = thumbs.generate(config, "image", source, "e" * 32)
    stamp = first.stat().st_mtime_ns
    again = thumbs.generate(config, "image", source, "e" * 32)

    assert again.stat().st_mtime_ns == stamp


def test_a_broken_source_returns_none_rather_than_raising(config, tmp_path):
    source = tmp_path / "broken.png"
    source.write_bytes(b"\x89PNG\r\n\x1a\n" + b"nonsense" * 4)

    assert thumbs.generate(config, "image", source, "f" * 32) is None


class TestRasteriser:
    def test_a_cube_renders_a_solid_silhouette(self):
        vertices, faces = _cube()
        image = thumbs.render_mesh(vertices, faces, 64)
        alpha = np.asarray(image)[:, :, 3]

        assert alpha.max() == 255
        coverage = (alpha > 0).mean()
        assert 0.2 < coverage < 0.85, "a cube should fill much of the frame, not all"

    def test_faces_are_shaded_differently(self):
        """A single flat tone means the lighting is not doing anything."""
        vertices, faces = _cube()
        image = thumbs.render_mesh(vertices, faces, 64)
        pixels = np.asarray(image)
        lit = pixels[pixels[:, :, 3] > 0][:, 0]

        assert len(np.unique(lit)) >= 3

    def test_the_nearer_face_wins(self):
        """Two overlapping triangles at different depths: z-buffer, not paint order."""
        far = np.array([[-1, -1, 5], [1, -1, 5], [0, 1, 5]], dtype=np.float32)
        near = np.array([[-1, -1, -5], [1, -1, -5], [0, 1, -5]], dtype=np.float32)
        vertices = np.vstack([far, near])
        faces = np.array([[0, 1, 2], [3, 4, 5]])

        # Drawn far-first and near-first must agree.
        forward = np.asarray(thumbs.render_mesh(vertices, faces, 48))
        backward = np.asarray(
            thumbs.render_mesh(vertices, faces[::-1].copy(), 48)
        )
        assert np.array_equal(forward, backward)

    def test_a_mesh_over_the_cap_still_renders(self):
        """Regression: _project decimates locally, so the loop must not use
        the original face count or it walks off the end of every array."""
        count = thumbs.MAX_RENDER_TRIANGLES + 5_000
        rng = np.random.default_rng(7)
        vertices = rng.normal(0, 1, (count * 3, 3)).astype(np.float32)
        faces = np.arange(count * 3).reshape(count, 3)

        image = thumbs.render_mesh(vertices, faces, 32)
        assert image.size == (32, 32)

    def test_an_empty_mesh_does_not_crash(self):
        image = thumbs.render_mesh(
            np.zeros((3, 3), dtype=np.float32), np.array([[0, 1, 2]]), 16
        )
        assert image.size == (16, 16)


class TestWaveform:
    def test_a_loud_signal_draws_taller_than_a_quiet_one(self):
        loud = np.sin(np.linspace(0, 500, 20_000)) * 30000
        image = thumbs.render_waveform(loud, 64)
        pixels = np.asarray(image)[:, :, :3]

        drawn = (pixels != np.array(thumbs.WAVEFORM_BACKGROUND)).any(axis=2)
        assert drawn.mean() > 0.3

    def test_silence_draws_a_flat_line(self):
        image = thumbs.render_waveform(np.zeros(8000, dtype=np.float32), 64)
        pixels = np.asarray(image)[:, :, :3]
        drawn = (pixels != np.array(thumbs.WAVEFORM_BACKGROUND)).any(axis=2)

        assert 0 < drawn.mean() < 0.1


def _cube():
    """Unit cube as 12 triangles."""
    vertices = np.array(
        [
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
            [0, 4, 5], [0, 5, 1], [3, 2, 6], [3, 6, 7],
            [0, 3, 7], [0, 7, 4], [1, 5, 6], [1, 6, 2],
        ]
    )
    return vertices, faces
