"""Each image heuristic against a positive and its near miss.

These detectors fail by over-triggering, so a test that only proves the positive
proves almost nothing: a function returning ``True`` unconditionally passes it.
Every case here is therefore a pair.
"""

from __future__ import annotations

import pytest

from assetkeep.probe import image as image_probe

from . import fixtures


def tags_for(img, tmp_path, name="fixture.png"):
    path = tmp_path / name
    img.save(path)
    result = image_probe.probe(path)
    return {tag for tag, _, _ in result.tags}, result


def test_upscaled_pixel_art_reports_its_block_size(tmp_path):
    tags, result = tags_for(fixtures.upscaled_pixel_art(block=4), tmp_path)
    assert "pixel-art" in tags
    assert result.attributes["block_size"] == 4


def test_organic_texture_is_not_pixel_art(tmp_path):
    tags, result = tags_for(fixtures.organic_texture(), tmp_path)
    assert "pixel-art" not in tags
    assert "block_size" not in result.attributes


def test_native_resolution_sprite_is_pixel_art_at_lower_confidence(tmp_path):
    """Small and flat, with no measurable period: the weaker of the two signals."""
    path = tmp_path / "sprite.png"
    fixtures.upscaled_pixel_art(block=1, cells=32).quantize(colors=16).save(path)
    result = image_probe.probe(path)

    confidences = {tag: conf for tag, _, conf in result.tags}
    assert confidences["pixel-art"] == 0.6
    assert result.attributes["block_size"] == 1


def test_spritesheet_grid_is_measured(tmp_path):
    tags, result = tags_for(fixtures.spritesheet(cols=3, rows=2), tmp_path)
    assert "spritesheet" in tags
    assert (result.attributes["cols"], result.attributes["rows"]) == (3, 2)
    assert result.attributes["frame_count"] == 6


def test_single_sprite_is_not_a_spritesheet(tmp_path):
    tags, result = tags_for(fixtures.single_sprite(), tmp_path)
    assert "spritesheet" not in tags
    assert "cols" not in result.attributes


def test_fully_transparent_image_is_not_a_spritesheet(tmp_path):
    """Empty cells satisfy every boundary test there is."""
    from PIL import Image

    tags, _ = tags_for(Image.new("RGBA", (128, 128), (0, 0, 0, 0)), tmp_path)
    assert "spritesheet" not in tags


def test_tiling_texture_is_tileable(tmp_path):
    tags, result = tags_for(fixtures.tiling_texture(), tmp_path)
    assert "tileable" in tags
    assert result.attributes["seam_score"] < image_probe.SEAM_THRESHOLD


def test_off_period_texture_is_not_tileable(tmp_path):
    tags, _ = tags_for(fixtures.non_tiling_texture(), tmp_path)
    assert "tileable" not in tags


def test_transparent_sprite_is_not_called_tileable(tmp_path):
    """Its opposite edges match because both are empty, which proves nothing."""
    tags, _ = tags_for(fixtures.single_sprite(), tmp_path)
    assert "tileable" not in tags


def test_normal_map_is_detected(tmp_path):
    tags, _ = tags_for(fixtures.normal_map(), tmp_path)
    assert "normal-map" in tags


def test_blue_sky_is_not_a_normal_map(tmp_path):
    tags, _ = tags_for(fixtures.blue_sky(), tmp_path)
    assert "normal-map" not in tags


def test_two_tone_image_is_a_mask(tmp_path):
    tags, _ = tags_for(fixtures.two_tone_mask(), tmp_path)
    assert "mask" in tags


def test_greyscale_photo_is_not_a_mask(tmp_path):
    tags, _ = tags_for(fixtures.greyscale_photo(), tmp_path)
    assert "mask" not in tags


def test_alpha_is_reported_only_when_present(tmp_path):
    transparent, _ = tags_for(fixtures.single_sprite(), tmp_path, "a.png")
    opaque, _ = tags_for(fixtures.tiling_texture(), tmp_path, "b.png")
    assert "has-alpha" in transparent
    assert "has-alpha" not in opaque


def test_structural_attributes(tmp_path):
    _, result = tags_for(fixtures.spritesheet(cols=3, rows=2, cell=32), tmp_path)
    assert result.attributes["width"] == 96
    assert result.attributes["height"] == 64
    assert result.attributes["aspect"] == 1.5
    assert result.attributes["bytes"] > 0
    assert result.dhash is not None
    assert len(result.palette) == 24


def test_a_corrupt_file_degrades_rather_than_raising(tmp_path):
    from assetkeep import probe as probe_module

    path = tmp_path / "broken.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"garbage" * 8)

    result = probe_module.probe(path, "image")
    assert result.error is not None
    assert result.tags == []


@pytest.mark.parametrize("suffix", [".aseprite", ".ase"])
def test_aseprite_header_is_parsed_without_a_decoder(tmp_path, suffix):
    import struct

    path = tmp_path / f"hero{suffix}"
    # file size, magic, frames, width, height, depth, then padding to 128.
    path.write_bytes(
        struct.pack("<IHHHHH", 1024, 0xA5E0, 8, 48, 64, 32) + bytes(114)
    )

    result = image_probe.probe(path)
    assert result.attributes["width"] == 48
    assert result.attributes["height"] == 64
    assert result.attributes["frame_count"] == 8
    assert {tag for tag, _, _ in result.tags} == {"pixel-art", "animation"}


class TestGuardsFoundOnRealData:
    """Regressions for false positives the calibration library produced.

    Each of these passed every test above and was still wrong. They are kept
    apart because they document a specific failure that was observed rather
    than one that was anticipated.
    """

    def test_a_periodic_photo_texture_is_not_pixel_art(self, tmp_path):
        """Blocks with detail inside them are not blocks."""
        tags, result = tags_for(fixtures.periodic_photo_texture(), tmp_path)
        assert "pixel-art" not in tags
        assert "block_size" not in result.attributes

    def test_a_block_too_coarse_for_the_image_is_rejected(self):
        """A 48 px sprite was reported as a 30 px grid: 1.6 blocks across."""
        import numpy as np

        sprite = np.asarray(fixtures.upscaled_pixel_art(block=4, cells=12).convert("RGB"))
        assert image_probe._plausible_upscale(sprite, 4)
        assert not image_probe._plausible_upscale(sprite, 30)

    def test_a_block_that_divides_neither_axis_is_rejected(self):
        import numpy as np

        art = np.asarray(fixtures.upscaled_pixel_art(block=4, cells=32).convert("RGB"))
        assert not image_probe._plausible_upscale(art, 6)  # 128 % 6 == 2

    def test_a_flat_border_does_not_make_an_atlas_tileable(self, tmp_path):
        """Its edges match because both are the same solid colour."""
        tags, _ = tags_for(fixtures.flat_bordered_atlas(), tmp_path)
        assert "tileable" not in tags

    def test_one_sprite_with_an_empty_half_is_not_a_sheet(self, tmp_path):
        tags, result = tags_for(fixtures.sprite_with_empty_half(), tmp_path)
        assert "spritesheet" not in tags
        assert "cols" not in result.attributes

    def test_a_two_frame_sheet_is_tagged_with_lower_confidence(self, tmp_path):
        _, result = tags_for(fixtures.spritesheet(cols=2, rows=1), tmp_path)
        confidences = {tag: conf for tag, _, conf in result.tags}
        assert confidences["spritesheet"] == 0.6

        _, bigger = tags_for(fixtures.spritesheet(cols=3, rows=2), tmp_path, "b.png")
        assert {t: c for t, _, c in bigger.tags}["spritesheet"] == 0.8


def test_exr_dimensions_come_from_the_header(tmp_path):
    import struct

    body = b"dataWindow\x00box2i\x00" + struct.pack("<i4i", 16, 0, 0, 255, 127)
    path = tmp_path / "bake.exr"
    path.write_bytes(image_probe.EXR_MAGIC + struct.pack("<i", 2) + body + b"\x00")

    result = image_probe.probe(path)
    assert (result.attributes["width"], result.attributes["height"]) == (256, 128)
    assert result.error is None
