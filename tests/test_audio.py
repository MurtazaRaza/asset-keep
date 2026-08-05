"""Audio: the probe's fields, the loudness measurement, and the waveform tile.

Three rules shape what is and is not tested here, all of them the suite's
existing ones applied to a new kind.

**No codec is required.** WAV goes through the stdlib, so the header tests read
real files. Everything ffprobe answers is tested against the JSON it returns,
which is a string, and the loudness and drawing code is tested against numpy
arrays built in the fixture. Nothing here spawns ffmpeg, so nothing here passes
or fails depending on whether Homebrew has been run on this machine.

**Each threshold gets its near miss.** `sfx` at 2.0 seconds and not at 2.1,
`is:clipping` at full scale and not a decibel below it. These detectors fail by
over-triggering, the same way the image ones do.

**What the real files do is measured, not asserted.** The 215-file calibration
run is written up in IMPLEMENTATION.md under M6. A test that asserted "this
sound effect is tagged sfx" against somebody's Unity project would be a test of
that project.
"""

from __future__ import annotations

import json
import subprocess

import numpy as np
import pytest

from assetkeep import db, scan, search, thumbs
from assetkeep.config import Config, RootConfig
from assetkeep.probe import audio

from . import fixtures


# --- header fields ----------------------------------------------------------


def test_wav_header_is_read_in_process(tmp_path, monkeypatch):
    """The stdlib path, and that it really is the stdlib path.

    Worth pinning: the whole reason WAV is special-cased is that it avoids a
    subprocess per file, measured at 3.1 ms against ffprobe's 26 ms. A refactor
    that quietly routed WAV through ffprobe would keep every field correct and
    lose the only thing the special case is for.
    """
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: pytest.fail("WAV should not spawn ffprobe")
    )
    monkeypatch.setattr(audio, "decode_pcm", lambda path: None)

    path = fixtures.wav(tmp_path / "hit.wav", seconds=0.5, rate=22050, channels=2)
    result = audio.probe(path)

    assert result.error is None
    assert result.attributes["duration"] == pytest.approx(0.5, abs=0.01)
    assert result.attributes["sample_rate"] == 22050
    assert result.attributes["channels"] == 2
    assert result.attributes["bit_depth"] == 16
    assert result.attributes["codec"] == "pcm"


@pytest.mark.parametrize("depth", [8, 16, 24])
def test_every_pcm_depth_is_reported(tmp_path, monkeypatch, depth):
    monkeypatch.setattr(audio, "decode_pcm", lambda path: None)
    path = fixtures.wav(tmp_path / f"d{depth}.wav", seconds=0.2, depth=depth)
    assert audio.probe(path).attributes["bit_depth"] == depth


def test_pcm_bit_depth_comes_from_bits_per_sample():
    """The M6 bug, pinned.

    ffprobe leaves ``bits_per_raw_sample`` unset for 16-bit PCM and fills in
    ``bits_per_sample``, so reading only the first recorded no depth for the
    single most common audio format in a game project - 19 of the 215 files
    measured, every WAV that fell through to ffprobe.
    """
    assert audio._bit_depth({"bits_per_sample": 16, "bits_per_raw_sample": None}) == 16


def test_lossy_bit_depth_is_absent_rather_than_zero():
    """vorbis and mp3 both answer ``0``, which is not a depth of zero."""
    assert audio._bit_depth({"bits_per_sample": 0}) is None


def test_ffprobe_fields_are_read_from_its_json(tmp_path, monkeypatch):
    payload = {
        "streams": [
            {
                "codec_type": "audio",
                "codec_name": "vorbis",
                "sample_rate": "48000",
                "channels": 2,
                "bits_per_sample": 0,
                "duration": "3.5",
                "bit_rate": "192000",
            }
        ],
        "format": {"duration": "3.5"},
    }
    _stub_ffprobe(monkeypatch, payload)
    monkeypatch.setattr(audio, "decode_pcm", lambda path: None)

    path = tmp_path / "music.ogg"
    path.write_bytes(b"not really an ogg")
    result = audio.probe(path)

    assert result.attributes["codec"] == "vorbis"
    assert result.attributes["sample_rate"] == 48000
    assert result.attributes["bit_depth"] is None
    assert result.attributes["bitrate"] == 192000


def test_duration_falls_back_to_the_container(tmp_path, monkeypatch):
    """OGG routinely omits the stream-level duration and only has a format one."""
    payload = {
        "streams": [{"codec_type": "audio", "codec_name": "vorbis", "channels": 1}],
        "format": {"duration": "12.0"},
    }
    _stub_ffprobe(monkeypatch, payload)
    monkeypatch.setattr(audio, "decode_pcm", lambda path: None)

    path = tmp_path / "ambience.ogg"
    path.write_bytes(b"x")
    assert audio.probe(path).attributes["duration"] == 12.0


def test_a_wav_the_stdlib_refuses_falls_through_to_ffprobe(tmp_path, monkeypatch):
    """19 of 155 real WAVs are float32, which `wave` will not open."""
    payload = {
        "streams": [
            {
                "codec_type": "audio",
                "codec_name": "pcm_f32le",
                "sample_rate": "44100",
                "channels": 1,
                "bits_per_sample": 32,
                "duration": "1.0",
            }
        ],
        "format": {},
    }
    _stub_ffprobe(monkeypatch, payload)
    monkeypatch.setattr(audio, "decode_pcm", lambda path: None)

    path = tmp_path / "float.wav"
    path.write_bytes(b"RIFF....WAVEfmt not-a-format-the-stdlib-knows")
    result = audio.probe(path)

    assert result.error is None
    assert result.attributes["codec"] == "pcm_f32le"
    assert result.attributes["bit_depth"] == 32


def _stub_ffprobe(monkeypatch, payload: dict) -> None:
    monkeypatch.setattr(audio.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(
        audio.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(
            a[0], 0, json.dumps(payload).encode(), b""
        ),
    )


# --- duration tags ----------------------------------------------------------


@pytest.mark.parametrize(
    "seconds, expected",
    [(0.4, "sfx"), (2.0, "sfx"), (2.1, None), (29.9, None), (30.0, "music")],
)
def test_duration_tags_and_their_near_misses(tmp_path, monkeypatch, seconds, expected):
    """The gap between the thresholds is left untagged on purpose."""
    monkeypatch.setattr(audio, "decode_pcm", lambda path: None)
    # Written at a low rate so a 30-second fixture is not 5 MB on disk.
    path = fixtures.wav(tmp_path / "clip.wav", seconds=seconds, rate=8000)

    names = [name for name, _, _ in audio.probe(path).tags]
    assert names == ([expected] if expected else [])


# --- loudness ---------------------------------------------------------------


@pytest.mark.parametrize(
    "amplitude, expected_db", [(1.0, 0.0), (0.5, -6.0), (0.1, -20.0), (0.01, -40.0)]
)
def test_peak_is_measured_in_dbfs(amplitude, expected_db):
    peak, _ = audio.loudness(fixtures.samples(amplitude=amplitude))
    assert peak == pytest.approx(expected_db, abs=0.2)


def test_rms_is_below_peak_for_a_sine():
    """3 dB below, which is what a sine's crest factor is, and a check that the
    two numbers are actually different measurements rather than one twice."""
    peak, rms = audio.loudness(fixtures.samples(amplitude=1.0))
    assert peak - rms == pytest.approx(3.0, abs=0.2)


def test_silence_hits_the_floor_rather_than_negative_infinity():
    assert audio.loudness(np.zeros(1000)) == (
        audio.SILENCE_FLOOR_DB,
        audio.SILENCE_FLOOR_DB,
    )


def test_empty_input_is_silence_not_a_crash():
    assert audio.loudness(np.array([])) == (
        audio.SILENCE_FLOOR_DB,
        audio.SILENCE_FLOOR_DB,
    )


def test_constant_dc_reads_as_full_scale_with_no_crest():
    """LoftDrop.wav. Peak says full scale, RMS says the same, and the gap
    between them being zero is what says there is no sound in here at all."""
    peak, rms = audio.loudness(fixtures.dc_offset_samples())
    assert peak == pytest.approx(0.0, abs=0.01)
    assert peak - rms == pytest.approx(0.0, abs=0.01)


def test_probe_records_loudness_from_decoded_samples(tmp_path, monkeypatch):
    monkeypatch.setattr(
        audio, "decode_pcm", lambda path: fixtures.samples(amplitude=0.5)
    )
    path = fixtures.wav(tmp_path / "hit.wav", seconds=0.2)
    attributes = audio.probe(path).attributes

    assert attributes["peak_db"] == pytest.approx(-6.0, abs=0.2)
    assert "rms_db" in attributes


def test_no_ffmpeg_means_no_loudness_not_a_failed_probe(tmp_path, monkeypatch):
    """The degradation rule: header fields still land, the file still indexes."""
    monkeypatch.setattr(audio, "decode_pcm", lambda path: None)
    path = fixtures.wav(tmp_path / "hit.wav", seconds=0.2)
    attributes = audio.probe(path).attributes

    assert attributes["duration"] == pytest.approx(0.2, abs=0.01)
    assert "peak_db" not in attributes


# --- the waveform tile ------------------------------------------------------


def test_a_quiet_file_draws_shorter_than_a_loud_one():
    """The M6 change. Peak-normalising drew these two identically, which made
    the one question a wall of waveforms should answer unanswerable."""
    loud = thumbs.render_waveform(fixtures.samples(amplitude=1.0), 128)
    quiet = thumbs.render_waveform(fixtures.samples(amplitude=0.05), 128)
    assert thumbs.drawn_rows(quiet) < thumbs.drawn_rows(loud) / 2


def test_quiet_is_still_visible_rather_than_a_flat_line():
    """Which is why the amplitude is square-rooted: at -26 dB a linear scale
    puts this file inside three pixels of a 128 px tile."""
    quiet = thumbs.render_waveform(fixtures.samples(amplitude=0.05), 128)
    assert thumbs.drawn_rows(quiet) > 8


def test_silence_draws_one_row(tmp_path):
    assert thumbs.drawn_rows(thumbs.render_waveform(np.zeros(8000), 128)) == 1


def test_a_clip_shorter_than_the_tile_is_wide_keeps_all_its_samples():
    """The reshape this replaced dropped anything under 256 samples entirely
    and drew it as silence."""
    tiny = np.array([0.0, 32767.0, -32768.0, 0.0] * 10, dtype=np.float32)
    assert thumbs.drawn_rows(thumbs.render_waveform(tiny, 128)) > 8


def test_full_scale_columns_are_drawn_in_the_warning_colour():
    image = thumbs.render_waveform(fixtures.samples(amplitude=1.0), 128)
    assert _colour_count(image, thumbs.CLIPPING_COLOR) > 0


def test_a_file_below_full_scale_carries_no_warning_colour():
    """The near miss: -1 dB is loud and is not clipping."""
    image = thumbs.render_waveform(fixtures.samples(amplitude=0.89), 128)
    assert _colour_count(image, thumbs.CLIPPING_COLOR) == 0
    assert _colour_count(image, thumbs.WAVEFORM_COLOR) > 0


def test_the_tile_and_the_filter_agree_on_what_clipping_means():
    """Found in the calibration library, not by a test.

    ``MMSequencingBass1.wav`` peaks at -0.03 dB, which ``is:clipping`` matched
    and the tile drew clean, because the tile compared a hard-coded 0.999
    against the square-rooted amplitude - an effective -0.017 dB. Everything
    between the two thresholds was a file the search called clipped and the
    picture called fine.
    """
    # -0.09 dB: inside the gap the two thresholds used to leave between them.
    inside = fixtures.samples(amplitude=0.99)
    peak, _ = audio.loudness(inside)
    assert peak >= audio.CLIPPING_DB  # the filter matches
    assert _colour_count(thumbs.render_waveform(inside, 128), thumbs.CLIPPING_COLOR) > 0

    # -0.13 dB: below both, and they still agree.
    outside = fixtures.samples(amplitude=0.985)
    peak, _ = audio.loudness(outside)
    assert peak < audio.CLIPPING_DB
    assert _colour_count(thumbs.render_waveform(outside, 128), thumbs.CLIPPING_COLOR) == 0


def test_constant_dc_draws_as_a_solid_clipped_block():
    image = thumbs.render_waveform(fixtures.dc_offset_samples(), 128)
    assert _colour_count(image, thumbs.WAVEFORM_COLOR) == 0
    assert _colour_count(image, thumbs.CLIPPING_COLOR) > 128


def _colour_count(image, colour) -> int:
    array = np.asarray(image.convert("RGB"))
    return int((array == np.array(colour, dtype=array.dtype)).all(axis=2).sum())


# --- in the index -----------------------------------------------------------


@pytest.fixture
def library(tmp_path, monkeypatch):
    """A scanned root of audio, with loudness stubbed to known levels.

    Stubbed rather than decoded for the reason at the top of this file, and set
    per file by amplitude in the name, so the search assertions below read as
    the thing they are checking.
    """
    levels = {"loud": 1.0, "quiet": 0.004, "normal": 0.3}

    def decode(path):
        return fixtures.samples(amplitude=levels.get(path.stem.split("_")[0], 0.3))

    monkeypatch.setattr(audio, "decode_pcm", decode)

    root = tmp_path / "sounds"
    root.mkdir()
    fixtures.wav(root / "loud_hit.wav", seconds=0.5, channels=1, rate=44100)
    fixtures.wav(root / "quiet_wind.wav", seconds=0.5, channels=2, rate=48000)
    fixtures.wav(root / "normal_step.wav", seconds=0.5, channels=2, rate=96000)

    cfg = Config(
        db_path=tmp_path / "index.db",
        roots=(RootConfig(path=root),),
        source_path=tmp_path / "config.toml",
    )
    conn = db.connect(cfg.db_path)
    scan.scan(conn, cfg)
    yield conn
    conn.close()


def titles(conn, query):
    return sorted(row["title"] for row in search.search(conn, query))


def test_audio_is_indexed_as_audio(library):
    assert titles(library, "kind:audio") == ["loud_hit", "normal_step", "quiet_wind"]


def test_channels_filter_separates_mono_from_stereo(library):
    assert titles(library, "kind:audio channels:1") == ["loud_hit"]
    assert titles(library, "is:mono") == ["loud_hit"]
    assert titles(library, "is:stereo") == ["normal_step", "quiet_wind"]


def test_sample_rate_filter(library):
    """`rate:>48000` is the query that finds bytes wasted on a sound effect."""
    assert titles(library, "rate:>48000") == ["normal_step"]
    assert titles(library, "rate:44100") == ["loud_hit"]


def test_clipping_and_silence_are_findable_without_typing_a_number(library):
    assert titles(library, "is:clipping") == ["loud_hit"]
    assert titles(library, "is:silent") == ["quiet_wind"]


def test_peak_takes_a_negative_threshold(library):
    """Decibels are the only negative numbers in the grammar, and `-6` at the
    front of a value must not be read as the `-tag:` negation."""
    assert titles(library, "peak:>-3db") == ["loud_hit"]
    assert titles(library, "peak:<-30db") == ["quiet_wind"]


def test_bit_depth_and_bitrate_are_recorded_for_pcm(library):
    assert titles(library, "depth:16") == ["loud_hit", "normal_step", "quiet_wind"]
    # 44100 * 1 * 16 for the mono file, which is the only one below 1 Mbps.
    assert titles(library, "bitrate:<1000000") == ["loud_hit"]
