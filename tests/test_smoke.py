"""Lightweight smoke tests — no model download or audio required.

These verify the package imports, the CLI parses arguments, and the pure
helpers behave. Run with: pytest
"""

from transcribe import __version__
from transcribe.audio import AudioPreprocessor, StereoSplitter
from transcribe.cli import build_parser
from transcribe.pipeline import _srt_timestamp


def test_version():
    assert isinstance(__version__, str) and __version__


def test_supported_formats():
    assert AudioPreprocessor.supported("a.wav")
    assert AudioPreprocessor.supported("CALL.MP3")
    assert not AudioPreprocessor.supported("notes.txt")


def test_srt_timestamp():
    assert _srt_timestamp(0) == "00:00:00,000"
    assert _srt_timestamp(3661.5) == "01:01:01,500"


def test_merge_orders_by_start_and_labels():
    left = [{"start": 0.0, "end": 1.0, "text": "hi"}]
    right = [{"start": 0.5, "end": 1.5, "text": "yo"}]
    merged = StereoSplitter.merge(left, right, labels=("Agent", "Customer"))
    assert [s["speaker"] for s in merged] == ["Agent", "Customer"]
    assert merged[0]["start"] <= merged[1]["start"]


def test_cli_parses_file_arg():
    args = build_parser().parse_args(["--file", "x.wav", "--language", "bn"])
    assert args.file == "x.wav"
    assert args.language == "bn"


def test_cli_parses_elevenlabs_engine():
    args = build_parser().parse_args(
        ["--file", "x.wav", "--engine", "elevenlabs", "--language", "ben"]
    )
    assert args.engine == "elevenlabs"
    assert args.language == "ben"


def test_cli_requires_a_source():
    import pytest

    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_elevenlabs_word_grouping():
    from transcribe.elevenlabs_engine import _words_to_segments

    words = [
        {"text": "hello", "start": 0.1, "end": 0.4, "speaker_id": "speaker_0", "type": "word"},
        {"text": "world", "start": 0.5, "end": 0.9, "speaker_id": "speaker_0", "type": "word"},
        {"text": "hi",    "start": 3.5, "end": 4.0, "speaker_id": "speaker_1", "type": "word"},
        {"text": "ok",    "start": 6.5, "end": 6.8, "speaker_id": "speaker_0", "type": "word"},
    ]
    segs, text = _words_to_segments(words, ("Agent", "Customer"))
    assert len(segs) == 3
    assert segs[0]["speaker"] == "Agent"
    assert segs[1]["speaker"] == "Customer"
    assert segs[2]["speaker"] == "Agent"
    assert "hello world" in segs[0]["text"]
