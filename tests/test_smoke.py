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


def test_cli_parses_gemini_engine():
    args = build_parser().parse_args(
        ["--file", "x.wav", "--engine", "gemini", "--language", "bn", "--google-api-key", "k"]
    )
    assert args.engine == "gemini"
    assert args.google_api_key == "k"


def test_cli_requires_a_source():
    import pytest

    with pytest.raises(SystemExit):
        build_parser().parse_args([])


