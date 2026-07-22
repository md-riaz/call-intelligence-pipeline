"""Lightweight smoke tests — no API calls or audio required.

These verify the package imports, the CLI parses arguments, and the pure
helpers behave. Run with: pytest
"""

from transcribe import __version__
from transcribe.audio import AudioPreprocessor, StereoSplitter
from transcribe.cli import build_parser
from transcribe.analyze_cli import build_parser as build_analyze_parser
from transcribe.pipeline import _srt_timestamp


def test_version():
    assert isinstance(__version__, str) and __version__


def test_supported_formats():
    assert AudioPreprocessor.supported("a.wav")
    assert AudioPreprocessor.supported("CALL.MP3")
    assert not AudioPreprocessor.supported("notes.txt")


def test_supported_extensionless_audio_probe(tmp_path, monkeypatch):
    audio = tmp_path / "savedly_sample"
    audio.write_bytes(b"not actually decoded because ffprobe is mocked")

    class Result:
        returncode = 0
        stdout = "audio\n"

    def fake_run(cmd, capture_output, text):
        assert cmd[0] == "ffprobe"
        assert str(audio) == cmd[-1]
        assert capture_output is True
        assert text is True
        return Result()

    monkeypatch.setattr("transcribe.audio.subprocess.run", fake_run)

    assert AudioPreprocessor.supported(audio)


def test_unsupported_extensionless_file_probe_failure(tmp_path, monkeypatch):
    unknown = tmp_path / "notes"
    unknown.write_text("plain text")

    class Result:
        returncode = 1
        stdout = ""

    monkeypatch.setattr("transcribe.audio.subprocess.run", lambda *a, **k: Result())

    assert not AudioPreprocessor.supported(unknown)


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


def test_cli_parses_google_api_key():
    args = build_parser().parse_args(
        ["--file", "x.wav", "--language", "bn", "--google-api-key", "k"]
    )
    assert args.google_api_key == "k"


def test_cli_model_flag():
    args = build_parser().parse_args(
        ["--file", "x.wav", "--model", "gemini-3.1-flash-lite"]
    )
    assert args.model == "gemini-3.1-flash-lite"


def test_cli_model_flag_25flash():
    args = build_parser().parse_args(
        ["--file", "x.wav", "--model", "gemini-2.5-flash"]
    )
    assert args.model == "gemini-2.5-flash"


def test_cli_requires_a_source():
    import pytest
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_analyze_cli_parses_file_arg():
    args = build_analyze_parser().parse_args(["--file", "transcripts/call.json"])
    assert args.file == "transcripts/call.json"
    assert not args.reanalyze


def test_analyze_cli_parses_batch_args():
    args = build_analyze_parser().parse_args(
        ["--input", "transcripts/", "--reanalyze", "--no-csv"]
    )
    assert args.input == "transcripts/"
    assert args.reanalyze
    assert args.no_csv


def test_analyze_cli_model_flag():
    args = build_analyze_parser().parse_args(
        ["--file", "t.json", "--model", "gemini-3.1-flash-lite"]
    )
    assert args.model == "gemini-3.1-flash-lite"


def test_analyze_csv_columns():
    from transcribe.analyze import _CSV_COLUMNS
    required = {"call_id", "brand", "issue_category", "resolution",
                "agent_score", "customer_sentiment", "coaching_tip"}
    assert required.issubset(set(_CSV_COLUMNS))


def test_key_pool_single_key():
    from transcribe.key_pool import GeminiKeyPool
    pool = GeminiKeyPool(["fake-key-1"])
    assert pool.total_count == 1
    assert pool.active_count == 1


def test_key_pool_deduplicates():
    from transcribe.key_pool import GeminiKeyPool
    pool = GeminiKeyPool(["k1", "k2", "k1", "k3"])
    assert pool.total_count == 3


def test_key_pool_rotation():
    from transcribe.key_pool import GeminiKeyPool, AllKeysExhaustedError
    pool = GeminiKeyPool(["k1", "k2"])
    pool._clients = {0: "client0", 1: "client1"}
    _, idx = pool.get_client()
    assert idx == 0
    pool.mark_exhausted(0)
    _, idx = pool.get_client()
    assert idx == 1
    pool.mark_exhausted(1)
    try:
        pool.get_client()
        assert False, "Should have raised AllKeysExhaustedError"
    except AllKeysExhaustedError:
        pass


def test_cli_parses_engine_whisper_sam15000():
    args = build_parser().parse_args(
        ["--file", "x.wav", "--engine", "whisper-sam15000", "--language", "bn"]
    )
    assert args.engine == "whisper-sam15000"
    assert args.language == "bn"




def test_cli_parses_engine_whisper_tugstugi():
    args = build_parser().parse_args(
        ["--file", "x.wav", "--engine", "whisper-tugstugi", "--language", "bn"]
    )
    assert args.engine == "whisper-tugstugi"
    assert args.language == "bn"


def test_tugstugi_factory_uses_default_model(monkeypatch):
    from transcribe.backends import base

    captured = {}

    class FakeBackend:
        def __init__(self, model_id=None, temp_dir=None, provider_name=""):
            captured["model_id"] = model_id
            captured["provider_name"] = provider_name

    monkeypatch.setattr(
        "transcribe.backends.whisper_sam15000.WhisperSam15000Backend", FakeBackend
    )
    backend = base.create_backend("whisper-tugstugi")
    assert isinstance(backend, FakeBackend)
    assert captured["model_id"] == "bengaliAI/tugstugi_bengaliai-asr_whisper-medium"
    assert captured["provider_name"] == "whisper-tugstugi"

def test_backend_factory_rejects_unknown_engine():
    import pytest
    from transcribe.backends.base import create_backend

    with pytest.raises(ValueError):
        create_backend("not-real")


def test_whisper_timestamp_pair():
    from transcribe.backends.whisper_sam15000 import _timestamp_pair

    assert _timestamp_pair((None, 2.5)) == (0.0, 2.5)
    assert _timestamp_pair((1, None)) == (1.0, 1.0)


def test_whisper_segments_to_text():
    from transcribe.backends.whisper_sam15000 import _segments_to_text

    text = _segments_to_text([{"speaker": "Agent", "text": "হ্যালো"}])
    assert text == "[Agent]: হ্যালো"
