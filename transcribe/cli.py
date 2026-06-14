"""Command-line interface: `transcribe`."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import __version__
from .config import load_config
from .pipeline import TranscriptionPipeline

MODELS = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]


def _setup_logging(output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                str(Path(output_dir) / "transcription.log"), encoding="utf-8"
            ),
        ],
    )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="transcribe",
        description="Transcribe audio call recordings into speaker-labelled text "
        "using faster-whisper. Works with any language and any ffmpeg-readable format.",
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", "-i", help="Directory of recordings (processed recursively)")
    src.add_argument("--file", "-f", help="A single audio file")

    ap.add_argument("--output", "-o", default="./transcripts", help="Output directory")
    ap.add_argument(
        "--language", "-l", default=None,
        help="ISO language code to force (e.g. en, bn, hi, ar, es). "
        "Omit or use 'auto' to auto-detect.",
    )
    ap.add_argument("--model", "-m", default=None, choices=MODELS,
                    help="Whisper model (default: from config.env or large-v3)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                    help="Compute device (default: auto)")
    ap.add_argument("--compute-type", default=None,
                    help="ctranslate2 compute type (default: int8 on CPU, float16 on GPU)")
    ap.add_argument("--days", "-d", type=int, default=None,
                    help="With --input: only files modified in the last N days")
    ap.add_argument("--labels", default="Speaker A,Speaker B",
                    help="Comma-separated labels for stereo channels "
                    "(e.g. 'Agent,Customer'). Default: 'Speaker A,Speaker B'")
    ap.add_argument("--no-separate-speakers", action="store_true",
                    help="Mix stereo down to mono instead of transcribing channels separately")
    ap.add_argument("--no-srt", action="store_true", help="Do not write .srt subtitles")
    ap.add_argument("--reprocess", action="store_true",
                    help="Re-transcribe files even if already in processed_files.json")
    ap.add_argument(
        "--engine", default="gemini", choices=["whisper", "gemini"],
        help="Transcription backend. 'gemini' = Google Gemini Flash API, free tier, "
        "excellent Bengali accuracy (default). Requires GOOGLE_API_KEY env var or "
        "--google-api-key. 'whisper' = local faster-whisper, fully offline/private.",
    )
    ap.add_argument("--google-api-key", default=None,
                    help="Google AI Studio API key for Gemini (overrides GOOGLE_API_KEY env var). "
                    "Get a free key at https://aistudio.google.com")
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.output)
    log = logging.getLogger(__name__)

    cfg = load_config()
    model = args.model or cfg.get("WHISPER_MODEL", "large-v3")
    language = None if (args.language in (None, "auto")) else args.language
    labels = tuple((args.labels.split(",", 1) + ["Speaker B"])[:2])

    log.info(
        "audio-transcription-pipeline %s | engine=%s | model=%s | lang=%s | device=%s",
        __version__, args.engine, model, language or "auto", args.device,
    )

    pipeline = TranscriptionPipeline(
        model_size=model,
        output_dir=args.output,
        language=language,
        device=args.device,
        compute_type=args.compute_type,
        speaker_labels=labels,
        separate_speakers=not args.no_separate_speakers,
        write_srt=not args.no_srt,
        engine=args.engine,
        google_api_key=args.google_api_key,
    )

    if args.file:
        if not Path(args.file).exists():
            log.error("File not found: %s", args.file)
            return 1
        t = pipeline.process_file(args.file, skip_done=not args.reprocess)
        return 0 if t.status in ("success", "skipped") else 1

    pipeline.process_directory(args.input, days=args.days)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
