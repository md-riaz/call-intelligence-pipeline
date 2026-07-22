"""Command-line interface: `transcribe`."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import __version__
from .config import load_config
from .pipeline import TranscriptionPipeline

_GEMINI_DEFAULT = "gemini-3.1-flash-lite"


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
        "using the local Bengali whisper-bn ASR backend.",
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", "-i", help="Directory of recordings (processed recursively)")
    src.add_argument("--file", "-f", help="A single audio file")

    ap.add_argument("--output", "-o", default="./transcripts", help="Output directory")
    ap.add_argument(
        "--language", "-l", default=None,
        help="ISO language code to force (e.g. bn, en, hi). Omit for auto-detect.",
    )
    ap.add_argument(
        "--engine",
        choices=("whisper-bn", "whisper-sam15000", "whisper-tugstugi", "gemini"),
        default=None,
        help="ASR engine. Defaults to MODEL_PROVIDER or whisper-bn.",
    )
    ap.add_argument(
        "--model", "-m", default=None,
        help="Backend model ID. For Gemini: gemini-3.1-flash-lite (default) or "
             "gemini-2.5-flash. For whisper-sam15000, omit unless overriding "
             "WHISPER_MODEL. For whisper-tugstugi, omit to use BengaliAI Tugstugi.",
    )
    ap.add_argument("--days", "-d", type=int, default=None,
                    help="With --input: only files modified in the last N days")
    ap.add_argument("--labels", default="Speaker A,Speaker B",
                    help="Comma-separated speaker labels (e.g. 'Agent,Customer')")
    ap.add_argument("--no-srt", action="store_true", help="Do not write .srt subtitles")
    ap.add_argument("--reprocess", action="store_true",
                    help="Re-transcribe files even if already done")
    ap.add_argument("--google-api-key", default=None,
                    help="Google AI Studio API key (overrides GOOGLE_API_KEY). "
                    "Supports comma-separated multiple keys for rate-limit rotation.")
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.output)
    log = logging.getLogger(__name__)

    cfg = load_config()
    language = None if (args.language in (None, "auto")) else args.language
    labels = tuple((args.labels.split(",", 1) + ["Speaker B"])[:2])
    engine = args.engine or cfg.get("MODEL_PROVIDER", "whisper-bn")
    if engine == "gemini":
        gemini_model = args.model or cfg.get("GEMINI_MODEL", _GEMINI_DEFAULT)
        whisper_model = cfg.get("WHISPER_MODEL")
        model_for_log = gemini_model
    else:
        gemini_model = cfg.get("GEMINI_MODEL", _GEMINI_DEFAULT)
        if engine == "whisper-tugstugi":
            # Keep Tugstugi pluggable even when Docker/host config sets the
            # legacy SAM15K WHISPER_MODEL default. Use --model for explicit
            # Tugstugi overrides instead of inheriting the global SAM15K value.
            whisper_model = (
                args.model or "bengaliAI/tugstugi_bengaliai-asr_whisper-medium"
            )
        else:
            whisper_model = args.model or cfg.get(
                "WHISPER_MODEL", "bitwisemind/sam_15000_clean_text_full_model"
            )
        model_for_log = whisper_model

    log.info(
        "call-intelligence-pipeline %s | engine=%s | model=%s | lang=%s",
        __version__, engine, model_for_log, language or "auto",
    )

    pipeline = TranscriptionPipeline(
        output_dir=args.output,
        language=language,
        speaker_labels=labels,
        write_srt=not args.no_srt,
        google_api_key=args.google_api_key,
        gemini_model_id=gemini_model,
        engine=engine,
        whisper_model_id=whisper_model,
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
