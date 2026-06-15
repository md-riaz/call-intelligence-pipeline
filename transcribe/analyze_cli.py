"""Command-line interface: `transcribe-analyze`."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import __version__
from .analyze import CallAnalyzer
from .config import load_config


def _setup_logging(output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                str(Path(output_dir) / "analysis.log"), encoding="utf-8"
            ),
        ],
    )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="transcribe-analyze",
        description="Analyze call transcript JSON files with Google Gemini. "
        "Scores agent quality, detects issue type and resolution, sentiment, "
        "and writes a summary CSV for management review.",
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--file", "-f", help="A single transcript .json file to analyze")
    src.add_argument(
        "--input", "-i",
        help="Directory of transcript .json files (from `transcribe` output)",
    )
    ap.add_argument(
        "--google-api-key", default=None,
        help="Google AI Studio API key (overrides GOOGLE_API_KEY env var)",
    )
    ap.add_argument(
        "--model", "-m", default=None,
        help="Gemini model ID for analysis (default: gemini-2.5-flash). "
             "Use gemini-3.1-flash-lite for 500 RPD free tier.",
    )
    ap.add_argument(
        "--reanalyze", action="store_true",
        help="Re-analyze files that already have an analysis block",
    )
    ap.add_argument(
        "--no-csv", action="store_true",
        help="Skip writing analysis_summary.csv (batch mode only)",
    )
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    output_dir = args.input if args.input else str(Path(args.file).parent)
    _setup_logging(output_dir)
    log = logging.getLogger(__name__)
    log.info("transcribe-analyze %s", __version__)

    cfg = load_config()
    model = args.model or cfg.get("GEMINI_MODEL", "gemini-3.1-flash-lite")
    log.info("Analysis model: %s", model)

    analyzer = CallAnalyzer(api_key=args.google_api_key, model_id=model)

    if args.file:
        path = Path(args.file)
        if not path.exists():
            log.error("File not found: %s", args.file)
            return 1
        try:
            result = analyzer.analyze_file(path, reanalyze=args.reanalyze)
            if result:
                _print_report(result)
        except Exception as e:
            log.error("Analysis failed: %s", e)
            return 1
        return 0

    # Batch mode
    d = Path(args.input)
    if not d.exists():
        log.error("Directory not found: %s", args.input)
        return 1
    results = analyzer.analyze_directory(
        d,
        reanalyze=args.reanalyze,
        write_csv=not args.no_csv,
    )
    if results:
        _print_batch_summary(results)
    return 0


def _print_report(a) -> None:
    """Print a human-readable single-call report to stdout."""
    flags = "\n    ".join(a.agent_flags) if a.agent_flags else "none"
    strengths = "\n    ".join(a.strengths) if a.strengths else "none"
    print(f"""
{'=' * 60}
Call ID  : {a.call_id}
Brand    : {a.brand}
Category : {a.issue_category}
Issue    : {a.issue_summary}
{'=' * 60}
Resolution  : {a.resolution.upper()} — {a.resolution_note}
Sentiment   : {a.customer_sentiment} ({a.sentiment_score}/5)
Agent score : {a.agent_score}/100
{'=' * 60}
Flags    :
    {flags}
Strengths:
    {strengths}
Coaching : {a.coaching_tip}
{'=' * 60}""")


def _print_batch_summary(results: list) -> None:
    """Print a compact batch summary table."""
    print(f"\n{'=' * 70}")
    print(f"{'CALL ID':<20} {'BRAND':<12} {'SCORE':>5}  {'RESOLUTION':<12} {'SENTIMENT'}")
    print(f"{'-' * 70}")
    for a in sorted(results, key=lambda x: x.agent_score):
        print(f"{a.call_id:<20} {a.brand:<12} {a.agent_score:>5}  {a.resolution:<12} {a.customer_sentiment}")
    scores = [a.agent_score for a in results]
    print(f"{'-' * 70}")
    print(f"Analyzed: {len(results)}  |  Avg score: {sum(scores)//len(scores)}/100  |  "
          f"Resolved: {sum(1 for a in results if a.resolution == 'resolved')}/{len(results)}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    raise SystemExit(main())
