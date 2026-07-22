"""
Call quality analysis using any OpenAI-compatible Chat Completions endpoint.

Reads transcript .json files produced by the transcription pipeline, sends
the conversation text to a configured OpenAI-compatible model, and appends an
'analysis' block back into the same JSON file.

Configure with environment variables or CLI flags:
    OPENAI_API_KEY       API key for the compatible endpoint
    OPENAI_BASE_URL      Base URL, default https://api.openai.com/v1
    OPENAI_MODEL         Model name, default gpt-4o-mini

Usage:
    transcribe-analyze --file transcripts/call.json
    transcribe-analyze --input transcripts/
    transcribe-analyze --input transcripts/ --reanalyze
"""

from __future__ import annotations

import csv
import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://api.openai.com/v1"
_DEFAULT_MODEL = "gpt-4o-mini"

_PROMPT_TEMPLATE = """\
You are a call quality analyst. Analyze the following customer service call transcript.
The transcript uses [Agent] and [Customer] speaker labels.

TRANSCRIPT:
{transcript}

Return ONLY a valid JSON object with exactly these fields (no markdown, no explanation):
{{
  "brand":              "the company or product/service name mentioned in the call, or null if not identifiable",
  "issue_category":     one of ["billing","technical","sales","complaint","inquiry","other"],
  "issue_summary":      "one sentence - what did the customer need?",
  "resolution":         one of ["resolved","unresolved","escalated","partial"],
  "resolution_note":    "one sentence - what was resolved or left open?",
  "customer_sentiment": one of ["satisfied","neutral","frustrated","angry"],
  "sentiment_score":    integer 1 (very negative) to 5 (very positive),
  "agent_score":        integer 0-100 overall quality score,
  "agent_flags":        ["specific problems observed - empty list if none"],
  "strengths":          ["specific things the agent did well - empty list if none"],
  "coaching_tip":       "one actionable sentence to improve this agent's performance"
}}
"""


@dataclass
class CallAnalysis:
    call_id: str
    brand: str
    issue_category: str
    issue_summary: str
    resolution: str
    resolution_note: str
    customer_sentiment: str
    sentiment_score: int
    agent_score: int
    agent_flags: list
    strengths: list
    coaching_tip: str
    analyzed_at: str
    model_used: str


class CallAnalyzer:
    """Analyzes transcript JSON files via an OpenAI-compatible endpoint."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_id: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout_seconds: int = 120,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL") or _DEFAULT_BASE_URL).rstrip("/")
        self.model_id = model_id or os.getenv("OPENAI_MODEL") or _DEFAULT_MODEL
        self.timeout_seconds = timeout_seconds
        if not self.api_key:
            raise SystemExit(
                "OpenAI-compatible analysis API key is required. "
                "Set OPENAI_API_KEY or pass --api-key."
            )
        log.info(
            "CallAnalyzer ready (provider=openai-compatible, base_url=%s, model=%s)",
            self.base_url,
            self.model_id,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _chat_completions_url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _call_model(self, prompt: str) -> str:
        """Call an OpenAI-compatible /chat/completions endpoint."""
        payload = {
            "model": self.model_id,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a precise call QA analyst. Return strict JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self._chat_completions_url(),
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"OpenAI-compatible analysis request failed: HTTP {exc.code}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"OpenAI-compatible analysis request failed: {exc.reason}"
            ) from exc

        try:
            parsed = json.loads(body)
            return parsed["choices"][0]["message"]["content"] or "{}"
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Unexpected analysis response shape: {body[:500]}") from exc

    # ------------------------------------------------------------------
    # Single file
    # ------------------------------------------------------------------

    def analyze_file(
        self,
        json_path,
        reanalyze: bool = False,
    ) -> Optional[CallAnalysis]:
        """Analyze one transcript JSON. Appends 'analysis' block in-place.

        Returns the CallAnalysis, or None if the file was skipped.
        """
        path = Path(json_path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        if not reanalyze and "analysis" in data:
            log.info("  Skipping (already analyzed): %s", path.name)
            return None

        transcript = data.get("full_text", "").strip()
        if not transcript:
            log.warning("  No transcript text in %s - skipping", path.name)
            return None

        call_id = data.get("call_id", path.stem)
        log.info("  Analyzing: %s", path.name)
        t0 = time.time()

        prompt = _PROMPT_TEMPLATE.format(transcript=transcript)

        try:
            raw = self._call_model(prompt).strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
            result = json.loads(raw)
        except Exception as e:
            log.error("  OpenAI-compatible analysis failed for %s: %s", path.name, e)
            raise

        analysis = CallAnalysis(
            call_id=call_id,
            brand=result.get("brand", "unknown"),
            issue_category=result.get("issue_category", "other"),
            issue_summary=result.get("issue_summary", ""),
            resolution=result.get("resolution", "unresolved"),
            resolution_note=result.get("resolution_note", ""),
            customer_sentiment=result.get("customer_sentiment", "neutral"),
            sentiment_score=int(result.get("sentiment_score", 3)),
            agent_score=int(result.get("agent_score", 0)),
            agent_flags=result.get("agent_flags", []),
            strengths=result.get("strengths", []),
            coaching_tip=result.get("coaching_tip", ""),
            analyzed_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            model_used=f"openai-compatible/{self.model_id}",
        )

        # Append analysis into the transcript JSON in-place
        data["analysis"] = asdict(analysis)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        elapsed = round(time.time() - t0, 2)
        log.info(
            "  Done: score=%d  resolution=%-10s  sentiment=%s  (%.1fs)",
            analysis.agent_score, analysis.resolution,
            analysis.customer_sentiment, elapsed,
        )
        return analysis

    # ------------------------------------------------------------------
    # Batch
    # ------------------------------------------------------------------

    def analyze_directory(
        self,
        transcripts_dir,
        reanalyze: bool = False,
        write_csv: bool = True,
    ) -> list[CallAnalysis]:
        """Analyze all transcript JSONs in a directory.

        Skips processed_files.json. Writes analysis_summary.csv covering
        ALL analyzed calls (including ones analyzed in prior runs).
        """
        d = Path(transcripts_dir)
        json_files = sorted(
            f for f in d.glob("*.json")
            if f.name not in ("processed_files.json",)
        )

        if not json_files:
            log.warning("No transcript JSON files found in %s", d)
            return []

        log.info("Found %d transcript(s) to process", len(json_files))
        newly_analyzed: list[CallAnalysis] = []
        for i, f in enumerate(json_files, 1):
            log.info("[%d/%d] %s", i, len(json_files), f.name)
            try:
                result = self.analyze_file(f, reanalyze=reanalyze)
                if result:
                    newly_analyzed.append(result)
            except Exception as e:
                log.error("  Failed: %s - %s", f.name, e)

        # Rebuild CSV from ALL analyzed calls in the directory
        if write_csv:
            all_analyses = _collect_analyses(d)
            if all_analyses:
                csv_path = d / "analysis_summary.csv"
                _write_csv(all_analyses, csv_path)
                log.info("Summary CSV updated: %s (%d calls)", csv_path, len(all_analyses))

        return newly_analyzed


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _collect_analyses(directory: Path) -> list[dict]:
    """Read the 'analysis' block from every analyzed transcript JSON."""
    rows = []
    for f in sorted(directory.glob("*.json")):
        if f.name in ("processed_files.json",):
            continue
        try:
            with open(f, encoding="utf-8") as fh:
                data = json.load(fh)
            if "analysis" not in data:
                continue
            row = {
                "call_id":          data.get("call_id", f.stem),
                "filename":         data.get("filename", f.name),
                "date":             data.get("date", ""),
                "duration_min":     round(data.get("duration_seconds", 0) / 60, 1),
                "word_count":       data.get("word_count", ""),
                **data["analysis"],
            }
            # Flatten list fields to semicolon-separated strings for CSV
            row["agent_flags"] = "; ".join(row.get("agent_flags") or [])
            row["strengths"]   = "; ".join(row.get("strengths") or [])
            rows.append(row)
        except Exception:
            pass
    return rows


_CSV_COLUMNS = [
    "call_id", "filename", "date", "duration_min", "word_count",
    "brand", "issue_category", "issue_summary",
    "resolution", "resolution_note",
    "customer_sentiment", "sentiment_score", "agent_score",
    "agent_flags", "strengths", "coaching_tip",
    "analyzed_at", "model_used",
]


def _write_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
