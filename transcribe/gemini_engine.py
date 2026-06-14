"""
Google Gemini Flash transcription backend.

Gemini 1.5/2.0 Flash has strong Bengali support and a generous free tier
via Google AI Studio:
  - 1,500 requests/day free
  - 15 requests/minute
  - Audio: 1 token per second (~150 tokens for a 2.5-min call)

Get a free API key at: https://aistudio.google.com

Requires:
    pip install google-genai
    export GOOGLE_API_KEY=your_key_here

Privacy: audio is uploaded to Google's servers.
"""

from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# Prompt that asks Gemini to return structured speaker-labelled Bengali text.
# We also ask for approximate timestamps so we can build SRT output.
_PROMPT = """\
Transcribe this audio call recording accurately.

Rules:
- Transcribe EXACTLY what is said — do not summarise or paraphrase.
- Use the native script of the language spoken (e.g. Bengali script for Bengali).
- Label each speaker turn as [Speaker 1]: or [Speaker 2]: (or Agent/Customer if those roles are clear from context).
- Include approximate timestamps in [MM:SS] format at the start of each turn.
- If you cannot hear a section clearly, write [inaudible].

Format example:
[00:00] [Agent]: আসসালামু আলাইকুম।
[00:03] [Customer]: হ্যালো, আমি জানতে চাই...

Output ONLY the transcript. No explanation, no preamble.
"""

# Regex to parse "[MM:SS] [Speaker]: text" lines from Gemini's response.
_LINE_RE = re.compile(
    r"^\[?(\d{1,2}:\d{2})\]?\s*\[?([^\]:\n]+)\]?:\s*(.+)$",
    re.MULTILINE,
)


def _parse_transcript(text: str, speaker_labels: tuple) -> tuple[list, str]:
    """Parse Gemini's free-text response into our standard segment list."""
    segments = []
    label_map: dict[str, str] = {}
    raw_lines = []

    for m in _LINE_RE.finditer(text):
        ts_str, raw_spk, content = m.group(1), m.group(2).strip(), m.group(3).strip()
        # Map "Speaker 1"/"Agent" → user's labels
        key = raw_spk.lower().replace(" ", "")
        if key not in label_map:
            idx = len(label_map)
            label_map[key] = speaker_labels[idx] if idx < len(speaker_labels) else raw_spk
        speaker = label_map[key]

        # Parse MM:SS → seconds
        parts = ts_str.split(":")
        start = int(parts[0]) * 60 + float(parts[1])
        segments.append({
            "start": start,
            "end": start,        # end filled in next pass
            "speaker": speaker,
            "text": content,
            "avg_logprob": 0.0,
            "no_speech_prob": 0.0,
            "compression_ratio": 1.0,
        })
        raw_lines.append(f"[{speaker}]: {content}")

    # Fill in end times
    for i, seg in enumerate(segments[:-1]):
        seg["end"] = segments[i + 1]["start"]
    if segments:
        segments[-1]["end"] = segments[-1]["start"] + 5.0

    # If parsing produced nothing (Gemini returned plain text), wrap it all
    # in a single segment so output files are still written.
    if not segments and text.strip():
        segments = [{
            "start": 0.0, "end": 0.0, "speaker": "",
            "text": text.strip(),
            "avg_logprob": 0.0, "no_speech_prob": 0.0, "compression_ratio": 1.0,
        }]
        raw_lines = [text.strip()]

    full_text = "\n".join(raw_lines)
    return segments, full_text


class GeminiTranscriber:
    """Wraps Google Gemini Flash for audio transcription."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_id: str = "gemini-2.5-flash",
    ):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY", "")
        self.model_id = model_id
        if not self.api_key:
            raise ValueError(
                "Google API key not set. "
                "Pass api_key= or set the GOOGLE_API_KEY environment variable. "
                "Get a free key at https://aistudio.google.com"
            )
        try:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        except ImportError:
            raise SystemExit(
                "google-genai package not installed. Run: pip install google-genai"
            )
        log.info("Gemini transcription backend ready (model=%s)", self.model_id)

    def transcribe(
        self,
        audio_path,
        language: Optional[str] = None,
        speaker_labels: tuple = ("Speaker A", "Speaker B"),
        diarize: bool = True,  # accepted for interface parity; diarization is always on via prompt
    ) -> dict:
        """Upload audio to Gemini and transcribe.

        Gemini handles stereo natively — no manual channel splitting needed.
        For files over ~20 MB the File API is used automatically.
        """
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        size_mb = path.stat().st_size / 1_048_576
        log.info("Transcribing via Gemini %s: %s (%.1f MB)", self.model_id, path.name, size_mb)
        t = time.time()

        from google import genai
        from google.genai import types

        mime = _mime_type(path.suffix)

        # Build a language hint if the caller specified one.
        lang_hint = ""
        if language:
            lang_hint = f"\nThe language spoken is primarily {language} (ISO code). "

        prompt = _PROMPT + lang_hint

        if size_mb > 19:
            # Use the File API for large files (avoids base64 bloat in the request).
            log.info("  File > 19 MB — uploading via Gemini File API...")
            uploaded = self._client.files.upload(file=str(path), config={"mime_type": mime})
            parts = [
                types.Part.from_uri(file_uri=uploaded.uri, mime_type=mime),
                types.Part.from_text(text=prompt),
            ]
            # Clean up the uploaded file after we're done.
            cleanup_file = uploaded.name
        else:
            audio_bytes = path.read_bytes()
            parts = [
                types.Part.from_bytes(data=audio_bytes, mime_type=mime),
                types.Part.from_text(text=prompt),
            ]
            cleanup_file = None

        response = self._client.models.generate_content(
            model=self.model_id,
            contents=parts,
        )

        if cleanup_file:
            try:
                self._client.files.delete(name=cleanup_file)
            except Exception:
                pass

        raw_text = response.text or ""
        segments, full_text = _parse_transcript(raw_text, speaker_labels)
        elapsed = time.time() - t

        log.info(
            "  Done: %d segments, took %.1fs",
            len(segments), elapsed,
        )
        return {
            "segments": segments,
            "full_text": full_text,
            "language_detected": language or "auto",
            "language_confidence": 1.0,
            "duration_seconds": segments[-1]["end"] if segments else 0.0,
            "processing_time_seconds": round(elapsed, 2),
            "speakers_separated": True,
        }


def _mime_type(suffix: str) -> str:
    return {
        ".mp3": "audio/mp3",
        ".wav": "audio/wav",
        ".ogg": "audio/ogg",
        ".opus": "audio/opus",
        ".flac": "audio/flac",
        ".m4a": "audio/mp4",
        ".aac": "audio/aac",
        ".amr": "audio/amr",
    }.get(suffix.lower(), "audio/wav")
