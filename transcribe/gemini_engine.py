"""
Google Gemini transcription backend.

Default model: gemini-3.1-flash-lite
  - 500 RPD free, 15 RPM
  - Excellent Bengali accuracy on 8 kHz phone audio
  - Handles stereo speaker separation natively via prompt

Also available: gemini-2.5-flash (20 RPD free, slightly higher quality)

Multiple API keys multiply free-tier quota automatically — see config.example.env.
Get free keys at: https://aistudio.google.com
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Optional

from .key_pool import AllKeysExhaustedError, GeminiKeyPool, _is_rate_limit_error

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
    """Wraps Google Gemini for audio transcription with key-pool rotation."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_id: str = "gemini-3.1-flash-lite",
        key_pool: Optional[GeminiKeyPool] = None,
    ):
        self.model_id = model_id
        try:
            from google import genai  # noqa: F401 — validate install early
        except ImportError:
            raise SystemExit(
                "google-genai package not installed. Run: pip install '.[gemini]'"
            )
        self._pool = key_pool or GeminiKeyPool.from_env(explicit_key=api_key)
        log.info(
            "Gemini transcription backend ready (model=%s, keys=%d)",
            self.model_id, self._pool.total_count,
        )

    def transcribe(
        self,
        audio_path,
        language: Optional[str] = None,
        speaker_labels: tuple = ("Speaker A", "Speaker B"),
        diarize: bool = True,  # accepted for interface parity; diarization is always on via prompt
    ) -> dict:
        """Upload audio to Gemini and transcribe.

        Gemini handles stereo natively — no manual channel splitting needed.
        Files over ~19 MB are sent via the File API. On 429 RESOURCE_EXHAUSTED
        the next key in the pool is tried automatically.
        """
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        size_mb = path.stat().st_size / 1_048_576
        log.info("Transcribing via Gemini %s: %s (%.1f MB)", self.model_id, path.name, size_mb)
        t = time.time()

        from google.genai import types

        mime = _mime_type(path.suffix)
        lang_hint = f"\nThe language spoken is primarily {language} (ISO code). " if language else ""
        prompt = _PROMPT + lang_hint

        while True:
            client, key_idx = self._pool.get_client()
            uploaded_name = None
            try:
                if size_mb > 19:
                    log.info("  File > 19 MB — uploading via Gemini File API...")
                    uploaded = client.files.upload(
                        file=str(path), config={"mime_type": mime}
                    )
                    uploaded_name = uploaded.name
                    parts = [
                        types.Part.from_uri(file_uri=uploaded.uri, mime_type=mime),
                        types.Part.from_text(text=prompt),
                    ]
                else:
                    parts = [
                        types.Part.from_bytes(data=path.read_bytes(), mime_type=mime),
                        types.Part.from_text(text=prompt),
                    ]

                response = client.models.generate_content(
                    model=self.model_id,
                    contents=parts,
                )
                break  # success

            except Exception as e:
                if uploaded_name:
                    try:
                        client.files.delete(name=uploaded_name)
                    except Exception:
                        pass
                    uploaded_name = None
                if _is_rate_limit_error(e):
                    self._pool.mark_exhausted(key_idx)
                    # AllKeysExhaustedError raised by get_client() on next iteration
                    continue
                raise
            finally:
                if uploaded_name:
                    try:
                        client.files.delete(name=uploaded_name)
                    except Exception:
                        pass

        raw_text = response.text or ""
        segments, full_text = _parse_transcript(raw_text, speaker_labels)
        elapsed = time.time() - t

        log.info("  Done: %d segments, took %.1fs", len(segments), elapsed)
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
