"""
ElevenLabs Scribe transcription backend.

Scribe v2 dramatically outperforms Whisper on Bengali phone audio because it
was trained specifically for real-world speech conditions. Use this backend
when accuracy matters more than cost or privacy.

Cost: ~$0.22/hour of audio (~$0.009 per 2.5-minute call).
Privacy: audio is sent to ElevenLabs' servers.

Requires:
    pip install elevenlabs
    export ELEVENLABS_API_KEY=your_key_here
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# Pause between sentences of the same speaker before starting a new segment.
_SEGMENT_GAP_S = 1.5


def _words_to_segments(words: list, speaker_labels: tuple) -> tuple[list, str]:
    """Group word-level ElevenLabs output into sentence segments.

    ElevenLabs returns one dict per word with keys: text, start, end,
    speaker_id, type ('word' | 'spacing' | 'audio_event').
    We group consecutive words from the same speaker into segments,
    breaking on pauses > _SEGMENT_GAP_S or speaker changes.
    """
    label_map: dict[str, str] = {}
    segments: list[dict] = []
    cur: dict | None = None

    for w in words:
        if w.get("type") != "word":
            continue
        raw_spk = w.get("speaker_id") or "speaker_0"
        # Map speaker_0/speaker_1 → the user's labels (Agent, Customer, …)
        if raw_spk not in label_map:
            idx = len(label_map)
            label_map[raw_spk] = (
                speaker_labels[idx] if idx < len(speaker_labels) else raw_spk
            )
        speaker = label_map[raw_spk]
        start = w.get("start", 0.0)
        end = w.get("end", start)

        gap = (start - cur["end"]) if cur else 0
        if cur is None or cur["speaker"] != speaker or gap > _SEGMENT_GAP_S:
            if cur:
                segments.append(cur)
            cur = {"start": start, "end": end, "speaker": speaker,
                   "text": w["text"], "avg_logprob": 0.0,
                   "no_speech_prob": 0.0, "compression_ratio": 1.0}
        else:
            cur["text"] += " " + w["text"]
            cur["end"] = end

    if cur:
        segments.append(cur)

    full_parts: list[str] = []
    prev_spk = None
    for seg in segments:
        spk = seg.get("speaker", "")
        if spk and spk != prev_spk:
            full_parts.append(f"[{spk}]: {seg['text']}")
            prev_spk = spk
        elif spk:
            full_parts.append(seg["text"])
        else:
            full_parts.append(seg["text"])

    return segments, "\n".join(full_parts)


class ElevenLabsTranscriber:
    """Wraps the ElevenLabs Scribe v2 Speech-to-Text API."""

    def __init__(self, api_key: Optional[str] = None, model_id: str = "scribe_v2"):
        self.api_key = api_key or os.environ.get("ELEVENLABS_API_KEY", "")
        self.model_id = model_id
        if not self.api_key:
            raise ValueError(
                "ElevenLabs API key not set. "
                "Pass api_key= or set the ELEVENLABS_API_KEY environment variable."
            )
        try:
            from elevenlabs import ElevenLabs as _EL
            self._client = _EL(api_key=self.api_key)
        except ImportError:
            raise SystemExit(
                "elevenlabs package not installed. Run: pip install elevenlabs"
            )
        log.info("ElevenLabs Scribe backend ready (model=%s)", self.model_id)

    def transcribe(
        self,
        audio_path,
        language: Optional[str] = None,
        speaker_labels: tuple = ("Speaker A", "Speaker B"),
        diarize: bool = True,
    ) -> dict:
        """Transcribe via Scribe. Sends the original file (stereo or mono).

        For stereo recordings this is better than our manual L/R split because
        Scribe's multi-channel diarization uses the channel information directly
        and avoids the double-transcription cost.
        """
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        log.info("Transcribing via ElevenLabs Scribe: %s", path.name)
        t = time.time()

        with open(path, "rb") as fh:
            result = self._client.speech_to_text.convert(
                file=fh,
                model_id=self.model_id,
                # ISO-639-3 code or None for auto-detect.
                # Scribe auto-detects reliably; forcing helps on very short clips.
                language_code=language,
                # use_multi_channel assigns speakers by L/R channel, which is
                # mutually exclusive with ML-based diarization per the API spec.
                use_multi_channel=True,
                diarize=False,
                detect_speaker_roles=False,
                tag_audio_events=False,
            )

        words = [
            {"text": w.text, "start": w.start, "end": w.end,
             "speaker_id": getattr(w, "speaker_id", None),
             "type": getattr(w, "type", "word")}
            for w in (result.words or [])
        ]
        segments, full_text = _words_to_segments(words, speaker_labels)
        duration = getattr(result, "audio_duration_secs", 0.0) or 0.0
        lang_detected = getattr(result, "language_code", language or "")
        lang_prob = getattr(result, "language_probability", 1.0) or 1.0

        elapsed = time.time() - t
        log.info(
            "  Done: %.0fs audio, %d segments, lang=%s (%.0f%%), took %.1fs",
            duration, len(segments), lang_detected, lang_prob * 100, elapsed,
        )
        return {
            "segments": segments,
            "full_text": full_text,
            "language_detected": lang_detected,
            "language_confidence": round(lang_prob, 4),
            "duration_seconds": round(duration, 2),
            "processing_time_seconds": round(elapsed, 2),
            # ElevenLabs handles speaker separation natively; skip our split.
            "speakers_separated": diarize,
        }
