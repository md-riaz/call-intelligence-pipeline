"""
Whisper transcription engine (faster-whisper wrapper).

The decoding parameters here are tuned to be robust on real-world call audio
(8 kHz / low-bitrate / noisy phone recordings). In particular they defend
against the most common Whisper failure mode on such audio: *repetition loops*,
where the model emits the same token thousands of times
(e.g. "ababababab...") and reports high confidence for it.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


def resolve_device(device: str, compute_type: Optional[str]) -> tuple[str, str]:
    """Pick a sensible (device, compute_type) pair.

    device="auto" uses the GPU if ctranslate2 reports CUDA devices, else CPU.
    compute_type defaults to float16 on GPU and int8 on CPU (int8 keeps the
    large-v3 model under ~3 GB RAM, which is why it runs fine on small VPSes).
    """
    if device == "auto":
        try:
            import ctranslate2

            if ctranslate2.get_cuda_device_count() > 0:
                return "cuda", compute_type or "float16"
        except Exception:
            pass
        return "cpu", compute_type or "int8"

    if compute_type:
        return device, compute_type
    return device, ("float16" if device == "cuda" else "int8")


class WhisperTranscriber:
    """Loads a faster-whisper model and transcribes audio robustly."""

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "auto",
        compute_type: Optional[str] = None,
        download_root: Optional[str] = None,
    ):
        self.model_size = model_size
        self.device, self.compute_type = resolve_device(device, compute_type)
        self.download_root = download_root or os.path.expanduser(
            "~/.cache/whisper_models"
        )
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise SystemExit(
                "faster-whisper is not installed. Run: pip install faster-whisper"
            )
        log.info(
            "Loading Whisper %s (%s/%s)...",
            self.model_size,
            self.device,
            self.compute_type,
        )
        log.info("First run downloads the model (~3 GB for large-v3). Please wait...")
        t = time.time()
        self.model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
            download_root=self.download_root,
        )
        log.info("Model loaded in %.1fs", time.time() - t)

    def transcribe(self, audio_path, language: Optional[str] = None) -> dict:
        """Transcribe a 16 kHz mono WAV (or anything ffmpeg/whisper can read).

        language=None lets Whisper auto-detect. Pass an ISO code (e.g. "bn",
        "en", "hi", "ar") to force a language — recommended when you already
        know it, as it avoids mis-detection on short or noisy clips.
        """
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        log.info("Transcribing: %s", Path(audio_path).name)
        t = time.time()
        segments, info = self.model.transcribe(
            str(audio_path),
            language=language,
            task="transcribe",
            beam_size=5,
            best_of=5,
            # --- Anti-hallucination decoding (see module docstring) ---
            # Temperature fallback: if a chunk decodes as a repetition loop or
            # low-confidence garbage, faster-whisper retries at a higher
            # temperature instead of committing to the bad output.
            temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            # CRITICAL on noisy audio: leaving this True makes Whisper feed its
            # own hallucinated repetitions back in as context, locking it into
            # loops. Must be False.
            condition_on_previous_text=False,
            # Block n-gram / token repetition (the other half of the loop fix).
            no_repeat_ngram_size=3,
            repetition_penalty=1.1,
            # Reject hallucinated/looping output: a repetition loop has a very
            # high gzip compression ratio; silence/noise has low log-prob.
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            # Voice-activity detection trims non-speech. threshold=0.5 is a good
            # balance; lower values let line noise through as "speech".
            vad_filter=True,
            vad_parameters=dict(
                threshold=0.5,
                min_silence_duration_ms=500,
                speech_pad_ms=400,
            ),
            word_timestamps=True,
        )

        seg_list = []
        full_parts = []
        for seg in segments:
            seg_list.append(
                {
                    "start": round(seg.start, 2),
                    "end": round(seg.end, 2),
                    "text": seg.text.strip(),
                    "avg_logprob": round(seg.avg_logprob, 4),
                    "no_speech_prob": round(seg.no_speech_prob, 4),
                    "compression_ratio": round(seg.compression_ratio, 4),
                }
            )
            full_parts.append(seg.text.strip())

        return {
            "segments": seg_list,
            "full_text": " ".join(full_parts),
            "language_detected": info.language,
            "language_confidence": round(info.language_probability, 4),
            "duration_seconds": round(info.duration, 2),
            "processing_time_seconds": round(time.time() - t, 2),
        }
