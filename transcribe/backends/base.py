"""Common ASR backend interface and provider factory."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Iterable, Optional, TypedDict


class ASRResult(TypedDict):
    """Standard transcript payload consumed by TranscriptionPipeline."""

    segments: list[dict]
    full_text: str
    language_detected: str
    language_confidence: float
    duration_seconds: float
    processing_time_seconds: float
    speakers_separated: bool


class ASRBackend(ABC):
    """Base class for transcription providers.

    Backends must return the same schema regardless of implementation so the
    call analyzer can continue consuming transcript JSON without changes.
    """

    name: str
    model_id: str

    @abstractmethod
    def transcribe(
        self,
        audio_file: str,
        language: Optional[str] = None,
        speaker_labels: tuple[str, str] = ("Speaker A", "Speaker B"),
    ) -> ASRResult:
        """Transcribe one audio file into the standard transcript schema."""

    def transcribe_many(
        self,
        audio_files: Iterable[str],
        language: Optional[str] = None,
        speaker_labels: tuple[str, str] = ("Speaker A", "Speaker B"),
    ) -> list[ASRResult]:
        """Batch API. Providers may override for true model-level batching."""
        return [
            self.transcribe(path, language=language, speaker_labels=speaker_labels)
            for path in audio_files
        ]


def create_backend(
    engine: Optional[str] = None,
    *,
    google_api_key: Optional[str] = None,
    gemini_model_id: str = "gemini-3.1-flash-lite",
    whisper_model_id: Optional[str] = None,
    temp_dir: Optional[str] = None,
) -> ASRBackend:
    """Instantiate an ASR backend by CLI/config engine name."""
    selected = (engine or os.getenv("MODEL_PROVIDER") or "whisper-bn").strip().lower()
    if selected == "gemini":
        from .gemini import GeminiBackend

        return GeminiBackend(api_key=google_api_key, model_id=gemini_model_id)

    if selected in {"whisper-bn", "whisper_bn", "whisper-sam15000", "sam15000", "whisper_sam15000"}:
        from .whisper_sam15000 import WhisperSam15000Backend

        return WhisperSam15000Backend(
            model_id=whisper_model_id,
            temp_dir=temp_dir,
            provider_name="whisper-bn" if selected in {"whisper-bn", "whisper_bn"} else "whisper-sam15000",
        )

    if selected in {"whisper-tugstugi", "tugstugi", "bengaliai-whisper-medium"}:
        from .whisper_sam15000 import TUGSTUGI_MODEL, WhisperSam15000Backend

        return WhisperSam15000Backend(
            model_id=whisper_model_id or TUGSTUGI_MODEL,
            temp_dir=temp_dir,
            provider_name="whisper-tugstugi",
        )

    raise ValueError(
        f"Unsupported ASR engine: {engine!r}. "
        "Supported engines: whisper-bn"
    )
