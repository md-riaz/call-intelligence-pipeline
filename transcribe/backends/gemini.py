"""Gemini ASR backend adapter."""

from __future__ import annotations

from typing import Optional

from .base import ASRBackend, ASRResult
from ..gemini_engine import GeminiTranscriber


class GeminiBackend(ASRBackend):
    """Adapter that exposes the existing Gemini transcriber via ASRBackend."""

    name = "gemini"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_id: str = "gemini-3.1-flash-lite",
    ):
        self.transcriber = GeminiTranscriber(api_key=api_key, model_id=model_id)
        self.model_id = self.transcriber.model_id

    def transcribe(
        self,
        audio_file: str,
        language: Optional[str] = None,
        speaker_labels: tuple[str, str] = ("Speaker A", "Speaker B"),
    ) -> ASRResult:
        return self.transcriber.transcribe(
            audio_file, language=language, speaker_labels=speaker_labels
        )
