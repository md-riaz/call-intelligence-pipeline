"""Local HuggingFace Whisper SAM15K Bengali ASR backend."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Iterable, Optional

from .base import ASRBackend, ASRResult
from ..audio import AudioPreprocessor, StereoSplitter

log = logging.getLogger(__name__)

_DEFAULT_MODEL = "bitwisemind/sam_15000_clean_text_full_model"
_LANGUAGE_NAMES = {
    "bn": "bengali",
    "ben": "bengali",
    "bangla": "bengali",
    "bengali": "bengali",
}


class WhisperSam15000Backend(ASRBackend):
    """GPU-backed Bengali Whisper provider using HuggingFace Transformers.

    The model/processor/pipeline are cached at the class level so repeated
    transcriptions in the same worker do not reload weights. Stereo FusionPBX
    recordings are split by channel instead of diarized: channel 0 maps to the
    first speaker label and channel 1 maps to the second speaker label.
    """

    name = "whisper-sam15000"
    _pipeline = None
    _processor = None
    _model_id_loaded: Optional[str] = None

    def __init__(self, model_id: Optional[str] = None, temp_dir: Optional[str] = None):
        self.model_id = model_id or os.getenv("WHISPER_MODEL") or _DEFAULT_MODEL
        self.temp_dir = Path(temp_dir or os.getenv("ASR_TEMP_DIR") or "./transcripts/_temp")
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_loaded()

    @classmethod
    def _ensure_dependencies(cls) -> None:
        try:
            import torch  # noqa: F401
            from transformers import (  # noqa: F401
                WhisperForConditionalGeneration,
                WhisperProcessor,
                pipeline,
            )
        except ImportError as exc:
            raise SystemExit(
                "Whisper SAM15K dependencies are not installed. "
                "Install with: pip install '.[sam15000]' or use the Docker image."
            ) from exc

    def _ensure_loaded(self) -> None:
        if self.__class__._pipeline is not None and self.__class__._model_id_loaded == self.model_id:
            return

        self._ensure_dependencies()
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required for whisper-sam15000 but torch.cuda.is_available() is false"
            )

        log.info("Loading local ASR model %s on CUDA with float16", self.model_id)
        t0 = time.time()
        processor = WhisperProcessor.from_pretrained(self.model_id)
        model = WhisperForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).to("cuda")
        model.eval()

        asr = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch.float16,
            device="cuda:0",
        )
        self.__class__._processor = processor
        self.__class__._pipeline = asr
        self.__class__._model_id_loaded = self.model_id
        log.info("Loaded %s in %.1fs", self.model_id, time.time() - t0)

    @property
    def _asr(self):
        self._ensure_loaded()
        return self.__class__._pipeline

    @property
    def _loaded_processor(self):
        self._ensure_loaded()
        return self.__class__._processor

    def transcribe(
        self,
        audio_file: str,
        language: Optional[str] = None,
        speaker_labels: tuple[str, str] = ("Speaker A", "Speaker B"),
    ) -> ASRResult:
        path = Path(audio_file)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")

        t0 = time.time()
        if StereoSplitter.is_stereo(str(path)):
            left, right = StereoSplitter.split(str(path), self.temp_dir)
            left_result, right_result = self._transcribe_batch(
                [left, right], language=language, speaker_labels=speaker_labels
            )
            segments = StereoSplitter.merge(
                left_result["segments"], right_result["segments"], labels=speaker_labels
            )
            full_text = _segments_to_text(segments)
            duration = max(
                left_result.get("duration_seconds", 0.0),
                right_result.get("duration_seconds", 0.0),
                AudioPreprocessor.duration(str(path)),
            )
            return {
                "segments": segments,
                "full_text": full_text,
                "language_detected": language or left_result.get("language_detected") or "bn",
                "language_confidence": 1.0,
                "duration_seconds": round(duration, 2),
                "processing_time_seconds": round(time.time() - t0, 2),
                "speakers_separated": True,
            }

        mono = AudioPreprocessor.convert(str(path), self.temp_dir)
        result = self._transcribe_one(mono, language=language)
        label = speaker_labels[0] if speaker_labels else "Speaker A"
        for seg in result["segments"]:
            seg["speaker"] = label
        result["full_text"] = _segments_to_text(result["segments"])
        result["speakers_separated"] = False
        result["processing_time_seconds"] = round(time.time() - t0, 2)
        return result

    def transcribe_many(
        self,
        audio_files: Iterable[str],
        language: Optional[str] = None,
        speaker_labels: tuple[str, str] = ("Speaker A", "Speaker B"),
    ) -> list[ASRResult]:
        return [
            self.transcribe(path, language=language, speaker_labels=speaker_labels)
            for path in audio_files
        ]

    def _transcribe_batch(
        self,
        audio_files: list[str],
        language: Optional[str],
        speaker_labels: tuple[str, str],
    ) -> list[ASRResult]:
        # HuggingFace ASR pipeline accepts a list and reuses the loaded model.
        generate_kwargs = self._generate_kwargs(language)
        outputs = self._asr(
            audio_files,
            return_timestamps=True,
            batch_size=int(os.getenv("ASR_BATCH_SIZE", "2")),
            generate_kwargs=generate_kwargs,
        )
        if isinstance(outputs, dict):
            outputs = [outputs]
        return [
            self._normalize_pipeline_output(out, path, language=language)
            for out, path in zip(outputs, audio_files)
        ]

    def _transcribe_one(self, audio_file: str, language: Optional[str]) -> ASRResult:
        output = self._asr(
            audio_file,
            return_timestamps=True,
            generate_kwargs=self._generate_kwargs(language),
        )
        return self._normalize_pipeline_output(output, audio_file, language=language)

    def _generate_kwargs(self, language: Optional[str]) -> dict:
        if not language or language == "auto":
            return {"task": "transcribe"}

        lang = _LANGUAGE_NAMES.get(language.lower(), language.lower())
        processor = self._loaded_processor
        forced_ids = processor.get_decoder_prompt_ids(language=lang, task="transcribe")
        return {"forced_decoder_ids": forced_ids, "task": "transcribe"}

    @staticmethod
    def _normalize_pipeline_output(output: dict, audio_file: str, language: Optional[str]) -> ASRResult:
        chunks = output.get("chunks") or []
        segments = []
        for chunk in chunks:
            start, end = _timestamp_pair(chunk.get("timestamp"))
            text = (chunk.get("text") or "").strip()
            if not text:
                continue
            segments.append(
                {
                    "start": round(start, 2),
                    "end": round(end, 2),
                    "speaker": "",
                    "text": text,
                    "avg_logprob": 0.0,
                    "no_speech_prob": 0.0,
                    "compression_ratio": 1.0,
                }
            )

        if not segments and (output.get("text") or "").strip():
            duration = AudioPreprocessor.duration(audio_file)
            segments = [
                {
                    "start": 0.0,
                    "end": round(duration, 2),
                    "speaker": "",
                    "text": output["text"].strip(),
                    "avg_logprob": 0.0,
                    "no_speech_prob": 0.0,
                    "compression_ratio": 1.0,
                }
            ]

        duration = max(AudioPreprocessor.duration(audio_file), segments[-1]["end"] if segments else 0.0)
        return {
            "segments": segments,
            "full_text": " ".join(s["text"] for s in segments),
            "language_detected": language or "bn",
            "language_confidence": 1.0 if language else 0.0,
            "duration_seconds": round(duration, 2),
            "processing_time_seconds": 0.0,
            "speakers_separated": False,
        }


def _timestamp_pair(value) -> tuple[float, float]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        start = 0.0 if value[0] is None else float(value[0])
        end = start if value[1] is None else float(value[1])
        return start, end
    return 0.0, 0.0


def _segments_to_text(segments: list[dict]) -> str:
    lines = []
    for seg in segments:
        speaker = seg.get("speaker") or "Speaker"
        text = (seg.get("text") or "").strip()
        if text:
            lines.append(f"[{speaker}]: {text}")
    return "\n".join(lines)
