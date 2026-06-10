"""
audio-transcription-pipeline
============================

A general-purpose pipeline that turns audio call recordings (or any speech
audio) into accurate, speaker-labelled transcripts using faster-whisper.

Works with any language Whisper supports (auto-detected by default), any audio
format ffmpeg can read, on CPU or GPU.
"""

from .engine import WhisperTranscriber
from .audio import AudioPreprocessor, StereoSplitter
from .pipeline import TranscriptionPipeline, CallTranscript

__version__ = "1.0.0"

__all__ = [
    "WhisperTranscriber",
    "AudioPreprocessor",
    "StereoSplitter",
    "TranscriptionPipeline",
    "CallTranscript",
    "__version__",
]
