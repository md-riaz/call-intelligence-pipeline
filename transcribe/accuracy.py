"""Quick accuracy / sanity checker: `transcribe-check`.

Run this on one recording after install to confirm the model, audio conversion,
and decoding all work end-to-end before launching a big batch. It prints the
transcript plus a quality verdict, and explicitly flags repetition loops (the
classic Whisper failure on noisy phone audio) instead of being fooled by them.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from .audio import DEFAULT_FILTER, AudioPreprocessor
from .config import load_config
from .engine import resolve_device

MODELS = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]


def check(audio_path: str, model_size: str, language, device: str) -> int:
    print(f"\n{'=' * 55}")
    print(f"  Transcription self-test | model={model_size}")
    print(f"{'=' * 55}\n")

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("ERROR: faster-whisper not installed. Run: pip install faster-whisper")
        return 1

    duration = AudioPreprocessor.duration(audio_path)
    channels = AudioPreprocessor.channels(audio_path)
    print(f"File     : {Path(audio_path).name}")
    print(f"Duration : {duration:.1f}s ({duration / 60:.1f} min)")
    print(
        f"Channels : {channels} "
        f"({'stereo' if channels == 2 else 'mono'})"
    )

    # Always pre-convert to a clean 16 kHz mono WAV.
    import tempfile

    tmp_wav = tempfile.mktemp(suffix="_16k.wav")
    print("Pre-converting -> 16 kHz mono WAV (band-pass + loudness normalize)")
    conv = AudioPreprocessor.convert(
        audio_path, Path(tmp_wav).parent, audio_filter=DEFAULT_FILTER
    )
    # AudioPreprocessor.convert names the file <stem>_16k.wav; use that path.
    transcribe_path = conv
    print("Conversion done\n")

    dev, ctype = resolve_device(device, None)
    print(f"Loading model {model_size} ({dev}/{ctype})...")
    t0 = time.time()
    model = WhisperModel(
        model_size, device=dev, compute_type=ctype,
        download_root=os.path.expanduser("~/.cache/whisper_models"),
    )
    print(f"Model loaded in {time.time() - t0:.1f}s\n")
    print(f"Transcribing (lang={language or 'auto'}, anti-repetition on)...\n")

    t1 = time.time()
    segments, info = model.transcribe(
        transcribe_path, language=language, beam_size=5, best_of=5,
        temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        condition_on_previous_text=False,
        no_repeat_ngram_size=3, repetition_penalty=1.1,
        compression_ratio_threshold=2.4, log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
        vad_filter=True,
        vad_parameters=dict(threshold=0.5, min_silence_duration_ms=500, speech_pad_ms=400),
        word_timestamps=True,
    )
    all_segs = list(segments)
    elapsed = time.time() - t1

    print(f"{'-' * 55}")
    print(f"Language : {info.language} ({info.language_probability:.1%} confidence)")
    rt = duration / elapsed if elapsed else 0
    print(f"Time     : {elapsed:.1f}s for {duration:.0f}s audio ({rt:.1f}x realtime)")
    print(f"Segments : {len(all_segs)}")
    print(f"{'-' * 55}\nTRANSCRIPT:\n")

    for seg in all_segs:
        m1, s1 = divmod(int(seg.start), 60)
        looped = getattr(seg, "compression_ratio", 0) > 2.4
        if looped:
            flag = "R"   # repetition / hallucination
        elif seg.avg_logprob > -0.3:
            flag = "+"
        elif seg.avg_logprob > -0.5:
            flag = "~"
        else:
            flag = "-"
        print(f"  [{m1:02d}:{s1:02d}] {flag} {seg.text.strip()}")

    avg_conf = sum(s.avg_logprob for s in all_segs) / len(all_segs) if all_segs else 0
    looped_segs = sum(1 for s in all_segs if getattr(s, "compression_ratio", 0) > 2.4)

    print(f"\n{'-' * 55}\nQUALITY:\n")
    print(f"  Avg confidence  : {avg_conf:.3f}  (0=perfect, -1=poor)")
    print(f"  Repetition loops: {looped_segs}/{len(all_segs)}  (hallucinated; should be 0)")

    print(f"\n{'-' * 55}\nVERDICT:")
    if not all_segs:
        print("  X  0 segments. Audio likely filtered as silence — check the recording.")
    elif looped_segs > 0:
        # Checked before avg_conf: a loop has a deceptively GOOD avg_logprob.
        print("  X  Repetition/hallucination detected (same token repeated).")
        print("     Causes: weak model on noisy audio, or too-quiet input.")
        print("     Fix: use --model large-v3 (and keep anti-repetition decoding on).")
    elif avg_conf > -0.3:
        print("  OK  Excellent quality. Pipeline ready for production.")
    elif avg_conf > -0.45:
        print("  OK  Good quality. Minor noise won't affect usefulness.")
    else:
        print("  ~  Low confidence. Likely heavy noise, strong accent, or a")
        print("     very low-quality source (8 kHz / low bitrate / very quiet).")
        if model_size != "large-v3":
            print("     Try: --model large-v3 for best accuracy.")
        else:
            print("     Already on large-v3 — a higher-quality source would help most.")

    print(f"{'=' * 55}\n")

    try:
        os.remove(transcribe_path)
    except OSError:
        pass
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="transcribe-check",
        description="Self-test transcription accuracy on one recording.",
    )
    ap.add_argument("--file", "-f", required=True, help="Path to an audio recording")
    ap.add_argument("--model", "-m", default=None, choices=MODELS)
    ap.add_argument("--language", "-l", default=None,
                    help="Force a language code, or omit for auto-detect")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args(argv)

    if not Path(args.file).exists():
        print(f"ERROR: File not found: {args.file}")
        return 1

    cfg = load_config()
    model = args.model or cfg.get("WHISPER_MODEL", "large-v3")
    language = None if args.language in (None, "auto") else args.language
    return check(args.file, model, language, args.device)


if __name__ == "__main__":
    raise SystemExit(main())
