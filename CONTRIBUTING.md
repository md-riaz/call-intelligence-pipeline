# Contributing

Thanks for your interest in improving **audio-transcription-pipeline**!

## Development setup

```bash
git clone <your-fork-url>
cd audio-transcription-pipeline
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
# Install ffmpeg via your OS package manager (apt/brew/choco).
```

## Running checks

```bash
pytest            # fast smoke tests — no model download needed
ruff check .      # lint
```

The smoke tests deliberately avoid downloading a model or touching real audio so
they run in seconds in CI. If you add behavior that needs a model, guard it
behind a marker so the default `pytest` run stays fast.

## Guidelines

- **Keep it general-purpose.** This project is not tied to any phone system,
  language, or vendor. Avoid hard-coding paths, languages, or speaker roles —
  expose them as options with sensible defaults.
- **Don't weaken the anti-hallucination decoding.** The parameters in
  `transcribe/engine.py` (`condition_on_previous_text=False`,
  `no_repeat_ngram_size`, `compression_ratio_threshold`, the temperature
  fallback list) prevent repetition-loop garbage on noisy phone audio. If you
  change them, explain why and show before/after output on a noisy sample.
- **Never commit audio or transcripts.** They may contain personal data and are
  git-ignored for that reason.
- Match the existing style; keep functions small and commented where the
  *why* isn't obvious.

## Reporting issues

Please include: OS, Python version, the exact command, the model used, and a
description of the audio (sample rate, mono/stereo, language, roughly how noisy).
If you can, attach a short **non-sensitive** clip that reproduces the problem.
