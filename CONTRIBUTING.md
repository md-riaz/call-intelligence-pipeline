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

To test the Gemini backend locally, also install its dependency and set your key:

```bash
pip install -e ".[dev,gemini]"
export GOOGLE_API_KEY=your_key_here   # free at https://aistudio.google.com
```

## Running checks

```bash
pytest            # fast smoke tests — no model download or API key needed
ruff check .      # lint
```

The smoke tests deliberately avoid downloading a model or calling any API so
they run in seconds in CI. If you add behaviour that needs a model or API call,
guard it behind a marker so the default `pytest` run stays fast.

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
- **Adding a new backend.** Create `transcribe/<name>_engine.py` with a class
  that exposes a `.transcribe(audio_path, language, speaker_labels)` method
  returning a dict with the standard keys (`segments`, `full_text`,
  `language_detected`, `language_confidence`, `duration_seconds`,
  `processing_time_seconds`, `speakers_separated`). Wire it into `pipeline.py`
  and the `--engine` choices in `cli.py`. Add an optional pip extra in
  `pyproject.toml` and document it in `README.md`.
- Match the existing style; keep functions small and comment only where the
  *why* isn't obvious from the code.

## Reporting issues

Please include: OS, Python version, the exact command, the backend (`whisper` or
`gemini`), the model/API used, and a description of the audio (sample rate,
mono/stereo, language, roughly how noisy). If you can, attach a short
**non-sensitive** clip that reproduces the problem.
