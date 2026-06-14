# audio-transcription-pipeline

Turn **audio call recordings** (or any speech audio) into accurate,
speaker-labelled transcripts — in **any language**, on **CPU or GPU**, from a
single command.

Built on [faster-whisper](https://github.com/SYSTRAN/faster-whisper). Designed
and tuned for the messy reality of real call-center audio: 8 kHz phone lines,
low bitrates, quiet levels, and background noise — the conditions where naïve
Whisper setups produce repetition-loop garbage.

> Works with recordings from **any** source — IP-PBX/SIP systems (FreeSWITCH,
> Asterisk, FusionPBX, 3CX), softphones, mobile call recorders, Zoom/Meet
> exports, voicemail, podcasts, interviews. It's just audio in, transcript out.

---

## Features

- **Any language** — auto-detected by default, or force a language code
  (`--language bn`, `en`, `hi`, `ar`, `es`, …). Whisper supports ~99 languages.
- **Any format** — `wav, mp3, ogg, opus, flac, m4a, aac, gsm, amr` (anything
  ffmpeg can read).
- **Speaker separation for stereo calls** — many phone systems record the two
  parties on separate left/right channels. Each channel is transcribed
  independently for clean per-speaker text, with configurable labels
  (`--labels "Agent,Customer"`).
- **Robust on bad audio** — band-pass + loudness normalization, plus
  decoding tuned to prevent the classic Whisper *repetition-loop* failure
  (see [Accuracy & tuning](#accuracy--tuning)).
- **Batch or single file** — process one recording or recurse a whole folder,
  optionally limited to the last N days. Already-processed files are skipped.
- **Three output formats per recording** — `.json` (structured, with
  timestamps and confidence), `.txt` (human-readable), and `.srt` (subtitles).
- **CPU or GPU** — runs on a small VPS (large-v3 int8 ≈ 3 GB RAM) or accelerates
  on CUDA automatically.
- **Installable CLI** — `transcribe` and `transcribe-check`, plus a Python API.

---

## Requirements

- **Python 3.9+**
- **ffmpeg** (system package — *not* installed by pip)
- ~3 GB RAM for the `large-v3` model in int8 (less for smaller models)
- Optional: an NVIDIA GPU + CUDA for faster runs

---

## Installation

### Option A — one command (fresh Debian/Ubuntu host)

```bash
git clone https://github.com/your-org/audio-transcription-pipeline.git
cd audio-transcription-pipeline
bash setup.sh
```

`setup.sh` installs ffmpeg + Python, creates `.venv`, installs the package,
recommends a model based on available RAM, and offers to download it.

### Option B — manual (macOS / Windows / any Linux)

```bash
# 1. Install ffmpeg
#    macOS:   brew install ffmpeg
#    Windows: choco install ffmpeg     (or scoop install ffmpeg)
#    Linux:   sudo apt install ffmpeg

# 2. Create an environment and install the package
git clone https://github.com/your-org/audio-transcription-pipeline.git
cd audio-transcription-pipeline
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install .
```

The Whisper model downloads automatically on first run (cached in
`~/.cache/whisper_models`).

---

## Quick start

```bash
# Sanity-check on one recording (prints transcript + a quality verdict)
transcribe-check --file samples/your-call.wav

# Transcribe one file, auto-detecting the language
transcribe --file samples/your-call.wav --output ./transcripts

# Force Bengali and label the two stereo channels as Agent / Customer
transcribe --file call.wav --language bn --labels "Agent,Customer"

# Batch a whole folder (recursive), only the last 7 days
transcribe --input /path/to/recordings --output ./transcripts --days 7
```

Each recording produces `transcripts/<name>.json`, `.txt`, and `.srt`.

---

## Usage

### `transcribe` — main CLI

| Option | Description | Default |
|---|---|---|
| `--file, -f` | Transcribe a single audio file | — |
| `--input, -i` | Transcribe a directory (recursive) | — |
| `--output, -o` | Output directory | `./transcripts` |
| `--language, -l` | Force an ISO code, or `auto` to detect | `auto` |
| `--model, -m` | `tiny`/`base`/`small`/`medium`/`large-v2`/`large-v3` | `large-v3` |
| `--device` | `auto`/`cpu`/`cuda` | `auto` |
| `--compute-type` | ctranslate2 compute type | `int8` (CPU), `float16` (GPU) |
| `--days, -d` | With `--input`: only files modified in last N days | all |
| `--labels` | Comma-separated labels for stereo channels | `Speaker A,Speaker B` |
| `--no-separate-speakers` | Mix stereo to mono instead of splitting channels | off |
| `--no-srt` | Skip writing `.srt` subtitles | off |
| `--reprocess` | Re-transcribe files even if already done | off |

(`--file` and `--input` are mutually exclusive; one is required.)

### `transcribe-check` — self-test

Runs one recording end-to-end and prints a verdict. Use it after install or
when results look off:

```bash
transcribe-check --file call.wav --model large-v3 --language bn
```

It explicitly flags **repetition loops** (`R`) rather than reporting them as
high-confidence success.

### Python API

```python
from transcribe import TranscriptionPipeline

pipe = TranscriptionPipeline(
    model_size="large-v3",
    language="bn",                       # or None to auto-detect
    speaker_labels=("Agent", "Customer"),
    output_dir="./transcripts",
)
result = pipe.process_file("call.wav")
print(result.full_text)
```

---

## Output

For a recording `call123.wav` you get:

- **`call123.json`** — structured: per-segment `start`/`end`, `text`,
  `speaker`, `avg_logprob`, `no_speech_prob`, `compression_ratio`, plus
  detected language, duration, model, and word count.
- **`call123.txt`** — a readable transcript with speaker turns and a timed
  segment list.
- **`call123.srt`** — subtitles (speaker-prefixed), usable in any video player.

A `processed_files.json` index lets re-runs skip work already done (override
with `--reprocess`).

---

## How it works

```
audio file ──> ffmpeg preprocess ──> Whisper (faster-whisper) ──> JSON / TXT / SRT
              (16 kHz mono, band-pass,        (anti-repetition
               loudness normalize)             decoding)

stereo? ──> split L/R channels ──> transcribe each ──> merge into one
            (per-channel normalize)                    speaker-labelled timeline
```

1. **Preprocess** every input to 16 kHz mono WAV, apply a telephony band-pass
   (200–3400 Hz), and **loudness-normalize** (EBU R128) so quiet calls are
   lifted to a consistent level.
2. **Stereo recordings** are split into left/right channels and transcribed
   separately, then merged into a single chronological, speaker-labelled
   transcript. (Disable with `--no-separate-speakers`.)
3. **Transcribe** with faster-whisper using decoding parameters tuned against
   hallucination on noisy audio.

---

## Accuracy & tuning

**Use `large-v3`.** It is dramatically more accurate and more noise-robust for
non-English languages than `medium`/`small`, and in int8 it only needs ~3 GB
RAM. This is the single biggest factor in transcript quality.

### The repetition-loop trap

On low-quality phone audio, default Whisper settings frequently fall into a
**repetition loop** — emitting one token thousands of times
(e.g. `বববববব…` or `yeahyeahyeah…`) *with high reported confidence*. Naïve
quality checks that only look at `avg_logprob` will call this "excellent".

This pipeline prevents it by default (`transcribe/engine.py`):

- `condition_on_previous_text=False` — stops the model feeding its own
  repetitions back in as context (the main cause of the lock-in).
- `no_repeat_ngram_size=3`, `repetition_penalty=1.1` — block token/n-gram loops.
- `temperature=[0.0, 0.2, … 1.0]` — retry a bad chunk at higher temperature
  instead of committing to garbage.
- `compression_ratio_threshold=2.4`, `log_prob_threshold=-1.0`,
  `no_speech_threshold=0.6` — reject looped/silent/low-probability output.
- VAD `threshold=0.5` — trims non-speech without treating line noise as speech.

`transcribe-check` reports a `Repetition loops` count that should be **0**.

### Getting the best results

- **Force the language** if you know it (`--language bn`) — avoids
  mis-detection on short or noisy clips.
- **Keep speaker separation on** for stereo call recordings — per-channel
  transcription is much cleaner than a mixed-down mono track.
- **Source quality matters most.** 8 kHz / low-bitrate / very quiet recordings
  cap achievable accuracy no matter the settings. If you control the recorder,
  capture at a higher bitrate and healthy gain.

---

## Google Gemini backend (free, recommended for Bengali)

The default Whisper backend struggles with Bengali phone audio. Google Gemini Flash is
**free** (up to 1,500 calls/day via Google AI Studio) and produces excellent Bengali
transcriptions because Google's models have extensive South Asian language training data.

**Cost:** Free tier — 1,500 requests/day, 15 requests/minute.
**Privacy:** Audio is sent to Google's servers.
**Accuracy:** Excellent for Bengali.

### Setup

```bash
pip install ".[gemini]"
export GOOGLE_API_KEY=your_key_here   # free key at https://aistudio.google.com
```

### Usage

```bash
# Single file
transcribe --file call.wav --engine gemini --language bn

# Batch
transcribe --input /path/to/recordings --engine gemini \
    --language bn --labels "Agent,Customer"
```

Or via the Python API:

```python
from transcribe import TranscriptionPipeline

pipe = TranscriptionPipeline(
    engine="gemini",
    google_api_key="your_key",   # or set GOOGLE_API_KEY
    language="bn",
    speaker_labels=("Agent", "Customer"),
    output_dir="./transcripts",
)
result = pipe.process_file("call.wav")
print(result.full_text)
```

Gemini handles stereo calls natively via structured prompting — no manual channel
splitting needed. Timestamps in the output are approximate (Gemini does not return
word-level timing), so SRT files use the model's best-effort `[MM:SS]` estimates.

### Choosing a backend

| | `whisper` (default) | `gemini` |
|---|---|---|
| Cost | Free | Free (1,500 req/day) |
| Privacy | Audio stays local | Sent to Google |
| Bengali accuracy | Poor on phone audio | Excellent |
| Other languages | Good (99 languages) | Good (100+ languages) |
| Timestamps | Word-level | Approximate (MM:SS) |
| Offline | Yes | No |
| GPU acceleration | Yes (`--device cuda`) | N/A |

**Recommendation:** Use `--engine gemini` for Bengali — it's free and significantly better
than Whisper on phone audio.

---

## GPU usage

If you have an NVIDIA GPU with CUDA and cuDNN available, `--device auto` (the
default) will use it. To force it and pick a compute type:

```bash
transcribe --file call.wav --device cuda --compute-type float16
```

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| Output is one character/word repeated | Repetition loop — ensure you're on `large-v3` and haven't weakened the decoding params. Run `transcribe-check`. |
| `0 segments` / empty transcript | Audio filtered as silence. Confirm the file actually contains speech; preprocessing already resamples to 16 kHz. |
| Wrong language detected | Force it: `--language <code>`. |
| `ffmpeg: command not found` | Install ffmpeg via your OS package manager. |
| Stereo not separated | The file may be mono, or you passed `--no-separate-speakers`. |
| Slow on CPU | Expected (~0.5–1× realtime for large-v3). Use a smaller model or a GPU. |
| Low confidence but readable | Usually genuine poor source audio; large-v3 is near the ceiling for it. |

---

## Project structure

```
audio-transcription-pipeline/
├── transcribe/
│   ├── engine.py           # faster-whisper wrapper + anti-repetition decoding
│   ├── audio.py            # ffmpeg preprocessing & stereo channel splitting
│   ├── pipeline.py         # orchestration + JSON/TXT/SRT output
│   ├── cli.py              # `transcribe` CLI
│   ├── accuracy.py         # `transcribe-check` self-test
│   ├── gemini_engine.py    # Google Gemini Flash backend
│   └── config.py           # optional config.env loader
├── tests/             # fast smoke tests (no model download)
├── samples/           # put your own audio here (git-ignored)
├── setup.sh           # one-command installer (Debian/Ubuntu)
├── pyproject.toml
├── requirements.txt
├── CONTRIBUTING.md
└── LICENSE            # MIT
```

---

## Privacy

Call recordings and transcripts can contain personal data. Audio and the
`transcripts/` directory are **git-ignored** — never commit real recordings to a
public repository.

---

## License

[MIT](LICENSE) — free to use, modify, and distribute. Whisper models and
faster-whisper carry their own licenses.

---

## Acknowledgements

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [ffmpeg](https://ffmpeg.org/)
