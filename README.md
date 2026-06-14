# call-intelligence-pipeline

Turn **call recordings** into transcripts, quality scores, and agent coaching insights —
in **any language**, from a single command.

Two backends are available:

- **Google Gemini** (`--engine gemini`, **default**) — sends audio to Google's API.
  Free tier gives 1,500 requests/day via Google AI Studio. Produces **excellent Bengali
  transcriptions** from the same 8 kHz call recordings that trip up Whisper.
- **Whisper** (`--engine whisper`) — runs entirely on your own hardware, completely
  free, no API key. Best for privacy-sensitive deployments or offline environments.
  Works well for English and many languages; struggles with Bengali on phone audio.

> Works with recordings from **any** source — IP-PBX/SIP systems (FreeSWITCH,
> Asterisk, FusionPBX, 3CX), softphones, mobile call recorders, Zoom/Meet
> exports, voicemail, podcasts, interviews. It's just audio in, transcript out.

---

## Features

- **Two backends** — local Whisper (free, private, offline) or Google Gemini
  (free API tier, excellent Bengali accuracy).
- **Any language** — auto-detected by default, or force a code (`--language bn`,
  `en`, `hi`, `ar`, `es`, …). Whisper supports ~99 languages; Gemini 100+.
- **Any format** — `wav, mp3, ogg, opus, flac, m4a, aac, gsm, amr` (anything
  ffmpeg can read).
- **Speaker separation for stereo calls** — many phone systems record the two
  parties on separate left/right channels. Each channel is transcribed
  independently, with configurable labels (`--labels "Agent,Customer"`).
- **Robust on bad audio** — band-pass + loudness normalization (EBU R128), plus
  Whisper decoding tuned to prevent the classic *repetition-loop* failure mode.
- **Batch or single file** — process one recording or recurse a whole folder,
  optionally limited to the last N days. Already-processed files are skipped.
- **Three output formats per recording** — `.json` (structured, with timestamps
  and confidence), `.txt` (human-readable), and `.srt` (subtitles).
- **CPU or GPU** — Whisper runs on a small VPS (large-v3 int8 ≈ 3 GB RAM) or
  accelerates on CUDA automatically.
- **Installable CLI** — `transcribe`, `transcribe-analyze`, and `transcribe-check`, plus a Python API.

---

## Requirements

- **Python 3.9+**
- **ffmpeg** (system package — *not* installed by pip)
- For Whisper: ~3 GB RAM for `large-v3` int8 (less for smaller models)
- For Gemini: a free API key from [aistudio.google.com](https://aistudio.google.com)

---

## Installation

### Option A — one command (fresh Debian/Ubuntu host)

```bash
git clone https://github.com/md-riaz/call-intelligence-pipeline.git
cd audio-transcription-pipeline
bash setup.sh
```

`setup.sh` installs ffmpeg + Python, creates `.venv`, installs the package,
recommends a Whisper model based on available RAM, and offers to download it.

### Option B — manual (macOS / Windows / any Linux)

```bash
# 1. Install ffmpeg
#    macOS:   brew install ffmpeg
#    Windows: choco install ffmpeg     (or scoop install ffmpeg)
#    Linux:   sudo apt install ffmpeg

# 2. Clone and install
git clone https://github.com/md-riaz/call-intelligence-pipeline.git
cd audio-transcription-pipeline
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install .

# 3. Optional: Gemini backend
pip install ".[gemini]"
export GOOGLE_API_KEY=your_key_here   # free key at https://aistudio.google.com
```

The Whisper model downloads automatically on first run (cached in
`~/.cache/whisper_models`).

---

## Quick start

```bash
# 1. Transcribe — Bengali call recording (Gemini is the default engine)
transcribe --file call.wav --language bn --labels "Agent,Customer"

# 2. Analyze — score the transcript for quality, sentiment, and flags
transcribe-analyze --file transcripts/call.json

# Batch: transcribe then analyze an entire folder
transcribe --input /recordings --language bn --labels "Agent,Customer" --days 7
transcribe-analyze --input transcripts/    # writes analysis_summary.csv

# Use local Whisper instead (no API key, audio stays on your machine)
transcribe --file call.wav --engine whisper --language bn
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
| `--engine` | `gemini` or `whisper` | `gemini` |
| `--language, -l` | Force an ISO code, or `auto` to detect | `auto` |
| `--labels` | Comma-separated labels for stereo channels | `Speaker A,Speaker B` |
| `--days, -d` | With `--input`: only files modified in last N days | all |
| `--model, -m` | Whisper model size (`tiny`/`base`/`small`/`medium`/`large-v3`) | `large-v3` |
| `--device` | `auto`/`cpu`/`cuda` (Whisper only) | `auto` |
| `--compute-type` | ctranslate2 compute type (Whisper only) | `int8` on CPU |
| `--google-api-key` | Google AI Studio API key (overrides `GOOGLE_API_KEY` env var) | — |
| `--no-separate-speakers` | Mix stereo to mono instead of splitting channels | off |
| `--no-srt` | Skip writing `.srt` subtitles | off |
| `--reprocess` | Re-transcribe files even if already done | off |

(`--file` and `--input` are mutually exclusive; one is required.)

### `transcribe-analyze` — call quality analysis

Reads transcript `.json` files and scores them with Gemini. Appends an
`analysis` block to each JSON in-place and writes `analysis_summary.csv`
for management review.

```bash
# Analyze one transcript
transcribe-analyze --file transcripts/call.json

# Analyze a whole folder, regenerate summary CSV
transcribe-analyze --input transcripts/

# Re-score everything (e.g. after updating the rubric)
transcribe-analyze --input transcripts/ --reanalyze
```

| Option | Description |
|---|---|
| `--file, -f` | Single transcript `.json` to analyze |
| `--input, -i` | Directory of transcript `.json` files |
| `--google-api-key` | Overrides `GOOGLE_API_KEY` env var |
| `--reanalyze` | Re-analyze files that already have a score |
| `--no-csv` | Skip writing `analysis_summary.csv` |

Each analyzed file gets an `analysis` block added to its JSON:

```json
"analysis": {
  "brand": "pbx.bd",
  "issue_category": "billing",
  "issue_summary": "Customer requested a BDT 4000 invoice for manual recharge.",
  "resolution": "partial",
  "customer_sentiment": "frustrated",
  "sentiment_score": 2,
  "agent_score": 75,
  "agent_flags": ["slow response on urgent request"],
  "strengths": ["polite throughout", "confirmed phone number back to customer"],
  "coaching_tip": "Acknowledge recurring issues and provide a timeline or escalation path."
}
```

The `analysis_summary.csv` has one row per call — import into Excel or any
BI tool for agent scorecards and trend analysis.

---

### `transcribe-check` — self-test (Whisper only)

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

# Gemini (default — free cloud API, excellent Bengali accuracy)
pipe = TranscriptionPipeline(
    engine="gemini",
    google_api_key="your_key",   # or set GOOGLE_API_KEY env var
    language="bn",
    speaker_labels=("Agent", "Customer"),
    output_dir="./transcripts",
)

# Whisper (local, offline, audio stays on your machine)
pipe = TranscriptionPipeline(
    engine="whisper",
    model_size="large-v3",
    language="bn",
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

### Whisper path (default)

```
audio file ──> ffmpeg preprocess ──> faster-whisper ──> JSON / TXT / SRT
              (16 kHz mono WAV,       (large-v3, anti-
               band-pass 200–3400 Hz,  repetition decoding)
               EBU R128 loudnorm)

stereo? ──> split L/R channels ──> transcribe each ──> merge timeline
            (per-channel loudnorm)                     (speaker-labelled)
```

1. **Preprocess** to 16 kHz mono WAV with telephony band-pass and loudness
   normalization so quiet calls reach a consistent level.
2. **Stereo recordings** split into left/right channels, transcribed separately,
   merged into a chronological speaker-labelled transcript.
3. **Transcribe** with faster-whisper using decoding parameters tuned to prevent
   hallucination on noisy audio.

### Gemini path (`--engine gemini`)

```
audio file ──> Google Gemini API ──> structured prompt ──> JSON / TXT / SRT
              (original file sent;    (speaker labels +
               stereo handled         [MM:SS] timestamps)
               natively)
```

The original audio file is sent directly to Gemini. No local preprocessing is
needed — Gemini handles stereo diarization via structured prompting. Timestamps
are approximate (`[MM:SS]` granularity rather than word-level milliseconds).

---

## Choosing a backend

| | `gemini` (**default**) | `whisper` |
|---|---|---|
| Cost | Free (1,500 req/day) | Free |
| Privacy | Sent to Google | Audio stays local |
| Bengali accuracy | Excellent | Poor on phone audio |
| Other languages | Good (100+ languages) | Good (99 languages) |
| Timestamps | Approximate (MM:SS) | Word-level (ms) |
| Offline | No | Yes |
| GPU acceleration | N/A | Yes (`--device cuda`) |
| API key required | Yes (free at [aistudio.google.com](https://aistudio.google.com)) | No |

**Switch to Whisper** (`--engine whisper`) only when audio must not leave your
servers, you're running offline, or you need millisecond-precise word timestamps.

---

## Accuracy & tuning (Whisper)

**Use `large-v3`.** It is dramatically more accurate and more noise-robust for
non-English languages than `medium`/`small`, and in int8 it only needs ~3 GB
RAM. This is the single biggest factor in Whisper transcript quality.

### The repetition-loop trap

On low-quality phone audio, default Whisper settings frequently fall into a
**repetition loop** — emitting one token thousands of times
(e.g. `বববববব…` or `yeahyeahyeah…`) *with high reported confidence*. Naïve
quality checks that only look at `avg_logprob` will call this "excellent".

This pipeline prevents it by default (`transcribe/engine.py`):

- `condition_on_previous_text=False` — stops the model feeding its own
  repetitions back as context (the main cause of lock-in).
- `no_repeat_ngram_size=3`, `repetition_penalty=1.1` — block token/n-gram loops.
- `temperature=[0.0, 0.2, … 1.0]` — retry a bad chunk at higher temperature
  instead of committing to garbage.
- `compression_ratio_threshold=2.4`, `log_prob_threshold=-1.0`,
  `no_speech_threshold=0.6` — reject looped/silent/low-probability output.
- VAD `threshold=0.5` — trims non-speech without treating line noise as speech.

`transcribe-check` reports a `Repetition loops` count that should be **0**.

### Getting the best Whisper results

- **Force the language** (`--language bn`) — avoids mis-detection on short or
  noisy clips.
- **Keep speaker separation on** for stereo recordings — per-channel
  transcription is much cleaner than a mixed-down mono track.
- **Source quality matters most.** 8 kHz / low-bitrate recordings cap achievable
  accuracy regardless of settings. If you control the recorder, capture at a
  higher bitrate.

---

## Google Gemini backend

### Setup

```bash
pip install ".[gemini]"
export GOOGLE_API_KEY=your_key_here   # free key at https://aistudio.google.com
```

### Usage

```bash
# Single file — Bengali call recording
transcribe --file call.wav --engine gemini --language bn --labels "Agent,Customer"

# Batch — last 7 days of recordings
transcribe --input /recordings --engine gemini \
    --language bn --labels "Agent,Customer" --days 7
```

### Notes

- The model used is `gemini-2.5-flash` by default.
- Audio is uploaded to Google's servers — do not use for recordings subject to
  strict data-residency requirements.
- Free tier: 1,500 requests/day, 15 requests/minute. A typical 2–3 minute call
  costs ~150 audio tokens, well within the daily limit for most call centres.
- Timestamps in `.srt` output are the model's best-effort `[MM:SS]` estimates,
  not millisecond-accurate word timing.

---

## GPU usage (Whisper)

If you have an NVIDIA GPU with CUDA and cuDNN available, `--device auto` (the
default) will use it automatically. To force it:

```bash
transcribe --file call.wav --device cuda --compute-type float16
```

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| Whisper output is one character/word repeated | Repetition loop — ensure you're on `large-v3`. Run `transcribe-check`. |
| `0 segments` / empty Whisper transcript | Audio filtered as silence. Confirm the file has speech; preprocessing resamples to 16 kHz automatically. |
| Gemini `429 RESOURCE_EXHAUSTED` | Free-tier daily limit reached, or the model version has no free quota. Try again the next day, or check [ai.dev/rate-limit](https://ai.dev/rate-limit). |
| Wrong language detected | Force it: `--language <code>`. |
| `ffmpeg: command not found` | Install ffmpeg via your OS package manager. |
| Stereo not separated (Whisper) | The file may be mono, or you passed `--no-separate-speakers`. |
| Slow on CPU (Whisper) | Expected (~0.5–1× realtime for large-v3). Use a smaller model or a GPU. |
| Low confidence but readable (Whisper) | Usually genuine poor source audio; large-v3 is near the ceiling for it. |

---

## Project structure

```
call-intelligence-pipeline/
├── transcribe/
│   ├── engine.py           # faster-whisper wrapper + anti-repetition decoding
│   ├── audio.py            # ffmpeg preprocessing & stereo channel splitting
│   ├── pipeline.py         # orchestration + JSON/TXT/SRT output
│   ├── cli.py              # `transcribe` CLI
│   ├── accuracy.py         # `transcribe-check` self-test (Whisper only)
│   ├── gemini_engine.py    # Google Gemini Flash transcription backend
│   ├── analyze.py          # call quality analysis (brand, resolution, score, flags)
│   ├── analyze_cli.py      # `transcribe-analyze` CLI
│   └── config.py           # optional config.env loader
├── tests/                  # fast smoke tests (no model download required)
├── samples/                # put your own audio here (git-ignored)
├── setup.sh                # one-command installer (Debian/Ubuntu)
├── pyproject.toml
├── requirements.txt
├── CONTRIBUTING.md
└── LICENSE                 # MIT
```

---

## Privacy

Call recordings and transcripts can contain personal data.

- Audio files and the `transcripts/` directory are **git-ignored** — never commit
  real recordings to a public repository.
- With `--engine whisper` (default), audio never leaves your machine.
- With `--engine gemini`, audio is sent to Google's servers for processing.

---

## License

[MIT](LICENSE) — free to use, modify, and distribute. Whisper models and
faster-whisper carry their own licenses. Use of the Gemini API is subject to
[Google's Terms of Service](https://ai.google.dev/terms).

---

## Acknowledgements

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [Google Gemini](https://ai.google.dev/)
- [ffmpeg](https://ffmpeg.org/)
