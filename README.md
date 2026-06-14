# call-intelligence-pipeline

Turn **call recordings** into transcripts, quality scores, and agent coaching insights —
in **any language**, from a single command.

Powered by **Google Gemini** (free API, 1,500 calls/day). Handles real-world call
center audio — 8 kHz phone lines, low bitrates, noisy environments — with excellent
accuracy for Bengali and other South Asian languages.

> Works with recordings from **any** source — IP-PBX/SIP systems (FreeSWITCH,
> Asterisk, FusionPBX, 3CX), softphones, mobile call recorders, Zoom/Meet exports,
> voicemail, podcasts, interviews. Audio in, intelligence out.

---

## Features

- **Transcription** — accurate, speaker-labelled transcripts via Google Gemini Flash.
  Handles stereo call recordings natively (Agent / Customer on separate channels).
- **Call quality analysis** — per-call scoring: issue type, resolution status, customer
  sentiment, agent behavior flags, strengths, and a concrete coaching tip.
- **Batch summary CSV** — one row per call, ready to import into Excel or any BI tool
  for agent scorecards and trend analysis.
- **Any language** — auto-detected, or force a code (`--language bn`, `en`, `hi`, `ar`, …).
- **Any format** — `wav, mp3, ogg, opus, flac, m4a, aac, gsm, amr`.
- **Batch or single file** — process one recording or recurse a whole folder, optionally
  limited to the last N days. Already-processed files are skipped.
- **Three output formats** — `.json` (structured), `.txt` (readable), `.srt` (subtitles).
- **Offline fallback** — swap to local Whisper (`--engine whisper`) when audio must not
  leave your servers. Note: Whisper accuracy on Bengali phone audio is poor.
- **Installable CLI** — `transcribe`, `transcribe-analyze`, `transcribe-check`.

---

## Requirements

- **Python 3.9+**
- **ffmpeg** (system package — *not* installed by pip)
- A free Google AI Studio API key — [aistudio.google.com](https://aistudio.google.com)

---

## Installation

### Option A — one command (fresh Debian/Ubuntu host)

```bash
git clone https://github.com/md-riaz/call-intelligence-pipeline.git
cd call-intelligence-pipeline
bash setup.sh
```

### Option B — manual (macOS / Windows / any Linux)

```bash
# 1. Install ffmpeg
#    macOS:   brew install ffmpeg
#    Windows: choco install ffmpeg     (or scoop install ffmpeg)
#    Linux:   sudo apt install ffmpeg

# 2. Clone and install
git clone https://github.com/md-riaz/call-intelligence-pipeline.git
cd call-intelligence-pipeline
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install ".[gemini]"

# 3. Set your API key (free at https://aistudio.google.com)
export GOOGLE_API_KEY=your_key_here
```

---

## Quick start

```bash
# 1. Transcribe a Bengali call recording
transcribe --file call.wav --language bn --labels "Agent,Customer"

# 2. Analyze the transcript — scores, sentiment, flags
transcribe-analyze --file transcripts/call.json

# Batch: transcribe + analyze an entire folder
transcribe --input /recordings --language bn --labels "Agent,Customer" --days 7
transcribe-analyze --input transcripts/    # writes analysis_summary.csv
```

Each recording produces `transcripts/<name>.json`, `.txt`, and `.srt`.

---

## Usage

### `transcribe` — transcription CLI

| Option | Description | Default |
|---|---|---|
| `--file, -f` | Transcribe a single audio file | — |
| `--input, -i` | Transcribe a directory (recursive) | — |
| `--output, -o` | Output directory | `./transcripts` |
| `--language, -l` | Force an ISO code, or `auto` to detect | `auto` |
| `--labels` | Comma-separated speaker labels for stereo channels | `Speaker A,Speaker B` |
| `--days, -d` | With `--input`: only files modified in last N days | all |
| `--google-api-key` | Google AI Studio API key (overrides `GOOGLE_API_KEY`) | — |
| `--no-separate-speakers` | Mix stereo to mono instead of splitting channels | off |
| `--no-srt` | Skip writing `.srt` subtitles | off |
| `--reprocess` | Re-transcribe files even if already done | off |
| `--engine` | `gemini` (default) or `whisper` (offline fallback) | `gemini` |

(`--file` and `--input` are mutually exclusive; one is required.)

---

### `transcribe-analyze` — call quality analysis

Reads transcript `.json` files, scores them with Gemini, and appends an `analysis`
block to each JSON. Also writes `analysis_summary.csv` covering all analyzed calls.

```bash
# Analyze one transcript
transcribe-analyze --file transcripts/call.json

# Analyze a whole folder + regenerate CSV
transcribe-analyze --input transcripts/

# Re-score everything (e.g. after refining the prompt)
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
  "brand": "Alpha PBX",
  "issue_category": "billing",
  "issue_summary": "Customer requested a BDT 4000 invoice for manual recharge.",
  "resolution": "partial",
  "resolution_note": "Invoice promised but balance update not confirmed on the call.",
  "customer_sentiment": "frustrated",
  "sentiment_score": 2,
  "agent_score": 90,
  "agent_flags": ["did not address customer's recurring manual recharge complaint"],
  "strengths": ["polite throughout", "confirmed contact number back to customer"],
  "coaching_tip": "Acknowledge recurring issues and give a timeline or escalation path."
}
```

The `analysis_summary.csv` has one row per call — import into Excel or any BI tool
for weekly agent scorecards and management trend reports.

---

### `transcribe-check` — Whisper self-test

Only relevant when using `--engine whisper`. Runs one file end-to-end and flags
repetition loops (a common Whisper failure on noisy audio):

```bash
transcribe-check --file call.wav --language en
```

---

### Python API

```python
from transcribe import TranscriptionPipeline
from transcribe.analyze import CallAnalyzer

# Transcribe
pipe = TranscriptionPipeline(
    engine="gemini",
    google_api_key="your_key",   # or set GOOGLE_API_KEY env var
    language="bn",
    speaker_labels=("Agent", "Customer"),
    output_dir="./transcripts",
)
result = pipe.process_file("call.wav")

# Analyze
analyzer = CallAnalyzer(api_key="your_key")
analysis = analyzer.analyze_file("transcripts/call.json")
print(analysis.agent_score, analysis.coaching_tip)
```

---

## Output

For a recording `call123.wav` you get:

- **`call123.json`** — structured: per-segment `start`/`end`, `text`, `speaker`,
  confidence fields, detected language, duration, model, word count. After analysis,
  also contains the `analysis` block shown above.
- **`call123.txt`** — readable transcript with speaker turns and timed segment list.
- **`call123.srt`** — subtitles (speaker-prefixed), playable in any video player.
- **`analysis_summary.csv`** — batch analysis summary, one row per call.

---

## How it works

```
call recording ──> Google Gemini API ──> transcript (JSON / TXT / SRT)
                   (original file;                    │
                    stereo handled natively)           ▼
                                           Gemini analysis prompt
                                                       │
                                                       ▼
                                           analysis block appended to JSON
                                           + analysis_summary.csv updated
```

1. The audio file is sent to Gemini with a structured transcription prompt.
   Gemini handles stereo speaker separation natively — no local preprocessing needed.
   Timestamps in the output are approximate (`[MM:SS]` granularity).
2. Each transcript JSON is then passed to a second Gemini call with a quality
   analysis prompt. It returns brand, issue category, FCR, sentiment, agent score,
   flags, strengths, and a coaching tip as structured JSON.
3. The analysis block is written back into the transcript JSON in-place, and
   `analysis_summary.csv` is regenerated from all analyzed files.

---

## Offline / privacy mode (`--engine whisper`)

If audio cannot leave your servers, pass `--engine whisper` to use local
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) instead of Gemini.

```bash
pip install .   # faster-whisper is a core dependency
transcribe --file call.wav --engine whisper --language en
```

**Limitations when using Whisper:**
- Bengali accuracy is poor on low-quality phone audio (8 kHz, noisy lines).
  Use Gemini for Bengali calls.
- Requires ~3 GB RAM for the `large-v3` model (int8).
- `transcribe-analyze` still requires a Gemini API key — analysis is always cloud-based.
- `--device cuda` enables GPU acceleration if CUDA is available.

---

## Choosing a backend

| | `gemini` (**default**) | `whisper` |
|---|---|---|
| Bengali accuracy | Excellent | Poor on phone audio |
| Other languages | Good (100+ languages) | Good (99 languages) |
| Cost | Free (1,500 req/day) | Free |
| Privacy | Audio sent to Google | Audio stays local |
| Timestamps | Approximate (MM:SS) | Word-level (ms) |
| Offline | No | Yes |
| API key | Yes (free) | No |

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Gemini `429 RESOURCE_EXHAUSTED` | Free-tier daily limit hit. Wait until the next day or check [ai.dev/rate-limit](https://ai.dev/rate-limit). |
| Empty or garbled transcript | Confirm the file has speech. Try `--language bn` to force language detection. |
| Wrong speaker labels | Adjust `--labels "Agent,Customer"` to match your recording convention. |
| `ffmpeg: command not found` | Install via your OS package manager (`apt`/`brew`/`choco`). |
| Whisper repetition loop | Use `--engine gemini` instead, or run `transcribe-check` to diagnose. |

---

## Project structure

```
call-intelligence-pipeline/
├── transcribe/
│   ├── gemini_engine.py    # Google Gemini transcription backend (default)
│   ├── analyze.py          # call quality analysis (score, sentiment, flags)
│   ├── analyze_cli.py      # `transcribe-analyze` CLI
│   ├── pipeline.py         # orchestration + JSON/TXT/SRT output
│   ├── cli.py              # `transcribe` CLI
│   ├── audio.py            # ffmpeg preprocessing & stereo channel splitting
│   ├── engine.py           # faster-whisper backend (offline fallback)
│   ├── accuracy.py         # `transcribe-check` (Whisper self-test)
│   └── config.py           # optional config.env loader
├── tests/                  # fast smoke tests (no model download required)
├── samples/                # put your own audio here (git-ignored)
├── setup.sh                # one-command installer (Debian/Ubuntu)
├── pyproject.toml
├── CONTRIBUTING.md
└── LICENSE                 # MIT
```

---

## Privacy

Call recordings can contain personal data.

- Audio files and `transcripts/` are **git-ignored** — never commit real recordings.
- With `--engine gemini` (default), audio is sent to Google's servers.
- With `--engine whisper`, audio never leaves your machine — but Bengali accuracy
  will be significantly lower.

---

## License

[MIT](LICENSE) — free to use, modify, and distribute. Use of the Gemini API is
subject to [Google's Terms of Service](https://ai.google.dev/terms).

---

## Acknowledgements

- [Google Gemini](https://ai.google.dev/)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (offline fallback)
- [ffmpeg](https://ffmpeg.org/)
