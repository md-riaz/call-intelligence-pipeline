# call-intelligence-pipeline

Turn **call recordings** into transcripts, quality scores, and agent coaching insights —
in **any language**, from a command line or a browser.

Powered by **Google Gemini** — free tier 500 requests/day, excellent Bengali accuracy.

> Tested on real Bengali call center audio (8 kHz phone lines, noisy environments).
> `gemini-3.1-flash-lite` is the default — 500 RPD free, excellent Bengali accuracy.

> Works with recordings from **any** source — IP-PBX/SIP systems (FreeSWITCH,
> Asterisk, FusionPBX, 3CX), softphones, mobile call recorders, Zoom/Meet exports,
> voicemail, podcasts, interviews. Audio in, intelligence out.

---

## Features

- **Transcription** — accurate, speaker-labelled transcripts. Gemini handles stereo
  natively via a structured prompt — no manual channel splitting needed.
- **Call quality analysis** — per-call scoring: issue type, resolution status, customer
  sentiment, agent behavior flags, strengths, and a concrete coaching tip.
- **Batch summary CSV** — one row per call, ready for Excel or any BI tool.
- **Any language** — auto-detected, or force a code (`--language bn`, `en`, `hi`, `ar`, …).
- **Any format** — `wav, mp3, ogg, opus, flac, m4a, aac, gsm, amr`.
- **Batch or single file** — process one recording or recurse a whole folder, optionally
  limited to the last N days. Already-processed files are skipped.
- **Three output formats** — `.json` (structured), `.txt` (readable), `.srt` (subtitles).
- **Multi-key rotation** — add multiple API keys to multiply your free-tier quota.
  3 Gemini keys × 500 RPD = 1,500 free calls/day.
- **Web UI** — drag-and-drop demo interface with real-time per-file progress, transcript
  viewer, and analysis dashboard (agent score, sentiment, flags, coaching tips).
- **Installable CLI** — `transcribe`, `transcribe-analyze`.

---

## Requirements

- **Python 3.9+**
- **ffmpeg** (system package — *not* installed by pip)
- A free Gemini API key — [aistudio.google.com](https://aistudio.google.com), no card needed

**Install ffmpeg:**

| OS | Command |
|---|---|
| Ubuntu / Debian | `sudo apt install ffmpeg` |
| macOS | `brew install ffmpeg` |
| Windows | `choco install ffmpeg` or `scoop install ffmpeg` |

---

## Setup

There are three ways to run this project, depending on your use case.

---

### Option A — Automated installer (fresh Debian/Ubuntu VPS)

Installs ffmpeg, Python, a virtualenv, and prompts for your API key. Recommended for
a clean server.

```bash
git clone https://github.com/md-riaz/call-intelligence-pipeline.git
cd call-intelligence-pipeline
bash setup.sh
```

The script writes your key and model preference to `config.env` (git-ignored). After it
finishes, the `transcribe` and `transcribe-analyze` commands are available inside the
virtualenv.

---

### Option B — Manual CLI install (macOS / Windows / any Linux)

```bash
# 1. Install ffmpeg (see table above)

# 2. Clone the repo
git clone https://github.com/md-riaz/call-intelligence-pipeline.git
cd call-intelligence-pipeline

# 3. Create and activate a virtualenv
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 4. Install the package
pip install ".[gemini]"

# 5. Set your API key
#    Option A — persistent (recommended): copy the example and fill it in
cp config.example.env config.env
# then open config.env and set GOOGLE_API_KEY=AIzaSy...
#
#    Option B — one-off shell export
export GOOGLE_API_KEY=your_key_here      # Windows: $env:GOOGLE_API_KEY="..."
```

Verify the install:

```bash
transcribe --version
transcribe-analyze --version
```

---

### Option C — Demo web UI

A browser-based interface: drop audio files, pick models, click Analyze, and watch
per-file results stream in. No command line needed after setup.

#### C1 — Local development (single machine)

```bash
# 1. Follow steps 1–4 of Option B above first (ffmpeg + virtualenv + pip install)

# 2. Install demo server dependencies
pip install -r demo/requirements.txt

# 3. Set your API key (in config.env or as an env var — same as Option B)

# 4. Start the server
cd demo
python app.py
```

Open **http://localhost:3433** in your browser.  
Enter your Gemini API key(s) directly in the UI — no config file needed for the demo.

To use a different port:

```bash
PORT=8080 python app.py
```

#### C2 — Production server (nginx + supervisor)

Use this to host the demo on a VPS so teammates can access it over the network.

**Step 1 — Install the package**

```bash
# As root or a deploy user:
git clone https://github.com/md-riaz/call-intelligence-pipeline.git /opt/call-intelligence
cd /opt/call-intelligence

python3 -m venv .venv
source .venv/bin/activate
pip install ".[gemini]"
pip install -r demo/requirements.txt

# Optional but recommended — gives gunicorn async worker support
pip install gevent
```

**Step 2 — Configure nginx**

```bash
sudo cp demo/nginx.conf /etc/nginx/sites-available/call-intelligence
sudo ln -s /etc/nginx/sites-available/call-intelligence \
           /etc/nginx/sites-enabled/call-intelligence

# Edit the file if you want to change the port (default: 3433) or add a domain
sudo nano /etc/nginx/sites-available/call-intelligence

sudo nginx -t && sudo systemctl reload nginx
```

The config sets `proxy_buffering off` and `X-Accel-Buffering: no` so Server-Sent
Events (live progress) reach the browser immediately. It also raises `client_max_body_size`
to 200 MB and proxy timeouts to 300 s for large audio files.

**Step 3 — Configure supervisor**

```bash
sudo mkdir -p /var/log/call-intelligence

sudo cp demo/supervisor.conf /etc/supervisor/conf.d/call-intelligence.conf

# Edit the file — update REPO_DIR path if you cloned somewhere other than /opt/call-intelligence
sudo nano /etc/supervisor/conf.d/call-intelligence.conf

sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start call-intelligence-demo
sudo supervisorctl status    # should show RUNNING
```

The supervisor config runs gunicorn with `--worker-class gthread --threads 4` — this
handles multiple simultaneous SSE connections without blocking. For higher concurrency,
switch to `--worker-class gevent` (see comments in `demo/supervisor.conf`).

**Step 4 — Open the UI**

Navigate to **http://your-server-ip:3433** from any browser on the network.

**Updating to a new version:**

```bash
cd /opt/call-intelligence
git pull
source .venv/bin/activate
pip install ".[gemini]"
pip install -r demo/requirements.txt
sudo supervisorctl restart call-intelligence-demo
```

---

## Quick start — CLI

```bash
# Transcribe a Bengali call recording
transcribe --file call.wav --language bn --labels "Agent,Customer"

# Use a higher-quality model (20 RPD free instead of 500)
transcribe --file call.wav --model gemini-2.5-flash --language bn

# Analyze the transcript — scores, sentiment, flags
transcribe-analyze --file transcripts/call.json

# Batch: transcribe + analyze an entire folder
transcribe --input /recordings --language bn --labels "Agent,Customer" --days 7
transcribe-analyze --input transcripts/    # writes analysis_summary.csv
```

Each recording produces `transcripts/<name>.json`, `.txt`, and `.srt`.

---

## CLI reference

### `transcribe`

| Option | Description | Default |
|---|---|---|
| `--file, -f` | Transcribe a single audio file | — |
| `--input, -i` | Transcribe a directory (recursive) | — |
| `--output, -o` | Output directory | `./transcripts` |
| `--language, -l` | Force an ISO code, or `auto` to detect | `auto` |
| `--model, -m` | Gemini model ID (see table below) | `gemini-3.1-flash-lite` |
| `--labels` | Comma-separated speaker labels | `Speaker A,Speaker B` |
| `--days, -d` | With `--input`: only files modified in last N days | all |
| `--google-api-key` | Gemini API key (overrides `GOOGLE_API_KEY`) | — |
| `--no-srt` | Skip writing `.srt` subtitles | off |
| `--reprocess` | Re-transcribe files even if already done | off |

(`--file` and `--input` are mutually exclusive; one is required.)

**Model options:**

| Model | RPD (free) | Notes |
|---|---|---|
| `gemini-3.1-flash-lite` | **500** | Default — best free-tier throughput |
| `gemini-2.5-flash` | 20 | Higher quality, lower free quota |

---

### `transcribe-analyze`

Reads transcript `.json` files, scores them with Gemini, and appends an `analysis`
block to each JSON. Also writes `analysis_summary.csv` covering all analyzed calls.

```bash
transcribe-analyze --file transcripts/call.json
transcribe-analyze --input transcripts/
transcribe-analyze --input transcripts/ --reanalyze      # re-score everything
transcribe-analyze --input transcripts/ --model gemini-2.5-flash
```

| Option | Description |
|---|---|
| `--file, -f` | Single transcript `.json` to analyze |
| `--input, -i` | Directory of transcript `.json` files |
| `--model, -m` | Gemini model ID (default: `gemini-3.1-flash-lite`) |
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

### Python API

```python
from transcribe import TranscriptionPipeline
from transcribe.analyze import CallAnalyzer

# Transcribe
pipe = TranscriptionPipeline(
    gemini_model_id="gemini-3.1-flash-lite",
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

## Multi-key rotation

Each Google AI Studio project has its own daily quota. Add multiple keys to multiply
your free-tier limit — rotation is automatic on `429 RESOURCE_EXHAUSTED`.

**In `config.env`:**

```bash
# Option A — comma-separated (3 keys × 500 RPD = 1,500 free calls/day)
GOOGLE_API_KEYS=AIza...key1,AIza...key2,AIza...key3

# Option B — numbered (easier to enable/disable one at a time)
GOOGLE_API_KEY_1=AIza...key1
GOOGLE_API_KEY_2=AIza...key2
GOOGLE_API_KEY_3=AIza...key3
```

**In the web UI:** paste all keys comma-separated into the API Keys field — the server
handles rotation the same way.

**Via CLI:**

```bash
transcribe --file call.wav --google-api-key AIza...key
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
call recording ──> Gemini API ──> transcript (JSON / TXT / SRT)
                   (stereo speaker                   │
                    separation via prompt)            ▼
                                             Gemini analysis prompt
                                                     │
                                                     ▼
                                         analysis block appended to JSON
                                         + analysis_summary.csv updated
```

1. The audio file is sent directly to Gemini. Stereo speaker separation is handled
   natively via the transcription prompt — no preprocessing needed.
2. Timestamps in the output are approximate (`[MM:SS]` granularity).
3. Each transcript JSON is passed to Gemini with a quality analysis prompt. It returns
   brand, issue category, resolution, sentiment, agent score, flags, strengths, and a
   coaching tip as structured JSON.

The web UI streams results back to the browser via Server-Sent Events — each file's
progress updates in real time as transcription and analysis complete.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `429 RESOURCE_EXHAUSTED` | Daily limit hit. Add more API keys via `GOOGLE_API_KEYS` in `config.env` (or comma-separate them in the UI), or wait until the next day. See [rate limits](https://ai.google.dev/gemini-api/docs/rate-limits). |
| Empty or garbled transcript | Confirm the file has speech. Try forcing the language (e.g. `--language bn`). |
| Speaker labels reversed | Gemini infers roles from context. If the agent answers first and is labelled `Customer`, flip the labels: `--labels "Customer,Agent"` (or swap them in the UI). |
| `ffmpeg: command not found` | Install via your OS package manager (`apt`/`brew`/`choco`). |
| Web UI shows no progress (SSE broken) | Check that nginx has `proxy_buffering off` and the `X-Accel-Buffering: no` response header — both are set in the provided `demo/nginx.conf`. |
| Web UI upload rejected | File exceeds nginx `client_max_body_size` (default 200 MB in the provided config). Increase it or compress the recording first. |
| gunicorn times out mid-request | Raise `--timeout` in `supervisor.conf`. Default is 300 s, which covers most files. Very long calls may need more. |
| `ModuleNotFoundError: transcribe` | Run `pip install ".[gemini]"` from the repo root before starting the demo server. |

---

## Project structure

```
call-intelligence-pipeline/
├── transcribe/
│   ├── gemini_engine.py    # Google Gemini transcription backend
│   ├── key_pool.py         # multi-key rotation (auto on 429)
│   ├── analyze.py          # call quality analysis (score, sentiment, flags)
│   ├── analyze_cli.py      # `transcribe-analyze` CLI
│   ├── pipeline.py         # orchestration + JSON/TXT/SRT output
│   ├── cli.py              # `transcribe` CLI
│   ├── audio.py            # ffmpeg preprocessing
│   └── config.py           # config.env loader
├── demo/
│   ├── app.py              # Flask web server (SSE streaming)
│   ├── requirements.txt    # flask, werkzeug, gunicorn
│   ├── static/
│   │   └── index.html      # single-page drag/drop UI
│   ├── nginx.conf          # reverse proxy on port 3433
│   └── supervisor.conf     # process manager config
├── tests/                  # fast smoke tests (no API calls required)
├── samples/                # put your own audio here (git-ignored)
├── setup.sh                # one-command installer (Debian/Ubuntu)
├── config.example.env      # copy to config.env for persistent defaults
├── pyproject.toml
├── CONTRIBUTING.md
└── LICENSE                 # MIT
```

---

## Privacy

Call recordings can contain personal data.

- Audio files and `transcripts/` are **git-ignored** — never commit real recordings.
- Audio is sent to the Google Gemini API for transcription and analysis.
- The demo web server keeps uploaded files only for the duration of processing and
  deletes them immediately afterwards — nothing is stored on disk persistently.
- Review [Google's data usage policies](https://ai.google.dev/gemini-api/terms) before
  processing recordings that contain personal data.

---

## License

[MIT](LICENSE) — free to use, modify, and distribute. Use of the Gemini API is subject
to [Google AI Terms](https://ai.google.dev/terms).

---

## Acknowledgements

- [Google Gemini](https://ai.google.dev/)
- [ffmpeg](https://ffmpeg.org/)
