# whisper-bn

A simplified Bengali ASR service for call recordings. The public API exposes one transcription engine, `whisper-bn`, backed by the SAM15K Bengali Whisper model, with durable SQLite job tracking. Call QA analysis is separate and works with any OpenAI-compatible Chat Completions endpoint.

## Public FastAPI service

Start the API:

```bash
pip install ".[sam15000,api]"
uvicorn transcribe.api:app --host 0.0.0.0 --port 3433
```

Docker starts the same FastAPI app by default:

```bash
docker compose up -d --build call-intelligence-pipeline
```

Endpoints:

- `GET /health` returns service health and engine name.
- `POST /v1/transcriptions` accepts multipart audio upload and returns `202 {"job_id": ..., "status": "queued", "engine": "whisper-bn"}`.
- `GET /v1/transcriptions/{job_id}` returns queued, processing, completed, or failed job state. Completed jobs include the full transcript JSON inline as `transcript`, not just an artifact path.
- `GET /v1/transcriptions?limit=100` lists recent jobs. Completed jobs include inline `transcript` when the JSON artifact is still available.
- `GET /docs` opens the interactive OpenAPI docs.

Example:

```bash
curl -F "file=@call.wav" -F "language=bn" -F "labels=Agent,Customer"   http://localhost:3433/v1/transcriptions

curl http://localhost:3433/v1/transcriptions/JOB_ID
```

Completed job responses include both compatibility paths and the inline transcript payload:

```json
{
  "id": "JOB_ID",
  "status": "completed",
  "engine": "whisper-bn",
  "result_json_path": "./transcripts/call.json",
  "result_path": "./transcripts/call.json",
  "transcript": {
    "call_id": "call",
    "segments": [
      {"start": 0.0, "end": 2.4, "speaker": "Agent", "text": "হ্যালো"}
    ],
    "full_text": "[Agent]: হ্যালো",
    "status": "success"
  }
}
```

Queue state is stored in SQLite at `ASR_QUEUE_DB`, default `./transcripts/transcription_queue.sqlite3`. Uploaded audio is stored under `ASR_UPLOAD_DIR`, default `./transcripts/uploads`. Transcript JSON/TXT/SRT outputs are written to `ASR_OUTPUT_DIR`, default `./transcripts`.

## Local bulk CLI

The CLI remains available for local/batch transcription. It now defaults to `whisper-bn`:

```bash
transcribe --file call.wav --language bn --labels "Agent,Customer" --output transcripts
transcribe --input samples --language bn --labels "Agent,Customer" --output transcripts
```

The public FastAPI service always invokes `whisper-bn`.

## OpenAI-compatible call QA analysis

`transcribe-analyze` does not require Gemini. It calls any OpenAI-compatible `/v1/chat/completions` endpoint.

```bash
export OPENAI_API_KEY=sk-...
export OPENAI_BASE_URL=https://api.openai.com/v1
export OPENAI_MODEL=gpt-4o-mini
transcribe-analyze --file transcripts/call.json

transcribe-analyze --input transcripts --reanalyze --no-csv   --api-key sk-...   --base-url https://your-gateway.example.com/v1   --model your-model-name
```

The analyzer writes an `analysis` block back into each transcript JSON and stores `analysis.model_used` as `openai-compatible/<model>`.

## Requirements

- Python 3.9+
- ffmpeg
- For local ASR: PyTorch, Transformers, librosa, soundfile, and a GPU strongly recommended
- For QA analysis only: an OpenAI-compatible API key, base URL, and model name

## Docker notes

The compose file uses host networking on the current experiment host to avoid Docker bridge DNS issues. It mounts:

- `./samples:/app/samples:ro`
- `./transcripts:/app/transcripts`
- `call-intelligence-models:/models`

Do not prune the `call-intelligence-models` volume if you want to preserve the Hugging Face model cache.

## Output files

For each successful transcription, the pipeline writes:

- `<call_id>.json` structured transcript
- `<call_id>.txt` readable transcript
- `<call_id>.srt` subtitles

Audio samples and transcripts may contain private call data and are git-ignored.
