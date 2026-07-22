# Docker deployment

This repo runs the public `whisper-bn` FastAPI transcription queue in Docker.

## Build and run

```bash
docker compose up -d --build call-intelligence-pipeline
```

Open:

```text
http://SERVER_IP:3433/docs
```

Health check:

```bash
curl http://SERVER_IP:3433/health
```

Submit a transcription job:

```bash
curl -F "file=@samples/call.wav" -F "language=bn" -F "labels=Agent,Customer"   http://SERVER_IP:3433/v1/transcriptions
```

Check a job:

```bash
curl http://SERVER_IP:3433/v1/transcriptions/JOB_ID
```

## Volumes

- `./samples` mounts read-only to `/app/samples` for optional local input audio.
- `./transcripts` mounts read-write to `/app/transcripts` for uploads, SQLite queue state, and outputs.
- `call-intelligence-models` mounts at `/models` for Hugging Face cache.

Do not prune `call-intelligence-models` if you want to preserve model downloads.

## Settings

```bash
PORT=3433
MODEL_PROVIDER=whisper-bn
WHISPER_MODEL=bitwisemind/sam_15000_clean_text_full_model
ASR_QUEUE_DB=/app/transcripts/transcription_queue.sqlite3
ASR_UPLOAD_DIR=/app/transcripts/uploads
ASR_OUTPUT_DIR=/app/transcripts
CUDA_VISIBLE_DEVICES=0
```

OpenAI-compatible QA analysis is a CLI feature. Provide these only when running `transcribe-analyze`:

```bash
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
```

## Host reverse proxy

This compose file uses `network_mode: host` because the current server's Docker bridge DNS cannot resolve package repositories or external API hosts. The app listens on `0.0.0.0:${PORT:-3433}` on the host.

If you later fix Docker daemon DNS and want Traefik Docker-label routing, create a separate override file that removes `network_mode: host`, restores a bridge network and `ports`, then adds Traefik labels. Do not edit production Traefik services directly.
