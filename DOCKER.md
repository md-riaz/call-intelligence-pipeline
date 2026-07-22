# Docker deployment

This repo can run as a Dockerized Flask demo app, with the CLI available inside the same image.

## Build

```bash
docker compose build
```

## Run the demo UI

```bash
docker compose up -d
```

Open:

```text
http://SERVER_IP:3433
```

The demo UI lets you paste Gemini API keys directly in the browser. You can also pass keys as environment variables:

```bash
GOOGLE_API_KEYS="key1,key2,key3" docker compose up -d
```

Optional settings:

```bash
# The container uses host networking on this server, so PORT is the host port.
PORT=3433
GEMINI_MODEL=gemini-3.1-flash-lite
WEB_CONCURRENCY=2
WEB_THREADS=4
WEB_TIMEOUT=300
```

## Run CLI commands in Docker

Single file:

```bash
docker compose run --rm call-intelligence-pipeline \
  transcribe --file /app/samples/call.wav --output /app/transcripts --language bn --labels "Agent,Customer"
```

Batch folder:

```bash
docker compose run --rm call-intelligence-pipeline \
  transcribe --input /app/samples --output /app/transcripts --language bn --labels "Agent,Customer"
```

Analyze transcripts:

```bash
docker compose run --rm call-intelligence-pipeline \
  transcribe-analyze --input /app/transcripts
```

## Volumes

- `./samples` mounts read-only to `/app/samples` for input audio.
- `./transcripts` mounts read-write to `/app/transcripts` for outputs.

Audio samples and transcripts may contain private call data and are git-ignored.

## Host reverse proxy

This compose file uses `network_mode: host` because the current server's Docker bridge DNS cannot resolve package repositories or external API hosts. The app listens on `0.0.0.0:${PORT:-3433}` on the host.

If you later fix Docker daemon DNS and want Traefik Docker-label routing, create a separate override file that removes `network_mode: host`, restores a bridge network and `ports`, then adds Traefik labels. Do not edit production Traefik services directly.
