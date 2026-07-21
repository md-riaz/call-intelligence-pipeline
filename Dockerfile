# syntax=docker/dockerfile:1
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=3433

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE requirements.txt ./
COPY transcribe ./transcribe
COPY demo/requirements.txt ./demo/requirements.txt

RUN python -m pip install --upgrade pip \
    && pip install ".[gemini]" \
    && pip install -r requirements.txt \
    && pip install -r demo/requirements.txt

COPY demo ./demo
COPY config.example.env ./config.example.env

RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && mkdir -p /app/samples /app/transcripts \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 3433

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c 'import os, urllib.request; urllib.request.urlopen("http://127.0.0.1:%s/" % os.environ.get("PORT", "3433"), timeout=3).read(1)' || exit 1

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-3433} --workers ${WEB_CONCURRENCY:-2} --worker-class gthread --threads ${WEB_THREADS:-4} --timeout ${WEB_TIMEOUT:-300} demo.app:app"]
