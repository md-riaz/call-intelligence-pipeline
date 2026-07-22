# syntax=docker/dockerfile:1
# CUDA 12.8 runtime keeps the image compatible with newer NVIDIA GPUs such as
# RTX 50-series for the local whisper-bn ASR service.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=3433 \
    HF_HOME=/models \
    TRANSFORMERS_CACHE=/models \
    MODEL_PROVIDER=whisper-bn \
    OPENAI_BASE_URL=https://api.openai.com/v1 \
    OPENAI_MODEL=gpt-4o-mini \
    WHISPER_MODEL=bitwisemind/sam_15000_clean_text_full_model \
    CUDA_VISIBLE_DEVICES=0 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-venv \
        python3-pip \
        python-is-python3 \
        ffmpeg \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv \
    && /opt/venv/bin/python -m pip install --upgrade pip setuptools wheel

COPY pyproject.toml README.md LICENSE requirements.txt ./
COPY transcribe ./transcribe
COPY demo/requirements.txt ./demo/requirements.txt

RUN pip install --index-url https://download.pytorch.org/whl/cu128 torch \
    && pip install ".[sam15000,api]" \
    && pip install -r requirements.txt \
    && pip install -r demo/requirements.txt

COPY demo ./demo
COPY config.example.env ./config.example.env

RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && mkdir -p /app/samples /app/transcripts /models \
    && chown -R appuser:appuser /app /models /opt/venv

USER appuser

EXPOSE 3433

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c 'import os, urllib.request; urllib.request.urlopen("http://127.0.0.1:%s/" % os.environ.get("PORT", "3433"), timeout=3).read(1)' || exit 1

CMD ["sh", "-c", "uvicorn transcribe.api:app --host 0.0.0.0 --port ${PORT:-3433}"]
