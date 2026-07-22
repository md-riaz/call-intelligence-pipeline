"""FastAPI service exposing the Bengali whisper-bn transcription queue."""

from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path
from typing import Optional

from .audio import AudioPreprocessor
from .pipeline import TranscriptionPipeline
from .queue import TranscriptionQueue

try:
    from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import JSONResponse
except ImportError as exc:  # pragma: no cover
    raise SystemExit("FastAPI dependencies are missing. Install with: pip install '.[api]'") from exc

APP_NAME = "whisper-bn"
UPLOAD_DIR = Path(os.getenv("ASR_UPLOAD_DIR", "./transcripts/uploads"))
OUTPUT_DIR = Path(os.getenv("ASR_OUTPUT_DIR", "./transcripts"))
DB_PATH = os.getenv("ASR_QUEUE_DB", str(OUTPUT_DIR / "transcription_queue.sqlite3"))

app = FastAPI(
    title="whisper-bn transcription API",
    version="1.0.0",
    description=(
        "Public Bengali speech-to-text API backed by the SAM15K whisper-bn engine. "
        "Upload one audio file, receive a queued transcription job, then poll the job "
        "endpoint until it completes. Only the `whisper-bn` engine is exposed."
    ),
)
queue = TranscriptionQueue(DB_PATH)


@app.get(
    "/health",
    summary="Health",
    description="Returns service readiness and the public engine name.",
    responses={
        200: {
            "description": "The API process is running and ready to accept transcription jobs.",
            "content": {
                "application/json": {
                    "example": {"status": "ok", "engine": "whisper-bn"}
                }
            },
        }
    },
)
def health() -> dict:
    return {"status": "ok", "engine": APP_NAME}


@app.post(
    "/v1/transcriptions",
    summary="Create Transcription",
    description=(
        "Upload an audio file and create an asynchronous Bengali transcription job. "
        "The response returns a `job_id` immediately. Poll `GET /v1/transcriptions/{job_id}` "
        "until `status` becomes `completed` or `failed`."
    ),
    responses={
        202: {
            "description": "The audio was accepted and queued for transcription.",
            "content": {
                "application/json": {
                    "example": {
                        "job_id": "01J4W4Q0R7M7MZ3M8P0N9B4K2T",
                        "status": "queued",
                        "engine": "whisper-bn",
                    }
                }
            },
        },
        400: {
            "description": "Unsupported uploaded audio format.",
            "content": {
                "application/json": {
                    "example": {"detail": "Unsupported audio format: .txt"}
                }
            },
        },
    },
)
async def create_transcription(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(
        ...,
        description=(
            "Audio file to transcribe. Supported formats are those accepted by the "
            "service audio preprocessor, such as wav, mp3, m4a, flac, ogg, and webm."
        ),
    ),
    language: Optional[str] = Form(
        "bn",
        description=(
            "Language hint for ASR. Use `bn` for Bengali, which is the recommended "
            "default for this service. Use `auto` or leave empty to let the backend "
            "auto-detect when supported."
        ),
        examples=["bn", "auto"],
    ),
    labels: str = Form(
        "Agent,Customer",
        description=(
            "Comma-separated speaker labels used in the transcript output. Provide two "
            "labels in call order, for example `Agent,Customer`, `Rep,Caller`, or "
            "`Doctor,Patient`."
        ),
        examples=["Agent,Customer", "Rep,Caller", "Doctor,Patient"],
    ),
) -> JSONResponse:
    suffix = Path(file.filename or "audio").suffix.lower()
    if suffix and suffix not in AudioPreprocessor.SUPPORTED:
        raise HTTPException(status_code=400, detail=f"Unsupported audio format: {suffix}")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file.filename or "audio.wav").name
    path = UPLOAD_DIR / f"{os.urandom(8).hex()}_{safe_name}"
    with path.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)

    job = queue.create_job(
        audio_path=str(path),
        output_dir=str(OUTPUT_DIR),
        language=None if language in (None, "", "auto") else language,
        labels=labels,
    )
    background_tasks.add_task(_drain_queue)
    return JSONResponse(status_code=202, content={"job_id": job.id, "status": job.status, "engine": APP_NAME})


@app.get(
    "/v1/transcriptions/{job_id}",
    summary="Get Transcription",
    description=(
        "Return the queued job state. When complete, `result_path` points to the JSON "
        "transcript written by the service."
    ),
    responses={
        200: {
            "description": "Current transcription job state.",
            "content": {
                "application/json": {
                    "example": {
                        "id": "01J4W4Q0R7M7MZ3M8P0N9B4K2T",
                        "status": "completed",
                        "engine": "whisper-bn",
                        "audio_path": "./transcripts/uploads/example.wav",
                        "output_dir": "./transcripts",
                        "language": "bn",
                        "labels": "Agent,Customer",
                        "result_path": "./transcripts/example.json",
                        "error": None,
                        "created_at": "2026-07-22T11:10:00Z",
                        "updated_at": "2026-07-22T11:11:30Z",
                    }
                }
            },
        },
        404: {
            "description": "No transcription job exists for the supplied job_id.",
            "content": {"application/json": {"example": {"detail": "Job not found"}}},
        },
    },
)
def get_transcription(job_id: str) -> dict:
    job = queue.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return TranscriptionQueue.job_to_dict(job)


@app.get(
    "/v1/transcriptions",
    summary="List Transcriptions",
    description="List recent transcription jobs, newest first, for integration dashboards or polling tools.",
    responses={
        200: {
            "description": "Recent transcription jobs.",
            "content": {
                "application/json": {
                    "example": {
                        "jobs": [
                            {
                                "id": "01J4W4Q0R7M7MZ3M8P0N9B4K2T",
                                "status": "completed",
                                "engine": "whisper-bn",
                                "language": "bn",
                                "labels": "Agent,Customer",
                                "result_path": "./transcripts/example.json",
                                "error": None,
                            }
                        ]
                    }
                }
            },
        }
    },
)
def list_transcriptions(limit: int = 100) -> dict:
    return {"jobs": [TranscriptionQueue.job_to_dict(job) for job in queue.list_jobs(limit=limit)]}


async def _drain_queue() -> None:
    while True:
        job = queue.claim_next()
        if not job:
            return
        try:
            labels = tuple((job.labels.split(",", 1) + ["Customer"])[:2])
            pipeline = TranscriptionPipeline(
                output_dir=job.output_dir,
                language=job.language,
                speaker_labels=labels,
                engine="whisper-bn",
                whisper_model_id=os.getenv("WHISPER_MODEL"),
            )
            transcript = await asyncio.to_thread(pipeline.process_file, job.audio_path, False)
            result_path = str(Path(job.output_dir) / f"{transcript.call_id}.json")
            if transcript.status != "success":
                queue.fail_job(job.id, transcript.error or "Transcription failed")
            else:
                queue.complete_job(job.id, result_path)
        except Exception as exc:  # noqa: BLE001
            queue.fail_job(job.id, str(exc))
