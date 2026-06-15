"""
Demo web server for call-intelligence-pipeline.

Runs a small Flask app that accepts audio file uploads, streams progress via
Server-Sent Events, transcribes with Gemini, runs call quality analysis, and
returns structured results to the browser.

Quick start (development):
    cd demo
    pip install -r requirements.txt
    pip install "..[gemini]"        # from repo root: pip install ".[gemini]"
    python app.py                   # listens on 0.0.0.0:3433

Production (gunicorn + nginx):
    gunicorn --bind 127.0.0.1:5000 --workers 2 --worker-class gthread \\
             --threads 4 --timeout 300 app:app
    # then configure nginx to proxy port 3433 → 127.0.0.1:5000
    # see nginx.conf and supervisor.conf in this directory
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

from flask import Flask, Response, jsonify, request, stream_with_context
from werkzeug.utils import secure_filename

# Allow running from the demo/ subdirectory without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB upload limit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".ogg", ".opus", ".flac", ".m4a", ".aac", ".amr", ".gsm"}


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/api/process", methods=["POST"])
def process():
    api_keys_str = request.form.get("api_keys", "").strip()
    transcribe_model = request.form.get("transcribe_model", "gemini-3.1-flash-lite")
    analyze_model = request.form.get("analyze_model", "gemini-3.1-flash-lite")
    language = request.form.get("language", "").strip() or None
    labels_str = request.form.get("labels", "Agent,Customer").strip()
    files = request.files.getlist("files")

    if not api_keys_str:
        return jsonify({"error": "At least one Gemini API key is required"}), 400
    if not files or not any(f.filename for f in files):
        return jsonify({"error": "No files uploaded"}), 400

    api_keys = [k.strip() for k in api_keys_str.split(",") if k.strip()]
    speaker_labels = tuple((labels_str.split(",", 1) + ["Speaker B"])[:2])

    # Save all files to a temp dir before streaming begins (can't read form data mid-stream)
    tmpdir = tempfile.mkdtemp(prefix="cip_demo_")
    saved = []
    for f in files:
        if not f.filename:
            continue
        suffix = Path(f.filename).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            continue
        safe_name = secure_filename(f.filename) or f"audio_{len(saved)}{suffix}"
        path = os.path.join(tmpdir, safe_name)
        f.save(path)
        saved.append({"original": f.filename, "path": path})

    if not saved:
        shutil.rmtree(tmpdir, ignore_errors=True)
        return jsonify({"error": "No valid audio files. Allowed: wav mp3 ogg opus flac m4a aac amr"}), 400

    def generate():
        try:
            from dataclasses import asdict

            from transcribe.analyze import CallAnalyzer
            from transcribe.gemini_engine import GeminiTranscriber
            from transcribe.key_pool import GeminiKeyPool

            pool = GeminiKeyPool(api_keys)
            transcriber = GeminiTranscriber(model_id=transcribe_model, key_pool=pool)
            analyzer = CallAnalyzer(model_id=analyze_model, key_pool=pool)
            total = len(saved)

            for i, finfo in enumerate(saved):
                orig, path = finfo["original"], finfo["path"]

                # ── Step 1: Transcribe ────────────────────────────────────────
                yield _sse({"type": "progress", "file": orig, "index": i, "total": total, "step": "transcribing"})
                try:
                    tr = transcriber.transcribe(path, language=language, speaker_labels=speaker_labels)
                except Exception as exc:
                    log.error("Transcription failed for %s: %s", orig, exc)
                    yield _sse({"type": "error", "file": orig, "index": i, "total": total,
                                "message": f"Transcription failed: {exc}"})
                    continue

                # ── Step 2: Analyze ───────────────────────────────────────────
                yield _sse({"type": "progress", "file": orig, "index": i, "total": total, "step": "analyzing"})
                temp_json = Path(tmpdir) / f"{Path(orig).stem}_{i}.json"
                transcript_data = {
                    "call_id": Path(orig).stem,
                    "filename": orig,
                    "full_text": tr["full_text"],
                    "segments": tr["segments"],
                    "duration_seconds": tr["duration_seconds"],
                    "language_detected": tr["language_detected"],
                    "model_used": f"gemini/{transcribe_model}",
                    "word_count": len(tr["full_text"].split()),
                    "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "status": "success",
                }
                with open(temp_json, "w", encoding="utf-8") as f:
                    json.dump(transcript_data, f, ensure_ascii=False, indent=2)

                analysis_dict = None
                try:
                    result = analyzer.analyze_file(temp_json, reanalyze=True)
                    if result:
                        analysis_dict = asdict(result)
                except Exception as exc:
                    log.warning("Analysis failed for %s: %s", orig, exc)

                yield _sse({
                    "type": "result",
                    "file": orig,
                    "index": i,
                    "total": total,
                    "transcript": tr["full_text"],
                    "segments": tr["segments"],
                    "duration": tr["duration_seconds"],
                    "language": tr["language_detected"],
                    "analysis": analysis_dict,
                })

            yield _sse({"type": "done", "total": total})

        except Exception as exc:
            log.exception("Fatal stream error")
            yield _sse({"type": "fatal", "message": str(exc)})
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # tell nginx not to buffer SSE
        },
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3433))
    print(f"\n  Call Intelligence Pipeline demo → http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
