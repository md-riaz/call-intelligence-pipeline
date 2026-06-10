#!/bin/bash
# ============================================================
# audio-transcription-pipeline — one-command installer
# For a fresh Ubuntu 20.04/22.04/24.04 or Debian 11/12 host.
#
# Installs ffmpeg + Python, creates a virtualenv, installs this package,
# and (optionally) downloads a Whisper model.
#
# Usage (from the repo root):
#   bash setup.sh
#
# On macOS / Windows / other Linux, skip this script and install manually:
#   1. install ffmpeg via your package manager
#   2. python3 -m venv .venv && source .venv/bin/activate
#   3. pip install .
# ============================================================

set -e

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; RED='\033[0;31m'; BOLD='\033[1m'; NC='\033[0m'
log()  { echo -e "${GREEN}[ok]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC}  $1"; }
err()  { echo -e "${RED}[x]${NC}  $1"; exit 1; }
info() { echo -e "${BLUE}[i]${NC}  $1"; }
sec()  { echo -e "\n${BOLD}-- $1 --${NC}"; }

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

echo -e "\n${BOLD}=== audio-transcription-pipeline setup ===${NC}\n"

# --- Privileges (only needed for apt) ---------------------------------------
SUDO=""
if [[ "$EUID" -ne 0 ]]; then
    command -v sudo &>/dev/null && SUDO="sudo" || warn "Not root and no sudo — apt steps may fail."
fi

# --- Detect OS ---------------------------------------------------------------
sec "Detecting OS"
if [[ -f /etc/os-release ]]; then
    . /etc/os-release
    info "OS: $NAME $VERSION_ID"
    if [[ "$ID" != "ubuntu" && "$ID" != "debian" && "$ID_LIKE" != *debian* ]]; then
        warn "This installer targets Debian/Ubuntu. For other systems, see README."
    fi
else
    warn "Cannot detect OS — assuming Debian/Ubuntu."
fi

# --- Resources & model recommendation ---------------------------------------
sec "Checking resources"
TOTAL_RAM_GB=$(( $(awk '/MemTotal/{print $2}' /proc/meminfo) / 1024 / 1024 ))
CPU_CORES=$(nproc 2>/dev/null || echo "?")
info "RAM: ${TOTAL_RAM_GB}GB | CPUs: ${CPU_CORES}"

# large-v3 with int8 only needs ~3GB RAM at runtime and is dramatically more
# accurate AND more robust against repetition-loop hallucinations than medium,
# so prefer it whenever there is enough RAM.
if   [[ "$TOTAL_RAM_GB" -ge 8 ]]; then
    MODEL="large-v3"; info "Recommended model: large-v3 (best accuracy)"
elif [[ "$TOTAL_RAM_GB" -ge 4 ]]; then
    MODEL="medium";   warn "RAM < 8GB — recommending 'medium' (lower accuracy)"
else
    MODEL="small";    warn "RAM < 4GB — recommending 'small' (basic accuracy)"
fi

# --- System dependencies -----------------------------------------------------
sec "Installing system dependencies (ffmpeg, python3, venv)"
$SUDO apt-get update -qq
DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y -qq \
    python3 python3-pip python3-venv python3-dev \
    ffmpeg git curl ca-certificates build-essential >/dev/null
command -v ffmpeg >/dev/null || err "ffmpeg install failed"
log "ffmpeg: $(ffmpeg -version 2>&1 | head -1 | cut -d' ' -f3)"
log "python: $(python3 --version)"

# --- Virtualenv + package ----------------------------------------------------
sec "Creating virtualenv and installing the package"
python3 -m venv "$REPO_DIR/.venv"
# shellcheck disable=SC1091
source "$REPO_DIR/.venv/bin/activate"
pip install --upgrade pip wheel --quiet
pip install --quiet .
python3 -c "import faster_whisper" || err "faster-whisper import failed"
log "Package installed in $REPO_DIR/.venv"

# --- config.env --------------------------------------------------------------
cat > "$REPO_DIR/config.env" <<EOF
# Written by setup.sh — the default model used when --model is not passed.
WHISPER_MODEL=$MODEL
EOF
log "Wrote config.env (WHISPER_MODEL=$MODEL)"

# --- Optional model download -------------------------------------------------
sec "Whisper model: $MODEL"
info "Sizes: large-v3 ~3GB | medium ~1.5GB | small ~500MB. Cached in ~/.cache/whisper_models"
read -r -p "Download '$MODEL' now? (recommended) [Y/n]: " ans
if [[ "$ans" != "n" && "$ans" != "N" ]]; then
    python3 - <<PYEOF
import os
from faster_whisper import WhisperModel
cache = os.path.expanduser("~/.cache/whisper_models")
os.makedirs(cache, exist_ok=True)
print("Downloading $MODEL (do not interrupt)...")
WhisperModel("$MODEL", device="cpu", compute_type="int8", download_root=cache)
print("Model ready.")
PYEOF
    log "Model '$MODEL' downloaded"
else
    warn "Skipped — it will auto-download on first run."
fi

# --- Done --------------------------------------------------------------------
echo -e "\n${BOLD}${GREEN}=== Setup complete ===${NC}\n"
cat <<EOF
Activate the environment in each new shell:
    source $REPO_DIR/.venv/bin/activate

Self-test on one recording:
    transcribe-check --file /path/to/call.wav

Transcribe a single file (auto-detect language):
    transcribe --file /path/to/call.wav --output ./transcripts

Batch a folder, forcing a language and labelling stereo channels:
    transcribe --input /path/to/recordings --output ./transcripts \\
        --language bn --labels "Agent,Customer"

Tip: run long batches inside tmux so an SSH drop won't kill them:
    tmux new -s transcribe
EOF
