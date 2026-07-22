#!/bin/bash
# ============================================================
# call-intelligence-pipeline - one-command installer
# For a fresh Ubuntu 20.04/22.04/24.04 or Debian 11/12 host.
#
# Installs ffmpeg + Python, creates a virtualenv, installs this
# package with the whisper-bn backend, and writes a config.env.
#
# Usage (from the repo root):
#   bash setup.sh
#
# On macOS / Windows / other Linux, skip this script and install manually:
#   1. install ffmpeg via your package manager
#   2. python3 -m venv .venv && source .venv/bin/activate
#   3. pip install ".[sam15000]"
#   4. export OPENAI_API_KEY=your_key   (optional for call QA analysis)
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

echo -e "\n${BOLD}=== call-intelligence-pipeline setup ===${NC}\n"

# --- Privileges (only needed for apt) ---------------------------------------
SUDO=""
if [[ "$EUID" -ne 0 ]]; then
    command -v sudo &>/dev/null && SUDO="sudo" || warn "Not root and no sudo - apt steps may fail."
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
    warn "Cannot detect OS - assuming Debian/Ubuntu."
fi

# --- System dependencies -----------------------------------------------------
sec "Installing system dependencies (ffmpeg, python3)"
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
pip install --quiet ".[sam15000]"
python3 -c "import transcribe" || err "transcribe package import failed"
log "Package installed in $REPO_DIR/.venv"

# --- API keys ----------------------------------------------------------------
sec "API keys"
info "whisper-bn local ASR installed. For call QA, set an OpenAI-compatible endpoint."
info "See config.example.env for OPENAI_BASE_URL, OPENAI_MODEL, and OPENAI_API_KEY."
echo ""

read -r -p "OpenAI-compatible API key for QA (leave blank to set later): " OPENAI_API_KEY
read -r -p "Analysis model [gpt-4o-mini]: " OPENAI_MODEL
OPENAI_MODEL="${OPENAI_MODEL:-gpt-4o-mini}"

# --- config.env --------------------------------------------------------------
sec "Writing config.env"
{
    echo "# Written by setup.sh"
    echo "MODEL_PROVIDER=whisper-bn"
    echo "OPENAI_BASE_URL=https://api.openai.com/v1"
    echo "OPENAI_MODEL=${OPENAI_MODEL}"
    [[ -n "$OPENAI_API_KEY" ]] && echo "OPENAI_API_KEY=${OPENAI_API_KEY}"
} > "$REPO_DIR/config.env"
log "Wrote config.env (MODEL_PROVIDER=whisper-bn, OPENAI_MODEL=${OPENAI_MODEL})"
[[ -z "$OPENAI_API_KEY" ]] && warn "No QA key set - add OPENAI_API_KEY to config.env before running transcribe-analyze."

# --- Done --------------------------------------------------------------------
echo -e "\n${BOLD}${GREEN}=== Setup complete ===${NC}\n"
cat <<EOF
Activate the environment in each new shell:
    source $REPO_DIR/.venv/bin/activate

Transcribe a single Bengali call recording:
    transcribe --file /path/to/call.wav --language bn --labels "Agent,Customer"

Batch a whole folder:
    transcribe --input /path/to/recordings --language bn --labels "Agent,Customer"

Analyze transcripts for quality scores:
    transcribe-analyze --input ./transcripts

Tip: run long batches inside tmux so an SSH drop won't kill them:
    tmux new -s transcribe
EOF
