#!/bin/bash
# ============================================================
# call-intelligence-pipeline — one-command installer
# For a fresh Ubuntu 20.04/22.04/24.04 or Debian 11/12 host.
#
# Installs ffmpeg + Python, creates a virtualenv, installs this
# package with the Gemini backend, and writes a config.env.
#
# Usage (from the repo root):
#   bash setup.sh
#
# On macOS / Windows / other Linux, skip this script and install manually:
#   1. install ffmpeg via your package manager
#   2. python3 -m venv .venv && source .venv/bin/activate
#   3. pip install ".[gemini]"
#   4. export GOOGLE_API_KEY=your_key   (get free key at aistudio.google.com)
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
pip install --quiet ".[gemini]"
python3 -c "from google import genai" || err "google-genai import failed"
log "Package installed in $REPO_DIR/.venv"

# --- API keys ----------------------------------------------------------------
sec "API keys"
info "Get a free Gemini key (no card needed): https://aistudio.google.com"
info "Up to 500 free transcriptions/day per key (gemini-3.1-flash-lite)"
info "Add multiple keys to multiply your daily quota — see config.example.env"
echo ""

read -r -p "Gemini API key (leave blank to set later): " GEMINI_KEY
read -r -p "Gemini model   [gemini-3.1-flash-lite]:    " GEMINI_MODEL
GEMINI_MODEL="${GEMINI_MODEL:-gemini-3.1-flash-lite}"

# --- config.env --------------------------------------------------------------
sec "Writing config.env"
{
    echo "# Written by setup.sh"
    echo "GEMINI_MODEL=${GEMINI_MODEL}"
    [[ -n "$GEMINI_KEY" ]] && echo "GOOGLE_API_KEY=${GEMINI_KEY}"
} > "$REPO_DIR/config.env"
log "Wrote config.env (GEMINI_MODEL=${GEMINI_MODEL})"
[[ -z "$GEMINI_KEY" ]] && warn "No Gemini key set — add GOOGLE_API_KEY to config.env before use."

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
