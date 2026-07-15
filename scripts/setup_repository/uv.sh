#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PATH="${HOME}/.local/bin:${PATH}"

log() {
  printf '[uv.sh] %s\n' "$*"
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    printf '[uv.sh] Missing required command: %s\n' "$1" >&2
    exit 1
  fi
}

install_uv() {
  require_command curl
  log "Installing uv via the official installer"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
}

if ! command -v uv >/dev/null 2>&1; then
  install_uv
else
  log "uv is already installed: $(uv --version)"
fi

log "Creating or updating the project environment with uv"
cd "${REPO_ROOT}"
uv python install 3.10
uv sync --python 3.10

log "Done"
