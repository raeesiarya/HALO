#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LMLM_DIR="${REPO_ROOT}/../LMLM"

export PATH="${HOME}/.local/bin:${PATH}"

log() {
  printf '[setup_repository.sh] %s\n' "$*"
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    printf '[setup_repository.sh] Missing required command: %s\n' "$1" >&2
    exit 1
  fi
}

if [[ ! -d "${LMLM_DIR}" ]]; then
  log "LMLM checkout not found — cloning from GitHub"
  require_command git
  git clone https://github.com/kilian-group/LMLM.git "${LMLM_DIR}"
else
  log "Found sibling LMLM checkout at ${LMLM_DIR}"
fi

log "Bootstrapping uv and Python dependencies"
"${SCRIPT_DIR}/uv.sh"

log "Bootstrapping Claude Code and Ruflo"
"${SCRIPT_DIR}/claude.sh"

cat <<'EOF'

Repository setup complete.

Useful next commands:
  source .venv/bin/activate
  .venv/bin/pytest
  claude   # authenticate on first run

EOF

log "Done"
