#!/usr/bin/env bash
# Pull audit results from the GPU instance down to this machine with rsync.
# Run this FROM YOUR LOCAL COMPUTER (the one that can ssh into the instance),
# not from the instance itself.
#
# Reads settings from a .env file (default: <repo>/.env, override with
# ENV_FILE=/path/to/env). Required: PULL_HOST (e.g. ubuntu@1.2.3.4).
# Optional: PULL_PORT (default 22), PULL_KEY (ssh private key path),
# PULL_REMOTE_DIR (default HALO/outputs/trex, relative to the remote home),
# PULL_LOCAL_DIR (default <repo>/results).
#
# Transfers are incremental and resumable: rerun the same command after each
# audit phase and only new/changed files are copied.
#
# Usage:
#   scripts/pull_results_rsync.sh                # pull outputs/trex
#   scripts/pull_results_rsync.sh --dry-run      # extra flags go to rsync
#   PULL_REMOTE_DIR=HALO/outputs/popqa scripts/pull_results_rsync.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ENV_FILE="${ENV_FILE:-$REPO_ROOT/.env}"
if [ -f "$ENV_FILE" ]; then
    set -a
    # shellcheck disable=SC1090
    . "$ENV_FILE"
    set +a
fi

: "${PULL_HOST:?PULL_HOST must be set in $ENV_FILE (e.g. ubuntu@1.2.3.4)}"
PULL_PORT="${PULL_PORT:-22}"
PULL_REMOTE_DIR="${PULL_REMOTE_DIR:-HALO/outputs/trex}"
PULL_LOCAL_DIR="${PULL_LOCAL_DIR:-results}"
case "$PULL_LOCAL_DIR" in /*|~*) ;; *) PULL_LOCAL_DIR="$REPO_ROOT/$PULL_LOCAL_DIR" ;; esac

SSH_CMD="ssh -p $PULL_PORT -o ConnectTimeout=10"
if [ -n "${PULL_KEY:-}" ]; then
    SSH_CMD="$SSH_CMD -i $PULL_KEY"
fi

mkdir -p "$PULL_LOCAL_DIR"

# Skip the giant raw provenance dumps (per-fact retrieval traces) by default;
# the analysis-ready metrics live in small CSV/PNG files. Set PULL_MAX_SIZE
# to another rsync size (e.g. 500m) or empty (PULL_MAX_SIZE=) to pull all.
PULL_MAX_SIZE="${PULL_MAX_SIZE-100m}"
SIZE_OPTS=()
if [ -n "$PULL_MAX_SIZE" ]; then
    SIZE_OPTS=(--max-size "$PULL_MAX_SIZE")
    echo "Skipping files larger than $PULL_MAX_SIZE (set PULL_MAX_SIZE= to pull everything)"
fi

# Trailing slashes: copy the *contents* of the remote dir into the local dir.
echo "Pulling $PULL_HOST:$PULL_REMOTE_DIR/ -> $PULL_LOCAL_DIR/"
rsync -avz --partial --progress ${SIZE_OPTS[@]+"${SIZE_OPTS[@]}"} -e "$SSH_CMD" "$@" \
    "$PULL_HOST:${PULL_REMOTE_DIR%/}/" \
    "${PULL_LOCAL_DIR%/}/"

echo "Done. Results are in $PULL_LOCAL_DIR"
