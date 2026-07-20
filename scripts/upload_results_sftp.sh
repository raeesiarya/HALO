#!/usr/bin/env bash
# Upload audit results to another machine over SFTP (OpenSSH scp).
#
# Reads connection settings from a .env file (default: <repo>/.env, override
# with ENV_FILE=/path/to/env). Required vars: SFTP_HOST, SFTP_USER.
# Optional: SFTP_PORT (default 22), SFTP_REMOTE_DIR (default results),
# SFTP_KEY (path to a private key), SFTP_PASS (password auth; requires
# sshpass to be installed — key auth is preferred).
#
# By default it tars up OUTPUT_DIR (same default as the audit suite,
# outputs/trex) into a single timestamped archive and uploads that.
# Set TARBALL=0 to copy the directory tree as-is instead.
#
# Usage:
#   scripts/upload_results_sftp.sh                 # upload outputs/trex
#   OUTPUT_DIR=outputs/popqa scripts/upload_results_sftp.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ENV_FILE="${ENV_FILE:-$REPO_ROOT/.env}"
if [ ! -f "$ENV_FILE" ]; then
    echo "error: $ENV_FILE not found; copy env.example to .env and fill it in" >&2
    exit 1
fi
set -a
# shellcheck disable=SC1090
. "$ENV_FILE"
set +a

: "${SFTP_HOST:?SFTP_HOST must be set in $ENV_FILE}"
: "${SFTP_USER:?SFTP_USER must be set in $ENV_FILE}"
SFTP_PORT="${SFTP_PORT:-22}"
SFTP_REMOTE_DIR="${SFTP_REMOTE_DIR:-results}"

OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/trex}"
case "$OUTPUT_DIR" in /*) ;; *) OUTPUT_DIR="$PWD/$OUTPUT_DIR" ;; esac
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "error: results directory $OUTPUT_DIR does not exist" >&2
    exit 1
fi

SSH_OPTS=(-p "$SFTP_PORT")
SCP_OPTS=(-P "$SFTP_PORT")
if [ -n "${SFTP_KEY:-}" ]; then
    SSH_OPTS+=(-i "$SFTP_KEY")
    SCP_OPTS+=(-i "$SFTP_KEY")
fi

# Key auth by default; SFTP_PASS falls back to sshpass if it's installed.
RUN=()
if [ -n "${SFTP_PASS:-}" ]; then
    if command -v sshpass >/dev/null 2>&1; then
        export SSHPASS="$SFTP_PASS"
        RUN=(sshpass -e)
    else
        echo "error: SFTP_PASS is set but sshpass is not installed." >&2
        echo "       Install sshpass, or unset SFTP_PASS and use SSH key auth" >&2
        echo "       (ssh-copy-id ${SFTP_USER}@${SFTP_HOST})." >&2
        exit 1
    fi
fi

DEST="$SFTP_USER@$SFTP_HOST"
echo "Ensuring remote directory $SFTP_REMOTE_DIR exists on $DEST"
"${RUN[@]}" ssh "${SSH_OPTS[@]}" "$DEST" "mkdir -p '$SFTP_REMOTE_DIR'"

TARBALL="${TARBALL:-1}"
if [ "$TARBALL" = "1" ]; then
    STAMP="$(date +%Y%m%d_%H%M%S)"
    ARCHIVE="$(mktemp -d)/$(basename "$OUTPUT_DIR")_${STAMP}.tar.gz"
    echo "Archiving $OUTPUT_DIR -> $ARCHIVE"
    tar -czf "$ARCHIVE" -C "$(dirname "$OUTPUT_DIR")" "$(basename "$OUTPUT_DIR")"
    echo "Uploading $(du -h "$ARCHIVE" | cut -f1) to $DEST:$SFTP_REMOTE_DIR/"
    "${RUN[@]}" scp "${SCP_OPTS[@]}" "$ARCHIVE" "$DEST:$SFTP_REMOTE_DIR/"
    rm -f "$ARCHIVE"
else
    echo "Copying $OUTPUT_DIR -> $DEST:$SFTP_REMOTE_DIR/"
    "${RUN[@]}" scp "${SCP_OPTS[@]}" -r "$OUTPUT_DIR" "$DEST:$SFTP_REMOTE_DIR/"
fi

echo "Upload complete."
