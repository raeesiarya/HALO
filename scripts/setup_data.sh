#!/usr/bin/env bash
# Set up all cross-model audit data in one go:
#   scripts/setup_colmlm.sh   prompt sets + the fineweb+wiki index (~1.05 TB)
#   scripts/setup_nulls.sh    NULLs checkpoint, title mapping, corpus, closure
#                             artifact (litgpt dependency group included)
# Each half is its own script so a machine that runs one paradigm sets up
# only that. SKIP_COLMLM=1 / SKIP_NULLS=1 skip a half; every other knob
# (INDEX_DIR, SKIP_INDEX, NULLS_*, SKIP_CORPUS, SKIP_CLOSURE, HF_TOKEN, ...)
# passes through to the halves.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ "${SKIP_COLMLM:-0}" = "1" ]; then
    echo "SKIP_COLMLM=1 — skipping scripts/setup_colmlm.sh"
else
    "$REPO_ROOT/scripts/setup_colmlm.sh"
fi

if [ "${SKIP_NULLS:-0}" = "1" ]; then
    echo "SKIP_NULLS=1 — skipping scripts/setup_nulls.sh"
else
    "$REPO_ROOT/scripts/setup_nulls.sh"
fi
