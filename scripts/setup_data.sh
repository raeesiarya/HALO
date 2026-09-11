#!/usr/bin/env bash
# Set up all cross-model audit data in one go:
#   scripts/setup_colmlm.sh   prompt sets + the fineweb+wiki index (~1.05 TB)
#   scripts/setup_nulls.sh    NULLs checkpoint, title mapping, corpus, closure
#                             artifact (litgpt dependency group included)
# Each half is its own script so a machine that runs one paradigm sets up
# only that. SKIP_COLMLM=1 / SKIP_NULLS=1 skip a half; every other knob
# (INDEX_DIR, SKIP_INDEX, NULLS_*, SKIP_CORPUS, SKIP_CLOSURE, HF_TOKEN, ...)
# passes through to the halves.
# By default the script detaches and logs to logs/setup_data.log, and each
# half to its own logs/<half>.log (tail -F them); HALO_SETUP_FOREGROUND=1
# runs everything attached.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/_detach.sh
source "$REPO_ROOT/scripts/_detach.sh"
detach_setup setup_data "$@"

run_half() {
    local half="$1"
    if [ "${HALO_SETUP_DETACHED:-0}" = "1" ]; then
        # The half inherits HALO_SETUP_DETACHED, so it runs in place and
        # brackets its own log with start/finish lines.
        echo "-> $half  (tail -F $REPO_ROOT/logs/$half.log)"
        "$REPO_ROOT/scripts/$half.sh" > "$REPO_ROOT/logs/$half.log" 2>&1
    else
        "$REPO_ROOT/scripts/$half.sh"
    fi
}

if [ "${SKIP_COLMLM:-0}" = "1" ]; then
    echo "SKIP_COLMLM=1 — skipping scripts/setup_colmlm.sh"
else
    run_half setup_colmlm
fi

if [ "${SKIP_NULLS:-0}" = "1" ]; then
    echo "SKIP_NULLS=1 — skipping scripts/setup_nulls.sh"
else
    run_half setup_nulls
fi
