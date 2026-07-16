#!/usr/bin/env bash
# Run the HALO audit on Co-LMLM.
#
# Invoke from the HALO repo, but it executes inside the public Co-LMLM
# environment (Co-LMLM ships its own `lmlm` package, so it must run there):
# it clones/syncs the checkout if needed, cd's into it, and puts HALO's src
# on PYTHONPATH.
#
# Optional:  CO_LMLM_DIR (defaults to ../Co-LMLM next to this repo; cloned
#            from GitHub if absent), INDEX_DIR, PROMPTS, OUTPUT_DIR
# Extra flags (e.g. --closure, --radius-grid, --adversarial) are passed through:
#   ./scripts/run_audit_co_lmlm.sh --closure geometric,semantic --radius-grid 0.95:0.70:0.05
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CO_LMLM_REPO_URL="https://github.com/lil-lab/Co-LMLM.git"
CO_LMLM_DIR="${CO_LMLM_DIR:-$(dirname "$REPO_ROOT")/Co-LMLM}"

INDEX_DIR="${INDEX_DIR:-$REPO_ROOT/data/co-lmlm-wiki-index}"
PROMPTS="${PROMPTS:-$REPO_ROOT/data/prompts.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/popqa}"

if [ ! -d "$CO_LMLM_DIR" ]; then
    echo "Co-LMLM checkout not found; cloning $CO_LMLM_REPO_URL -> $CO_LMLM_DIR"
    git clone "$CO_LMLM_REPO_URL" "$CO_LMLM_DIR"
elif [ ! -f "$CO_LMLM_DIR/src/lmlm/eval/hf_generate.py" ]; then
    echo "error: $CO_LMLM_DIR exists but does not look like the public Co-LMLM checkout" >&2
    echo "       (missing src/lmlm/eval/hf_generate.py); set CO_LMLM_DIR to the right path" >&2
    exit 1
fi

cd "$CO_LMLM_DIR"
echo "Syncing the Co-LMLM environment (uv sync) ..."
uv sync

PYTHONPATH="$REPO_ROOT/src:src${PYTHONPATH:+:$PYTHONPATH}" \
uv run python -m halo.run_audit \
    --backend co-lmlm \
    --index-path "$INDEX_DIR" \
    --prompt-files "$PROMPTS" \
    --bootstrap-oracle-from-full \
    --wandb-activation on \
    --output-dir "$OUTPUT_DIR" \
    "$@"
