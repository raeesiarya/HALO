#!/usr/bin/env bash
# Build the audit prompt sets (shared by setup_colmlm.sh and setup_nulls.sh):
#   T-REx        -> data/prompts_trex.jsonl        (default corpus)
#   PopQA        -> data/prompts.jsonl
#   Google-RE    -> data/prompts_googlere.jsonl
#   CounterFact  -> data/prompts_counterfact.jsonl
#   ZsRE         -> data/prompts_zsre.jsonl
#
# The prep scripts are seeded, so a rebuild is byte-identical unless a prep
# script changed. The scheduler and the reuse store fingerprint the prompt
# files: a changed file re-runs that set's jobs for every model, which is the
# intended behavior (never edit a prompt file by hand).
#
#   SKIP_PROMPTS=1        skip entirely
#   PROMPTS_IF_MISSING=1  only build sets whose file is absent
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ "${SKIP_PROMPTS:-0}" = "1" ]; then
    echo "SKIP_PROMPTS=1 — not building prompt sets."
    exit 0
fi

SCRIPTS=(prepare_trex_audit.py prepare_popqa_audit.py prepare_googlere_audit.py prepare_counterfact_audit.py prepare_zsre_audit.py)
OUTPUTS=(prompts_trex.jsonl prompts.jsonl prompts_googlere.jsonl prompts_counterfact.jsonl prompts_zsre.jsonl)

for i in "${!SCRIPTS[@]}"; do
    output="$REPO_ROOT/data/${OUTPUTS[$i]}"
    if [ "${PROMPTS_IF_MISSING:-0}" = "1" ] && [ -f "$output" ]; then
        echo "  ${OUTPUTS[$i]} present — kept."
        continue
    fi
    echo "  building ${OUTPUTS[$i]} ..."
    uv run python "$REPO_ROOT/data/${SCRIPTS[$i]}"
done
