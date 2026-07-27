#!/usr/bin/env bash
# Standard audit for each deletion-set policy, with separate output paths.
#
# All five policies share one FULL pass (FULL never consults the manifest)
# and reuse manifest-independent DEL-OFF rows from earlier results, so each
# policy only generates its own DEL-ON arm. FULL_DIR/REUSE_FROM come from
# the audit suite when delegated; standalone runs default to a FULL dir
# under this matrix's output root and chain the oracle run's results.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/trex_policy_matrix}"
DEL_OFF_MODE="${DEL_OFF_MODE:-null-retrieval}"

# Mirrors the default in run_audit_co_lmlm.sh; only used to name paths.
STEM="$(basename "${PROMPTS:-$REPO_ROOT/data/prompts_trex.jsonl}" .jsonl)"
FULL_DIR="${FULL_DIR:-$BASE_OUTPUT_DIR/${STEM}_full}"
export FULL_DIR

run_policy() {
    local label="$1"
    shift
    echo "=== Deletion policy: $label ==="
    OUTPUT_DIR="$BASE_OUTPUT_DIR/$label" \
    "$REPO_ROOT/scripts/run_audit_co_lmlm.sh" \
        --co-lmlm-del-off-mode "$DEL_OFF_MODE" \
        "$@"
}

run_policy oracle "$@"
# The oracle results now exist; later policies reuse their DEL-OFF rows
# (and any DEL-ON whose manifest happens to coincide) on top of whatever
# REUSE_FROM the caller provided.
REUSE_FROM="${REUSE_FROM:+$REUSE_FROM }$BASE_OUTPUT_DIR/oracle/${STEM}_results.jsonl"
export REUSE_FROM
run_policy geometric --closure geometric "$@"
run_policy value --closure value "$@"
run_policy provenance --closure provenance "$@"
run_policy hybrid --closure geometric,value,provenance "$@"
