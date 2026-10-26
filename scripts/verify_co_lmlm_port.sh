#!/usr/bin/env bash
# Check that this machine's Co-LMLM runtime reproduces the canonical runs.
#
# Re-audits the first N facts of one prompt set exactly like the canonical
# oracle policy row (bootstrap oracle, no closure, same DEL-OFF mode) and
# compares the rows with the canonical results downloaded into OUT_ROOT.
# Run it once after setting up a new platform (e.g. the Linux/aarch64 port
# in envs/co-lmlm-aarch64) and before mixing its results with x86 ones.
#
#   ./scripts/verify_co_lmlm_port.sh [SET] [N]     (default: zsre 200)
#
# Needs the index (scripts/setup_colmlm.sh) and
# $OUT_ROOT/co-lmlm/<set>/policy_matrix/oracle/<stem>_results.jsonl.
# Optional: OUT_ROOT (default out-cross-model), MIN_AGREEMENT (0.98),
# plus the knobs of run_audit_co_lmlm.sh (CO_LMLM_DIR, INDEX_DIR, ...).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SET="${1:-zsre}"
LIMIT="${2:-200}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/out-cross-model}"
case "$OUT_ROOT" in /*) ;; *) OUT_ROOT="$PWD/$OUT_ROOT" ;; esac

case "$SET" in
    popqa) PROMPTS="$REPO_ROOT/data/prompts.jsonl" ;;
    trex|googlere|counterfact|zsre) PROMPTS="$REPO_ROOT/data/prompts_$SET.jsonl" ;;
    *) echo "error: unknown set '$SET'" >&2; exit 1 ;;
esac
STEM="$(basename "$PROMPTS" .jsonl)"
REFERENCE="$OUT_ROOT/co-lmlm/$SET/policy_matrix/oracle/${STEM}_results.jsonl"
REFERENCE_CONFIG="$OUT_ROOT/co-lmlm/$SET/policy_matrix/oracle/${STEM}_audit_config.json"
CHECK_DIR="$OUT_ROOT/_port_check/co-lmlm/$SET"

for path in "$PROMPTS" "$REFERENCE"; do
    if [ ! -f "$path" ]; then
        echo "error: $path not found" >&2
        exit 1
    fi
done

# The canonical prompt file must be the one the reference audited.
if [ -f "$REFERENCE_CONFIG" ]; then
    expected="$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['prompt_sha256'])" "$REFERENCE_CONFIG")"
    actual="$(sha256sum "$PROMPTS" | cut -d' ' -f1)"
    if [ "$expected" != "$actual" ]; then
        echo "error: $PROMPTS differs from the prompt file the reference audited" >&2
        echo "       (sha256 $actual vs $expected)" >&2
        exit 1
    fi
    DEL_OFF_MODE="$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('del_off_mode') or 'null-retrieval')" "$REFERENCE_CONFIG")"
fi
DEL_OFF_MODE="${DEL_OFF_MODE:-null-retrieval}"

echo "=== Re-auditing the first $LIMIT $SET facts (oracle row, DEL-OFF $DEL_OFF_MODE) ==="
rm -rf "$CHECK_DIR"
PROMPTS="$PROMPTS" OUTPUT_DIR="$CHECK_DIR" FULL_DIR="" REUSE_FROM="" SHARD="" \
    "$REPO_ROOT/scripts/run_audit_co_lmlm.sh" \
    --co-lmlm-del-off-mode "$DEL_OFF_MODE" \
    --limit "$LIMIT"

echo "=== Comparing with $REFERENCE ==="
uv run --project "$REPO_ROOT" python "$REPO_ROOT/scripts/compare_audit_results.py" \
    --reference "$REFERENCE" \
    --candidate "$CHECK_DIR/${STEM}_results.jsonl" \
    --min-agreement "${MIN_AGREEMENT:-0.98}" \
    --json "$CHECK_DIR/port_check.json"
