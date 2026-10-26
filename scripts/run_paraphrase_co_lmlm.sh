#!/usr/bin/env bash
# Audit frozen deletion sets under paraphrased prompts (Co-LMLM).
#
# Each deletion policy built D(f) from the canonical eval prompt; this
# re-asks every fact through paraphrases while D(f) stays exactly as the
# canonical run persisted it. The per-policy prompt files come from
# `python -m halo.paraphrase build` (suite phase paraphrase-build, or the
# scheduler's paraphrase-build job): rows carry the frozen manifest, so the
# audit runs with no --closure and the oracle bootstrap never fires (the
# builder drops empty manifests).
#
# The gate policy (oracle) runs first; every other policy passes its
# results as --reuse-from, so FULL and DEL-OFF — both manifest-independent
# — are generated once per prompt set, and DEL-ON is served wherever the
# backend certifies the manifests equivalent. This mirrors
# run_policy_matrix_co_lmlm.sh.
#
# Required:  OUTPUT_DIR (the paraphrase root, <set>/paraphrase),
#            PARAPHRASE_STEM (prompt stem, <prompts>_paraphrase)
# Optional:  POLICIES (default: every policy whose prompt file exists),
#            PARAPHRASE_GATE (default: oracle), DEL_OFF_MODE, plus the
#            CO_LMLM_DIR / INDEX_DIR knobs of run_audit_co_lmlm.sh.
# Extra flags (e.g. --shard I/N, --log-file, --limit) are passed through.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR (the paraphrase root) is required}"
PARAPHRASE_STEM="${PARAPHRASE_STEM:?PARAPHRASE_STEM is required}"
DEL_OFF_MODE="${DEL_OFF_MODE:-null-retrieval}"
GATE="${PARAPHRASE_GATE:-oracle}"
case "$BASE_OUTPUT_DIR" in /*) ;; *) BASE_OUTPUT_DIR="$PWD/$BASE_OUTPUT_DIR" ;; esac

ALL_POLICIES="oracle geometric value provenance hybrid factq"
EXPLICIT_POLICIES="${POLICIES:-}"
POLICIES="${POLICIES:-$ALL_POLICIES}"
for requested in $POLICIES; do
    case " $ALL_POLICIES " in
        *" $requested "*) ;;
        *)
            echo "error: unknown policy '$requested' in POLICIES" >&2
            echo "       valid policies: $ALL_POLICIES" >&2
            exit 1
            ;;
    esac
done

# Gate first so the others can reuse its FULL/DEL-OFF rows.
ORDERED=""
case " $POLICIES " in *" $GATE "*) ORDERED="$GATE" ;; esac
for policy in $POLICIES; do
    [ "$policy" = "$GATE" ] || ORDERED="$ORDERED $policy"
done

GATE_RESULTS="$BASE_OUTPUT_DIR/$GATE/${PARAPHRASE_STEM}_results.jsonl"
ran=0
for policy in $ORDERED; do
    prompts="$BASE_OUTPUT_DIR/$policy/${PARAPHRASE_STEM}.jsonl"
    if [ ! -f "$prompts" ]; then
        if [ -n "$EXPLICIT_POLICIES" ]; then
            echo "error: $prompts not found; run the paraphrase-build phase first" >&2
            exit 1
        fi
        echo "  (no paraphrase prompts for $policy; skipped)"
        continue
    fi
    reuse=""
    if [ "$policy" != "$GATE" ] && [ -f "$GATE_RESULTS" ]; then
        reuse="$GATE_RESULTS"
    fi
    echo "=== Paraphrase audit: $policy ==="
    # A per-policy FULL_DIR, never the canonical one: the canonical shared
    # FULL pass is keyed on the canonical facts and must not see this file.
    # Setting it turns on persistence, which writes the audit config the
    # next policy's --reuse-from needs.
    PROMPTS="$prompts" OUTPUT_DIR="$BASE_OUTPUT_DIR/$policy" \
    FULL_DIR="$BASE_OUTPUT_DIR/$policy/${PARAPHRASE_STEM}_full" \
    REUSE_FROM="$reuse" \
        "$REPO_ROOT/scripts/run_audit_co_lmlm.sh" \
        --co-lmlm-del-off-mode "$DEL_OFF_MODE" \
        "$@"
    ran=$((ran + 1))
done

if [ "$ran" -eq 0 ]; then
    echo "error: no paraphrase prompt files under $BASE_OUTPUT_DIR" >&2
    exit 1
fi
