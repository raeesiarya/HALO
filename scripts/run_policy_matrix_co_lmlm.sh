#!/usr/bin/env bash
# Standard audit for each deletion-set policy, with separate output paths.
#
# All five policies share one FULL pass (FULL never consults the manifest)
# and reuse manifest-independent DEL-OFF rows from earlier results, so each
# policy only generates its own DEL-ON arm. FULL_DIR/REUSE_FROM come from
# the audit suite when delegated; standalone runs default to a FULL dir
# under this matrix's output root and chain the oracle run's results.
#
# POLICIES (space-separated subset of "oracle geometric value provenance
# hybrid factq") restricts which policies run — the cross-model scheduler
# uses it to run oracle first and the remaining policies in parallel.
# Oracle-result reuse is chained whenever the oracle results exist on disk,
# whether they were produced by this invocation or an earlier one.
#
# factq (the <FACT-q> query-ensemble rule) is valid but not in the default
# set: it needs FACTQ_VECTORS, the .npz produced by `python -m
# halo.factq_embed`, which does not exist until the factq-questions and
# factq-embed suite phases have run. The suite's factq-policy phase (and
# the cross-model scheduler) invoke it with POLICIES=factq once it has.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/trex_policy_matrix}"
DEL_OFF_MODE="${DEL_OFF_MODE:-null-retrieval}"

# Mirrors the default in run_audit_co_lmlm.sh; only used to name paths.
STEM="$(basename "${PROMPTS:-$REPO_ROOT/data/prompts_trex.jsonl}" .jsonl)"
FULL_DIR="${FULL_DIR:-$BASE_OUTPUT_DIR/${STEM}_full}"
export FULL_DIR

ALL_POLICIES="oracle geometric value provenance hybrid factq"
DEFAULT_POLICIES="oracle geometric value provenance hybrid"
POLICIES="${POLICIES:-$DEFAULT_POLICIES}"

# Reject typos rather than silently skipping a policy the user asked for.
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

policy_enabled() {
    case " $POLICIES " in *" $1 "*) return 0 ;; *) return 1 ;; esac
}

run_policy() {
    local label="$1"
    shift
    echo "=== Deletion policy: $label ==="
    OUTPUT_DIR="$BASE_OUTPUT_DIR/$label" \
    "$REPO_ROOT/scripts/run_audit_co_lmlm.sh" \
        --co-lmlm-del-off-mode "$DEL_OFF_MODE" \
        "$@"
}

if policy_enabled oracle; then
    run_policy oracle "$@"
fi
# When finalized oracle results exist (from this run or an earlier job),
# later policies reuse their DEL-OFF rows (and any DEL-ON whose manifest
# happens to coincide) on top of whatever REUSE_FROM the caller provided.
# Shard-only oracle runs have no results file yet, so they add nothing.
ORACLE_RESULTS="$BASE_OUTPUT_DIR/oracle/${STEM}_results.jsonl"
if [ -f "$ORACLE_RESULTS" ]; then
    REUSE_FROM="${REUSE_FROM:+$REUSE_FROM }$ORACLE_RESULTS"
    export REUSE_FROM
fi
if policy_enabled geometric; then
    run_policy geometric --closure geometric "$@"
fi
if policy_enabled value; then
    run_policy value --closure value "$@"
fi
if policy_enabled provenance; then
    run_policy provenance --closure provenance "$@"
fi
if policy_enabled hybrid; then
    run_policy hybrid --closure geometric,value,provenance "$@"
fi
if policy_enabled factq; then
    # Yair's query-ensemble rule, isolated: the closure is exactly what the
    # generated-question <FACT-q> queries retrieve.
    if [ -z "${FACTQ_VECTORS:-}" ] || [ ! -f "$FACTQ_VECTORS" ]; then
        echo "error: the factq policy needs FACTQ_VECTORS pointing at the .npz" >&2
        echo "       from 'python -m halo.factq_embed' (run the factq-questions" >&2
        echo "       and factq-embed suite phases first)" >&2
        exit 1
    fi
    run_policy factq --closure factq --factq-vectors "$FACTQ_VECTORS" \
        --factq-threshold "${FACTQ_THRESHOLD:-0.7}" "$@"
fi
