#!/usr/bin/env bash
# Submit the full cross-model audit — every prompt set x every model — as a
# detached scheduler run and return immediately, mirroring
# scripts/run_suite_parallel_co_lmlm.sh: the launcher validates fast, nothing
# streams to the terminal, and every job logs to its own file.
#
# Where the Co-LMLM launcher fans out one whole suite per prompt set, this
# one submits the dependency-aware job graph (src/halo/scheduler.py) for
# EVERYTHING — all four models, prep included, one command:
#   - Co-LMLM: the standard phase gates sweep/adversarial/del-off/policy/
#     factq, the sweep fans out per radius, the policy matrix runs oracle
#     first and the four remaining policies in parallel, the factq chain
#     runs question generation -> <FACT-q> embedding -> the factq policy
#     row (Co-LMLM only — no other backend has the retrieval index).
#   - The two closed-book baselines run alongside from the start.
#   - NULLs (nulls-wiki-1b): the prep chain runs as jobs in the same graph
#     (title-embedding build -> per-set source-title augmentation -> the
#     striped verification gate emitting the gated prompt set), then the
#     standard audit, the complementary DEL-OFF sensitivity run, and — when
#     the shared-encoder closure artifact exists — the source-closure build
#     plus one breadth-k sweep group per k, and (with the corpus present)
#     the value/hybrid policy rows. Requires the artifacts from
#     ./scripts/setup_nulls.sh (checkpoint, title_to_index.pkl, the closure
#     artifact for sweep/policy, the corpus for policy).
# SCHEDULER_SHARDS=N stripes every fact-striped phase (Co-LMLM and NULLs,
# gate included) into N single-GPU jobs each.
#
# Tail the run:
#   driver:   tail -F $OUT_ROOT/_scheduler.log
#   one job:  tail -F $OUT_ROOT/_scheduler_jobs/<model>.<set>.<phase>.log
#
# Config (env), matching run_suite_parallel_co_lmlm.sh where it applies:
#   SETS             space- or comma-separated prompt sets (default: all five)
#   OUT_ROOT         parent output dir (default: $REPO_ROOT/out-cross-model)
#   GPUS             comma-separated GPU ids (default: 0,1,2,3,4,5,6,7)
#   MAX_PARALLEL     concurrent jobs (default: len(GPUS))
#   CO_LMLM_DIR      Co-LMLM checkout (default: ../Co-LMLM; cloned if absent)
#   INDEX_DIR        fineweb+wiki index (default: data/co-lmlm-fineweb-wiki-index)
#   MODELS           subset of co-lmlm,standard-lm-360m-fw,smollm2-360m,
#                    nulls-wiki-1b (default: all four)
#   SUITE_WORKERS    worker processes inside unsharded Co-LMLM jobs (default: 1)
#   SCHEDULER_SHARDS fact-shard jobs per striped phase (default: 1)
#   FACTQ_NUM_QUESTIONS / FACTQ_THRESHOLD / FACTQ_ADAPTER
#                    factq-chain knobs (defaults: 5 / 0.7 / the released
#                    lil-lab/CoLMLM-Question-Generator adapter)
#   NULLS_CHECKPOINT_DIR / NULLS_TITLE_TO_INDEX / NULLS_TITLE_EMBEDDINGS /
#   NULLS_CLOSURE_EMBEDDINGS / NULLS_DEL_OFF_MODE / NULLS_GATE_MARGIN /
#   NULLS_SWEEP_K_GRID / NULLS_POLICY_K / NULLS_CORPUS_DIR / NULLS_PHASES
#                    NULLs artifact paths and knobs (defaults: data/ paths,
#                    sinks-zero, 0.0, 1,2,4,8,16, 4, all phases)
# Extra flags after the script name are forwarded to every audit job, e.g.
#   ./scripts/run_suite_parallel_cross_model.sh --limit 200
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/out-cross-model}"
export OUT_ROOT

# Everything in one: the default model list includes nulls-wiki-1b, so a bare
# invocation submits the complete cross-model graph. Narrow with MODELS=...
# on machines without the NULLs artifacts.
MODELS="${MODELS:-co-lmlm,standard-lm-360m-fw,smollm2-360m,nulls-wiki-1b}"

# SETS/GPUS/CO_LMLM_DIR/INDEX_DIR/SUITE_WORKERS/SCHEDULER_SHARDS/NULLS_* are
# read from the environment by the scheduler itself; the launcher only maps
# the knobs that lack an environment default. The scheduler validates the
# plan (unknown sets, missing prompt files, missing index, missing NULLs
# artifacts) before detaching, so a bad invocation fails here instead of in
# a background log.
SCHEDULER_ARGS=(--detach --models "$MODELS")
[ -n "${MAX_PARALLEL:-}" ] && SCHEDULER_ARGS+=(--max-parallel "$MAX_PARALLEL")

# Audit flags go to the scheduler after "--" so they reach every job.
if [ "$#" -gt 0 ]; then
    SCHEDULER_ARGS+=(-- "$@")
fi

exec "$REPO_ROOT/scripts/run_cross_model_scheduler.sh" "${SCHEDULER_ARGS[@]}"
