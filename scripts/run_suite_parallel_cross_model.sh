#!/usr/bin/env bash
# Submit the full cross-model audit — every prompt set x every model — as a
# detached scheduler run and return immediately, mirroring
# scripts/run_suite_parallel_co_lmlm.sh: the launcher validates fast, nothing
# streams to the terminal, and every job logs to its own file.
#
# Where the Co-LMLM launcher fans out one whole suite per prompt set, this
# one submits the dependency-aware job graph (src/halo/scheduler.py): per
# prompt set the Co-LMLM standard phase gates sweep/adversarial/del-off/
# policy, the sweep fans out per radius, the policy matrix runs oracle first
# and the four remaining policies in parallel, SCHEDULER_SHARDS=N stripes
# the fact-striped phases into N single-GPU jobs each, and the two
# parametric baselines run alongside from the start.
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
#   MODELS           subset of co-lmlm,standard-lm-360m-fw,smollm2-360m
#                    (default: all three)
#   SUITE_WORKERS    worker processes inside unsharded Co-LMLM jobs (default: 1)
#   SCHEDULER_SHARDS fact-shard jobs per striped phase (default: 1)
# Extra flags after the script name are forwarded to every audit job, e.g.
#   ./scripts/run_suite_parallel_cross_model.sh --limit 200
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/out-cross-model}"
export OUT_ROOT

# SETS/GPUS/CO_LMLM_DIR/INDEX_DIR/SUITE_WORKERS/SCHEDULER_SHARDS are read
# from the environment by the scheduler itself; the launcher only maps the
# knobs that lack an environment default. The scheduler validates the plan
# (unknown sets, missing prompt files, missing index) before detaching, so
# a bad invocation fails here instead of in a background log.
SCHEDULER_ARGS=(--detach)
[ -n "${MAX_PARALLEL:-}" ] && SCHEDULER_ARGS+=(--max-parallel "$MAX_PARALLEL")
[ -n "${MODELS:-}" ] && SCHEDULER_ARGS+=(--models "$MODELS")

# Audit flags go to the scheduler after "--" so they reach every job.
if [ "$#" -gt 0 ]; then
    SCHEDULER_ARGS+=(-- "$@")
fi

exec "$REPO_ROOT/scripts/run_cross_model_scheduler.sh" "${SCHEDULER_ARGS[@]}"
