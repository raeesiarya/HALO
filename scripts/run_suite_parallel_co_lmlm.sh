#!/usr/bin/env bash
# Submit the full Co-LMLM audit suite for every prompt set in parallel — one
# process per set, one GPU per set — and return immediately. Nothing streams to
# the terminal: the run is detached and every set logs to its own file.
#
#   trex        data/prompts_trex.jsonl        -> $OUT_ROOT/trex/suite.log
#   popqa       data/prompts.jsonl             -> $OUT_ROOT/popqa/suite.log
#   googlere    data/prompts_googlere.jsonl    -> $OUT_ROOT/googlere/suite.log
#   counterfact data/prompts_counterfact.jsonl -> $OUT_ROOT/counterfact/suite.log
#   zsre        data/prompts_zsre.jsonl        -> $OUT_ROOT/zsre/suite.log
#
# Tail any set on its own, e.g.  tail -f out/counterfact/suite.log
# The one-time environment warmup and the final per-set pass/fail summary land
# in $OUT_ROOT/_driver.log.
#
# This is a thin fan-out over scripts/run_audit_suite_co_lmlm.sh: each set gets
# its own process with PROMPTS/OUTPUT_DIR set, and all other suite knobs
# (SUITE_PHASES, RADIUS_GRID, STANDARD_CLOSURE, ...) plus any extra CLI flags in
# "$@" are inherited unchanged, so every set runs the identical protocol.
#
# GPUs: defaults to 8 (0..7), round-robin one per set — with six sets that
# leaves GPUs 6 and 7 idle. Override the pool or the cap with:
#   GPUS="0,1,2,3"     GPU ids to round-robin across sets.
#   MAX_PARALLEL=N     cap concurrent sets (default: len(GPUS)). Extra sets queue.
#
# Config (env):
#   SETS         space-separated subset of the names above (default: all)
#   OUT_ROOT     parent output dir (default: $REPO_ROOT/out)
#   GPUS         comma-separated GPU ids (default: 0,1,2,3,4,5,6,7)
#   MAX_PARALLEL concurrent sets (default: len(GPUS))
#   CO_LMLM_DIR  Co-LMLM checkout (default: ../Co-LMLM; cloned if absent)
#   INDEX_DIR    wiki index (default: $REPO_ROOT/data/co-lmlm-wiki-index)
# Extra flags after the script name are forwarded to every suite invocation,
# e.g.  ./scripts/run_suite_parallel_co_lmlm.sh --limit 200
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUITE="$REPO_ROOT/scripts/run_audit_suite_co_lmlm.sh"

CO_LMLM_REPO_URL="https://github.com/lil-lab/Co-LMLM.git"
CO_LMLM_DIR="${CO_LMLM_DIR:-$(dirname "$REPO_ROOT")/Co-LMLM}"
INDEX_DIR="${INDEX_DIR:-$REPO_ROOT/data/co-lmlm-wiki-index}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/out}"
DRIVER_LOG="$OUT_ROOT/_driver.log"

DEFAULT_SETS="trex popqa googlere counterfact zsre"
SETS="${SETS:-$DEFAULT_SETS}"

# name -> prompt file (popqa keeps the legacy prompts.jsonl name).
prompt_file_for() {
    case "$1" in
        trex)        echo "$REPO_ROOT/data/prompts_trex.jsonl" ;;
        popqa)       echo "$REPO_ROOT/data/prompts.jsonl" ;;
        googlere)    echo "$REPO_ROOT/data/prompts_googlere.jsonl" ;;
        counterfact) echo "$REPO_ROOT/data/prompts_counterfact.jsonl" ;;
        zsre)        echo "$REPO_ROOT/data/prompts_zsre.jsonl" ;;
        *)           return 1 ;;
    esac
}

# GPU pool (default: all 8) and concurrency cap, shared by driver and launcher.
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
gpu_ids=()
[ -n "$GPUS" ] && IFS=',' read -r -a gpu_ids <<< "$GPUS"
set_count=0
for _ in $SETS; do set_count=$((set_count + 1)); done
if [ -n "${MAX_PARALLEL:-}" ]; then
    max_parallel="$MAX_PARALLEL"
elif [ "${#gpu_ids[@]}" -gt 0 ]; then
    max_parallel="${#gpu_ids[@]}"
else
    max_parallel="$set_count"
fi
[ "$max_parallel" -lt 1 ] && max_parallel=1

# ---------------------------------------------------------------------------
# Detached driver: warmup once, then fan the sets out. Runs in the background so
# the launcher can return immediately; all its output goes to $DRIVER_LOG.
# ---------------------------------------------------------------------------
run_driver() {
    # One-time warmup so the parallel suites don't race on the shared clone, the
    # uv lock, or the dpkg lock. Each suite still preflights faiss, but by then
    # OpenBLAS and the synced venv already exist.
    if [ ! -d "$CO_LMLM_DIR" ]; then
        echo "Co-LMLM checkout not found; cloning $CO_LMLM_REPO_URL -> $CO_LMLM_DIR"
        git clone "$CO_LMLM_REPO_URL" "$CO_LMLM_DIR"
    fi
    echo "Warming up the Co-LMLM environment (uv sync) ..."
    ( cd "$CO_LMLM_DIR" && uv sync )
    if command -v apt-get >/dev/null 2>&1 && command -v dpkg >/dev/null 2>&1; then
        if ! dpkg -s libopenblas0 >/dev/null 2>&1; then
            echo "Pre-installing OpenBLAS (shared by all parallel suites) ..."
            sudo_cmd=""
            [ "$(id -u)" -ne 0 ] && sudo_cmd="sudo"
            $sudo_cmd apt-get update && $sudo_cmd apt-get install -y libopenblas0 || true
        fi
    fi

    echo "Launching audit suites: [$SETS]  (concurrency $max_parallel, gpus ${gpu_ids[*]:-default})"

    # Portable job pool: throttle by counting launched sets whose .exit marker is
    # not yet present (works without bash 4.3 `wait -n`).
    local launched_names=()
    count_active() {
        local n=0 name
        for name in "${launched_names[@]:-}"; do
            [ -z "$name" ] && continue
            [ -f "$OUT_ROOT/$name/.exit" ] || n=$((n + 1))
        done
        echo "$n"
    }

    local index=0 name prompts outdir gpu
    for name in $SETS; do
        while [ "$(count_active)" -ge "$max_parallel" ]; do
            sleep 2
        done

        prompts="$(prompt_file_for "$name")"
        outdir="$OUT_ROOT/$name"
        mkdir -p "$outdir"
        rm -f "$outdir/.exit"

        gpu=""
        if [ "${#gpu_ids[@]}" -gt 0 ]; then
            gpu="${gpu_ids[$((index % ${#gpu_ids[@]}))]}"
            echo "  -> $name  (GPU $gpu)  -> $outdir/suite.log"
        else
            echo "  -> $name  -> $outdir/suite.log"
        fi

        # The trap records the suite's exit code even on non-zero exit, so the
        # pool can free the slot and the summary can report per-set status.
        (
            trap 'echo $? > "'"$outdir"'/.exit"' EXIT
            export CO_LMLM_DIR INDEX_DIR
            export PROMPTS="$prompts" OUTPUT_DIR="$outdir"
            [ -n "$gpu" ] && export CUDA_VISIBLE_DEVICES="$gpu"
            "$SUITE" "$@"
        ) > "$outdir/suite.log" 2>&1 &

        launched_names+=("$name")
        index=$((index + 1))
    done

    wait

    echo
    echo "=== Parallel audit summary ==="
    local code
    for name in "${launched_names[@]}"; do
        code="$(cat "$OUT_ROOT/$name/.exit" 2>/dev/null || echo "??")"
        if [ "$code" = "0" ]; then
            echo "  [ok]   $name -> $OUT_ROOT/$name"
        else
            echo "  [FAIL] $name (exit $code) -> $OUT_ROOT/$name/suite.log"
        fi
    done
    echo "=== Done ($SETS) ==="
}

# Re-entry point for the detached driver process.
if [ "${1:-}" = "__driver" ]; then
    shift
    run_driver "$@"
    exit 0
fi

# ---------------------------------------------------------------------------
# Launcher (foreground): validate fast, then detach the driver and return.
# ---------------------------------------------------------------------------
for name in $SETS; do
    prompts="$(prompt_file_for "$name")" || {
        echo "error: unknown set '$name' (valid: $DEFAULT_SETS)" >&2
        exit 1
    }
    if [ ! -f "$prompts" ]; then
        echo "error: prompt file for '$name' not found: $prompts" >&2
        echo "       build it first with scripts/setup_data.sh" >&2
        exit 1
    fi
done

# Pre-create the log files so `tail -f` works the instant this returns.
mkdir -p "$OUT_ROOT"
: > "$DRIVER_LOG"
for name in $SETS; do
    mkdir -p "$OUT_ROOT/$name"
    : > "$OUT_ROOT/$name/suite.log"
done

# Detach: nohup + background survives the terminal closing; the driver's stdout
# and stderr go to the driver log, per-set output to each set's suite.log.
nohup bash "$0" __driver "$@" </dev/null >"$DRIVER_LOG" 2>&1 &
disown 2>/dev/null || true

# Minimal receipt: where the job went and how to watch it. No run output here.
echo "Submitted $set_count audit set(s), detached. Output root: $OUT_ROOT"
echo "  driver log:  tail -f $DRIVER_LOG"
for name in $SETS; do
    echo "  $name:  tail -f $OUT_ROOT/$name/suite.log"
done
