#!/usr/bin/env bash
# Run the HALO audit on Co-LMLM.
#
# Invoke from the HALO repo, but it executes inside the public Co-LMLM
# environment (Co-LMLM ships its own `lmlm` package, so it must run there):
# it clones/syncs the checkout if needed, cd's into it, and puts HALO's src
# on PYTHONPATH.
#
# Optional:  CO_LMLM_DIR (defaults to ../Co-LMLM next to this repo; cloned
#            from GitHub if absent), INDEX_DIR, PROMPTS, OUTPUT_DIR.
#            FULL_DIR (shared FULL-pass dir -> --full-dir), REUSE_FROM
#            (space-separated results files -> --reuse-from), SHARD
#            (I/N -> --shard), and LOG_FILE (-> --log-file) wire the
#            reuse/sharding machinery; unset, the invocation is unchanged.
# Extra flags (e.g. --closure, --radius-grid, --adversarial) are passed through:
#   ./scripts/run_audit_co_lmlm.sh --closure geometric --radius-grid 0.95:0.70:0.05
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CO_LMLM_REPO_URL="https://github.com/lil-lab/Co-LMLM.git"
CO_LMLM_DIR="${CO_LMLM_DIR:-$(dirname "$REPO_ROOT")/Co-LMLM}"

INDEX_DIR="${INDEX_DIR:-$REPO_ROOT/data/co-lmlm-fineweb-wiki-index}"
# T-REx slot-filling is the default audit corpus (in-context prompts, native
# continuation format); use PROMPTS=data/prompts.jsonl for the PopQA set.
PROMPTS="${PROMPTS:-$REPO_ROOT/data/prompts_trex.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/trex}"

# The audit runs from inside the Co-LMLM checkout, so anchor relative
# overrides to the invocation directory before we cd away.
case "$INDEX_DIR" in /*) ;; *) INDEX_DIR="$PWD/$INDEX_DIR" ;; esac
case "$PROMPTS" in /*) ;; *) PROMPTS="$PWD/$PROMPTS" ;; esac
case "$OUTPUT_DIR" in /*) ;; *) OUTPUT_DIR="$PWD/$OUTPUT_DIR" ;; esac

# Optional reuse/sharding wiring (see the audit suite). Paths are anchored
# before the cd below; unset variables add no flags at all.
EXTRA_FLAGS=()
if [ -n "${FULL_DIR:-}" ]; then
    case "$FULL_DIR" in /*) ;; *) FULL_DIR="$PWD/$FULL_DIR" ;; esac
    EXTRA_FLAGS+=(--full-dir "$FULL_DIR")
fi
if [ -n "${REUSE_FROM:-}" ]; then
    for reuse_path in $REUSE_FROM; do
        case "$reuse_path" in /*) ;; *) reuse_path="$PWD/$reuse_path" ;; esac
        EXTRA_FLAGS+=(--reuse-from "$reuse_path")
    done
fi
if [ -n "${SHARD:-}" ]; then
    EXTRA_FLAGS+=(--shard "$SHARD")
fi
if [ -n "${LOG_FILE:-}" ]; then
    case "$LOG_FILE" in /*) ;; *) LOG_FILE="$PWD/$LOG_FILE" ;; esac
    EXTRA_FLAGS+=(--log-file "$LOG_FILE")
fi

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

# faiss-gpu-cu12-cuvs ships without an RPATH to the RAPIDS and CUDA wheel
# directories, so importing faiss fails on libcuvs.so / librmm.so / libraft.so
# even though the wheels are installed. Point the dynamic loader at every
# wheel lib dir in the synced environment.
rapids_libs="$(
    find "$CO_LMLM_DIR"/.venv/lib/python*/site-packages -maxdepth 3 -type d \
        \( -name lib64 -o -name lib \) 2>/dev/null | paste -sd: -
)"
case ":${LD_LIBRARY_PATH:-}:" in
    *":$rapids_libs:"*) ;;  # already applied by a calling script
    *)
        if [ -n "$rapids_libs" ]; then
            export LD_LIBRARY_PATH="$rapids_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        fi
        ;;
esac

halo_audit() {
    PYTHONPATH="$REPO_ROOT/src:src${PYTHONPATH:+:$PYTHONPATH}" \
    uv run python -m halo.run_audit \
        --backend co-lmlm \
        --index-path "$INDEX_DIR" \
        --prompt-files "$PROMPTS" \
        --bootstrap-oracle-from-full \
        --output-dir "$OUTPUT_DIR" \
        ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"} \
        "$@"
}

# SUITE_WORKERS=N stripes this invocation into N fact-shard workers plus a
# finalize run — the same pattern the audit suite uses for its own phases.
# Sweeps shard by radius (--sweep-shard-radii), not by fact, and explicit
# SHARD/--shard invocations already are a single worker, so both fall
# through to the plain run.
case " $* " in
    *" --radius-grid "* | *" --shard "*) SUITE_WORKERS=1 ;;
esac
if [ "${SUITE_WORKERS:-1}" -gt 1 ] && [ -z "${SHARD:-}" ]; then
    pids=()
    for ((i = 0; i < SUITE_WORKERS; i++)); do
        halo_audit --shard "$i/$SUITE_WORKERS" \
            --log-file "$OUTPUT_DIR/run_audit.shard$i.log" "$@" &
        pids+=($!)
    done
    failed=0
    for pid in "${pids[@]}"; do
        wait "$pid" || failed=1
    done
    if [ "$failed" -ne 0 ]; then
        echo "error: an audit shard worker failed; see $OUTPUT_DIR/run_audit.shard*.log" >&2
        exit 1
    fi
fi
halo_audit "$@"
