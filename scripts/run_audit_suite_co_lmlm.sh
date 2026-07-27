#!/usr/bin/env bash
# Run the full HALO audit suite on Co-LMLM: the standard three-state audit,
# the entanglement sweep, and the adversarial-closure evaluation.
#
# Invoke from the HALO repo, but it executes inside the public Co-LMLM
# environment (Co-LMLM ships its own `lmlm` package, so it must run there):
# it clones/syncs the checkout if needed, cd's into it, and puts HALO's src
# on PYTHONPATH.
#
# The three phases are separate evaluation modes and run sequentially — they
# share one GPU, and the sweep/adversarial phases share one FULL pass (the
# <prompts>_full/ directory). Every phase is resumable, so if the suite dies
# partway, re-running it skips everything already on disk.
#
# Wall-clock tip: phase 2 shards cleanly by radius. Run one single-radius
# sweep first (e.g. RADIUS_GRID=0.95:0.95:0.05) so the shared FULL pass is
# materialized, then launch the remaining radii as parallel processes (one
# GPU each, same OUTPUT_DIR — each writes its own sweep_rho_*.jsonl), and
# finally re-run the full grid: it resumes every per-radius file and only
# computes the analysis. Do NOT shard by prompt file subsets — neighbor
# sets N(f) are defined within a prompt file.
#
# The suite runs every published evaluation by default: the three phases
# above, plus the DEL-OFF sensitivity control and the deletion-policy matrix.
# SUITE_PHASES narrows that when you need it:
#   all (default) = standard,sweep,adversarial,del-off,policy
#   core          = standard,sweep,adversarial
#   or an explicit comma-separated subset, e.g. SUITE_PHASES=standard,policy
# Narrowing is mainly useful for resuming a partial run or iterating on one
# phase; the del-off and policy phases are a full standard audit per variant
# (one and four respectively, after de-duplication against phase 1), so the
# default suite is roughly three times the cost of `core`.
#
# Optional:  CO_LMLM_DIR (defaults to ../Co-LMLM next to this repo; cloned
#            from GitHub if absent), INDEX_DIR, PROMPTS, OUTPUT_DIR,
#            SUITE_PHASES, STANDARD_CLOSURE, SWEEP_CLOSURE,
#            ADVERSARIAL_CLOSURE, RADIUS_GRID, NEIGHBOR_MODE,
#            NEIGHBOR_MIN_COUNT, DEL_OFF_MODE
# Extra flags are passed through to every phase, so keep --limit consistent
# across re-runs: the shared FULL pass is resumed wholesale and only covers
# the facts it was built with.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CO_LMLM_REPO_URL="https://github.com/lil-lab/Co-LMLM.git"
CO_LMLM_DIR="${CO_LMLM_DIR:-$(dirname "$REPO_ROOT")/Co-LMLM}"

INDEX_DIR="${INDEX_DIR:-$REPO_ROOT/data/co-lmlm-wiki-index}"
# T-REx slot-filling is the default audit corpus (in-context prompts, native
# continuation format); use PROMPTS=data/prompts.jsonl for the PopQA set.
PROMPTS="${PROMPTS:-$REPO_ROOT/data/prompts_trex.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/trex}"

# The audit runs from inside the Co-LMLM checkout, so anchor relative
# overrides to the invocation directory before we cd away.
case "$INDEX_DIR" in /*) ;; *) INDEX_DIR="$PWD/$INDEX_DIR" ;; esac
case "$PROMPTS" in /*) ;; *) PROMPTS="$PWD/$PROMPTS" ;; esac
case "$OUTPUT_DIR" in /*) ;; *) OUTPUT_DIR="$PWD/$OUTPUT_DIR" ;; esac

# Radius-dependent evaluations use geometric closure only.
STANDARD_CLOSURE="${STANDARD_CLOSURE:-${CLOSURE:-geometric,value}}"
SWEEP_CLOSURE="${SWEEP_CLOSURE:-geometric}"
ADVERSARIAL_CLOSURE="${ADVERSARIAL_CLOSURE:-geometric}"
RADIUS_GRID="${RADIUS_GRID:-0.95:0.70:0.05}"
NEIGHBOR_MODE="${NEIGHBOR_MODE:-cosine}"
NEIGHBOR_MIN_COUNT="${NEIGHBOR_MIN_COUNT:-5}"
DEL_OFF_MODE="${DEL_OFF_MODE:-null-retrieval}"

ALL_PHASES="standard,sweep,adversarial,del-off,policy"
case "${SUITE_PHASES:-all}" in
    core) SUITE_PHASES="standard,sweep,adversarial" ;;
    all)  SUITE_PHASES="$ALL_PHASES" ;;
esac

phase_enabled() {
    case ",$SUITE_PHASES," in *",$1,"*) return 0 ;; *) return 1 ;; esac
}

# Reject typos rather than silently skipping a phase the user asked for.
for requested in ${SUITE_PHASES//,/ }; do
    case ",$ALL_PHASES," in
        *",$requested,"*) ;;
        *)
            echo "error: unknown phase '$requested' in SUITE_PHASES" >&2
            echo "       valid phases: $ALL_PHASES (or 'core' / 'all')" >&2
            exit 1
            ;;
    esac
done

PHASE_TOTAL=0
for requested in ${SUITE_PHASES//,/ }; do
    PHASE_TOTAL=$((PHASE_TOTAL + 1))
done
PHASE_INDEX=0

announce() {
    PHASE_INDEX=$((PHASE_INDEX + 1))
    echo "=== Phase $PHASE_INDEX/$PHASE_TOTAL: $1 ==="
}

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
# wheel lib dir in the synced environment; exported so the delegated phases
# inherit it rather than recomputing it.
rapids_libs="$(
    find "$CO_LMLM_DIR"/.venv/lib/python*/site-packages -maxdepth 3 -type d \
        \( -name lib64 -o -name lib \) 2>/dev/null | paste -sd: -
)"
case ":${LD_LIBRARY_PATH:-}:" in
    *":$rapids_libs:"*) ;;
    *)
        if [ -n "$rapids_libs" ]; then
            export LD_LIBRARY_PATH="$rapids_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        fi
        ;;
esac

# faiss needs two kinds of native library that a bare `uv sync` does not make
# loadable: OpenBLAS (a *system* package, absent on minimal cloud images) and
# the RAPIDS/CUDA libs shipped inside the faiss-gpu-cu12-cuvs wheel without an
# RPATH (put on the path by LD_LIBRARY_PATH above). Preflight the import here so
# the suite either self-heals or fails with an actionable message, instead of
# crashing deep in phase 1 behind Co-LMLM's misleading "public release required"
# wrapper (it catches the underlying ImportError and re-raises).
faiss_import_error() {
    # Echo the missing-library line from a failed `import faiss`; empty if it imports.
    uv run python -c "import faiss" 2>&1 >/dev/null \
        | grep -oE 'lib[^:]*\.so[^:]*: cannot open shared object file' | head -n1 || true
}

ensure_faiss_importable() {
    local err
    err="$(faiss_import_error)"
    [ -z "$err" ] && return 0

    # OpenBLAS is a system package, not a wheel; install it if we can.
    if printf '%s' "$err" | grep -q "libopenblas"; then
        if command -v apt-get >/dev/null 2>&1; then
            echo "faiss needs OpenBLAS ($err); installing libopenblas0 ..."
            local sudo=""
            [ "$(id -u)" -ne 0 ] && sudo="sudo"
            $sudo apt-get update && $sudo apt-get install -y libopenblas0
        else
            echo "error: faiss needs OpenBLAS but no apt-get to install it: $err" >&2
            echo "       install it manually (Debian/Ubuntu: apt-get install libopenblas0)" >&2
            exit 1
        fi
        err="$(faiss_import_error)"
        [ -z "$err" ] && return 0
    fi

    # Anything left is a wheel-shipped lib that should already be on
    # LD_LIBRARY_PATH — surface the path so a missing wheel is diagnosable.
    echo "error: faiss cannot load a native library: $err" >&2
    echo "       LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<unset>}" >&2
    echo "       RAPIDS/CUDA libs ship in the faiss-gpu-cu12-cuvs wheel; confirm it" >&2
    echo "       installed (find $CO_LMLM_DIR/.venv -name 'libcuvs.so*') and that its" >&2
    echo "       lib dir is on the path above." >&2
    exit 1
}

echo "Checking faiss native libraries ..."
ensure_faiss_importable

run_audit() {
    PYTHONPATH="$REPO_ROOT/src:src${PYTHONPATH:+:$PYTHONPATH}" \
    uv run python -m halo.run_audit \
        --backend co-lmlm \
        --index-path "$INDEX_DIR" \
        --prompt-files "$PROMPTS" \
        --bootstrap-oracle-from-full \
        --co-lmlm-del-off-mode "$DEL_OFF_MODE" \
        --output-dir "$OUTPUT_DIR" \
        "$@"
}

if phase_enabled standard; then
    announce "standard audit (L(f), R(f), I(f), probe, closure manifests)"
    run_audit --closure "$STANDARD_CLOSURE" "$@"
fi

if phase_enabled sweep; then
    announce "entanglement sweep (operating curves, G(f))"
    run_audit --closure "$SWEEP_CLOSURE" --radius-grid "$RADIUS_GRID" \
        --neighbor-mode "$NEIGHBOR_MODE" \
        --neighbor-min-count "$NEIGHBOR_MIN_COUNT" "$@"
fi

if phase_enabled adversarial; then
    announce "adversarial closure (attack attribution, margin predictor)"
    run_audit --closure "$ADVERSARIAL_CLOSURE" --adversarial "$@"
fi

# The remaining phases delegate to the standalone scripts so the two entry
# points cannot drift apart. Each re-enters the Co-LMLM environment; uv sync
# is a no-op once warm.

if phase_enabled del-off; then
    # Phase 1 is already the "$DEL_OFF_MODE" arm of this comparison, so run
    # only the complementary control when both phases are enabled.
    if phase_enabled standard; then
        case "$DEL_OFF_MODE" in
            null-retrieval) sensitivity_modes="forbid-token" ;;
            *)              sensitivity_modes="null-retrieval" ;;
        esac
        announce "DEL-OFF sensitivity ($sensitivity_modes; $DEL_OFF_MODE arm is phase 1)"
    else
        sensitivity_modes="null-retrieval forbid-token"
        announce "DEL-OFF sensitivity ($sensitivity_modes)"
    fi
    CO_LMLM_DIR="$CO_LMLM_DIR" INDEX_DIR="$INDEX_DIR" PROMPTS="$PROMPTS" \
    OUTPUT_DIR="$OUTPUT_DIR/del_off_sensitivity" \
    STANDARD_CLOSURE="$STANDARD_CLOSURE" DEL_OFF_MODES="$sensitivity_modes" \
        "$REPO_ROOT/scripts/run_del_off_sensitivity_co_lmlm.sh" "$@"
fi

if phase_enabled policy; then
    # Note: if STANDARD_CLOSURE is overridden to match one of the policy
    # labels, that policy is audited twice (once here, once as phase 1).
    announce "deletion-policy matrix (oracle, geometric, value, provenance, hybrid)"
    CO_LMLM_DIR="$CO_LMLM_DIR" INDEX_DIR="$INDEX_DIR" PROMPTS="$PROMPTS" \
    OUTPUT_DIR="$OUTPUT_DIR/policy_matrix" DEL_OFF_MODE="$DEL_OFF_MODE" \
        "$REPO_ROOT/scripts/run_policy_matrix_co_lmlm.sh" "$@"
fi

echo "=== Audit suite complete ($SUITE_PHASES) ==="
