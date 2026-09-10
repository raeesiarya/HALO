#!/usr/bin/env bash
# Set up the NULLs (nulls-wiki-1b) audit's artifacts, all released by the
# authors on 2026-09-10:
#   0. install the litgpt dependency group (the SeqTD checkpoint loader)
#   1. (re)build the audit prompt sets (scripts/build_prompts.sh; seeded, so
#      unchanged sets are byte-identical and nothing is invalidated)
#   2. download the checkpoint  gauravrghosal/NULLS-Wikipedia-Full:final/*
#        -> data/nulls-wikipedia-full/{lit_model.pth,model_config.yaml,tokenizer*}
#      (~12 GB; the repo keeps these under final/, we flatten them because
#      the scheduler and the backend address the directory that holds
#      lit_model.pth directly)
#   3. download the training-time title -> source-index mapping
#        -> data/nulls-title-to-index.pkl   (179 MB)
#      Sinks are keyed on this mapping; it is the authors' artifact and is
#      deliberately never reconstructed (a guessed mapping can silently
#      mis-address sinks).
#   4. download the training corpus  gauravrghosal/wiki_nulls_corpus
#        -> data/nulls-wiki-corpus/train_raw/en/*.parquet   (~11.6 GB)
#      Bijective with the mapping (6,407,814 rows; index = row position).
#      The breadth-k sweep (article texts for the shared-encoder closure
#      artifact) and the value/hybrid policy rows need it; the standard
#      and DEL-OFF phases do not.
#   5. build the shared-encoder closure artifact from those texts
#        -> data/nulls-closure-embeddings.npz   (~20 GB, GPU-hours)
#      The cross-model scheduler adds the sweep phases when this file
#      exists; the routing-space artifact (titles, MiniLM) is built by the
#      scheduler itself as a prep job.
#
# Every step is idempotent and resumable. Knobs:
#   NULLS_CHECKPOINT_DIR / NULLS_TITLE_TO_INDEX / NULLS_CORPUS_DIR /
#   NULLS_CLOSURE_EMBEDDINGS   artifact locations (defaults above; the
#                              scheduler reads the same variables)
#   SKIP_CORPUS=1              skip step 4 (and therefore 5)
#   SKIP_CLOSURE=1             skip step 5 — the audit's standard/DEL-OFF
#                              phases can start right away; re-run this
#                              script (or the build) later and re-submit the
#                              suite to add the sweep
#   SKIP_PROMPTS=1             skip step 1
#   HF_TOKEN                   if a repo is gated
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

NULLS_REPO="gauravrghosal/NULLS-Wikipedia-Full"
CORPUS_REPO="gauravrghosal/wiki_nulls_corpus"
NULLS_CHECKPOINT_DIR="${NULLS_CHECKPOINT_DIR:-$REPO_ROOT/data/nulls-wikipedia-full}"
NULLS_TITLE_TO_INDEX="${NULLS_TITLE_TO_INDEX:-$REPO_ROOT/data/nulls-title-to-index.pkl}"
NULLS_CORPUS_DIR="${NULLS_CORPUS_DIR:-$REPO_ROOT/data/nulls-wiki-corpus}"
NULLS_CLOSURE_EMBEDDINGS="${NULLS_CLOSURE_EMBEDDINGS:-$REPO_ROOT/data/nulls-closure-embeddings.npz}"
# Modern `hf` CLI, isolated via uvx (the repo's pinned huggingface_hub is
# too old for it; no effect on the audit environment).
HF_CLI=(uvx --from "huggingface_hub[hf_xet]>=1.25" hf)
# Empty-array expansion is an "unbound variable" under set -u on bash < 4.4
# (macOS ships 3.2), hence the ${arr[@]+"${arr[@]}"} idiom below.
TOKEN_ARGS=(${HF_TOKEN:+--token "$HF_TOKEN"})

echo "[0/5] Syncing the environment (litgpt is a default dependency group) ..."
uv sync

# Always rebuild: the prep scripts are seeded (unchanged sets come out
# byte-identical, so nothing is invalidated), and a stale prompt file is a
# silent failure here — PopQA sets built before 2026-09-11 carry no subject
# and every PopQA fact would be dropped by prep-augment.
echo "[1/5] Building audit prompts ..."
"$REPO_ROOT/scripts/build_prompts.sh"

if [ -f "$NULLS_CHECKPOINT_DIR/lit_model.pth" ]; then
    echo "[2/5] NULLs checkpoint already present at $NULLS_CHECKPOINT_DIR"
else
    echo "[2/5] Downloading NULLs checkpoint $NULLS_REPO:final/* -> $NULLS_CHECKPOINT_DIR (~12 GB) ..."
    mkdir -p "$NULLS_CHECKPOINT_DIR"
    "${HF_CLI[@]}" download "$NULLS_REPO" ${TOKEN_ARGS[@]+"${TOKEN_ARGS[@]}"} \
        --include 'final/*' --local-dir "$NULLS_CHECKPOINT_DIR"
fi
# Flatten final/ -> checkpoint dir (also completes an interrupted earlier
# flatten: everything still under final/ moves up).
if [ -d "$NULLS_CHECKPOINT_DIR/final" ]; then
    echo "      flattening $NULLS_CHECKPOINT_DIR/final/ -> $NULLS_CHECKPOINT_DIR/"
    for f in "$NULLS_CHECKPOINT_DIR"/final/*; do
        [ -e "$f" ] && mv -f "$f" "$NULLS_CHECKPOINT_DIR/"
    done
    rmdir "$NULLS_CHECKPOINT_DIR/final"
fi
for required in lit_model.pth model_config.yaml tokenizer.json tokenizer_config.json; do
    if [ ! -f "$NULLS_CHECKPOINT_DIR/$required" ]; then
        echo "ERROR: $NULLS_CHECKPOINT_DIR/$required missing after download." >&2
        exit 1
    fi
done

if [ -f "$NULLS_TITLE_TO_INDEX" ]; then
    echo "[3/5] title_to_index.pkl already present at $NULLS_TITLE_TO_INDEX"
else
    echo "[3/5] Downloading title_to_index.pkl -> $NULLS_TITLE_TO_INDEX (179 MB) ..."
    staging="$(dirname "$NULLS_TITLE_TO_INDEX")/.nulls-download"
    mkdir -p "$staging"
    "${HF_CLI[@]}" download "$NULLS_REPO" title_to_index.pkl ${TOKEN_ARGS[@]+"${TOKEN_ARGS[@]}"} \
        --local-dir "$staging"
    mv -f "$staging/title_to_index.pkl" "$NULLS_TITLE_TO_INDEX"
    rm -rf "$staging"
fi

if [ "${SKIP_CORPUS:-0}" = "1" ]; then
    echo "[4/5] SKIP_CORPUS=1 — skipping the training corpus (breadth-k sweep needs it)."
else
    echo "[4/5] Downloading training corpus $CORPUS_REPO -> $NULLS_CORPUS_DIR (~11.6 GB) ..."
    mkdir -p "$NULLS_CORPUS_DIR"
    # Resumable: files already present are skipped.
    "${HF_CLI[@]}" download "$CORPUS_REPO" --repo-type dataset ${TOKEN_ARGS[@]+"${TOKEN_ARGS[@]}"} \
        --include 'train_raw/en/*' --local-dir "$NULLS_CORPUS_DIR"
fi

if [ "${SKIP_CLOSURE:-0}" = "1" ]; then
    echo "[5/5] SKIP_CLOSURE=1 — not building $NULLS_CLOSURE_EMBEDDINGS."
    echo "      The suite runs the standard and DEL-OFF phases without it;"
    echo "      build it later and re-submit the suite to add the sweep."
elif [ -f "$NULLS_CLOSURE_EMBEDDINGS" ]; then
    echo "[5/5] Closure artifact already present at $NULLS_CLOSURE_EMBEDDINGS"
elif [ ! -d "$NULLS_CORPUS_DIR/train_raw/en" ]; then
    echo "[5/5] Corpus absent — cannot build the closure artifact (SKIP_CORPUS=1?)."
else
    echo "[5/5] Building shared-encoder closure embeddings -> $NULLS_CLOSURE_EMBEDDINGS"
    echo "      (6.4M article texts through all-mpnet-base-v2: GPU-hours; the"
    echo "       output is ~20 GB. SKIP_CLOSURE=1 to defer.)"
    uv run python "$REPO_ROOT/scripts/build_nulls_title_embeddings.py" \
        --mode closure \
        --title-to-index "$NULLS_TITLE_TO_INDEX" \
        --texts "$NULLS_CORPUS_DIR/train_raw/en" \
        --output "$NULLS_CLOSURE_EMBEDDINGS"
fi

echo
echo "Done."
echo "  Checkpoint:      $NULLS_CHECKPOINT_DIR"
echo "  title_to_index:  $NULLS_TITLE_TO_INDEX"
echo "  Corpus:          $NULLS_CORPUS_DIR"
echo "  Closure artifact: $NULLS_CLOSURE_EMBEDDINGS$([ -f "$NULLS_CLOSURE_EMBEDDINGS" ] || echo ' (absent: no sweep phase)')"
echo "Run everything (all models, NULLs prep chain included) with:"
echo "  ./scripts/run_suite_parallel_cross_model.sh"
echo "NULLs alone, fast smoke check:"
echo "  MODELS=nulls-wiki-1b NULLS_PHASES=standard,del-off ./scripts/run_suite_parallel_cross_model.sh --limit 200"
