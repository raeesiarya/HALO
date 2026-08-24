#!/usr/bin/env bash
# Set up the cross-model audit's data, all in one:
#   1. build the audit prompt sets:
#        T-REx        -> data/prompts_trex.jsonl        (default corpus)
#        PopQA        -> data/prompts.jsonl
#        Google-RE    -> data/prompts_googlere.jsonl
#        CounterFact  -> data/prompts_counterfact.jsonl
#        ZsRE         -> data/prompts_zsre.jsonl
#   2. download the fineweb+wiki index bucket
#        -> data/co-lmlm-fineweb-wiki-index  (~1.05 TB!)
#   3. download the NULLs Wikipedia checkpoint
#        -> data/nulls-wikipedia-full        (~12 GB)
#
# The Co-LMLM model is NOT downloaded here: the audit's loader fetches it from
# Hugging Face on first use and it is cached by transformers. The index, being
# a custom FAISS+sqlite artifact, must be local — as must the NULLs checkpoint
# (litgpt layout; plain transformers cannot load it).
#
# The NULLs audit additionally needs data/nulls-title-to-index.pkl, the
# training-time title -> source-index mapping. It is NOT publicly released as
# of 2026-08-24 and is deliberately not reconstructable here: a guessed
# mapping can pass spot checks while silently mis-addressing sinks. Request
# it from the authors (gghosal@andrew.cmu.edu, github.com/AR-FORUM/NULLS;
# the gauravrghosal/wikipedia_full_8x HF repo may come to carry it).
# Everything downstream of it (title embeddings, augmented prompts, the
# verification gate, closures) is built automatically as jobs by the
# cross-model scheduler.
#
# Override locations with INDEX_DIR=/path NULLS_CHECKPOINT_DIR=/path.
# Set HF_TOKEN if a bucket is gated. SKIP_INDEX=1 / SKIP_NULLS=1 skip the
# respective downloads.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

INDEX_REPO="lil-lab/co-lmlm-360m-fw-fineweb-wiki-index"
INDEX_DIR="${INDEX_DIR:-$REPO_ROOT/data/co-lmlm-fineweb-wiki-index}"
# Released bucket files (faiss.index ~228 GB, fineweb_with_fullwiki_entries.db
# ~713 GB, faiss_id_to_entry_id.db ~134 GB): ~1.05 TB total.
INDEX_FILES=(faiss.index fineweb_with_fullwiki_entries.db faiss_id_to_entry_id.db index_config.json manifest.json)
# The index lives in a Xet-backed HF bucket. Xet reconstructs files from
# content-addressed chunks and does not serve plain HTTP range requests, so
# curl -C - (resume) gets a 400 — the download must go through the Xet-aware
# `hf buckets` CLI. huggingface_hub in this repo is too old to have it, so run
# a recent one isolated via uvx (no effect on the audit environment's pins).
HF_CLI=(uvx --from "huggingface_hub[hf_xet]>=1.25" hf)

NULLS_REPO="gauravrghosal/NULLS-Wikipedia-Full"
NULLS_CHECKPOINT_DIR="${NULLS_CHECKPOINT_DIR:-$REPO_ROOT/data/nulls-wikipedia-full}"

echo "[1/3] Building audit prompts (T-REx default corpus, then the rest) ..."
uv run python "$REPO_ROOT/data/prepare_trex_audit.py"
uv run python "$REPO_ROOT/data/prepare_popqa_audit.py"
uv run python "$REPO_ROOT/data/prepare_googlere_audit.py"
uv run python "$REPO_ROOT/data/prepare_counterfact_audit.py"
uv run python "$REPO_ROOT/data/prepare_zsre_audit.py"

if [ "${SKIP_INDEX:-0}" = "1" ]; then
    echo "[2/3] SKIP_INDEX=1 — skipping the fineweb+wiki index download."
else
    echo "[2/3] Downloading fineweb+wiki index $INDEX_REPO -> $INDEX_DIR (~1.05 TB) ..."
    mkdir -p "$INDEX_DIR"
    for file in "${INDEX_FILES[@]}"; do
        echo "  -> $file"
        # Re-running is safe and resumes: the Xet client dedupes chunks it
        # already has, so an interrupted download picks up where it left off.
        "${HF_CLI[@]}" buckets cp ${HF_TOKEN:+--token "$HF_TOKEN"} \
            "hf://buckets/$INDEX_REPO/$file" "$INDEX_DIR/$file"
    done
fi

if [ "${SKIP_NULLS:-0}" = "1" ]; then
    echo "[3/3] SKIP_NULLS=1 — skipping the NULLs checkpoint download."
elif [ -f "$NULLS_CHECKPOINT_DIR/lit_model.pth" ]; then
    echo "[3/3] NULLs checkpoint already present at $NULLS_CHECKPOINT_DIR"
else
    echo "[3/3] Downloading NULLs checkpoint $NULLS_REPO -> $NULLS_CHECKPOINT_DIR (~12 GB) ..."
    uv run huggingface-cli download "$NULLS_REPO" --local-dir "$NULLS_CHECKPOINT_DIR"
fi

if [ ! -f "$REPO_ROOT/data/nulls-title-to-index.pkl" ]; then
    echo
    echo "NOTE: data/nulls-title-to-index.pkl is missing (see the header of this"
    echo "      script). The NULLs jobs in the cross-model suite need it; the"
    echo "      Co-LMLM and closed-book baselines run without it."
fi

echo
echo "Done."
echo "  Prompts:          $REPO_ROOT/data/prompts_trex.jsonl (default), plus"
echo "                    prompts.jsonl (PopQA), prompts_googlere.jsonl,"
echo "                    prompts_counterfact.jsonl, prompts_zsre.jsonl"
echo "  Index:            $INDEX_DIR"
echo "  NULLs checkpoint: $NULLS_CHECKPOINT_DIR"
echo "Run everything (all models, all phases, NULLs prep included) with:"
echo "  ./scripts/run_suite_parallel_cross_model.sh"
