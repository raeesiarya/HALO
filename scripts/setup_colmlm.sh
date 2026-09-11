#!/usr/bin/env bash
# Set up the Co-LMLM audit's data:
#   1. build the audit prompt sets (scripts/build_prompts.sh)
#   2. download the fineweb+wiki index bucket
#        -> data/co-lmlm-fineweb-wiki-index  (~1.05 TB!)
#
# The Co-LMLM model is NOT downloaded here: the audit's loader fetches it from
# Hugging Face on first use and it is cached by transformers. The index, being
# a custom FAISS+sqlite artifact, must be local.
#
# The NULLs artifacts have their own script (scripts/setup_nulls.sh);
# scripts/setup_data.sh runs both.
#
# Override the index location with INDEX_DIR=/path. Set HF_TOKEN if the
# bucket is gated. SKIP_INDEX=1 skips the download, SKIP_PROMPTS=1 the
# prompt build. By default the script detaches and logs to
# logs/setup_colmlm.log (tail -F it); HALO_SETUP_FOREGROUND=1 runs attached.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/_detach.sh
source "$REPO_ROOT/scripts/_detach.sh"
detach_setup setup_colmlm "$@"

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

echo "[1/2] Building audit prompts ..."
"$REPO_ROOT/scripts/build_prompts.sh"

if [ "${SKIP_INDEX:-0}" = "1" ]; then
    echo "[2/2] SKIP_INDEX=1 — skipping the fineweb+wiki index download."
else
    echo "[2/2] Downloading fineweb+wiki index $INDEX_REPO -> $INDEX_DIR (~1.05 TB) ..."
    mkdir -p "$INDEX_DIR"
    for file in "${INDEX_FILES[@]}"; do
        echo "  -> $file"
        # Re-running is safe and resumes: the Xet client dedupes chunks it
        # already has, so an interrupted download picks up where it left off.
        "${HF_CLI[@]}" buckets cp ${HF_TOKEN:+--token "$HF_TOKEN"} \
            "hf://buckets/$INDEX_REPO/$file" "$INDEX_DIR/$file"
    done
fi

echo
echo "Done."
echo "  Prompts: $REPO_ROOT/data/prompts_trex.jsonl (default), plus"
echo "           prompts.jsonl (PopQA), prompts_googlere.jsonl,"
echo "           prompts_counterfact.jsonl, prompts_zsre.jsonl"
echo "  Index:   $INDEX_DIR"
echo "Run the Co-LMLM suite with ./scripts/run_audit_suite_co_lmlm.sh, or"
echo "everything (all models) with ./scripts/run_suite_parallel_cross_model.sh"
echo "after ./scripts/setup_nulls.sh."
