#!/usr/bin/env bash
# Set up a Co-LMLM audit run:
#   1. build the audit prompt sets:
#        T-REx        -> data/prompts_trex.jsonl        (default corpus)
#        PopQA        -> data/prompts.jsonl
#        Google-RE    -> data/prompts_googlere.jsonl
#        CounterFact  -> data/prompts_counterfact.jsonl
#        ZsRE         -> data/prompts_zsre.jsonl
#   2. download the fineweb+wiki index bucket
#        -> data/co-lmlm-fineweb-wiki-index  (~1.05 TB!)
#
# The model is NOT downloaded here: the audit's loader fetches it from Hugging
# Face on first use (the loader fetches the released model and
# is cached by transformers). The index, being a custom FAISS+sqlite artifact,
# must be local — that is what this script fetches.
#
# Override the index location with INDEX_DIR=/path ./scripts/setup_data.sh
# Set HF_TOKEN if the bucket is gated.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

INDEX_REPO="lil-lab/co-lmlm-360m-fw-fineweb-wiki-index"
INDEX_DIR="${INDEX_DIR:-$REPO_ROOT/data/co-lmlm-fineweb-wiki-index}"
# Released bucket files (faiss.index ~228 GB, fineweb_with_fullwiki_entries.db
# ~713 GB, faiss_id_to_entry_id.db ~134 GB): ~1.05 TB total.
INDEX_FILES=(faiss.index fineweb_with_fullwiki_entries.db faiss_id_to_entry_id.db index_config.json manifest.json)
INDEX_BASE_URL="https://huggingface.co/buckets/$INDEX_REPO/resolve"

echo "[1/2] Building audit prompts (T-REx default corpus, then the rest) ..."
uv run python "$REPO_ROOT/data/prepare_trex_audit.py"
uv run python "$REPO_ROOT/data/prepare_popqa_audit.py"
uv run python "$REPO_ROOT/data/prepare_googlere_audit.py"
uv run python "$REPO_ROOT/data/prepare_counterfact_audit.py"
uv run python "$REPO_ROOT/data/prepare_zsre_audit.py"

echo "[2/2] Downloading fineweb+wiki index $INDEX_REPO -> $INDEX_DIR (~1.05 TB) ..."
mkdir -p "$INDEX_DIR"
for file in "${INDEX_FILES[@]}"; do
    echo "  -> $file"
    curl -L --fail --retry 3 -C - -o "$INDEX_DIR/$file" \
        ${HF_TOKEN:+-H "Authorization: Bearer $HF_TOKEN"} \
        "$INDEX_BASE_URL/$file"
done

echo
echo "Done."
echo "  Prompts: $REPO_ROOT/data/prompts_trex.jsonl (default), plus"
echo "           prompts.jsonl (PopQA), prompts_googlere.jsonl, prompts_counterfact.jsonl,"
echo "           prompts_zsre.jsonl"
echo "  Index:   $INDEX_DIR"
echo "Run the audit with (the model is fetched automatically):"
echo "  halo-audit --backend co-lmlm --index-path $INDEX_DIR \\"
echo "    --prompt-files $REPO_ROOT/data/prompts_trex.jsonl --bootstrap-oracle-from-full \\"
echo "    --output-dir outputs/trex"
