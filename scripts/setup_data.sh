#!/usr/bin/env bash
# Set up a PopQA Co-LMLM audit run:
#   1. build the PopQA audit prompts   -> data/prompts.jsonl
#   2. download the wiki index bucket  -> data/co-lmlm-wiki-index  (~113 GB!)
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

INDEX_REPO="lil-lab/co-lmlm-360m-fw-wiki-index"
INDEX_DIR="${INDEX_DIR:-$REPO_ROOT/data/co-lmlm-wiki-index}"
# Released bucket files (faiss.index ~59 GB, entries.db ~49 GB, mapping ~5.5 GB).
INDEX_FILES=(faiss.index entries.db faiss_id_to_entry_id.txt index_config.json manifest.json)
INDEX_BASE_URL="https://huggingface.co/buckets/$INDEX_REPO/resolve"

echo "[1/2] Building PopQA audit prompts ..."
uv run python "$REPO_ROOT/data/prepare_popqa_audit.py"

echo "[2/2] Downloading wiki index $INDEX_REPO -> $INDEX_DIR (~113 GB) ..."
mkdir -p "$INDEX_DIR"
for file in "${INDEX_FILES[@]}"; do
    echo "  -> $file"
    curl -L --fail --retry 3 -C - -o "$INDEX_DIR/$file" \
        ${HF_TOKEN:+-H "Authorization: Bearer $HF_TOKEN"} \
        "$INDEX_BASE_URL/$file"
done

echo
echo "Done."
echo "  Prompts: $REPO_ROOT/data/prompts.jsonl"
echo "  Index:   $INDEX_DIR"
echo "Run the audit with (the model is fetched automatically):"
echo "  halo-audit --backend co-lmlm --index-path $INDEX_DIR \\"
echo "    --prompt-files $REPO_ROOT/data/prompts.jsonl --bootstrap-oracle-from-full \\"
echo "    --output-dir outputs/popqa"
