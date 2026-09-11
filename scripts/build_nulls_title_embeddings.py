"""Build source-embedding artifacts for the NULLs audit.

Two distinct spaces, two modes (docs/NULLS_AUDIT_DESIGN.md §4):

- ``--mode routing`` (default): embeddings of article *titles* with
  ``all-MiniLM-L6-v2`` — a faithful reproduction of upstream's next-closest
  routing space (`compute_truth_ratio.py`), consumed by the backend for
  DEL-ON routing and placebo selection. The cross-model scheduler runs this
  as a shared prep job.
- ``--mode closure``: embeddings of article *text* with the shared external
  encoder E that defines cross-substrate closure neighborhoods. ``--texts``
  is the released training corpus (``gauravrghosal/wiki_nulls_corpus``,
  downloaded by ``scripts/setup_nulls.sh``): a directory of parquet shards,
  one parquet file, or a jsonl — anything with ``title`` and ``text``
  columns. Shards stream one row batch at a time and each text is cut to
  ``--max-text-chars`` before it is held, so memory is the output matrix
  plus one batch (never the corpus). Never use either model's own
  representations here.

Output: ``.npz`` written uncompressed — float32 embeddings do not compress,
and every consumer would otherwise decompress ~10-20 GB on load — with
``embeddings`` (float32, L2-normalized rows, row-aligned with
``title_to_index`` insertion order), ``encoder``, ``mode`` and ``dtype`` (the
encoder's compute precision; storage is always float32). Closure mode
computes in bf16 by default — ~6.4M article texts in fp32 take ~19 h on a
GB10 — while routing mode stays fp32 to match upstream's routing space.

Usage:
  uv run python scripts/build_nulls_title_embeddings.py \
      --title-to-index data/nulls-title-to-index.pkl \
      --output data/nulls-title-embeddings.npz
  uv run python scripts/build_nulls_title_embeddings.py --mode closure \
      --title-to-index data/nulls-title-to-index.pkl \
      --texts data/nulls-wiki-corpus/train_raw/en \
      --output data/nulls-closure-embeddings.npz
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
from tqdm import tqdm

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from models.nulls_wiki.corpus import iter_text_batches as _iter_corpus_batches  # noqa: E402

ROUTING_ENCODER = "all-MiniLM-L6-v2"  # upstream's routing space
CLOSURE_ENCODER = "sentence-transformers/all-mpnet-base-v2"  # shared E, design §4
DEFAULT_BATCH = {"routing": 1024, "closure": 256}
DEFAULT_DTYPE = {"routing": "fp32", "closure": "bf16"}
TORCH_DTYPES = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}
PARQUET_ROWS_PER_BATCH = 8192


def load_title_positions(path: Path) -> dict[str, int]:
    """title -> row position in ``title_to_index`` insertion order.

    Rows are aligned with insertion order (the artifact contract), not with
    the mapping's integer values — the released mapping has both identical,
    but the contract is what the consumers rely on.
    """
    with open(path, "rb") as handle:
        title_to_index = pickle.load(handle)
    return {title: position for position, title in enumerate(title_to_index)}


def iter_text_batches(
    path: Path, *, max_chars: int, rows_per_batch: int = PARQUET_ROWS_PER_BATCH
) -> Iterator[tuple[list[str], list[str]]]:
    """(titles, truncated texts) batches — models.nulls_wiki.corpus, shared
    with the value-source closure builder."""
    return _iter_corpus_batches(path, max_chars=max_chars, rows_per_batch=rows_per_batch)


def encode_closure(
    positions: dict[str, int],
    batches: Iterator[tuple[list[str], list[str]]],
    *,
    encoder,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Stream the corpus into a preallocated (titles x dim) matrix.

    Returns the matrix and a boolean coverage vector (which rows received a
    text). Corpus rows whose title is not in the mapping are ignored.
    """
    dim = int(encoder.get_sentence_embedding_dimension())
    embeddings = np.zeros((len(positions), dim), dtype=np.float32)
    covered = np.zeros(len(positions), dtype=bool)
    # The released corpus is bijective with the mapping, so the source count
    # is also the corpus row count the bar runs to.
    with tqdm(total=len(positions), unit="rows", unit_scale=True, desc="closure") as bar:
        for titles, texts in batches:
            rows, kept_texts = [], []
            for title, text in zip(titles, texts):
                position = positions.get(title)
                if position is None:
                    continue
                rows.append(position)
                kept_texts.append(text)
            if rows:
                vectors = encoder.encode(
                    kept_texts,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
                embeddings[rows] = vectors.astype(np.float32, copy=False)
                covered[rows] = True
            bar.update(len(titles))
            bar.set_postfix(covered=f"{int(covered.sum()):,}", refresh=False)
    return embeddings, covered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--title-to-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("routing", "closure"), default="routing")
    parser.add_argument(
        "--encoder",
        default=None,
        help="Override the encoder (defaults: routing=all-MiniLM-L6-v2 to "
        "match upstream; closure=all-mpnet-base-v2 as the shared E).",
    )
    parser.add_argument(
        "--texts",
        type=Path,
        default=None,
        help="closure mode: directory of parquet shards, one parquet file, "
        "or a jsonl, with 'title' and 'text' columns.",
    )
    parser.add_argument("--max-text-chars", type=int, default=2000)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=f"encoder batch size (defaults: {DEFAULT_BATCH}).",
    )
    parser.add_argument(
        "--dtype",
        choices=tuple(TORCH_DTYPES),
        default=None,
        help=f"encoder compute precision (defaults: {DEFAULT_DTYPE}); the "
        "stored embeddings are float32 either way.",
    )
    args = parser.parse_args()

    encoder_name = args.encoder or (
        ROUTING_ENCODER if args.mode == "routing" else CLOSURE_ENCODER
    )
    batch_size = args.batch_size or DEFAULT_BATCH[args.mode]
    dtype = args.dtype or DEFAULT_DTYPE[args.mode]
    positions = load_title_positions(args.title_to_index)
    print(
        f"{len(positions):,} sources; encoder {encoder_name}; mode {args.mode}; "
        f"dtype {dtype}"
    )

    import torch
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(
        encoder_name, model_kwargs={"torch_dtype": getattr(torch, TORCH_DTYPES[dtype])}
    )

    if args.mode == "routing":
        titles = list(positions)
        embeddings = encoder.encode(
            titles,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32, copy=False)
    else:
        if args.texts is None:
            raise SystemExit("--mode closure requires --texts (title/text corpus).")
        embeddings, covered = encode_closure(
            positions,
            iter_text_batches(args.texts, max_chars=args.max_text_chars),
            encoder=encoder,
            batch_size=batch_size,
        )
        if not covered.all():
            missing = [
                title for title, position in positions.items() if not covered[position]
            ]
            report = args.output.with_suffix(".missing.txt")
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text("\n".join(missing) + "\n", encoding="utf-8")
            raise SystemExit(
                f"{len(missing):,} of {len(positions):,} sources have no text in "
                f"{args.texts} (list: {report}); closure embeddings must cover "
                "every source or neighborhoods are silently biased."
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        embeddings=embeddings,
        encoder=np.str_(encoder_name),
        mode=np.str_(args.mode),
        dtype=np.str_(dtype),
    )
    print(
        f"wrote {embeddings.shape} embeddings ({encoder_name}, {dtype}) to {args.output}"
    )


if __name__ == "__main__":
    main()
