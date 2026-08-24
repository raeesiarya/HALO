"""Build source-embedding artifacts for the NULLs audit.

Two distinct spaces, two modes (docs/NULLS_AUDIT_DESIGN.md §4):

- ``--mode routing`` (default): embeddings of article *titles* with
  ``all-MiniLM-L6-v2`` — a faithful reproduction of upstream's next-closest
  routing space (`compute_truth_ratio.py`), consumed by the backend for
  DEL-ON routing and placebo selection.
- ``--mode closure``: embeddings of article *text* with the shared external
  encoder E that defines cross-substrate closure neighborhoods. Requires
  ``--texts``, a jsonl/parquet with ``title`` and ``text`` columns from the
  training dump. Never use either model's own representations here.

Output: ``.npz`` with ``embeddings`` (float32, row-aligned with
``title_to_index`` insertion order), ``encoder``, and ``mode``.

Usage:
  uv run python scripts/build_nulls_title_embeddings.py \
      --title-to-index data/nulls-title-to-index.pkl \
      --output data/nulls-title-embeddings.npz
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

ROUTING_ENCODER = "all-MiniLM-L6-v2"  # upstream's routing space
CLOSURE_ENCODER = "sentence-transformers/all-mpnet-base-v2"  # shared E, design §4


def _load_texts(path: Path) -> dict[str, str]:
    if path.suffix == ".parquet":
        import pandas as pd

        frame = pd.read_parquet(path, columns=["title", "text"])
        return dict(zip(frame["title"], frame["text"]))
    import json

    texts: dict[str, str] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            texts[row["title"]] = row["text"]
    return texts


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
        help="closure mode: jsonl/parquet with 'title' and 'text' columns; "
        "the first --max-text-chars of each text are encoded.",
    )
    parser.add_argument("--max-text-chars", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()

    encoder_name = args.encoder or (
        ROUTING_ENCODER if args.mode == "routing" else CLOSURE_ENCODER
    )
    with open(args.title_to_index, "rb") as handle:
        title_to_index = pickle.load(handle)
    titles = list(title_to_index)

    if args.mode == "routing":
        inputs = titles
    else:
        if args.texts is None:
            raise SystemExit("--mode closure requires --texts (title/text corpus).")
        texts = _load_texts(args.texts)
        missing = [title for title in titles if title not in texts]
        if missing:
            raise SystemExit(
                f"{len(missing)} titles have no text in {args.texts} "
                f"(first: {missing[:3]!r}); closure embeddings must cover "
                "every source or neighborhoods are silently biased."
            )
        inputs = [texts[title][: args.max_text_chars] for title in titles]

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(encoder_name)
    embeddings = model.encode(
        inputs,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        embeddings=embeddings,
        encoder=np.str_(encoder_name),
        mode=np.str_(args.mode),
    )
    print(f"wrote {embeddings.shape} embeddings ({encoder_name}) to {args.output}")


if __name__ == "__main__":
    main()
