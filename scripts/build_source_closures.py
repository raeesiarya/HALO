"""Materialize source-level closure prompt files (design §4-5).

Reads an augmented prompt set (scripts/augment_source_titles.py), expands
each fact's deletion manifest under a source-level predicate in the shared
encoder space E, and writes one prompt file per breadth k. The same files
drive both substrates: nulls-wiki-1b executes the manifests as sink
exclusions, co-lmlm as search-time source filtering.

The k-grid IS the sweep axis (matched-collateral curves are computed by the
analysis over the per-k audit outputs); k=0 is the provenance anchor and is
identical to the augmented input's manifests.

Usage:
  uv run python scripts/build_source_closures.py \
      --prompts data/prompts_trex_nulls.jsonl \
      --title-to-index data/nulls-title-to-index.pkl \
      --closure-embeddings data/nulls-closure-embeddings.npz \
      --predicate geometric --k-grid 1,2,4,8,16 \
      --output-dir data/source_closures/trex
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from halo.interventions.source_closure import (  # noqa: E402
    SOURCE_PREDICATES,
    SourceSpace,
    build_closure_prompt_file,
)


def _texts_lookup(path: Path | None):
    if path is None:
        return None
    import pandas as pd

    frame = pd.read_parquet(path, columns=["title", "text"])
    table = dict(zip(frame["title"], frame["text"]))
    return table.get


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--title-to-index", type=Path, required=True)
    parser.add_argument(
        "--closure-embeddings",
        type=Path,
        required=True,
        help="Shared-E artifact from build_nulls_title_embeddings.py "
        "--mode closure (routing-space artifacts are rejected).",
    )
    parser.add_argument("--predicate", choices=SOURCE_PREDICATES, default="geometric")
    parser.add_argument("--k-grid", default="1,2,4,8,16")
    parser.add_argument("--envelope-k", type=int, default=500)
    parser.add_argument(
        "--texts",
        type=Path,
        default=None,
        help="Parquet with title/text columns; required for value/hybrid.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    with open(args.title_to_index, "rb") as handle:
        titles = list(pickle.load(handle))
    space = SourceSpace.load(titles=titles, embeddings_path=args.closure_embeddings)
    source_texts = _texts_lookup(args.texts)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.prompts.stem
    reports = []
    for k_text in args.k_grid.split(","):
        k = int(k_text)
        output = args.output_dir / f"{stem}_{args.predicate}_k{k}.jsonl"
        report = build_closure_prompt_file(
            args.prompts,
            output,
            space=space,
            predicate=args.predicate if k > 0 else "provenance",
            k=k,
            envelope_k=args.envelope_k,
            source_texts=source_texts,
        )
        reports.append(report)
        print(json.dumps(report))
    (args.output_dir / f"{stem}_{args.predicate}_report.json").write_text(
        json.dumps(reports, indent=2)
    )


if __name__ == "__main__":
    main()
