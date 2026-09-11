"""Materialize source-level closure prompt files (design §4-5, §7).

Reads an augmented (gated) prompt set, expands each fact's deletion manifest
under one or more source-level predicates in the shared encoder space E,
and writes one prompt file per (predicate, breadth k). The same files drive
both substrates: nulls-wiki-1b executes the manifests as sink exclusions,
co-lmlm as search-time source filtering.

- ``geometric``: gold + k nearest sources; the k-grid IS the sweep axis.
- ``provenance``: gold only (k = 0), identical to the augmented input.
- ``value``: gold + envelope sources whose *text* mentions the answer — the
  oracle evaluation rule of the policy matrix (§7); needs ``--texts``.
- ``hybrid``: gold + (geometric-k ∩ value).

Cost structure: neighbors are computed once for every distinct gold source
as blocked matrix products (``SourceSpace.nearest_many``), and the answer-
mention check for value/hybrid is ONE streaming pass over the corpus that
keeps only the hits — the corpus is never held in memory. Several
predicates in one invocation share both.

Usage:
  uv run python scripts/build_source_closures.py \\
      --prompts <gated>.jsonl \\
      --title-to-index data/nulls-title-to-index.pkl \\
      --closure-embeddings data/nulls-closure-embeddings.npz \\
      --predicate geometric --k-grid 1,2,4,8,16 --output-dir <out>
  uv run python scripts/build_source_closures.py ... \\
      --predicate value,hybrid --k-grid 4 --texts data/nulls-wiki-corpus \\
      --output-dir <out>
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from halo.core.equivalence import normalize_text  # noqa: E402
from halo.interventions.source_closure import (  # noqa: E402
    SOURCE_PREDICATES,
    SourceSpace,
    answer_mentioned,
    build_closure_prompt_file,
    iter_prompt_rows,
    normalized_answer_aliases,
)
from models.nulls_wiki.corpus import iter_text_batches  # noqa: E402


def load_rows(prompts: Path) -> list[dict]:
    with open(prompts, encoding="utf-8") as handle:
        return list(iter_prompt_rows(handle))


def stream_value_hits(
    corpus: Path,
    rows: Sequence[Mapping],
    envelopes: Mapping[str, Sequence[tuple[str, float]]],
    *,
    envelope_k: int,
    log_every: int = 200,
) -> dict[int, set[str]]:
    """row index -> envelope titles whose text mentions the row's answer.

    One pass over the corpus: an article is normalized once and checked
    against every fact whose envelope contains it; nothing but the hits is
    retained.
    """
    needed: dict[str, list[int]] = defaultdict(list)
    aliases: dict[int, tuple[str, ...]] = {}
    for index, row in enumerate(rows):
        gold = row.get("source_title")
        if gold not in envelopes:
            continue
        aliases[index] = normalized_answer_aliases(row)
        for title, _ in envelopes[gold][:envelope_k]:
            needed[title].append(index)
    hits: dict[int, set[str]] = defaultdict(set)
    seen_titles = 0
    started = time.time()
    for batch_index, (titles, texts) in enumerate(
        iter_text_batches(corpus, max_chars=None), start=1
    ):
        for title, text in zip(titles, texts):
            facts = needed.get(title)
            if not facts:
                continue
            seen_titles += 1
            normalized = normalize_text(text)
            for index in facts:
                if answer_mentioned(normalized, aliases[index]):
                    hits[index].add(title)
        if batch_index % log_every == 0:
            print(
                f"  corpus pass: {seen_titles:,}/{len(needed):,} envelope sources "
                f"seen, {sum(len(h) for h in hits.values()):,} hits "
                f"({time.time() - started:.0f}s)",
                flush=True,
            )
    missing = len(needed) - seen_titles
    if missing:
        print(f"  note: {missing:,} envelope sources have no text in {corpus}", flush=True)
    return hits


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
    parser.add_argument(
        "--predicate",
        default="geometric",
        help=f"comma-separated subset of {', '.join(SOURCE_PREDICATES)}",
    )
    parser.add_argument("--k-grid", default="1,2,4,8,16")
    parser.add_argument("--envelope-k", type=int, default=500)
    parser.add_argument(
        "--texts",
        type=Path,
        default=None,
        help="Corpus (parquet directory/file or jsonl with title/text); "
        "required for value/hybrid.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    predicates = [p.strip() for p in args.predicate.split(",") if p.strip()]
    unknown = [p for p in predicates if p not in SOURCE_PREDICATES]
    if unknown:
        raise SystemExit(f"Unknown predicates {unknown}; valid: {SOURCE_PREDICATES}")
    needs_texts = any(p in ("value", "hybrid") for p in predicates)
    if needs_texts and args.texts is None:
        raise SystemExit("--predicate value/hybrid requires --texts.")
    k_grid = [int(part) for part in args.k_grid.split(",") if part.strip()]

    with open(args.title_to_index, "rb") as handle:
        titles = list(pickle.load(handle))
    space = SourceSpace.load(titles=titles, embeddings_path=args.closure_embeddings)
    rows = load_rows(args.prompts)
    golds = [row.get("source_title") for row in rows if row.get("source_title") in space]

    # Neighbors once per distinct gold, deep enough for every predicate.
    depth = max([args.envelope_k] if needs_texts else [0] + k_grid)
    t = time.time()
    envelopes = space.nearest_many(golds, depth) if depth > 0 else {}
    print(f"neighbors: {len(envelopes):,} distinct sources x {depth} in {time.time()-t:.0f}s", flush=True)

    value_hits = None
    if needs_texts:
        t = time.time()
        value_hits = stream_value_hits(args.texts, rows, envelopes, envelope_k=args.envelope_k)
        print(f"value hits: {sum(len(h) for h in value_hits.values()):,} over "
              f"{len(value_hits):,} facts in {time.time()-t:.0f}s", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.prompts.stem
    reports = []
    for predicate in predicates:
        # provenance and value ignore k; write them once under k0.
        ks = [0] if predicate in ("provenance", "value") else k_grid
        for k in ks:
            output = args.output_dir / f"{stem}_{predicate}_k{k}.jsonl"
            report = build_closure_prompt_file(
                args.prompts,
                output,
                space=space,
                predicate=predicate,
                k=k,
                envelope_k=args.envelope_k,
                neighbors=envelopes,
                value_hits=value_hits,
            )
            reports.append(report)
            print(json.dumps(report), flush=True)
        (args.output_dir / f"{stem}_{predicate}_report.json").write_text(
            json.dumps([r for r in reports if r["predicate"] == predicate], indent=2)
        )


if __name__ == "__main__":
    main()
