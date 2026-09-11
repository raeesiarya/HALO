"""NULLs verification gate — thin CLI over ``models.nulls_wiki.gate``.

The gate's definition and rationale live in the package module. Run modes
(the cross-model scheduler drives all of them as jobs):

- full: score every fact, write ``<output>`` + ``<output>.summary.json``
  and, with ``--emit-passed-prompts``, the gated prompt set the audits
  consume.
- ``--shard I/N``: score facts with ``row_index %% N == I`` into
  ``<output>.shardIofN.jsonl`` (no summary, no gated prompts).
- ``--merge N``: no model load; concatenate the N shard files into
  ``<output>``, write the summary and the gated prompt set.

Usage:
  uv run python scripts/nulls_verification_gate.py \
      --prompts <augmented>.jsonl --checkpoint-dir data/nulls-wikipedia-full \
      --title-to-index data/nulls-title-to-index.pkl \
      --title-embeddings data/nulls-title-embeddings.npz \
      --output <out>/gate.jsonl --emit-passed-prompts <out>/gated.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from models.nulls_wiki import gate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--title-to-index", type=Path)
    parser.add_argument(
        "--title-embeddings",
        type=Path,
        default=None,
        help="Enables the far-placebo similarity ceiling; without it the "
        "placebo is only guaranteed to be a different title.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--margin",
        type=float,
        default=0.0,
        help="Required Sink-On minus placebo log-likelihood margin (nats). "
        "0 records margins without excluding on them; calibrate from the "
        "summary's margin distribution.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--placebo-ceiling", type=float, default=0.5)
    parser.add_argument("--shard", default=None, metavar="I/N")
    parser.add_argument(
        "--merge",
        type=int,
        default=None,
        metavar="N",
        help="Merge N shard files (no model load) and finalize.",
    )
    parser.add_argument(
        "--emit-passed-prompts",
        type=Path,
        default=None,
        help="Write the gated prompt set (rows whose fact passed) here; the "
        "cross-model scheduler points the NULLs audits at this file.",
    )
    args = parser.parse_args()

    if args.shard is not None and args.merge is not None:
        raise SystemExit("--shard and --merge are mutually exclusive.")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.merge is not None:
        summary = gate.finalize(
            gate.read_shards(args.output, args.merge),
            output=args.output,
            prompts_path=args.prompts,
            margin=args.margin,
            emit_passed_prompts=args.emit_passed_prompts,
        )
        print(json.dumps(summary, indent=2))
        return

    if args.checkpoint_dir is None or args.title_to_index is None:
        raise SystemExit("--checkpoint-dir and --title-to-index are required to score.")

    from models.nulls_wiki.backend import NullsWikiAuditBackend

    backend = NullsWikiAuditBackend.from_release(
        checkpoint_dir=args.checkpoint_dir,
        title_to_index_path=args.title_to_index,
        title_embeddings_path=args.title_embeddings,
        placebo_similarity_ceiling=args.placebo_ceiling,
    )

    if args.shard is not None:
        index, count = gate.parse_shard(args.shard)
        records = gate.score_rows(
            backend,
            args.prompts,
            margin=args.margin,
            placebo_ceiling=args.placebo_ceiling,
            shard=(index, count),
            limit=args.limit,
        )
        target = gate.write_shard(records, args.output, index, count)
        print(f"gate shard {args.shard}: {len(records)} records -> {target}")
        return

    records = gate.score_rows(
        backend,
        args.prompts,
        margin=args.margin,
        placebo_ceiling=args.placebo_ceiling,
        limit=args.limit,
    )
    summary = gate.finalize(
        records,
        output=args.output,
        prompts_path=args.prompts,
        margin=args.margin,
        emit_passed_prompts=args.emit_passed_prompts,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
