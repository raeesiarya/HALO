"""Augment a prompt set with NULLs source titles and provenance manifests.

Thin CLI over ``models.nulls_wiki.titles`` (the logic and its rationale live
there): resolves each fact's Wikipedia source article against the NULLs
training mapping and writes ``source_title`` plus the provenance-source
deletion manifest. Unresolved rows are dropped and reported in
``<output>.exclusions.json``. Driven as a job by the cross-model scheduler.

Usage:
  uv run python scripts/augment_source_titles.py \
      --prompts data/prompts_trex.jsonl \
      --title-to-index data/nulls-title-to-index.pkl \
      --output data/prompts_trex_nulls.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from models.nulls_wiki.titles import augment  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--title-to-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = augment(args.prompts, args.title_to_index, args.output)
    print(
        f"kept {report['kept']} rows, dropped {report['dropped']} "
        f"(exclusion rate {report['exclusion_rate']:.4f}); "
        f"methods: {report['resolution_methods']}"
    )


if __name__ == "__main__":
    main()
