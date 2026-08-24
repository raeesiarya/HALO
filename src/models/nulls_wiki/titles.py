"""Source-title resolution and prompt augmentation for the NULLs audit.

Resolution is deliberately conservative: exact subject match, else the
subject under MediaWiki's own title-identity rule (underscores are spaces,
whitespace collapses, only the first character is case-insensitive). Nothing
beyond that rule — no full case-folding, no alias fallback — because a
lexical guess that lands on a *different real article* is a silently wrong
sink identity the downstream verification gate cannot always catch.

Unresolved rows are DROPPED from the augmented output and counted: the
exclusion rate is a substantive statistic (title/dump mismatch,
docs/NULLS_AUDIT_DESIGN.md §9) and is written to
``<output>.exclusions.json``. Whether a resolved title really addresses the
fact's sink is validated empirically by the gate (models.nulls_wiki.gate).
"""

from __future__ import annotations

import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping


def normalize_title(title: str) -> str:
    """MediaWiki title identity: underscores are spaces, runs of whitespace
    collapse, and only the FIRST character is case-insensitive. This is the
    wiki's own equivalence rule — two strings with the same normal form name
    the same page — so matching under it introduces no guessing. Anything
    looser (full case-folding, aliases) can land on a different real article
    and silently corrupt the sink identity."""
    collapsed = " ".join(title.replace("_", " ").split())
    if not collapsed:
        return collapsed
    return collapsed[0].upper() + collapsed[1:]


def build_title_resolver(titles: Iterable[str]):
    """Exact-first resolver with a MediaWiki-normalized fallback index.

    Ambiguous normal forms (several distinct stored titles collapsing to
    one form) are dropped from the fallback rather than guessed.
    """
    exact = set()
    normalized: dict[str, str | None] = {}
    for title in titles:
        exact.add(title)
        key = normalize_title(title)
        if key in normalized and normalized[key] != title:
            normalized[key] = None  # ambiguous: refuse to guess
        else:
            normalized.setdefault(key, title)

    def resolve(candidates: Iterable[str]) -> tuple[str, str] | None:
        seen: list[str] = []
        for candidate in candidates:
            candidate = str(candidate).strip()
            if not candidate or candidate in seen:
                continue
            seen.append(candidate)
            if candidate in exact:
                return candidate, "exact"
        for candidate in seen:
            resolved = normalized.get(normalize_title(candidate))
            if resolved is not None:
                return resolved, "mediawiki-normalized"
        return None

    return resolve


def candidate_titles(row: Mapping) -> list[str]:
    # The subject only: T-REx facts are extracted from the subject article's
    # abstract, so the subject title IS the provenance source. Aliases are
    # deliberately not consulted — an alias that happens to be a real title
    # names a different article, not this fact's source.
    if row.get("subject"):
        return [str(row["subject"])]
    return []


def augment(prompts_path: Path, title_to_index_path: Path, output_path: Path) -> dict:
    """Write the augmented prompt set (source_title + provenance manifest)
    and the exclusions report; return the report."""
    with open(title_to_index_path, "rb") as handle:
        title_to_index = pickle.load(handle)
    resolve = build_title_resolver(title_to_index)

    kept = 0
    dropped: list[dict] = []
    method_counts: Counter[str] = Counter()
    with open(prompts_path, encoding="utf-8") as source, open(
        output_path, "w", encoding="utf-8"
    ) as sink:
        for line in source:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            resolution = resolve(candidate_titles(row))
            if resolution is None:
                dropped.append(
                    {
                        "fact_id": row.get("fact_id"),
                        "prompt_id": row.get("prompt_id"),
                        "subject": row.get("subject"),
                    }
                )
                continue
            title, method = resolution
            method_counts[method] += 1
            row["source_title"] = title
            row["deletion_manifest"] = {
                "source_ids": [title],
                "strategy": "provenance-source",
            }
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1

    report = {
        "prompts": str(prompts_path),
        "title_to_index": str(title_to_index_path),
        "kept": kept,
        "dropped": len(dropped),
        "exclusion_rate": (
            len(dropped) / (kept + len(dropped)) if kept + len(dropped) else None
        ),
        "resolution_methods": dict(method_counts),
        "dropped_rows": dropped,
    }
    report_path = output_path.with_suffix(output_path.suffix + ".exclusions.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report
