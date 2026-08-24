"""Source-level deletion closures over the shared encoder space E.

docs/NULLS_AUDIT_DESIGN.md §2/§4-5: cross-substrate closures are computed at
source (document) granularity in one external embedding space, ahead of the
audit, and written into prompt-row deletion manifests (``source_ids`` =
article titles). Both substrates then execute the same manifest — Co-LMLM by
search-time source filtering, NULLs by sink-mask exclusion — so the closure
rule is held fixed and only the deletion operator varies.

Predicates (all include the gold source):

- ``provenance``: the gold source alone (k = 0; every sweep's anchor).
- ``geometric``: gold + the k nearest sources in E-space (the breadth-k
  sweep axis of design §5 — sweeps parametrize by k, never by raw radius).
- ``value``: gold + envelope sources whose *text* mentions the gold answer
  (whole-phrase, alias-aware). An oracle evaluation rule, not a deployable
  policy, exactly like the entry-level value filter.
- ``hybrid``: gold + (geometric-k ∩ value).

``value``/``hybrid`` scan a top-``envelope_k`` E-space envelope rather than
all sources, mirroring the entry-level closure's envelope bound; envelope
truncation is recorded in the manifest metadata (no silent caps).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from halo.core.equivalence import build_alias_set, prompt_row_aliases
from halo.core.metrics import contains_match

SOURCE_PREDICATES = ("provenance", "geometric", "value", "hybrid")


@dataclass
class SourceSpace:
    """Titles with row-aligned, L2-normalized embeddings in the shared E."""

    titles: tuple[str, ...]
    embeddings: np.ndarray
    encoder: str | None = None

    def __post_init__(self) -> None:
        if self.embeddings.shape[0] != len(self.titles):
            raise ValueError(
                f"{self.embeddings.shape[0]} embedding rows for "
                f"{len(self.titles)} titles; the artifact must be row-aligned."
            )
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        self.embeddings = (self.embeddings / norms).astype(np.float32)
        self._position = {title: i for i, title in enumerate(self.titles)}

    @classmethod
    def load(
        cls, *, titles: list[str] | tuple[str, ...], embeddings_path: str | Path
    ) -> "SourceSpace":
        with np.load(embeddings_path, allow_pickle=False) as archive:
            embeddings = np.asarray(archive["embeddings"], dtype=np.float32)
            encoder = str(archive["encoder"]) if "encoder" in archive.files else None
            mode = str(archive["mode"]) if "mode" in archive.files else None
        if mode == "routing":
            raise ValueError(
                f"{embeddings_path} is a routing-space artifact (title "
                "strings, model-matched); closures must use the shared "
                "encoder E over source text (build with --mode closure)."
            )
        return cls(titles=tuple(titles), embeddings=embeddings, encoder=encoder)

    def __contains__(self, title: str) -> bool:
        return title in self._position

    def nearest(self, title: str, k: int) -> list[tuple[str, float]]:
        """The k nearest sources to ``title`` (excluding itself)."""
        if k <= 0:
            return []
        scores = self.embeddings @ self.embeddings[self._position[title]]
        fetch = min(len(self.titles), k + 1)
        candidates = np.argpartition(-scores, fetch - 1)[:fetch]
        ranked = candidates[np.argsort(-scores[candidates])]
        return [
            (self.titles[position], float(scores[position]))
            for position in ranked
            if position != self._position[title]
        ][:k]


def _mentions_answer(text: str, row: Mapping) -> bool:
    aliases = build_alias_set(
        str(row["gold_object"]), prompt_row_aliases(dict(row), "object_aliases")
    )
    return bool(contains_match(text, str(row["gold_object"]), list(aliases)))


def expand_row(
    row: Mapping,
    *,
    space: SourceSpace,
    predicate: str,
    k: int,
    envelope_k: int = 500,
    source_texts: Callable[[str], str | None] | None = None,
) -> dict:
    """Return a copy of ``row`` with the closure manifest for ``predicate``."""
    if predicate not in SOURCE_PREDICATES:
        raise ValueError(f"predicate must be one of {SOURCE_PREDICATES}.")
    gold = row.get("source_title")
    if not gold:
        raise ValueError(
            "Row has no source_title; run scripts/augment_source_titles.py first."
        )
    if gold not in space:
        raise ValueError(f"source_title {gold!r} is not in the source space.")

    metadata: dict = {
        "predicate": f"{predicate}-source",
        "k": k,
        "encoder": space.encoder,
    }
    members: list[str] = [gold]
    if predicate == "geometric":
        members += [title for title, _ in space.nearest(gold, k)]
    elif predicate in ("value", "hybrid"):
        if source_texts is None:
            raise ValueError(f"{predicate!r} requires source texts.")
        envelope = space.nearest(gold, envelope_k)
        metadata["envelope_k"] = envelope_k
        metadata["envelope_truncated"] = len(envelope) == envelope_k
        value_hits = []
        for title, _ in envelope:
            text = source_texts(title)
            if text is not None and _mentions_answer(text, row):
                value_hits.append(title)
        if predicate == "value":
            members += value_hits
        else:
            geometric = {title for title, _ in envelope[:k]}
            members += [title for title in value_hits if title in geometric]
        metadata["value_hits"] = len(value_hits)

    expanded = dict(row)
    expanded["deletion_manifest"] = {
        "source_ids": sorted(dict.fromkeys(members)),
        "strategy": f"{predicate}-source",
        "metadata": metadata,
    }
    return expanded


def build_closure_prompt_file(
    prompts_path: Path,
    output_path: Path,
    *,
    space: SourceSpace,
    predicate: str,
    k: int,
    envelope_k: int = 500,
    source_texts: Callable[[str], str | None] | None = None,
) -> dict:
    kept = skipped = 0
    sizes: list[int] = []
    with open(prompts_path, encoding="utf-8") as source, open(
        output_path, "w", encoding="utf-8"
    ) as sink:
        for line in source:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("source_title") not in space:
                skipped += 1
                continue
            expanded = expand_row(
                row,
                space=space,
                predicate=predicate,
                k=k,
                envelope_k=envelope_k,
                source_texts=source_texts,
            )
            sizes.append(len(expanded["deletion_manifest"]["source_ids"]))
            sink.write(json.dumps(expanded, ensure_ascii=False) + "\n")
            kept += 1
    return {
        "predicate": predicate,
        "k": k,
        "kept": kept,
        "skipped_out_of_space": skipped,
        "mean_manifest_size": float(np.mean(sizes)) if sizes else None,
        "output": str(output_path),
    }
