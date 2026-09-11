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
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from halo.core.embeddings import normalize_rows_inplace
from halo.core.equivalence import build_alias_set, normalize_text, prompt_row_aliases

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
        # ~20 GB for the released corpus (6.4M x 768 float32): normalize in
        # place rather than allocating two more full copies. A float32
        # C-contiguous input array is modified in place.
        self.embeddings = normalize_rows_inplace(self.embeddings)
        self._position = {title: i for i, title in enumerate(self.titles)}

    @classmethod
    def load(
        cls, *, titles: list[str] | tuple[str, ...], embeddings_path: str | Path
    ) -> "SourceSpace":
        with np.load(embeddings_path, allow_pickle=False) as archive:
            embeddings = archive["embeddings"]  # float32 from the builder: no copy
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

    def nearest_many(
        self,
        titles: Iterable[str],
        k: int,
        *,
        query_chunk: int = 1024,
        row_chunk: int = 262_144,
    ) -> dict[str, list[tuple[str, float]]]:
        """``nearest`` for many titles at once, as blocked matrix products.

        One matvec per query streams the whole (6.4M x 768) matrix from
        memory each time; a prompt set's 20k gold sources would read
        ~400 TB. Blocking queries and rows turns that into GEMMs that read
        the matrix once per query block, keeping a running top-k per query.
        Results equal ``nearest(title, k)`` per title (ties aside).
        """
        unique = list(dict.fromkeys(titles))
        if k <= 0 or not unique:
            return {title: [] for title in unique}
        n_rows = self.embeddings.shape[0]
        fetch = min(n_rows, k + 1)  # over-fetch one for the query itself
        out: dict[str, list[tuple[str, float]]] = {}
        for start in range(0, len(unique), query_chunk):
            block_titles = unique[start : start + query_chunk]
            queries = self.embeddings[[self._position[t] for t in block_titles]]
            best_scores = np.full((len(block_titles), fetch), -np.inf, dtype=np.float32)
            best_rows = np.zeros((len(block_titles), fetch), dtype=np.int64)
            for row_start in range(0, n_rows, row_chunk):
                rows = self.embeddings[row_start : row_start + row_chunk]
                scores = rows @ queries.T  # (rows, queries)
                take = min(fetch, scores.shape[0])
                local = np.argpartition(-scores, take - 1, axis=0)[:take].T  # (queries, take)
                local_scores = np.take_along_axis(scores.T, local, axis=1)
                merged_scores = np.concatenate([best_scores, local_scores], axis=1)
                merged_rows = np.concatenate([best_rows, local + row_start], axis=1)
                keep = np.argpartition(-merged_scores, fetch - 1, axis=1)[:, :fetch]
                best_scores = np.take_along_axis(merged_scores, keep, axis=1)
                best_rows = np.take_along_axis(merged_rows, keep, axis=1)
            for i, title in enumerate(block_titles):
                order = np.argsort(-best_scores[i])
                self_position = self._position[title]
                out[title] = [
                    (self.titles[int(best_rows[i, j])], float(best_scores[i, j]))
                    for j in order
                    if int(best_rows[i, j]) != self_position
                ][:k]
        return out


def answer_mentioned(normalized_text: str, normalized_aliases: Sequence[str]) -> bool:
    """The value-source rule: the source text mentions the answer or an
    alias as a whole phrase, after HALO's normalization on both sides —
    the same whole-phrase containment the entry-level value filter and
    ``metrics.contains_match`` apply."""
    padded = f" {normalized_text} "
    return any(alias and f" {alias} " in padded for alias in normalized_aliases)


def normalized_answer_aliases(row: Mapping) -> tuple[str, ...]:
    aliases = build_alias_set(
        str(row["gold_object"]), prompt_row_aliases(dict(row), "object")
    )
    return tuple(normalize_text(alias) for alias in aliases)


def _mentions_answer(text: str, row: Mapping) -> bool:
    return answer_mentioned(normalize_text(text), normalized_answer_aliases(row))


def iter_prompt_rows(lines: Iterable[str]) -> Iterable[dict]:
    """Prompt rows from jsonl lines, blank lines skipped — the row indexing
    ``build_closure_prompt_file`` and its callers share."""
    for line in lines:
        line = line.strip()
        if line:
            yield json.loads(line)


def expand_row(
    row: Mapping,
    *,
    space: SourceSpace,
    predicate: str,
    k: int,
    envelope_k: int = 500,
    source_texts: Callable[[str], str | None] | None = None,
    neighbors: Sequence[tuple[str, float]] | None = None,
    value_hits: Iterable[str] | None = None,
) -> dict:
    """Return a copy of ``row`` with the closure manifest for ``predicate``.

    ``neighbors`` (the gold source's E-space neighbors, most similar first,
    at least ``max(k, envelope_k)`` long) and ``value_hits`` (envelope
    titles whose text mentions the answer) let a caller precompute the
    expensive parts in bulk — ``SourceSpace.nearest_many`` and a single
    streaming pass over the corpus — instead of one matvec and one text
    lookup per row. Without them the row is expanded on its own.
    """
    if predicate not in SOURCE_PREDICATES:
        raise ValueError(f"predicate must be one of {SOURCE_PREDICATES}.")
    gold = row.get("source_title")
    if not gold:
        raise ValueError(
            "Row has no source_title; run scripts/augment_source_titles.py first."
        )
    if gold not in space:
        raise ValueError(f"source_title {gold!r} is not in the source space.")

    def _neighbors(count: int) -> list[tuple[str, float]]:
        if neighbors is not None:
            if len(neighbors) < min(count, len(space.titles) - 1):
                raise ValueError(
                    f"Precomputed neighbors for {gold!r} hold {len(neighbors)} "
                    f"entries but {count} are needed."
                )
            return list(neighbors[:count])
        return space.nearest(gold, count)

    metadata: dict = {
        "predicate": f"{predicate}-source",
        "k": k,
        "encoder": space.encoder,
    }
    members: list[str] = [gold]
    if predicate == "geometric":
        members += [title for title, _ in _neighbors(k)]
    elif predicate in ("value", "hybrid"):
        envelope = _neighbors(envelope_k)
        metadata["envelope_k"] = envelope_k
        metadata["envelope_truncated"] = len(envelope) == envelope_k
        if value_hits is not None:
            envelope_titles = {title for title, _ in envelope}
            value_hits = [title for title in value_hits if title in envelope_titles]
        elif source_texts is not None:
            value_hits = []
            for title, _ in envelope:
                text = source_texts(title)
                if text is not None and _mentions_answer(text, row):
                    value_hits.append(title)
        else:
            raise ValueError(f"{predicate!r} requires source texts or value_hits.")
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
    neighbors: Mapping[str, Sequence[tuple[str, float]]] | None = None,
    value_hits: Mapping[int, Iterable[str]] | None = None,
) -> dict:
    """Write the closure prompt file for one predicate and breadth.

    ``neighbors`` maps gold titles to precomputed E-space neighbors and
    ``value_hits`` maps a row's index in ``prompts_path`` (blank lines
    excluded) to the envelope titles mentioning its answer; see
    ``expand_row``. Rows whose gold source is outside the space are
    skipped and counted.
    """
    kept = skipped = 0
    sizes: list[int] = []
    with open(prompts_path, encoding="utf-8") as source, open(
        output_path, "w", encoding="utf-8"
    ) as sink:
        for row_index, row in enumerate(iter_prompt_rows(source)):
            gold = row.get("source_title")
            if gold not in space:
                skipped += 1
                continue
            expanded = expand_row(
                row,
                space=space,
                predicate=predicate,
                k=k,
                envelope_k=envelope_k,
                source_texts=source_texts,
                neighbors=None if neighbors is None else neighbors.get(gold),
                value_hits=None if value_hits is None else value_hits.get(row_index, ()),
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
