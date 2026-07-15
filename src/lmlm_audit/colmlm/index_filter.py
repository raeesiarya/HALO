from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np

from lmlm_audit.colmlm.errors import (
    CoLMLMIntegrationError,
    ExclusionSearchExhaustedError,
)
from lmlm_audit.core.examples import AuditExample


def _candidate_id(candidate: Any) -> str:
    value = getattr(candidate, "id", None)
    if value is None and isinstance(candidate, Mapping):
        value = candidate.get("id")
        if value is None:
            value = candidate.get("entry_id")
    return "" if value is None else str(value)


def _candidate_metadata(candidate: Any) -> dict[str, Any]:
    value = getattr(candidate, "metadata", None)
    if value is None and isinstance(candidate, Mapping):
        value = candidate.get("metadata")
    return dict(value) if isinstance(value, Mapping) else {}


def _candidate_source_id(candidate: Any) -> str | None:
    metadata = _candidate_metadata(candidate)
    for key in ("source_id", "source", "document_id", "sample_id", "url"):
        value = metadata.get(key)
        if value is not None:
            return str(value)
    return None


def _candidate_text(candidate: Any) -> str:
    value = getattr(candidate, "text_value", None)
    if value is None and isinstance(candidate, Mapping):
        value = candidate.get("text_value")
        if value is None:
            value = candidate.get("value")
    return "" if value is None else str(value)


def _candidate_score(candidate: Any) -> float | None:
    value = getattr(candidate, "score", None)
    if value is None and isinstance(candidate, Mapping):
        value = candidate.get("score")
    return None if value is None else float(value)


def _serialize_candidate(
    candidate: Any,
    example: AuditExample,
    support_judge: Callable[[Any, AuditExample], Mapping[str, Any]],
) -> dict[str, Any]:
    metadata = _candidate_metadata(candidate)
    text_key = getattr(candidate, "text_key", None)
    if text_key is None and isinstance(candidate, Mapping):
        text_key = candidate.get("text_key")
    result = {
        "entry_id": _candidate_id(candidate),
        "source_id": _candidate_source_id(candidate),
        "value": _candidate_text(candidate),
        "text_key": text_key,
        "score": _candidate_score(candidate),
        "metadata": metadata,
    }
    result.update(dict(support_judge(candidate, example)))
    return result


@dataclass
class _FilteringSearchIndex:
    base_index: Any
    example: AuditExample
    excluded_entry_ids: frozenset[str]
    excluded_source_ids: frozenset[str]
    support_judge: Callable[[Any, AuditExample], Mapping[str, Any]]
    exclude_all: bool = False
    exclude_supporting: bool = False
    # The example whose answer the semantic backstop judges against. Under a
    # sweep, neighbor prompts run with the *target* fact's manifest, so this
    # differs from `example` (which drives trace serialization).
    backstop_example: AuditExample | None = None
    max_filter_overfetch: int = 4096
    events: list[dict[str, Any]] = field(default_factory=list)
    query_embeddings: list[np.ndarray | None] = field(default_factory=list)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_index, name)

    def _is_excluded(self, candidate: Any) -> bool:
        entry_id = _candidate_id(candidate)
        source_id = _candidate_source_id(candidate)
        return entry_id in self.excluded_entry_ids or (
            source_id is not None and source_id in self.excluded_source_ids
        )

    def search(
        self,
        query_vector: Any,
        top_k: int = 1,
        similarity_threshold: float | None = None,
    ) -> list[Any]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        try:
            query_array = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError):
            query_array = None
        self.query_embeddings.append(query_array)
        entry_exclusion_count = len(self.excluded_entry_ids)
        source_exclusions_are_unbounded = bool(self.excluded_source_ids)
        exclusions_are_unbounded = (
            source_exclusions_are_unbounded or self.exclude_supporting
        )
        if self.exclude_all:
            # Every candidate is discarded, so over-fetching buys nothing.
            extra = 0
        elif exclusions_are_unbounded:
            extra = self.max_filter_overfetch
        else:
            extra = min(entry_exclusion_count, self.max_filter_overfetch)
        search_k = top_k + extra
        raw = self.base_index.search(
            query_vector,
            top_k=search_k,
            similarity_threshold=similarity_threshold,
        )
        if raw and isinstance(raw[0], list):
            if len(raw) != 1:
                raise CoLMLMIntegrationError(
                    "Co-LMLM generator issued a single query but the index returned "
                    f"{len(raw)} result lists."
                )
            raw = raw[0]
        candidates = list(raw or [])
        deleted: list[Any] = []
        retained: list[Any] = []
        for candidate in candidates:
            excluded = self.exclude_all or self._is_excluded(candidate)
            if not excluded and self.exclude_supporting:
                # Semantic-closure backstop: also null any candidate the
                # support judge marks as expressing the target answer, even
                # when the materialized closure missed it.
                excluded = bool(
                    dict(
                        self.support_judge(
                            candidate, self.backstop_example or self.example
                        )
                    ).get("supports_target")
                )
            (deleted if excluded else retained).append(candidate)
        selected = retained[:top_k]

        event = {
            "event_index": len(self.events),
            "threshold": similarity_threshold,
            "requested_top_k": top_k,
            "searched_top_k": search_k,
            "exclude_all": self.exclude_all,
            "exclude_supporting": self.exclude_supporting,
            "query_embedding_captured": query_array is not None,
            "query_dim": None if query_array is None else int(query_array.size),
            "query_l2_norm": (
                None
                if query_array is None
                else float(np.linalg.norm(query_array))
            ),
            "all_candidates": [
                _serialize_candidate(candidate, self.example, self.support_judge)
                for candidate in candidates
            ],
            "deleted_candidates": [
                _serialize_candidate(candidate, self.example, self.support_judge)
                for candidate in deleted
            ],
            "retained_candidates": [
                _serialize_candidate(candidate, self.example, self.support_judge)
                for candidate in retained
            ],
            "selected_candidate": (
                _serialize_candidate(selected[0], self.example, self.support_judge)
                if selected
                else None
            ),
        }
        self.events.append(event)

        bounded_out = (
            not self.exclude_all
            and (
                exclusions_are_unbounded
                or entry_exclusion_count > self.max_filter_overfetch
            )
            and len(candidates) == search_k
            and len(selected) < top_k
        )
        if bounded_out:
            raise ExclusionSearchExhaustedError(
                "The exclusion filter exhausted its over-retrieval budget before "
                "finding enough retained candidates. Increase max_filter_overfetch "
                "or use a native FAISS ID selector."
            )
        return selected
