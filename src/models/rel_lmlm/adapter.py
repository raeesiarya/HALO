from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np

from lmlm_audit.core.equivalence import values_equivalent
from lmlm_audit.core.examples import AuditExample
from models.rel_lmlm.database import (
    TargetFact,
    candidate_supports_target_fact,
    triple_id,
)


def _candidate_field(candidate: Any, key: str) -> Any:
    value = getattr(candidate, key, None)
    if value is None and isinstance(candidate, Mapping):
        value = candidate.get(key)
    if value is None:
        metadata = getattr(candidate, "metadata", None)
        if metadata is None and isinstance(candidate, Mapping):
            metadata = candidate.get("metadata")
        if isinstance(metadata, Mapping):
            value = metadata.get(key)
    return value


def rel_support_judge(candidate: Any, example: AuditExample) -> dict[str, Any]:
    """Support judge over triple candidates: full (s, r, o) equivalence when
    the candidate and example both carry subject/relation, else value-level
    answer equivalence."""
    subject = _candidate_field(candidate, "subject")
    relation = _candidate_field(candidate, "relation")
    obj = (
        _candidate_field(candidate, "object")
        or _candidate_field(candidate, "text_value")
        or _candidate_field(candidate, "value")
        or ""
    )

    if (
        subject is not None
        and relation is not None
        and example.subject is not None
        and example.relation is not None
    ):
        target = TargetFact(
            fact_id=example.fact_id if isinstance(example.fact_id, int) else None,
            subject=example.subject,
            subject_aliases=example.subject_aliases,
            relation=example.relation,
            relation_aliases=example.relation_aliases,
            object=example.ground_truth,
            object_aliases=example.object_aliases,
        )
        *_, supports = candidate_supports_target_fact(
            (str(subject), str(relation), str(obj)), target
        )
        method = "triple-equivalence"
    else:
        supports = values_equivalent(
            str(obj),
            example.ground_truth,
            right_aliases=example.object_aliases,
        )
        method = "value-equivalence"

    return {
        "supports_target": bool(supports),
        "support_method": method,
        "support_confidence": 1.0 if supports else 0.0,
    }


@dataclass
class TripleSearchIndex:
    """Adapts the rel-LMLM TopkRetriever's FAISS index to the search
    interface the closure builder and sweep/adversarial runners consume:
    ``search(query_vector, top_k, similarity_threshold)`` returning
    candidates with ``id``/``score``/``text_value``/``metadata``."""

    db_manager: Any

    def _retriever(self) -> Any:
        if getattr(self.db_manager, "topk_retriever", None) is None:
            self.db_manager.init_topk_retriever()
        return self.db_manager.topk_retriever

    def search(
        self,
        query_vector: Any,
        top_k: int = 1,
        similarity_threshold: float | None = None,
    ) -> list[Any]:
        retriever = self._retriever()
        query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        norm = float(np.linalg.norm(query))
        if norm > 0.0:
            query = query / norm

        available = len(getattr(retriever, "id_to_triplet", {})) or top_k
        distances, indices = retriever.index.search(query, min(top_k, available))

        results = []
        for distance, index in zip(distances[0], indices[0]):
            if index == -1 or index not in retriever.id_to_triplet:
                continue
            score = float(distance)
            if similarity_threshold is not None and score < similarity_threshold:
                continue
            subject, relation, obj = retriever.id_to_triplet[index]
            results.append(
                SimpleNamespace(
                    id=triple_id(subject, relation, obj),
                    score=score,
                    text_value=obj,
                    text_key=f"{subject} {relation}",
                    metadata={
                        "subject": subject,
                        "relation": relation,
                        "object": obj,
                    },
                    vector=None,
                )
            )
        results.sort(key=lambda candidate: -candidate.score)
        return results


def build_search_index(backend: Any) -> "TripleSearchIndex":
    return TripleSearchIndex(backend.base_db_manager)
