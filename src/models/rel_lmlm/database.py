import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from lmlm_audit.core.equivalence import prompt_row_aliases, values_equivalent
from lmlm_audit.core.states import DatabaseState


def triple_id(subject: str, relation: str, obj: str) -> str:
    """Stable, unambiguous entry ID for a bare (s, r, o) triple — the
    upstream database has no row IDs."""
    return json.dumps([subject, relation, obj], ensure_ascii=False)


LOOKUP_PATTERNS = [
    r"\[dblookup\('((?:[^'\\]|\\.)+)',\s*'((?:[^'\\]|\\.)+)'\)\s*->",
    r"\[dblookup\('(.+?)',\s*'(.+?)'\)\s*->",
    r"<\|db_entity\|>(.+?)<\|db_relationship\|>(.+?)<\|db_return\|>",
]


@dataclass(frozen=True)
class TargetFact:
    fact_id: int | None
    subject: str
    subject_aliases: tuple[str, ...]
    relation: str
    relation_aliases: tuple[str, ...]
    object: str
    object_aliases: tuple[str, ...]


def target_fact_from_prompt_row(prompt_row: dict[str, Any]) -> TargetFact:
    return TargetFact(
        fact_id=prompt_row.get("fact_id"),
        subject=prompt_row["subject"],
        subject_aliases=prompt_row_aliases(prompt_row, "subject"),
        relation=prompt_row["relation"],
        relation_aliases=prompt_row_aliases(prompt_row, "relation"),
        object=prompt_row["gold_object"],
        object_aliases=prompt_row_aliases(prompt_row, "object"),
    )


def is_deleted_triplet(triplet: tuple[str, str, str], target_fact: TargetFact) -> bool:
    subject, relation, obj = triplet
    return (
        values_equivalent(
            subject, target_fact.subject, right_aliases=target_fact.subject_aliases
        )
        and values_equivalent(
            relation,
            target_fact.relation,
            right_aliases=target_fact.relation_aliases,
        )
        and values_equivalent(
            obj, target_fact.object, right_aliases=target_fact.object_aliases
        )
    )


def candidate_supports_target_fact(
    triplet: tuple[str, str, str],
    target_fact: TargetFact,
) -> tuple[bool, bool, bool, bool]:
    subject, relation, obj = triplet
    matches_subject = values_equivalent(
        subject,
        target_fact.subject,
        right_aliases=target_fact.subject_aliases,
    )
    matches_relation = values_equivalent(
        relation,
        target_fact.relation,
        right_aliases=target_fact.relation_aliases,
    )
    matches_object = values_equivalent(
        obj,
        target_fact.object,
        right_aliases=target_fact.object_aliases,
    )
    supports_target_fact = matches_subject and matches_relation and matches_object
    return (
        matches_subject,
        matches_relation,
        matches_object,
        supports_target_fact,
    )


def extract_lookup_query(prompt: str) -> tuple[str, str]:
    matches = {
        tuple(match)
        for pattern in LOOKUP_PATTERNS
        for match in re.findall(pattern, prompt)
    }

    if not matches:
        raise ValueError(f"No valid dblookup pattern found in prompt: {prompt}")

    if len(matches) > 1:
        raise ValueError(
            f"Multiple dblookup matches found: {matches} in prompt: {prompt}"
        )

    entity, relationship = matches.pop()
    return entity, relationship


def retrieve_triplet_candidates(
    topk_retriever: Any,
    entity: str,
    relation: str,
    threshold: float | None = None,
) -> tuple[list[tuple[str, str, str, float]], np.ndarray | None]:
    query_text = (
        f"{topk_retriever._normalize_text(entity)} "
        f"{topk_retriever._normalize_text(relation)}"
    )
    query_embedding = topk_retriever.model.encode(
        [query_text],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    distances, indices = topk_retriever.index.search(
        query_embedding,
        topk_retriever.top_k,
    )

    effective_threshold = (
        threshold if threshold is not None else topk_retriever.default_threshold
    )

    results: list[tuple[str, str, str, float]] = []
    for distance, index in zip(distances[0], indices[0]):
        if index == -1 or index not in topk_retriever.id_to_triplet:
            continue
        if distance < effective_threshold:
            continue

        subject, relation_name, value = topk_retriever.id_to_triplet[index]
        results.append((subject, relation_name, value, float(distance)))

    results.sort(key=lambda item: item[-1], reverse=True)
    try:
        captured_query = np.asarray(
            query_embedding, dtype=np.float32
        ).reshape(-1)
    except (TypeError, ValueError):
        captured_query = None
    return results, captured_query


class AuditDatabaseManager:
    def __init__(
        self,
        base_db_manager: Any,
        state: DatabaseState,
        target_fact: TargetFact | None = None,
        *,
        deletion_manifest: Any = None,
        injections: tuple[Any, ...] = (),
    ) -> None:
        self.base_db_manager = base_db_manager
        self.state = state
        self.target_fact = target_fact
        self.injections = tuple(injections)

        # Closure-manifest deletion (parity with the Co-LMLM backend):
        # explicit triple-ID exclusion plus the value-level semantic
        # backstop, judged against the *target* fact's answer.
        self.use_closure = False
        self.excluded_entry_ids: frozenset[str] = frozenset()
        self.backstop_answer: tuple[str, tuple[str, ...]] | None = None
        if (
            deletion_manifest is not None
            and getattr(deletion_manifest, "strategy", None) == "closure"
        ):
            self.use_closure = True
            self.excluded_entry_ids = frozenset(deletion_manifest.entry_ids)
            metadata = (
                deletion_manifest.metadata
                if isinstance(deletion_manifest.metadata, Mapping)
                else {}
            )
            predicates = metadata.get("predicates_active")
            semantic_target = metadata.get("semantic_target")
            if (
                isinstance(predicates, (list, tuple))
                and "semantic" in predicates
                and isinstance(semantic_target, Mapping)
            ):
                self.backstop_answer = (
                    str(semantic_target.get("ground_truth", "")),
                    tuple(
                        str(alias)
                        for alias in (
                            semantic_target.get("object_aliases") or ()
                        )
                    ),
                )

        self.database_name = getattr(base_db_manager, "database_name", None)
        self.database_org_file = getattr(base_db_manager, "database_org_file", [])
        self.database = getattr(base_db_manager, "database", {})
        self.topk_retriever = getattr(base_db_manager, "topk_retriever", None)
        self.last_trace: dict[str, Any] | None = None
        self.captured_query_embeddings: list[np.ndarray | None] = []

    def _is_candidate_deleted(self, triplet: tuple[str, str, str]) -> bool:
        if self.use_closure:
            if triple_id(*triplet) in self.excluded_entry_ids:
                return True
            if self.backstop_answer is not None:
                answer, aliases = self.backstop_answer
                return values_equivalent(
                    triplet[2], answer, right_aliases=aliases
                )
            return False
        return self.target_fact is not None and is_deleted_triplet(
            triplet, self.target_fact
        )

    def init_topk_retriever(self, *args: Any, **kwargs: Any) -> None:
        if getattr(self.base_db_manager, "topk_retriever", None) is None:
            self.base_db_manager.init_topk_retriever(*args, **kwargs)
        self.topk_retriever = self.base_db_manager.topk_retriever

    def reset_trace(self) -> None:
        self.last_trace = None
        self.captured_query_embeddings = []

    def _candidate_trace_entry(
        self,
        candidate: tuple[str, str, str, float],
    ) -> dict[str, Any]:
        subject, relation, obj, score = candidate
        matches_subject = False
        matches_relation = False
        matches_object = False
        supports_target_fact = False
        if self.target_fact is not None:
            (
                matches_subject,
                matches_relation,
                matches_object,
                supports_target_fact,
            ) = candidate_supports_target_fact(candidate[:3], self.target_fact)

        return {
            "entry_id": triple_id(subject, relation, obj),
            "subject": subject,
            "relation": relation,
            "object": obj,
            "value": obj,
            "score": score,
            "matches_subject": matches_subject,
            "matches_relation": matches_relation,
            "matches_object": matches_object,
            "supports_target_fact": supports_target_fact,
            "matches_deleted_fact": (
                self.target_fact is not None
                and is_deleted_triplet(candidate[:3], self.target_fact)
            ),
        }

    def _injected_trace_entry(
        self, injection: Any, score: float
    ) -> dict[str, Any]:
        return {
            "entry_id": str(injection.entry_id),
            "subject": "<synthetic>",
            "relation": str(injection.template),
            "object": str(injection.value),
            "value": str(injection.value),
            "score": score,
            "synthetic": True,
            "target_cosine": float(injection.target_cosine),
            "matches_subject": False,
            "matches_relation": False,
            "matches_object": False,
            "supports_target_fact": False,
            "matches_deleted_fact": False,
        }

    def _scored_injections(
        self,
        query_embedding: np.ndarray | None,
        threshold: float | None,
    ) -> list[tuple[Any, float]]:
        if not self.injections or query_embedding is None:
            return []
        norm = float(np.linalg.norm(query_embedding))
        if norm == 0.0:
            return []
        unit_query = query_embedding / norm
        effective_threshold = (
            threshold
            if threshold is not None
            else getattr(self.topk_retriever, "default_threshold", None)
        )
        scored: list[tuple[Any, float]] = []
        for injection in self.injections:
            vector = np.asarray(injection.vector, dtype=np.float32).reshape(-1)
            if vector.size != unit_query.size:
                continue
            score = float(np.dot(unit_query, vector))
            if effective_threshold is not None and score < effective_threshold:
                continue
            scored.append((injection, score))
        return scored

    def retrieve_from_database(
        self, prompt: str, threshold: float | None = None
    ) -> str:
        trace: dict[str, Any] = {
            "state": self.state.value,
            "retrieval_enabled": True,
            "lookup_query": None,
            "threshold": threshold,
            "all_candidates": [],
            "deleted_candidates": [],
            "retained_candidates": [],
            "selected_candidate": None,
            "selected_value": None,
            "error": None,
        }
        is_passthrough_state = self.state is DatabaseState.FULL or (
            self.target_fact is None and not self.use_closure
        )

        query_embedding: np.ndarray | None = None
        scored_injections: list[tuple[Any, float]] = []
        try:
            entity, relationship = extract_lookup_query(prompt)
            trace["lookup_query"] = {
                "entity": entity.strip(),
                "relation": relationship.strip(),
            }

            self.init_topk_retriever()
            candidates, query_embedding = retrieve_triplet_candidates(
                self.topk_retriever,
                entity=entity,
                relation=relationship,
                threshold=threshold,
            )
            self.captured_query_embeddings.append(query_embedding)
            scored_injections = self._scored_injections(
                query_embedding, threshold
            )
            trace["all_candidates"] = [
                self._candidate_trace_entry(candidate) for candidate in candidates
            ] + [
                self._injected_trace_entry(injection, score)
                for injection, score in scored_injections
            ]
        except Exception as exc:
            trace["error"] = str(exc)
            if is_passthrough_state:
                value = self.base_db_manager.retrieve_from_database(
                    prompt, threshold=threshold
                )
                trace["selected_value"] = value
                self.last_trace = trace
                return value
            self.last_trace = trace
            raise

        if is_passthrough_state:
            value = self.base_db_manager.retrieve_from_database(
                prompt, threshold=threshold
            )
            trace["retained_candidates"] = trace["all_candidates"]
            trace["selected_value"] = value

            selected_candidate = next(
                (
                    candidate
                    for candidate in trace["retained_candidates"]
                    if candidate["object"] == value
                ),
                None,
            )
            trace["selected_candidate"] = selected_candidate
            self.last_trace = trace
            return value

        # Real candidates and scored injections compete in one ranking; the
        # deletion predicates apply uniformly (injections carry synthetic
        # identities, so only the semantic backstop can catch them).
        pool: list[tuple[str, Any, float]] = [
            ("real", candidate, candidate[3]) for candidate in candidates
        ] + [
            ("synthetic", injection, score)
            for injection, score in scored_injections
        ]
        pool.sort(key=lambda item: -item[2])

        deleted_entries: list[dict[str, Any]] = []
        retained: list[tuple[str, Any, float]] = []
        for kind, item, score in pool:
            if kind == "real":
                excluded = self._is_candidate_deleted(item[:3])
                entry = self._candidate_trace_entry(item)
            else:
                excluded = self._is_candidate_deleted(
                    ("<synthetic>", str(item.template), str(item.value))
                )
                entry = self._injected_trace_entry(item, score)
            if excluded:
                deleted_entries.append(entry)
            else:
                retained.append((kind, item, score))

        trace["deleted_candidates"] = deleted_entries
        trace["retained_candidates"] = [
            self._candidate_trace_entry(item)
            if kind == "real"
            else self._injected_trace_entry(item, score)
            for kind, item, score in retained
        ]

        if not retained:
            trace["error"] = (
                f"No retrieval results for entity={entity!r}, relationship={relationship!r}"
            )
            self.last_trace = trace
            raise ValueError(trace["error"])

        kind, selected, score = retained[0]
        if kind == "real":
            trace["selected_candidate"] = self._candidate_trace_entry(selected)
            trace["selected_value"] = selected[2]
        else:
            trace["selected_candidate"] = self._injected_trace_entry(
                selected, score
            )
            trace["selected_value"] = str(selected.value)
        self.last_trace = trace
        return trace["selected_value"]


def build_state_db_manager(
    base_db_manager: Any,
    prompt_row: dict[str, Any],
    state: DatabaseState,
    *,
    deletion_manifest: Any = None,
    injections: tuple[Any, ...] = (),
) -> Any:
    has_closure = (
        deletion_manifest is not None
        and getattr(deletion_manifest, "strategy", None) == "closure"
    )
    target_fact = None
    if (
        prompt_row.get("subject") is not None
        and prompt_row.get("relation") is not None
    ):
        target_fact = target_fact_from_prompt_row(prompt_row)
    elif state is not DatabaseState.FULL and not has_closure:
        raise ValueError(
            "Deleted states need either a subject/relation target fact or a "
            "closure deletion manifest."
        )
    return AuditDatabaseManager(
        base_db_manager=base_db_manager,
        state=state,
        target_fact=target_fact,
        deletion_manifest=deletion_manifest,
        injections=injections,
    )
