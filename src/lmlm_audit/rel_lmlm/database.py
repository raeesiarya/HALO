import re
from dataclasses import dataclass
from typing import Any

from lmlm_audit.core.equivalence import prompt_row_aliases, values_equivalent
from lmlm_audit.core.states import DatabaseState


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
) -> list[tuple[str, str, str, float]]:
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
    return results


class AuditDatabaseManager:
    def __init__(
        self,
        base_db_manager: Any,
        state: DatabaseState,
        target_fact: TargetFact | None = None,
    ) -> None:
        self.base_db_manager = base_db_manager
        self.state = state
        self.target_fact = target_fact

        self.database_name = getattr(base_db_manager, "database_name", None)
        self.database_org_file = getattr(base_db_manager, "database_org_file", [])
        self.database = getattr(base_db_manager, "database", {})
        self.topk_retriever = getattr(base_db_manager, "topk_retriever", None)
        self.last_trace: dict[str, Any] | None = None

    def init_topk_retriever(self, *args: Any, **kwargs: Any) -> None:
        if getattr(self.base_db_manager, "topk_retriever", None) is None:
            self.base_db_manager.init_topk_retriever(*args, **kwargs)
        self.topk_retriever = self.base_db_manager.topk_retriever

    def reset_trace(self) -> None:
        self.last_trace = None

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
            "subject": subject,
            "relation": relation,
            "object": obj,
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
        is_passthrough_state = (
            self.state is DatabaseState.FULL or self.target_fact is None
        )

        try:
            entity, relationship = extract_lookup_query(prompt)
            trace["lookup_query"] = {
                "entity": entity.strip(),
                "relation": relationship.strip(),
            }

            self.init_topk_retriever()
            candidates = retrieve_triplet_candidates(
                self.topk_retriever,
                entity=entity,
                relation=relationship,
                threshold=threshold,
            )
            trace["all_candidates"] = [
                self._candidate_trace_entry(candidate) for candidate in candidates
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

        remaining_candidates: list[tuple[str, str, str, float]] = []
        deleted_candidates: list[tuple[str, str, str, float]] = []
        for candidate in candidates:
            if is_deleted_triplet(candidate[:3], self.target_fact):
                deleted_candidates.append(candidate)
            else:
                remaining_candidates.append(candidate)

        trace["deleted_candidates"] = [
            self._candidate_trace_entry(candidate) for candidate in deleted_candidates
        ]
        trace["retained_candidates"] = [
            self._candidate_trace_entry(candidate) for candidate in remaining_candidates
        ]

        if not remaining_candidates:
            trace["error"] = (
                f"No retrieval results for entity={entity!r}, relationship={relationship!r}"
            )
            self.last_trace = trace
            raise ValueError(trace["error"])

        selected_candidate = remaining_candidates[0]
        trace["selected_candidate"] = self._candidate_trace_entry(selected_candidate)
        trace["selected_value"] = selected_candidate[2]
        self.last_trace = trace
        return selected_candidate[2]


def build_state_db_manager(
    base_db_manager: Any,
    prompt_row: dict[str, Any],
    state: DatabaseState,
) -> Any:
    return AuditDatabaseManager(
        base_db_manager=base_db_manager,
        state=state,
        target_fact=target_fact_from_prompt_row(prompt_row),
    )
