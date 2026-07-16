from __future__ import annotations

import dataclasses
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from halo.interventions.judge import default_support_judge
from halo.interventions.errors import AuditIntegrationError
from halo.interventions.filtering import (
    _candidate_id,
    _candidate_score,
    _candidate_source_id,
    _candidate_text,
)
from halo.core.examples import AuditExample, DeletionManifest

VALID_PREDICATES = ("geometric", "semantic", "provenance")


@dataclass(frozen=True)
class ClosureConfig:
    predicates: tuple[str, ...] = VALID_PREDICATES
    radius: float = 0.85
    envelope_top_k: int = 500
    max_closure_size: int = 10_000

    def __post_init__(self) -> None:
        object.__setattr__(self, "predicates", tuple(dict.fromkeys(self.predicates)))
        unknown = [
            predicate
            for predicate in self.predicates
            if predicate not in VALID_PREDICATES
        ]
        if unknown:
            raise ValueError(
                f"Unknown closure predicates {unknown!r}; "
                f"valid predicates are {list(VALID_PREDICATES)!r}."
            )
        if not self.predicates:
            raise ValueError("At least one closure predicate is required.")
        if not -1.0 <= self.radius <= 1.0:
            raise ValueError("radius must be a cosine similarity in [-1, 1].")
        if self.envelope_top_k < 1:
            raise ValueError("envelope_top_k must be at least 1.")
        if self.max_closure_size < 1:
            raise ValueError("max_closure_size must be at least 1.")

    def is_active(self, predicate: str) -> bool:
        return predicate in self.predicates


@dataclass(frozen=True)
class ClosureEntry:
    entry_id: str
    score: float | None
    source_id: str | None
    value: str
    caught_by: tuple[str, ...]


@dataclass(frozen=True)
class ClosureResult:
    example_key: str
    entries: tuple[ClosureEntry, ...]
    source_ids: tuple[str, ...]
    truncated: bool
    config: ClosureConfig
    index_nprobe: int | None = None
    target_answer: str = ""
    target_aliases: tuple[str, ...] = ()
    # Geometry for the margin predictor: the best deleted score and the best
    # score among observed candidates that survive this closure.
    s_del: float | None = None
    s_surv: float | None = None
    top_survivors: tuple[ClosureEntry, ...] = ()

    @property
    def margin(self) -> float | None:
        if self.s_del is None or self.s_surv is None:
            return None
        return self.s_del - self.s_surv

    def to_manifest(self) -> DeletionManifest:
        entry_counts = {
            predicate: sum(1 for entry in self.entries if predicate in entry.caught_by)
            for predicate in (*VALID_PREDICATES, "oracle")
        }
        # Per-entry attribution stays out of the manifest (it is embedded in
        # every JSONL result row); the full listing goes to the closure
        # artifact instead.
        metadata: dict[str, Any] = {
            "predicates_active": list(self.config.predicates),
            "radius": self.config.radius,
            "envelope_top_k": self.config.envelope_top_k,
            "max_closure_size": self.config.max_closure_size,
            "truncated": self.truncated,
            "entry_counts": entry_counts,
            "index_nprobe": self.index_nprobe,
        }
        if self.config.is_active("semantic"):
            # The run-time backstop must judge candidates against the target
            # fact's answer — not the answer of whichever prompt happens to
            # run under this manifest (neighbor prompts in a sweep).
            metadata["semantic_target"] = {
                "ground_truth": self.target_answer,
                "object_aliases": list(self.target_aliases),
            }
        return DeletionManifest(
            entry_ids=tuple(entry.entry_id for entry in self.entries),
            source_ids=self.source_ids,
            strategy="closure",
            metadata=metadata,
        )

    def as_dict(self) -> dict[str, Any]:
        manifest = self.to_manifest()
        return {
            "example_key": self.example_key,
            "manifest_id": manifest.manifest_id,
            "strategy": "closure",
            "predicates_active": list(self.config.predicates),
            "radius": self.config.radius,
            "envelope_top_k": self.config.envelope_top_k,
            "max_closure_size": self.config.max_closure_size,
            "truncated": self.truncated,
            "index_nprobe": self.index_nprobe,
            "s_del": self.s_del,
            "s_surv": self.s_surv,
            "margin": self.margin,
            "top_survivors": [
                {
                    "entry_id": entry.entry_id,
                    "score": entry.score,
                    "source_id": entry.source_id,
                    "value": entry.value,
                }
                for entry in self.top_survivors
            ],
            "source_ids": list(self.source_ids),
            "entries": [
                {
                    "entry_id": entry.entry_id,
                    "score": entry.score,
                    "source_id": entry.source_id,
                    "value": entry.value,
                    "caught_by": list(entry.caught_by),
                }
                for entry in self.entries
            ],
        }


def _flatten_single_query(raw: Any) -> list[Any]:
    if raw and isinstance(raw, list) and isinstance(raw[0], list):
        if len(raw) != 1:
            raise AuditIntegrationError(
                "The closure builder issued a single query but the index "
                f"returned {len(raw)} result lists."
            )
        return list(raw[0])
    return list(raw or [])


def _index_nprobe(index: Any) -> int | None:
    config = getattr(index, "config", None)
    value = getattr(config, "ivf_nprobe", None)
    return int(value) if isinstance(value, int) else None


def build_closure_family(
    *,
    index: Any,
    example: AuditExample,
    query_vector: Any | None,
    config: ClosureConfig,
    radii: tuple[float, ...],
    seed_candidates: tuple[Any, ...] = (),
    seed_source_ids: tuple[str, ...] = (),
    support_judge: Callable[[Any, AuditExample], Mapping[str, Any]] = (
        default_support_judge
    ),
    example_key: str = "",
) -> dict[float, ClosureResult]:
    """Closures at several radii from a single index search.

    Geometric closure sets are nested as the radius shrinks, so one search at
    the smallest radius yields per-radius membership by score; semantic,
    provenance, and oracle members are radius-independent.
    """
    if not radii:
        raise ValueError("At least one closure radius is required.")
    radii = tuple(dict.fromkeys(float(radius) for radius in radii))
    for radius in radii:
        if not -1.0 <= radius <= 1.0:
            raise ValueError("radius must be a cosine similarity in [-1, 1].")

    needs_search = config.is_active("geometric") or config.is_active("semantic")
    if needs_search and query_vector is None:
        raise ValueError(
            "Geometric/semantic closure predicates need the FULL query "
            "embedding, but none was captured for this fact."
        )

    shared_caught: dict[str, set[str]] = {}
    details: dict[str, dict[str, Any]] = {}

    def note(target: dict[str, set[str]], candidate: Any, predicate: str) -> None:
        entry_id = _candidate_id(candidate)
        if not entry_id:
            return
        target.setdefault(entry_id, set()).add(predicate)
        details.setdefault(
            entry_id,
            {
                "score": _candidate_score(candidate),
                "source_id": _candidate_source_id(candidate),
                "value": _candidate_text(candidate),
            },
        )

    query = (
        np.asarray(query_vector, dtype=np.float32).reshape(-1) if needs_search else None
    )

    geometric_hits: list[Any] = []
    page_full = False
    last_score: float | None = None
    if config.is_active("geometric"):
        geometric_hits = _flatten_single_query(
            index.search(
                query,
                top_k=config.max_closure_size,
                similarity_threshold=min(radii),
            )
        )
        # A full page means the smallest radius may hold more entries than we
        # fetched; the closure must never silently read as complete. Larger
        # radii are only affected once they dip below the lowest fetched
        # score (search returns the top of the ranking first).
        page_full = len(geometric_hits) >= config.max_closure_size
        scores = [
            score
            for score in (_candidate_score(candidate) for candidate in geometric_hits)
            if score is not None
        ]
        last_score = min(scores) if scores else None

    envelope: list[Any] = []
    if config.is_active("semantic"):
        envelope = _flatten_single_query(
            index.search(
                query,
                top_k=config.envelope_top_k,
                similarity_threshold=None,
            )
        )
        for candidate in envelope:
            if dict(support_judge(candidate, example)).get("supports_target"):
                note(shared_caught, candidate, "semantic")

    source_ids = (
        tuple(dict.fromkeys(str(value) for value in seed_source_ids))
        if config.is_active("provenance")
        else ()
    )
    if source_ids:
        # Attribution only: the actual provenance exclusion happens by
        # source ID at search time, unbounded by what we saw here.
        for candidate in envelope:
            candidate_source = _candidate_source_id(candidate)
            if candidate_source is not None and candidate_source in source_ids:
                note(shared_caught, candidate, "provenance")

    for candidate in seed_candidates:
        note(shared_caught, candidate, "oracle")

    def sort_key(entry_id: str) -> tuple[int, float, str]:
        score = details[entry_id]["score"]
        return (0 if score is not None else 1, -(score or 0.0), entry_id)

    # Every observed candidate with a score, for the margin geometry
    # (s_del / s_surv). Envelope and geometric pages overlap; keep the best
    # score per entry.
    observed_scores: dict[str, float] = {}
    for candidate in (*geometric_hits, *envelope):
        entry_id = _candidate_id(candidate)
        score = _candidate_score(candidate)
        if not entry_id or score is None:
            continue
        if entry_id not in observed_scores or score > observed_scores[entry_id]:
            observed_scores[entry_id] = score
            details.setdefault(
                entry_id,
                {
                    "score": score,
                    "source_id": _candidate_source_id(candidate),
                    "value": _candidate_text(candidate),
                },
            )

    family: dict[float, ClosureResult] = {}
    for radius in radii:
        caught = {
            entry_id: set(predicates) for entry_id, predicates in shared_caught.items()
        }
        for candidate in geometric_hits:
            score = _candidate_score(candidate)
            # A missing score cannot be compared against the radius; err
            # toward deletion at every radius.
            if score is None or score >= radius:
                note(caught, candidate, "geometric")
        truncated = page_full and (last_score is None or radius <= last_score)
        deleted_scores = [
            observed_scores[entry_id]
            for entry_id in caught
            if entry_id in observed_scores
        ]
        survivors = sorted(
            (
                (entry_id, score)
                for entry_id, score in observed_scores.items()
                if entry_id not in caught
                and details[entry_id]["source_id"] not in source_ids
            ),
            key=lambda item: (-item[1], item[0]),
        )
        top_survivors = tuple(
            ClosureEntry(
                entry_id=entry_id,
                score=score,
                source_id=details[entry_id]["source_id"],
                value=details[entry_id]["value"],
                caught_by=(),
            )
            for entry_id, score in survivors[:5]
        )
        entries = tuple(
            ClosureEntry(
                entry_id=entry_id,
                score=details[entry_id]["score"],
                source_id=details[entry_id]["source_id"],
                value=details[entry_id]["value"],
                caught_by=tuple(sorted(caught[entry_id])),
            )
            for entry_id in sorted(caught, key=sort_key)
        )
        family[radius] = ClosureResult(
            example_key=example_key,
            entries=entries,
            source_ids=source_ids,
            truncated=truncated,
            config=dataclasses.replace(config, radius=radius),
            index_nprobe=_index_nprobe(index),
            target_answer=example.ground_truth,
            target_aliases=tuple(example.object_aliases),
            s_del=max(deleted_scores) if deleted_scores else None,
            s_surv=survivors[0][1] if survivors else None,
            top_survivors=top_survivors,
        )
    return family


def build_closure(
    *,
    index: Any,
    example: AuditExample,
    query_vector: Any | None,
    config: ClosureConfig,
    seed_candidates: tuple[Any, ...] = (),
    seed_source_ids: tuple[str, ...] = (),
    support_judge: Callable[[Any, AuditExample], Mapping[str, Any]] = (
        default_support_judge
    ),
    example_key: str = "",
) -> ClosureResult:
    family = build_closure_family(
        index=index,
        example=example,
        query_vector=query_vector,
        config=config,
        radii=(config.radius,),
        seed_candidates=seed_candidates,
        seed_source_ids=seed_source_ids,
        support_judge=support_judge,
        example_key=example_key,
    )
    return family[config.radius]


def _example_key(example: AuditExample) -> str:
    for value in (example.prompt_id, example.fact_id):
        if value is not None:
            return str(value)
    return "fact"


def _safe_filename(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", key)


def write_closure_artifact(closure: ClosureResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(closure.as_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def full_selected_candidate(
    full_result: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    trace = full_result.get("retrieval_trace") or {}
    selected = trace.get("selected_candidate")
    return selected if isinstance(selected, Mapping) else None


def full_query_vector(full_result: Mapping[str, Any]) -> Any | None:
    """The captured query vector for the FULL retrieval event whose
    candidate was selected (falling back to the first captured event)."""
    trace = full_result.get("retrieval_trace") or {}
    selected_event_index = None
    for event in trace.get("retrieval_events") or []:
        if event.get("selected_candidate"):
            selected_event_index = event.get("event_index")
            break

    for item in full_result.get("_query_embeddings") or []:
        if (
            selected_event_index is None
            or item.get("event_index") == selected_event_index
        ):
            return item.get("vector")
    return None


def build_closure_manifest_from_full(
    *,
    index: Any,
    example: AuditExample,
    full_result: Mapping[str, Any],
    config: ClosureConfig,
    support_judge: Callable[[Any, AuditExample], Mapping[str, Any]] = (
        default_support_judge
    ),
    artifact_dir: Path | None = None,
) -> DeletionManifest:
    selected = full_selected_candidate(full_result) or {}
    seed_entry_id = selected.get("entry_id")
    if not seed_entry_id:
        raise ValueError("FULL produced no selected entry; cannot build a closure.")

    query_vector = full_query_vector(full_result)
    seed_source = selected.get("source_id")
    key = _example_key(example)
    closure = build_closure(
        index=index,
        example=example,
        query_vector=query_vector,
        config=config,
        seed_candidates=(selected,),
        seed_source_ids=(str(seed_source),) if seed_source is not None else (),
        support_judge=support_judge,
        example_key=key,
    )
    if artifact_dir is not None:
        write_closure_artifact(closure, artifact_dir / f"{_safe_filename(key)}.json")
    return closure.to_manifest()
