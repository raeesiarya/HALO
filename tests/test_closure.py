import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from halo.core.backend import audit_example
from models.co_lmlm.backend import CoLMLMAuditBackend
from halo.interventions.closure import (
    ClosureConfig,
    build_closure,
    build_closure_manifest_from_full,
)
from halo.core.examples import AuditExample
from halo.cli.runner import run_backend_audit
from halo.core.states import DatabaseState


@dataclass
class FakeSearchResult:
    id: str
    score: float
    text_value: str
    text_key: str | None = None
    metadata: dict = field(default_factory=dict)


def _unit_vector(cosine: float) -> np.ndarray:
    return np.asarray(
        [cosine, math.sqrt(max(0.0, 1.0 - cosine * cosine))], dtype=np.float32
    )


@dataclass
class FakeVectorEntry:
    entry_id: str
    vector: np.ndarray
    text: str
    source_id: str


class FakeVectorIndex:
    """Cosine search over explicit vectors, mimicking the public index API."""

    def __init__(self, entries: list[FakeVectorEntry]) -> None:
        self.entries = list(entries)
        self.calls: list[tuple[int, float | None]] = []

    def search(self, query, top_k=1, similarity_threshold=None):
        self.calls.append((top_k, similarity_threshold))
        normalized_query = np.asarray(query, dtype=np.float32).reshape(-1)
        normalized_query = normalized_query / np.linalg.norm(normalized_query)
        results = []
        for entry in self.entries:
            vector = entry.vector / np.linalg.norm(entry.vector)
            score = float(np.dot(normalized_query, vector))
            if similarity_threshold is None or score >= similarity_threshold:
                results.append(
                    FakeSearchResult(
                        id=entry.entry_id,
                        score=score,
                        text_value=entry.text,
                        metadata={"sample_id": entry.source_id},
                    )
                )
        results.sort(key=lambda result: -result.score)
        return results[:top_k]


def _example() -> AuditExample:
    return AuditExample.from_prompt_row(
        {
            "prompt_id": "capital-direct",
            "fact_id": "france-capital",
            "prompt_text": "What is the capital of France?",
            "gold_object": "Paris",
        }
    )


def _vector_index() -> FakeVectorIndex:
    return FakeVectorIndex(
        [
            FakeVectorEntry("target-entry", _unit_vector(0.95), "Paris", "wiki:France"),
            FakeVectorEntry("near-neighbor", _unit_vector(0.90), "Lyon", "wiki:Lyon"),
            FakeVectorEntry(
                "alias-entry",
                _unit_vector(0.50),
                "The capital is Paris",
                "wiki:ParisCity",
            ),
            FakeVectorEntry("far-entry", _unit_vector(0.10), "Berlin", "wiki:France"),
        ]
    )


QUERY = _unit_vector(1.0)


def _seed_candidate() -> dict:
    return {
        "entry_id": "target-entry",
        "score": 0.95,
        "value": "Paris",
        "metadata": {"sample_id": "wiki:France"},
    }


def test_full_closure_attributes_each_predicate() -> None:
    closure = build_closure(
        index=_vector_index(),
        example=_example(),
        query_vector=QUERY,
        config=ClosureConfig(radius=0.8),
        seed_candidates=(_seed_candidate(),),
        seed_source_ids=("wiki:France",),
        example_key="capital-direct",
    )

    caught = {entry.entry_id: entry.caught_by for entry in closure.entries}
    assert caught == {
        "target-entry": ("geometric", "oracle", "provenance", "semantic"),
        "near-neighbor": ("geometric",),
        "alias-entry": ("semantic",),
        "far-entry": ("provenance",),
    }
    assert closure.source_ids == ("wiki:France",)
    assert closure.truncated is False

    manifest = closure.to_manifest()
    assert manifest.strategy == "closure"
    assert manifest.entry_ids == (
        "alias-entry",
        "far-entry",
        "near-neighbor",
        "target-entry",
    )
    assert manifest.source_ids == ("wiki:France",)
    assert manifest.metadata["entry_counts"] == {
        "geometric": 2,
        "semantic": 2,
        "provenance": 2,
        "oracle": 1,
    }
    assert manifest.metadata["truncated"] is False


def test_geometric_truncation_is_flagged() -> None:
    closure = build_closure(
        index=_vector_index(),
        example=_example(),
        query_vector=QUERY,
        config=ClosureConfig(predicates=("geometric",), radius=0.8, max_closure_size=1),
    )

    assert closure.truncated is True
    assert [entry.entry_id for entry in closure.entries] == ["target-entry"]
    assert closure.to_manifest().metadata["truncated"] is True


def test_provenance_only_closure_never_searches() -> None:
    index = _vector_index()
    closure = build_closure(
        index=index,
        example=_example(),
        query_vector=None,
        config=ClosureConfig(predicates=("provenance",)),
        seed_candidates=(_seed_candidate(),),
        seed_source_ids=("wiki:France",),
    )

    assert index.calls == []
    assert closure.source_ids == ("wiki:France",)
    assert [entry.entry_id for entry in closure.entries] == ["target-entry"]


def test_geometric_predicate_requires_query_vector() -> None:
    with pytest.raises(ValueError, match="query embedding"):
        build_closure(
            index=_vector_index(),
            example=_example(),
            query_vector=None,
            config=ClosureConfig(predicates=("geometric",)),
        )


def test_config_rejects_unknown_predicates_and_bad_radius() -> None:
    with pytest.raises(ValueError, match="Unknown closure predicates"):
        ClosureConfig(predicates=("geometric", "psychic"))
    with pytest.raises(ValueError, match="radius"):
        ClosureConfig(radius=1.5)
    with pytest.raises(ValueError, match="predicate is required"):
        ClosureConfig(predicates=())


def test_build_closure_manifest_from_full_writes_artifact(tmp_path) -> None:
    full_result = {
        "retrieval_trace": {
            "selected_candidate": _seed_candidate() | {"source_id": "wiki:France"},
            "retrieval_events": [
                {"event_index": 0, "selected_candidate": _seed_candidate()}
            ],
        },
        "_query_embeddings": [{"event_index": 0, "vector": QUERY}],
    }

    manifest = build_closure_manifest_from_full(
        index=_vector_index(),
        example=_example(),
        full_result=full_result,
        config=ClosureConfig(radius=0.8),
        artifact_dir=tmp_path,
    )

    assert manifest.strategy == "closure"
    assert "near-neighbor" in manifest.entry_ids

    artifact_path = tmp_path / "capital-direct.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["manifest_id"] == manifest.manifest_id
    assert artifact["source_ids"] == ["wiki:France"]
    caught = {entry["entry_id"]: entry["caught_by"] for entry in artifact["entries"]}
    assert caught["near-neighbor"] == ["geometric"]
    assert "oracle" in caught["target-entry"]


def test_manifest_from_full_requires_a_selected_entry() -> None:
    with pytest.raises(ValueError, match="no selected entry"):
        build_closure_manifest_from_full(
            index=_vector_index(),
            example=_example(),
            full_result={"retrieval_trace": {}},
            config=ClosureConfig(),
        )


class FakeGenerator:
    def __init__(self, index) -> None:
        self.index = index
        self.generation_config = SimpleNamespace(max_new_tokens=64)
        self.retrieval_config = SimpleNamespace(similarity_threshold=0.7)

    def generate(self, prompt):
        results = self.index.search(
            QUERY,
            top_k=1,
            similarity_threshold=self.retrieval_config.similarity_threshold,
        )
        if not results:
            return SimpleNamespace(
                text=f"{prompt} unknown.",
                num_retrievals=0,
                failed_retrievals=1,
            )
        selected = results[0]
        return SimpleNamespace(
            text=(f"{prompt}<FACT>{selected.text_value}</FACT> {selected.text_value}."),
            num_retrievals=1,
            failed_retrievals=0,
        )


def test_semantic_backstop_nulls_supporting_candidates_missed_by_closure() -> None:
    # alias-entry sits above the generation threshold so it would be
    # retrieved once target-entry is deleted — unless the backstop fires.
    index = FakeVectorIndex(
        [
            FakeVectorEntry("target-entry", _unit_vector(0.95), "Paris", "wiki:France"),
            FakeVectorEntry(
                "alias-entry",
                _unit_vector(0.93),
                "Paris, the city of light",
                "wiki:ParisCity",
            ),
            FakeVectorEntry("near-neighbor", _unit_vector(0.90), "Lyon", "wiki:Lyon"),
        ]
    )
    backend = CoLMLMAuditBackend(FakeGenerator(index))
    example = AuditExample.from_prompt_row(
        {
            "prompt_id": "capital-direct",
            "prompt_text": "What is the capital of France?",
            "gold_object": "Paris",
            "deletion_manifest": {
                # Closure deliberately misses alias-entry; the backstop must
                # catch it because the semantic predicate is active.
                "entry_ids": ["target-entry"],
                "strategy": "closure",
                "metadata": {"predicates_active": ["semantic"]},
            },
        }
    )

    result = audit_example(backend, example, DatabaseState.DEL_ON)
    trace = result["retrieval_trace"]
    assert result["model_output"] == "Lyon"
    deleted_ids = {item["entry_id"] for item in trace["deleted_candidates"]}
    assert "target-entry" in deleted_ids
    assert "alias-entry" in deleted_ids
    assert trace["retrieval_events"][0]["exclude_supporting"] is True


def test_runner_builds_closure_manifest_from_full(tmp_path) -> None:
    index = _vector_index()
    backend = CoLMLMAuditBackend(FakeGenerator(index))
    prompt_path = tmp_path / "prompts.jsonl"
    prompt_path.write_text(
        '{"prompt_id":"p1","fact_id":"f1","prompt_text":"What is the capital of France?",'
        '"gold_object":"Paris"}\n',
        encoding="utf-8",
    )
    artifact_dir = tmp_path / "closures"

    def manifest_builder(example, full_result):
        return build_closure_manifest_from_full(
            index=index,
            example=example,
            full_result=full_result,
            config=ClosureConfig(radius=0.8),
            artifact_dir=artifact_dir,
        )

    results = run_backend_audit(
        prompt_path=prompt_path,
        backend=backend,
        states=[
            DatabaseState.FULL,
            DatabaseState.DEL_ON,
            DatabaseState.DEL_OFF,
        ],
        bootstrap_oracle_from_full=True,
        manifest_builder=manifest_builder,
    )

    json.dumps(results)
    full_row, del_on_row, del_off_row = results
    manifest = full_row["deletion_manifest"]
    assert manifest["strategy"] == "closure"
    assert set(manifest["entry_ids"]) == {
        "target-entry",
        "near-neighbor",
        "alias-entry",
        "far-entry",
    }
    assert manifest["source_ids"] == ["wiki:France"]
    assert del_on_row["deletion_manifest"] == manifest
    assert del_off_row["deletion_manifest"] == manifest

    # Both candidates above the generation threshold are in the closure, so
    # DEL-ON retains nothing and falls back to parametric decoding.
    del_on_deleted = {
        item["entry_id"] for item in del_on_row["retrieval_trace"]["deleted_candidates"]
    }
    assert del_on_deleted == {"target-entry", "near-neighbor"}
    assert del_on_row["model_output"] == "unknown"
    assert (artifact_dir / "p1.json").is_file()
