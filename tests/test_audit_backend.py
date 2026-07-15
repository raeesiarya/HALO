import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lmlm_audit.core.backend import (
    AuditObservation,
    audit_example,
    default_retrieval_trace,
    validate_intervention_results,
)
from lmlm_audit.core.examples import AuditExample, DeletionManifest
from lmlm_audit.core.states import DatabaseState


class FakeBackend:
    def __init__(self) -> None:
        self.calls = []

    def generate(self, example, state, *, max_new_tokens=12):
        self.calls.append((example, state, max_new_tokens))
        return AuditObservation(
            model_output="Paris",
            retrieval_trace={
                "state": state.value,
                "retrieval_enabled": state is not DatabaseState.DEL_OFF,
                "retrieval_triggered": True,
                "retained_candidates": [],
            },
            generation_metadata={"answer_source": "retrieved_value"},
        )


def test_example_requires_prompt_and_ground_truth() -> None:
    with pytest.raises(ValueError, match="prompt_text"):
        AuditExample.from_prompt_row({"gold_object": "Paris"})

    with pytest.raises(ValueError, match="gold_object"):
        AuditExample.from_prompt_row({"prompt_text": "Capital?"})


def test_example_allows_schema_free_colmlm_target() -> None:
    example = AuditExample.from_prompt_row(
        {
            "fact_id": "france-capital",
            "prompt_text": "What is the capital of France?",
            "gold_object": "Paris",
            "answer_aliases": ["Paris, France"],
            "source_ids": ["wiki:France"],
            "oracle_entry_ids": ["wiki:France:17"],
        }
    )

    assert example.subject is None
    assert example.relation is None
    assert example.object_aliases == ("Paris, France",)
    assert example.source_row["source_ids"] == ["wiki:France"]
    assert example.deletion_manifest.entry_ids == ("wiki:France:17",)
    assert example.deletion_manifest.source_ids == ("wiki:France",)


def test_nested_deletion_manifest_is_stable_and_serializable() -> None:
    manifest = DeletionManifest.from_prompt_row(
        {
            "deletion_manifest": {
                "entry_ids": ["entry-2", "entry-1"],
                "source_ids": ["source-1"],
                "strategy": "oracle-plus-source",
                "metadata": {"radius": 0.2},
            }
        }
    )

    serialized = manifest.as_dict()
    assert serialized["entry_ids"] == ["entry-1", "entry-2"]
    assert serialized["strategy"] == "oracle-plus-source"
    assert serialized["manifest_id"] == manifest.manifest_id


def test_manifest_id_is_independent_of_id_order_and_duplicates() -> None:
    first = DeletionManifest(entry_ids=("entry-2", "entry-1", "entry-1"))
    second = DeletionManifest(entry_ids=("entry-1", "entry-2"))

    assert first.entry_ids == ("entry-1", "entry-2")
    assert first.manifest_id == second.manifest_id


def test_nested_empty_manifest_does_not_fall_back_to_legacy_ids() -> None:
    manifest = DeletionManifest.from_prompt_row(
        {
            "oracle_entry_ids": ["legacy-entry"],
            "deletion_manifest": {"entry_ids": [], "source_ids": []},
        }
    )

    assert manifest.is_empty


def test_audit_example_uses_backend_and_serializes_common_schema() -> None:
    backend = FakeBackend()
    example = AuditExample.from_prompt_row(
        {
            "fact_id": 1,
            "prompt_text": "What is the capital of France?",
            "gold_object": "Paris",
        }
    )

    result = audit_example(
        backend,
        example,
        DatabaseState.DEL_ON,
        max_new_tokens=20,
    )

    assert backend.calls == [(example, DatabaseState.DEL_ON, 20)]
    assert result["state"] == "DEL-ON"
    assert result["prompt_id"] is None
    assert result["deletion_manifest"]["manifest_id"]
    assert result["model_output"] == "Paris"
    assert result["retrieval_trace"]["retrieval_triggered"] is True
    assert result["generation_metadata"]["answer_source"] == "retrieved_value"


def test_audit_example_supplies_default_trace() -> None:
    class NoTraceBackend:
        def generate(self, example, state, *, max_new_tokens=12):
            return AuditObservation(model_output="unknown")

    example = AuditExample.from_prompt_row(
        {"prompt_text": "Capital?", "gold_object": "Paris"}
    )
    result = audit_example(NoTraceBackend(), example, DatabaseState.DEL_OFF)

    assert result["retrieval_trace"] == default_retrieval_trace(
        DatabaseState.DEL_OFF
    )
    assert result["retrieval_trace"]["retrieval_enabled"] is False


def test_audit_example_marks_error_trace_incomplete_by_default() -> None:
    class ErrorTraceBackend:
        def generate(self, example, state, *, max_new_tokens=12):
            return AuditObservation(
                model_output="unknown",
                retrieval_trace={"state": state.value, "error": "search failed"},
            )

    example = AuditExample(prompt="Capital?", ground_truth="Paris")
    result = audit_example(ErrorTraceBackend(), example, DatabaseState.DEL_ON)

    assert result["retrieval_trace"]["trace_available"] is True
    assert result["retrieval_trace"]["trace_complete"] is False


def test_cross_state_validation_rejects_deleted_selected_entry() -> None:
    manifest = DeletionManifest(entry_ids=("deleted-entry",))
    example = AuditExample(
        prompt="Capital?",
        ground_truth="Paris",
        fact_id="france-capital",
        deletion_manifest=manifest,
    )
    results = []
    for state in (DatabaseState.DEL_ON, DatabaseState.DEL_OFF):
        trace = default_retrieval_trace(state)
        if state is DatabaseState.DEL_ON:
            trace.update(
                {
                    "trace_available": True,
                    "trace_complete": True,
                    "selected_candidate": {"entry_id": "deleted-entry"},
                }
            )
        results.append(
            {
                "fact_id": example.fact_id,
                "prompt_id": example.prompt_id,
                "prompt": example.prompt,
                "ground_truth": example.ground_truth,
                "state": state.value,
                "deletion_manifest": manifest.as_dict(),
                "retrieval_trace": trace,
            }
        )

    with pytest.raises(ValueError, match="selected an entry"):
        validate_intervention_results(
            results,
            expected_states=[DatabaseState.DEL_ON, DatabaseState.DEL_OFF],
        )


def test_cross_state_validation_checks_every_retrieval_event() -> None:
    manifest = DeletionManifest(source_ids=("deleted-source",))
    results = []
    for state in (DatabaseState.DEL_ON, DatabaseState.DEL_OFF):
        trace = default_retrieval_trace(state)
        if state is DatabaseState.DEL_ON:
            trace.update(
                {
                    "trace_available": True,
                    "trace_complete": True,
                    "retrieval_events": [
                        {
                            "selected_candidate": {
                                "entry_id": "another-entry",
                                "source_id": "deleted-source",
                            }
                        }
                    ],
                }
            )
        results.append(
            {
                "fact_id": "france-capital",
                "prompt_id": "p1",
                "prompt": "Capital?",
                "ground_truth": "Paris",
                "state": state.value,
                "deletion_manifest": manifest.as_dict(),
                "retrieval_trace": trace,
            }
        )

    with pytest.raises(ValueError, match="selected a source"):
        validate_intervention_results(
            results,
            expected_states=[DatabaseState.DEL_ON, DatabaseState.DEL_OFF],
        )
