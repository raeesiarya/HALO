import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

from lmlm_audit.core.examples import AuditExample
from lmlm_audit.core.states import DatabaseState, retrieval_enabled


@dataclass(frozen=True)
class AuditObservation:
    model_output: str
    retrieval_trace: Mapping[str, Any] | None = None
    generation_metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class AuditBackend(Protocol):
    def generate(
        self,
        example: AuditExample,
        state: DatabaseState,
        *,
        max_new_tokens: int = 12,
    ) -> AuditObservation:
        ...


def default_retrieval_trace(state: DatabaseState) -> dict[str, Any]:
    return {
        "state": state.value,
        "trace_available": False,
        "trace_complete": False,
        "retrieval_enabled": retrieval_enabled(state),
        "retrieval_triggered": False,
        "threshold_fallback": False,
        "lookup_query": None,
        "threshold": None,
        "all_candidates": [],
        "deleted_candidates": [],
        "retained_candidates": [],
        "selected_candidate": None,
        "selected_value": None,
        "retrieval_events": [],
        "error": None,
    }


def audit_example(
    backend: AuditBackend,
    example: AuditExample,
    state: DatabaseState,
    *,
    max_new_tokens: int = 12,
) -> dict[str, Any]:
    observation = backend.generate(
        example,
        state,
        max_new_tokens=max_new_tokens,
    )
    retrieval_trace = default_retrieval_trace(state)
    if observation.retrieval_trace is not None:
        supplied_trace = dict(observation.retrieval_trace)
        supplied_state = supplied_trace.get("state")
        if supplied_state is not None and supplied_state != state.value:
            raise ValueError(
                f"Backend returned trace state {supplied_state!r} for {state.value!r}."
            )
        retrieval_trace.update(supplied_trace)
        retrieval_trace["state"] = state.value
        if "trace_available" not in supplied_trace:
            retrieval_trace["trace_available"] = True
        if "trace_complete" not in supplied_trace:
            retrieval_trace["trace_complete"] = supplied_trace.get("error") is None

    return {
        "fact_id": example.fact_id,
        "prompt_id": example.prompt_id,
        "subject": example.subject,
        "subject_aliases": list(example.subject_aliases),
        "relation": example.relation,
        "relation_aliases": list(example.relation_aliases),
        "state": state.value,
        "prompt": example.prompt,
        "ground_truth": example.ground_truth,
        "object_aliases": list(example.object_aliases),
        "deletion_manifest": example.deletion_manifest.as_dict(),
        "model_output": observation.model_output,
        "retrieval_trace": retrieval_trace,
        "generation_metadata": dict(observation.generation_metadata),
    }


def validate_intervention_results(
    results: list[dict[str, Any]],
    *,
    expected_states: list[DatabaseState],
) -> None:
    if not results:
        return
    states = [result.get("state") for result in results]
    expected = [state.value for state in expected_states]
    if states != expected:
        raise ValueError(f"Intervention states {states!r} do not match {expected!r}.")

    identity_fields = ("fact_id", "prompt_id", "prompt", "ground_truth")
    for field_name in identity_fields:
        values = {json.dumps(result.get(field_name), sort_keys=True) for result in results}
        if len(values) != 1:
            raise ValueError(f"Cross-state rows disagree on {field_name!r}.")

    manifests = {
        json.dumps(result.get("deletion_manifest"), sort_keys=True)
        for result in results
    }
    if len(manifests) != 1:
        raise ValueError("Cross-state rows do not share one deletion manifest.")

    for result in results:
        trace = result.get("retrieval_trace") or {}
        if trace.get("state") != result.get("state"):
            raise ValueError("Result state and retrieval-trace state disagree.")

        if result.get("state") == DatabaseState.DEL_OFF.value:
            if trace.get("retrieval_enabled") is not False:
                raise ValueError("DEL-OFF must disable retrieval.")
            if trace.get("retrieval_triggered") is True:
                raise ValueError("DEL-OFF unexpectedly triggered retrieval.")
            if trace.get("retrieval_events"):
                raise ValueError("DEL-OFF unexpectedly recorded retrieval events.")

        if result.get("state") == DatabaseState.DEL_ON.value:
            manifest = result.get("deletion_manifest") or {}
            excluded_entry_ids = set(manifest.get("entry_ids") or [])
            excluded_source_ids = set(manifest.get("source_ids") or [])
            selected_candidates = [trace.get("selected_candidate")]
            selected_candidates.extend(
                event.get("selected_candidate")
                for event in trace.get("retrieval_events") or []
            )
            for selected in selected_candidates:
                if not isinstance(selected, Mapping):
                    continue
                if selected.get("entry_id") in excluded_entry_ids:
                    raise ValueError(
                        "DEL-ON selected an entry from its deletion manifest."
                    )
                if selected.get("source_id") in excluded_source_ids:
                    raise ValueError(
                        "DEL-ON selected a source from its deletion manifest."
                    )
