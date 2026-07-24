import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from halo.core.backend import audit_example
from halo.core.embeddings import QueryEmbeddingSink
from halo.interventions.judge import default_support_judge
from models.co_lmlm.backend import extract_colmlm_answer
from models.co_lmlm.backend import CoLMLMAuditBackend
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


class FakeIndex:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def search(self, _query, top_k=1, similarity_threshold=None):
        self.calls.append((top_k, similarity_threshold))
        return [
            result
            for result in self.results
            if similarity_threshold is None or result.score >= similarity_threshold
        ][:top_k]


class FakeGenerator:
    def __init__(self, index):
        self.index = index
        self.generation_config = SimpleNamespace(max_new_tokens=64)
        self.retrieval_config = SimpleNamespace(similarity_threshold=0.7)
        self.no_retrieval_calls = []

    def generate(self, prompt):
        results = self.index.search(
            [1.0],
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
            t_generate_s=0.1,
            t_encode_s=0.2,
            t_search_s=0.3,
            gen_decoded_tokens=4,
        )

    def generate_no_retrieval(self, prompt):
        self.no_retrieval_calls.append(prompt)
        return SimpleNamespace(
            text=f"{prompt} Paris.",
            num_retrievals=0,
            failed_retrievals=0,
        )


def _example() -> AuditExample:
    return AuditExample.from_prompt_row(
        {
            "prompt_id": "capital-direct",
            "fact_id": "france-capital",
            "prompt_text": "What is the capital of France?",
            "gold_object": "Paris",
            "oracle_entry_ids": ["target-entry"],
            "source_ids": [],
        }
    )


def _backend():
    index = FakeIndex(
        [
            FakeSearchResult(
                id="target-entry",
                score=0.95,
                text_value="Paris",
                metadata={"source_id": "wiki:France"},
            ),
            FakeSearchResult(
                id="neighbor-entry",
                score=0.90,
                text_value="Lyon",
                metadata={"source_id": "wiki:Lyon"},
            ),
        ]
    )
    generator = FakeGenerator(index)
    return CoLMLMAuditBackend(generator), generator, index


def test_full_uses_public_generation_and_records_stable_entry_id() -> None:
    backend, generator, index = _backend()
    result = audit_example(backend, _example(), DatabaseState.FULL, max_new_tokens=20)

    assert result["model_output"] == "Paris"
    assert result["retrieval_trace"]["selected_candidate"]["entry_id"] == "target-entry"
    assert result["retrieval_trace"]["num_retrievals"] == 1
    assert result["retrieval_trace"]["trace_complete"] is True
    assert result["deletion_manifest"]["entry_ids"] == ["target-entry"]
    assert generator.index is index
    assert generator.generation_config.max_new_tokens == 64


def test_del_on_filters_oracle_id_without_mutating_base_index() -> None:
    backend, generator, index = _backend()
    result = audit_example(backend, _example(), DatabaseState.DEL_ON)

    trace = result["retrieval_trace"]
    assert result["model_output"] == "Lyon"
    assert trace["selected_candidate"]["entry_id"] == "neighbor-entry"
    assert [item["entry_id"] for item in trace["deleted_candidates"]] == [
        "target-entry"
    ]
    assert trace["deletion_manifest_id"] == result["deletion_manifest"]["manifest_id"]
    assert index.calls == [(2, 0.7)]
    assert generator.index is index


def test_del_off_default_nulls_retrieval_but_keeps_fact_emission_path() -> None:
    backend, generator, index = _backend()
    result = audit_example(backend, _example(), DatabaseState.DEL_OFF)

    trace = result["retrieval_trace"]
    assert result["model_output"] == "unknown"
    assert trace["retrieval_enabled"] is False
    assert trace["del_off_mode"] == "null-retrieval"
    assert trace["retrieval_triggered"] is True
    assert trace["retained_candidates"] == []
    assert trace["selected_candidate"] is None
    assert [item["entry_id"] for item in trace["deleted_candidates"]] == [
        "target-entry"
    ]
    assert trace["threshold_fallback"] is True
    # No over-fetch when everything is excluded anyway.
    assert index.calls == [(1, 0.7)]
    assert generator.no_retrieval_calls == []


def test_del_off_forbid_token_mode_preserves_legacy_path() -> None:
    index = FakeIndex([])
    generator = FakeGenerator(index)
    backend = CoLMLMAuditBackend(generator, del_off_mode="forbid-token")
    result = audit_example(backend, _example(), DatabaseState.DEL_OFF)

    trace = result["retrieval_trace"]
    assert result["model_output"] == "Paris"
    assert trace["retrieval_enabled"] is False
    assert trace["del_off_mode"] == "forbid-token"
    assert trace["retrieval_events"] == []
    assert trace["num_retrievals"] == 0
    assert generator.no_retrieval_calls == [_example().prompt]
    assert index.calls == []


def test_unknown_del_off_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="del_off_mode"):
        CoLMLMAuditBackend(FakeGenerator(FakeIndex([])), del_off_mode="bogus")


def test_source_manifest_filters_every_matching_candidate() -> None:
    index = FakeIndex(
        [
            FakeSearchResult(
                id="source-entry-1",
                score=0.95,
                text_value="Paris",
                metadata={"source_id": "wiki:France"},
            ),
            FakeSearchResult(
                id="source-entry-2",
                score=0.94,
                text_value="Paris, France",
                metadata={"source_id": "wiki:France"},
            ),
            FakeSearchResult(
                id="neighbor-entry",
                score=0.90,
                text_value="Lyon",
                metadata={"source_id": "wiki:Lyon"},
            ),
        ]
    )
    generator = FakeGenerator(index)
    backend = CoLMLMAuditBackend(generator, max_filter_overfetch=4)
    example = AuditExample.from_prompt_row(
        {
            "fact_id": "france-capital",
            "prompt_text": "What is the capital of France?",
            "gold_object": "Paris",
            "deletion_manifest": {
                "entry_ids": [],
                "source_ids": ["wiki:France"],
                "strategy": "source",
            },
        }
    )

    result = audit_example(backend, example, DatabaseState.DEL_ON)
    trace = result["retrieval_trace"]
    assert result["model_output"] == "Lyon"
    assert [item["entry_id"] for item in trace["deleted_candidates"]] == [
        "source-entry-1",
        "source-entry-2",
    ]
    assert index.calls == [(5, 0.7)]


def test_exhausted_overfetch_widens_search_until_a_retained_candidate() -> None:
    # 6 excluded entries from one source, then a retained neighbor. With
    # max_filter_overfetch=4 the first fetch (top_k + 4 = 5) is entirely
    # excluded, so the filter must widen and retry instead of raising.
    excluded = [
        FakeSearchResult(
            id=f"source-entry-{i}",
            score=0.99 - i * 0.01,
            text_value="Paris",
            metadata={"source_id": "wiki:France"},
        )
        for i in range(6)
    ]
    retained = FakeSearchResult(
        id="neighbor-entry",
        score=0.90,
        text_value="Lyon",
        metadata={"source_id": "wiki:Lyon"},
    )
    index = FakeIndex(excluded + [retained])
    generator = FakeGenerator(index)
    backend = CoLMLMAuditBackend(generator, max_filter_overfetch=4)
    example = AuditExample.from_prompt_row(
        {
            "fact_id": "france-capital",
            "prompt_text": "What is the capital of France?",
            "gold_object": "Paris",
            "deletion_manifest": {
                "entry_ids": [],
                "source_ids": ["wiki:France"],
                "strategy": "source",
            },
        }
    )

    result = audit_example(backend, example, DatabaseState.DEL_ON)
    trace = result["retrieval_trace"]
    assert result["model_output"] == "Lyon"
    assert index.calls == [(5, 0.7), (10, 0.7)]
    event = trace["retrieval_events"][0]
    assert event["searched_top_k"] == 10
    assert event["widened_search_attempts"] == 1
    assert trace["selected_candidate"]["entry_id"] == "neighbor-entry"


def test_exhausted_widening_ceiling_still_raises() -> None:
    excluded = [
        FakeSearchResult(
            id=f"source-entry-{i}",
            score=0.99 - i * 0.001,
            text_value="Paris",
            metadata={"source_id": "wiki:France"},
        )
        for i in range(64)
    ]
    index = FakeIndex(excluded)
    generator = FakeGenerator(index)
    backend = CoLMLMAuditBackend(
        generator, max_filter_overfetch=4, max_filter_search_k=16
    )
    example = AuditExample.from_prompt_row(
        {
            "fact_id": "france-capital",
            "prompt_text": "What is the capital of France?",
            "gold_object": "Paris",
            "deletion_manifest": {
                "entry_ids": [],
                "source_ids": ["wiki:France"],
                "strategy": "source",
            },
        }
    )

    from halo.interventions.errors import ExclusionSearchExhaustedError

    with pytest.raises(ExclusionSearchExhaustedError, match="widening"):
        audit_example(backend, example, DatabaseState.DEL_ON)
    # 5 → 10 → 17 (top_k + ceiling), then stop.
    assert index.calls == [(5, 0.7), (10, 0.7), (17, 0.7)]


def test_deleted_states_require_a_manifest() -> None:
    backend, _, _ = _backend()
    example = AuditExample.from_prompt_row(
        {"prompt_text": "Capital?", "gold_object": "Paris"}
    )

    with pytest.raises(ValueError, match="requires deletion"):
        backend.generate(example, DatabaseState.DEL_ON)


def test_answer_extractor_removes_retrieval_scaffolding() -> None:
    raw = "Question?<FACT> Paris</FACT> Paris. More explanation."
    assert extract_colmlm_answer(raw, "Question?") == "Paris"


def test_answer_extractor_handles_output_without_fact_blocks() -> None:
    raw = "Question? The answer is Paris."
    assert extract_colmlm_answer(raw, "Question?") == "Paris"


def test_support_judge_matches_whole_normalized_phrases() -> None:
    candidate = FakeSearchResult(
        id="russia",
        score=0.9,
        text_value="Russia is a country.",
    )
    example = AuditExample(prompt="Where?", ground_truth="US")

    assert default_support_judge(candidate, example)["supports_target"] is False


def test_runner_can_bootstrap_reviewable_oracle_manifest_from_full(tmp_path) -> None:
    backend, _, _ = _backend()
    prompt_path = tmp_path / "prompts.jsonl"
    prompt_path.write_text(
        '{"prompt_id":"p1","fact_id":"f1","prompt_text":"What is the capital of France?",'
        '"gold_object":"Paris"}\n',
        encoding="utf-8",
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
    )

    manifest_ids = {result["deletion_manifest"]["manifest_id"] for result in results}
    assert len(manifest_ids) == 1
    assert results[0]["deletion_manifest"]["entry_ids"] == ["target-entry"]
    assert results[1]["retrieval_trace"]["selected_candidate"]["entry_id"] == (
        "neighbor-entry"
    )


class FakeNestedIndex(FakeIndex):
    """Mimics RetrieverIndex.search, which wraps results per query."""

    def search(self, query, top_k=1, similarity_threshold=None):
        return [super().search(query, top_k, similarity_threshold)]


def test_nested_result_lists_are_flattened_for_single_queries() -> None:
    index = FakeNestedIndex(
        [
            FakeSearchResult(
                id="target-entry",
                score=0.95,
                text_value="Paris",
                metadata={"source_id": "wiki:France"},
            ),
        ]
    )
    backend = CoLMLMAuditBackend(FakeGenerator(index))
    result = audit_example(backend, _example(), DatabaseState.FULL)

    assert result["model_output"] == "Paris"
    assert result["retrieval_trace"]["selected_candidate"]["entry_id"] == (
        "target-entry"
    )


def test_full_captures_query_embeddings_outside_the_trace() -> None:
    backend, _, _ = _backend()
    result = audit_example(backend, _example(), DatabaseState.FULL)

    embeddings = result["_query_embeddings"]
    assert [item["event_index"] for item in embeddings] == [0]
    np.testing.assert_array_equal(
        embeddings[0]["vector"], np.asarray([1.0], dtype=np.float32)
    )

    event = result["retrieval_trace"]["retrieval_events"][0]
    assert event["query_embedding_captured"] is True
    assert event["query_dim"] == 1
    assert event["query_l2_norm"] == pytest.approx(1.0)


def test_runner_routes_embeddings_to_sidecar_and_keeps_results_serializable(
    tmp_path,
) -> None:
    backend, _, _ = _backend()
    prompt_path = tmp_path / "prompts.jsonl"
    prompt_path.write_text(
        '{"prompt_id":"p1","fact_id":"f1","prompt_text":"What is the capital of France?",'
        '"gold_object":"Paris"}\n',
        encoding="utf-8",
    )
    sink = QueryEmbeddingSink()

    results = run_backend_audit(
        prompt_path=prompt_path,
        backend=backend,
        states=[
            DatabaseState.FULL,
            DatabaseState.DEL_ON,
            DatabaseState.DEL_OFF,
        ],
        bootstrap_oracle_from_full=True,
        embedding_sink=sink,
    )

    assert all("_query_embeddings" not in result for result in results)
    json.dumps(results)

    sidecar_path = tmp_path / "query_embeddings.npz"
    sink.save(sidecar_path)
    with np.load(sidecar_path) as stored:
        assert sorted(stored.keys()) == [
            "p1/DEL-OFF/event0",
            "p1/DEL-ON/event0",
            "p1/FULL/event0",
        ]
        np.testing.assert_array_equal(
            stored["p1/FULL/event0"], np.asarray([1.0], dtype=np.float32)
        )


def test_public_loader_arguments_map_to_release_factory() -> None:
    generator = FakeGenerator(FakeIndex([]))
    captured: dict = {}

    def fake_loader(**kwargs):
        captured.update(kwargs)
        return generator

    loader = SimpleNamespace(load_retriever_generator=fake_loader)

    with patch(
        "models.co_lmlm.backend.importlib.import_module", return_value=loader
    ) as load_module:
        backend = CoLMLMAuditBackend.from_public_release(
            model_path="model",
            index_path="index",
            db_path="entries.db",
            similarity_threshold=0.7,
        )

    assert backend.generator is generator
    load_module.assert_called_once_with("lmlm.eval.hf_generate")
    # Device/dtype/attn/sqlite are auto-resolved, not passed by the caller.
    assert captured["retrieval_top_k"] == 1
    assert captured["similarity_threshold"] == 0.7
    assert captured["use_sqlite_id_mapping"] is False  # no mapping .db at "index"
    assert captured["device"] in ("cuda:0", "mps", "cpu")
    assert captured["torch_dtype"] in ("bfloat16", "float32")
    assert "attn_implementation" in captured


# --- capability hooks (reuse fast paths) ------------------------------------


from halo.core.examples import DeletionManifest  # noqa: E402
from models.co_lmlm.backend import (  # noqa: E402
    full_trace_unaffected,
    manifest_reuse_fingerprint,
)


def _closure_manifest(entry_ids, source_ids=(), *, radius=0.9, target="Paris"):
    return DeletionManifest(
        entry_ids=tuple(entry_ids),
        source_ids=tuple(source_ids),
        strategy="closure",
        metadata={
            "predicates_active": ["geometric", "value", "provenance"],
            "radius": radius,
            "value_target": {"ground_truth": target, "object_aliases": []},
        },
    )


def _full_row(candidates, *, state="FULL", complete=True, injected=0):
    return {
        "model_output": "Paris",
        "retrieval_trace": {
            "state": state,
            "trace_available": True,
            "trace_complete": complete,
            "retrieval_events": [
                {
                    "all_candidates": list(candidates),
                    "injected_candidates_count": injected,
                }
            ],
        },
    }


def test_manifest_fingerprint_ignores_bookkeeping_metadata() -> None:
    tight = _closure_manifest(("e1",), ("s1",), radius=0.9)
    loose = _closure_manifest(("e1",), ("s1",), radius=0.7)
    assert manifest_reuse_fingerprint(tight) == manifest_reuse_fingerprint(loose)
    assert manifest_reuse_fingerprint(tight) != manifest_reuse_fingerprint(
        _closure_manifest(("e1", "e2"), ("s1",))
    )
    assert manifest_reuse_fingerprint(tight) != manifest_reuse_fingerprint(
        _closure_manifest(("e1",), ("s1",), target="Warsaw")
    )


def test_manifest_fingerprint_declines_without_value_target() -> None:
    manifest = DeletionManifest(
        entry_ids=("e1",),
        strategy="closure",
        metadata={"predicates_active": ["value"]},
    )
    assert manifest_reuse_fingerprint(manifest) is None


def test_backend_hooks_make_no_claim_while_injections_are_active() -> None:
    # full_row_unaffected is only enabled on an exact index (see below); use
    # that mode here so the test isolates the injection gate.
    backend = CoLMLMAuditBackend(
        FakeGenerator(FakeIndex([])), assume_exact_index=True
    )
    manifest = _closure_manifest(("e1",))
    row = _full_row([{"entry_id": "other", "source_id": "s9", "value": "Rome"}])
    assert backend.manifest_fingerprint(manifest) is not None
    assert backend.full_row_unaffected(row, manifest) is True
    backend.injections = (object(),)
    assert backend.manifest_fingerprint(manifest) is None
    assert backend.full_row_unaffected(row, manifest) is False


def test_full_row_reuse_makes_no_claim_on_an_approximate_index() -> None:
    # Default (approximate index): DEL-ON over-fetches relative to the FULL
    # pass, so a deeper search can change the top-1 even when the manifest
    # deletes nothing the FULL pass saw. full_row_unaffected must decline; the
    # fingerprint hook is unaffected by search depth and stays sound.
    backend = CoLMLMAuditBackend(FakeGenerator(FakeIndex([])))
    manifest = _closure_manifest(("e1",))
    row = _full_row([{"entry_id": "other", "source_id": "s9", "value": "Rome"}])
    assert backend.assume_exact_index is False
    assert backend.full_row_unaffected(row, manifest) is False
    assert backend.manifest_fingerprint(manifest) is not None


def test_full_trace_unaffected_checks_all_three_filters() -> None:
    manifest = _closure_manifest(("e1",), ("s1",), target="Paris")
    clean = [{"entry_id": "e9", "source_id": "s9", "value": "Warsaw"}]
    assert full_trace_unaffected(
        _full_row(clean), manifest, default_support_judge
    )
    assert not full_trace_unaffected(
        _full_row([{"entry_id": "e1", "source_id": "s9", "value": "Warsaw"}]),
        manifest,
        default_support_judge,
    )
    assert not full_trace_unaffected(
        _full_row([{"entry_id": "e9", "source_id": "s1", "value": "Warsaw"}]),
        manifest,
        default_support_judge,
    )
    assert not full_trace_unaffected(
        _full_row([{"entry_id": "e9", "source_id": "s9", "value": "Paris"}]),
        manifest,
        default_support_judge,
    )


def test_full_trace_unaffected_is_conservative_about_the_trace() -> None:
    manifest = _closure_manifest(("e1",))
    clean = [{"entry_id": "e9", "source_id": "s9", "value": "Warsaw"}]
    assert full_trace_unaffected(_full_row([]), manifest, default_support_judge)
    assert not full_trace_unaffected(
        _full_row(clean, state="DEL-ON"), manifest, default_support_judge
    )
    assert not full_trace_unaffected(
        _full_row(clean, complete=False), manifest, default_support_judge
    )
    assert not full_trace_unaffected(
        _full_row(clean, injected=1), manifest, default_support_judge
    )
    assert not full_trace_unaffected(None, manifest, default_support_judge)
