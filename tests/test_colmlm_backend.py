from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lmlm_audit.core.backend import audit_example
from lmlm_audit.colmlm.answers import _default_support_judge, extract_colmlm_answer
from lmlm_audit.colmlm.backend import CoLMLMAuditBackend
from lmlm_audit.core.examples import AuditExample
from lmlm_audit.cli.runner import run_backend_audit
from lmlm_audit.core.states import DatabaseState


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
            text=(
                f"{prompt}<FACT>{selected.text_value}</FACT> "
                f"{selected.text_value}."
            ),
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


def test_del_off_uses_public_no_retrieval_path_and_never_searches() -> None:
    backend, generator, index = _backend()
    result = audit_example(backend, _example(), DatabaseState.DEL_OFF)

    assert result["model_output"] == "Paris"
    assert result["retrieval_trace"]["retrieval_enabled"] is False
    assert result["retrieval_trace"]["retrieval_events"] == []
    assert result["retrieval_trace"]["num_retrievals"] == 0
    assert generator.no_retrieval_calls == [_example().prompt]
    assert index.calls == []


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


def test_support_judge_matches_whole_normalized_phrases() -> None:
    candidate = FakeSearchResult(
        id="russia",
        score=0.9,
        text_value="Russia is a country.",
    )
    example = AuditExample(prompt="Where?", ground_truth="US")

    assert _default_support_judge(candidate, example)["supports_target"] is False


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

    manifest_ids = {
        result["deletion_manifest"]["manifest_id"] for result in results
    }
    assert len(manifest_ids) == 1
    assert results[0]["deletion_manifest"]["entry_ids"] == ["target-entry"]
    assert results[1]["retrieval_trace"]["selected_candidate"]["entry_id"] == (
        "neighbor-entry"
    )


def test_public_loader_arguments_map_to_release_factory() -> None:
    generator = FakeGenerator(FakeIndex([]))
    loader = SimpleNamespace(load_retriever_generator=lambda **_kwargs: generator)

    with patch("lmlm_audit.colmlm.backend.importlib.import_module", return_value=loader) as load_module:
        backend = CoLMLMAuditBackend.from_public_release(
            model_path="model",
            index_path="index",
            db_path="entries.db",
            use_sqlite_id_mapping=True,
            similarity_threshold=0.7,
        )

    assert backend.generator is generator
    load_module.assert_called_once_with("lmlm.eval.hf_generate")
