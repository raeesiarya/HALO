import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from halo.interventions.adversary import (
    AdversarialConfig,
    build_injections,
    survivor_key,
    survivor_value,
    template_evades_judge,
)
from models.co_lmlm.backend import CoLMLMAuditBackend
from halo.interventions.closure import ClosureConfig, build_closure_family
from halo.core.backend import audit_example
from halo.core.examples import AuditExample
from halo.core.metrics import auroc
from halo.cli.reporting import write_adversarial_outputs
from halo.cli.runner import run_adversarial_eval
from halo.core.states import DatabaseState


# --- survivor construction -------------------------------------------------


def test_survivor_key_hits_exact_target_cosine() -> None:
    rng = np.random.default_rng(0)
    query = rng.normal(size=8)
    key = survivor_key(query, 0.83, rng)
    unit_query = query / np.linalg.norm(query)
    assert np.linalg.norm(key) == pytest.approx(1.0, abs=1e-5)
    assert float(np.dot(key, unit_query)) == pytest.approx(0.83, abs=1e-5)


def test_survivor_key_rejects_zero_query() -> None:
    with pytest.raises(ValueError, match="zero query"):
        survivor_key(np.zeros(4), 0.8, np.random.default_rng(0))


@pytest.mark.parametrize("answer", ["Paris", "New York"])
def test_evading_templates_beat_the_judge_and_verbatim_does_not(
    answer,
) -> None:
    example = AuditExample(prompt="Q?", ground_truth=answer)
    assert template_evades_judge(answer, "hyphenated", example)
    assert template_evades_judge(answer, "letter-spaced", example)
    assert template_evades_judge(answer, "prefix-cue", example)
    assert not template_evades_judge(answer, "verbatim", example)


def test_adversarial_config_validation() -> None:
    with pytest.raises(ValueError, match="epsilon"):
        AdversarialConfig(epsilons=(0.0,))
    with pytest.raises(ValueError, match="templates"):
        AdversarialConfig(templates=("psychic",))
    with pytest.raises(ValueError, match="topology"):
        AdversarialConfig(topology="mesh")


def test_build_injections_topologies_are_deterministic() -> None:
    example = AuditExample(prompt="Q?", ground_truth="Paris")
    query = np.asarray([1.0, 0.0, 0.0, 0.0])

    def build(topology):
        return build_injections(
            example=example,
            query_vector=query,
            config=AdversarialConfig(
                rho=0.85, epsilons=(0.05,), topology=topology, count=2, seed=1
            ),
            epsilon=0.05,
            template="hyphenated",
            fact_seed=42,
        )

    single = build("single")
    assert len(single) == 1
    assert single[0].target_cosine == pytest.approx(0.8)

    aliased = build("aliased")
    assert len(aliased) == 2
    assert all("Pa-ris" in entry.value for entry in aliased)

    collided = build("collided")
    assert len(collided) == 3
    decoys = [entry for entry in collided if entry.template == "decoy"]
    assert len(decoys) == 2
    assert all(entry.target_cosine == pytest.approx(0.8) for entry in decoys)

    saturated = build("saturated")
    decoys = [entry for entry in saturated if entry.template == "decoy"]
    assert all(entry.target_cosine == pytest.approx(0.9) for entry in decoys)

    again = build("single")
    np.testing.assert_array_equal(single[0].vector, again[0].vector)


# --- auroc -----------------------------------------------------------------


def test_auroc() -> None:
    assert auroc([0.9, 0.8, 0.2, 0.1], [True, True, False, False]) == 1.0
    assert auroc([0.1, 0.2, 0.8, 0.9], [True, True, False, False]) == 0.0
    assert auroc([0.5, 0.5, 0.5, 0.5], [True, True, False, False]) == 0.5
    assert auroc([0.9, 0.1], [True, True]) is None
    with pytest.raises(ValueError):
        auroc([0.5], [True, False])


# --- fakes -----------------------------------------------------------------


@dataclass
class FakeSearchResult:
    id: str
    score: float
    text_value: str
    text_key: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class FakeVectorEntry:
    entry_id: str
    vector: np.ndarray
    text: str
    source_id: str


class FakeVectorIndex:
    def __init__(self, entries):
        self.entries = list(entries)

    def search(self, query, top_k=1, similarity_threshold=None):
        normalized = np.asarray(query, dtype=np.float32).reshape(-1)
        normalized = normalized / np.linalg.norm(normalized)
        results = []
        for entry in self.entries:
            vector = entry.vector / np.linalg.norm(entry.vector)
            score = float(np.dot(normalized, vector))
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


QUERY = np.asarray([1.0, 0.0], dtype=np.float32)
PROMPT = "What is the capital of France?"


class FakeGenerator:
    def __init__(self, index):
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
            text=(f"{prompt}<FACT>{selected.text_value}</FACT> {selected.text_value}"),
            num_retrievals=1,
            failed_retrievals=0,
        )


def _example() -> AuditExample:
    return AuditExample.from_prompt_row(
        {
            "prompt_id": "pA",
            "prompt_text": PROMPT,
            "gold_object": "Paris",
        }
    )


def _cos_vec(cosine: float) -> np.ndarray:
    return np.asarray([cosine, float(np.sqrt(1.0 - cosine * cosine))], dtype=np.float32)


# --- injection through the filter/backend ----------------------------------


def test_injected_survivor_is_retrieved_after_deletion() -> None:
    index = FakeVectorIndex(
        [FakeVectorEntry("entry-a", QUERY.copy(), "Paris", "wiki:France")]
    )
    generator = FakeGenerator(index)
    backend = CoLMLMAuditBackend(generator)
    injections = build_injections(
        example=_example(),
        query_vector=QUERY,
        config=AdversarialConfig(rho=0.85, epsilons=(0.05,)),
        epsilon=0.05,
        template="verbatim",
    )
    backend.injections = injections
    example = AuditExample.from_prompt_row(
        {
            "prompt_id": "pA",
            "prompt_text": PROMPT,
            "gold_object": "Paris",
            "deletion_manifest": {"entry_ids": ["entry-a"]},
        }
    )

    result = audit_example(backend, example, DatabaseState.DEL_ON)
    trace = result["retrieval_trace"]
    assert result["model_output"] == "Paris"
    selected = trace["selected_candidate"]
    assert selected["entry_id"] == "adv-verbatim-0"
    assert selected["metadata"]["synthetic"] is True
    assert selected["score"] == pytest.approx(0.8, abs=1e-5)
    assert trace["retrieval_events"][0]["injected_candidates_count"] == 1


def test_semantic_backstop_catches_verbatim_but_not_evading_survivor() -> None:
    index = FakeVectorIndex(
        [FakeVectorEntry("entry-a", QUERY.copy(), "Paris", "wiki:France")]
    )
    backend = CoLMLMAuditBackend(FakeGenerator(index))
    base_row = {
        "prompt_id": "pA",
        "prompt_text": PROMPT,
        "gold_object": "Paris",
        "deletion_manifest": {
            "entry_ids": ["entry-a"],
            "strategy": "closure",
            "metadata": {
                "predicates_active": ["semantic"],
                "semantic_target": {
                    "ground_truth": "Paris",
                    "object_aliases": [],
                },
            },
        },
    }
    config = AdversarialConfig(rho=0.85, epsilons=(0.05,))

    for template, survives in (("verbatim", False), ("hyphenated", True)):
        backend.injections = build_injections(
            example=_example(),
            query_vector=QUERY,
            config=config,
            epsilon=0.05,
            template=template,
        )
        result = audit_example(
            backend,
            AuditExample.from_prompt_row(base_row),
            DatabaseState.DEL_ON,
        )
        selected = result["retrieval_trace"]["selected_candidate"]
        if survives:
            assert selected["entry_id"] == f"adv-{template}-0"
        else:
            assert selected is None


# --- margin geometry from the closure --------------------------------------


def test_closure_records_survivor_margin() -> None:
    index = FakeVectorIndex(
        [
            FakeVectorEntry("entry-a", _cos_vec(0.95), "Paris", "wiki:France"),
            FakeVectorEntry("distractor", _cos_vec(0.75), "Berlin", "wiki:Berlin"),
        ]
    )
    closure = build_closure_family(
        index=index,
        example=_example(),
        query_vector=QUERY,
        config=ClosureConfig(radius=0.85),
        radii=(0.85,),
    )[0.85]

    assert closure.s_del == pytest.approx(0.95)
    assert closure.s_surv == pytest.approx(0.75)
    assert closure.margin == pytest.approx(0.2)
    assert [entry.entry_id for entry in closure.top_survivors] == ["distractor"]


# --- end-to-end ------------------------------------------------------------


def _write_prompt(tmp_path: Path) -> Path:
    prompt_path = tmp_path / "prompts.jsonl"
    prompt_path.write_text(
        json.dumps({"prompt_id": "pA", "prompt_text": PROMPT, "gold_object": "Paris"})
        + "\n",
        encoding="utf-8",
    )
    return prompt_path


def test_adversarial_eval_end_to_end(tmp_path) -> None:
    index = FakeVectorIndex(
        [FakeVectorEntry("entry-a", QUERY.copy(), "Paris", "wiki:France")]
    )
    backend = CoLMLMAuditBackend(FakeGenerator(index))
    prompt_path = _write_prompt(tmp_path)
    output_dir = tmp_path / "adversarial"

    summary = run_adversarial_eval(
        prompt_path,
        backend,
        index=index,
        closure_config=ClosureConfig(predicates=("geometric",), radius=0.85),
        adversarial_config=AdversarialConfig(
            rho=0.85,
            epsilons=(0.05,),
            templates=("verbatim", "hyphenated"),
        ),
        output_dir=output_dir,
    )

    assert summary["attacked_facts"] == 1
    evasion = {row["template"]: row["evasion_rate"] for row in summary["evasion"]}
    # Geometric-only closure: no backstop, so the verbatim survivor restores
    # the fact; the fake model cannot decode the hyphenated paraphrase.
    assert evasion["verbatim"] == 1.0
    assert evasion["hyphenated"] == 0.0
    assert summary["margins"][0]["s_del"] == pytest.approx(1.0)
    # Injections stay out of the backend between calls.
    assert backend.injections == ()

    outputs = write_adversarial_outputs(summary, output_dir)
    assert outputs["evasion"].is_file()
    assert outputs["margins"].is_file()
    assert (output_dir / "closures" / "pA.json").is_file()


def test_adversarial_eval_resumes_without_regenerating(tmp_path) -> None:
    index = FakeVectorIndex(
        [FakeVectorEntry("entry-a", QUERY.copy(), "Paris", "wiki:France")]
    )
    backend = CoLMLMAuditBackend(FakeGenerator(index))
    prompt_path = _write_prompt(tmp_path)
    kwargs = dict(
        index=index,
        closure_config=ClosureConfig(predicates=("geometric",), radius=0.85),
        adversarial_config=AdversarialConfig(
            rho=0.85, epsilons=(0.05,), templates=("verbatim",)
        ),
        output_dir=tmp_path / "adversarial",
    )

    first = run_adversarial_eval(prompt_path, backend, **kwargs)
    assert first["executed_generations"] == 3  # del-off, baseline, 1 attack

    second = run_adversarial_eval(prompt_path, backend, **kwargs)
    assert second["executed_generations"] == 0
    assert second["evasion"] == first["evasion"]
