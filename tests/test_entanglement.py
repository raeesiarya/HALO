import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lmlm_audit.colmlm.backend import CoLMLMAuditBackend
from lmlm_audit.colmlm.closure import ClosureConfig, build_closure_family
from lmlm_audit.core.entanglement import compute_entanglement
from lmlm_audit.core.examples import AuditExample
from lmlm_audit.core.neighbors import (
    NeighborConfig,
    compute_cosine_neighbors,
    compute_same_source_neighbors,
)
from lmlm_audit.cli.reporting import write_entanglement_outputs
from lmlm_audit.cli.run_audit import parse_radius_grid
from lmlm_audit.cli.runner import run_entanglement_sweep


# --- pure metrics ----------------------------------------------------------


def _row(key, output, truth, *, target=None, rho=None, role=None):
    row = {
        "prompt_id": key,
        "model_output": output,
        "ground_truth": truth,
    }
    if target is not None:
        row["sweep"] = {"target_key": target, "rho": rho, "role": role}
    return row


def test_compute_entanglement_reports_a_positive_gap() -> None:
    full_rows = [_row("t", "Paris", "Paris"), _row("n1", "Warsaw", "Warsaw")]
    sweep_rows = [
        # rho 0.9: deletion ineffective (E=0), neighbor intact (X=0).
        _row("t", "Paris", "Paris", target="t", rho=0.9, role="target"),
        _row("n1", "Warsaw", "Warsaw", target="t", rho=0.9, role="neighbor"),
        # rho 0.5: deletion works (E=1) but breaks the neighbor (X=1).
        _row("t", "unknown", "Paris", target="t", rho=0.5, role="target"),
        _row("n1", "unknown", "Warsaw", target="t", rho=0.5, role="neighbor"),
    ]

    entanglement = compute_entanglement(
        sweep_rows, full_rows, {"t": ["n1"]}
    )
    curve = entanglement["t"]["curve"]
    assert [point["rho"] for point in curve] == [0.9, 0.5]
    assert [point["efficacy"] for point in curve] == [0.0, 1.0]
    assert [point["collateral"] for point in curve] == [0.0, 1.0]
    assert entanglement["t"]["gap"] == 1.0


def test_collateral_ignores_neighbors_wrong_under_full() -> None:
    full_rows = [_row("t", "Paris", "Paris"), _row("n1", "Krakow", "Warsaw")]
    sweep_rows = [
        _row("t", "unknown", "Paris", target="t", rho=0.8, role="target"),
        _row("n1", "unknown", "Warsaw", target="t", rho=0.8, role="neighbor"),
    ]

    entanglement = compute_entanglement(
        sweep_rows, full_rows, {"t": ["n1"]}
    )
    point = entanglement["t"]["curve"][0]
    # The neighbor was already wrong under FULL, so it cannot be collateral.
    assert point["collateral"] == 0.0
    assert point["neighbors_full_correct"] == 0
    assert entanglement["t"]["gap"] == 0.0
    assert entanglement["t"]["gap_rho"] == 0.8


def test_fact_with_no_neighbors_has_zero_collateral() -> None:
    full_rows = [_row("t", "Paris", "Paris")]
    sweep_rows = [
        _row("t", "unknown", "Paris", target="t", rho=0.8, role="target"),
    ]
    entanglement = compute_entanglement(sweep_rows, full_rows, {"t": []})
    assert entanglement["t"]["gap"] == 0.0


# --- neighbors -------------------------------------------------------------


def test_cosine_neighbors_respect_ball_and_cap() -> None:
    embeddings = {
        "a": np.asarray([1.0, 0.0]),
        "b": np.asarray([0.8, 0.6]),
        "c": np.asarray([-1.0, 0.0]),
        "empty": np.asarray([0.0, 0.0]),
    }
    neighbors = compute_cosine_neighbors(
        embeddings, NeighborConfig(mode="cosine", ball=0.5, cap=20)
    )
    assert [item["neighbor"] for item in neighbors["a"]] == ["b"]
    assert neighbors["a"][0]["cosine"] == pytest.approx(0.8)
    assert neighbors["c"] == []
    assert neighbors["empty"] == []

    capped = compute_cosine_neighbors(
        {
            "a": np.asarray([1.0, 0.0]),
            "b": np.asarray([0.9, math.sqrt(1 - 0.81)]),
            "d": np.asarray([0.8, 0.6]),
        },
        NeighborConfig(mode="cosine", ball=0.5, cap=1),
    )
    assert [item["neighbor"] for item in capped["a"]] == ["b"]


def test_same_source_neighbors() -> None:
    neighbors = compute_same_source_neighbors(
        {"a": "s1", "b": "s1", "c": "s2", "d": None},
        NeighborConfig(mode="same-source"),
    )
    assert [item["neighbor"] for item in neighbors["a"]] == ["b"]
    assert neighbors["c"] == []
    assert neighbors["d"] == []


def test_neighbor_config_validation() -> None:
    with pytest.raises(ValueError, match="mode"):
        NeighborConfig(mode="psychic")
    with pytest.raises(ValueError, match="cap"):
        NeighborConfig(cap=0)


# --- closure family --------------------------------------------------------


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
        self.calls = []

    def search(self, query, top_k=1, similarity_threshold=None):
        self.calls.append((top_k, similarity_threshold))
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


QUERY_A = np.asarray([1.0, 0.0], dtype=np.float32)
QUERY_B = np.asarray([0.6, 0.8], dtype=np.float32)


def _two_fact_index() -> FakeVectorIndex:
    return FakeVectorIndex(
        [
            FakeVectorEntry("entry-a", QUERY_A.copy(), "Paris", "wiki:France"),
            FakeVectorEntry("entry-b", QUERY_B.copy(), "Warsaw", "wiki:Poland"),
        ]
    )


def _example_a() -> AuditExample:
    return AuditExample.from_prompt_row(
        {
            "prompt_id": "pA",
            "prompt_text": "What is the capital of France?",
            "gold_object": "Paris",
        }
    )


def test_closure_family_sets_are_nested_and_share_one_search() -> None:
    index = _two_fact_index()
    family = build_closure_family(
        index=index,
        example=_example_a(),
        query_vector=QUERY_A,
        config=ClosureConfig(predicates=("geometric",)),
        radii=(0.9, 0.5),
    )

    ids_tight = {entry.entry_id for entry in family[0.9].entries}
    ids_loose = {entry.entry_id for entry in family[0.5].entries}
    assert ids_tight == {"entry-a"}
    assert ids_loose == {"entry-a", "entry-b"}
    assert ids_tight <= ids_loose
    assert family[0.9].config.radius == 0.9
    assert family[0.5].config.radius == 0.5
    # One geometric search at the loosest radius serves both closures.
    assert index.calls == [(10_000, 0.5)]
    assert family[0.9].truncated is False


def test_closure_family_truncation_only_flags_affected_radii() -> None:
    index = _two_fact_index()
    family = build_closure_family(
        index=index,
        example=_example_a(),
        query_vector=QUERY_A,
        config=ClosureConfig(predicates=("geometric",), max_closure_size=1),
        radii=(0.9, 0.5),
    )
    # The page holds only entry-a (score 1.0); anything at or below that
    # score may be missing, so both radii are conservatively flagged.
    assert family[0.5].truncated is True
    assert family[0.9].truncated is True


# --- sweep runner ----------------------------------------------------------


class SweepFakeGenerator:
    """Routes each prompt to its own query vector."""

    def __init__(self, index, queries: dict[str, np.ndarray]) -> None:
        self.index = index
        self.queries = dict(queries)
        self.generation_config = SimpleNamespace(max_new_tokens=64)
        self.retrieval_config = SimpleNamespace(similarity_threshold=0.7)
        self.generate_calls = 0

    def generate(self, prompt):
        self.generate_calls += 1
        results = self.index.search(
            self.queries[prompt],
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
        )


PROMPT_A = "What is the capital of France?"
PROMPT_B = "What is the capital of Poland?"


def _sweep_setup():
    index = _two_fact_index()
    generator = SweepFakeGenerator(
        index, {PROMPT_A: QUERY_A, PROMPT_B: QUERY_B}
    )
    backend = CoLMLMAuditBackend(generator)
    return index, generator, backend


def _write_sweep_prompts(tmp_path: Path) -> Path:
    prompt_path = tmp_path / "prompts.jsonl"
    prompt_path.write_text(
        json.dumps(
            {
                "prompt_id": "pA",
                "prompt_text": PROMPT_A,
                "gold_object": "Paris",
            }
        )
        + "\n"
        + json.dumps(
            {
                "prompt_id": "pB",
                "prompt_text": PROMPT_B,
                "gold_object": "Warsaw",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return prompt_path


def test_entanglement_sweep_end_to_end(tmp_path) -> None:
    index, generator, backend = _sweep_setup()
    prompt_path = _write_sweep_prompts(tmp_path)
    output_dir = tmp_path / "sweep"

    summary = run_entanglement_sweep(
        prompt_path,
        backend,
        index=index,
        radii=(0.9, 0.5),
        closure_config=ClosureConfig(),
        neighbor_config=NeighborConfig(mode="cosine", ball=0.5, cap=20),
        output_dir=output_dir,
    )

    assert summary["swept_facts"] == 2
    assert summary["skipped_facts"] == []
    # 2 facts x 2 radii x (1 target + 1 neighbor)
    assert summary["planned_generations"] == 8
    assert summary["executed_generations"] == 8

    for fact in ("pA", "pB"):
        curve = summary["entanglement"][fact]["curve"]
        # Tight radius: fact forgotten, neighbor untouched -> clean deletion.
        assert curve[0] == {
            "rho": 0.9,
            "efficacy": 1.0,
            "collateral": 0.0,
            "neighbor_count": 1,
            "neighbors_observed": 1,
            "neighbors_full_correct": 1,
            "gap_term": 0.0,
        }
        # Loose radius: the neighbor's entry falls inside the closure.
        assert curve[1]["collateral"] == 1.0
        assert summary["entanglement"][fact]["gap"] == 0.0
        assert summary["entanglement"][fact]["gap_rho"] == 0.9

    assert (output_dir / "full_results.jsonl").is_file()
    assert (output_dir / "full_query_embeddings.npz").is_file()
    assert (output_dir / "neighbors.json").is_file()
    assert (output_dir / "sweep_rho_0.9000.jsonl").is_file()
    assert (output_dir / "sweep_rho_0.5000.jsonl").is_file()

    outputs = write_entanglement_outputs(summary["entanglement"], output_dir)
    assert outputs["curves"].is_file()
    assert outputs["gaps"].is_file()
    assert outputs["figure"].stat().st_size > 0


def test_entanglement_sweep_resumes_without_regenerating(tmp_path) -> None:
    index, generator, backend = _sweep_setup()
    prompt_path = _write_sweep_prompts(tmp_path)
    output_dir = tmp_path / "sweep"
    kwargs = dict(
        index=index,
        radii=(0.9, 0.5),
        closure_config=ClosureConfig(),
        neighbor_config=NeighborConfig(mode="cosine", ball=0.5, cap=20),
        output_dir=output_dir,
    )

    first = run_entanglement_sweep(prompt_path, backend, **kwargs)
    calls_after_first = generator.generate_calls

    second = run_entanglement_sweep(prompt_path, backend, **kwargs)
    assert generator.generate_calls == calls_after_first
    assert second["executed_generations"] == 0
    assert second["entanglement"] == first["entanglement"]


# --- CLI grid parsing ------------------------------------------------------


def test_parse_radius_grid() -> None:
    assert parse_radius_grid("0.95:0.70:0.05") == (
        0.95,
        0.9,
        0.85,
        0.8,
        0.75,
        0.7,
    )
    with pytest.raises(ValueError, match="start:stop:step"):
        parse_radius_grid("0.9:0.5")
    with pytest.raises(ValueError, match="descend"):
        parse_radius_grid("0.5:0.9:0.1")
    with pytest.raises(ValueError, match="positive"):
        parse_radius_grid("0.9:0.5:0")
