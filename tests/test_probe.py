import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from halo.core.probe import (
    ProbeConfig,
    ProbeSample,
    answer_features,
    compute_delta_rep,
    load_labels_and_behavioral,
    load_probe_samples,
    probe_audit_outputs,
    run_probe,
)


# --- planted structure -----------------------------------------------------

COMPOSITIONAL_ANSWERS = [
    f"{first} {second}"
    for first in ("red", "yellow")
    for second in ("blue", "green", "cat", "dog")
]


def _labels(answers):
    return {
        f"fact{i}": {"ground_truth": answer, "aliases": ()}
        for i, answer in enumerate(answers)
    }


def test_ranking_probe_decodes_linearly_structured_embeddings() -> None:
    # Embeddings ARE the answer features: the readout is exactly linear and
    # held-out answers stay decodable because their character n-grams are
    # shared across training facts.
    samples = [
        ProbeSample(f"s{i}", f"fact{i}", answer_features(answer, 256))
        for i, answer in enumerate(COMPOSITIONAL_ANSWERS)
    ]
    report = run_probe(
        samples,
        _labels(COMPOSITIONAL_ANSWERS),
        ProbeConfig(mode="ranking", folds=4, ridge_lambda=0.001, feature_dim=256),
    )
    assert report.summary["l_rep_hat"] >= 0.6
    assert report.summary["label_unseen_count"] == 0


def test_ranking_probe_is_at_chance_on_random_embeddings() -> None:
    rng = np.random.default_rng(7)
    samples = []
    for i in range(len(COMPOSITIONAL_ANSWERS)):
        vector = rng.normal(size=256).astype(np.float32)
        samples.append(
            ProbeSample(f"s{i}", f"fact{i}", vector / np.linalg.norm(vector))
        )
    report = run_probe(
        samples,
        _labels(COMPOSITIONAL_ANSWERS),
        ProbeConfig(mode="ranking", folds=4, ridge_lambda=0.001, feature_dim=256),
    )
    assert report.summary["l_rep_hat"] <= 0.4


# --- classification mode ---------------------------------------------------


def _clustered_sample(fact, class_axis, rng):
    vector = np.zeros(16, dtype=np.float32)
    vector[class_axis] = 1.0
    vector += rng.normal(scale=0.01, size=16).astype(np.float32)
    return ProbeSample(f"s-{fact}", fact, vector / np.linalg.norm(vector))


def test_classification_probe_with_shared_answers() -> None:
    rng = np.random.default_rng(0)
    labels = {}
    samples = []
    for i in range(6):
        fact = f"fact{i}"
        answer = "Paris" if i % 2 == 0 else "Warsaw"
        labels[fact] = {"ground_truth": answer, "aliases": ()}
        samples.append(_clustered_sample(fact, i % 2, rng))

    report = run_probe(
        samples, labels, ProbeConfig(mode="classification", folds=5)
    )
    assert report.summary["l_rep_hat"] == 1.0
    assert report.summary["label_unseen_count"] == 0
    assert report.summary["candidates"] == 2


def test_classification_flags_singleton_labels_as_unseen() -> None:
    rng = np.random.default_rng(0)
    labels = {}
    samples = []
    for i in range(6):
        fact = f"fact{i}"
        answer = "Paris" if i % 2 == 0 else "Warsaw"
        labels[fact] = {"ground_truth": answer, "aliases": ()}
        samples.append(_clustered_sample(fact, i % 2, rng))
    labels["fact-berlin"] = {"ground_truth": "Berlin", "aliases": ()}
    samples.append(_clustered_sample("fact-berlin", 2, rng))

    report = run_probe(
        samples, labels, ProbeConfig(mode="classification", folds=5)
    )
    berlin = next(
        row for row in report.per_fact if row["fact"] == "fact-berlin"
    )
    assert berlin["label_unseen"] is True
    assert report.summary["label_unseen_count"] == 1


def test_samples_of_one_fact_share_a_fold() -> None:
    rng = np.random.default_rng(0)
    labels = {
        f"fact{i}": {"ground_truth": "Paris" if i % 2 == 0 else "Warsaw", "aliases": ()}
        for i in range(4)
    }
    samples = []
    for i in range(4):
        samples.append(_clustered_sample(f"fact{i}", i % 2, rng))
        # A second embedding of the same fact (e.g. another prompt variant).
        samples.append(
            ProbeSample(
                f"s-{i}-b",
                f"fact{i}",
                samples[-1].vector,
            )
        )
    report = run_probe(
        samples, labels, ProbeConfig(mode="classification", folds=2)
    )
    for row in report.per_fact:
        assert row["samples"] == 2
        assert row["samples_scored"] == 2
        # Both samples were scored out-of-fold together: their agreement
        # makes l_rep integral.
        assert row["l_rep"] in (0.0, 1.0)


def test_probe_config_validation() -> None:
    with pytest.raises(ValueError, match="mode"):
        ProbeConfig(mode="psychic")
    with pytest.raises(ValueError, match="folds"):
        ProbeConfig(folds=1)
    with pytest.raises(ValueError, match="ridge_lambda"):
        ProbeConfig(ridge_lambda=0.0)


# --- loaders ---------------------------------------------------------------


def test_load_probe_samples_handles_both_key_formats(tmp_path) -> None:
    sidecar = tmp_path / "embeddings.npz"
    np.savez(
        sidecar,
        **{
            "f1/FULL/event0": np.asarray([1.0, 0.0]),
            "f1/DEL-ON/event0": np.asarray([9.0, 9.0]),
            "f2/FULL/event1": np.asarray([8.0, 8.0]),
            "f3": np.asarray([0.0, 2.0]),
            "f4/FULL/event0": np.asarray([0.0, 0.0]),
        },
    )
    samples = load_probe_samples([sidecar], state="FULL")
    by_fact = {sample.fact: sample for sample in samples}
    # DEL-ON, non-event0, and zero-norm vectors are skipped.
    assert set(by_fact) == {"f1", "f3"}
    np.testing.assert_allclose(by_fact["f3"].vector, [0.0, 1.0])


def _result_row(key, state, output, truth):
    return {
        "prompt_id": key,
        "state": state,
        "model_output": output,
        "ground_truth": truth,
    }


def test_load_labels_and_behavioral(tmp_path) -> None:
    results = tmp_path / "results.jsonl"
    rows = [
        _result_row("f1", "FULL", "Paris", "Paris"),
        _result_row("f1", "DEL-OFF", "Paris", "Paris"),
        _result_row("f1", "DEL-OFF", "unknown", "Paris"),
        _result_row("f2", "FULL", "Warsaw", "Warsaw"),
        _result_row("f2", "DEL-ON", "unknown", "Warsaw"),
    ]
    results.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    labels, behavioral = load_labels_and_behavioral([results])
    assert labels["f1"]["ground_truth"] == "Paris"
    assert behavioral == {"f1": 0.5}
    assert "f2" not in behavioral


def test_conflicting_ground_truths_raise(tmp_path) -> None:
    results = tmp_path / "results.jsonl"
    rows = [
        _result_row("f1", "FULL", "Paris", "Paris"),
        _result_row("f1", "FULL", "Lyon", "Lyon"),
    ]
    results.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Conflicting ground truths"):
        load_labels_and_behavioral([results])


# --- delta_rep -------------------------------------------------------------


def test_compute_delta_rep() -> None:
    per_fact = [
        {"fact": "f1", "l_rep": 1.0},
        {"fact": "f2", "l_rep": 1.0},
        {"fact": "f3", "l_rep": 0.0},
        {"fact": "no-behavioral", "l_rep": 1.0},
    ]
    delta = compute_delta_rep(per_fact, {"f1": 0.0, "f2": 1.0, "f3": 0.0})
    assert delta["facts_common"] == 3
    assert delta["l_rep_hat"] == pytest.approx(2 / 3)
    assert delta["l_hat"] == pytest.approx(1 / 3)
    assert delta["delta_rep"] == pytest.approx(1 / 3)

    empty = compute_delta_rep(per_fact, {})
    assert empty["facts_common"] == 0
    assert empty["delta_rep"] is None


# --- CLI -------------------------------------------------------------------


def test_probe_audit_outputs_end_to_end(tmp_path) -> None:
    rng = np.random.default_rng(0)
    results = tmp_path / "prompts_results.jsonl"
    rows = []
    vectors = {}
    for i in range(6):
        key = f"fact{i}"
        answer = "Paris" if i % 2 == 0 else "Warsaw"
        rows.append(_result_row(key, "FULL", answer, answer))
        rows.append(_result_row(key, "DEL-OFF", "unknown", answer))
        sample = _clustered_sample(key, i % 2, rng)
        vectors[f"{key}/FULL/event0"] = sample.vector
    results.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    sidecar = tmp_path / "prompts_query_embeddings.npz"
    np.savez(sidecar, **vectors)

    summary = probe_audit_outputs(
        results_paths=[results],
        embeddings_paths=[sidecar],
        output_dir=tmp_path,
        config=ProbeConfig(mode="classification"),
        stem="prompts",
    )

    assert summary is not None
    assert summary["mode"] == "classification"
    assert summary["l_rep_hat"] == 1.0
    # Every DEL-OFF row was wrong, so the probe decodes what behavior hides.
    assert summary["l_hat"] == 0.0
    assert summary["delta_rep"] == 1.0
    assert summary["facts_common"] == 6

    with (tmp_path / "prompts_probe_summary.csv").open() as handle:
        summary_row = next(csv.DictReader(handle))
    assert float(summary_row["delta_rep"]) == 1.0
    with (tmp_path / "prompts_probe_per_fact.csv").open() as handle:
        per_fact = list(csv.DictReader(handle))
    assert len(per_fact) == 6


def test_probe_audit_outputs_skips_when_too_few_facts(tmp_path) -> None:
    results = tmp_path / "r_results.jsonl"
    results.write_text(
        json.dumps(_result_row("f1", "FULL", "Paris", "Paris")) + "\n",
        encoding="utf-8",
    )
    sidecar = tmp_path / "r_query_embeddings.npz"
    np.savez(sidecar, **{"f1/FULL/event0": np.asarray([1.0, 0.0])})
    assert (
        probe_audit_outputs(
            results_paths=[results],
            embeddings_paths=[sidecar],
            output_dir=tmp_path,
            stem="r",
        )
        is None
    )
