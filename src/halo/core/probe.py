from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from halo.core.entanglement import fact_key
from halo.core.equivalence import build_alias_set, normalize_text
from halo.core.metrics import _result_is_correct

VALID_PROBE_MODES = ("ranking", "classification")

# Fewer facts than this and a fact-disjoint fit is meaningless; the probe is
# skipped rather than run on noise.
MIN_PROBE_FACTS = 4


@dataclass(frozen=True)
class ProbeConfig:
    mode: str = "ranking"
    folds: int = 5
    seed: int = 0
    ridge_lambda: float = 1.0
    feature_dim: int = 2048

    def __post_init__(self) -> None:
        if self.mode not in VALID_PROBE_MODES:
            raise ValueError(
                f"Unknown probe mode {self.mode!r}; valid modes are "
                f"{list(VALID_PROBE_MODES)!r}."
            )
        if self.folds < 2:
            raise ValueError("folds must be at least 2.")
        if self.ridge_lambda <= 0:
            raise ValueError("ridge_lambda must be positive.")
        if self.feature_dim < 8:
            raise ValueError("feature_dim must be at least 8.")


@dataclass(frozen=True)
class ProbeSample:
    sample_id: str
    fact: str
    vector: np.ndarray


@dataclass(frozen=True)
class ProbeReport:
    per_fact: list[dict[str, Any]]
    summary: dict[str, Any]


def load_probe_samples(
    paths: Iterable[Path],
    state: str = "FULL",
) -> list[ProbeSample]:
    """Load query embeddings from sidecar .npz files.

    Supports both sidecar key formats: ``{fact}/{state}/event{n}`` from
    standard runs (only the requested state's event0 is used) and flat
    ``{fact}`` keys from sweep FULL passes. Vectors are L2-normalized on
    load to match the index's cosine geometry.
    """
    samples: list[ProbeSample] = []
    for path in paths:
        with np.load(path) as stored:
            for key in stored.files:
                parts = key.split("/")
                if len(parts) == 3:
                    fact, key_state, event = parts
                    if key_state != state or event != "event0":
                        continue
                elif len(parts) == 1:
                    fact = key
                else:
                    continue
                vector = np.asarray(stored[key], dtype=np.float32).reshape(-1)
                norm = float(np.linalg.norm(vector))
                if norm == 0.0:
                    continue
                samples.append(
                    ProbeSample(
                        sample_id=f"{Path(path).stem}:{key}",
                        fact=fact,
                        vector=vector / norm,
                    )
                )
    return samples


def load_labels_and_behavioral(
    paths: Iterable[Path],
) -> tuple[dict[str, dict[str, Any]], dict[str, float]]:
    """Per-fact labels (from FULL rows) and behavioral parametric leakage
    L(f) (mean DEL-OFF correctness) from audit result JSONL files."""
    labels: dict[str, dict[str, Any]] = {}
    del_off: dict[str, list[float]] = {}
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = fact_key(row)
                if not key:
                    continue
                state = row.get("state")
                if state == "FULL":
                    label = {
                        "ground_truth": str(row.get("ground_truth", "")),
                        "aliases": tuple(row.get("object_aliases") or ()),
                    }
                    existing = labels.get(key)
                    if existing is None:
                        labels[key] = label
                    elif normalize_text(
                        existing["ground_truth"]
                    ) != normalize_text(label["ground_truth"]):
                        raise ValueError(
                            f"Conflicting ground truths for fact {key!r}: "
                            f"{existing['ground_truth']!r} vs "
                            f"{label['ground_truth']!r}."
                        )
                elif state == "DEL-OFF":
                    del_off.setdefault(key, []).append(
                        1.0 if _result_is_correct(row) else 0.0
                    )
    behavioral = {
        key: sum(values) / len(values) for key, values in del_off.items()
    }
    return labels, behavioral


def answer_features(answer: str, dim: int) -> np.ndarray:
    """Deterministic hashed character-trigram features of an answer."""
    text = f"##{normalize_text(answer)}##"
    vector = np.zeros(dim, dtype=np.float32)
    for start in range(len(text) - 2):
        gram = text[start : start + 3]
        digest = hashlib.md5(gram.encode("utf-8")).hexdigest()
        vector[int(digest, 16) % dim] += 1.0
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0.0 else vector


def _ridge_fit(
    features: np.ndarray, targets: np.ndarray, ridge_lambda: float
) -> np.ndarray:
    gram = features.T @ features
    gram += ridge_lambda * np.eye(gram.shape[0], dtype=features.dtype)
    return np.linalg.solve(gram, features.T @ targets)


def _alias_norms(label: Mapping[str, Any]) -> set[str]:
    return {
        normalize_text(alias)
        for alias in build_alias_set(
            label["ground_truth"], list(label["aliases"])
        )
        if normalize_text(alias)
    }


def run_probe(
    samples: list[ProbeSample],
    labels: Mapping[str, Mapping[str, Any]],
    config: ProbeConfig,
) -> ProbeReport:
    usable = [sample for sample in samples if sample.fact in labels]
    dropped_no_label = len(samples) - len(usable)
    facts = sorted({sample.fact for sample in usable})
    if len(facts) < 2:
        raise ValueError(
            "The probe needs embeddings for at least two labeled facts; "
            f"got {len(facts)}."
        )

    folds = min(config.folds, len(facts))
    rng = np.random.default_rng(config.seed)
    order = list(facts)
    rng.shuffle(order)
    fold_of = {fact: position % folds for position, fact in enumerate(order)}

    features = np.stack([sample.vector for sample in usable])

    # Candidate answers, deduplicated by normalized form (first surface wins).
    candidate_by_norm: dict[str, str] = {}
    for fact in facts:
        answer = str(labels[fact]["ground_truth"])
        candidate_by_norm.setdefault(normalize_text(answer), answer)
    candidates = sorted(candidate_by_norm.values(), key=normalize_text)
    candidate_norms = [normalize_text(answer) for answer in candidates]

    if config.mode == "ranking":
        candidate_features = np.stack(
            [answer_features(answer, config.feature_dim) for answer in candidates]
        )
        targets = np.stack(
            [
                answer_features(
                    str(labels[sample.fact]["ground_truth"]), config.feature_dim
                )
                for sample in usable
            ]
        )
    else:
        class_index = {norm: idx for idx, norm in enumerate(candidate_norms)}
        targets = np.zeros((len(usable), len(candidates)), dtype=np.float32)
        for row_idx, sample in enumerate(usable):
            targets[
                row_idx,
                class_index[normalize_text(str(labels[sample.fact]["ground_truth"]))],
            ] = 1.0

    predictions: dict[int, str] = {}
    train_classes_of_fold: dict[int, set[str]] = {}
    for fold in range(folds):
        train_idx = [
            idx
            for idx, sample in enumerate(usable)
            if fold_of[sample.fact] != fold
        ]
        test_idx = [
            idx
            for idx, sample in enumerate(usable)
            if fold_of[sample.fact] == fold
        ]
        if not train_idx or not test_idx:
            train_classes_of_fold[fold] = set()
            continue
        train_classes_of_fold[fold] = {
            normalize_text(str(labels[usable[idx].fact]["ground_truth"]))
            for idx in train_idx
        }
        weights = _ridge_fit(
            features[train_idx], targets[train_idx], config.ridge_lambda
        )
        scores = features[test_idx] @ weights
        if config.mode == "ranking":
            scores = scores @ candidate_features.T
        winners = np.argmax(scores, axis=1)
        for offset, idx in enumerate(test_idx):
            predictions[idx] = candidate_norms[int(winners[offset])]

    per_fact: list[dict[str, Any]] = []
    for fact in facts:
        label = labels[fact]
        alias_norms = _alias_norms(label)
        sample_idx = [
            idx for idx, sample in enumerate(usable) if sample.fact == fact
        ]
        scored = [idx for idx in sample_idx if idx in predictions]
        correct = [
            1.0 if predictions[idx] in alias_norms else 0.0 for idx in scored
        ]
        fold = fold_of[fact]
        label_unseen = (
            config.mode == "classification"
            and normalize_text(str(label["ground_truth"]))
            not in train_classes_of_fold.get(fold, set())
        )
        per_fact.append(
            {
                "fact": fact,
                "answer": label["ground_truth"],
                "fold": fold,
                "samples": len(sample_idx),
                "samples_scored": len(scored),
                "l_rep": (sum(correct) / len(correct)) if correct else None,
                "predicted": (
                    candidate_by_norm.get(predictions[scored[0]])
                    if scored
                    else None
                ),
                "label_unseen": label_unseen,
            }
        )

    scored_facts = [row for row in per_fact if row["l_rep"] is not None]
    summary = {
        "mode": config.mode,
        "folds": folds,
        "seed": config.seed,
        "ridge_lambda": config.ridge_lambda,
        "feature_dim": config.feature_dim,
        "facts": len(facts),
        "samples": len(usable),
        "samples_dropped_no_label": dropped_no_label,
        "candidates": len(candidates),
        "label_unseen_count": sum(
            1 for row in per_fact if row["label_unseen"]
        ),
        "l_rep_hat": (
            sum(row["l_rep"] for row in scored_facts) / len(scored_facts)
            if scored_facts
            else None
        ),
    }
    return ProbeReport(per_fact=per_fact, summary=summary)


def compute_delta_rep(
    per_fact: list[dict[str, Any]],
    behavioral: Mapping[str, float],
) -> dict[str, Any]:
    """Delta_rep = mean L_rep - mean behavioral L over facts having both."""
    common = [
        row
        for row in per_fact
        if row["l_rep"] is not None and row["fact"] in behavioral
    ]
    if not common:
        return {
            "facts_common": 0,
            "l_rep_hat": None,
            "l_hat": None,
            "delta_rep": None,
        }
    l_rep_hat = sum(row["l_rep"] for row in common) / len(common)
    l_hat = sum(behavioral[row["fact"]] for row in common) / len(common)
    return {
        "facts_common": len(common),
        "l_rep_hat": l_rep_hat,
        "l_hat": l_hat,
        "delta_rep": l_rep_hat - l_hat,
    }


def probe_audit_outputs(
    results_paths: Iterable[Path],
    embeddings_paths: Iterable[Path],
    output_dir: Path,
    *,
    config: ProbeConfig | None = None,
    state: str = "FULL",
    stem: str = "",
) -> dict[str, Any] | None:
    """Fit the representational-leakage probe on a finished audit's results +
    query-embedding sidecar and write the probe CSVs.

    Returns a summary dict, or None when there are too few labeled facts with
    embeddings to fit a meaningful fact-disjoint probe (the common case for a
    tiny prompt file — skipped, not an error).
    """
    config = config or ProbeConfig()
    samples = load_probe_samples(embeddings_paths, state=state)
    labels, behavioral = load_labels_and_behavioral(results_paths)
    labeled_facts = {s.fact for s in samples if s.fact in labels}
    if len(labeled_facts) < max(MIN_PROBE_FACTS, config.folds):
        return None

    report = run_probe(samples, labels, config)
    delta = compute_delta_rep(report.per_fact, behavioral)

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{stem}_" if stem else ""
    per_fact_path = output_dir / f"{prefix}probe_per_fact.csv"
    summary_path = output_dir / f"{prefix}probe_summary.csv"

    per_fact_rows = [
        {**row, "behavioral_l": behavioral.get(row["fact"])}
        for row in report.per_fact
    ]
    with per_fact_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_fact_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_fact_rows)

    summary_row = {**report.summary, **delta}
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_row.keys()))
        writer.writeheader()
        writer.writerow(summary_row)

    return {
        **report.summary,
        **delta,
        "per_fact_path": per_fact_path,
        "summary_path": summary_path,
    }
