from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from halo.core.entanglement import fact_key
from halo.core.equivalence import build_alias_set, normalize_text
from halo.core.metrics import _result_is_correct
from halo.core.probe import (
    ProbeConfig,
    ProbeSample,
    assign_group_folds,
    compute_delta_rep,
    load_probe_samples,
    run_probe,
)


DATASET_STEMS: dict[str, str] = {
    "trex": "prompts_trex",
    "popqa": "prompts",
    "googlere": "prompts_googlere",
    "counterfact": "prompts_counterfact",
    "zsre": "prompts_zsre",
}

FREQUENCY_BINS: tuple[tuple[str, int, int | None], ...] = (
    ("0", 0, 0),
    ("1", 1, 1),
    ("2-4", 2, 4),
    ("5-9", 5, 9),
    ("10-24", 10, 24),
    ("25-49", 25, 49),
    ("50-99", 50, 99),
    ("100+", 100, None),
)

_WORD_PATTERN = re.compile(r"\w+(?:[-']\w+)*", re.UNICODE)


@dataclass
class AuditFact:
    dataset: str
    fact_id: str
    prompt: str = ""
    ground_truth: str = ""
    aliases: tuple[str, ...] = ()
    subject: str = ""
    relation: str = ""
    full_correct: bool | None = None
    del_on_correct: bool | None = None
    del_off_correct: bool | None = None
    value_count_top_k: int = 0
    geometric_count: int = 0
    envelope_top_k: int | None = None
    del_on_output: str = ""
    del_off_output: str = ""
    del_on_selected: dict[str, Any] = field(default_factory=dict)

    @property
    def answer_norms(self) -> set[str]:
        return {
            normalize_text(value)
            for value in build_alias_set(self.ground_truth, list(self.aliases))
            if normalize_text(value)
        }

    @property
    def proposition_group(self) -> str:
        """Conservative canonical group used for probe splitting.

        Subject-bearing datasets can contain several surface records for the
        same proposition (notably ZsRE).  If a subject is unavailable, as in
        PopQA, falling back to the fact ID avoids incorrectly merging every
        question that happens to share an answer.
        """
        subject = normalize_text(self.subject)
        if not subject:
            return f"fact:{self.fact_id}"
        relation = normalize_text(self.relation)
        answer = normalize_text(self.ground_truth)
        return f"proposition:{subject}\x1f{relation}\x1f{answer}"


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(name)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: Iterable[bool | float | int]) -> float | None:
    materialized = [float(value) for value in values]
    return sum(materialized) / len(materialized) if materialized else None


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _fact_id(row: Mapping[str, Any]) -> str:
    return fact_key(row) or str(row.get("prompt_id") or row.get("fact_id") or "")


def load_audit_facts(path: Path, dataset: str) -> dict[str, AuditFact]:
    """Stream a three-state result file into compact per-fact records."""
    facts: dict[str, AuditFact] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = _fact_id(row)
            if not key:
                continue
            fact = facts.setdefault(key, AuditFact(dataset=dataset, fact_id=key))
            state = str(row.get("state") or "")
            if state == "FULL":
                fact.prompt = str(row.get("prompt") or "")
                fact.ground_truth = str(row.get("ground_truth") or "")
                fact.aliases = tuple(str(value) for value in row.get("object_aliases") or ())
                fact.subject = str(row.get("subject") or "")
                fact.relation = str(row.get("relation") or "")
                fact.full_correct = _result_is_correct(row)
                metadata = (row.get("deletion_manifest") or {}).get("metadata") or {}
                counts = metadata.get("entry_counts") or {}
                fact.value_count_top_k = int(counts.get("value") or 0)
                fact.geometric_count = int(counts.get("geometric") or 0)
                envelope = metadata.get("envelope_top_k")
                fact.envelope_top_k = int(envelope) if envelope is not None else None
            elif state == "DEL-ON":
                fact.del_on_correct = _result_is_correct(row)
                fact.del_on_output = str(row.get("model_output") or "")
                fact.del_on_selected = dict(
                    (row.get("retrieval_trace") or {}).get("selected_candidate") or {}
                )
            elif state == "DEL-OFF":
                fact.del_off_correct = _result_is_correct(row)
                fact.del_off_output = str(row.get("model_output") or "")
    return facts


def load_full_correctness(path: Path) -> dict[str, bool]:
    """Load only the FULL arm of a plain-LM result file."""
    correctness: dict[str, bool] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("state") != "FULL":
                continue
            key = _fact_id(row)
            if key:
                correctness[key] = _result_is_correct(row)
    return correctness


def _alias_correct(prediction: str, fact: AuditFact) -> bool:
    return normalize_text(prediction) in fact.answer_norms


def _prior_baselines(
    facts: Mapping[str, AuditFact],
    fold_of: Mapping[str, int],
) -> dict[str, float | int | None]:
    global_correct = 0
    relation_correct = 0
    answer_seen = 0
    for fold in sorted(set(fold_of.values())):
        train = [fact for key, fact in facts.items() if fold_of[key] != fold]
        test = [fact for key, fact in facts.items() if fold_of[key] == fold]
        global_counts = Counter(normalize_text(fact.ground_truth) for fact in train)
        if not global_counts:
            continue
        global_prediction = min(
            global_counts,
            key=lambda answer: (-global_counts[answer], answer),
        )
        relation_counts: dict[str, Counter[str]] = defaultdict(Counter)
        for fact in train:
            relation_counts[normalize_text(fact.relation)][
                normalize_text(fact.ground_truth)
            ] += 1
        train_answers = set(global_counts)
        for fact in test:
            relation = normalize_text(fact.relation)
            counts = relation_counts.get(relation) if relation else None
            relation_prediction = (
                min(counts, key=lambda answer: (-counts[answer], answer))
                if counts
                else global_prediction
            )
            global_correct += _alias_correct(global_prediction, fact)
            relation_correct += _alias_correct(relation_prediction, fact)
            answer_seen += normalize_text(fact.ground_truth) in train_answers
    count = len(facts)
    return {
        "global_majority_accuracy": _safe_rate(global_correct, count),
        "relation_majority_accuracy": _safe_rate(relation_correct, count),
        "answer_seen_rate": _safe_rate(answer_seen, count),
    }


def probe_distribution_diagnostics(
    dataset: str,
    facts: Mapping[str, AuditFact],
    *,
    folds: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ids = sorted(facts)
    canonical = {key: facts[key].proposition_group for key in ids}
    legacy_fold, _, _ = assign_group_folds(
        ids, requested_folds=folds, seed=seed
    )
    grouped_fold, group_of, effective_folds = assign_group_folds(
        ids,
        requested_folds=folds,
        seed=seed,
        fold_groups=canonical,
    )
    group_members: dict[str, list[str]] = defaultdict(list)
    for key, group in group_of.items():
        group_members[group].append(key)
    duplicate_groups = {
        group: members for group, members in group_members.items() if len(members) > 1
    }
    legacy_split_groups = sum(
        len({legacy_fold[key] for key in members}) > 1
        for members in duplicate_groups.values()
    )
    answers = Counter(normalize_text(fact.ground_truth) for fact in facts.values())
    probabilities = [count / len(facts) for count in answers.values()]
    entropy = -sum(p * math.log(p) for p in probabilities if p)
    normalized_entropy = (
        entropy / math.log(len(answers)) if len(answers) > 1 else 0.0
    )
    priors = _prior_baselines(facts, grouped_fold)
    summary = {
        "dataset": dataset,
        "facts": len(facts),
        "candidate_answers": len(answers),
        "uniform_chance": 1.0 / len(answers) if answers else None,
        "empirical_top_answer_rate": max(answers.values()) / len(facts) if facts else None,
        "normalized_answer_entropy": normalized_entropy,
        "relations": len({normalize_text(fact.relation) for fact in facts.values()}),
        "canonical_groups": len(group_members),
        "duplicate_groups": len(duplicate_groups),
        "facts_in_duplicate_groups": sum(map(len, duplicate_groups.values())),
        "largest_canonical_group": max(map(len, group_members.values())),
        "duplicate_groups_split_by_legacy_folds": legacy_split_groups,
        "folds": effective_folds,
        **priors,
    }
    assignments = [
        {
            "dataset": dataset,
            "fact_id": key,
            "canonical_group": group_of[key],
            "canonical_group_size": len(group_members[group_of[key]]),
            "legacy_fact_fold": legacy_fold[key],
            "canonical_group_fold": grouped_fold[key],
        }
        for key in ids
    ]
    return summary, assignments


def _frequency_bin(value: int) -> str:
    for label, low, high in FREQUENCY_BINS:
        if value >= low and (high is None or value <= high):
            return label
    raise AssertionError(f"No frequency bin for {value}.")


def matched_frequency_rows(
    dataset: str,
    facts: Mapping[str, AuditFact],
    standard: Mapping[str, bool],
    smollm2: Mapping[str, bool],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in sorted(facts):
        fact = facts[key]
        rows.append(
            {
                "dataset": dataset,
                "fact_id": key,
                "canonical_group": fact.proposition_group,
                "subject": fact.subject,
                "relation": fact.relation,
                "ground_truth": fact.ground_truth,
                "full_correct": fact.full_correct,
                "co_lmlm_del_off_correct": fact.del_off_correct,
                "standard_lm_correct": standard.get(key),
                "smollm2_correct": smollm2.get(key),
                # This is deliberately named as a local top-K proxy rather
                # than "fact frequency": an answer mention can support an
                # unrelated proposition, and entries outside the retrieval
                # envelope were not counted by the audit.
                "answer_bearing_neighbors_top_k": fact.value_count_top_k,
                "retrieval_envelope_top_k": fact.envelope_top_k,
                "geometric_neighbors": fact.geometric_count,
                "frequency_bin": _frequency_bin(fact.value_count_top_k),
            }
        )
    return rows


def frequency_bin_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for cohort_name, cohort_test in (
        ("answer_mention", lambda row: True),
        ("full_correct", lambda row: row.get("full_correct") is True),
    ):
        cohort = [row for row in rows if cohort_test(row)]
        for label, _, _ in FREQUENCY_BINS:
            members = [row for row in cohort if row["frequency_bin"] == label]
            if not members:
                continue
            output.append(
                {
                    "dataset": members[0]["dataset"],
                    "cohort": cohort_name,
                    "frequency_bin": label,
                    "facts": len(members),
                    "co_lmlm_del_off_accuracy": _mean(
                        row["co_lmlm_del_off_correct"]
                        for row in members
                        if row["co_lmlm_del_off_correct"] is not None
                    ),
                    "standard_lm_accuracy": _mean(
                        row["standard_lm_correct"]
                        for row in members
                        if row["standard_lm_correct"] is not None
                    ),
                    "smollm2_accuracy": _mean(
                        row["smollm2_correct"]
                        for row in members
                        if row["smollm2_correct"] is not None
                    ),
                }
            )
    return output


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2:
        return None
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = float(
        np.sqrt((left_centered @ left_centered) * (right_centered @ right_centered))
    )
    return float(left_centered @ right_centered / denominator) if denominator else None


def _logistic_frequency_effect(
    frequencies: np.ndarray,
    outcomes: np.ndarray,
) -> dict[str, float | None]:
    """Univariate logit effect per doubling of ``frequency + 1``."""
    if len(outcomes) < 3 or len(set(outcomes.tolist())) < 2:
        return {
            "odds_ratio_per_doubling": None,
            "odds_ratio_ci_low": None,
            "odds_ratio_ci_high": None,
        }
    x = np.column_stack(
        [np.ones(len(frequencies)), np.log2(frequencies.astype(np.float64) + 1.0)]
    )
    beta = np.zeros(2, dtype=np.float64)
    information = None
    for _ in range(100):
        linear = np.clip(x @ beta, -30.0, 30.0)
        probabilities = 1.0 / (1.0 + np.exp(-linear))
        weights = np.maximum(probabilities * (1.0 - probabilities), 1e-9)
        information = x.T @ (weights[:, None] * x)
        information += np.eye(2) * 1e-8
        step = np.linalg.solve(information, x.T @ (outcomes - probabilities))
        beta += step
        if float(np.max(np.abs(step))) < 1e-9:
            break
    if information is None:
        return {
            "odds_ratio_per_doubling": None,
            "odds_ratio_ci_low": None,
            "odds_ratio_ci_high": None,
        }
    covariance = np.linalg.inv(information)
    standard_error = float(np.sqrt(max(covariance[1, 1], 0.0)))
    slope = float(beta[1])
    exp_safe = lambda value: math.exp(float(np.clip(value, -50.0, 50.0)))
    return {
        "odds_ratio_per_doubling": exp_safe(slope),
        "odds_ratio_ci_low": exp_safe(slope - 1.96 * standard_error),
        "odds_ratio_ci_high": exp_safe(slope + 1.96 * standard_error),
    }


def frequency_association_summary(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    outcome_fields = (
        ("co-lmlm-del-off", "co_lmlm_del_off_correct"),
        ("standard-lm-360m-fw", "standard_lm_correct"),
        ("smollm2-360m", "smollm2_correct"),
    )
    for cohort_name, cohort_test in (
        ("answer_mention", lambda row: True),
        ("full_correct", lambda row: row.get("full_correct") is True),
    ):
        cohort = [row for row in rows if cohort_test(row)]
        for model, field_name in outcome_fields:
            members = [row for row in cohort if row.get(field_name) is not None]
            frequencies = np.asarray(
                [row["answer_bearing_neighbors_top_k"] for row in members],
                dtype=np.float64,
            )
            outcomes = np.asarray(
                [float(row[field_name]) for row in members], dtype=np.float64
            )
            spearman = _pearson(_average_ranks(frequencies), _average_ranks(outcomes))
            effect = _logistic_frequency_effect(frequencies, outcomes)
            output.append(
                {
                    "dataset": members[0]["dataset"] if members else "",
                    "cohort": cohort_name,
                    "model": model,
                    "facts": len(members),
                    "correct": int(outcomes.sum()) if len(outcomes) else 0,
                    "spearman_rho": spearman,
                    **effect,
                }
            )
    return output


def matched_model_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for cohort_name, cohort_test in (
        ("answer_mention", lambda row: True),
        ("full_correct", lambda row: row.get("full_correct") is True),
    ):
        cohort = [row for row in rows if cohort_test(row)]
        output.append(
            {
                "dataset": cohort[0]["dataset"] if cohort else "",
                "cohort": cohort_name,
                "facts": len(cohort),
                "co_lmlm_del_off_accuracy": _mean(
                    row["co_lmlm_del_off_correct"] for row in cohort
                ),
                "standard_lm_accuracy": _mean(
                    row["standard_lm_correct"]
                    for row in cohort
                    if row["standard_lm_correct"] is not None
                ),
                "smollm2_accuracy": _mean(
                    row["smollm2_correct"]
                    for row in cohort
                    if row["smollm2_correct"] is not None
                ),
            }
        )
    return output


def _acronym(text: str) -> str:
    tokens = _WORD_PATTERN.findall(text)
    return "".join(token[0] for token in tokens if token).casefold()


def _variant_signals(value: str, fact: AuditFact) -> tuple[float, str]:
    value_norm = normalize_text(value)
    answers = sorted(fact.answer_norms)
    similarity = max(
        (SequenceMatcher(None, value_norm, answer).ratio() for answer in answers),
        default=0.0,
    )
    compact_value = re.sub(r"[^a-z0-9]", "", value.casefold())
    if compact_value and any(
        compact_value == _acronym(answer) or _acronym(value) == re.sub(
            r"[^a-z0-9]", "", answer.casefold()
        )
        for answer in (fact.ground_truth, *fact.aliases)
    ):
        return similarity, "acronym-or-abbreviation"
    if similarity >= 0.86:
        return similarity, "high-string-similarity"
    value_tokens = set(_WORD_PATTERN.findall(value_norm))
    answer_tokens = set().union(
        *[set(_WORD_PATTERN.findall(answer)) for answer in answers]
    ) if answers else set()
    if value_tokens and answer_tokens and (
        value_tokens <= answer_tokens or answer_tokens <= value_tokens
    ):
        return similarity, "token-subset"
    return similarity, "needs-semantic-review"


def oracle_miss_candidates(
    dataset: str,
    facts: Mapping[str, AuditFact],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    full_correct = [fact for fact in facts.values() if fact.full_correct]
    survivors = [fact for fact in full_correct if fact.del_on_correct]
    candidates = [
        fact
        for fact in full_correct
        if fact.del_on_correct and fact.del_off_correct is False
    ]
    rows: list[dict[str, Any]] = []
    for fact in sorted(candidates, key=lambda item: item.fact_id):
        selected = fact.del_on_selected
        value = str(selected.get("value") or "")
        similarity, signal = _variant_signals(value, fact)
        rows.append(
            {
                "dataset": dataset,
                "fact_id": fact.fact_id,
                "prompt": fact.prompt,
                "ground_truth": fact.ground_truth,
                "aliases": " | ".join(fact.aliases),
                "del_on_output": fact.del_on_output,
                "del_off_output": fact.del_off_output,
                "retrieved_value": value,
                "retrieved_entry_id": selected.get("entry_id"),
                "retrieved_source_id": selected.get("source_id"),
                "retrieved_score": selected.get("score"),
                "lexical_similarity": similarity,
                "automatic_review_signal": signal,
                "review_label": "",
                "review_notes": "",
            }
        )
    summary = {
        "dataset": dataset,
        "full_correct_facts": len(full_correct),
        "post_deletion_survivors": len(survivors),
        "retrieval_mediated_candidates": len(candidates),
        "candidate_rate_given_full": _safe_rate(len(candidates), len(full_correct)),
        "candidate_share_of_survivors": _safe_rate(len(candidates), len(survivors)),
        "high_string_similarity_or_abbreviation": sum(
            row["automatic_review_signal"]
            in {"high-string-similarity", "acronym-or-abbreviation"}
            for row in rows
        ),
    }
    return rows, summary


def _hashed_prompt_vector(text: str, dim: int) -> np.ndarray:
    tokens = _WORD_PATTERN.findall(normalize_text(text))
    features = [*tokens, *(f"{left}_{right}" for left, right in zip(tokens, tokens[1:]))]
    vector = np.zeros(dim, dtype=np.float32)
    for feature in features:
        digest = hashlib.md5(feature.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[index] += sign
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


def hashed_prompt_samples(
    facts: Mapping[str, AuditFact], dim: int
) -> list[ProbeSample]:
    return [
        ProbeSample(
            sample_id=f"prompt-bow:{key}",
            fact=key,
            vector=_hashed_prompt_vector(fact.prompt, dim),
        )
        for key, fact in sorted(facts.items())
    ]


class MeanTokenEmbedder:
    """Reusable frozen embedding lookup for the prompt representation control."""

    def __init__(
        self,
        *,
        model_path: str,
        device: str,
        trust_remote_code: bool,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=trust_remote_code
            )
        except AttributeError:
            # The released Co-LMLM tokenizer_config.json represents
            # ``extra_special_tokens`` as a list, while Transformers 4.49
            # expects a mapping. The actual added-token vocabulary remains.
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=trust_remote_code,
                extra_special_tokens={},
            )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=(
                torch.bfloat16 if device.startswith("cuda") else torch.float32
            ),
        )
        model.to(device)
        model.eval()
        self.model_path = model_path
        self.device = device
        self.torch = torch
        self.tokenizer = tokenizer
        self.model = model
        self.embedding = model.get_input_embeddings()

    def encode(
        self,
        facts: Mapping[str, AuditFact],
        *,
        batch_size: int,
    ) -> list[ProbeSample]:
        items = sorted(facts.items())
        samples: list[ProbeSample] = []
        with self.torch.no_grad():
            for start in range(0, len(items), batch_size):
                batch = items[start : start + batch_size]
                encoded = self.tokenizer(
                    [fact.prompt for _, fact in batch],
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                input_ids = encoded["input_ids"].to(self.device)
                mask = encoded["attention_mask"].to(self.device).unsqueeze(-1)
                vectors = self.embedding(input_ids)
                means = (vectors * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
                means = self.torch.nn.functional.normalize(
                    means.float(), dim=1
                ).cpu().numpy()
                for (key, _), vector in zip(batch, means, strict=True):
                    samples.append(
                        ProbeSample(
                            sample_id=f"mean-token:{key}",
                            fact=key,
                            vector=np.asarray(vector, dtype=np.float32),
                        )
                    )
        return samples


def mean_token_embedding_samples(
    facts: Mapping[str, AuditFact],
    *,
    model_path: str,
    batch_size: int,
    device: str,
    trust_remote_code: bool,
) -> list[ProbeSample]:
    """One-shot public wrapper for the mean input-token control."""
    embedder = MeanTokenEmbedder(
        model_path=model_path,
        device=device,
        trust_remote_code=trust_remote_code,
    )
    return embedder.encode(facts, batch_size=batch_size)


def _probe_labels(facts: Mapping[str, AuditFact]) -> dict[str, dict[str, Any]]:
    return {
        key: {
            "ground_truth": fact.ground_truth,
            "aliases": fact.aliases,
            "subject": fact.subject,
            "relation": fact.relation,
            "prompt": fact.prompt,
        }
        for key, fact in facts.items()
    }


def _run_control_probe(
    *,
    representation: str,
    split: str,
    samples: list[ProbeSample],
    facts: Mapping[str, AuditFact],
    config: ProbeConfig,
    fold_groups: Mapping[str, str] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    report = run_probe(
        samples,
        _probe_labels(facts),
        config,
        fold_groups=fold_groups,
    )
    behavioral = {
        key: float(fact.del_off_correct)
        for key, fact in facts.items()
        if fact.del_off_correct is not None
    }
    delta = compute_delta_rep(report.per_fact, behavioral)
    summary = {
        "dataset": next(iter(facts.values())).dataset,
        "representation": representation,
        "split": split,
        **report.summary,
        **delta,
    }
    rows = [
        {
            "dataset": summary["dataset"],
            "representation": representation,
            "split": split,
            **row,
            "behavioral_l": behavioral.get(row["fact"]),
        }
        for row in report.per_fact
    ]
    return summary, rows


def _discover_query_embeddings(dataset_dir: Path, stem: str) -> Path | None:
    candidates = (
        dataset_dir / f"{stem}_query_embeddings.npz",
        dataset_dir / f"{stem}_full" / "full_query_embeddings.npz",
    )
    return next((path for path in candidates if path.is_file()), None)


def _preserve_review_labels(path: Path, rows: list[dict[str, Any]]) -> None:
    if not path.is_file():
        _write_csv(path, rows)
        return
    existing: dict[tuple[str, str], dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            existing[(row.get("dataset", ""), row.get("fact_id", ""))] = {
                "review_label": row.get("review_label", ""),
                "review_notes": row.get("review_notes", ""),
            }
    for row in rows:
        row.update(existing.get((row["dataset"], row["fact_id"]), {}))
    _write_csv(path, rows)


def _format_rate(value: Any) -> str:
    return "n/a" if value is None else f"{100 * float(value):.2f}%"


def _write_markdown_summary(
    path: Path,
    diagnostics: Sequence[Mapping[str, Any]],
    models: Sequence[Mapping[str, Any]],
    frequency_associations: Sequence[Mapping[str, Any]],
    oracle: Sequence[Mapping[str, Any]],
    probe_controls: Sequence[Mapping[str, Any]],
) -> None:
    lines = [
        "# Follow-up analyses",
        "",
        "Generated from the completed audit artifacts. The stored frequency "
        "variable is the number of answer-bearing candidates in the audit's "
        "top-K retrieval envelope; it is a local density proxy, not a global "
        "fact-frequency count.",
        "",
        "## Probe/dataset diagnostics",
        "",
        "| Dataset | Facts | Answers | Canonical duplicates | Legacy-split groups | Global prior | Relation prior |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in diagnostics:
        lines.append(
            f"| {row['dataset']} | {row['facts']} | {row['candidate_answers']} | "
            f"{row['facts_in_duplicate_groups']} | "
            f"{row['duplicate_groups_split_by_legacy_folds']} | "
            f"{_format_rate(row['global_majority_accuracy'])} | "
            f"{_format_rate(row['relation_majority_accuracy'])} |"
        )
    lines.extend(
        [
            "",
            "## Matched parametric answerability",
            "",
            "| Dataset | Cohort | Facts | Co-LMLM DEL-OFF | STANDARD-360M-FW | SmolLM2-360M |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in models:
        lines.append(
            f"| {row['dataset']} | {row['cohort']} | {row['facts']} | "
            f"{_format_rate(row['co_lmlm_del_off_accuracy'])} | "
            f"{_format_rate(row['standard_lm_accuracy'])} | "
            f"{_format_rate(row['smollm2_accuracy'])} |"
        )
    lines.extend(
        [
            "",
            "## Frequency-density association",
            "",
            "Odds ratios are per doubling of `(top-K answer-bearing neighbors + 1)`.",
            "",
            "| Dataset | Cohort | Model | Spearman rho | Odds ratio | 95% CI |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in frequency_associations:
        if row["cohort"] != "answer_mention":
            continue
        odds = row["odds_ratio_per_doubling"]
        interval = (
            "n/a"
            if odds is None
            else f"{row['odds_ratio_ci_low']:.2f}–{row['odds_ratio_ci_high']:.2f}"
        )
        rho = row["spearman_rho"]
        lines.append(
            f"| {row['dataset']} | {row['cohort']} | {row['model']} | "
            f"{('n/a' if rho is None else f'{rho:.3f}')} | "
            f"{('n/a' if odds is None else f'{odds:.2f}')} | {interval} |"
        )
    lines.extend(
        [
            "",
            "## Value-filter residual candidates",
            "",
            "These are an upper bound for paraphrase/alias misses and require "
            "the manual labels in `oracle_miss_adjudication.csv`.",
            "",
            "| Dataset | Full-correct | Candidates | Rate | Share of survivors | Lexical/abbrev signal |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in oracle:
        lines.append(
            f"| {row['dataset']} | {row['full_correct_facts']} | "
            f"{row['retrieval_mediated_candidates']} | "
            f"{_format_rate(row['candidate_rate_given_full'])} | "
            f"{_format_rate(row['candidate_share_of_survivors'])} | "
            f"{row['high_string_similarity_or_abbreviation']} |"
        )
    if probe_controls:
        lines.extend(
            [
                "",
                "## Linear probe controls",
                "",
                "| Dataset | Representation | Split | Accuracy | Behavioral L | Delta |",
                "|---|---|---|---:|---:|---:|",
            ]
        )
        for row in probe_controls:
            lines.append(
                f"| {row['dataset']} | {row['representation']} | {row['split']} | "
                f"{_format_rate(row['l_rep_hat'])} | {_format_rate(row['l_hat'])} | "
                f"{_format_rate(row['delta_rep'])} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_followups(args: argparse.Namespace) -> dict[str, Any]:
    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    selected = list(DATASET_STEMS) if args.datasets == "all" else [
        value.strip() for value in args.datasets.split(",") if value.strip()
    ]
    unknown = [value for value in selected if value not in DATASET_STEMS]
    if unknown:
        raise ValueError(f"Unknown datasets {unknown}; choose from {list(DATASET_STEMS)}.")

    diagnostics_rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []
    matched_rows: list[dict[str, Any]] = []
    frequency_rows: list[dict[str, Any]] = []
    frequency_association_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    oracle_summary_rows: list[dict[str, Any]] = []
    probe_summary_rows: list[dict[str, Any]] = []
    probe_per_fact_rows: list[dict[str, Any]] = []
    skipped_probe_controls: list[str] = []
    mean_embedder = (
        MeanTokenEmbedder(
            model_path=args.embedding_model,
            device=args.embedding_device,
            trust_remote_code=args.trust_remote_code,
        )
        if args.embedding_model
        else None
    )

    for dataset in selected:
        stem = DATASET_STEMS[dataset]
        dataset_dir = results_root / "co-lmlm" / dataset
        audit_path = dataset_dir / f"{stem}_results.jsonl"
        if not audit_path.is_file():
            raise FileNotFoundError(f"Missing Co-LMLM results: {audit_path}")
        facts = load_audit_facts(audit_path, dataset)
        diagnostics, assignments = probe_distribution_diagnostics(
            dataset, facts, folds=args.folds, seed=args.seed
        )
        diagnostics_rows.append(diagnostics)
        assignment_rows.extend(assignments)

        standard_path = (
            results_root / "standard-lm-360m-fw" / dataset / f"{stem}_results.jsonl"
        )
        smollm_path = (
            results_root / "smollm2-360m" / dataset / f"{stem}_results.jsonl"
        )
        standard = load_full_correctness(standard_path) if standard_path.is_file() else {}
        smollm2 = load_full_correctness(smollm_path) if smollm_path.is_file() else {}
        joined = matched_frequency_rows(dataset, facts, standard, smollm2)
        matched_rows.extend(joined)
        frequency_rows.extend(frequency_bin_summary(joined))
        frequency_association_rows.extend(frequency_association_summary(joined))
        model_rows.extend(matched_model_summary(joined))

        value_path = dataset_dir / "policy_matrix" / "value" / f"{stem}_results.jsonl"
        if value_path.is_file():
            value_facts = load_audit_facts(value_path, dataset)
            candidates, summary = oracle_miss_candidates(dataset, value_facts)
            oracle_rows.extend(candidates)
            oracle_summary_rows.append(summary)

        proposition_groups = {
            key: fact.proposition_group for key, fact in facts.items()
        }
        probe_config = ProbeConfig(
            folds=args.folds,
            seed=args.seed,
            ridge_lambda=args.ridge_lambda,
            feature_dim=args.answer_feature_dim,
        )
        if not args.skip_bow_probe:
            summary, rows = _run_control_probe(
                representation=f"hashed-prompt-{args.prompt_feature_dim}",
                split="canonical-proposition",
                samples=hashed_prompt_samples(facts, args.prompt_feature_dim),
                facts=facts,
                config=probe_config,
                fold_groups=proposition_groups,
            )
            probe_summary_rows.append(summary)
            probe_per_fact_rows.extend(rows)

        query_path = _discover_query_embeddings(dataset_dir, stem)
        if query_path is not None:
            query_samples = load_probe_samples([query_path], state="FULL")
            for split, groups in (
                ("legacy-fact-id", None),
                ("canonical-proposition", proposition_groups),
            ):
                summary, rows = _run_control_probe(
                    representation="co-lmlm-query",
                    split=split,
                    samples=query_samples,
                    facts=facts,
                    config=probe_config,
                    fold_groups=groups,
                )
                probe_summary_rows.append(summary)
                probe_per_fact_rows.extend(rows)
        else:
            skipped_probe_controls.append(
                f"{dataset}: query-embedding sidecar not found"
            )

        if mean_embedder is not None:
            mean_samples = mean_embedder.encode(
                facts, batch_size=args.embedding_batch_size
            )
            summary, rows = _run_control_probe(
                representation=f"mean-input-token:{args.embedding_model}",
                split="canonical-proposition",
                samples=mean_samples,
                facts=facts,
                config=probe_config,
                fold_groups=proposition_groups,
            )
            probe_summary_rows.append(summary)
            probe_per_fact_rows.extend(rows)

    _write_csv(output_dir / "probe_dataset_diagnostics.csv", diagnostics_rows)
    _write_csv(output_dir / "probe_group_assignments.csv", assignment_rows)
    _write_csv(output_dir / "matched_memorization_per_fact.csv", matched_rows)
    _write_csv(output_dir / "frequency_bins.csv", frequency_rows)
    _write_csv(
        output_dir / "frequency_associations.csv", frequency_association_rows
    )
    _write_csv(output_dir / "matched_model_summary.csv", model_rows)
    _write_csv(output_dir / "oracle_miss_summary.csv", oracle_summary_rows)
    _preserve_review_labels(output_dir / "oracle_miss_adjudication.csv", oracle_rows)
    _write_csv(output_dir / "probe_controls_summary.csv", probe_summary_rows)
    _write_csv(output_dir / "probe_controls_per_fact.csv", probe_per_fact_rows)
    _write_markdown_summary(
        output_dir / "README.md",
        diagnostics_rows,
        model_rows,
        frequency_association_rows,
        oracle_summary_rows,
        probe_summary_rows,
    )
    manifest = {
        "results_root": str(results_root.resolve()),
        "datasets": selected,
        "query_probe_skips": skipped_probe_controls,
        "mean_token_embedding_model": args.embedding_model,
        "frequency_warning": (
            "answer_bearing_neighbors_top_k is query-local answer density, "
            "not a global proposition-frequency count"
        ),
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "datasets": len(selected),
        "facts": len(matched_rows),
        "oracle_candidates": len(oracle_rows),
        "probe_controls": len(probe_summary_rows),
        "query_probe_skips": skipped_probe_controls,
        "output_dir": output_dir,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run probe controls, frequency/memorization analysis, and "
            "answer-filter diagnostics over completed HALO outputs."
        )
    )
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("reports/followup_analyses")
    )
    parser.add_argument(
        "--datasets",
        default="all",
        help="Comma-separated dataset names or 'all'.",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ridge-lambda", type=float, default=1.0)
    parser.add_argument("--answer-feature-dim", type=int, default=2048)
    parser.add_argument("--prompt-feature-dim", type=int, default=256)
    parser.add_argument(
        "--skip-bow-probe",
        action="store_true",
        help="Skip the no-model hashed prompt lexical baseline.",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help=(
            "Optional Hugging Face/local Co-LMLM checkpoint used for the "
            "mean frozen input-token-embedding baseline."
        ),
    )
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args(argv)
    if args.folds < 2:
        parser.error("--folds must be at least 2")
    if args.prompt_feature_dim < 8:
        parser.error("--prompt-feature-dim must be at least 8")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    summary = run_followups(parse_args(argv))
    print(
        f"Analyzed {summary['facts']} facts across {summary['datasets']} datasets; "
        f"wrote {summary['oracle_candidates']} oracle-miss candidates and "
        f"{summary['probe_controls']} probe-control summaries to "
        f"{summary['output_dir']}."
    )
    for warning in summary["query_probe_skips"]:
        print(f"warning: {warning}")


if __name__ == "__main__":
    main()
