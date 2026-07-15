import re
from collections import Counter
from typing import Any

from equivalence import normalize_text, values_equivalent


TOKEN_PATTERN = re.compile(r"\d+\.\d+|\w+(?:[-']\w+)*", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(str(text).casefold())


def normalize_answer(text: str) -> str:
    return normalize_text(text)


def exact_match(
    prediction: str,
    ground_truth: str,
    ground_truth_aliases: list[str] | tuple[str, ...] | None = None,
) -> float:
    return float(
        values_equivalent(
            prediction,
            ground_truth,
            right_aliases=ground_truth_aliases,
        )
    )


def contains_match(
    prediction: str,
    ground_truth: str,
    ground_truth_aliases: list[str] | tuple[str, ...] | None = None,
) -> float:
    if exact_match(prediction, ground_truth, ground_truth_aliases=ground_truth_aliases):
        return 1.0

    normalized_prediction = normalize_answer(prediction)
    candidate_truths = [ground_truth, *(ground_truth_aliases or ())]

    if not normalized_prediction or not candidate_truths:
        return 0.0

    for candidate_truth in candidate_truths:
        normalized_ground_truth = normalize_answer(candidate_truth)
        if not normalized_ground_truth:
            continue
        if (
            normalized_ground_truth in normalized_prediction
            or normalized_prediction in normalized_ground_truth
        ):
            return 1.0

    return 0.0


def is_unknown(prediction: str) -> float:
    normalized_prediction = normalize_answer(prediction)
    unknown_values = {
        "",
        "unknown",
        "n a",
        "na",
        "none",
        "no answer",
        "i don't know",
        "i do not know",
        "i don t know",
    }
    return float(normalized_prediction in unknown_values)


def precision_recall_f1(
    prediction: str,
    ground_truth: str,
    ground_truth_aliases: list[str] | tuple[str, ...] | None = None,
) -> dict[str, float]:
    if exact_match(prediction, ground_truth, ground_truth_aliases=ground_truth_aliases):
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    pred_tokens = tokenize(prediction)
    gold_tokens = tokenize(ground_truth)

    if not pred_tokens and not gold_tokens:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    if not pred_tokens or not gold_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)
    overlap = sum((pred_counter & gold_counter).values())

    if overlap == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)

    return {"precision": precision, "recall": recall, "f1": f1}


def score_prediction(
    prediction: str,
    ground_truth: str,
    ground_truth_aliases: list[str] | tuple[str, ...] | None = None,
) -> dict[str, float]:
    overlap_scores = precision_recall_f1(
        prediction,
        ground_truth,
        ground_truth_aliases=ground_truth_aliases,
    )
    return {
        "exact_match": exact_match(
            prediction,
            ground_truth,
            ground_truth_aliases=ground_truth_aliases,
        ),
        "contains_match": contains_match(
            prediction,
            ground_truth,
            ground_truth_aliases=ground_truth_aliases,
        ),
        "unknown": is_unknown(prediction),
        **overlap_scores,
    }


def count(results: list[dict[str, Any]]) -> int:
    return len(results)


def _average_metric(
    results: list[dict[str, Any]],
    metric_name: str,
) -> float:
    if not results:
        return 0.0

    total = 0.0
    for result in results:
        scores = score_prediction(
            result["model_output"],
            result["ground_truth"],
            ground_truth_aliases=result.get("object_aliases"),
        )
        total += scores[metric_name]
    return total / len(results)


def exact_match_rate(results: list[dict[str, Any]]) -> float:
    return _average_metric(results, "exact_match")


def contains_match_rate(results: list[dict[str, Any]]) -> float:
    return _average_metric(results, "contains_match")


def unknown_rate(results: list[dict[str, Any]]) -> float:
    return _average_metric(results, "unknown")


def precision_rate(results: list[dict[str, Any]]) -> float:
    return _average_metric(results, "precision")


def recall_rate(results: list[dict[str, Any]]) -> float:
    return _average_metric(results, "recall")


def f1_rate(results: list[dict[str, Any]]) -> float:
    return _average_metric(results, "f1")


def _result_group_key(result: dict[str, Any]) -> tuple[Any, str, str]:
    return (
        result.get("fact_id"),
        result.get("prompt", ""),
        result.get("ground_truth", ""),
    )


def _group_results_by_fact(
    results: list[dict[str, Any]],
) -> dict[tuple[Any, str, str], dict[str, dict[str, Any]]]:
    grouped: dict[tuple[Any, str, str], dict[str, dict[str, Any]]] = {}
    for result in results:
        group_key = _result_group_key(result)
        grouped.setdefault(group_key, {})[result["state"]] = result
    return grouped


def _eligible_state_groups(
    results: list[dict[str, Any]],
) -> list[dict[str, dict[str, Any]]]:
    grouped_results = _group_results_by_fact(results)
    return [
        state_results
        for state_results in grouped_results.values()
        if "DEL-ON" in state_results and "DEL-OFF" in state_results
    ]


def paired_count(results: list[dict[str, Any]]) -> int:
    return len(_eligible_state_groups(results))


def parametric_leakage(results: list[dict[str, Any]]) -> float:
    eligible_groups = _eligible_state_groups(results)
    if not eligible_groups:
        return 0.0

    leakage_total = 0.0
    for state_results in eligible_groups:
        del_off_result = state_results["DEL-OFF"]
        leakage_total += exact_match(
            del_off_result["model_output"],
            del_off_result["ground_truth"],
            ground_truth_aliases=del_off_result.get("object_aliases"),
        )

    return leakage_total / len(eligible_groups)


def retrieval_mediated_correctness(results: list[dict[str, Any]]) -> float:
    eligible_groups = _eligible_state_groups(results)
    if not eligible_groups:
        return 0.0

    retrieval_total = 0.0
    for state_results in eligible_groups:
        del_on_result = state_results["DEL-ON"]
        del_off_result = state_results["DEL-OFF"]

        del_on_correct = exact_match(
            del_on_result["model_output"],
            del_on_result["ground_truth"],
            ground_truth_aliases=del_on_result.get("object_aliases"),
        )
        del_off_correct = exact_match(
            del_off_result["model_output"],
            del_off_result["ground_truth"],
            ground_truth_aliases=del_off_result.get("object_aliases"),
        )

        retrieval_total += float(del_on_correct == 1.0 and del_off_correct == 0.0)

    return retrieval_total / len(eligible_groups)


def trace_has_gold_equivalent(result: dict[str, Any]) -> bool:
    retrieval_trace = result.get("retrieval_trace") or {}
    retained_candidates = retrieval_trace.get("retained_candidates") or []

    for candidate in retained_candidates:
        if candidate.get("supports_target_fact") is True:
            return True

        subject_matches = values_equivalent(
            candidate.get("subject", ""),
            result.get("subject", ""),
            right_aliases=result.get("subject_aliases"),
        )
        relation_matches = values_equivalent(
            candidate.get("relation", ""),
            result.get("relation", ""),
            right_aliases=result.get("relation_aliases"),
        )
        object_matches = values_equivalent(
            candidate.get("object", ""),
            result["ground_truth"],
            right_aliases=result.get("object_aliases"),
        )
        if subject_matches and relation_matches and object_matches:
            return True

    return False


def retrieval_artifact_rate(results: list[dict[str, Any]]) -> float:
    eligible_groups = _eligible_state_groups(results)
    if not eligible_groups:
        return 0.0

    artifact_total = 0.0
    for state_results in eligible_groups:
        del_on_result = state_results["DEL-ON"]
        del_on_correct = exact_match(
            del_on_result["model_output"],
            del_on_result["ground_truth"],
            ground_truth_aliases=del_on_result.get("object_aliases"),
        )
        artifact_total += float(
            del_on_correct == 1.0 and not trace_has_gold_equivalent(del_on_result)
        )

    return artifact_total / len(eligible_groups)


def metrics_total(results: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "count": count(results),
        "exact_match": exact_match_rate(results),
        "contains_match": contains_match_rate(results),
        "unknown_rate": unknown_rate(results),
        "precision": precision_rate(results),
        "recall": recall_rate(results),
        "f1": f1_rate(results),
        "paired_count": paired_count(results),
        "parametric_leakage": parametric_leakage(results),
        "retrieval_mediated_correctness": retrieval_mediated_correctness(results),
        "retrieval_artifact_rate": retrieval_artifact_rate(results),
    }
