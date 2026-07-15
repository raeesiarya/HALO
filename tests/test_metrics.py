import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lmlm_audit.core.metrics import (
    _average_metric,
    _eligible_state_groups,
    _group_results_by_fact,
    contains_match,
    contains_match_rate,
    count,
    exact_match,
    exact_match_rate,
    f1_rate,
    full_correct_paired_count,
    is_unknown,
    metrics_total,
    normalize_answer,
    paired_count,
    parametric_leakage,
    parametric_leakage_given_full,
    post_deletion_survival_given_full,
    precision_rate,
    precision_recall_f1,
    recall_rate,
    retrieval_artifact_rate,
    retrieval_artifact_eligible_count,
    retrieval_artifact_full_eligible_count,
    retrieval_artifact_rate_given_full,
    retrieval_mediated_correctness,
    retrieval_mediated_correctness_given_full,
    score_prediction,
    trace_has_gold_equivalent,
    trace_is_complete,
    unknown_rate,
)



def _result(
    fact_id=1,
    prompt_id=None,
    prompt="Q?",
    ground_truth="Answer",
    model_output="Answer",
    state="FULL",
    object_aliases=None,
    subject="Subject",
    subject_aliases=None,
    relation="Relation",
    relation_aliases=None,
    retrieval_trace=None,
):
    return {
        "fact_id": fact_id,
        "prompt_id": prompt_id,
        "prompt": prompt,
        "ground_truth": ground_truth,
        "model_output": model_output,
        "state": state,
        "object_aliases": object_aliases or [],
        "subject": subject,
        "subject_aliases": subject_aliases or [],
        "relation": relation,
        "relation_aliases": relation_aliases or [],
        "retrieval_trace": retrieval_trace,
    }


def _del_on_off_pair(
    fact_id=1,
    prompt="Q?",
    ground_truth="GT",
    del_on_output="GT",
    del_off_output="GT",
    object_aliases=None,
    subject="S",
    relation="R",
    del_on_trace=None,
):
    return [
        _result(
            fact_id=fact_id,
            prompt=prompt,
            ground_truth=ground_truth,
            model_output=del_on_output,
            state="DEL-ON",
            object_aliases=object_aliases,
            subject=subject,
            relation=relation,
            retrieval_trace=del_on_trace,
        ),
        _result(
            fact_id=fact_id,
            prompt=prompt,
            ground_truth=ground_truth,
            model_output=del_off_output,
            state="DEL-OFF",
            object_aliases=object_aliases,
            subject=subject,
            relation=relation,
        ),
    ]



def test_normalize_answer() -> None:
    assert normalize_answer("Spice Girls!") == "spice girls"
    assert normalize_answer("$69.7 million") == "69.7 million"


def test_normalize_answer_empty():
    assert normalize_answer("") == ""


def test_normalize_answer_punctuation_only():
    assert normalize_answer("!!!") == ""


def test_normalize_answer_uppercase():
    assert normalize_answer("NASA") == "nasa"


def test_normalize_answer_unicode():
    result = normalize_answer("Jørgensen")
    assert result == "jørgensen"


def test_normalize_answer_idempotent():
    once = normalize_answer("Hello, World!")
    twice = normalize_answer(once)
    assert once == twice



class TestExactMatch:
    def test_identical(self):
        assert exact_match("Paris", "Paris") == 1.0

    def test_case_insensitive(self):
        assert exact_match("paris", "Paris") == 1.0

    def test_different(self):
        assert exact_match("Paris", "Berlin") == 0.0

    def test_via_alias(self):
        assert exact_match("UK", "United Kingdom", ground_truth_aliases=["UK"]) == 1.0

    def test_empty_both(self):
        assert exact_match("", "") == 0.0

    def test_left_empty(self):
        assert exact_match("", "Paris") == 0.0

    def test_right_empty(self):
        assert exact_match("Paris", "") == 0.0

    def test_partial_word_overlap_not_exact(self):
        assert exact_match("Spice", "Spice Girls") == 0.0

    def test_with_punctuation(self):
        assert exact_match("Spice Girls!", "Spice Girls") == 1.0

    def test_alias_none(self):
        assert exact_match("Paris", "Paris", ground_truth_aliases=None) == 1.0



def test_contains_match() -> None:
    assert contains_match("Spice Girls, a girl group", "Spice Girls") == 1.0
    assert contains_match("Dutch", "Jørgensen") == 0.0


class TestContainsMatch:
    def test_exact_implies_contains(self):
        assert contains_match("Paris", "Paris") == 1.0

    def test_prediction_contains_ground_truth(self):
        assert contains_match("She was born in Paris, France", "Paris") == 1.0

    def test_ground_truth_contains_prediction(self):
        assert contains_match("Girls", "Spice Girls") == 1.0

    def test_no_overlap(self):
        assert contains_match("Berlin", "Paris") == 0.0

    def test_alias_provides_contains(self):
        assert contains_match("UK", "United Kingdom", ground_truth_aliases=["UK"]) == 1.0

    def test_empty_prediction(self):
        assert contains_match("", "Paris") == 0.0

    def test_empty_ground_truth(self):
        assert contains_match("", "") == 0.0

    def test_both_nonempty_no_substring(self):
        assert contains_match("cat", "dog") == 0.0

    def test_case_insensitive_contains(self):
        assert contains_match("spice girls are cool", "Spice Girls") == 1.0



def test_is_unknown() -> None:
    assert is_unknown("unknown") == 1.0
    assert is_unknown("") == 1.0
    assert is_unknown("I don't know") == 1.0
    assert is_unknown("Spice Girls") == 0.0


class TestIsUnknown:
    @pytest.mark.parametrize(
        "text",
        [
            "",
            "unknown",
            "Unknown",
            "UNKNOWN",
            "n/a",
            "N/A",
            "na",
            "NA",
            "none",
            "None",
            "no answer",
            "No Answer",
            "i don't know",
            "I don't know",
            "I do not know",
            "i do not know",
            "I don t know",
        ],
    )
    def test_is_unknown_truthy(self, text):
        assert is_unknown(text) == 1.0, f"Expected is_unknown({text!r}) == 1.0"

    @pytest.mark.parametrize(
        "text",
        [
            "Spice Girls",
            "Paris",
            "42",
            "yes",
            "no",
            "maybe",
            "Richard Mthetwa",
        ],
    )
    def test_is_unknown_falsy(self, text):
        assert is_unknown(text) == 0.0, f"Expected is_unknown({text!r}) == 0.0"



class TestPrecisionRecallF1:
    def test_exact_match_gives_perfect_scores(self):
        result = precision_recall_f1("Richard Mthetwa", "Richard Mthetwa")
        assert result == {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    def test_both_empty(self):
        result = precision_recall_f1("", "")
        assert result == {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    def test_empty_prediction(self):
        result = precision_recall_f1("", "Paris")
        assert result == {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    def test_empty_ground_truth(self):
        result = precision_recall_f1("Paris", "")
        assert result == {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    def test_no_token_overlap(self):
        result = precision_recall_f1("cat", "dog")
        assert result == {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    def test_partial_overlap(self):
        result = precision_recall_f1(
            "Sanskrit scholars, poets, musicians",
            "family of musicians",
        )
        assert result["exact_match"] if "exact_match" in result else True
        assert result["precision"] == pytest.approx(0.25, abs=1e-6)
        assert result["recall"] == pytest.approx(1 / 3, abs=1e-6)
        assert result["f1"] == pytest.approx(0.286, abs=1e-3)

    def test_f1_harmonic_mean(self):
        result = precision_recall_f1("Paris", "Paris France")
        assert result["precision"] == pytest.approx(1.0)
        assert result["recall"] == pytest.approx(0.5)
        assert result["f1"] == pytest.approx(2 / 3, abs=1e-6)

    def test_alias_match_returns_perfect(self):
        result = precision_recall_f1("Jorgensen", "Jørgensen", ["Jorgensen"])
        assert result == {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    def test_repeated_tokens(self):
        result = precision_recall_f1("cat cat", "cat")
        assert result["precision"] == pytest.approx(0.5)
        assert result["recall"] == pytest.approx(1.0)
        assert result["f1"] == pytest.approx(2 / 3, abs=1e-6)



def test_score_prediction_exact_match() -> None:
    scores = score_prediction("Richard Mthetwa", "Richard Mthetwa")
    assert scores["exact_match"] == 1.0
    assert scores["contains_match"] == 1.0
    assert scores["unknown"] == 0.0
    assert scores["precision"] == 1.0
    assert scores["recall"] == 1.0
    assert scores["f1"] == 1.0


def test_score_prediction_alias_match() -> None:
    scores = score_prediction(
        "Jorgensen",
        "Jørgensen",
        ground_truth_aliases=["Jorgensen"],
    )
    assert scores["exact_match"] == 1.0
    assert scores["contains_match"] == 1.0
    assert scores["precision"] == 1.0
    assert scores["recall"] == 1.0
    assert scores["f1"] == 1.0


def test_score_prediction_partial_overlap() -> None:
    scores = score_prediction(
        "Sanskrit scholars, poets, musicians",
        "family of musicians",
    )
    assert scores["exact_match"] == 0.0
    assert scores["contains_match"] == 0.0
    assert scores["unknown"] == 0.0
    assert scores["precision"] == 0.25
    assert scores["recall"] == 1 / 3
    assert round(scores["f1"], 3) == 0.286


class TestScorePrediction:
    def test_unknown_prediction(self):
        scores = score_prediction("unknown", "Paris")
        assert scores["unknown"] == 1.0
        assert scores["exact_match"] == 0.0

    def test_empty_prediction(self):
        scores = score_prediction("", "Paris")
        assert scores["unknown"] == 1.0

    def test_all_keys_present(self):
        scores = score_prediction("x", "y")
        assert set(scores.keys()) == {"exact_match", "contains_match", "unknown", "precision", "recall", "f1"}

    def test_contains_but_not_exact(self):
        scores = score_prediction("the Spice Girls band", "Spice Girls")
        assert scores["exact_match"] == 0.0
        assert scores["contains_match"] == 1.0



def test_count_empty():
    assert count([]) == 0


def test_count_nonempty():
    assert count([_result(), _result()]) == 2



class TestAverageMetric:
    def test_empty_list(self):
        assert _average_metric([], "exact_match") == 0.0

    def test_all_correct(self):
        results = [_result(model_output="Answer", ground_truth="Answer") for _ in range(3)]
        assert _average_metric(results, "exact_match") == 1.0

    def test_none_correct(self):
        results = [_result(model_output="Wrong", ground_truth="Answer") for _ in range(3)]
        assert _average_metric(results, "exact_match") == 0.0

    def test_half_correct(self):
        results = [
            _result(model_output="Answer", ground_truth="Answer"),
            _result(model_output="Wrong", ground_truth="Answer"),
        ]
        assert _average_metric(results, "exact_match") == 0.5



def test_metrics_total() -> None:
    summary = metrics_total(
        [
            {
                "ground_truth": "Spice Girls",
                "model_output": "Spice Girls",
                "state": "FULL",
            },
            {
                "ground_truth": "Bihar, India",
                "model_output": "1956",
                "state": "FULL",
            },
        ]
    )

    assert summary["count"] == 2
    assert summary["exact_match"] == 0.5
    assert summary["contains_match"] == 0.5
    assert summary["unknown_rate"] == 0.0
    assert summary["precision"] == 0.5
    assert summary["recall"] == 0.5
    assert summary["f1"] == 0.5
    assert summary["retrieval_artifact_rate"] == 0.0


def test_unknown_rate() -> None:
    value = unknown_rate(
        [
            {
                "ground_truth": "Spice Girls",
                "model_output": "unknown",
                "state": "FULL",
            },
            {
                "ground_truth": "Bihar, India",
                "model_output": "",
                "state": "FULL",
            },
        ]
    )

    assert value == 1.0


def test_rate_helpers() -> None:
    results = [
        {
            "ground_truth": "Spice Girls",
            "model_output": "Spice Girls",
            "state": "FULL",
        },
        {
            "ground_truth": "Bihar, India",
            "model_output": "1956",
            "state": "FULL",
        },
    ]

    assert exact_match_rate(results) == 0.5
    assert precision_rate(results) == 0.5
    assert recall_rate(results) == 0.5
    assert f1_rate(results) == 0.5


class TestRateHelpers:
    def test_empty_list(self):
        assert exact_match_rate([]) == 0.0
        assert contains_match_rate([]) == 0.0
        assert unknown_rate([]) == 0.0
        assert precision_rate([]) == 0.0
        assert recall_rate([]) == 0.0
        assert f1_rate([]) == 0.0

    def test_all_correct(self):
        results = [_result(model_output="Answer", ground_truth="Answer") for _ in range(5)]
        assert exact_match_rate(results) == 1.0
        assert contains_match_rate(results) == 1.0
        assert unknown_rate(results) == 0.0

    def test_all_unknown(self):
        results = [_result(model_output="unknown", ground_truth="Answer") for _ in range(5)]
        assert unknown_rate(results) == 1.0
        assert exact_match_rate(results) == 0.0

    def test_with_aliases(self):
        results = [
            _result(
                model_output="Jorgensen",
                ground_truth="Jørgensen",
                object_aliases=["Jorgensen"],
            )
        ]
        assert exact_match_rate(results) == 1.0

    def test_contains_but_not_exact(self):
        results = [
            _result(model_output="the Spice Girls group", ground_truth="Spice Girls")
        ]
        assert exact_match_rate(results) == 0.0
        assert contains_match_rate(results) == 1.0



class TestMetricsTotal:
    def test_empty_results(self):
        summary = metrics_total([])
        assert summary["count"] == 0
        assert summary["exact_match"] == 0.0
        assert summary["paired_count"] == 0
        assert summary["parametric_leakage"] == 0.0

    def test_all_keys_present(self):
        summary = metrics_total([_result()])
        expected_keys = {
            "count", "exact_match", "contains_match", "unknown_rate",
            "precision", "recall", "f1", "paired_count",
            "parametric_leakage", "retrieval_mediated_correctness",
            "retrieval_artifact_rate",
        }
        assert expected_keys <= set(summary.keys())

    def test_single_result(self):
        summary = metrics_total([_result(model_output="Answer", ground_truth="Answer")])
        assert summary["count"] == 1
        assert summary["exact_match"] == 1.0

    def test_all_unknown(self):
        results = [_result(model_output="unknown", ground_truth="Paris") for _ in range(4)]
        summary = metrics_total(results)
        assert summary["unknown_rate"] == 1.0
        assert summary["exact_match"] == 0.0



class TestGroupResultsByFact:
    def test_single_result(self):
        r = _result(fact_id=1, prompt="Q?", ground_truth="A", state="FULL")
        grouped = _group_results_by_fact([r])
        assert len(grouped) == 1

    def test_groups_by_fact_id_and_prompt(self):
        r1 = _result(fact_id=1, prompt="Q1?", ground_truth="A", state="DEL-ON")
        r2 = _result(fact_id=1, prompt="Q1?", ground_truth="A", state="DEL-OFF")
        r3 = _result(fact_id=2, prompt="Q2?", ground_truth="B", state="DEL-ON")
        grouped = _group_results_by_fact([r1, r2, r3])
        assert len(grouped) == 2

    def test_same_fact_different_state(self):
        r1 = _result(fact_id=1, prompt="Q?", ground_truth="A", state="DEL-ON")
        r2 = _result(fact_id=1, prompt="Q?", ground_truth="A", state="DEL-OFF")
        grouped = _group_results_by_fact([r1, r2])
        assert len(grouped) == 1
        group = next(iter(grouped.values()))
        assert "DEL-ON" in group
        assert "DEL-OFF" in group

    def test_same_prompt_text_with_different_prompt_ids_stays_separate(self):
        r1 = _result(prompt_id="p1", state="DEL-ON")
        r2 = _result(prompt_id="p2", state="DEL-OFF")

        assert len(_group_results_by_fact([r1, r2])) == 2

    def test_different_deletion_manifests_stay_separate(self):
        r1 = _result(state="DEL-ON")
        r2 = _result(state="DEL-OFF")
        r1["deletion_manifest"] = {"manifest_id": "first"}
        r2["deletion_manifest"] = {"manifest_id": "second"}

        assert len(_group_results_by_fact([r1, r2])) == 2

    def test_empty_list(self):
        assert _group_results_by_fact([]) == {}



class TestEligibleStateGroups:
    def test_requires_both_del_on_and_del_off(self):
        results = [_result(fact_id=1, prompt="Q?", ground_truth="A", state="DEL-ON")]
        assert _eligible_state_groups(results) == []

    def test_both_present(self):
        results = _del_on_off_pair()
        groups = _eligible_state_groups(results)
        assert len(groups) == 1

    def test_full_state_alone_not_eligible(self):
        results = [_result(fact_id=1, state="FULL")]
        assert _eligible_state_groups(results) == []

    def test_two_pairs(self):
        results = _del_on_off_pair(fact_id=1, prompt="Q1?", ground_truth="A")
        results += _del_on_off_pair(fact_id=2, prompt="Q2?", ground_truth="B")
        groups = _eligible_state_groups(results)
        assert len(groups) == 2

    def test_empty(self):
        assert _eligible_state_groups([]) == []



class TestPairedCount:
    def test_zero_no_pairs(self):
        assert paired_count([]) == 0

    def test_one_pair(self):
        assert paired_count(_del_on_off_pair()) == 1

    def test_two_pairs(self):
        r = _del_on_off_pair(fact_id=1, prompt="Q1?", ground_truth="A")
        r += _del_on_off_pair(fact_id=2, prompt="Q2?", ground_truth="B")
        assert paired_count(r) == 2

    def test_full_only_no_pairs(self):
        results = [_result(state="FULL")]
        assert paired_count(results) == 0


def test_full_conditioned_metrics_ignore_unknown_full_facts() -> None:
    known = [
        _result(fact_id=1, prompt="Q1?", ground_truth="A", state="FULL", model_output="A"),
        _result(
            fact_id=1,
            prompt="Q1?",
            ground_truth="A",
            state="DEL-ON",
            model_output="A",
            retrieval_trace={
                "trace_available": True,
                "trace_complete": True,
                "retained_candidates": [],
            },
        ),
        _result(fact_id=1, prompt="Q1?", ground_truth="A", state="DEL-OFF", model_output="wrong"),
    ]
    unknown = [
        _result(fact_id=2, prompt="Q2?", ground_truth="B", state="FULL", model_output="wrong"),
        _result(fact_id=2, prompt="Q2?", ground_truth="B", state="DEL-ON", model_output="B"),
        _result(fact_id=2, prompt="Q2?", ground_truth="B", state="DEL-OFF", model_output="B"),
    ]

    results = known + unknown
    assert paired_count(results) == 2
    assert full_correct_paired_count(results) == 1
    assert post_deletion_survival_given_full(results) == 1.0
    assert parametric_leakage_given_full(results) == 0.0
    assert retrieval_mediated_correctness_given_full(results) == 1.0
    assert retrieval_artifact_rate_given_full(results) == 1.0
    assert retrieval_artifact_full_eligible_count(results) == 1


def test_full_conditioned_metrics_require_all_three_states() -> None:
    results = [
        _result(state="FULL", model_output="Answer"),
        _result(state="DEL-ON", model_output="Answer"),
    ]

    assert full_correct_paired_count(results) == 0
    assert post_deletion_survival_given_full(results) == 0.0


def test_trace_supports_schema_free_semantic_judgment() -> None:
    result = _result(
        subject=None,
        relation=None,
        retrieval_trace={
            "retained_candidates": [
                {
                    "entry_id": "wiki:france:17",
                    "value": "Paris is France's capital city.",
                    "supports_target": True,
                }
            ]
        },
    )

    assert trace_has_gold_equivalent(result) is True



def test_cross_state_metrics() -> None:
    results = [
        {
            "fact_id": 1,
            "prompt": "What is Geri Halliwell famous for?",
            "subject": "Geri Halliwell",
            "subject_aliases": [],
            "relation": "Famous For",
            "relation_aliases": [],
            "ground_truth": "Spice Girls",
            "state": "DEL-ON",
            "model_output": "Spice Girls",
            "object_aliases": [],
            "retrieval_trace": {
                "trace_available": True,
                "trace_complete": True,
                "retained_candidates": [
                    {
                        "subject": "Geri Halliwell",
                        "relation": "Famous For",
                        "object": "Spice Girls",
                        "supports_target_fact": True,
                    },
                ]
            },
        },
        {
            "fact_id": 1,
            "prompt": "What is Geri Halliwell famous for?",
            "subject": "Geri Halliwell",
            "subject_aliases": [],
            "relation": "Famous For",
            "relation_aliases": [],
            "ground_truth": "Spice Girls",
            "state": "DEL-OFF",
            "model_output": "unknown",
            "object_aliases": [],
        },
        {
            "fact_id": 2,
            "prompt": "What is Nozinja's birth name?",
            "subject": "Nozinja",
            "subject_aliases": [],
            "relation": "Birth Name",
            "relation_aliases": [],
            "ground_truth": "Richard Mthetwa",
            "state": "DEL-ON",
            "model_output": "Richard Mthetwa",
            "object_aliases": [],
            "retrieval_trace": {
                "trace_available": True,
                "trace_complete": True,
                "retained_candidates": [
                    {
                        "subject": "Nozinja",
                        "relation": "Birth Place",
                        "object": "Richard Mthetwa",
                        "supports_target_fact": False,
                    },
                ]
            },
        },
        {
            "fact_id": 2,
            "prompt": "What is Nozinja's birth name?",
            "subject": "Nozinja",
            "subject_aliases": [],
            "relation": "Birth Name",
            "relation_aliases": [],
            "ground_truth": "Richard Mthetwa",
            "state": "DEL-OFF",
            "model_output": "Richard Mthetwa",
            "object_aliases": [],
        },
    ]

    assert paired_count(results) == 2
    assert parametric_leakage(results) == 0.5
    assert retrieval_mediated_correctness(results) == 0.5
    assert retrieval_artifact_rate(results) == 0.5


class TestParametricLeakage:
    def test_empty(self):
        assert parametric_leakage([]) == 0.0

    def test_no_pairs(self):
        assert parametric_leakage([_result(state="FULL")]) == 0.0

    def test_full_leakage(self):
        results = _del_on_off_pair(del_off_output="GT", ground_truth="GT")
        assert parametric_leakage(results) == 1.0

    def test_zero_leakage(self):
        results = _del_on_off_pair(del_off_output="WRONG", ground_truth="GT")
        assert parametric_leakage(results) == 0.0

    def test_half_leakage(self):
        r1 = _del_on_off_pair(fact_id=1, prompt="Q1?", ground_truth="GT", del_off_output="GT")
        r2 = _del_on_off_pair(fact_id=2, prompt="Q2?", ground_truth="GT", del_off_output="WRONG")
        assert parametric_leakage(r1 + r2) == 0.5



class TestRetrievalMediatedCorrectness:
    def test_empty(self):
        assert retrieval_mediated_correctness([]) == 0.0

    def test_del_on_correct_del_off_wrong(self):
        results = _del_on_off_pair(
            del_on_output="GT", del_off_output="WRONG", ground_truth="GT"
        )
        assert retrieval_mediated_correctness(results) == 1.0

    def test_both_correct(self):
        results = _del_on_off_pair(del_on_output="GT", del_off_output="GT", ground_truth="GT")
        assert retrieval_mediated_correctness(results) == 0.0

    def test_both_wrong(self):
        results = _del_on_off_pair(del_on_output="W", del_off_output="W", ground_truth="GT")
        assert retrieval_mediated_correctness(results) == 0.0

    def test_del_on_wrong_del_off_correct_not_counted(self):
        results = _del_on_off_pair(del_on_output="W", del_off_output="GT", ground_truth="GT")
        assert retrieval_mediated_correctness(results) == 0.0



def test_retrieval_artifact_rate() -> None:
    results = [
        {
            "fact_id": 1,
            "prompt": "What is Geri Halliwell famous for?",
            "subject": "Geri Halliwell",
            "subject_aliases": [],
            "relation": "Famous For",
            "relation_aliases": [],
            "ground_truth": "Spice Girls",
            "state": "DEL-ON",
            "model_output": "Spice Girls",
            "object_aliases": [],
            "retrieval_trace": {
                "trace_available": True,
                "trace_complete": True,
                "retained_candidates": [
                    {
                        "subject": "Geri Halliwell",
                        "relation": "Famous For",
                        "object": "girl group",
                        "supports_target_fact": False,
                    },
                ]
            },
        },
        {
            "fact_id": 1,
            "prompt": "What is Geri Halliwell famous for?",
            "subject": "Geri Halliwell",
            "subject_aliases": [],
            "relation": "Famous For",
            "relation_aliases": [],
            "ground_truth": "Spice Girls",
            "state": "DEL-OFF",
            "model_output": "unknown",
            "object_aliases": [],
        },
    ]

    assert trace_has_gold_equivalent(results[0]) is False
    assert retrieval_artifact_rate(results) == 1.0


def test_trace_has_gold_equivalent_requires_subject_and_relation_match() -> None:
    result = {
        "subject": "Vishnevka",
        "subject_aliases": [],
        "relation": "Rural Localities As Of 2012",
        "relation_aliases": [],
        "ground_truth": "1",
        "object_aliases": [],
        "retrieval_trace": {
            "retained_candidates": [
                {
                    "subject": "Novgorod Oblast",
                    "relation": "Vishnevka Rural Localities As Of 2012",
                    "object": "1",
                    "supports_target_fact": False,
                }
            ]
        },
    }

    assert trace_has_gold_equivalent(result) is False


class TestTraceHasGoldEquivalent:
    def _result_with_trace(self, candidates, subject="S", relation="R", ground_truth="GT"):
        return {
            "subject": subject,
            "subject_aliases": [],
            "relation": relation,
            "relation_aliases": [],
            "ground_truth": ground_truth,
            "object_aliases": [],
            "retrieval_trace": {"retained_candidates": candidates},
        }

    def test_supports_target_fact_true(self):
        result = self._result_with_trace(
            [{"subject": "S", "relation": "R", "object": "GT", "supports_target_fact": True}]
        )
        assert trace_has_gold_equivalent(result) is True

    def test_all_candidates_wrong(self):
        result = self._result_with_trace(
            [{"subject": "X", "relation": "Y", "object": "Z", "supports_target_fact": False}]
        )
        assert trace_has_gold_equivalent(result) is False

    def test_no_trace(self):
        result = {
            "subject": "S",
            "subject_aliases": [],
            "relation": "R",
            "relation_aliases": [],
            "ground_truth": "GT",
            "object_aliases": [],
            "retrieval_trace": None,
        }
        assert trace_has_gold_equivalent(result) is False

    def test_empty_retained_candidates(self):
        result = self._result_with_trace([])
        assert trace_has_gold_equivalent(result) is False

    def test_subject_mismatch_not_gold(self):
        result = self._result_with_trace(
            [{"subject": "WRONG", "relation": "R", "object": "GT", "supports_target_fact": False}],
            subject="S",
            relation="R",
            ground_truth="GT",
        )
        assert trace_has_gold_equivalent(result) is False

    def test_relation_mismatch_not_gold(self):
        result = self._result_with_trace(
            [{"subject": "S", "relation": "WRONG_REL", "object": "GT", "supports_target_fact": False}],
            subject="S",
            relation="R",
            ground_truth="GT",
        )
        assert trace_has_gold_equivalent(result) is False

    def test_fallback_triple_matching(self):
        result = self._result_with_trace(
            [{"subject": "S", "relation": "R", "object": "GT", "supports_target_fact": False}],
            subject="S",
            relation="R",
            ground_truth="GT",
        )
        assert trace_has_gold_equivalent(result) is True

    def test_multiple_candidates_one_matches(self):
        result = self._result_with_trace(
            [
                {"subject": "X", "relation": "Y", "object": "Z", "supports_target_fact": False},
                {"subject": "S", "relation": "R", "object": "GT", "supports_target_fact": False},
            ],
            subject="S",
            relation="R",
            ground_truth="GT",
        )
        assert trace_has_gold_equivalent(result) is True



class TestRetrievalArtifactRate:
    def test_empty(self):
        assert retrieval_artifact_rate([]) == 0.0

    def test_missing_trace_is_not_treated_as_negative_evidence(self):
        results = _del_on_off_pair(
            del_on_output="GT",
            del_off_output="WRONG",
            ground_truth="GT",
        )
        assert trace_is_complete(results[0]) is False
        assert retrieval_artifact_eligible_count(results) == 0
        assert retrieval_artifact_rate(results) == 0.0

    def test_complete_empty_trace_counts_as_no_gold_evidence(self):
        results = _del_on_off_pair(
            del_on_output="GT",
            del_off_output="WRONG",
            ground_truth="GT",
            del_on_trace={
                "trace_available": True,
                "trace_complete": True,
                "retained_candidates": [],
            },
        )
        assert retrieval_artifact_eligible_count(results) == 1
        assert retrieval_artifact_rate(results) == 1.0

    def test_gold_evidence_present_not_artifact(self):
        trace = {
            "trace_available": True,
            "trace_complete": True,
            "retained_candidates": [
                {"subject": "S", "relation": "R", "object": "GT", "supports_target_fact": True}
            ]
        }
        results = _del_on_off_pair(
            del_on_output="GT",
            del_off_output="WRONG",
            ground_truth="GT",
            subject="S",
            relation="R",
            del_on_trace=trace,
        )
        assert retrieval_artifact_rate(results) == 0.0

    def test_wrong_answer_no_artifact_even_without_evidence(self):
        results = _del_on_off_pair(del_on_output="WRONG", del_off_output="WRONG", ground_truth="GT")
        assert retrieval_artifact_rate(results) == 0.0



def test_score_metrics_logged_to_wandb(wandb_run):
    """Grouped bar chart of score_prediction metrics for various scenarios."""
    import matplotlib.pyplot as plt
    import numpy as np

    scenarios = {
        "exact": score_prediction("Paris", "Paris"),
        "alias": score_prediction("Jorgensen", "Jørgensen", ["Jorgensen"]),
        "partial": score_prediction("the Spice Girls", "Spice Girls"),
        "wrong": score_prediction("Berlin", "Paris"),
        "unknown": score_prediction("unknown", "Paris"),
    }
    metrics_names = ["exact_match", "contains_match", "precision", "recall", "f1"]
    x = np.arange(len(metrics_names))
    width = 0.15

    if wandb_run is not None:
        try:
            import wandb

            fig, ax = plt.subplots(figsize=(12, 5))
            for i, (label, scores) in enumerate(scenarios.items()):
                values = [scores[m] for m in metrics_names]
                ax.bar(x + i * width, values, width, label=label)
            ax.set_xticks(x + width * 2)
            ax.set_xticklabels(metrics_names)
            ax.set_ylim(0, 1.1)
            ax.set_ylabel("Score")
            ax.set_title("score_prediction metrics by scenario")
            ax.legend()
            plt.tight_layout()
            wandb_run.log({"metrics/score_comparison": wandb.Image(fig)})
            plt.close(fig)
        except Exception:
            pass

    assert scenarios["exact"]["exact_match"] == 1.0
    assert scenarios["wrong"]["exact_match"] == 0.0
    assert scenarios["unknown"]["unknown"] == 1.0


def test_cross_state_metrics_logged_to_wandb(wandb_run):
    """Bar chart of cross-state metrics over a synthetic result set."""
    import matplotlib.pyplot as plt

    pairs = [
        (True, False),
        (True, True),
        (False, False),
        (True, False),
    ]
    results = []
    for i, (on_correct, off_correct) in enumerate(pairs):
        results += _del_on_off_pair(
            fact_id=i + 1,
            prompt=f"Q{i}?",
            ground_truth="GT",
            del_on_output="GT" if on_correct else "WRONG",
            del_off_output="GT" if off_correct else "WRONG",
        )

    pl = parametric_leakage(results)
    rmc = retrieval_mediated_correctness(results)
    rar = retrieval_artifact_rate(results)

    if wandb_run is not None:
        try:
            import wandb

            fig, ax = plt.subplots()
            metrics_map = {
                "parametric_leakage": pl,
                "retrieval_mediated": rmc,
                "artifact_rate": rar,
            }
            ax.bar(
                metrics_map.keys(),
                metrics_map.values(),
                color=["tomato", "steelblue", "goldenrod"],
            )
            ax.set_ylim(0, 1.0)
            ax.set_ylabel("Rate")
            ax.set_title("Cross-state audit metrics")
            plt.tight_layout()
            wandb_run.log({"metrics/cross_state": wandb.Image(fig)})
            plt.close(fig)
        except Exception:
            pass

    assert 0.0 <= pl <= 1.0
    assert 0.0 <= rmc <= 1.0
    assert 0.0 <= rar <= 1.0


def test_precision_recall_scatter_logged_to_wandb(wandb_run):
    """Scatter plot of precision vs recall across token-overlap scenarios."""
    import matplotlib.pyplot as plt

    test_cases = [
        ("cat", "cat"),
        ("Paris France", "Paris"),
        ("the quick brown fox", "brown fox"),
        ("hello world foo", "hello"),
        ("cats and dogs", "dogs and cats"),
        ("nothing", "completely different"),
    ]
    precisions, recalls, f1s = [], [], []
    for pred, truth in test_cases:
        result = precision_recall_f1(pred, truth)
        precisions.append(result["precision"])
        recalls.append(result["recall"])
        f1s.append(result["f1"])

    if wandb_run is not None:
        try:
            import wandb

            fig, ax = plt.subplots()
            scatter = ax.scatter(recalls, precisions, c=f1s, cmap="RdYlGn", s=80, vmin=0, vmax=1)
            plt.colorbar(scatter, ax=ax, label="F1")
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.set_title("Precision vs Recall (colour = F1)")
            plt.tight_layout()
            wandb_run.log({"metrics/precision_recall_scatter": wandb.Image(fig)})
            plt.close(fig)
        except Exception:
            pass

    assert all(0.0 <= p <= 1.0 for p in precisions)
    assert all(0.0 <= r <= 1.0 for r in recalls)
