import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from halo.core.examples import AuditExample
from halo.factq_embed import (
    first_query_embedding,
    plan_embeddings,
    question_example,
)
from halo.interventions.factq import (
    FACTQ_QUESTIONS_FIELD,
    build_generator_prompt,
    dedupe_questions,
    factq_vector_key,
    load_factq_vectors,
    mark_fact_span,
    parse_generator_output,
    prompt_row_key,
    save_factq_vectors,
)


def test_mark_fact_span_wraps_the_answer_mention() -> None:
    marked = mark_fact_span("Marie Curie was born in Warsaw in 1867.", "Warsaw")
    assert marked == (
        "Marie Curie was born in <FACT>1<FACT_ID>Warsaw</FACT> in 1867."
    )


def test_mark_fact_span_is_case_insensitive_but_preserves_document_casing() -> None:
    marked = mark_fact_span("The capital is PARIS today.", "Paris")
    assert marked == "The capital is <FACT>1<FACT_ID>PARIS</FACT> today."


def test_mark_fact_span_prefers_the_longest_matching_candidate() -> None:
    # "New York City" contains "New York"; the longer alias must win so the
    # generator sees the full mention as the fact span.
    marked = mark_fact_span(
        "She moved to New York City in 1998.",
        "New York",
        aliases=("New York City",),
    )
    assert marked == (
        "She moved to <FACT>1<FACT_ID>New York City</FACT> in 1998."
    )


def test_mark_fact_span_falls_back_to_aliases() -> None:
    marked = mark_fact_span(
        "The Big Apple never sleeps.", "New York", aliases=("Big Apple",)
    )
    assert marked == "The <FACT>1<FACT_ID>Big Apple</FACT> never sleeps."


def test_mark_fact_span_returns_none_when_no_candidate_occurs() -> None:
    assert mark_fact_span("Nothing relevant here.", "Paris") is None
    assert mark_fact_span("Nothing relevant here.", "", aliases=("",)) is None


def test_mark_fact_span_marks_only_the_first_occurrence() -> None:
    marked = mark_fact_span("Paris is Paris.", "Paris")
    assert marked == "<FACT>1<FACT_ID>Paris</FACT> is Paris."


def test_mark_fact_span_treats_regex_metacharacters_literally() -> None:
    marked = mark_fact_span("Version 1.5 (beta) shipped.", "1.5 (beta)")
    assert marked == "Version <FACT>1<FACT_ID>1.5 (beta)</FACT> shipped."


def test_build_generator_prompt_matches_the_model_card_format() -> None:
    prompt = build_generator_prompt("Doc with <FACT>1<FACT_ID>span</FACT>.")
    assert prompt == (
        "Doc with <FACT>1<FACT_ID>span</FACT>.<DOC_SEP>\n\n"
        "What are the question and paraphrased answer for <FACT>1<FACT_ID>?\n"
    )


def test_parse_generator_output_extracts_question_and_answer() -> None:
    assert parse_generator_output(
        "<QUESTION>Where was Marie Curie born?</QUESTION>"
        "<ANSWER>Warsaw</ANSWER><|im_end|>"
    ) == ("Where was Marie Curie born?", "Warsaw")


def test_parse_generator_output_tolerates_whitespace_and_newlines() -> None:
    assert parse_generator_output(
        "<QUESTION>\nWhere?\n</QUESTION>\n<ANSWER>\nThere\n</ANSWER>"
    ) == ("Where?", "There")


def test_parse_generator_output_rejects_malformed_completions() -> None:
    assert parse_generator_output("no tags at all") is None
    assert parse_generator_output("<QUESTION>dangling question") is None
    assert (
        parse_generator_output("<QUESTION></QUESTION><ANSWER>orphan</ANSWER>")
        is None
    )


def test_dedupe_questions_folds_case_and_preserves_order() -> None:
    assert dedupe_questions(
        [("Where?", "A"), ("where?", "B"), ("When?", "C")]
    ) == [("Where?", "A"), ("When?", "C")]


def test_prompt_row_key_prefers_prompt_id_then_fact_id() -> None:
    assert prompt_row_key({"prompt_id": "p1", "fact_id": "f1"}) == "p1"
    assert prompt_row_key({"fact_id": "f1"}) == "f1"
    assert prompt_row_key({}) == ""


def test_factq_vector_key_rejects_slashes_in_the_fact_key() -> None:
    with pytest.raises(ValueError, match="collides"):
        factq_vector_key("a/b", 0)


def test_vectors_round_trip_preserves_question_order_and_gaps(tmp_path) -> None:
    path = tmp_path / "vectors.npz"
    save_factq_vectors(
        path,
        {
            # Question 1 issued no search: an index gap, not a shift.
            "fact-a": {0: [1.0, 0.0], 2: [0.0, 1.0]},
            "fact-b": {0: [0.5, 0.5]},
        },
    )
    loaded = load_factq_vectors(path)

    assert set(loaded) == {"fact-a", "fact-b"}
    np.testing.assert_array_equal(loaded["fact-a"][0], [1.0, 0.0])
    np.testing.assert_array_equal(loaded["fact-a"][1], [0.0, 1.0])
    assert len(loaded["fact-a"]) == 2
    assert loaded["fact-a"][0].dtype == np.float32


def test_load_factq_vectors_rejects_foreign_key_layouts(tmp_path) -> None:
    path = tmp_path / "other.npz"
    np.savez_compressed(path, **{"fact/FULL/event0": np.zeros(2)})
    with pytest.raises(ValueError, match="Unexpected key"):
        load_factq_vectors(path)


def test_enriched_row_fields_survive_prompt_row_parsing() -> None:
    example = AuditExample.from_prompt_row(
        {
            "prompt_id": "p1",
            "prompt_text": "What is the capital of France?",
            "gold_object": "Paris",
            FACTQ_QUESTIONS_FIELD: ["Which city is France's capital?"],
        }
    )
    assert example.source_row[FACTQ_QUESTIONS_FIELD] == [
        "Which city is France's capital?"
    ]


def test_first_query_embedding_takes_the_earliest_captured_event() -> None:
    result = {
        "_query_embeddings": [
            {"event_index": 2, "vector": [2.0]},
            {"event_index": 0, "vector": None},
            {"event_index": 1, "vector": [1.0]},
        ]
    }
    assert first_query_embedding(result) == [1.0]
    assert first_query_embedding({"_query_embeddings": []}) is None
    assert first_query_embedding({}) is None


def _record(key: str, index: int, question: str, captured: bool) -> dict:
    return {
        "key": key,
        "question_index": index,
        "question": question,
        "captured": captured,
    }


def test_plan_embeddings_reuses_matching_prior_work() -> None:
    questions = [("f1", 0, "Where?"), ("f1", 1, "When?"), ("f2", 0, "Who?")]
    prior_vectors = {"f1": {0: [1.0]}}
    prior_records = {
        ("f1", 0): _record("f1", 0, "Where?", captured=True),
        # A recorded no-search miss is itself the result: no re-embedding.
        ("f1", 1): _record("f1", 1, "When?", captured=False),
    }

    vectors, records, todo = plan_embeddings(
        questions, prior_vectors, prior_records
    )

    assert vectors == {"f1": {0: [1.0]}}
    assert set(records) == {("f1", 0), ("f1", 1)}
    assert todo == [("f2", 0, "Who?")]


def test_plan_embeddings_invalidates_stale_or_incomplete_priors() -> None:
    questions = [("f1", 0, "Where exactly?"), ("f1", 1, "When?")]
    prior_records = {
        # The questions file was regenerated: text changed, vector is stale.
        ("f1", 0): _record("f1", 0, "Where?", captured=True),
        # Log says captured but the vector is missing from the store.
        ("f1", 1): _record("f1", 1, "When?", captured=True),
    }

    vectors, records, todo = plan_embeddings(
        questions, {"f1": {0: [1.0]}}, prior_records
    )

    assert vectors == {}
    assert records == {}
    assert todo == [("f1", 0, "Where exactly?"), ("f1", 1, "When?")]


def test_finalize_merges_complete_shards_without_a_backend(tmp_path) -> None:
    """A finalize run whose shard artifacts already cover every question
    must merge them into the canonical outputs and delete the partials —
    without ever loading the model (missing index would fail otherwise)."""
    import json

    from halo.factq_embed import main as embed_main

    questions_path = tmp_path / "prompts_factq.jsonl"
    rows = [
        {"prompt_id": "f1", "prompt_text": "p1", "gold_object": "a1",
         FACTQ_QUESTIONS_FIELD: ["Where?", "When?"]},
        {"prompt_id": "f2", "prompt_text": "p2", "gold_object": "a2",
         FACTQ_QUESTIONS_FIELD: ["Who?"]},
    ]
    questions_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    output_path = tmp_path / "prompts_factq_vectors.npz"

    # Shard 0 owns f1 (row 0), shard 1 owns f2 (row 1); "When?" was a
    # recorded no-search miss.
    save_factq_vectors(
        tmp_path / "prompts_factq_vectors.shard0-of-2.npz",
        {"f1": {0: [1.0, 0.0]}},
    )
    (tmp_path / "prompts_factq_vectors.shard0-of-2_embed_log.jsonl").write_text(
        json.dumps(_record("f1", 0, "Where?", captured=True)) + "\n"
        + json.dumps(_record("f1", 1, "When?", captured=False)) + "\n",
        encoding="utf-8",
    )
    save_factq_vectors(
        tmp_path / "prompts_factq_vectors.shard1-of-2.npz",
        {"f2": {0: [0.0, 1.0]}},
    )
    (tmp_path / "prompts_factq_vectors.shard1-of-2_embed_log.jsonl").write_text(
        json.dumps(_record("f2", 0, "Who?", captured=True)) + "\n",
        encoding="utf-8",
    )

    embed_main(
        ["--questions", str(questions_path),
         "--index-path", str(tmp_path / "no-such-index"),
         "--output", str(output_path)]
    )

    merged = load_factq_vectors(output_path)
    assert set(merged) == {"f1", "f2"}
    np.testing.assert_array_equal(merged["f1"][0], [1.0, 0.0])
    log_lines = [
        json.loads(line)
        for line in (tmp_path / "prompts_factq_vectors_embed_log.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert {(line["key"], line["question_index"]) for line in log_lines} == {
        ("f1", 0), ("f1", 1), ("f2", 0)
    }
    assert not list(tmp_path.glob("*.shard*"))


def test_question_example_swaps_the_prompt_and_keeps_the_gold() -> None:
    example = question_example(
        {
            "prompt_id": "p1",
            "prompt_text": "What is the capital of France?",
            "gold_object": "Paris",
        },
        "Which city is France's capital?",
    )
    assert example.prompt == "Which city is France's capital?"
    assert example.ground_truth == "Paris"
    assert example.prompt_id == "p1"
