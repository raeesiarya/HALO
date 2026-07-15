"""
Comprehensive unit tests for src/lmlm-audit/run_audit.py.

All pure-Python helpers (no model weights, no GPU) are exercised with normal,
boundary, and edge-case inputs.  Functions that require a live model are only
tested through mocks.  Several tests log diagnostic plots to W&B.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/lmlm-audit"))

import run_audit
from run_audit import (
    AuditJob,
    AuditLogger,
    _default_retrieval_trace,
    choose_answer,
    clean_answer,
    compute_generation_budget,
    discover_all_audit_jobs,
    discover_custom_audit_jobs,
    discover_released_audit_jobs,
    extract_lookup_values,
    generate_answer,
    infer_prompt_paths_for_database,
    load_prompts,
    log_metrics_to_wandb,
    parse_args,
    prepare_prompt,
    resolve_audit_jobs,
    retrieve_lookup_value,
    run_audit as run_audit_fn,
    run_prompt_audit,
    save_results,
    setup_wandb,
    write_metrics_csvs,
)
from database_states import DatabaseState


# ===========================================================================
# clean_answer – original tests
# ===========================================================================


def test_clean_answer_strips_db_markup_and_html_tags() -> None:
    answer = clean_answer(
        "&lt;/poem&gt; <|db_entity|>Madhur Jaffrey<|db_relationship|>"
        "Award<|db_return|>Madison Sharma<|db_end|> Biography"
    )
    assert answer == "Biography"


def test_clean_answer_strips_standalone_db_special_tokens() -> None:
    answer = clean_answer('"<|db_entity|> Spice Girls <|db_return|>"')
    assert answer == "Spice Girls"


# ===========================================================================
# clean_answer – extended
# ===========================================================================


class TestCleanAnswer:
    def test_strips_leading_trailing_whitespace(self):
        assert clean_answer("  Paris  ") == "Paris"

    def test_strips_answer_prefix_single(self):
        assert clean_answer("Answer: Paris") == "Paris"

    def test_strips_answer_prefix_repeated(self):
        assert clean_answer("Answer: Answer: Paris") == "Paris"

    def test_strips_answer_prefix_case_insensitive(self):
        assert clean_answer("answer: Paris") == "Paris"

    def test_strips_the_answer_is_prefix(self):
        assert clean_answer("The answer is Paris") == "Paris"

    def test_strips_it_is_prefix(self):
        assert clean_answer("It is Paris") == "Paris"

    def test_strips_its_prefix(self):
        assert clean_answer("It's Paris") == "Paris"

    def test_stops_at_question_marker(self):
        result = clean_answer("Paris\nQuestion: What is the capital?")
        assert result == "Paris"

    def test_stops_at_context_marker(self):
        result = clean_answer("Paris\nContext: France is a country.")
        assert result == "Paris"

    def test_stops_at_double_newline(self):
        result = clean_answer("Paris\n\nSome extra text")
        assert result == "Paris"

    def test_stops_at_fact_marker(self):
        result = clean_answer("Paris\nFact: France is in Europe.")
        assert result == "Paris"

    def test_stops_at_answer_marker(self):
        result = clean_answer("Paris\nAnswer: Berlin")
        assert result == "Paris"

    def test_unescapes_html_entities(self):
        result = clean_answer("&amp; &lt; &gt;")
        assert "&" in result

    def test_removes_db_markup_span(self):
        result = clean_answer(
            "<|db_entity|>X<|db_relationship|>Y<|db_return|>Z<|db_end|>"
        )
        assert "<|db_entity|>" not in result
        assert "<|db_end|>" not in result

    def test_removes_standalone_db_token(self):
        result = clean_answer("Hello <|db_return|> World")
        assert "<|db_return|>" not in result

    def test_removes_html_tags(self):
        result = clean_answer("<b>Paris</b>")
        assert "<b>" not in result
        assert "Paris" in result

    def test_keeps_first_sentence(self):
        result = clean_answer("Paris. It is the capital of France.")
        assert result == "Paris"

    def test_strips_trailing_punctuation(self):
        result = clean_answer("Paris.")
        assert result == "Paris"

    def test_strips_trailing_quotes(self):
        result = clean_answer('"Paris"')
        assert result == "Paris"

    def test_empty_string(self):
        result = clean_answer("")
        assert result == ""

    def test_only_whitespace(self):
        result = clean_answer("   ")
        assert result == ""

    def test_only_db_markup(self):
        result = clean_answer(
            "<|db_entity|>X<|db_relationship|>Y<|db_return|>Z<|db_end|>"
        )
        # The markup is removed; only surrounding whitespace remains → empty or the value before it
        assert result.strip() == "" or result.isspace() or result == ""

    def test_collapses_internal_spaces(self):
        result = clean_answer("New    York")
        assert result == "New York"

    def test_strips_leading_comma(self):
        result = clean_answer(",Paris")
        assert result == "Paris"

    def test_strips_trailing_semicolon(self):
        result = clean_answer("Paris;")
        assert result == "Paris"

    def test_complex_combined(self):
        result = clean_answer(
            'Answer: The answer is "Paris"\nQuestion: What city?'
        )
        assert result == "Paris"

    def test_answer_prefix_mixed_case(self):
        assert clean_answer("ANSWER: Paris") == "Paris"

    def test_exclamation_sentence_split(self):
        # The regex splits AFTER the punctuation; ! is not in the final strip
        # chars, so it is preserved in the first segment.
        result = clean_answer("Paris! It is wonderful.")
        assert result == "Paris!"

    def test_question_sentence_split(self):
        result = clean_answer("Paris? Yes, Paris.")
        assert result == "Paris?"


# ===========================================================================
# extract_lookup_values
# ===========================================================================


class TestExtractLookupValues:
    TEMPLATE = (
        "<|db_entity|>{entity}<|db_relationship|>{rel}<|db_return|>{value}<|db_end|>"
    )

    def test_single_lookup(self):
        raw = self.TEMPLATE.format(entity="Hexol", rel="First Described By", value="Jorgensen")
        result = extract_lookup_values(raw)
        assert result == ["Jorgensen"]

    def test_multiple_distinct_lookups(self):
        raw = (
            self.TEMPLATE.format(entity="A", rel="R", value="X")
            + self.TEMPLATE.format(entity="B", rel="S", value="Y")
        )
        result = extract_lookup_values(raw)
        assert "X" in result
        assert "Y" in result
        assert len(result) == 2

    def test_deduplicates_repeated_value(self):
        raw = (
            self.TEMPLATE.format(entity="A", rel="R", value="X")
            + self.TEMPLATE.format(entity="A", rel="R", value="X")
        )
        result = extract_lookup_values(raw)
        assert result.count("X") == 1

    def test_no_lookup_returns_empty(self):
        assert extract_lookup_values("plain text") == []

    def test_empty_string(self):
        assert extract_lookup_values("") == []

    def test_value_cleaned_before_return(self):
        # value with answer prefix should be cleaned
        raw = self.TEMPLATE.format(entity="A", rel="R", value="Answer: Paris")
        result = extract_lookup_values(raw)
        assert result == ["Paris"]

    def test_empty_value_ignored(self):
        raw = self.TEMPLATE.format(entity="A", rel="R", value="")
        result = extract_lookup_values(raw)
        assert result == []

    def test_multiline_value(self):
        raw = self.TEMPLATE.format(entity="A", rel="R", value="Paris\nFrance")
        result = extract_lookup_values(raw)
        # clean_answer truncates at \n\n but a single \n is fine here
        assert len(result) >= 1


# ===========================================================================
# choose_answer
# ===========================================================================


class TestChooseAnswer:
    def test_lookup_value_preferred_for_fact_query(self):
        answer, source = choose_answer("What is the capital?", "Berlin", ["Paris"])
        assert answer == "Paris"
        assert source == "lookup_value"

    def test_lookup_value_preferred_for_fill_blank(self):
        answer, source = choose_answer("The capital is ____.", "Berlin", ["Paris"])
        assert answer == "Paris"
        assert source == "lookup_value"

    def test_processed_text_used_when_no_lookup(self):
        answer, source = choose_answer("The capital is Paris.", "Paris", [])
        assert answer == "Paris"
        assert source == "postprocessed_text"

    def test_lookup_value_used_as_fallback_for_non_fact_query(self):
        # non-question, no blank → lookup not preferred, but postprocessed_text is empty
        answer, source = choose_answer("Tell me about Paris.", "", ["Paris"])
        assert answer == "Paris"
        assert source == "lookup_value"

    def test_empty_when_nothing_available(self):
        answer, source = choose_answer("Some prompt.", "", [])
        assert answer == ""
        assert source == "empty"

    def test_question_mark_triggers_lookup_preference(self):
        answer, source = choose_answer("Where was she born?", "France", ["Paris"])
        assert answer == "Paris"

    def test_blank_triggers_lookup_preference(self):
        answer, source = choose_answer("She was born in ____", "France", ["Paris"])
        assert answer == "Paris"

    def test_first_lookup_value_used(self):
        answer, source = choose_answer("Q?", "", ["First", "Second"])
        assert answer == "First"

    def test_non_question_uses_postprocessed_text(self):
        answer, source = choose_answer("Describe Paris.", "The City of Light", ["Paris"])
        assert answer == "The City of Light"
        assert source == "postprocessed_text"


# ===========================================================================
# compute_generation_budget
# ===========================================================================


class TestComputeGenerationBudget:
    def _make_tokenizer(self, token_count: int):
        tok = MagicMock()
        tok.encode.return_value = list(range(token_count))
        return tok

    def test_minimum_is_32(self):
        tok = self._make_tokenizer(0)
        result = compute_generation_budget(tok, "", target_answer_tokens=0)
        assert result >= 32

    def test_includes_prompt_length(self):
        tok = self._make_tokenizer(100)
        result = compute_generation_budget(tok, "x" * 100, target_answer_tokens=12)
        # 100 + 12 + 16 = 128 > 32
        assert result == 128

    def test_includes_slack(self):
        tok = self._make_tokenizer(10)
        result = compute_generation_budget(tok, "short", target_answer_tokens=5)
        # 10 + 5 + 16 = 31 → max(32, 31) = 32
        assert result == 32

    def test_large_prompt(self):
        tok = self._make_tokenizer(1000)
        result = compute_generation_budget(tok, "x" * 1000, target_answer_tokens=12)
        assert result == 1028  # 1000 + 12 + 16

    def test_returns_int(self):
        tok = self._make_tokenizer(50)
        result = compute_generation_budget(tok, "prompt", target_answer_tokens=10)
        assert isinstance(result, int)


# ===========================================================================
# prepare_prompt
# ===========================================================================


class TestPreparePrompt:
    def test_strips_whitespace(self):
        assert prepare_prompt("  hello  ") == "hello"

    def test_no_change_needed(self):
        assert prepare_prompt("hello") == "hello"

    def test_empty(self):
        assert prepare_prompt("") == ""

    def test_newline_stripped(self):
        assert prepare_prompt("hello\n") == "hello"

    def test_internal_content_preserved(self):
        text = "What is the capital of France?"
        assert prepare_prompt(text) == text


# ===========================================================================
# retrieve_lookup_value
# ===========================================================================


class TestRetrieveLookupValue:
    def test_no_db_manager_returns_unknown(self):
        model = MagicMock()
        del model.db_manager  # AttributeError → getattr returns None
        model.db_manager = None
        result = retrieve_lookup_value(model, "some query")
        assert result == "unknown"

    def test_successful_retrieval(self):
        db = MagicMock()
        db.retrieve_from_database.return_value = "Paris"
        model = MagicMock()
        model.db_manager = db
        result = retrieve_lookup_value(model, "query")
        assert result == "Paris"

    def test_exception_with_top1_fallback(self):
        db = MagicMock()
        # First call raises, second (threshold=-1) succeeds
        db.retrieve_from_database.side_effect = [
            ValueError("no result"),
            "Paris",
        ]
        model = MagicMock()
        model.db_manager = db
        model.fallback_policy = "top1_anyway"
        result = retrieve_lookup_value(model, "query")
        assert result == "Paris"

    def test_exception_with_non_top1_policy_returns_unknown(self):
        db = MagicMock()
        db.retrieve_from_database.side_effect = ValueError("fail")
        model = MagicMock()
        model.db_manager = db
        model.fallback_policy = "raise"
        result = retrieve_lookup_value(model, "query")
        assert result == "unknown"

    def test_both_calls_fail_returns_unknown(self):
        db = MagicMock()
        db.retrieve_from_database.side_effect = ValueError("fail")
        model = MagicMock()
        model.db_manager = db
        model.fallback_policy = "top1_anyway"
        result = retrieve_lookup_value(model, "query")
        assert result == "unknown"


# ===========================================================================
# load_prompts
# ===========================================================================


class TestLoadPrompts:
    def test_loads_valid_jsonl(self, tmp_path):
        p = tmp_path / "prompts.jsonl"
        records = [{"id": 1, "text": "hello"}, {"id": 2, "text": "world"}]
        p.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
        result = load_prompts(p)
        assert len(result) == 2
        assert result[0]["id"] == 1
        assert result[1]["text"] == "world"

    def test_skips_blank_lines(self, tmp_path):
        p = tmp_path / "prompts.jsonl"
        p.write_text(
            '{"id": 1}\n\n{"id": 2}\n   \n{"id": 3}\n', encoding="utf-8"
        )
        result = load_prompts(p)
        assert len(result) == 3

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.write_text("", encoding="utf-8")
        result = load_prompts(p)
        assert result == []

    def test_single_record(self, tmp_path):
        p = tmp_path / "single.jsonl"
        p.write_text('{"fact_id": 42}\n', encoding="utf-8")
        result = load_prompts(p)
        assert len(result) == 1
        assert result[0]["fact_id"] == 42

    def test_unicode_content(self, tmp_path):
        p = tmp_path / "unicode.jsonl"
        p.write_text('{"text": "Jørgensen"}\n', encoding="utf-8")
        result = load_prompts(p)
        assert result[0]["text"] == "Jørgensen"


# ===========================================================================
# save_results
# ===========================================================================


class TestSaveResults:
    def test_creates_parent_directories(self, tmp_path):
        output = tmp_path / "a" / "b" / "results.jsonl"
        save_results([{"key": "val"}], output)
        assert output.exists()

    def test_saves_each_result_as_jsonl(self, tmp_path):
        output = tmp_path / "out.jsonl"
        results = [{"id": 1}, {"id": 2}]
        save_results(results, output)
        lines = output.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"id": 1}
        assert json.loads(lines[1]) == {"id": 2}

    def test_empty_results(self, tmp_path):
        output = tmp_path / "empty.jsonl"
        save_results([], output)
        assert output.read_text() == ""

    def test_unicode_preserved(self, tmp_path):
        output = tmp_path / "unicode.jsonl"
        save_results([{"text": "Jørgensen"}], output)
        data = json.loads(output.read_text(encoding="utf-8").strip())
        assert data["text"] == "Jørgensen"

    def test_overwrites_existing_file(self, tmp_path):
        output = tmp_path / "out.jsonl"
        output.write_text("old content\n", encoding="utf-8")
        save_results([{"new": True}], output)
        data = json.loads(output.read_text(encoding="utf-8").strip())
        assert data == {"new": True}


# ===========================================================================
# _default_retrieval_trace
# ===========================================================================


class TestDefaultRetrievalTrace:
    def test_full_state(self):
        trace = _default_retrieval_trace(DatabaseState.FULL)
        assert trace["state"] == "FULL"
        assert trace["retrieval_enabled"] is True
        assert trace["lookup_query"] is None
        assert trace["all_candidates"] == []
        assert trace["error"] is None

    def test_del_on_state(self):
        trace = _default_retrieval_trace(DatabaseState.DEL_ON)
        assert trace["state"] == "DEL-ON"
        assert trace["retrieval_enabled"] is True

    def test_del_off_state(self):
        trace = _default_retrieval_trace(DatabaseState.DEL_OFF)
        assert trace["state"] == "DEL-OFF"
        assert trace["retrieval_enabled"] is False

    def test_all_keys_present(self):
        trace = _default_retrieval_trace(DatabaseState.FULL)
        expected_keys = {
            "state", "retrieval_enabled", "lookup_query", "threshold",
            "all_candidates", "deleted_candidates", "retained_candidates",
            "selected_candidate", "selected_value", "error",
        }
        assert expected_keys <= set(trace.keys())

    def test_selected_value_none(self):
        trace = _default_retrieval_trace(DatabaseState.FULL)
        assert trace["selected_value"] is None


# ===========================================================================
# W&B visualisation tests
# ===========================================================================


def test_clean_answer_processing_logged_to_wandb(wandb_run):
    """Bar chart showing before/after clean_answer character counts."""
    import matplotlib.pyplot as plt

    test_inputs = [
        "Answer: Paris",
        "&lt;b&gt;Berlin&lt;/b&gt;",
        "<|db_entity|>X<|db_relationship|>Y<|db_return|>Value<|db_end|> Paris",
        "The answer is Rome.",
        '"Tokyo"',
        "London\nQuestion: follow-up?",
        "  Madrid  ",
        "Vienna; city of music.",
    ]
    before_lens = [len(t) for t in test_inputs]
    cleaned = [clean_answer(t) for t in test_inputs]
    after_lens = [len(c) for c in cleaned]

    if wandb_run is not None:
        try:
            import numpy as np
            import wandb

            x = np.arange(len(test_inputs))
            width = 0.4
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.bar(x - width / 2, before_lens, width, label="before", color="tomato")
            ax.bar(x + width / 2, after_lens, width, label="after", color="seagreen")
            ax.set_xticks(x)
            ax.set_xticklabels([t[:20] + "…" if len(t) > 20 else t for t in test_inputs], rotation=45, ha="right")
            ax.set_ylabel("Characters")
            ax.set_title("clean_answer: before vs after character count")
            ax.legend()
            plt.tight_layout()
            wandb_run.log({"run_audit/clean_answer_lengths": wandb.Image(fig)})
            plt.close(fig)
        except Exception:
            pass

    # Assertions: cleaned version should never be longer than original (after strip)
    for inp, out in zip(test_inputs, cleaned):
        assert len(out) <= len(inp.strip()) + 5  # small slack for unescape


def test_choose_answer_distribution_logged_to_wandb(wandb_run):
    """Pie chart of choose_answer source distribution over test cases."""
    import matplotlib.pyplot as plt
    from collections import Counter

    cases = [
        ("Q?", "Berlin", ["Paris"]),
        ("Q?", "", ["Paris"]),
        ("Statement.", "Paris", []),
        ("Statement.", "", []),
        ("Blank ____", "Berlin", ["Paris"]),
        ("Statement.", "", ["Paris"]),
    ]
    sources = [choose_answer(p, out, lv)[1] for p, out, lv in cases]
    counts = Counter(sources)

    if wandb_run is not None:
        try:
            import wandb

            fig, ax = plt.subplots()
            ax.pie(
                counts.values(),
                labels=counts.keys(),
                autopct="%1.0f%%",
                colors=["steelblue", "seagreen", "tomato"],
            )
            ax.set_title("choose_answer source distribution")
            plt.tight_layout()
            wandb_run.log({"run_audit/choose_answer_sources": wandb.Image(fig)})
            plt.close(fig)
        except Exception:
            pass

    assert set(sources) <= {"lookup_value", "postprocessed_text", "empty"}


# ===========================================================================
# generate_answer
# ===========================================================================


def _make_tokenizer_mock(token_count: int = 10):
    """A MagicMock tokenizer that plays well with generate_answer."""
    tok = MagicMock()
    tok.encode.return_value = list(range(token_count))
    tok.pad_token_id = 0
    tok.eos_token_id = 2
    tok.unk_token_id = 3
    # tokenizer(prompt, return_tensors="pt") returns an object whose .to() call
    # returns a dict-like with "input_ids" and "attention_mask". We want
    # inputs["input_ids"].shape[1] to be a real int.
    inputs = MagicMock()
    inputs["input_ids"].shape = (1, token_count)
    inputs["input_ids"].__getitem__.return_value = MagicMock()
    # inputs accessors
    def getitem(key):
        m = MagicMock()
        m.shape = (1, token_count)
        return m
    inputs.__getitem__.side_effect = getitem
    # tokenizer(...) returns something with .to(device) → inputs
    call_result = MagicMock()
    call_result.to.return_value = inputs
    tok.side_effect = lambda *a, **kw: call_result
    # convert_tokens_to_ids returns int
    tok.convert_tokens_to_ids.return_value = 42
    return tok


def _make_model_mock(raw_output: str = "Some answer text"):
    model = MagicMock()
    # parameters() must return a FRESH iterator each call (iter() is exhausted
    # after one next(), so returning the same iter object from side_effect
    # breaks after the first call).
    param = MagicMock()
    param.device = "cpu"
    model.parameters.side_effect = lambda: iter([param])
    model._decode_with_special_tokens.return_value = raw_output
    model.generate_with_lookup.return_value = raw_output
    model.post_process.return_value = raw_output
    return model


class TestGenerateAnswer:
    def test_no_dblookup_path_uses_generate_with_lookup(self):
        tok = _make_tokenizer_mock()
        model = _make_model_mock(raw_output="Paris is the capital.")
        result = generate_answer(
            model=model,
            tokenizer=tok,
            prompt_text="What is the capital of France?",
            enable_dblookup=False,
        )
        model.generate_with_lookup.assert_called_once()
        # model.generate for lookup path must NOT be called.
        model.generate.assert_not_called()
        assert isinstance(result, str)

    def test_dblookup_path_without_db_return_marker(self):
        """With dblookup enabled but raw_output has no <|db_return|>, go
        through post_process path just like the non-dblookup branch."""
        tok = _make_tokenizer_mock()
        model = _make_model_mock(raw_output="Paris.")
        result = generate_answer(
            model=model,
            tokenizer=tok,
            prompt_text="What is the capital of France?",
            enable_dblookup=True,
        )
        model.eval.assert_called()
        model.set_logits_bias.assert_called()
        model.generate.assert_called_once()
        assert result == "Paris"

    def test_dblookup_path_with_db_return_marker(self):
        """When raw_output contains <|db_return|>, the retrieved value is
        returned via clean_answer(retrieve_lookup_value(...))."""
        tok = _make_tokenizer_mock()
        raw = "prefix <|db_entity|>X<|db_relationship|>Y<|db_return|>Paris<|db_end|>"
        model = _make_model_mock(raw_output=raw)
        model.db_manager = MagicMock()
        model.db_manager.retrieve_from_database.return_value = "Paris"
        result = generate_answer(
            model=model,
            tokenizer=tok,
            prompt_text="What is the capital?",
            enable_dblookup=True,
        )
        assert result == "Paris"

    def test_prompt_whitespace_stripped(self):
        tok = _make_tokenizer_mock()
        model = _make_model_mock(raw_output="Answer")
        generate_answer(
            model=model,
            tokenizer=tok,
            prompt_text="   What is the capital?   ",
            enable_dblookup=False,
        )
        # Prepared prompt passed to generate_with_lookup should be stripped.
        call_kwargs = model.generate_with_lookup.call_args.kwargs
        assert call_kwargs["prompt"] == "What is the capital?"


# ===========================================================================
# run_prompt_audit
# ===========================================================================


class TestRunPromptAudit:
    def _prompt_row(self):
        return {
            "fact_id": 1,
            "subject": "Paris",
            "relation": "capital_of",
            "gold_object": "France",
            "prompt_text": "Paris is the capital of ____.",
        }

    def test_populates_metadata_fields(self, fake_base_manager):
        tok = _make_tokenizer_mock()
        model = _make_model_mock(raw_output="France")
        with patch.object(run_audit, "build_state_db_manager") as b:
            db_manager = MagicMock()
            db_manager.last_trace = None
            b.return_value = db_manager
            row = self._prompt_row()
            result = run_prompt_audit(
                base_db_manager=fake_base_manager,
                model=model,
                tokenizer=tok,
                prompt_row=row,
                state=DatabaseState.FULL,
            )
        assert result["fact_id"] == 1
        assert result["subject"] == "Paris"
        assert result["relation"] == "capital_of"
        assert result["ground_truth"] == "France"
        assert result["state"] == "FULL"
        assert result["prompt"] == "Paris is the capital of ____."
        assert "model_output" in result

    def test_uses_default_trace_when_db_manager_has_none(self, fake_base_manager):
        tok = _make_tokenizer_mock()
        model = _make_model_mock(raw_output="France")
        with patch.object(run_audit, "build_state_db_manager") as b:
            db_manager = MagicMock(spec=[])  # no last_trace attribute
            b.return_value = db_manager
            result = run_prompt_audit(
                base_db_manager=fake_base_manager,
                model=model,
                tokenizer=tok,
                prompt_row=self._prompt_row(),
                state=DatabaseState.DEL_OFF,
            )
        trace = result["retrieval_trace"]
        assert trace["state"] == "DEL-OFF"
        assert trace["retrieval_enabled"] is False

    def test_resets_trace_when_supported(self, fake_base_manager):
        tok = _make_tokenizer_mock()
        model = _make_model_mock(raw_output="France")
        with patch.object(run_audit, "build_state_db_manager") as b:
            db_manager = MagicMock()
            db_manager.last_trace = {"selected_value": "France"}
            b.return_value = db_manager
            run_prompt_audit(
                base_db_manager=fake_base_manager,
                model=model,
                tokenizer=tok,
                prompt_row=self._prompt_row(),
                state=DatabaseState.FULL,
            )
        db_manager.reset_trace.assert_called_once()


# ===========================================================================
# run_audit (loop)
# ===========================================================================


class TestRunAuditLoop:
    def _write_prompt_jsonl(self, tmp_path, rows):
        p = tmp_path / "prompts.jsonl"
        p.write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
        )
        return p

    def _prompt_rows(self):
        return [
            {
                "fact_id": i,
                "subject": f"S{i}",
                "relation": "R",
                "gold_object": f"O{i}",
                "prompt_text": f"prompt {i}",
            }
            for i in range(3)
        ]

    def test_produces_one_result_per_prompt_per_state(
        self, tmp_path, fake_base_manager
    ):
        p = self._write_prompt_jsonl(tmp_path, self._prompt_rows())
        tok = _make_tokenizer_mock()
        model = _make_model_mock(raw_output="answer")
        with patch.object(run_audit, "build_state_db_manager") as b:
            b.return_value = MagicMock()
            results = run_audit_fn(
                prompt_path=p,
                base_db_manager=fake_base_manager,
                model=model,
                tokenizer=tok,
                states=[DatabaseState.FULL, DatabaseState.DEL_ON],
            )
        # 3 prompts × 2 states
        assert len(results) == 6

    def test_limit_caps_prompts(self, tmp_path, fake_base_manager):
        p = self._write_prompt_jsonl(tmp_path, self._prompt_rows())
        tok = _make_tokenizer_mock()
        model = _make_model_mock(raw_output="answer")
        with patch.object(run_audit, "build_state_db_manager") as b:
            b.return_value = MagicMock()
            results = run_audit_fn(
                prompt_path=p,
                base_db_manager=fake_base_manager,
                model=model,
                tokenizer=tok,
                states=[DatabaseState.FULL],
                limit=2,
            )
        assert len(results) == 2

    def test_empty_prompt_file_returns_empty(self, tmp_path, fake_base_manager):
        p = tmp_path / "empty.jsonl"
        p.write_text("", encoding="utf-8")
        tok = _make_tokenizer_mock()
        model = _make_model_mock()
        results = run_audit_fn(
            prompt_path=p,
            base_db_manager=fake_base_manager,
            model=model,
            tokenizer=tok,
            states=[DatabaseState.FULL],
        )
        assert results == []


# ===========================================================================
# write_metrics_csvs
# ===========================================================================


class TestWriteMetricsCsvs:
    def test_writes_both_csvs(self, tmp_path):
        cross = [{"prompt_file": "a.jsonl", "paired_count": 10}]
        per_state = [
            {"prompt_file": "a.jsonl", "state": "FULL", "count": 5},
            {"prompt_file": "a.jsonl", "state": "DEL-OFF", "count": 5},
        ]
        c_path = tmp_path / "c.csv"
        p_path = tmp_path / "p.csv"
        write_metrics_csvs(cross, per_state, c_path, p_path)

        assert c_path.exists()
        assert p_path.exists()

        with c_path.open() as f:
            rows = list(csv.DictReader(f))
        assert rows == [{"prompt_file": "a.jsonl", "paired_count": "10"}]

        with p_path.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert rows[0]["state"] == "FULL"

    def test_creates_parent_dirs(self, tmp_path):
        c_path = tmp_path / "nested" / "c.csv"
        p_path = tmp_path / "nested" / "p.csv"
        write_metrics_csvs(
            [{"a": 1}],
            [{"b": 2}],
            c_path,
            p_path,
        )
        assert c_path.exists()
        assert p_path.exists()

    def test_empty_rows_skip_file_creation(self, tmp_path):
        c_path = tmp_path / "c.csv"
        p_path = tmp_path / "p.csv"
        write_metrics_csvs([], [], c_path, p_path)
        # With empty inputs, no content is written, so the files don't exist.
        assert not c_path.exists()
        assert not p_path.exists()


# ===========================================================================
# AuditLogger
# ===========================================================================


class TestAuditLogger:
    def test_writes_messages_and_newline(self, tmp_path, capsys):
        log = tmp_path / "nested" / "run.log"
        logger = AuditLogger(log)
        logger.print("hello", "world")
        logger.close()

        assert log.exists()
        content = log.read_text(encoding="utf-8")
        assert content == "hello world\n"
        # Printed to stdout too.
        captured = capsys.readouterr()
        assert "hello world" in captured.out

    def test_custom_separator_and_end(self, tmp_path):
        log = tmp_path / "run.log"
        logger = AuditLogger(log)
        logger.print("a", "b", sep="|", end="!")
        logger.close()
        assert log.read_text() == "a|b!"

    def test_appends_across_multiple_prints(self, tmp_path):
        log = tmp_path / "run.log"
        logger = AuditLogger(log)
        logger.print("first")
        logger.print("second")
        logger.close()
        assert log.read_text() == "first\nsecond\n"

    def test_creates_parent_directories(self, tmp_path):
        log = tmp_path / "a" / "b" / "run.log"
        logger = AuditLogger(log)
        logger.close()
        assert log.parent.exists()


# ===========================================================================
# discover_custom_audit_jobs / infer_prompt_paths_for_database
# ===========================================================================


class TestDiscoverCustomAuditJobs:
    def _build_tree(self, root: Path, domains: dict):
        """Create data/custom_databases-style directory tree.

        domains = {
            "countries": {
                "variants": ["alias", "collision"],
                "prompts": {"alias": ["a.jsonl"], "collision": ["c.jsonl"]},
            }
        }
        """
        for domain_name, spec in domains.items():
            domain_dir = root / domain_name
            prompts_root = domain_dir / "prompts"
            prompts_root.mkdir(parents=True)
            for variant in spec.get("variants", []):
                # DB JSON at the domain level.
                (domain_dir / f"{variant}.json").write_text("{}", encoding="utf-8")
                # Variant prompts dir.
                vd = prompts_root / variant
                vd.mkdir()
                for prompt_file in spec["prompts"].get(variant, []):
                    (vd / prompt_file).write_text("", encoding="utf-8")

    def test_discovers_all_variants(self, tmp_path, monkeypatch):
        root = tmp_path / "custom_databases"
        self._build_tree(
            root,
            {
                "countries": {
                    "variants": ["alias", "collision"],
                    "prompts": {
                        "alias": ["p1.jsonl"],
                        "collision": ["p2.jsonl", "p3.jsonl"],
                    },
                },
            },
        )
        monkeypatch.setattr(run_audit, "DEFAULT_CUSTOM_DATABASE_DIR", root)
        jobs = discover_custom_audit_jobs(tmp_path / "out")

        prompt_names = sorted(j.prompt_path.name for j in jobs)
        assert prompt_names == ["p1.jsonl", "p2.jsonl", "p3.jsonl"]
        # Output path nests under domain/variant/stem.
        assert all("countries" in str(j.output_path) for j in jobs)

    def test_skips_domain_without_prompts_dir(self, tmp_path, monkeypatch):
        root = tmp_path / "custom_databases"
        root.mkdir()
        # A domain dir exists but has NO prompts/ subdirectory.
        (root / "empty_domain").mkdir()
        monkeypatch.setattr(run_audit, "DEFAULT_CUSTOM_DATABASE_DIR", root)
        assert discover_custom_audit_jobs(tmp_path / "out") == []

    def test_skips_variant_without_matching_db_json(self, tmp_path, monkeypatch):
        root = tmp_path / "custom_databases"
        domain = root / "countries"
        (domain / "prompts" / "missing_db_variant").mkdir(parents=True)
        (domain / "prompts" / "missing_db_variant" / "p.jsonl").write_text(
            "", encoding="utf-8"
        )
        monkeypatch.setattr(run_audit, "DEFAULT_CUSTOM_DATABASE_DIR", root)
        # No countries/missing_db_variant.json exists → variant is skipped.
        assert discover_custom_audit_jobs(tmp_path / "out") == []

    def test_ignores_non_directories_at_domain_level(self, tmp_path, monkeypatch):
        root = tmp_path / "custom_databases"
        root.mkdir()
        (root / "stray_file.txt").write_text("ignore me", encoding="utf-8")
        monkeypatch.setattr(run_audit, "DEFAULT_CUSTOM_DATABASE_DIR", root)
        assert discover_custom_audit_jobs(tmp_path / "out") == []


class TestDiscoverReleasedAuditJobs:
    def _build_released(self, root: Path, prompt_files: list[str]) -> Path:
        root.mkdir(parents=True)
        (root / "lmlm_database.json").write_text("{}", encoding="utf-8")
        prompts_dir = root / "prompts"
        prompts_dir.mkdir()
        for name in prompt_files:
            (prompts_dir / name).write_text("", encoding="utf-8")
        return root

    def test_discovers_all_prompt_files(self, tmp_path, monkeypatch):
        root = tmp_path / "released_database"
        self._build_released(root, ["prompts_a.jsonl", "prompts_b.jsonl"])
        monkeypatch.setattr(run_audit, "DEFAULT_RELEASED_DATABASE_DIR", root)

        jobs = discover_released_audit_jobs(tmp_path / "out")

        assert sorted(j.prompt_path.name for j in jobs) == [
            "prompts_a.jsonl",
            "prompts_b.jsonl",
        ]
        assert all(j.database_path == root / "lmlm_database.json" for j in jobs)
        # Output nests under released_database/lmlm_database/<stem>.
        assert all(
            j.output_path
            == tmp_path
            / "out"
            / "released_database"
            / "lmlm_database"
            / f"{j.prompt_path.stem}_results.jsonl"
            for j in jobs
        )

    def test_returns_empty_when_dir_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            run_audit,
            "DEFAULT_RELEASED_DATABASE_DIR",
            tmp_path / "does_not_exist",
        )
        assert discover_released_audit_jobs(tmp_path / "out") == []

    def test_returns_empty_when_database_missing(self, tmp_path, monkeypatch):
        root = tmp_path / "released_database"
        root.mkdir()
        (root / "prompts").mkdir()
        (root / "prompts" / "p.jsonl").write_text("", encoding="utf-8")
        monkeypatch.setattr(run_audit, "DEFAULT_RELEASED_DATABASE_DIR", root)
        # No lmlm_database.json present → skip.
        assert discover_released_audit_jobs(tmp_path / "out") == []

    def test_returns_empty_when_prompts_dir_missing(self, tmp_path, monkeypatch):
        root = tmp_path / "released_database"
        root.mkdir()
        (root / "lmlm_database.json").write_text("{}", encoding="utf-8")
        monkeypatch.setattr(run_audit, "DEFAULT_RELEASED_DATABASE_DIR", root)
        # No prompts/ present → skip.
        assert discover_released_audit_jobs(tmp_path / "out") == []


class TestDiscoverAllAuditJobs:
    def test_combines_custom_and_released(self, tmp_path, monkeypatch):
        custom_root = tmp_path / "custom_databases"
        domain = custom_root / "countries"
        (domain / "prompts" / "alias").mkdir(parents=True)
        (domain / "alias.json").write_text("{}", encoding="utf-8")
        (domain / "prompts" / "alias" / "p1.jsonl").write_text("", encoding="utf-8")

        released_root = tmp_path / "released_database"
        released_root.mkdir()
        (released_root / "lmlm_database.json").write_text("{}", encoding="utf-8")
        (released_root / "prompts").mkdir()
        (released_root / "prompts" / "p2.jsonl").write_text("", encoding="utf-8")

        monkeypatch.setattr(run_audit, "DEFAULT_CUSTOM_DATABASE_DIR", custom_root)
        monkeypatch.setattr(run_audit, "DEFAULT_RELEASED_DATABASE_DIR", released_root)

        jobs = discover_all_audit_jobs(tmp_path / "out")
        prompt_names = sorted(j.prompt_path.name for j in jobs)
        assert prompt_names == ["p1.jsonl", "p2.jsonl"]
        databases = {j.database_path for j in jobs}
        assert databases == {
            domain / "alias.json",
            released_root / "lmlm_database.json",
        }


class TestInferPromptPathsForDatabase:
    def test_returns_matching_jsonl_files(self, tmp_path):
        db = tmp_path / "countries" / "alias.json"
        db.parent.mkdir(parents=True)
        db.write_text("{}", encoding="utf-8")
        prompts_dir = db.parent / "prompts" / "alias"
        prompts_dir.mkdir(parents=True)
        (prompts_dir / "a.jsonl").write_text("", encoding="utf-8")
        (prompts_dir / "b.jsonl").write_text("", encoding="utf-8")
        (prompts_dir / "not_a_prompt.txt").write_text("", encoding="utf-8")
        paths = infer_prompt_paths_for_database(db)
        names = sorted(p.name for p in paths)
        assert names == ["a.jsonl", "b.jsonl"]

    def test_returns_empty_when_sibling_dir_missing(self, tmp_path):
        db = tmp_path / "countries" / "alias.json"
        db.parent.mkdir(parents=True)
        db.write_text("{}", encoding="utf-8")
        assert infer_prompt_paths_for_database(db) == []


# ===========================================================================
# resolve_audit_jobs
# ===========================================================================


class TestResolveAuditJobs:
    def _args(self, **over):
        ns = argparse.Namespace(
            prompt_files=None,
            database_path=run_audit.DEFAULT_DATABASE_PATH,
            output_dir=Path("outputs/audit"),
        )
        for k, v in over.items():
            setattr(ns, k, v)
        return ns

    def test_uses_prompt_files_when_provided(self, tmp_path):
        files = [tmp_path / "p1.jsonl", tmp_path / "p2.jsonl"]
        for f in files:
            f.write_text("", encoding="utf-8")
        args = self._args(
            prompt_files=files,
            database_path=tmp_path / "db.json",
            output_dir=tmp_path / "out",
        )
        jobs = resolve_audit_jobs(args)
        assert len(jobs) == 2
        assert all(j.database_path == tmp_path / "db.json" for j in jobs)
        assert jobs[0].output_path == tmp_path / "out" / "p1_results.jsonl"

    def test_custom_db_infers_prompts_from_sibling_dir(self, tmp_path):
        db = tmp_path / "countries" / "alias.json"
        db.parent.mkdir(parents=True)
        db.write_text("{}", encoding="utf-8")
        pd = db.parent / "prompts" / "alias"
        pd.mkdir(parents=True)
        (pd / "q.jsonl").write_text("", encoding="utf-8")
        args = self._args(
            database_path=db,
            output_dir=tmp_path / "out",
        )
        jobs = resolve_audit_jobs(args)
        assert len(jobs) == 1
        assert jobs[0].prompt_path.name == "q.jsonl"
        # Output nests under domain/variant.
        assert "countries" in str(jobs[0].output_path)

    def test_fallback_to_discover_when_default_db_and_no_prompts(
        self, tmp_path, monkeypatch
    ):
        root = tmp_path / "custom_databases"
        root.mkdir()
        released = tmp_path / "released_database"
        released.mkdir()
        # Point the loader at empty custom_databases / released_database.
        monkeypatch.setattr(run_audit, "DEFAULT_CUSTOM_DATABASE_DIR", root)
        monkeypatch.setattr(run_audit, "DEFAULT_RELEASED_DATABASE_DIR", released)
        args = self._args(output_dir=tmp_path / "out")
        assert resolve_audit_jobs(args) == []

    def test_custom_db_without_sibling_prompts_falls_back_to_discover(
        self, tmp_path, monkeypatch
    ):
        # A non-default DB path that has no sibling prompts dir.
        db = tmp_path / "lonely.json"
        db.write_text("{}", encoding="utf-8")
        root = tmp_path / "custom_databases"
        root.mkdir()
        released = tmp_path / "released_database"
        released.mkdir()
        monkeypatch.setattr(run_audit, "DEFAULT_CUSTOM_DATABASE_DIR", root)
        monkeypatch.setattr(run_audit, "DEFAULT_RELEASED_DATABASE_DIR", released)
        args = self._args(database_path=db, output_dir=tmp_path / "out")
        # Falls through to discover_all_audit_jobs → empty.
        assert resolve_audit_jobs(args) == []


# ===========================================================================
# setup_wandb
# ===========================================================================


class TestSetupWandb:
    def test_raises_when_api_key_missing(self, tmp_path, monkeypatch):
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        # Patch load_dotenv so it doesn't pick up a real .env.
        fake_dotenv = MagicMock()
        fake_dotenv.load_dotenv = MagicMock()
        with patch.dict(sys.modules, {"dotenv": fake_dotenv}):
            with pytest.raises(RuntimeError, match="WANDB_API_KEY"):
                setup_wandb()

    def test_returns_wandb_after_login(self, monkeypatch):
        monkeypatch.setenv("WANDB_API_KEY", "k123")
        fake_dotenv = MagicMock()
        fake_dotenv.load_dotenv = MagicMock()
        fake_wandb = MagicMock()
        with patch.dict(
            sys.modules, {"dotenv": fake_dotenv, "wandb": fake_wandb}
        ):
            result = setup_wandb()
        assert result is fake_wandb
        fake_wandb.login.assert_called_once_with(key="k123", relogin=True)


# ===========================================================================
# log_metrics_to_wandb
# ===========================================================================


class TestLogMetricsToWandb:
    def test_logs_run_with_state_and_cross_state_metrics(self, tmp_path):
        wandb_module = MagicMock()
        run = MagicMock()
        wandb_module.init.return_value = run

        log_metrics_to_wandb(
            wandb_module=wandb_module,
            prompt_path=Path("prompts/x.jsonl"),
            state=DatabaseState.FULL,
            state_metrics={"f1": 0.8},
            cross_state_metrics={"paired_count": 5},
            model_name="m",
            database_path=Path("db.json"),
            max_new_tokens=12,
            limit=None,
        )

        wandb_module.init.assert_called_once()
        init_kwargs = wandb_module.init.call_args.kwargs
        assert init_kwargs["project"] == run_audit.WANDB_PROJECT
        assert init_kwargs["name"].endswith("_FULL")
        assert init_kwargs["config"]["state"] == "FULL"

        logged = run.log.call_args.args[0]
        assert "state/f1" in logged
        assert "cross_state/paired_count" in logged
        run.finish.assert_called_once()


# ===========================================================================
# AuditJob dataclass
# ===========================================================================


class TestAuditJob:
    def test_is_frozen(self):
        j = AuditJob(Path("p"), Path("d"), Path("o"))
        with pytest.raises(dataclasses_FrozenInstanceError := Exception):
            j.prompt_path = Path("other")  # type: ignore[misc]

    def test_equal_when_fields_equal(self):
        a = AuditJob(Path("p"), Path("d"), Path("o"))
        b = AuditJob(Path("p"), Path("d"), Path("o"))
        assert a == b
