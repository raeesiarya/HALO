import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lmlm_audit.core.equivalence import (
    _flatten_alias_values,
    _unique_preserving_order,
    build_alias_set,
    normalize_text,
    prompt_row_aliases,
    tokenize,
    values_equivalent,
)



class TestTokenize:
    def test_empty_string(self):
        assert tokenize("") == []

    def test_single_word(self):
        assert tokenize("hello") == ["hello"]

    def test_multiple_words(self):
        assert tokenize("hello world") == ["hello", "world"]

    def test_case_folding(self):
        assert tokenize("Hello WORLD") == ["hello", "world"]

    def test_all_uppercase(self):
        assert tokenize("NASA") == ["nasa"]

    def test_unicode_word_character(self):
        result = tokenize("Jørgensen")
        assert result == ["jørgensen"]

    def test_decimal_number_matched_first(self):
        assert tokenize("69.7") == ["69.7"]

    def test_decimal_inside_sentence(self):
        result = tokenize("$69.7 million")
        assert "69.7" in result
        assert "million" in result
        assert "$" not in " ".join(result)

    def test_integer_only(self):
        assert tokenize("1956") == ["1956"]

    def test_integer_in_sentence(self):
        result = tokenize("born in 1956")
        assert "1956" in result

    def test_hyphenated_word(self):
        assert tokenize("well-known") == ["well-known"]

    def test_apostrophe_contraction(self):
        assert tokenize("don't") == ["don't"]

    def test_apostrophe_possessive(self):
        assert tokenize("John's") == ["john's"]

    def test_punctuation_stripped(self):
        result = tokenize("Hello, world!")
        assert result == ["hello", "world"]

    def test_only_punctuation(self):
        assert tokenize("!@#$%^&*()") == []

    def test_multiple_spaces(self):
        result = tokenize("a   b")
        assert result == ["a", "b"]

    def test_newline(self):
        result = tokenize("line1\nline2")
        assert result == ["line1", "line2"]

    def test_tab(self):
        result = tokenize("col1\tcol2")
        assert result == ["col1", "col2"]

    def test_non_string_integer(self):
        result = tokenize(42)
        assert result == ["42"]

    def test_non_string_float(self):
        result = tokenize(3.14)
        assert "3" in result or "3.14" in result

    def test_mixed_unicode_accents(self):
        result = tokenize("naïve café")
        assert "naïve" in result
        assert "café" in result

    def test_leading_trailing_whitespace(self):
        result = tokenize("  hello  ")
        assert result == ["hello"]

    def test_number_adjacent_to_word(self):
        result = tokenize("2pac")
        assert result == ["2pac"]

    def test_currency_symbol_ignored(self):
        result = tokenize("$100")
        assert "100" in result



class TestNormalizeText:
    def test_empty(self):
        assert normalize_text("") == ""

    def test_single_word(self):
        assert normalize_text("Hello") == "hello"

    def test_multiple_words(self):
        assert normalize_text("Spice Girls!") == "spice girls"

    def test_decimal(self):
        assert normalize_text("$69.7 million") == "69.7 million"

    def test_collapses_spaces(self):
        result = normalize_text("a  b  c")
        assert result == "a b c"

    def test_unicode(self):
        assert normalize_text("Jørgensen") == "jørgensen"

    def test_punctuation_only(self):
        assert normalize_text("!!!") == ""

    def test_idempotent(self):
        once = normalize_text("Hello World!")
        twice = normalize_text(once)
        assert once == twice

    def test_preserves_hyphen_in_token(self):
        result = normalize_text("state-of-the-art")
        assert "state-of-the-art" in result



class TestFlattenAliasValues:
    def test_none(self):
        assert _flatten_alias_values(None) == []

    def test_empty_string(self):
        assert _flatten_alias_values("") == [""]

    def test_non_empty_string(self):
        assert _flatten_alias_values("hello") == ["hello"]

    def test_list_of_strings(self):
        assert _flatten_alias_values(["a", "b"]) == ["a", "b"]

    def test_tuple_of_strings(self):
        assert _flatten_alias_values(("a", "b")) == ["a", "b"]

    def test_set_of_strings(self):
        result = _flatten_alias_values({"a", "b"})
        assert sorted(result) == ["a", "b"]

    def test_nested_list(self):
        assert _flatten_alias_values([["a", "b"], "c"]) == ["a", "b", "c"]

    def test_none_inside_list(self):
        assert _flatten_alias_values([None, "a"]) == ["a"]

    def test_integer_scalar(self):
        assert _flatten_alias_values(42) == ["42"]

    def test_float_scalar(self):
        result = _flatten_alias_values(3.14)
        assert len(result) == 1

    def test_deeply_nested(self):
        result = _flatten_alias_values([[["a"], "b"], ["c"]])
        assert result == ["a", "b", "c"]

    def test_empty_list(self):
        assert _flatten_alias_values([]) == []

    def test_empty_tuple(self):
        assert _flatten_alias_values(()) == []

    def test_empty_set(self):
        assert _flatten_alias_values(set()) == []

    def test_list_with_integers(self):
        result = _flatten_alias_values([1, 2, 3])
        assert result == ["1", "2", "3"]

    def test_mixed_types(self):
        result = _flatten_alias_values(["a", 1, None, ["b"]])
        assert "a" in result
        assert "1" in result
        assert "b" in result
        assert None not in result



class TestUniquePreservingOrder:
    def test_empty(self):
        assert _unique_preserving_order([]) == ()

    def test_single_element(self):
        assert _unique_preserving_order(["hello"]) == ("hello",)

    def test_no_duplicates(self):
        result = _unique_preserving_order(["a", "b", "c"])
        assert result == ("a", "b", "c")

    def test_deduplicates_same_case(self):
        result = _unique_preserving_order(["paris", "paris"])
        assert result == ("paris",)

    def test_deduplicates_different_case(self):
        result = _unique_preserving_order(["Hello", "hello"])
        assert result == ("Hello",)

    def test_preserves_first_occurrence(self):
        result = _unique_preserving_order(["B", "A", "b"])
        assert result == ("B", "A")

    def test_skips_empty_strings(self):
        result = _unique_preserving_order(["", "a", ""])
        assert "" not in result
        assert "a" in result

    def test_skips_whitespace_only(self):
        result = _unique_preserving_order(["   ", "a"])
        assert result == ("a",)

    def test_deduplicates_unicode_normalized(self):
        result = _unique_preserving_order(["Über", "über"])
        assert len(result) == 1

    def test_order_of_unique_elements_preserved(self):
        result = _unique_preserving_order(["z", "a", "m"])
        assert result == ("z", "a", "m")

    def test_punctuation_difference_still_same_normalized(self):
        result = _unique_preserving_order(["hello!", "hello"])
        assert len(result) == 1



class TestBuildAliasSet:
    def test_canonical_only_no_aliases(self):
        result = build_alias_set("Paris")
        assert "Paris" in result
        assert len(result) == 1

    def test_canonical_with_none_aliases(self):
        result = build_alias_set("Paris", None)
        assert result == ("Paris",)

    def test_canonical_with_single_alias(self):
        result = build_alias_set("Paris", ["City of Light"])
        assert "Paris" in result
        assert "City of Light" in result
        assert len(result) == 2

    def test_deduplicates_alias_equal_to_canonical_normalized(self):
        result = build_alias_set("Paris", ["paris"])
        assert len(result) == 1

    def test_canonical_with_multiple_aliases(self):
        result = build_alias_set("UK", ["United Kingdom", "Britain"])
        assert len(result) == 3

    def test_nested_alias_list_flattened(self):
        result = build_alias_set("Paris", [["Lutece", "City of Light"]])
        assert len(result) == 3

    def test_empty_canonical(self):
        result = build_alias_set("")
        assert result == ()

    def test_alias_deduplication_preserves_canonical_first(self):
        result = build_alias_set("UK", ["UK", "United Kingdom"])
        uk_count = sum(1 for x in result if normalize_text(x) == "uk")
        assert uk_count == 1



class TestPromptRowAliases:
    def test_subject_field(self):
        row = {"subject_aliases": ["Geri Halliwell", "Geri"]}
        result = prompt_row_aliases(row, "subject")
        assert "Geri Halliwell" in result
        assert "Geri" in result

    def test_relation_field(self):
        row = {"relation_aliases": ["Born In"]}
        result = prompt_row_aliases(row, "relation")
        assert "Born In" in result

    def test_object_field_all_three_keys(self):
        row = {
            "object_aliases": ["alias1"],
            "gold_object_aliases": ["alias2"],
            "answer_aliases": ["alias3"],
        }
        result = prompt_row_aliases(row, "object")
        assert "alias1" in result
        assert "alias2" in result
        assert "alias3" in result

    def test_unknown_field_returns_empty(self):
        row = {"subject_aliases": ["x"]}
        result = prompt_row_aliases(row, "nonexistent_field")
        assert result == ()

    def test_missing_key_returns_empty(self):
        result = prompt_row_aliases({}, "subject")
        assert result == ()

    def test_deduplication_across_object_keys(self):
        row = {
            "object_aliases": ["Spice Girls"],
            "gold_object_aliases": ["Spice Girls"],
        }
        result = prompt_row_aliases(row, "object")
        spice_count = sum(
            1 for x in result if normalize_text(x) == "spice girls"
        )
        assert spice_count == 1

    def test_none_value_for_key(self):
        row = {"subject_aliases": None}
        result = prompt_row_aliases(row, "subject")
        assert result == ()

    def test_list_value(self):
        row = {"subject_aliases": ["a", "b", "c"]}
        result = prompt_row_aliases(row, "subject")
        assert set(result) == {"a", "b", "c"}

    def test_nested_list_value(self):
        row = {"subject_aliases": [["a", "b"]]}
        result = prompt_row_aliases(row, "subject")
        assert "a" in result
        assert "b" in result

    def test_empty_list_value(self):
        row = {"subject_aliases": []}
        result = prompt_row_aliases(row, "subject")
        assert result == ()



class TestValuesEquivalent:
    def test_identical_strings(self):
        assert values_equivalent("Paris", "Paris") is True

    def test_case_insensitive(self):
        assert values_equivalent("paris", "Paris") is True

    def test_different_strings(self):
        assert values_equivalent("Paris", "Berlin") is False

    def test_match_via_right_alias(self):
        assert values_equivalent("UK", "United Kingdom", right_aliases=["UK"]) is True

    def test_match_via_left_alias(self):
        assert values_equivalent("United Kingdom", "UK", left_aliases=["UK"]) is True

    def test_no_match_with_unrelated_alias(self):
        assert values_equivalent("Paris", "Berlin", right_aliases=["France"]) is False

    def test_both_empty(self):
        assert values_equivalent("", "") is False

    def test_left_empty_right_nonempty(self):
        assert values_equivalent("", "Paris") is False

    def test_right_empty_left_nonempty(self):
        assert values_equivalent("Paris", "") is False

    def test_unicode_via_alias(self):
        assert values_equivalent(
            "Jorgensen", "Jørgensen", right_aliases=["Jorgensen"]
        ) is True

    def test_normalized_match_punctuation(self):
        assert values_equivalent("Spice Girls!", "Spice Girls") is True

    def test_multiple_right_aliases(self):
        assert values_equivalent(
            "The Beatles",
            "Beatles",
            right_aliases=["The Beatles", "Fab Four"],
        ) is True

    def test_alias_as_tuple(self):
        assert values_equivalent("x", "y", right_aliases=("x",)) is True

    def test_alias_list_no_match(self):
        assert values_equivalent("x", "y", right_aliases=["z"]) is False

    def test_hyphenated_vs_space(self):
        assert values_equivalent("state-of-the-art", "state of the art") is False

    def test_both_aliases_needed_for_match(self):
        assert values_equivalent(
            "GB",
            "Germany",
            left_aliases=["Great Britain"],
            right_aliases=["GB"],
        ) is True

    def test_number_equivalence(self):
        assert values_equivalent("42", "42") is True

    def test_number_mismatch(self):
        assert values_equivalent("42", "43") is False

    def test_whitespace_normalization(self):
        assert values_equivalent("New  York", "New York") is True



def test_tokenize_lengths_logged_to_wandb(wandb_run):
    """Bar chart of tokenisation counts, logged to W&B."""
    import matplotlib.pyplot as plt

    texts = [
        "Paris",
        "United Kingdom",
        "Spice Girls!",
        "$69.7 million",
        "I don't know",
        "Jørgensen",
        "",
        "well-known author",
        "New York City",
        "!@#$",
    ]
    lengths = [len(tokenize(t)) for t in texts]

    if wandb_run is not None:
        try:
            import wandb

            fig, ax = plt.subplots(figsize=(9, 4))
            ax.barh(texts, lengths, color="steelblue")
            ax.set_xlabel("Token count")
            ax.set_title("Tokenisation length by input text")
            plt.tight_layout()
            wandb_run.log({"equivalence/tokenize_lengths": wandb.Image(fig)})
            plt.close(fig)
        except Exception:
            pass

    assert lengths[texts.index("")] == 0
    assert lengths[texts.index("Paris")] == 1
    assert lengths[texts.index("!@#$")] == 0


def test_equivalence_matrix_logged_to_wandb(wandb_run):
    """Heatmap of pairwise values_equivalent results, logged to W&B."""
    import matplotlib.pyplot as plt
    import numpy as np

    labels = ["Paris", "paris", "Berlin", "UK", "United Kingdom", ""]
    n = len(labels)
    matrix = np.zeros((n, n), dtype=int)
    for i, left in enumerate(labels):
        for j, right in enumerate(labels):
            matrix[i, j] = int(values_equivalent(left, right))

    if wandb_run is not None:
        try:
            import wandb

            fig, ax = plt.subplots(figsize=(7, 6))
            im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_yticklabels(labels)
            ax.set_title("values_equivalent pairwise matrix")
            plt.colorbar(im, ax=ax)
            plt.tight_layout()
            wandb_run.log(
                {"equivalence/equivalence_matrix": wandb.Image(fig)}
            )
            plt.close(fig)
        except Exception:
            pass

    for i in range(n):
        if labels[i]:
            assert matrix[i, i] == 1, f"Self-equivalence failed for {labels[i]!r}"
    assert matrix[labels.index("Paris"), labels.index("Berlin")] == 0
