"""The <FACT-q> query-ensemble deletion rule (the "factq" closure predicate).

Co-LMLM was trained to map questions to facts, so the most
training-aligned way to enumerate a fact's footprint in the index is to
ask the model itself: generate questions about the fact with the released
question-generator adapter, embed each question through Co-LMLM's own
<FACT-q> retrieval head, and delete the union of everything those queries
retrieve above the threshold.

This module holds the model-free pieces shared by the three stages:

1. ``data/generate_factq_questions.py`` marks the fact span in a DB entry
   and prompts the generator (helpers: ``mark_fact_span``,
   ``build_generator_prompt``, ``parse_generator_output``).
2. ``python -m halo.factq_embed`` runs each question through the Co-LMLM
   backend and captures the <FACT-q> query vector (helpers:
   ``factq_vector_key``, ``save_factq_vectors``).
3. The closure build consumes the vectors (``load_factq_vectors``) and
   attributes caught entries to the ``factq`` predicate.

The eval prompts themselves must never feed stage 1 or 2: deleting what
the eval question retrieves guarantees a successful forget without
measuring anything (the "cheating" failure mode).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np

# Enriched-prompt-row fields written by stage 1 and read by stage 2. They
# ride along unknown-field passthrough (AuditExample.source_row), so the
# audit prompt schema needs no change.
FACTQ_QUESTIONS_FIELD = "factq_questions"
FACTQ_ANSWERS_FIELD = "factq_answers"
FACTQ_SOURCE_ENTRY_FIELD = "factq_source_entry_id"

# The generator was trained on documents whose spans carry this exact
# markup (see the model card): <FACT>N<FACT_ID>span</FACT>.
_GENERATOR_QUESTION_PATTERN = re.compile(
    r"<QUESTION>(?P<question>.*?)</QUESTION>\s*<ANSWER>(?P<answer>.*?)</ANSWER>",
    re.DOTALL,
)

# npz key layout, mirroring the query-embedding sidecar convention
# ``{example_key}/{state}/event{n}``; the middle segment names this pass.
_VECTOR_STATE_SEGMENT = "FACTQ"


def prompt_row_key(row: Mapping[str, Any]) -> str:
    """The persistence key for a prompt row: prompt_id, then fact_id.

    Matches ``halo.core.entanglement.fact_key`` (and the closure build's
    ``_example_key``) so stage artifacts and closures agree on identity.
    """
    for field_name in ("prompt_id", "fact_id"):
        value = row.get(field_name)
        if value is not None:
            return str(value)
    return ""


def mark_fact_span(
    document: str, answer: str, aliases: tuple[str, ...] = ()
) -> str | None:
    """Wrap the first answer mention in ``document`` as fact span 1.

    Candidates are tried longest-first (the gold answer plus aliases) with
    a case-insensitive search; the document's own casing is preserved
    inside the tag. Returns None when no candidate occurs, which the
    caller must count — a span miss silently dropping a fact from the
    factq closure would read as coverage.
    """
    candidates = sorted(
        {text.strip() for text in (answer, *aliases) if text and text.strip()},
        key=len,
        reverse=True,
    )
    lowered = document.casefold()
    for candidate in candidates:
        position = lowered.find(candidate.casefold())
        if position < 0:
            continue
        end = position + len(candidate)
        span = document[position:end]
        return (
            f"{document[:position]}<FACT>1<FACT_ID>{span}</FACT>{document[end:]}"
        )
    return None


def build_generator_prompt(marked_document: str) -> str:
    """The generator's user message for fact span 1 (model-card format)."""
    return (
        f"{marked_document}<DOC_SEP>\n\n"
        "What are the question and paraphrased answer for <FACT>1<FACT_ID>?\n"
    )


def parse_generator_output(text: str) -> tuple[str, str] | None:
    """Extract (question, paraphrased answer) from a generator completion.

    Returns None for malformed output or an empty question; the caller
    counts parse misses.
    """
    match = _GENERATOR_QUESTION_PATTERN.search(text)
    if match is None:
        return None
    question = match.group("question").strip()
    answer = match.group("answer").strip()
    if not question:
        return None
    return question, answer


def dedupe_questions(
    pairs: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Order-preserving dedupe on the casefolded question text."""
    seen: set[str] = set()
    unique: list[tuple[str, str]] = []
    for question, answer in pairs:
        normalized = question.casefold()
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append((question, answer))
    return unique


def factq_vector_key(example_key: str, question_index: int) -> str:
    if "/" in example_key:
        raise ValueError(
            f"Fact key {example_key!r} contains '/', which collides with "
            "the npz key convention {key}/FACTQ/q{n}."
        )
    return f"{example_key}/{_VECTOR_STATE_SEGMENT}/q{question_index}"


def save_factq_vectors(
    path: Path, vectors: Mapping[str, Mapping[int, Any]]
) -> None:
    """``{example_key: {question_index: vector}}`` -> one compressed npz.

    Question indices may have gaps (a question whose generation never
    issued a search has no vector); indices keep vectors aligned with the
    enriched rows' question lists.
    """
    flat: dict[str, np.ndarray] = {}
    for example_key, by_index in vectors.items():
        for question_index, vector in by_index.items():
            flat[factq_vector_key(example_key, int(question_index))] = np.asarray(
                vector, dtype=np.float32
            ).reshape(-1)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **flat)


def load_factq_vector_map(path: Path) -> dict[str, dict[int, np.ndarray]]:
    """``{example_key: {question_index: vector}}``, preserving index gaps."""
    by_key: dict[str, dict[int, np.ndarray]] = {}
    with np.load(path) as stored:
        for stored_key in stored.files:
            example_key, segment, question = stored_key.rsplit("/", 2)
            if segment != _VECTOR_STATE_SEGMENT or not question.startswith("q"):
                raise ValueError(
                    f"Unexpected key {stored_key!r} in {path}; expected "
                    "{example_key}/FACTQ/q{n}."
                )
            by_key.setdefault(example_key, {})[
                int(question.removeprefix("q"))
            ] = stored[stored_key]
    return by_key


def load_factq_vectors(path: Path) -> dict[str, tuple[np.ndarray, ...]]:
    """Per-fact <FACT-q> query vectors, ordered by question index."""
    return {
        example_key: tuple(vector for _, vector in sorted(by_index.items()))
        for example_key, by_index in load_factq_vector_map(path).items()
    }
