from __future__ import annotations

import re
from typing import Any

from lmlm_audit.colmlm.index_filter import _candidate_text
from lmlm_audit.core.equivalence import normalize_text
from lmlm_audit.core.examples import AuditExample


def _default_support_judge(candidate: Any, example: AuditExample) -> dict[str, Any]:
    text = normalize_text(_candidate_text(candidate))
    answers = (example.ground_truth, *example.object_aliases)
    padded_text = f" {text} "
    supports = any(
        normalized and f" {normalized} " in padded_text
        for answer in answers
        if (normalized := normalize_text(answer))
    )
    return {
        "supports_target": supports,
        "support_method": "normalized-answer-mention",
        "support_confidence": 1.0 if supports else 0.0,
    }


_FACT_BLOCK_PATTERN = re.compile(r"<FACT>.*?</FACT>", re.DOTALL)
_SPECIAL_TOKEN_PATTERN = re.compile(r"</?[A-Z_]+>")


def extract_colmlm_answer(raw_text: str, prompt: str) -> str:
    completion = str(raw_text)
    if prompt and completion.startswith(prompt):
        completion = completion[len(prompt) :]
    completion = _FACT_BLOCK_PATTERN.sub(" ", completion)
    completion = _SPECIAL_TOKEN_PATTERN.sub(" ", completion)
    completion = re.sub(r"\s+", " ", completion).strip()
    for prefix in ("answer:", "the answer is", "it is", "it's"):
        if completion.casefold().startswith(prefix):
            completion = completion[len(prefix) :].strip()
            break
    completion = re.split(r"(?<=[.!?])\s+", completion, maxsplit=1)[0]
    return completion.strip(" \t\n\r\"'`,;:.")
