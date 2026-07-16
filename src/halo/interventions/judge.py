from __future__ import annotations

from typing import Any

from halo.core.equivalence import normalize_text
from halo.core.examples import AuditExample
from halo.interventions.filtering import _candidate_text


def default_support_judge(candidate: Any, example: AuditExample) -> dict[str, Any]:
    """Backend-agnostic support judge: does a retrieved candidate's value
    mention the target answer (or one of its aliases) as a whole phrase?"""
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
