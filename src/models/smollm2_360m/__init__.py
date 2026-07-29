"""Audit backend for SmolLM2-360M, a parametric closed-book baseline.

This package is just the model: a plain Hugging Face causal LM with no
retrieval index and no external memory. It size-matches the Co-LMLM release
(a 360M Llama-architecture decoder), so the two backends isolate one
variable: whether knowledge lives in the weights or in an external store.

All three database states still run; with no memory to delete from they
collapse to the same computation, so L(f) reads closed-book correctness and
R(f)/I(f) are identically zero. The deletion-closure machinery
(--closure/--radius-grid/--adversarial) has no referent here and is
rejected up front.
"""

from __future__ import annotations

import argparse
from typing import Any

from halo.core.backend import AuditBackend
from halo.registry import BackendSpec, register_backend

# The released model is fixed for the audit; nothing varies per run.
MODEL = "HuggingFaceTB/SmolLM2-360M"


def _build_backend(args: argparse.Namespace, _group_key: Any) -> AuditBackend:
    from models.smollm2_360m.backend import SmolLM2AuditBackend

    return SmolLM2AuditBackend.from_pretrained(model_path=MODEL)


def _search_index(backend: AuditBackend) -> Any:
    # There is no retrieval index. The standard audit never dereferences the
    # search index; the closure/sweep/adversarial modes that would are
    # rejected in _validate below.
    return None


def _group_key(args: argparse.Namespace, _job: Any) -> Any:
    # One parametric model serves every prompt file: one backend per run.
    return MODEL


def _validate(args: argparse.Namespace) -> None:
    if args.prompt_files is None:
        raise ValueError("smollm2-360m runs require explicit --prompt-files.")
    rejected = [
        flag
        for flag, active in (
            ("--closure", args.closure is not None),
            ("--radius-grid", args.radius_grid is not None),
            ("--adversarial", args.adversarial),
            ("--bootstrap-oracle-from-full", args.bootstrap_oracle_from_full),
        )
        if active
    ]
    if rejected:
        raise ValueError(
            f"Not supported for smollm2-360m: {', '.join(rejected)}. This "
            "backend is a parametric (closed-book) baseline with no "
            "retrieval index, so the deletion machinery has no referent. "
            "Run the standard three-state audit without these flags."
        )


register_backend(
    BackendSpec(
        name="smollm2-360m",
        build_backend=_build_backend,
        build_search_index=_search_index,
        group_key=_group_key,
        validate=_validate,
    )
)
