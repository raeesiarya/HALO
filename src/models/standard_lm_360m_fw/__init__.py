"""Audit backend for the Co-LMLM paper's Standard LM baseline (360M, FW).

This package is just the model: the released Standard LM checkpoint — the
same SmolLM2-360M architecture trained from scratch on the same FineWeb-Edu
corpus as CoLMLM-360M-FW, but on unannotated text with no retrieval. It is
the matched-data parametric control: against `co-lmlm` it isolates the
externalization recipe with training data held constant, while
`smollm2-360m` remains the off-the-shelf parametric reference.

All three database states still run; with no memory to delete from they
collapse to the same computation, so L(f) reads closed-book correctness and
R(f)/I(f) are identically zero. The deletion-closure machinery is rejected
up front, exactly as for `smollm2-360m`.
"""

from __future__ import annotations

import argparse
from typing import Any

from halo.core.backend import AuditBackend
from halo.registry import BackendSpec, register_backend

# The released model is fixed for the audit; nothing varies per run.
MODEL = "lil-lab/CoLMLM-Standard-LM-Baseline-360M-FW"


def _build_backend(args: argparse.Namespace, _group_key: Any) -> AuditBackend:
    from models.standard_lm_360m_fw.backend import StandardLMAuditBackend

    return StandardLMAuditBackend.from_pretrained(model_path=MODEL)


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
        raise ValueError("standard-lm-360m-fw runs require explicit --prompt-files.")
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
            f"Not supported for standard-lm-360m-fw: {', '.join(rejected)}. "
            "This backend is a parametric (closed-book) baseline with no "
            "retrieval index, so the deletion machinery has no referent. "
            "Run the standard three-state audit without these flags."
        )


register_backend(
    BackendSpec(
        name="standard-lm-360m-fw",
        build_backend=_build_backend,
        build_search_index=_search_index,
        group_key=_group_key,
        validate=_validate,
    )
)
