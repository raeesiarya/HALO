"""Audit backend for SmolLM2-1.7B, the parametric reference for NULLs.

Same role `smollm2-360m` plays for Co-LMLM, one scale up: an off-the-shelf
Hugging Face causal LM from the same family as the memory model under
audit, with no retrieval index and no external memory. NULLs-Wikipedia is
~1B (SmolLM2 tokenizer, n_embd 960 / 32 layers widened to MLP 8192), and
1.7B is the nearest released SmolLM2 size, so this backend brackets it on
parameter count where the 360M entries cannot.

It is a *reference*, not a matched control: SmolLM2-1.7B is trained on the
full SmolLM2 web corpus while NULLs is trained on Wikipedia only, so its
closed-book accuracy is an upper bracket that the training data explains as
much as the architecture (docs/NULLS_AUDIT_DESIGN.md §10). The data- and
parameter-matched controls for NULLs remain its own DEL-OFF modes
(`sinks-zero`, `placebo-sink`); no no-sink Wikipedia LM was released.

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
MODEL = "HuggingFaceTB/SmolLM2-1.7B"


def _build_backend(args: argparse.Namespace, _group_key: Any) -> AuditBackend:
    from models.smollm2_1_7b.backend import SmolLM2LargeAuditBackend

    return SmolLM2LargeAuditBackend.from_pretrained(model_path=MODEL)


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
        raise ValueError("smollm2-1.7b runs require explicit --prompt-files.")
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
            f"Not supported for smollm2-1.7b: {', '.join(rejected)}. This "
            "backend is a parametric (closed-book) baseline with no "
            "retrieval index, so the deletion machinery has no referent. "
            "Run the standard three-state audit without these flags."
        )


register_backend(
    BackendSpec(
        name="smollm2-1.7b",
        build_backend=_build_backend,
        build_search_index=_search_index,
        group_key=_group_key,
        validate=_validate,
    )
)
