"""Audit backend for the public Co-LMLM release."""

from __future__ import annotations

import argparse
from typing import Any

from lmlm_audit.core.backend import AuditBackend
from lmlm_audit.registry import BackendSpec, register_backend


def _build_backend(args: argparse.Namespace, _group_key: Any) -> AuditBackend:
    from models.co_lmlm.backend import CoLMLMAuditBackend

    attn_implementation = (
        None
        if str(args.attn_implementation).casefold() == "none"
        else args.attn_implementation
    )
    return CoLMLMAuditBackend.from_public_release(
        model_path=args.colmlm_model_path,
        index_path=args.index_path,
        db_path=args.entries_db_path,
        source_path=args.colmlm_source_path,
        use_sqlite_id_mapping=args.use_sqlite_id_mapping,
        del_off_mode=args.del_off_mode,
        device=args.device,
        torch_dtype=args.torch_dtype,
        attn_implementation=attn_implementation,
        max_new_tokens=args.max_new_tokens,
        similarity_threshold=args.similarity_threshold,
        retrieval_top_k=args.retrieval_top_k,
        **({"nprobe": args.nprobe} if args.nprobe is not None else {}),
    )


def _search_index(backend: AuditBackend) -> Any:
    from models.co_lmlm.adapter import build_search_index

    return build_search_index(backend)


def _group_key(args: argparse.Namespace, _job: Any) -> Any:
    # One index serves every prompt file, so all jobs share one backend.
    return args.index_path


def _validate(args: argparse.Namespace) -> None:
    if args.prompt_files is None:
        raise ValueError("Co-LMLM runs require explicit --prompt-files.")
    if args.colmlm_model_path is None:
        raise ValueError("Co-LMLM runs require --colmlm-model-path.")
    if args.index_path is None:
        raise ValueError("Co-LMLM runs require --index-path.")


register_backend(
    BackendSpec(
        name="colmlm",
        build_backend=_build_backend,
        build_search_index=_search_index,
        group_key=_group_key,
        validate=_validate,
    )
)
