"""Audit backend for the original relational LMLM."""

from __future__ import annotations

import argparse
from typing import Any

from lmlm_audit.core.backend import AuditBackend
from lmlm_audit.registry import BackendSpec, register_backend


def _build_backend(args: argparse.Namespace, group_key: Any) -> AuditBackend:
    from models.rel_lmlm.backend import (
        RelLMLMAuditBackend,
        load_model_and_tokenizer,
    )

    # group_key is the database path this group of prompt files audits.
    model, tokenizer = load_model_and_tokenizer(
        model_name=args.model_name,
        database_path=group_key,
    )
    return RelLMLMAuditBackend(
        base_db_manager=model.db_manager,
        model=model,
        tokenizer=tokenizer,
    )


def _search_index(backend: AuditBackend) -> Any:
    from models.rel_lmlm.adapter import build_search_index

    return build_search_index(backend)


def _group_key(args: argparse.Namespace, job: Any) -> Any:
    # Each database has its own retriever, so group jobs by database path.
    return job.database_path


def _validate(args: argparse.Namespace) -> None:
    closure = getattr(args, "closure", None)
    if closure is not None and "provenance" in closure:
        raise ValueError(
            "rel-LMLM triples carry no provenance metadata; use "
            "--closure geometric,semantic."
        )


register_backend(
    BackendSpec(
        name="rel-lmlm",
        build_backend=_build_backend,
        build_search_index=_search_index,
        group_key=_group_key,
        validate=_validate,
    )
)
