"""Audit backend for the public Co-LMLM release.

This package is just the model: the released checkpoint, its loader, and its
search adapter. The database being audited (the retrieval index) is a general
audit input (`--index-path`), not a property of the model.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from halo.core.backend import AuditBackend
from halo.registry import BackendSpec, register_backend

# The released model and how to reach the public source. These are fixed for
# the audit; the only thing that varies per run is where the index lives.
MODEL = "lil-lab/CoLMLM-360M-FW"
SOURCE_PATH = "."  # run from the public Co-LMLM checkout
SIMILARITY_THRESHOLD: float | None = None  # match the released eval config
NPROBE: int | None = None  # use the index's own nprobe


def _build_backend(args: argparse.Namespace, _group_key: Any) -> AuditBackend:
    from models.co_lmlm.backend import CoLMLMAuditBackend

    index_path = Path(args.index_path)
    return CoLMLMAuditBackend.from_public_release(
        model_path=MODEL,
        index_path=index_path,
        db_path=index_path / "entries.db",
        source_path=SOURCE_PATH,
        similarity_threshold=SIMILARITY_THRESHOLD,
        nprobe=NPROBE,
        max_new_tokens=args.max_new_tokens,
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


register_backend(
    BackendSpec(
        name="co-lmlm",
        build_backend=_build_backend,
        build_search_index=_search_index,
        group_key=_group_key,
        validate=_validate,
    )
)
