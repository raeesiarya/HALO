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
SIMILARITY_THRESHOLD: float | None = 0.7  # Co-LMLM paper factual-eval setting
NPROBE: int | None = None  # use the index's own nprobe


def _optional_similarity_threshold(value: str) -> float | None:
    if value.casefold() in {"none", "off", "disabled"}:
        return None
    threshold = float(value)
    if not -1.0 <= threshold <= 1.0:
        raise argparse.ArgumentTypeError(
            "similarity threshold must be in [-1, 1], or 'none'"
        )
    return threshold


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Co-LMLM controls")
    group.add_argument(
        "--co-lmlm-similarity-threshold",
        type=_optional_similarity_threshold,
        default=SIMILARITY_THRESHOLD,
        metavar="FLOAT|none",
        help=(
            "Retrieval threshold. Defaults to 0.7, the Co-LMLM factual "
            "evaluation setting. Pass 'none' only for a labeled "
            "threshold-disabled sensitivity run."
        ),
    )
    group.add_argument(
        "--co-lmlm-del-off-mode",
        choices=("null-retrieval", "forbid-token"),
        default="null-retrieval",
        help=(
            "DEL-OFF control: let <FACT> lookups return no candidate and then "
            "fall back to decoding, or forbid retrieval tokens entirely. "
            "Report both modes before interpreting parametric leakage."
        ),
    )
    group.add_argument(
        "--co-lmlm-assume-exact-index",
        action="store_true",
        help=(
            "Trust the sweep's full-pass reuse (full_row_unaffected). Only sound "
            "when the index's top-1 is independent of search depth, i.e. an exact "
            "(Flat) index. The public IVF/PQ wiki index is approximate, so leave "
            "this off: DEL-ON over-fetches relative to the FULL pass and a deeper "
            "search can change the top-1, which the reuse canary flags. Fingerprint "
            "reuse stays enabled either way."
        ),
    )


# Entry-store filename across index releases: the wiki-only bucket ships
# `entries.db`; the fineweb+wiki bucket ships `fineweb_with_fullwiki_entries.db`.
_ENTRY_DB_NAMES = ("fineweb_with_fullwiki_entries.db", "entries.db")


def _resolve_entry_db(index_path: Path) -> Path:
    for name in _ENTRY_DB_NAMES:
        candidate = index_path / name
        if candidate.exists():
            return candidate
    # Fall back to the newest release's name; the loader raises a clear error
    # if it is genuinely missing.
    return index_path / _ENTRY_DB_NAMES[0]


def _build_backend(args: argparse.Namespace, _group_key: Any) -> AuditBackend:
    from models.co_lmlm.backend import CoLMLMAuditBackend

    index_path = Path(args.index_path)
    return CoLMLMAuditBackend.from_public_release(
        model_path=MODEL,
        index_path=index_path,
        db_path=_resolve_entry_db(index_path),
        source_path=SOURCE_PATH,
        similarity_threshold=args.co_lmlm_similarity_threshold,
        nprobe=NPROBE,
        max_new_tokens=args.max_new_tokens,
        del_off_mode=args.co_lmlm_del_off_mode,
        assume_exact_index=args.co_lmlm_assume_exact_index,
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
        add_arguments=_add_arguments,
        validate=_validate,
        supports_oracle_bootstrap=True,
    )
)
