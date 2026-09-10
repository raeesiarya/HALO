"""Audit backend for NULLs (natively unlearnable LM), Wikipedia release.

Third deletion paradigm in the cross-model suite: knowledge lives in
source-keyed parametric sink neurons, and deletion = refusing to activate a
source's sink mask (docs/NULLS_AUDIT_DESIGN.md). Unlike the closed-book
baselines the three database states are real interventions, and unlike
Co-LMLM there is no external index: deletion manifests must carry Wikipedia
article titles in ``source_ids`` (the augmentation script writes the
provenance-source manifest into prompt rows; source-level closures extend
this).

Required artifacts, none derivable from the checkpoint; all released by the
authors (2026-09-10) and fetched by ``scripts/setup_nulls.sh``:

- checkpoint dir (``gauravrghosal/NULLS-Wikipedia-Full``, its ``final/``
  contents flattened): ``lit_model.pth`` + ``model_config.yaml`` +
  tokenizer files;
- ``title_to_index.pkl``: the training-time title -> integer mapping the
  sink masks are keyed on (6,407,814 raw MediaWiki titles, index = insertion
  position). Always the authors' artifact — a guessed mapping can silently
  mis-address sinks;
- title embeddings (required for DEL-ON routing, the placebo-sink DEL-OFF
  mode, and source-level geometric closures): built by
  ``scripts/build_nulls_title_embeddings.py`` (routing space as a scheduler
  prep job; closure space from the released corpus in setup).

The seq id fed to the model is ``index + vocab_size`` (49152), matching
upstream's tokenizer-side offset that keeps source ids disjoint from tokens.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from halo.core.backend import AuditBackend
from halo.registry import BackendSpec, register_backend

BACKEND_NAME = "nulls-wiki-1b"
DEFAULT_CHECKPOINT_DIR = "data/nulls-wikipedia-full"
DEFAULT_TITLE_TO_INDEX = "data/nulls-title-to-index.pkl"


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("nulls-wiki-1b")
    group.add_argument(
        "--nulls-checkpoint-dir",
        default=DEFAULT_CHECKPOINT_DIR,
        help="Released NULLs checkpoint directory (lit_model.pth + "
        "model_config.yaml + tokenizer).",
    )
    group.add_argument(
        "--nulls-title-to-index",
        default=DEFAULT_TITLE_TO_INDEX,
        help="Pickle mapping Wikipedia article title -> training source index.",
    )
    group.add_argument(
        "--nulls-title-embeddings",
        default=None,
        help="Title-embedding artifact (.npz from "
        "scripts/build_nulls_title_embeddings.py, or upstream's pickle). "
        "Required for DEL-ON routing and the placebo-sink DEL-OFF mode.",
    )
    group.add_argument(
        "--nulls-del-off-mode",
        choices=("sinks-zero", "placebo-sink"),
        default="sinks-zero",
        help="How DEL-OFF disables the sink memory: 'sinks-zero' is backbone-"
        "only; 'placebo-sink' activates a deterministic far-away source's "
        "mask (the sensitivity pair of design §6).",
    )
    group.add_argument(
        "--nulls-placebo-ceiling",
        type=float,
        default=0.5,
        help="Maximum cosine similarity (title-embedding space) between the "
        "placebo source and the audited source.",
    )


def _validate(args: argparse.Namespace) -> None:
    if args.prompt_files is None:
        raise ValueError("nulls-wiki-1b runs require explicit --prompt-files.")
    rejected = [
        flag
        for flag, active in (
            ("--adversarial", args.adversarial),
            ("--bootstrap-oracle-from-full", args.bootstrap_oracle_from_full),
            ("--closure", args.closure is not None),
            ("--radius-grid", args.radius_grid is not None),
        )
        if active
    ]
    if rejected:
        raise ValueError(
            f"Not supported for nulls-wiki-1b: {', '.join(rejected)}. "
            "Deletion for this backend is driven by source-title manifests "
            "in the prompt rows (scripts/augment_source_titles.py; "
            "source-level closures per docs/NULLS_AUDIT_DESIGN.md §4-5). "
            "The entry-level closure/adversarial machinery has no referent "
            "in a parametric sink pool."
        )
    if not Path(args.nulls_checkpoint_dir).is_dir():
        raise ValueError(
            f"--nulls-checkpoint-dir {args.nulls_checkpoint_dir!r} is not a "
            "directory; download gauravrghosal/NULLS-Wikipedia-Full first "
            "(scripts/setup_nulls.sh)."
        )
    if not Path(args.nulls_title_to_index).is_file():
        raise ValueError(
            f"--nulls-title-to-index {args.nulls_title_to_index!r} not found; "
            "scripts/setup_nulls.sh downloads the authors' title_to_index.pkl."
        )
    if args.nulls_del_off_mode == "placebo-sink" and not args.nulls_title_embeddings:
        raise ValueError(
            "--nulls-del-off-mode placebo-sink requires "
            "--nulls-title-embeddings (the placebo must be provably far from "
            "the audited source)."
        )


def _build_backend(args: argparse.Namespace, _group_key: Any) -> AuditBackend:
    from models.nulls_wiki.backend import NullsWikiAuditBackend

    return NullsWikiAuditBackend.from_release(
        checkpoint_dir=args.nulls_checkpoint_dir,
        title_to_index_path=args.nulls_title_to_index,
        title_embeddings_path=args.nulls_title_embeddings,
        del_off_mode=args.nulls_del_off_mode,
        placebo_similarity_ceiling=args.nulls_placebo_ceiling,
    )


def _search_index(backend: AuditBackend) -> Any:
    # No index over the audited store: the sink pool is addressed by title
    # manifests, not by vector search. Source-level closures are built ahead
    # of time into the prompt rows, so the standard audit never dereferences
    # this. (The closure-driving modes are rejected in _validate.)
    return None


def _group_key(args: argparse.Namespace, _job: Any) -> Any:
    return (
        args.nulls_checkpoint_dir,
        args.nulls_title_to_index,
        args.nulls_title_embeddings,
        args.nulls_del_off_mode,
    )


register_backend(
    BackendSpec(
        name=BACKEND_NAME,
        build_backend=_build_backend,
        build_search_index=_search_index,
        group_key=_group_key,
        add_arguments=_add_arguments,
        validate=_validate,
    )
)
