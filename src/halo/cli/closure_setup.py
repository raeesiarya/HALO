from __future__ import annotations

import argparse
from typing import Any

from halo.cli.jobs import AuditJob


def closure_config_from_args(args: argparse.Namespace) -> Any:
    from halo.interventions.closure import ClosureConfig

    return ClosureConfig(
        predicates=tuple(
            predicate.strip()
            for predicate in args.closure.split(",")
            if predicate.strip()
        ),
        radius=args.closure_radius,
        envelope_top_k=args.closure_envelope_k,
        max_closure_size=args.closure_max_size,
        factq_threshold=args.factq_threshold,
    )


def load_factq_vectors_from_args(args: argparse.Namespace) -> dict[str, tuple]:
    """The per-fact <FACT-q> vectors behind the factq predicate, keyed like
    the prompt rows. Empty when the predicate is inactive."""
    from halo.interventions.factq import load_factq_vectors

    config = closure_config_from_args(args)
    if not config.is_active("factq"):
        return {}
    if args.factq_vectors is None:
        raise ValueError(
            "--closure factq deletes what generated-question queries "
            "retrieve and requires --factq-vectors (produced by "
            "`python -m halo.factq_embed`)."
        )
    return load_factq_vectors(args.factq_vectors)


def make_closure_manifest_builder(
    backend: Any, search_index: Any, args: argparse.Namespace, job: AuditJob
) -> Any:
    from halo.interventions.closure import build_closure_manifest_from_full
    from halo.interventions.factq import prompt_row_key

    config = closure_config_from_args(args)
    factq_vectors = load_factq_vectors_from_args(args)
    artifact_dir = job.output_path.parent / f"{job.prompt_path.stem}_closures"

    def builder(example: Any, full_result: dict[str, Any]) -> Any:
        key = prompt_row_key(
            {"prompt_id": example.prompt_id, "fact_id": example.fact_id}
        )
        return build_closure_manifest_from_full(
            index=search_index,
            example=example,
            full_result=full_result,
            config=config,
            factq_query_vectors=factq_vectors.get(key, ()),
            support_judge=backend.support_judge,
            artifact_dir=artifact_dir,
        )

    return builder
