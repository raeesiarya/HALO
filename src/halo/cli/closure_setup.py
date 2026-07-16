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
    )


def make_closure_manifest_builder(
    backend: Any, search_index: Any, args: argparse.Namespace, job: AuditJob
) -> Any:
    from halo.interventions.closure import build_closure_manifest_from_full

    config = closure_config_from_args(args)
    artifact_dir = job.output_path.parent / f"{job.prompt_path.stem}_closures"

    def builder(example: Any, full_result: dict[str, Any]) -> Any:
        return build_closure_manifest_from_full(
            index=search_index,
            example=example,
            full_result=full_result,
            config=config,
            support_judge=backend.support_judge,
            artifact_dir=artifact_dir,
        )

    return builder
