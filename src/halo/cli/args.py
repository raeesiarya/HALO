from __future__ import annotations

import argparse
from pathlib import Path

from halo.cli.jobs import DEFAULT_INDEX_DIR, DEFAULT_OUTPUT_DIR
from halo.registry import available_backends, get_backend_spec
import models  # noqa: F401  (imports register the bundled backends)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    # Pre-parse the backend so the selected model can contribute its own
    # arguments; the CLI itself only defines model-agnostic flags.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument(
        "--backend",
        choices=available_backends(),
        default="co-lmlm",
        help="Inference backend to audit (registered under src/models).",
    )
    backend = pre.parse_known_args(argv)[0].backend

    parser = argparse.ArgumentParser(description="Run the prompt audit.", parents=[pre])
    parser.add_argument(
        "--prompt-files",
        nargs="+",
        type=Path,
        default=None,
        help="Prompt JSONL files to audit.",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=DEFAULT_INDEX_DIR,
        help=(
            "Directory of the memory/database being audited (the model's "
            "retrieval index)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where JSONL audit results will be written.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help=(
            "Optional path for a run log file. Defaults to <output-dir>/run_audit.log "
            "when omitted."
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=12,
        help="Maximum number of tokens to generate per prompt.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of prompts to run per file.",
    )
    parser.add_argument(
        "--bootstrap-oracle-from-full",
        action="store_true",
        help=(
            "For schema-free rows without a manifest, use a FULL selected entry "
            "that passes the support judge as the oracle deletion ID."
        ),
    )
    parser.add_argument(
        "--wandb-activation",
        "--wandb_activation",
        dest="wandb_activation",
        type=str,
        default="off",
        choices=["on", "off"],
        help="Enable or disable Weights & Biases logging.",
    )

    closure = parser.add_argument_group("deletion closure / interventions")
    closure.add_argument(
        "--closure",
        type=str,
        default=None,
        help=(
            "Comma-separated deletion-closure predicates materialized from the "
            "FULL pass: any of geometric, semantic, provenance."
        ),
    )
    closure.add_argument(
        "--closure-radius",
        type=float,
        default=0.85,
        help="Cosine radius for the geometric closure predicate.",
    )
    closure.add_argument(
        "--closure-envelope-k",
        type=int,
        default=500,
        help="Candidates fetched for the semantic/provenance closure envelope.",
    )
    closure.add_argument(
        "--closure-max-size",
        type=int,
        default=10_000,
        help=(
            "Maximum entries fetched for the geometric predicate; closures "
            "that hit this cap are flagged as truncated."
        ),
    )
    closure.add_argument(
        "--radius-grid",
        type=str,
        default=None,
        help=(
            "Descending cosine-radius grid 'start:stop:step' (e.g. "
            "0.95:0.70:0.05). Switches the run into an entanglement sweep; "
            "requires --closure."
        ),
    )
    closure.add_argument(
        "--neighbor-mode",
        choices=["cosine", "same-source"],
        default="cosine",
        help="How N(f) is defined for the entanglement sweep.",
    )
    closure.add_argument(
        "--neighbor-ball",
        type=float,
        default=0.5,
        help="Cosine ball for --neighbor-mode cosine.",
    )
    closure.add_argument(
        "--neighbor-cap",
        type=int,
        default=20,
        help="Maximum neighbors per fact in the entanglement sweep.",
    )
    closure.add_argument(
        "--adversarial",
        action="store_true",
        help=(
            "Run the adversarial-closure evaluation (Ev and the margin "
            "predictor) instead of a standard audit. Requires --closure."
        ),
    )
    closure.add_argument(
        "--adversarial-epsilons",
        type=str,
        default="0.01,0.02,0.05",
        help="Comma-separated epsilon offsets below rho for survivor keys.",
    )
    closure.add_argument(
        "--adversarial-templates",
        type=str,
        default="verbatim,hyphenated,letter-spaced,prefix-cue",
        help=(
            "Comma-separated survivor value templates; 'verbatim' is the "
            "non-evading control."
        ),
    )
    closure.add_argument(
        "--adversarial-topology",
        choices=["single", "aliased", "collided", "saturated"],
        default="single",
        help="Injection topology for the adversarial evaluation.",
    )
    closure.add_argument(
        "--adversarial-count",
        type=int,
        default=3,
        help="Survivors/decoys per injection for multi-entry topologies.",
    )
    closure.add_argument(
        "--adversarial-seed",
        type=int,
        default=0,
        help="Seed for survivor key directions.",
    )

    # The selected model contributes its own artifact paths and knobs.
    get_backend_spec(backend).add_arguments(parser)
    return parser.parse_args(argv)


def parse_radius_grid(spec: str) -> tuple[float, ...]:
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError(f"--radius-grid must be 'start:stop:step', got {spec!r}.")
    start, stop, step = (float(part) for part in parts)
    if step <= 0:
        raise ValueError("--radius-grid step must be positive.")
    if start < stop:
        raise ValueError("--radius-grid must descend (start >= stop).")
    radii: list[float] = []
    radius = start
    while radius >= stop - 1e-9:
        radii.append(round(radius, 6))
        radius -= step
    return tuple(radii)
