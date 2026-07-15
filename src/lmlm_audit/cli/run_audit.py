import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from lmlm_audit.core.backend import AuditBackend
from lmlm_audit.cli.jobs import (
    DEFAULT_DATABASE_PATH,
    DEFAULT_OUTPUT_DIR,
    AuditJob,
    resolve_audit_jobs,
)
from lmlm_audit.core.metrics import metrics_total
from lmlm_audit.rel_lmlm.backend import RelLMLMAuditBackend
from lmlm_audit.cli.reporting import (
    AuditLogger,
    log_metrics_to_wandb,
    save_results,
    setup_wandb,
    write_metrics_csvs,
)
from lmlm_audit.cli.runner import run_backend_audit
from lmlm_audit.core.states import DatabaseState


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the prompt audit.")
    parser.add_argument(
        "--backend",
        choices=["rel-lmlm", "colmlm"],
        default="rel-lmlm",
        help="Inference backend to audit.",
    )
    parser.add_argument(
        "--prompt-files",
        nargs="+",
        type=Path,
        default=None,
        help=(
            "Specific prompt JSONL files to audit. If omitted, run all prompt files for "
            "all custom databases under data/custom_databases."
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
        "--model-name",
        type=str,
        default="kilian-group/LMLM-llama2-382M",
        help="rel-LMLM model name or checkpoint. Use --colmlm-model-path for Co-LMLM.",
    )
    parser.add_argument(
        "--colmlm-model-path",
        type=Path,
        default=None,
        help="Local Co-LMLM checkpoint directory.",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=None,
        help="Local Co-LMLM retrieval-index directory.",
    )
    parser.add_argument(
        "--entries-db-path",
        type=Path,
        default=None,
        help="Optional Co-LMLM entries.db path used to resolve index results.",
    )
    parser.add_argument(
        "--colmlm-source-path",
        type=Path,
        default=None,
        help="Path to a public lil-lab/Co-LMLM checkout (or its src directory).",
    )
    parser.add_argument(
        "--use-sqlite-id-mapping",
        action="store_true",
        help="Use the large index's SQLite FAISS-ID to entry-ID mapping.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device passed to the public Co-LMLM loader.",
    )
    parser.add_argument(
        "--torch-dtype",
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
        help="Model dtype passed to the public Co-LMLM loader.",
    )
    parser.add_argument(
        "--attn-implementation",
        default="flash_attention_2",
        help="Transformers attention implementation for Co-LMLM; use 'none' to omit.",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.7,
        help="Co-LMLM retrieval similarity threshold.",
    )
    parser.add_argument(
        "--retrieval-top-k",
        type=int,
        default=1,
        help="Number of retained Co-LMLM candidates returned to generation.",
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
        "--database-path",
        type=Path,
        default=DEFAULT_DATABASE_PATH,
        help=(
            "Path to a specific database JSON file. When provided without --prompt-files, "
            "run all prompt files in the sibling prompts/<variant>/ directory if present."
        ),
    )
    parser.add_argument(
        "--disable-dblookup",
        action="store_true",
        help="Deprecated shortcut for running only the DEL-OFF state.",
    )
    parser.add_argument(
        "--states",
        nargs="+",
        default=[state.value for state in DatabaseState],
        choices=[state.value for state in DatabaseState],
        help="Database states to evaluate.",
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
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    log_path = args.log_file or (args.output_dir / "run_audit.log")
    logger = AuditLogger(log_path)

    try:
        logger.print(f"Logging run_audit output to {log_path}")

        if args.backend == "colmlm":
            if args.prompt_files is None:
                raise ValueError("Co-LMLM runs require explicit --prompt-files.")
            if args.colmlm_model_path is None:
                raise ValueError("Co-LMLM runs require --colmlm-model-path.")
            if args.index_path is None:
                raise ValueError("Co-LMLM runs require --index-path.")

        jobs = resolve_audit_jobs(args)
        if not jobs:
            raise FileNotFoundError(
                "No audit jobs found. Add custom prompts under data/custom_databases or "
                "pass --prompt-files explicitly."
            )

        state_values = [DatabaseState(state) for state in args.states]
        if args.disable_dblookup:
            state_values = [DatabaseState.DEL_OFF]
        states = state_values
        wandb_module = setup_wandb() if args.wandb_activation == "on" else None

        jobs_by_database: dict[Path, list[AuditJob]] = defaultdict(list)
        if args.backend == "colmlm":
            jobs_by_database[args.index_path].extend(jobs)
        else:
            for job in jobs:
                jobs_by_database[job.database_path].append(job)

        cross_state_rows: list[dict[str, Any]] = []
        per_state_rows: list[dict[str, Any]] = []

        for database_path in sorted(jobs_by_database):
            if args.backend == "colmlm":
                from lmlm_audit.colmlm.backend import CoLMLMAuditBackend

                attn_implementation = (
                    None
                    if args.attn_implementation.casefold() == "none"
                    else args.attn_implementation
                )
                backend: AuditBackend = CoLMLMAuditBackend.from_public_release(
                    model_path=args.colmlm_model_path,
                    index_path=args.index_path,
                    db_path=args.entries_db_path,
                    source_path=args.colmlm_source_path,
                    use_sqlite_id_mapping=args.use_sqlite_id_mapping,
                    device=args.device,
                    torch_dtype=args.torch_dtype,
                    attn_implementation=attn_implementation,
                    max_new_tokens=args.max_new_tokens,
                    similarity_threshold=args.similarity_threshold,
                    retrieval_top_k=args.retrieval_top_k,
                )
            else:
                from lmlm_audit.rel_lmlm.loader import load_model_and_tokenizer

                model, tokenizer = load_model_and_tokenizer(
                    model_name=args.model_name,
                    database_path=database_path,
                )
                backend = RelLMLMAuditBackend(
                    base_db_manager=model.db_manager,
                    model=model,
                    tokenizer=tokenizer,
                )

            for job in jobs_by_database[database_path]:
                logger.print(f"Prompt file: {job.prompt_path}")
                logger.print(f"Database used: {database_path}")
                logger.print("DB states: " + ", ".join(state.value for state in states))
                logger.print(
                    f"Running audit for {job.prompt_path} with database {database_path}"
                )
                results = run_backend_audit(
                    prompt_path=job.prompt_path,
                    backend=backend,
                    states=states,
                    max_new_tokens=args.max_new_tokens,
                    limit=args.limit,
                    bootstrap_oracle_from_full=(
                        args.backend == "colmlm" and args.bootstrap_oracle_from_full
                    ),
                )

                save_results(results, job.output_path)
                total_metrics = metrics_total(results)
                metrics_by_state = {
                    state.value: metrics_total(
                        [result for result in results if result["state"] == state.value]
                    )
                    for state in states
                }

                cross_state_rows.append(
                    {
                        "prompt_file": str(job.prompt_path),
                        "database_path": str(database_path),
                        **total_metrics,
                    }
                )
                for state in states:
                    per_state_rows.append(
                        {
                            "prompt_file": str(job.prompt_path),
                            "database_path": str(database_path),
                            "state": state.value,
                            **metrics_by_state[state.value],
                        }
                    )

                logger.print("Cross-state audit metrics:")
                logger.print(f"  Paired count: {total_metrics['paired_count']}")
                logger.print(
                    "  FULL-correct paired count: "
                    f"{total_metrics['full_correct_paired_count']}"
                )
                logger.print(
                    f"  Parametric leakage L(f): {total_metrics['parametric_leakage']:.3f}"
                )
                logger.print(
                    "  Retrieval-mediated correctness R(f): "
                    f"{total_metrics['retrieval_mediated_correctness']:.3f}"
                )
                logger.print(
                    f"  Retrieval artifact rate: {total_metrics['retrieval_artifact_rate']:.3f}"
                )
                logger.print(
                    "  Artifact-trace eligible count: "
                    f"{total_metrics['retrieval_artifact_eligible_count']}"
                )
                logger.print(
                    "  Post-deletion survival | FULL correct: "
                    f"{total_metrics['post_deletion_survival_given_full']:.3f}"
                )
                logger.print("Metrics by state:")
                for state in states:
                    metrics = metrics_by_state[state.value]
                    logger.print(f"{state.value}:")
                    logger.print(f"  Count: {metrics['count']}")
                    logger.print(f"  Exact match: {metrics['exact_match']:.3f}")
                    logger.print(f"  Contains match: {metrics['contains_match']:.3f}")
                    logger.print(f"  Unknown rate: {metrics['unknown_rate']:.3f}")
                    logger.print(f"  Precision: {metrics['precision']:.3f}")
                    logger.print(f"  Recall: {metrics['recall']:.3f}")
                    logger.print(f"  F1: {metrics['f1']:.3f}")
                    if wandb_module is not None:
                        log_metrics_to_wandb(
                            wandb_module=wandb_module,
                            prompt_path=job.prompt_path,
                            state=state,
                            state_metrics=metrics,
                            cross_state_metrics=total_metrics,
                            model_name=(
                                str(args.colmlm_model_path)
                                if args.backend == "colmlm"
                                else args.model_name
                            ),
                            database_path=database_path,
                            max_new_tokens=args.max_new_tokens,
                            limit=args.limit,
                        )
                        logger.print(f"  W&B run: {job.prompt_path.stem}_{state.value}")

        cross_state_csv_path = args.output_dir / "cross_state_metrics.csv"
        per_state_csv_path = args.output_dir / "per_state_metrics.csv"
        write_metrics_csvs(
            cross_state_rows=cross_state_rows,
            per_state_rows=per_state_rows,
            cross_state_path=cross_state_csv_path,
            per_state_path=per_state_csv_path,
        )
        logger.print(f"Wrote cross-state metrics CSV to {cross_state_csv_path}")
        logger.print(f"Wrote per-state metrics CSV to {per_state_csv_path}")
    finally:
        logger.close()


if __name__ == "__main__":
    main()
