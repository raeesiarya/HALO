from collections import defaultdict
from typing import Any

from halo.cli.args import parse_args, parse_radius_grid
from halo.cli.closure_setup import (
    closure_config_from_args,
    make_closure_manifest_builder,
)
from halo.cli.jobs import AuditJob, resolve_audit_jobs
from halo.core.embeddings import QueryEmbeddingSink
from halo.core.metrics import metrics_total
from halo.registry import get_backend_spec
from halo.cli.reporting import (
    AuditLogger,
    save_results,
    setup_wandb,
    start_wandb_run,
    wandb_log_image,
    wandb_log_metrics,
    wandb_log_output_artifacts,
    write_adversarial_outputs,
    write_entanglement_outputs,
    write_metrics_csvs,
)
from halo.cli.runner import (
    run_adversarial_eval,
    run_backend_audit,
    run_entanglement_sweep,
)
from halo.core.states import DatabaseState


def main() -> None:
    args = parse_args()
    log_path = args.log_file or (args.output_dir / "run_audit.log")
    logger = AuditLogger(log_path)
    run = None

    try:
        logger.print(f"Logging run_audit output to {log_path}")

        spec = get_backend_spec(args.backend)
        # Backend-specific argument checks (missing paths, unsupported
        # predicates) live with each model; the audit CLI only validates its
        # own generic flags below.
        spec.validate(args)

        if args.closure is not None:
            if args.backend == "co-lmlm" and (
                not args.bootstrap_oracle_from_full
                and args.radius_grid is None
                and not args.adversarial
            ):
                raise ValueError(
                    "--closure builds its manifest from the FULL pass and "
                    "requires --bootstrap-oracle-from-full."
                )
            if args.backend != "co-lmlm" and (
                args.radius_grid is None and not args.adversarial
            ):
                raise ValueError(
                    f"--closure with the {args.backend} backend is used "
                    "through --radius-grid or --adversarial."
                )

        if args.radius_grid is not None:
            if args.closure is None:
                raise ValueError(
                    "--radius-grid sweeps the closure radius and requires --closure."
                )
            parse_radius_grid(args.radius_grid)

        if args.adversarial:
            if args.closure is None:
                raise ValueError(
                    "--adversarial needs a deletion closure; pass --closure."
                )
            if args.radius_grid is not None:
                raise ValueError(
                    "--adversarial and --radius-grid are separate evaluation "
                    "modes; run them individually."
                )

        jobs = resolve_audit_jobs(args)
        if not jobs:
            raise FileNotFoundError(
                "No audit jobs found. Add custom prompts under data/custom_databases or "
                "pass --prompt-files explicitly."
            )

        # The audit is the three-way comparison; always run all states.
        states = list(DatabaseState)
        wandb_module = setup_wandb() if args.wandb_activation == "on" else None
        if wandb_module is not None:
            mode = (
                "adversarial"
                if args.adversarial
                else "sweep"
                if args.radius_grid is not None
                else "standard"
            )
            run = start_wandb_run(
                wandb_module,
                name=str(args.output_dir).replace("/", "__"),
                config={
                    "backend": args.backend,
                    "mode": mode,
                    "index_path": str(args.index_path),
                    "prompt_files": [str(p) for p in (args.prompt_files or [])],
                    "max_new_tokens": args.max_new_tokens,
                    "limit": args.limit,
                    "closure": args.closure,
                    "closure_radius": args.closure_radius,
                    "radius_grid": args.radius_grid,
                    "neighbor_mode": args.neighbor_mode,
                    "adversarial_topology": args.adversarial_topology,
                    "bootstrap_oracle_from_full": args.bootstrap_oracle_from_full,
                },
            )

        jobs_by_group: dict[Any, list[AuditJob]] = defaultdict(list)
        for job in jobs:
            jobs_by_group[spec.group_key(args, job)].append(job)

        cross_state_rows: list[dict[str, Any]] = []
        per_state_rows: list[dict[str, Any]] = []

        for database_path in sorted(jobs_by_group, key=str):
            backend = spec.build_backend(args, database_path)
            search_index = spec.build_search_index(backend)

            for job in jobs_by_group[database_path]:
                logger.print(f"Prompt file: {job.prompt_path}")
                logger.print(f"Database used: {database_path}")

                if args.adversarial:
                    from halo.interventions.adversary import AdversarialConfig

                    adversarial_dir = (
                        job.output_path.parent / f"{job.prompt_path.stem}_adversarial"
                    )
                    summary = run_adversarial_eval(
                        prompt_path=job.prompt_path,
                        backend=backend,
                        index=search_index,
                        closure_config=closure_config_from_args(args),
                        adversarial_config=AdversarialConfig(
                            rho=args.closure_radius,
                            epsilons=tuple(
                                float(value)
                                for value in args.adversarial_epsilons.split(",")
                                if value.strip()
                            ),
                            templates=tuple(
                                value.strip()
                                for value in args.adversarial_templates.split(",")
                                if value.strip()
                            ),
                            topology=args.adversarial_topology,
                            count=args.adversarial_count,
                            seed=args.adversarial_seed,
                        ),
                        output_dir=adversarial_dir,
                        max_new_tokens=args.max_new_tokens,
                        limit=args.limit,
                    )
                    outputs = write_adversarial_outputs(summary, adversarial_dir)
                    logger.print(
                        f"Adversarial: {summary['attacked_facts']}/"
                        f"{summary['facts']} facts at rho={summary['rho']}, "
                        f"topology={summary['topology']} "
                        f"({summary['executed_generations']} generations)."
                    )
                    if summary["skipped_facts"]:
                        logger.print(
                            "Skipped facts (no FULL selection/embedding): "
                            + ", ".join(summary["skipped_facts"])
                        )
                    for row in summary["evasion"]:
                        rate = row["evasion_rate"]
                        logger.print(
                            f"  Ev(rho={row['rho']}, eps={row['epsilon']}, "
                            f"{row['template']}): "
                            + (f"{rate:.3f}" if rate is not None else "n/a")
                            + f" over {row['facts']} facts"
                        )
                    if summary["margin_auroc"] is not None:
                        logger.print(
                            "  Margin-predictor AUROC (survivor proximity "
                            f"vs R(f)): {summary['margin_auroc']:.3f} over "
                            f"{summary['margin_auroc_facts']} facts"
                        )
                    else:
                        logger.print(
                            "  Margin-predictor AUROC: n/a (R(f) has a "
                            "single class or no scored facts)"
                        )
                    for label, path in outputs.items():
                        logger.print(f"Wrote adversarial {label} to {path}")
                    if run is not None:
                        stem = job.prompt_path.stem
                        for row in summary["evasion"]:
                            if row["evasion_rate"] is not None:
                                run.log(
                                    {
                                        f"{stem}/evasion/eps{row['epsilon']}"
                                        f"_{row['template']}": row["evasion_rate"]
                                    }
                                )
                        wandb_log_metrics(
                            run,
                            {
                                "margin_auroc": summary["margin_auroc"],
                                "margin_auroc_facts": summary["margin_auroc_facts"],
                                "attacked_facts": summary["attacked_facts"],
                                "executed_generations": summary["executed_generations"],
                            },
                            prefix=f"{stem}/adversarial/",
                        )
                    continue

                if args.radius_grid is not None:
                    from halo.core.neighbors import NeighborConfig

                    sweep_dir = job.output_path.parent / f"{job.prompt_path.stem}_sweep"
                    summary = run_entanglement_sweep(
                        prompt_path=job.prompt_path,
                        backend=backend,
                        index=search_index,
                        radii=parse_radius_grid(args.radius_grid),
                        closure_config=closure_config_from_args(args),
                        neighbor_config=NeighborConfig(
                            mode=args.neighbor_mode,
                            ball=args.neighbor_ball,
                            cap=args.neighbor_cap,
                        ),
                        output_dir=sweep_dir,
                        max_new_tokens=args.max_new_tokens,
                        limit=args.limit,
                    )
                    outputs = write_entanglement_outputs(
                        summary["entanglement"], sweep_dir
                    )
                    logger.print(
                        f"Sweep: {summary['swept_facts']}/{summary['facts']} "
                        f"facts over {len(summary['radii'])} radii "
                        f"({summary['executed_generations']} generations, "
                        f"{summary['planned_generations']} planned)."
                    )
                    if summary["skipped_facts"]:
                        logger.print(
                            "Skipped facts (no FULL selection/embedding): "
                            + ", ".join(summary["skipped_facts"])
                        )
                    gaps = [item["gap"] for item in summary["entanglement"].values()]
                    if gaps:
                        logger.print(
                            f"G(f): mean {sum(gaps) / len(gaps):.3f}, "
                            f"min {min(gaps):.3f}, max {max(gaps):.3f} "
                            f"over {len(gaps)} facts"
                        )
                    for label, path in outputs.items():
                        logger.print(f"Wrote entanglement {label} to {path}")
                    if run is not None:
                        stem = job.prompt_path.stem
                        if gaps:
                            wandb_log_metrics(
                                run,
                                {
                                    "g_mean": sum(gaps) / len(gaps),
                                    "g_min": min(gaps),
                                    "g_max": max(gaps),
                                    "facts": len(gaps),
                                    "swept_facts": summary["swept_facts"],
                                    "executed_generations": summary["executed_generations"],
                                },
                                prefix=f"{stem}/entanglement/",
                            )
                        wandb_log_image(
                            run,
                            wandb_module,
                            outputs.get("figure"),
                            f"{stem}/entanglement_curves",
                        )
                    continue

                logger.print("DB states: " + ", ".join(state.value for state in states))
                logger.print(
                    f"Running audit for {job.prompt_path} with database {database_path}"
                )
                # Both backends capture query embeddings now (Co-LMLM: the
                # <FACT> hidden state; rel-LMLM: the encoded lookup text).
                embedding_sink = QueryEmbeddingSink()
                manifest_builder = (
                    make_closure_manifest_builder(backend, search_index, args, job)
                    if args.backend == "co-lmlm" and args.closure is not None
                    else None
                )
                results = run_backend_audit(
                    prompt_path=job.prompt_path,
                    backend=backend,
                    states=states,
                    max_new_tokens=args.max_new_tokens,
                    limit=args.limit,
                    bootstrap_oracle_from_full=(
                        args.backend == "co-lmlm" and args.bootstrap_oracle_from_full
                    ),
                    embedding_sink=embedding_sink,
                    manifest_builder=manifest_builder,
                )

                save_results(results, job.output_path)
                probe_summary: dict[str, Any] | None = None
                if manifest_builder is not None:
                    logger.print(
                        "Wrote closure artifacts to "
                        f"{job.output_path.parent / f'{job.prompt_path.stem}_closures'}"
                    )
                if embedding_sink is not None and len(embedding_sink):
                    sidecar_path = job.output_path.with_name(
                        f"{job.prompt_path.stem}_query_embeddings.npz"
                    )
                    embedding_sink.save(sidecar_path)
                    logger.print(f"Wrote query-embedding sidecar to {sidecar_path}")

                    # Representational-leakage probe on this job's FULL
                    # embeddings — runs automatically, skips when too few facts.
                    from halo.core.probe import probe_audit_outputs

                    probe_summary = probe_audit_outputs(
                        results_paths=[job.output_path],
                        embeddings_paths=[sidecar_path],
                        output_dir=job.output_path.parent,
                        stem=job.prompt_path.stem,
                    )
                    if probe_summary is None:
                        logger.print(
                            "Probe skipped (too few labeled facts with embeddings)."
                        )
                    else:
                        delta = probe_summary["delta_rep"]
                        logger.print(
                            f"Probe L_rep: {probe_summary['l_rep_hat']:.3f}"
                            + (
                                f", behavioral L: {probe_summary['l_hat']:.3f}, "
                                f"Δ_rep: {delta:.3f} over "
                                f"{probe_summary['facts_common']} facts"
                                if delta is not None
                                else " (Δ_rep n/a: no DEL-OFF overlap)"
                            )
                        )
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

                if run is not None:
                    stem = job.prompt_path.stem
                    wandb_log_metrics(run, total_metrics, prefix=f"{stem}/cross_state/")
                    for state in states:
                        wandb_log_metrics(
                            run,
                            metrics_by_state[state.value],
                            prefix=f"{stem}/{state.value}/",
                        )
                    if probe_summary is not None:
                        wandb_log_metrics(
                            run, probe_summary, prefix=f"{stem}/probe/"
                        )

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

        if run is not None:
            # Upload every output file (results, all CSVs, plots, sidecars).
            wandb_log_output_artifacts(run, wandb_module, args.output_dir)
            logger.print("Uploaded output artifacts to W&B.")
    finally:
        if run is not None:
            run.finish()
        logger.close()


if __name__ == "__main__":
    main()
