from collections import defaultdict
from typing import Any

from halo.cli.args import parse_args, parse_radius_grid
from halo.cli.closure_setup import (
    closure_config_from_args,
    make_closure_manifest_builder,
)
from halo.cli.full_pass import FullPassStore
from halo.cli.jobs import AuditJob, resolve_audit_jobs
from halo.cli.persistence import (
    AUDIT_SCHEMA_VERSION,
    backend_resume_identity,
    ensure_resume_config,
    parse_shard,
    prompt_digest,
)
from halo.cli.reuse import GenerationReuseStore, ReuseContext
from halo.core.embeddings import QueryEmbeddingSink
from halo.core.metrics import metrics_total
from halo.registry import get_backend_spec
from halo.cli.reporting import (
    AuditLogger,
    save_results,
    write_adversarial_outputs,
    write_entanglement_outputs,
    write_metrics_csvs,
)
from halo.cli.runner import (
    StandardAuditPersistence,
    _load_examples,
    run_adversarial_eval,
    run_backend_audit,
    run_entanglement_sweep,
)
from halo.core.states import DatabaseState


def _parse_sweep_shard_radii(
    spec: str | None, radii: tuple[float, ...]
) -> tuple[float, ...] | None:
    """Resolve --sweep-shard-radii against the full grid: 'none' materializes
    only, 'I/N' takes a round-robin stripe, a comma list names grid members."""
    if spec is None:
        return None
    spec = spec.strip()
    if spec == "none":
        return ()
    if "/" in spec:
        index, count = parse_shard(spec)
        return tuple(radii[index::count])
    members = tuple(round(float(value), 6) for value in spec.split(",") if value.strip())
    unknown = [rho for rho in members if rho not in radii]
    if unknown:
        raise ValueError(
            f"--sweep-shard-radii members {unknown} are not in the radius "
            f"grid {list(radii)}."
        )
    return members


def _build_reuse_store(
    args: Any,
    backend: Any,
    job: AuditJob,
    reuse_paths: tuple,
    output_dir: Any,
    logger: AuditLogger,
) -> GenerationReuseStore:
    return GenerationReuseStore(
        backend=backend,
        context=ReuseContext(
            backend_identity=backend_resume_identity(backend),
            prompt_sha256=prompt_digest(job.prompt_path),
            limit=args.limit,
            max_new_tokens=args.max_new_tokens,
            del_off_mode=getattr(backend, "del_off_mode", None),
        ),
        source_paths=reuse_paths,
        canary_rate=args.reuse_canary_rate,
        max_new_tokens=args.max_new_tokens,
        output_dir=output_dir,
        log=logger.print,
    )


def main() -> None:
    args = parse_args()
    log_path = args.log_file or (args.output_dir / "run_audit.log")
    logger = AuditLogger(log_path)

    try:
        logger.print(f"Logging run_audit output to {log_path}")

        spec = get_backend_spec(args.backend)
        # Backend-specific argument checks (missing paths, unsupported
        # predicates) live with each model; the audit CLI only validates its
        # own generic flags below.
        spec.validate(args)

        if args.closure is not None:
            if spec.supports_oracle_bootstrap and (
                not args.bootstrap_oracle_from_full
                and args.radius_grid is None
                and not args.adversarial
            ):
                raise ValueError(
                    "--closure builds its manifest from the FULL pass and "
                    "requires --bootstrap-oracle-from-full."
                )
            if not spec.supports_oracle_bootstrap and (
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
            predicates = closure_config_from_args(args).predicates
            if predicates != ("geometric",):
                raise ValueError(
                    "--radius-grid must isolate the geometric predicate; "
                    "value/provenance members are radius-independent and "
                    "flatten the operating curve. Pass --closure geometric."
                )

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
            if "geometric" not in closure_config_from_args(args).predicates:
                raise ValueError(
                    "--adversarial places survivors relative to a geometric "
                    "radius and therefore requires geometric in --closure."
                )

        shard = parse_shard(args.shard) if args.shard is not None else None
        reuse_paths = tuple(args.reuse_from or ())
        if shard is not None and args.radius_grid is not None:
            raise ValueError(
                "--shard stripes facts and is incompatible with "
                "--radius-grid; use --sweep-shard-radii to shard the sweep "
                "by radius."
            )
        if args.sweep_shard_radii is not None and args.radius_grid is None:
            raise ValueError("--sweep-shard-radii requires --radius-grid.")

        jobs = resolve_audit_jobs(args)
        if not jobs:
            raise FileNotFoundError(
                "No audit jobs found. Add custom prompts under data/custom_databases or "
                "pass --prompt-files explicitly."
            )

        # The audit is the three-way comparison; always run all states.
        states = list(DatabaseState)

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

                # All evaluation modes can share one FULL pass per prompt
                # file; --full-dir overrides the default location.
                shared_full_dir = args.full_dir or (
                    job.output_path.parent / f"{job.prompt_path.stem}_full"
                )

                if args.adversarial:
                    from halo.interventions.adversary import AdversarialConfig

                    adversarial_dir = (
                        job.output_path.parent / f"{job.prompt_path.stem}_adversarial"
                    )
                    adversarial_reuse = (
                        _build_reuse_store(
                            args, backend, job, reuse_paths, adversarial_dir, logger
                        )
                        if reuse_paths
                        else None
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
                        full_dir=shared_full_dir,
                        reuse_store=adversarial_reuse,
                        shard=shard,
                    )
                    if summary["partial"]:
                        logger.print(
                            f"Adversarial shard {shard[0]}/{shard[1]} complete "
                            f"({summary['executed_generations']} generations, "
                            f"{summary['reused_del_off']} del-off reused); run "
                            "without --shard to merge and finalize."
                        )
                        continue
                    outputs = write_adversarial_outputs(summary, adversarial_dir)
                    logger.print(
                        f"Adversarial: {summary['attacked_facts']}/"
                        f"{summary['facts']} facts at rho={summary['rho']}, "
                        f"topology={summary['topology']} "
                        f"({summary['executed_generations']} generations, "
                        f"{summary['reused_del_off']} del-off reused)."
                    )
                    if adversarial_reuse is not None:
                        adversarial_reuse.write_manifest(
                            adversarial_dir / "reuse_manifest.json"
                        )
                    if summary["skipped_facts"]:
                        logger.print(
                            "Skipped facts outside the strict primary cohort: "
                            + ", ".join(summary["skipped_facts"])
                        )
                        for reason, count in summary.get(
                            "skipped_by_reason", {}
                        ).items():
                            logger.print(f"  {count}x {reason}")
                    for row in summary["evasion"]:
                        rate = row["evasion_rate"]
                        gain = row["attack_gain_rate"]
                        selected = row["target_selected_rate"]
                        logger.print(
                            f"  Attack(rho={row['rho']}, eps={row['epsilon']}, "
                            f"{row['template']}): "
                            + (f"post-correct={rate:.3f}" if rate is not None else "n/a")
                            + (f", gain={gain:.3f}" if gain is not None else ", gain=n/a")
                            + (
                                f", selected={selected:.3f}"
                                if selected is not None
                                else ", selected=n/a"
                            )
                            + (
                                f", gain|selected={row['gain_given_target_selected']:.3f}"
                                if row["gain_given_target_selected"] is not None
                                else ", gain|selected=n/a"
                            )
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
                    continue

                if args.radius_grid is not None:
                    from halo.core.neighbors import NeighborConfig

                    sweep_dir = job.output_path.parent / f"{job.prompt_path.stem}_sweep"
                    radii = parse_radius_grid(args.radius_grid)
                    summary = run_entanglement_sweep(
                        prompt_path=job.prompt_path,
                        backend=backend,
                        index=search_index,
                        radii=radii,
                        execute_radii=_parse_sweep_shard_radii(
                            args.sweep_shard_radii, radii
                        ),
                        closure_config=closure_config_from_args(args),
                        neighbor_config=NeighborConfig(
                            mode=args.neighbor_mode,
                            ball=args.neighbor_ball,
                            cap=args.neighbor_cap,
                            min_count=args.neighbor_min_count,
                        ),
                        output_dir=sweep_dir,
                        max_new_tokens=args.max_new_tokens,
                        limit=args.limit,
                        full_dir=shared_full_dir,
                        reuse_canary_rate=args.reuse_canary_rate,
                    )
                    outputs = write_entanglement_outputs(
                        summary["entanglement"], sweep_dir
                    )
                    logger.print(
                        f"Sweep: {summary['swept_facts']}/{summary['facts']} "
                        f"facts over {len(summary['radii'])} radii "
                        f"({summary['executed_generations']} generations, "
                        f"{summary['reused_generations']} reused "
                        f"[{summary['reused_fingerprint']} fingerprint, "
                        f"{summary['reused_full_pass']} full-pass, "
                        f"{summary['canary_checks']} canary-verified], "
                        f"{summary['planned_generations']} planned)."
                    )
                    if summary["skipped_facts"]:
                        logger.print(
                            "Skipped facts outside the strict primary cohort: "
                            + ", ".join(summary["skipped_facts"])
                        )
                        for reason, count in summary.get(
                            "skipped_by_reason", {}
                        ).items():
                            logger.print(f"  {count}x {reason}")
                    gaps = [
                        item["gap"]
                        for item in summary["entanglement"].values()
                        if item.get("gap") is not None
                    ]
                    if gaps:
                        logger.print(
                            f"G(f): mean {sum(gaps) / len(gaps):.3f}, "
                            f"min {min(gaps):.3f}, max {max(gaps):.3f} "
                            f"over {len(gaps)} facts"
                        )
                    for label, path in outputs.items():
                        logger.print(f"Wrote entanglement {label} to {path}")
                    if summary["partial"]:
                        logger.print(
                            "Partial sweep shard "
                            f"({summary['executed_radii']}) complete; run the "
                            "full grid to merge and compute the analysis."
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
                    if spec.supports_oracle_bootstrap and args.closure is not None
                    else None
                )
                stem = job.prompt_path.stem
                full_store = None
                reuse_store = None
                resume = None
                # Persistence, the shared FULL pass, and cross-phase reuse
                # activate only through their flags; a bare invocation runs
                # the historical in-memory path untouched.
                if args.full_dir or reuse_paths or shard is not None:
                    closure_payload = None
                    if args.closure is not None:
                        closure_config = closure_config_from_args(args)
                        closure_payload = {
                            "predicates": list(closure_config.predicates),
                            "radius": closure_config.radius,
                            "envelope_top_k": closure_config.envelope_top_k,
                            "max_closure_size": closure_config.max_closure_size,
                        }
                    ensure_resume_config(
                        args.output_dir / f"{stem}_audit_config.json",
                        {
                            "audit_schema_version": AUDIT_SCHEMA_VERSION,
                            "mode": "standard-audit",
                            "backend": backend_resume_identity(backend),
                            "del_off_mode": getattr(backend, "del_off_mode", None),
                            "prompt_sha256": prompt_digest(job.prompt_path),
                            "limit": args.limit,
                            "max_new_tokens": args.max_new_tokens,
                            "states": [state.value for state in states],
                            "bootstrap_oracle_from_full": bool(
                                spec.supports_oracle_bootstrap
                                and args.bootstrap_oracle_from_full
                            ),
                            "closure": closure_payload,
                        },
                        legacy_artifacts=(
                            job.output_path,
                            job.output_path.with_name(
                                f"{stem}_query_embeddings.npz"
                            ),
                            job.output_path.with_name(
                                f"{stem}_skipped_facts.jsonl"
                            ),
                        ),
                    )
                    resume = StandardAuditPersistence(
                        output_dir=args.output_dir,
                        stem=stem,
                        expected_states=states,
                        shard=shard,
                    )
                    if args.full_dir is not None:
                        full_store = FullPassStore(
                            backend=backend,
                            examples=_load_examples(job.prompt_path, args.limit),
                            prompt_path=job.prompt_path,
                            limit=args.limit,
                            output_dir=args.full_dir,
                            max_new_tokens=args.max_new_tokens,
                            shard=shard,
                        )
                    if reuse_paths:
                        reuse_store = _build_reuse_store(
                            args, backend, job, reuse_paths, args.output_dir, logger
                        )
                coverage_summary: dict[str, Any] = {}
                results = run_backend_audit(
                    prompt_path=job.prompt_path,
                    backend=backend,
                    states=states,
                    max_new_tokens=args.max_new_tokens,
                    limit=args.limit,
                    bootstrap_oracle_from_full=(
                        spec.supports_oracle_bootstrap
                        and args.bootstrap_oracle_from_full
                    ),
                    embedding_sink=embedding_sink,
                    manifest_builder=manifest_builder,
                    skip_log_path=job.output_path.with_name(
                        f"{job.prompt_path.stem}_skipped_facts.jsonl"
                    ),
                    coverage_summary=coverage_summary,
                    full_store=full_store,
                    reuse_store=reuse_store,
                    resume=resume,
                    shard=shard,
                )

                if shard is not None:
                    if full_store is not None:
                        full_store.close()
                    logger.print(
                        f"Shard {shard[0]}/{shard[1]} complete for {stem}; "
                        "run without --shard to merge and finalize."
                    )
                    continue

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
                if full_store is not None:
                    full_store.finalize()
                    logger.print(
                        f"FULL pass: {full_store.hits} reused, "
                        f"{full_store.generated_count} generated in "
                        f"{full_store.output_dir}"
                    )
                if reuse_store is not None:
                    reuse_store.write_manifest(
                        job.output_path.with_name(f"{stem}_reuse_manifest.json")
                    )
                    reuse_summary = reuse_store.summary()
                    logger.print(
                        "Cross-phase reuse: served "
                        f"{reuse_summary['served']} generations "
                        f"({reuse_summary['served_by_state']}), generated "
                        f"{reuse_summary['generated']}, "
                        f"{reuse_summary['canary_checks']} canary-verified."
                    )
                if resume is not None:
                    # Every canonical artifact is on disk; drop the partials.
                    resume.cleanup()
                total_metrics = metrics_total(results)
                del_off_mode = getattr(backend, "del_off_mode", None)
                closure_policy = (
                    ",".join(closure_config_from_args(args).predicates)
                    if args.closure is not None
                    else "oracle"
                )
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
                        "backend": args.backend,
                        "closure_policy": closure_policy,
                        "closure_radius": (
                            args.closure_radius if args.closure is not None else None
                        ),
                        "del_off_mode": del_off_mode,
                        "source_facts": coverage_summary.get("facts"),
                        "answer_mention_cohort_facts": coverage_summary.get(
                            "audited_facts"
                        ),
                        "coverage_skipped_facts": coverage_summary.get(
                            "skipped_facts"
                        ),
                        **total_metrics,
                    }
                )
                for state in states:
                    per_state_rows.append(
                        {
                            "prompt_file": str(job.prompt_path),
                            "database_path": str(database_path),
                            "backend": args.backend,
                            "closure_policy": closure_policy,
                            "closure_radius": (
                                args.closure_radius
                                if args.closure is not None
                                else None
                            ),
                            "state": state.value,
                            "del_off_mode": del_off_mode,
                            **metrics_by_state[state.value],
                        }
                    )

                logger.print("Cross-state audit metrics:")
                logger.print(f"  Closure policy: {closure_policy}")
                if del_off_mode is not None:
                    logger.print(f"  DEL-OFF control mode: {del_off_mode}")
                if coverage_summary:
                    logger.print(
                        "  Pre-answer answer-mention coverage: "
                        f"{coverage_summary['audited_facts']}/"
                        f"{coverage_summary['facts']} facts"
                    )
                    for reason, skipped_count in coverage_summary.get(
                        "skipped_by_reason", {}
                    ).items():
                        logger.print(f"    {skipped_count}x {reason}")
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
                    "  Retrieval interference I(f): "
                    f"{total_metrics['retrieval_interference']:.3f}"
                )
                logger.print(
                    "  Retrieval interference | FULL correct: "
                    f"{total_metrics['retrieval_interference_given_full']:.3f}"
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

        if shard is None:
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
