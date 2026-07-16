import dataclasses
import json
import zlib
from pathlib import Path
from typing import Any, Callable

import numpy as np
from tqdm import tqdm

from halo.core.backend import (
    AuditBackend,
    audit_example,
    validate_intervention_results,
)
from halo.core.embeddings import QueryEmbeddingSink, result_example_key
from halo.core.entanglement import compute_entanglement, fact_key
from halo.core.examples import AuditExample, DeletionManifest
from halo.core.neighbors import (
    NeighborConfig,
    compute_cosine_neighbors,
    compute_same_source_neighbors,
    neighbor_keys,
    write_neighbors_file,
)
from halo.core.states import DatabaseState


def load_prompts(prompts_path: Path) -> list[dict[str, Any]]:
    with prompts_path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_backend_prompt_audit(
    backend: AuditBackend,
    prompt_row: dict[str, Any],
    state: DatabaseState,
    max_new_tokens: int = 12,
) -> dict[str, Any]:
    return audit_example(
        backend=backend,
        example=AuditExample.from_prompt_row(prompt_row),
        state=state,
        max_new_tokens=max_new_tokens,
    )


def run_backend_audit(
    prompt_path: Path,
    backend: AuditBackend,
    states: list[DatabaseState],
    max_new_tokens: int = 12,
    limit: int | None = None,
    bootstrap_oracle_from_full: bool = False,
    embedding_sink: QueryEmbeddingSink | None = None,
    manifest_builder: (
        Callable[[AuditExample, dict[str, Any]], DeletionManifest] | None
    ) = None,
) -> list[dict[str, Any]]:
    if not states:
        raise ValueError("At least one audit state is required.")
    if len(states) != len(set(states)):
        raise ValueError("Audit states must not contain duplicates.")

    prompts = load_prompts(prompt_path)
    if limit is not None:
        prompts = prompts[:limit]

    results: list[dict[str, Any]] = []
    for row_index, prompt in enumerate(
        tqdm(
            prompts,
            desc=f"Auditing {prompt_path.stem}",
            unit="prompt",
        )
    ):
        example = AuditExample.from_prompt_row(prompt)
        prompt_results: list[dict[str, Any]] = []
        remaining_states = list(states)

        if bootstrap_oracle_from_full and example.deletion_manifest.is_empty:
            if not remaining_states or remaining_states[0] is not DatabaseState.FULL:
                raise ValueError(
                    "Oracle bootstrapping requires FULL to be the first requested state."
                )
            full_result = audit_example(
                backend=backend,
                example=example,
                state=DatabaseState.FULL,
                max_new_tokens=max_new_tokens,
            )
            selected = (full_result.get("retrieval_trace") or {}).get(
                "selected_candidate"
            ) or {}
            entry_id = selected.get("entry_id")
            if not entry_id:
                raise ValueError(
                    "FULL produced no selected entry ID; cannot bootstrap an oracle manifest."
                )
            if selected.get("supports_target") is not True:
                raise ValueError(
                    "FULL's selected entry did not pass the configured target-support "
                    "judge; supply a reviewed deletion manifest manually."
                )
            if manifest_builder is not None:
                manifest = manifest_builder(example, full_result)
                if manifest.is_empty:
                    raise ValueError(
                        "The manifest builder produced an empty deletion manifest."
                    )
            else:
                manifest = DeletionManifest(
                    entry_ids=(str(entry_id),),
                    strategy="oracle-from-full",
                    metadata={"bootstrap": "FULL.selected_candidate"},
                )
            example = dataclasses.replace(example, deletion_manifest=manifest)
            full_result["deletion_manifest"] = manifest.as_dict()
            full_result["retrieval_trace"]["deletion_manifest_id"] = (
                manifest.manifest_id
            )
            prompt_results.append(full_result)
            remaining_states = remaining_states[1:]

        for state in remaining_states:
            prompt_results.append(
                audit_example(
                    backend=backend,
                    example=example,
                    state=state,
                    max_new_tokens=max_new_tokens,
                )
            )
        validate_intervention_results(prompt_results, expected_states=states)
        for result in prompt_results:
            # Numpy arrays must never reach the JSONL writer; route them to
            # the sidecar (or drop them when no sink is configured).
            embeddings = result.pop("_query_embeddings", None)
            if embedding_sink is None or not embeddings:
                continue
            key = result_example_key(result, row_index)
            for item in embeddings:
                embedding_sink.add(
                    example_key=key,
                    state=str(result["state"]),
                    event_index=int(item["event_index"]),
                    vector=item["vector"],
                )
        results.extend(prompt_results)

    return results


def _load_examples(prompt_path: Path, limit: int | None) -> dict[str, AuditExample]:
    prompts = load_prompts(prompt_path)
    if limit is not None:
        prompts = prompts[:limit]
    examples: dict[str, AuditExample] = {}
    for row_index, prompt in enumerate(prompts):
        example = AuditExample.from_prompt_row(prompt)
        key = result_example_key(
            {"prompt_id": example.prompt_id, "fact_id": example.fact_id},
            row_index,
        )
        if key in examples:
            raise ValueError(f"Duplicate fact key {key!r} in {prompt_path}.")
        examples[key] = example
    return examples


def _full_pass(
    backend: AuditBackend,
    examples: dict[str, AuditExample],
    output_dir: Path,
    max_new_tokens: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray]]:
    """FULL over every prompt, capturing query embeddings. Resumed wholesale
    when both artifacts from a previous run exist."""
    from halo.interventions.closure import full_query_vector

    full_rows_path = output_dir / "full_results.jsonl"
    embeddings_path = output_dir / "full_query_embeddings.npz"
    full_rows: dict[str, dict[str, Any]] = {}
    vectors: dict[str, np.ndarray] = {}
    if full_rows_path.exists() and embeddings_path.exists():
        for row in load_prompts(full_rows_path):
            full_rows[fact_key(row)] = row
        with np.load(embeddings_path) as stored:
            vectors = {key: stored[key] for key in stored.files}
        return full_rows, vectors

    for key, example in tqdm(examples.items(), desc="FULL pass", unit="prompt"):
        row = audit_example(
            backend,
            example,
            DatabaseState.FULL,
            max_new_tokens=max_new_tokens,
        )
        vector = full_query_vector(row)
        row.pop("_query_embeddings", None)
        full_rows[key] = row
        if vector is not None:
            vectors[key] = np.asarray(vector, dtype=np.float32)
    with full_rows_path.open("w", encoding="utf-8") as handle:
        for row in full_rows.values():
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if vectors:
        np.savez_compressed(embeddings_path, **vectors)
    return full_rows, vectors


def run_entanglement_sweep(
    prompt_path: Path,
    backend: AuditBackend,
    *,
    index: Any,
    radii: tuple[float, ...],
    closure_config: Any,
    neighbor_config: NeighborConfig,
    output_dir: Path,
    max_new_tokens: int = 12,
    limit: int | None = None,
) -> dict[str, Any]:
    """Radius sweep for the entanglement analysis (E, X, G).

    Pass 1 runs FULL once over all prompts (capturing query embeddings and
    FULL-correctness); closures for every radius come from one search per
    fact; then each (fact, radius) runs the target prompt and all neighbor
    prompts under DEL-ON. Per-radius JSONL files make the sweep resumable:
    (target, role, subject) triples already on disk are skipped.
    """
    from halo.interventions.closure import (
        build_closure_family,
        full_query_vector,
        full_selected_candidate,
    )

    if not radii:
        raise ValueError("A radius sweep requires at least one radius.")
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = _load_examples(prompt_path, limit)
    full_rows, vectors = _full_pass(backend, examples, output_dir, max_new_tokens)

    # Closure families: one geometric search per fact covers every radius.
    families: dict[str, dict[float, Any]] = {}
    skipped: list[str] = []
    judge = getattr(backend, "support_judge", None)
    for key, example in examples.items():
        selected = full_selected_candidate(full_rows.get(key, {}))
        vector = vectors.get(key)
        if not selected or not selected.get("entry_id") or vector is None:
            skipped.append(key)
            continue
        seed_source = selected.get("source_id")
        family_kwargs: dict[str, Any] = {}
        if judge is not None:
            family_kwargs["support_judge"] = judge
        families[key] = build_closure_family(
            index=index,
            example=example,
            query_vector=vector,
            config=closure_config,
            radii=radii,
            seed_candidates=(selected,),
            seed_source_ids=((str(seed_source),) if seed_source is not None else ()),
            example_key=key,
            **family_kwargs,
        )

    # Neighbor sets over the facts that survived the FULL pass.
    if neighbor_config.mode == "cosine":
        raw_neighbors = compute_cosine_neighbors(
            {key: vectors[key] for key in families}, neighbor_config
        )
    else:
        sources = {
            key: (full_selected_candidate(full_rows[key]) or {}).get("source_id")
            for key in families
        }
        raw_neighbors = compute_same_source_neighbors(sources, neighbor_config)
    write_neighbors_file(raw_neighbors, neighbor_config, output_dir / "neighbors.json")
    neighbors = neighbor_keys(raw_neighbors)

    planned = sum(len(radii) * (1 + len(neighbors.get(key, []))) for key in families)
    executed = 0
    sweep_rows: list[dict[str, Any]] = []
    progress = tqdm(
        total=planned, desc=f"Sweeping {prompt_path.stem}", unit="generation"
    )
    for rho in radii:
        rho_path = output_dir / f"sweep_rho_{rho:.4f}.jsonl"
        done: set[tuple[str, str, str]] = set()
        if rho_path.exists():
            for row in load_prompts(rho_path):
                tag = row.get("sweep") or {}
                done.add(
                    (str(tag.get("target_key")), str(tag.get("role")), fact_key(row))
                )
                sweep_rows.append(row)
        with rho_path.open("a", encoding="utf-8") as handle:
            for key, family in families.items():
                manifest = family[rho].to_manifest()
                jobs = [("target", key)] + [
                    ("neighbor", neighbor_key)
                    for neighbor_key in neighbors.get(key, [])
                    if neighbor_key in examples
                ]
                for role, subject_key in jobs:
                    if (key, role, subject_key) in done:
                        progress.update(1)
                        continue
                    subject = dataclasses.replace(
                        examples[subject_key], deletion_manifest=manifest
                    )
                    row = audit_example(
                        backend,
                        subject,
                        DatabaseState.DEL_ON,
                        max_new_tokens=max_new_tokens,
                    )
                    row.pop("_query_embeddings", None)
                    row["sweep"] = {
                        "target_key": key,
                        "rho": rho,
                        "role": role,
                    }
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    sweep_rows.append(row)
                    executed += 1
                    progress.update(1)
    progress.close()

    entanglement = compute_entanglement(sweep_rows, list(full_rows.values()), neighbors)
    return {
        "prompt_file": str(prompt_path),
        "facts": len(examples),
        "swept_facts": len(families),
        "skipped_facts": skipped,
        "radii": list(radii),
        "planned_generations": planned,
        "executed_generations": executed,
        "entanglement": entanglement,
        "output_dir": str(output_dir),
    }


def run_adversarial_eval(
    prompt_path: Path,
    backend: Any,
    *,
    index: Any,
    closure_config: Any,
    adversarial_config: Any,
    output_dir: Path,
    max_new_tokens: int = 12,
    limit: int | None = None,
) -> dict[str, Any]:
    """Adversarial-closure evaluation: Ev(rho, epsilon) and the geometry-only
    margin predictor.

    Per fact: FULL -> closure at rho -> DEL-OFF and baseline DEL-ON rows
    (yielding R(f)) -> one injected DEL-ON per (epsilon, template). Rows are
    appended to a resumable JSONL keyed by (fact, role, epsilon, template).
    """
    from halo.interventions.adversary import build_injections
    from halo.interventions.closure import (
        build_closure_family,
        full_selected_candidate,
        write_closure_artifact,
    )
    from halo.core.metrics import _result_is_correct, auroc

    config = adversarial_config
    output_dir.mkdir(parents=True, exist_ok=True)
    examples = _load_examples(prompt_path, limit)
    full_rows, vectors = _full_pass(backend, examples, output_dir, max_new_tokens)

    closures: dict[str, Any] = {}
    skipped: list[str] = []
    judge = getattr(backend, "support_judge", None)
    for key, example in examples.items():
        selected = full_selected_candidate(full_rows.get(key, {}))
        vector = vectors.get(key)
        if not selected or not selected.get("entry_id") or vector is None:
            skipped.append(key)
            continue
        seed_source = selected.get("source_id")
        family_kwargs: dict[str, Any] = {}
        if judge is not None:
            family_kwargs["support_judge"] = judge
        closure = build_closure_family(
            index=index,
            example=example,
            query_vector=vector,
            config=closure_config,
            radii=(config.rho,),
            seed_candidates=(selected,),
            seed_source_ids=((str(seed_source),) if seed_source is not None else ()),
            example_key=key,
            **family_kwargs,
        )[config.rho]
        closures[key] = closure
        write_closure_artifact(closure, output_dir / "closures" / f"{key}.json")

    rows_path = output_dir / "adversarial_results.jsonl"
    done: set[tuple[str, str, str, str]] = set()
    rows: list[dict[str, Any]] = []
    if rows_path.exists():
        for row in load_prompts(rows_path):
            tag = row.get("adversarial") or {}
            done.add(
                (
                    str(tag.get("target_key")),
                    str(tag.get("role")),
                    str(tag.get("epsilon")),
                    str(tag.get("template")),
                )
            )
            rows.append(row)

    jobs: list[tuple[str, str, float | None, str | None]] = []
    for key in closures:
        jobs.append((key, "del-off", None, None))
        jobs.append((key, "baseline", None, None))
        for epsilon in config.epsilons:
            for template in config.templates:
                jobs.append((key, "attack", epsilon, template))

    executed = 0
    with rows_path.open("a", encoding="utf-8") as handle:
        for key, role, epsilon, template in tqdm(
            jobs, desc=f"Adversarial {prompt_path.stem}", unit="generation"
        ):
            done_key = (key, role, str(epsilon), str(template))
            if done_key in done:
                continue
            manifest = closures[key].to_manifest()
            subject = dataclasses.replace(examples[key], deletion_manifest=manifest)
            state = DatabaseState.DEL_OFF if role == "del-off" else DatabaseState.DEL_ON
            injections: tuple[Any, ...] = ()
            if role == "attack":
                injections = build_injections(
                    example=examples[key],
                    query_vector=vectors[key],
                    config=config,
                    epsilon=epsilon,
                    template=template,
                    fact_seed=zlib.crc32(key.encode("utf-8")),
                )
            backend.injections = injections
            try:
                row = audit_example(
                    backend,
                    subject,
                    state,
                    max_new_tokens=max_new_tokens,
                )
            finally:
                backend.injections = ()
            row.pop("_query_embeddings", None)
            row["adversarial"] = {
                "target_key": key,
                "role": role,
                "epsilon": epsilon,
                "template": template,
                "rho": config.rho,
                "topology": config.topology,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)
            executed += 1

    # Aggregate: R(f), Ev(rho, epsilon, template), margin AUROC.
    correctness: dict[tuple[str, str, str, str], bool] = {}
    for row in rows:
        tag = row.get("adversarial") or {}
        correctness[
            (
                str(tag.get("target_key")),
                str(tag.get("role")),
                str(tag.get("epsilon")),
                str(tag.get("template")),
            )
        ] = _result_is_correct(row)

    r_of: dict[str, bool] = {}
    for key in closures:
        baseline = correctness.get((key, "baseline", "None", "None"))
        del_off = correctness.get((key, "del-off", "None", "None"))
        if baseline is None or del_off is None:
            continue
        r_of[key] = bool(baseline and not del_off)

    evasion_rows: list[dict[str, Any]] = []
    for epsilon in config.epsilons:
        for template in config.templates:
            outcomes = [
                correctness[(key, "attack", str(epsilon), str(template))]
                for key in closures
                if (key, "attack", str(epsilon), str(template)) in correctness
            ]
            evasion_rows.append(
                {
                    "rho": config.rho,
                    "epsilon": epsilon,
                    "template": template,
                    "topology": config.topology,
                    "facts": len(outcomes),
                    "evasion_rate": (
                        sum(outcomes) / len(outcomes) if outcomes else None
                    ),
                }
            )

    margin_rows: list[dict[str, Any]] = []
    for key, closure in closures.items():
        margin_rows.append(
            {
                "fact": key,
                "rho": config.rho,
                "s_del": closure.s_del,
                "s_surv": closure.s_surv,
                "margin": closure.margin,
                "r_f": r_of.get(key),
            }
        )
    scored = [
        row
        for row in margin_rows
        if row["margin"] is not None and row["r_f"] is not None
    ]
    # Predictor: survivor proximity (-margin). A close survivor predicts
    # retrieval-mediated leakage before any deletion is run.
    margin_auroc = (
        auroc(
            [-row["margin"] for row in scored],
            [row["r_f"] for row in scored],
        )
        if scored
        else None
    )

    return {
        "prompt_file": str(prompt_path),
        "facts": len(examples),
        "attacked_facts": len(closures),
        "skipped_facts": skipped,
        "rho": config.rho,
        "epsilons": list(config.epsilons),
        "templates": list(config.templates),
        "topology": config.topology,
        "executed_generations": executed,
        "evasion": evasion_rows,
        "margins": margin_rows,
        "margin_auroc": margin_auroc,
        "margin_auroc_facts": len(scored),
        "output_dir": str(output_dir),
    }
