import dataclasses
import hashlib
import json
import zlib
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
from tqdm import tqdm

from halo.core.backend import (
    AuditBackend,
    audit_example,
    backend_full_row_unaffected,
    backend_manifest_fingerprint,
    validate_intervention_results,
)
from halo.interventions.errors import AuditIntegrationError
from halo.core.embeddings import QueryEmbeddingSink, result_example_key
from halo.core.entanglement import compute_entanglement, fact_key
from halo.core.examples import AuditExample, DeletionManifest
from halo.core.metrics import _result_is_correct
from halo.core.neighbors import (
    NeighborConfig,
    compute_cosine_neighbors,
    compute_same_source_neighbors,
    neighbor_keys,
    write_neighbors_file,
)
from halo.core.states import DatabaseState

AUDIT_SCHEMA_VERSION = 2


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
    skip_log_path: Path | None = None,
    coverage_summary: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if not states:
        raise ValueError("At least one audit state is required.")
    if len(states) != len(set(states)):
        raise ValueError("Audit states must not contain duplicates.")

    prompts = load_prompts(prompt_path)
    if limit is not None:
        prompts = prompts[:limit]

    results: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    progress = tqdm(
        prompts,
        desc=f"Auditing {prompt_path.stem}",
        unit="prompt",
    )
    for row_index, prompt in enumerate(progress):
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
            trace = full_result.get("retrieval_trace") or {}
            entry_id = selected.get("entry_id")
            # The schema-free bootstrap is an oracle answer-mention baseline,
            # not proposition verification. It is auditable only when the
            # selected lookup occurs before the visible answer.
            skip_reason = None
            if not entry_id:
                skip_reason = "FULL produced no selected entry ID"
            elif selected.get("supports_target") is not True:
                skip_reason = (
                    "FULL selected entry did not pass the answer-mention heuristic"
                )
            elif trace.get("selected_answer_visible_before_event") is True:
                skip_reason = (
                    "FULL answer was already visible before the answer-mention event"
                )
            elif trace.get("selected_answer_visible_before_event") is not False:
                skip_reason = (
                    "FULL answer-mention event has unknown temporal alignment"
                )
            manifest = None
            if skip_reason is None and manifest_builder is not None:
                manifest = manifest_builder(example, full_result)
                if manifest.is_empty:
                    skip_reason = "manifest builder produced an empty manifest"
            if skip_reason is not None:
                fact_id = str(
                    prompt.get("fact_id") or prompt.get("prompt_id") or row_index
                )
                skipped.append(
                    {
                        "fact_id": fact_id,
                        "reason": skip_reason,
                        "prompt_text": example.prompt,
                        "gold": example.ground_truth,
                        "selected_value": str(selected.get("value") or ""),
                    }
                )
                progress.set_postfix(skipped=len(skipped))
                continue
            if manifest is None:
                manifest = DeletionManifest(
                    entry_ids=(str(entry_id),),
                    strategy="oracle-from-full",
                    metadata={
                        "bootstrap": "FULL.selected_candidate",
                        "audited_event_index": trace.get("selected_event_index"),
                        "selection_policy": (
                            "pre-answer-normalized-answer-mention"
                        ),
                    },
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

    if bootstrap_oracle_from_full:
        audited = len(prompts) - len(skipped)
        by_reason = Counter(item["reason"] for item in skipped)
        if coverage_summary is not None:
            coverage_summary.update(
                {
                    "facts": len(prompts),
                    "audited_facts": audited,
                    "skipped_facts": len(skipped),
                    "skipped_by_reason": dict(by_reason),
                }
            )
        breakdown = "".join(
            f"\n  {count}x {reason}" for reason, count in by_reason.most_common()
        )
        tqdm.write(
            f"Oracle bootstrap coverage: {audited}/{len(prompts)} facts audited, "
            f"{len(skipped)} skipped." + breakdown
        )
        if skipped and skip_log_path is not None:
            skip_log_path.parent.mkdir(parents=True, exist_ok=True)
            with skip_log_path.open("w", encoding="utf-8") as handle:
                for item in skipped:
                    handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            tqdm.write(f"Wrote per-fact skip details to {skip_log_path}")
    elif coverage_summary is not None:
        coverage_summary.update(
            {
                "facts": len(prompts),
                "audited_facts": len(prompts),
                "skipped_facts": 0,
                "skipped_by_reason": {},
            }
        )

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


def _prompt_digest(prompt_path: Path) -> str:
    digest = hashlib.sha256()
    with prompt_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _backend_resume_identity(backend: AuditBackend) -> dict[str, Any]:
    """Stable backend fields that can change generated or retrieved outputs."""
    identity = {
        "class": f"{type(backend).__module__}.{type(backend).__qualname__}"
    }
    for field in ("model_path", "index_path", "release_source"):
        value = getattr(backend, field, None)
        if value is not None:
            identity[field] = str(value)
    retrieval_config = getattr(
        getattr(backend, "generator", None),
        "retrieval_config",
        None,
    )
    identity["similarity_threshold"] = getattr(
        retrieval_config,
        "similarity_threshold",
        getattr(backend, "similarity_threshold", None),
    )
    return identity


def _ensure_resume_config(
    path: Path,
    payload: dict[str, Any],
    *,
    legacy_artifacts: tuple[Path, ...] = (),
) -> None:
    """Prevent resumable artifacts from crossing experiment definitions."""
    normalized = json.loads(json.dumps(payload, sort_keys=True, default=str))
    if path.exists():
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored != normalized:
            raise AuditIntegrationError(
                f"Resume configuration mismatch in {path}. Use a fresh output "
                "directory rather than mixing experiment definitions."
            )
        return
    existing = [artifact for artifact in legacy_artifacts if artifact.exists()]
    if existing:
        raise AuditIntegrationError(
            f"Found resumable artifacts without {path.name}: {existing[0]}. "
            "They predate configuration guards; use a fresh output directory."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(normalized, indent=2, sort_keys=True), encoding="utf-8"
    )


def _full_pass(
    backend: AuditBackend,
    examples: dict[str, AuditExample],
    prompt_path: Path,
    limit: int | None,
    output_dir: Path,
    max_new_tokens: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray]]:
    """FULL over every prompt, capturing query embeddings. Resumed wholesale
    when both artifacts from a previous run exist."""
    from halo.interventions.closure import full_query_vector

    output_dir.mkdir(parents=True, exist_ok=True)
    full_rows_path = output_dir / "full_results.jsonl"
    embeddings_path = output_dir / "full_query_embeddings.npz"
    _ensure_resume_config(
        output_dir / "full_config.json",
        {
            "audit_schema_version": AUDIT_SCHEMA_VERSION,
            "mode": "full-pass",
            "backend": _backend_resume_identity(backend),
            "prompt_sha256": _prompt_digest(prompt_path),
            "limit": limit,
            "max_new_tokens": max_new_tokens,
        },
        legacy_artifacts=(full_rows_path, embeddings_path),
    )
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


def _primary_target_skip_reason(
    full_row: dict[str, Any] | None,
    query_vector: np.ndarray | None,
) -> str | None:
    """Return the primary-cohort exclusion reason, if any."""
    if not full_row:
        return "missing FULL result"
    selected = (full_row.get("retrieval_trace") or {}).get(
        "selected_candidate"
    ) or {}
    trace = full_row.get("retrieval_trace") or {}
    if not selected.get("entry_id"):
        return "FULL produced no selected entry ID"
    if selected.get("supports_target") is not True:
        return "FULL selected entry did not pass the answer-mention heuristic"
    if trace.get("selected_answer_visible_before_event") is True:
        return "FULL answer was already visible before the answer-mention event"
    if trace.get("selected_answer_visible_before_event") is not False:
        return "FULL answer-mention event has unknown temporal alignment"
    if not _result_is_correct(full_row):
        return "FULL answer was incorrect"
    if query_vector is None:
        return "FULL produced no query embedding"
    return None


def _canary_selected(
    target_key: str, role: str, subject_key: str, rho: float, rate: float
) -> bool:
    """Deterministic per-job canary draw, stable across resumes."""
    if rate <= 0.0:
        return False
    if rate >= 1.0:
        return True
    token = f"{target_key}|{role}|{subject_key}|{rho:.6f}".encode("utf-8")
    return (zlib.crc32(token) % 10_000) < int(rate * 10_000)


def _reused_sweep_row(
    source_row: dict[str, Any],
    *,
    manifest: DeletionManifest,
    rho: float,
    target_key: str,
    role: str,
    reused_from: str,
) -> dict[str, Any]:
    """A DEL-ON sweep row copied from an equivalent prior generation.

    Only the manifest bookkeeping and the sweep tag are rewritten; the trace
    keeps the source run's candidate lists (thinner than a live DEL-ON run
    would record), which is why the row is marked ``reused``.
    """
    row = json.loads(json.dumps(source_row, ensure_ascii=False))
    row.pop("_query_embeddings", None)
    row["state"] = DatabaseState.DEL_ON.value
    row["deletion_manifest"] = manifest.as_dict()
    trace = row.get("retrieval_trace")
    if isinstance(trace, dict):
        trace["state"] = DatabaseState.DEL_ON.value
        trace["retrieval_enabled"] = True
        trace["deletion_manifest_id"] = manifest.manifest_id
    row["sweep"] = {
        "target_key": target_key,
        "rho": rho,
        "role": role,
        "reused": reused_from,
    }
    return row


def _canary_trace_digest(row: Mapping[str, Any] | None) -> dict[str, Any]:
    """The parts of a row's retrieval trace that explain a canary mismatch."""
    if not isinstance(row, Mapping):
        return {}
    trace = row.get("retrieval_trace") or {}
    return {
        "state": trace.get("state"),
        "model_output": row.get("model_output"),
        "sweep": row.get("sweep"),
        "selected_candidate": trace.get("selected_candidate"),
        "selected_value": trace.get("selected_value"),
        "num_retrievals": trace.get("num_retrievals"),
        "failed_retrievals": trace.get("failed_retrievals"),
        "all_candidates_count": trace.get("all_candidates_count"),
        "deleted_candidates_count": trace.get("deleted_candidates_count"),
        # searched_top_k is the discriminator: the FULL pass searches at
        # k=1 while DEL-ON searches at k=1+|entry_ids|, so a mismatch here
        # means the two rows were never comparable to begin with.
        "events": [
            {
                "searched_top_k": event.get("searched_top_k"),
                "requested_top_k": event.get("requested_top_k"),
                "widened_search_attempts": event.get("widened_search_attempts"),
                "all_candidates_count": event.get("all_candidates_count"),
                "deleted_candidates_count": event.get("deleted_candidates_count"),
                "candidate_ids": [
                    candidate.get("entry_id")
                    for candidate in (event.get("all_candidates") or [])
                ],
            }
            for event in (trace.get("retrieval_events") or [])
        ],
    }


def _write_canary_failure_report(
    output_dir: Path,
    *,
    target_key: str,
    role: str,
    subject_key: str,
    rho: float,
    reused_from: str,
    source_origin: str | None,
    manifest: DeletionManifest,
    source_row: Mapping[str, Any],
    generated_row: Mapping[str, Any],
    full_row: Mapping[str, Any] | None,
) -> Path:
    """Dump both sides of a failed canary so the cause is diagnosable from
    the artifact alone, without re-running the sweep."""
    path = output_dir / "reuse_canary_failure.json"
    payload = {
        "target_key": target_key,
        "role": role,
        "subject_key": subject_key,
        "rho": rho,
        "served_by": reused_from,
        "originated_from": source_origin,
        "manifest": {
            "manifest_id": manifest.manifest_id,
            "strategy": manifest.strategy,
            "entry_ids": list(manifest.entry_ids),
            "source_ids": list(manifest.source_ids),
        },
        "reused": _canary_trace_digest(source_row),
        "generated": _canary_trace_digest(generated_row),
        "full_pass": _canary_trace_digest(full_row),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


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
    full_dir: Path | None = None,
    reuse_canary_rate: float = 0.0,
) -> dict[str, Any]:
    """Radius sweep for the entanglement analysis (E, X, G).

    Pass 1 runs FULL once over all prompts (capturing query embeddings and
    FULL-correctness); closures for every radius come from one search per
    fact; then each (fact, radius) runs the target prompt and all neighbor
    prompts under DEL-ON. Per-radius JSONL files make the sweep resumable:
    (target, role, subject) triples already on disk are skipped. `full_dir`
    holds the FULL-pass artifacts (defaults to `output_dir`); point the sweep
    and the adversarial evaluation at the same directory to share one pass.

    Backends can cut the generation count via two capability hooks (see
    ``halo.core.backend``): identical manifest fingerprints share one
    generation per subject, and manifests a backend certifies as unable to
    affect a subject reuse that subject's FULL row. Reused rows carry
    ``sweep.reused``. ``reuse_canary_rate`` re-executes that fraction of
    reusable jobs anyway and raises if the generated output disagrees with
    the reused row — a continuous soundness check on the hooks.
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
    full_rows, vectors = _full_pass(
        backend,
        examples,
        prompt_path,
        limit,
        full_dir or output_dir,
        max_new_tokens,
    )

    _ensure_resume_config(
        output_dir / "evaluation_config.json",
        {
            "audit_schema_version": AUDIT_SCHEMA_VERSION,
            "mode": "entanglement-sweep",
            "backend": _backend_resume_identity(backend),
            "prompt_sha256": _prompt_digest(prompt_path),
            "limit": limit,
            "max_new_tokens": max_new_tokens,
            "radii": list(radii),
            "closure": {
                "predicates": list(closure_config.predicates),
                "radius": closure_config.radius,
                "envelope_top_k": closure_config.envelope_top_k,
                "max_closure_size": closure_config.max_closure_size,
            },
            "neighbors": dataclasses.asdict(neighbor_config),
        },
        legacy_artifacts=tuple(output_dir.glob("sweep_rho_*.jsonl"))
        + (output_dir / "neighbors.json",),
    )

    # Closure families: one geometric search per fact covers every radius.
    families: dict[str, dict[float, Any]] = {}
    skipped: list[str] = []
    skipped_by_reason: Counter[str] = Counter()
    judge = getattr(backend, "support_judge", None)
    for key, example in examples.items():
        selected = full_selected_candidate(full_rows.get(key, {}))
        vector = vectors.get(key)
        skip_reason = _primary_target_skip_reason(full_rows.get(key), vector)
        if skip_reason is not None:
            skipped.append(key)
            skipped_by_reason[skip_reason] += 1
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

    if not 0.0 <= reuse_canary_rate <= 1.0:
        raise ValueError("reuse_canary_rate must be in [0, 1].")

    planned = sum(len(radii) * (1 + len(neighbors.get(key, []))) for key in families)
    executed = 0
    reused_fingerprint = 0
    reused_full_pass = 0
    canary_checks = 0
    sweep_rows: list[dict[str, Any]] = []
    done: dict[float, set[tuple[str, str, str]]] = {}
    rows_on_disk: dict[float, dict[tuple[str, str, str], dict[str, Any]]] = {}
    for rho in radii:
        rho_path = output_dir / f"sweep_rho_{rho:.4f}.jsonl"
        done[rho] = set()
        rows_on_disk[rho] = {}
        if rho_path.exists():
            for row in load_prompts(rho_path):
                tag = row.get("sweep") or {}
                triple = (
                    str(tag.get("target_key")),
                    str(tag.get("role")),
                    fact_key(row),
                )
                done[rho].add(triple)
                rows_on_disk[rho][triple] = row
                sweep_rows.append(row)

    # Cached rows carry their *origin* — the fast path that first vouched for
    # the generation — alongside the row. A row copied from the FULL pass and
    # then served again through the fingerprint cache is still, in substance,
    # a `full_row_unaffected` claim; without this the canary would attribute
    # the failure to `manifest_fingerprint` and send the reader to the wrong
    # hook. "executed" marks a genuine generation.
    generation_cache: dict[tuple[str, Any], tuple[dict[str, Any], str]] = {}
    handles = {
        rho: (output_dir / f"sweep_rho_{rho:.4f}.jsonl").open("a", encoding="utf-8")
        for rho in radii
    }
    progress = tqdm(
        total=planned, desc=f"Sweeping {prompt_path.stem}", unit="generation"
    )
    try:
        for key, family in families.items():
            manifests = {rho: family[rho].to_manifest() for rho in radii}
            fingerprints = {
                rho: backend_manifest_fingerprint(backend, manifests[rho])
                for rho in radii
            }
            jobs = [("target", key)] + [
                ("neighbor", neighbor_key)
                for neighbor_key in neighbors.get(key, [])
                if neighbor_key in examples
            ]
            for role, subject_key in jobs:
                for rho in radii:
                    triple = (key, role, subject_key)
                    fingerprint = fingerprints[rho]
                    cache_key = (
                        (subject_key, fingerprint)
                        if fingerprint is not None
                        else None
                    )
                    if triple in done[rho]:
                        if cache_key is not None:
                            resumed = rows_on_disk[rho][triple]
                            generation_cache.setdefault(
                                cache_key,
                                (
                                    resumed,
                                    str(
                                        (resumed.get("sweep") or {}).get("reused")
                                        or "executed"
                                    ),
                                ),
                            )
                        progress.update(1)
                        continue
                    manifest = manifests[rho]
                    reused_from: str | None = None
                    source_row: dict[str, Any] | None = None
                    source_origin: str | None = None
                    if cache_key is not None and cache_key in generation_cache:
                        reused_from = "fingerprint"
                        source_row, source_origin = generation_cache[cache_key]
                    elif backend_full_row_unaffected(
                        backend, full_rows.get(subject_key), manifest
                    ):
                        reused_from = "full-pass"
                        source_row = full_rows[subject_key]
                        source_origin = "full-pass"
                    canary = reused_from is not None and _canary_selected(
                        key, role, subject_key, rho, reuse_canary_rate
                    )
                    if reused_from is not None and not canary:
                        row = _reused_sweep_row(
                            source_row,
                            manifest=manifest,
                            rho=rho,
                            target_key=key,
                            role=role,
                            reused_from=reused_from,
                        )
                        if reused_from == "fingerprint":
                            reused_fingerprint += 1
                        else:
                            reused_full_pass += 1
                    else:
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
                        executed += 1
                        if reused_from is not None:
                            canary_checks += 1
                            row["sweep"]["canary_verified"] = reused_from
                            row["sweep"]["canary_origin"] = source_origin
                            if row["model_output"] != source_row["model_output"]:
                                report = _write_canary_failure_report(
                                    output_dir,
                                    target_key=key,
                                    role=role,
                                    subject_key=subject_key,
                                    rho=rho,
                                    reused_from=reused_from,
                                    source_origin=source_origin,
                                    manifest=manifest,
                                    source_row=source_row,
                                    generated_row=row,
                                    full_row=full_rows.get(subject_key),
                                )
                                hook = (
                                    "full_row_unaffected"
                                    if source_origin == "full-pass"
                                    else "manifest_fingerprint"
                                )
                                raise AuditIntegrationError(
                                    f"Reuse canary failed for target {key!r} "
                                    f"({role} {subject_key!r}, rho={rho}): the "
                                    f"{reused_from!r} fast path predicted "
                                    f"{source_row['model_output']!r} but the "
                                    "backend generated "
                                    f"{row['model_output']!r}. The reused row "
                                    f"originated from the {source_origin!r} "
                                    f"path, so the claim that failed is the "
                                    f"backend's {hook!r} hook. Evidence written "
                                    f"to {report}."
                                )
                    if cache_key is not None:
                        # A canary-verified row was really generated, so it
                        # vouches for itself regardless of which fast path
                        # would have served it.
                        generation_cache.setdefault(
                            cache_key,
                            (
                                row,
                                "executed"
                                if (reused_from is None or canary)
                                else reused_from,
                            ),
                        )
                    handles[rho].write(json.dumps(row, ensure_ascii=False) + "\n")
                    sweep_rows.append(row)
                    progress.update(1)
                    progress.set_postfix(
                        executed=executed,
                        reused=reused_fingerprint + reused_full_pass,
                        refresh=False,
                    )
    finally:
        progress.close()
        for handle in handles.values():
            handle.close()

    entanglement = compute_entanglement(sweep_rows, list(full_rows.values()), neighbors)
    return {
        "prompt_file": str(prompt_path),
        "facts": len(examples),
        "swept_facts": len(families),
        "skipped_facts": skipped,
        "skipped_by_reason": dict(skipped_by_reason),
        "radii": list(radii),
        "planned_generations": planned,
        "executed_generations": executed,
        "reused_generations": reused_fingerprint + reused_full_pass,
        "reused_fingerprint": reused_fingerprint,
        "reused_full_pass": reused_full_pass,
        "canary_checks": canary_checks,
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
    full_dir: Path | None = None,
) -> dict[str, Any]:
    """Adversarial-closure evaluation: Ev(rho, epsilon) and the geometry-only
    margin predictor.

    Per fact: FULL -> closure at rho -> DEL-OFF and baseline DEL-ON rows
    (yielding R(f)) -> one injected DEL-ON per (epsilon, template). Rows are
    appended to a resumable JSONL keyed by (fact, role, epsilon, template).
    `full_dir` holds the FULL-pass artifacts (defaults to `output_dir`) and
    can be shared with the entanglement sweep.
    """
    from halo.interventions.adversary import build_injections
    from halo.interventions.closure import (
        build_closure_family,
        full_selected_candidate,
        write_closure_artifact,
    )
    from halo.core.metrics import auroc

    config = adversarial_config
    output_dir.mkdir(parents=True, exist_ok=True)
    examples = _load_examples(prompt_path, limit)
    full_rows, vectors = _full_pass(
        backend,
        examples,
        prompt_path,
        limit,
        full_dir or output_dir,
        max_new_tokens,
    )

    _ensure_resume_config(
        output_dir / "evaluation_config.json",
        {
            "audit_schema_version": AUDIT_SCHEMA_VERSION,
            "mode": "adversarial",
            "backend": _backend_resume_identity(backend),
            "del_off_mode": getattr(backend, "del_off_mode", None),
            "prompt_sha256": _prompt_digest(prompt_path),
            "limit": limit,
            "max_new_tokens": max_new_tokens,
            "closure": {
                "predicates": list(closure_config.predicates),
                "radius": closure_config.radius,
                "envelope_top_k": closure_config.envelope_top_k,
                "max_closure_size": closure_config.max_closure_size,
            },
            "adversarial": dataclasses.asdict(config),
        },
        legacy_artifacts=(output_dir / "adversarial_results.jsonl",),
    )

    closures: dict[str, Any] = {}
    skipped: list[str] = []
    skipped_by_reason: Counter[str] = Counter()
    judge = getattr(backend, "support_judge", None)
    for key, example in examples.items():
        selected = full_selected_candidate(full_rows.get(key, {}))
        vector = vectors.get(key)
        skip_reason = _primary_target_skip_reason(full_rows.get(key), vector)
        if skip_reason is not None:
            skipped.append(key)
            skipped_by_reason[skip_reason] += 1
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

    # Aggregate R(f), attack attribution, and margin AUROC.
    result_rows: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    correctness: dict[tuple[str, str, str, str], bool] = {}
    for row in rows:
        tag = row.get("adversarial") or {}
        result_key = (
            str(tag.get("target_key")),
            str(tag.get("role")),
            str(tag.get("epsilon")),
            str(tag.get("template")),
        )
        result_rows[result_key] = row
        correctness[result_key] = _result_is_correct(row)

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
            attack_keys = [
                key
                for key in closures
                if (key, "attack", str(epsilon), str(template)) in correctness
            ]
            outcomes = [
                correctness[(key, "attack", str(epsilon), str(template))]
                for key in attack_keys
            ]
            baseline_outcomes = [
                correctness.get((key, "baseline", "None", "None"), False)
                for key in attack_keys
            ]
            target_selected: list[bool] = []
            for key in attack_keys:
                attack_row = result_rows[
                    (key, "attack", str(epsilon), str(template))
                ]
                selected = (
                    (attack_row.get("retrieval_trace") or {}).get(
                        "selected_candidate"
                    )
                    or {}
                )
                metadata = selected.get("metadata") or {}
                entry_id = str(selected.get("entry_id") or "")
                target_selected.append(
                    metadata.get("synthetic") is True
                    and entry_id.startswith(f"adv-{template}-")
                )
            selected_count = sum(target_selected)
            gains = [
                attack_correct and not baseline_correct
                for attack_correct, baseline_correct in zip(
                    outcomes, baseline_outcomes
                )
            ]
            regressions = [
                baseline_correct and not attack_correct
                for attack_correct, baseline_correct in zip(
                    outcomes, baseline_outcomes
                )
            ]
            selected_correct = sum(
                attack_correct and selected
                for attack_correct, selected in zip(outcomes, target_selected)
            )
            selected_gains = sum(
                gain and selected for gain, selected in zip(gains, target_selected)
            )
            evasion_rows.append(
                {
                    "rho": config.rho,
                    "epsilon": epsilon,
                    "template": template,
                    "topology": config.topology,
                    "facts": len(outcomes),
                    "selected_facts": selected_count,
                    "target_selected_rate": (
                        selected_count / len(outcomes) if outcomes else None
                    ),
                    "baseline_correct_rate": (
                        sum(baseline_outcomes) / len(outcomes) if outcomes else None
                    ),
                    "evasion_rate": (
                        sum(outcomes) / len(outcomes) if outcomes else None
                    ),
                    "post_attack_correct_rate": (
                        sum(outcomes) / len(outcomes) if outcomes else None
                    ),
                    "attack_gain_rate": (
                        sum(gains) / len(outcomes) if outcomes else None
                    ),
                    "attack_regression_rate": (
                        sum(regressions) / len(outcomes) if outcomes else None
                    ),
                    "correct_given_target_selected": (
                        selected_correct / selected_count if selected_count else None
                    ),
                    "gain_given_target_selected": (
                        selected_gains / selected_count if selected_count else None
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
        "skipped_by_reason": dict(skipped_by_reason),
        "rho": config.rho,
        "epsilons": list(config.epsilons),
        "templates": list(config.templates),
        "topology": config.topology,
        "closure_predicates": list(closure_config.predicates),
        "del_off_mode": getattr(backend, "del_off_mode", None),
        "executed_generations": executed,
        "evasion": evasion_rows,
        "margins": margin_rows,
        "margin_auroc": margin_auroc,
        "margin_auroc_facts": len(scored),
        "output_dir": str(output_dir),
    }
