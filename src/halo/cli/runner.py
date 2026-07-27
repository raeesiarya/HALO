import dataclasses
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
from halo.cli.full_pass import FullPassStore, run_full_pass
from halo.cli.persistence import (
    AUDIT_SCHEMA_VERSION,
    PartialLog,
    atomic_write_text,
    backend_resume_identity as _backend_resume_identity,
    ensure_resume_config as _ensure_resume_config,
    partial_name,
    partial_paths,
    prompt_digest as _prompt_digest,
    read_partial_jsonl,
)
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


class StandardAuditPersistence:
    """Per-fact append logs for the standard audit, one JSONL line per fact
    (rows + captured vectors together, so a torn write loses at most one
    fact and never leaves a partial state group)."""

    def __init__(
        self,
        *,
        output_dir: Path,
        stem: str,
        expected_states: list[DatabaseState],
        shard: tuple[int, int] | None = None,
    ) -> None:
        self._expected = [state.value for state in expected_states]
        self._results_stem = f"{stem}_results"
        self._skips_stem = f"{stem}_skipped_facts"
        self.output_dir = output_dir
        self.done_rows: dict[str, list[dict[str, Any]]] = {}
        self.done_vectors: dict[str, list[dict[str, Any]]] = {}
        self.done_skips: dict[str, dict[str, str]] = {}
        for path in partial_paths(output_dir, self._results_stem):
            for group in read_partial_jsonl(path):
                key = str(group.get("fact_key"))
                rows = group.get("rows") or []
                if [row.get("state") for row in rows] != self._expected:
                    continue
                self.done_rows.setdefault(key, rows)
                self.done_vectors.setdefault(key, group.get("vectors") or [])
        for path in partial_paths(output_dir, self._skips_stem):
            for entry in read_partial_jsonl(path):
                self.done_skips.setdefault(
                    str(entry.get("fact_key")), entry.get("entry") or {}
                )
        self._results_log = PartialLog(
            output_dir / partial_name(self._results_stem, shard)
        )
        self._skips_log = PartialLog(
            output_dir / partial_name(self._skips_stem, shard)
        )

    def record_fact(
        self,
        key: str,
        rows: list[dict[str, Any]],
        vectors: list[dict[str, Any]],
    ) -> None:
        self._results_log.write({"fact_key": key, "rows": rows, "vectors": vectors})

    def record_skip(self, key: str, entry: dict[str, str]) -> None:
        self._skips_log.write({"fact_key": key, "entry": entry})

    def close(self) -> None:
        self._results_log.close()
        self._skips_log.close()

    def cleanup(self) -> None:
        """Drop the partials once every canonical artifact is on disk."""
        self.close()
        for stem in (self._results_stem, self._skips_stem):
            for path in partial_paths(self.output_dir, stem):
                path.unlink()


def _persistence_fact_key(example: AuditExample) -> str:
    key = fact_key({"prompt_id": example.prompt_id, "fact_id": example.fact_id})
    if not key or "/" in key:
        raise AuditIntegrationError(
            "Resumable/reusable audits require a unique prompt_id/fact_id "
            f"without '/' per prompt row; got {key!r}."
        )
    return key


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
    *,
    full_store: FullPassStore | None = None,
    reuse_store: Any = None,
    resume: StandardAuditPersistence | None = None,
    shard: tuple[int, int] | None = None,
) -> list[dict[str, Any]]:
    if not states:
        raise ValueError("At least one audit state is required.")
    if len(states) != len(set(states)):
        raise ValueError("Audit states must not contain duplicates.")
    if shard is not None and resume is None:
        raise ValueError("Shard runs require persistence (resume).")

    prompts = load_prompts(prompt_path)
    if limit is not None:
        prompts = prompts[:limit]
    persistence_active = (
        full_store is not None or reuse_store is not None or resume is not None
    )

    results: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    progress = tqdm(
        prompts,
        desc=f"Auditing {prompt_path.stem}",
        unit="prompt",
    )
    for row_index, prompt in enumerate(progress):
        example = AuditExample.from_prompt_row(prompt)
        key = _persistence_fact_key(example) if persistence_active else None
        if shard is not None and row_index % shard[1] != shard[0]:
            continue
        if resume is not None:
            if key in resume.done_skips:
                skipped.append(resume.done_skips[key])
                progress.set_postfix(skipped=len(skipped))
                continue
            if key in resume.done_rows:
                resumed_rows = resume.done_rows[key]
                validate_intervention_results(
                    resumed_rows, expected_states=states
                )
                if embedding_sink is not None:
                    for item in resume.done_vectors.get(key, []):
                        embedding_sink.add(
                            example_key=key,
                            state=str(item["state"]),
                            event_index=int(item["event_index"]),
                            vector=np.asarray(item["vector"], dtype=np.float32),
                        )
                results.extend(resumed_rows)
                continue
        prompt_results: list[dict[str, Any]] = []
        remaining_states = list(states)

        if bootstrap_oracle_from_full and example.deletion_manifest.is_empty:
            if not remaining_states or remaining_states[0] is not DatabaseState.FULL:
                raise ValueError(
                    "Oracle bootstrapping requires FULL to be the first requested state."
                )
            full_result = None
            if full_store is not None:
                full_result = full_store.get_or_generate(key, example)
            elif reuse_store is not None:
                full_result = reuse_store.serve(
                    key=key,
                    state=DatabaseState.FULL,
                    example=example,
                    manifest=example.deletion_manifest,
                )
            if full_result is None:
                full_result = audit_example(
                    backend=backend,
                    example=example,
                    state=DatabaseState.FULL,
                    max_new_tokens=max_new_tokens,
                )
                if reuse_store is not None:
                    reuse_store.note_generated(DatabaseState.FULL)
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
                skip_entry = {
                    "fact_id": fact_id,
                    "reason": skip_reason,
                    "prompt_text": example.prompt,
                    "gold": example.ground_truth,
                    "selected_value": str(selected.get("value") or ""),
                }
                skipped.append(skip_entry)
                if resume is not None:
                    resume.record_skip(key, skip_entry)
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
            row = None
            if reuse_store is not None:
                row = reuse_store.serve(
                    key=key,
                    state=state,
                    example=example,
                    manifest=example.deletion_manifest,
                )
            if row is None:
                row = audit_example(
                    backend=backend,
                    example=example,
                    state=state,
                    max_new_tokens=max_new_tokens,
                )
                if reuse_store is not None:
                    reuse_store.note_generated(state)
            prompt_results.append(row)
        validate_intervention_results(prompt_results, expected_states=states)
        collected_vectors: list[dict[str, Any]] = []
        for result in prompt_results:
            # Numpy arrays must never reach the JSONL writer; route them to
            # the sidecar (or drop them when no sink is configured).
            embeddings = result.pop("_query_embeddings", None)
            if embedding_sink is None or not embeddings:
                continue
            sink_key = result_example_key(result, row_index)
            for item in embeddings:
                embedding_sink.add(
                    example_key=sink_key,
                    state=str(result["state"]),
                    event_index=int(item["event_index"]),
                    vector=item["vector"],
                )
                collected_vectors.append(
                    {
                        "state": str(result["state"]),
                        "event_index": int(item["event_index"]),
                        "vector": np.asarray(
                            item["vector"], dtype=np.float32
                        ).tolist(),
                    }
                )
        results.extend(prompt_results)
        if resume is not None:
            resume.record_fact(key, prompt_results, collected_vectors)

    if resume is not None:
        resume.close()
    if shard is not None:
        # Shard runs only append partials; the unsharded finalize invocation
        # owns coverage, the canonical skip log, and every other artifact.
        return results
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


# Resumable FULL-pass persistence lives in halo.cli.full_pass; this alias
# preserves the historical entry point for the sweep/adversarial callers.
_full_pass = run_full_pass


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
    from halo.interventions.closure import _safe_filename

    # Deterministic per-job name so concurrent radius shards never
    # overwrite each other's evidence.
    path = output_dir / (
        "reuse_canary_failure_"
        f"{_safe_filename(target_key)}_{role}_{_safe_filename(subject_key)}"
        f"_rho{rho:.4f}.json"
    )
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
    atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False))
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
    execute_radii: tuple[float, ...] | None = None,
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

    ``execute_radii`` narrows execution to a subset of the grid so parallel
    processes can each own some radii while sharing ``radii`` as the resume
    identity; an empty tuple materializes the FULL pass, closures, and
    neighbors without generating. Subset runs skip the entanglement analysis
    (summary field ``partial``) — the later full-grid invocation resumes
    every per-radius file and computes it. Note the fingerprint cache is
    per-process, so sharded runs may mark a row ``executed`` where a
    sequential run marks it ``reused`` (and vice versa); outputs and metrics
    are identical either way — that is exactly the fingerprint guarantee.
    """
    from halo.interventions.closure import (
        build_closure_family,
        full_query_vector,
        full_selected_candidate,
    )

    if not radii:
        raise ValueError("A radius sweep requires at least one radius.")
    if execute_radii is None:
        execute_radii = tuple(radii)
    else:
        execute_radii = tuple(execute_radii)
        unknown = [rho for rho in execute_radii if rho not in radii]
        if unknown:
            raise ValueError(
                f"Radii {unknown} are not members of the sweep grid {list(radii)}."
            )
    partial = execute_radii != tuple(radii)
    output_dir.mkdir(parents=True, exist_ok=True)

    full_pass_dir = full_dir or output_dir
    if execute_radii and partial and not (full_pass_dir / "full_results.jsonl").exists():
        raise AuditIntegrationError(
            "Radius-subset sweeps share one FULL pass and must not race to "
            f"create it; materialize {full_pass_dir} first (e.g. with "
            "--sweep-shard-radii none) and then launch the radius shards."
        )

    examples = _load_examples(prompt_path, limit)
    full_rows, vectors = _full_pass(
        backend,
        examples,
        prompt_path,
        limit,
        full_pass_dir,
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

    planned = sum(
        len(execute_radii) * (1 + len(neighbors.get(key, []))) for key in families
    )
    executed = 0
    reused_fingerprint = 0
    reused_full_pass = 0
    canary_checks = 0
    sweep_rows: list[dict[str, Any]] = []
    done: dict[float, set[tuple[str, str, str]]] = {}
    rows_on_disk: dict[float, dict[tuple[str, str, str], dict[str, Any]]] = {}
    for rho in execute_radii:
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
        for rho in execute_radii
    }
    progress = tqdm(
        total=planned, desc=f"Sweeping {prompt_path.stem}", unit="generation"
    )
    try:
        for key, family in families.items():
            manifests = {rho: family[rho].to_manifest() for rho in execute_radii}
            fingerprints = {
                rho: backend_manifest_fingerprint(backend, manifests[rho])
                for rho in execute_radii
            }
            jobs = [("target", key)] + [
                ("neighbor", neighbor_key)
                for neighbor_key in neighbors.get(key, [])
                if neighbor_key in examples
            ]
            for role, subject_key in jobs:
                for rho in execute_radii:
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

    # Subset runs hold only their own radii's rows; the entanglement curves
    # are computed by the full-grid invocation that resumes every file.
    entanglement = (
        {}
        if partial
        else compute_entanglement(sweep_rows, list(full_rows.values()), neighbors)
    )
    return {
        "prompt_file": str(prompt_path),
        "facts": len(examples),
        "swept_facts": len(families),
        "skipped_facts": skipped,
        "skipped_by_reason": dict(skipped_by_reason),
        "radii": list(radii),
        "executed_radii": list(execute_radii),
        "partial": partial,
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
    reuse_store: Any = None,
    shard: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Adversarial-closure evaluation: Ev(rho, epsilon) and the geometry-only
    margin predictor.

    Per fact: FULL -> closure at rho -> DEL-OFF and baseline DEL-ON rows
    (yielding R(f)) -> one injected DEL-ON per (epsilon, template). Rows are
    appended to resumable partials keyed by (fact, role, epsilon, template)
    and canonicalized into ``adversarial_results.jsonl`` at completion.
    `full_dir` holds the FULL-pass artifacts (defaults to `output_dir`) and
    can be shared with the entanglement sweep. ``reuse_store`` may serve the
    manifest-independent del-off rows from a standard-audit results file.
    ``shard`` stripes facts across parallel workers; the later unsharded
    invocation merges every stripe and computes the analysis.
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
    full_pass_dir = full_dir or output_dir
    if shard is not None and not (full_pass_dir / "full_results.jsonl").exists():
        raise AuditIntegrationError(
            "Adversarial shard runs share one FULL pass and must not race to "
            f"create it; materialize {full_pass_dir} first with an unsharded "
            "run (or the suite's standard/sweep phases)."
        )
    examples = _load_examples(prompt_path, limit)
    full_rows, vectors = _full_pass(
        backend,
        examples,
        prompt_path,
        limit,
        full_pass_dir,
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
    for fact_index, (key, example) in enumerate(examples.items()):
        # Shard runs stripe facts; closures and rows for other stripes come
        # from their own workers, merged by the unsharded finalize run.
        if shard is not None and fact_index % shard[1] != shard[0]:
            continue
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

    def _ingest(row: dict[str, Any]) -> None:
        tag = row.get("adversarial") or {}
        done_key = (
            str(tag.get("target_key")),
            str(tag.get("role")),
            str(tag.get("epsilon")),
            str(tag.get("template")),
        )
        if done_key in done:
            return
        done.add(done_key)
        rows.append(row)

    if rows_path.exists():
        for row in load_prompts(rows_path):
            _ingest(row)
    for path in partial_paths(output_dir, "adversarial_results"):
        for row in read_partial_jsonl(path):
            _ingest(row)

    jobs: list[tuple[str, str, float | None, str | None]] = []
    for key in closures:
        jobs.append((key, "del-off", None, None))
        jobs.append((key, "baseline", None, None))
        for epsilon in config.epsilons:
            for template in config.templates:
                jobs.append((key, "attack", epsilon, template))

    executed = 0
    reused_del_off = 0
    partial_log = PartialLog(
        output_dir / partial_name("adversarial_results", shard)
    )
    try:
        for key, role, epsilon, template in tqdm(
            jobs, desc=f"Adversarial {prompt_path.stem}", unit="generation"
        ):
            done_key = (key, role, str(epsilon), str(template))
            if done_key in done:
                continue
            manifest = closures[key].to_manifest()
            subject = dataclasses.replace(examples[key], deletion_manifest=manifest)
            state = DatabaseState.DEL_OFF if role == "del-off" else DatabaseState.DEL_ON
            row = None
            if role == "del-off" and reuse_store is not None:
                # DEL-OFF is manifest-independent, so the standard audit's
                # row for this fact is this generation, re-stamped.
                row = reuse_store.serve(
                    key=key,
                    state=DatabaseState.DEL_OFF,
                    example=examples[key],
                    manifest=manifest,
                    require_embeddings=False,
                )
                if row is not None:
                    reused_del_off += 1
            if row is None:
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
                executed += 1
            row.pop("_query_embeddings", None)
            row["adversarial"] = {
                "target_key": key,
                "role": role,
                "epsilon": epsilon,
                "template": template,
                "rho": config.rho,
                "topology": config.topology,
            }
            partial_log.write(row)
            done.add(done_key)
            rows.append(row)
    finally:
        partial_log.close()

    if shard is not None:
        # Shard runs only append partials; the unsharded finalize invocation
        # merges every stripe, canonicalizes, and computes the analysis.
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
            "reused_del_off": reused_del_off,
            "partial": True,
            "evasion": [],
            "margins": [],
            "margin_auroc": None,
            "margin_auroc_facts": 0,
            "output_dir": str(output_dir),
        }

    # Canonicalize in job order (a fresh run's append order) and drop the
    # partials; the canonical file now always means "complete".
    row_by_key = {}
    for row in rows:
        tag = row.get("adversarial") or {}
        row_by_key[
            (
                str(tag.get("target_key")),
                str(tag.get("role")),
                str(tag.get("epsilon")),
                str(tag.get("template")),
            )
        ] = row
    with rows_path.open("w", encoding="utf-8") as handle:
        for key, role, epsilon, template in jobs:
            row = row_by_key.get((key, role, str(epsilon), str(template)))
            if row is not None:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    for path in partial_paths(output_dir, "adversarial_results"):
        path.unlink()

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
        "reused_del_off": reused_del_off,
        "partial": False,
        "evasion": evasion_rows,
        "margins": margin_rows,
        "margin_auroc": margin_auroc,
        "margin_auroc_facts": len(scored),
        "output_dir": str(output_dir),
    }
