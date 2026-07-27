"""Cross-phase generation reuse.

Suite phases repeat generations whose outputs are provably identical: FULL
rows never consult the deletion manifest, DEL-OFF rows are manifest-
independent by construction, and DEL-ON rows are a pure function of the
retrieval-filter fingerprint. A backend certifies this per state through its
``cross_phase_fingerprint`` capability hook; this module serves equivalent
rows from earlier phases' results files instead of regenerating them.

Served rows are re-stamped with the consuming fact's manifest bookkeeping
(``deletion_manifest`` + ``retrieval_trace.deletion_manifest_id``) and keep
the source generation's timing metadata; no other field changes, and no
provenance is added in-row. Provenance lives in a per-phase
``<stem>_reuse_manifest.json`` sidecar.

A deterministic canary re-executes a fraction of served generations and
raises when the backend's claim fails, mirroring the sweep's reuse canary.
"""

from __future__ import annotations

import dataclasses
import json
import zlib
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from halo.cli.persistence import atomic_write_json
from halo.core.backend import (
    AuditBackend,
    audit_example,
    backend_cross_phase_fingerprint,
)
from halo.core.entanglement import fact_key
from halo.core.examples import AuditExample, DeletionManifest
from halo.core.states import DatabaseState
from halo.interventions.errors import AuditIntegrationError

_RESULTS_SUFFIX = "_results.jsonl"


def manifest_from_row(row: Mapping[str, Any]) -> DeletionManifest:
    """Rebuild a row's manifest; the constructor canonicalizes exactly the
    form ``as_dict`` wrote, so the recomputed id must match the stored one."""
    stored = row.get("deletion_manifest") or {}
    manifest = DeletionManifest(
        entry_ids=tuple(stored.get("entry_ids") or ()),
        source_ids=tuple(stored.get("source_ids") or ()),
        strategy=stored.get("strategy") or "oracle-entry",
        metadata=stored.get("metadata") or {},
    )
    stored_id = stored.get("manifest_id")
    if stored_id is not None and manifest.manifest_id != stored_id:
        raise AuditIntegrationError(
            f"Stored manifest id {stored_id!r} does not round-trip "
            f"({manifest.manifest_id!r}); the source row is corrupt."
        )
    return manifest


@dataclasses.dataclass(frozen=True)
class ReuseContext:
    """Identity of the consuming run, checked against each source's config."""

    backend_identity: dict[str, Any]
    prompt_sha256: str
    limit: int | None
    max_new_tokens: int
    del_off_mode: str | None


class ReuseSource:
    """One ``--reuse-from`` standard-audit results file plus its sidecars."""

    def __init__(
        self,
        results_path: Path,
        context: ReuseContext,
        *,
        log: Callable[[str], None] = print,
    ) -> None:
        self.results_path = results_path
        self.label = str(results_path)
        if not results_path.name.endswith(_RESULTS_SUFFIX):
            raise AuditIntegrationError(
                f"--reuse-from expects a <stem>{_RESULTS_SUFFIX} standard-audit "
                f"results file, got {results_path}."
            )
        stem = results_path.name.removesuffix(_RESULTS_SUFFIX)
        self._npz_path = results_path.with_name(f"{stem}_query_embeddings.npz")
        config_path = results_path.with_name(f"{stem}_audit_config.json")
        self._rows: dict[tuple[str, str], dict[str, Any]] = {}
        self._npz: Any = None
        self._npz_index: dict[tuple[str, str], list[tuple[int, str]]] | None = None
        self.del_off_compatible = False

        if not results_path.exists():
            log(f"Reuse source {results_path} not found; generating instead.")
            return
        if not config_path.exists():
            raise AuditIntegrationError(
                f"Reuse source {results_path} has no {config_path.name}; only "
                "standard-audit outputs written with resume configs can be "
                "reused."
            )
        config = json.loads(config_path.read_text(encoding="utf-8"))
        normalized = json.loads(
            json.dumps(context.backend_identity, sort_keys=True, default=str)
        )
        for field, expected in (
            ("backend", normalized),
            ("prompt_sha256", context.prompt_sha256),
            ("limit", context.limit),
            ("max_new_tokens", context.max_new_tokens),
        ):
            if config.get(field) != expected:
                raise AuditIntegrationError(
                    f"Reuse source {results_path} was produced under a "
                    f"different {field!r} ({config.get(field)!r} vs "
                    f"{expected!r}); refusing to mix experiment definitions."
                )
        self.del_off_compatible = config.get("del_off_mode") == context.del_off_mode

        with results_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if "sweep" in row or "adversarial" in row:
                    raise AuditIntegrationError(
                        f"{results_path} contains sweep/adversarial rows; "
                        "--reuse-from takes standard-audit results only."
                    )
                key = fact_key(row)
                if not key:
                    continue
                index_key = (key, str(row.get("state")))
                if index_key in self._rows:
                    raise AuditIntegrationError(
                        f"Duplicate (fact, state) {index_key!r} in {results_path}."
                    )
                self._rows[index_key] = row

    def row(self, key: str, state: DatabaseState) -> dict[str, Any] | None:
        return self._rows.get((key, state.value))

    @property
    def has_embeddings_sidecar(self) -> bool:
        return self._npz_path.exists()

    def embeddings(
        self, key: str, state: DatabaseState
    ) -> list[dict[str, Any]]:
        """Captured event vectors for (fact, state); empty when none were."""
        if not self.has_embeddings_sidecar:
            return []
        if self._npz is None:
            self._npz = np.load(self._npz_path)
            self._npz_index = {}
            for stored_key in self._npz.files:
                fact, event_state, event = stored_key.rsplit("/", 2)
                self._npz_index.setdefault((fact, event_state), []).append(
                    (int(event.removeprefix("event")), stored_key)
                )
        return [
            {"event_index": index, "vector": self._npz[stored_key]}
            for index, stored_key in sorted(
                self._npz_index.get((key, state.value), [])
            )
        ]


class GenerationReuseStore:
    """Serves equivalent generations from earlier phases, with a canary."""

    def __init__(
        self,
        *,
        backend: AuditBackend,
        context: ReuseContext,
        source_paths: tuple[Path, ...],
        canary_rate: float = 0.01,
        max_new_tokens: int = 12,
        output_dir: Path | None = None,
        log: Callable[[str], None] = print,
    ) -> None:
        if not 0.0 <= canary_rate <= 1.0:
            raise ValueError("reuse canary rate must be in [0, 1].")
        self._backend = backend
        self._context = context
        self._canary_rate = canary_rate
        self._max_new_tokens = max_new_tokens
        self._output_dir = output_dir
        self._log = log
        self._sources = [
            ReuseSource(path, context, log=log) for path in source_paths
        ]
        self.served: Counter[str] = Counter()
        self.generated: Counter[str] = Counter()
        self.canary_checks = 0
        self.entries: list[dict[str, Any]] = []

    def _canary_selected(self, key: str, state: DatabaseState) -> bool:
        if self._canary_rate <= 0.0:
            return False
        if self._canary_rate >= 1.0:
            return True
        token = f"{key}|{state.value}|cross-phase".encode("utf-8")
        return (zlib.crc32(token) % 10_000) < int(self._canary_rate * 10_000)

    def _canary_failure(
        self,
        *,
        key: str,
        state: DatabaseState,
        source: ReuseSource,
        source_row: Mapping[str, Any],
        generated_row: Mapping[str, Any],
        manifest: DeletionManifest,
    ) -> None:
        from halo.cli.runner import _canary_trace_digest
        from halo.interventions.closure import _safe_filename

        payload = {
            "fact_key": key,
            "state": state.value,
            "served_by": source.label,
            "manifest": {
                "manifest_id": manifest.manifest_id,
                "strategy": manifest.strategy,
                "entry_ids": list(manifest.entry_ids),
                "source_ids": list(manifest.source_ids),
            },
            "reused": _canary_trace_digest(source_row),
            "generated": _canary_trace_digest(generated_row),
        }
        report_dir = self._output_dir or source.results_path.parent
        report = report_dir / (
            f"reuse_canary_failure_{_safe_filename(key)}_{state.value}.json"
        )
        atomic_write_json(report, payload)
        raise AuditIntegrationError(
            f"Cross-phase reuse canary failed for fact {key!r} in "
            f"{state.value}: {source.label} predicted "
            f"{source_row['model_output']!r} but the backend generated "
            f"{generated_row['model_output']!r}. The claim that failed is the "
            "backend's 'cross_phase_fingerprint' hook. Evidence written to "
            f"{report}."
        )

    def serve(
        self,
        *,
        key: str,
        state: DatabaseState,
        example: AuditExample,
        manifest: DeletionManifest,
        require_embeddings: bool = True,
    ) -> dict[str, Any] | None:
        """A stamped copy of an equivalent earlier generation, or None to
        generate live. Runs the canary on a deterministic fraction of hits."""
        fingerprint = backend_cross_phase_fingerprint(
            self._backend, state, manifest
        )
        if fingerprint is None:
            return None
        for source in self._sources:
            if state is DatabaseState.DEL_OFF and not source.del_off_compatible:
                continue
            row = source.row(key, state)
            if row is None:
                continue
            source_fingerprint = backend_cross_phase_fingerprint(
                self._backend, state, manifest_from_row(row)
            )
            if source_fingerprint != fingerprint:
                continue
            if state is DatabaseState.DEL_OFF:
                trace_mode = (row.get("retrieval_trace") or {}).get("del_off_mode")
                if trace_mode != self._context.del_off_mode:
                    continue
            if (
                row.get("prompt") != example.prompt
                or row.get("ground_truth") != example.ground_truth
            ):
                raise AuditIntegrationError(
                    f"Reuse source {source.label} disagrees on prompt/answer "
                    f"for fact {key!r} despite matching prompt digests."
                )
            embeddings = source.embeddings(key, state)
            if (
                require_embeddings
                and not source.has_embeddings_sidecar
                and (row.get("retrieval_trace") or {}).get("retrieval_events")
            ):
                # The source captured events but its sidecar is gone; serving
                # would silently thin this phase's embedding sidecar.
                continue

            if self._canary_selected(key, state):
                subject = dataclasses.replace(
                    example, deletion_manifest=manifest
                )
                generated = audit_example(
                    self._backend,
                    subject,
                    state,
                    max_new_tokens=self._max_new_tokens,
                )
                generated.pop("_query_embeddings", None)
                self.canary_checks += 1
                if generated["model_output"] != row["model_output"]:
                    self._canary_failure(
                        key=key,
                        state=state,
                        source=source,
                        source_row=row,
                        generated_row=generated,
                        manifest=manifest,
                    )

            stamped = json.loads(json.dumps(row, ensure_ascii=False))
            stamped["deletion_manifest"] = manifest.as_dict()
            trace = stamped.get("retrieval_trace")
            if isinstance(trace, dict):
                trace["deletion_manifest_id"] = manifest.manifest_id
            if embeddings:
                stamped["_query_embeddings"] = embeddings
            self.served[state.value] += 1
            self.entries.append(
                {"fact_key": key, "state": state.value, "source": source.label}
            )
            return stamped
        return None

    def note_generated(self, state: DatabaseState) -> None:
        self.generated[state.value] += 1

    def summary(self) -> dict[str, Any]:
        return {
            "served": sum(self.served.values()),
            "generated": sum(self.generated.values()),
            "canary_checks": self.canary_checks,
            "served_by_state": dict(self.served),
            "generated_by_state": dict(self.generated),
            "sources": [source.label for source in self._sources],
        }

    def write_manifest(self, path: Path) -> None:
        atomic_write_json(path, {**self.summary(), "entries": self.entries})
