"""Shared, resumable FULL-pass store.

One FULL generation per fact, persisted incrementally and shared by every
consumer: the entanglement sweep and adversarial evaluation (rows + the
selected-event query vector) and the standard audit (rows + every captured
event vector, re-attached as ``_query_embeddings`` so closure building and
the query-embedding sidecar behave exactly as with a live generation).

Artifacts in the full dir:

- ``full_results.jsonl``            canonical rows, prompt order (existence
                                    means the pass is complete)
- ``full_query_embeddings.npz``     flat ``{fact_key}`` -> selected-event
                                    vector (legacy format, kept for the
                                    sweep/adversarial consumers)
- ``full_all_query_embeddings.npz`` ``{fact_key}/FULL/event{n}`` -> every
                                    captured event vector
- ``full_config.json``              resume-identity guard
- ``*.partial.jsonl``               append-only in-progress state, deleted
                                    at canonicalization
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from tqdm import tqdm

from halo.cli.persistence import (
    AUDIT_SCHEMA_VERSION,
    PartialLog,
    backend_resume_identity,
    ensure_resume_config,
    partial_name,
    partial_paths,
    prompt_digest,
    read_partial_jsonl,
)
from halo.core.backend import AuditBackend, audit_example
from halo.core.entanglement import fact_key
from halo.core.examples import AuditExample
from halo.core.states import DatabaseState
from halo.interventions.errors import AuditIntegrationError

_ROWS_STEM = "full_results"
_VECTORS_STEM = "full_query_embeddings"


def _all_events_key(key: str, event_index: int) -> str:
    return f"{key}/FULL/event{event_index}"


def _validate_example_keys(examples: dict[str, AuditExample]) -> None:
    """Persistence keys must round-trip through `fact_key` on the stored row."""
    for key, example in examples.items():
        expected = fact_key(
            {"prompt_id": example.prompt_id, "fact_id": example.fact_id}
        )
        if not expected or expected != key:
            raise AuditIntegrationError(
                "Resumable runs require a unique prompt_id/fact_id per prompt "
                f"row; got key {key!r} with persisted identity {expected!r}."
            )
        if "/" in key:
            raise AuditIntegrationError(
                f"Fact key {key!r} contains '/', which collides with the "
                "npz sidecar key convention {key}/{state}/event{n}."
            )


class FullPassStore:
    """FULL rows and their query vectors, generated on demand and persisted
    incrementally. ``get``/``generate`` return deep copies with
    ``_query_embeddings`` re-attached; internal state stays pristine."""

    def __init__(
        self,
        *,
        backend: AuditBackend,
        examples: dict[str, AuditExample],
        prompt_path: Path,
        limit: int | None,
        output_dir: Path,
        max_new_tokens: int,
        shard: tuple[int, int] | None = None,
    ) -> None:
        _validate_example_keys(examples)
        self._backend = backend
        self._examples = examples
        self._max_new_tokens = max_new_tokens
        self._shard = shard
        self.output_dir = output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        self._rows_path = output_dir / "full_results.jsonl"
        self._flat_npz_path = output_dir / "full_query_embeddings.npz"
        self._all_npz_path = output_dir / "full_all_query_embeddings.npz"
        ensure_resume_config(
            output_dir / "full_config.json",
            {
                "audit_schema_version": AUDIT_SCHEMA_VERSION,
                "mode": "full-pass",
                "backend": backend_resume_identity(backend),
                "prompt_sha256": prompt_digest(prompt_path),
                "limit": limit,
                "max_new_tokens": max_new_tokens,
            },
            legacy_artifacts=(self._rows_path, self._flat_npz_path),
        )

        self.hits = 0
        self.generated_count = 0
        self._rows: dict[str, dict[str, Any]] = {}
        self._events: dict[str, list[tuple[int, np.ndarray]]] = {}
        # None => complete legacy dir without the all-events sidecar; only
        # `get` consumers (the standard audit) need it, so only they raise.
        self._events_available = True
        self._row_log: PartialLog | None = None
        self._vector_log: PartialLog | None = None

        if self._rows_path.exists():
            self.complete = True
            for row in self._iter_jsonl(self._rows_path):
                self._rows[fact_key(row)] = row
            if self._all_npz_path.exists():
                with np.load(self._all_npz_path) as stored:
                    for stored_key in stored.files:
                        key, _, event = stored_key.rsplit("/", 2)
                        self._events.setdefault(key, []).append(
                            (int(event.removeprefix("event")), stored[stored_key])
                        )
            else:
                self._events_available = False
        else:
            self.complete = False
            for path in partial_paths(output_dir, _VECTORS_STEM):
                for item in read_partial_jsonl(path):
                    self._events.setdefault(str(item["key"]), []).append(
                        (
                            int(item["event_index"]),
                            np.asarray(item["vector"], dtype=np.float32),
                        )
                    )
            for path in partial_paths(output_dir, _ROWS_STEM):
                for row in read_partial_jsonl(path):
                    self._rows.setdefault(fact_key(row), row)
            # Vector lines are written before their row line, so a stored row
            # implies its vectors; orphan vectors belong to a torn fact.
            self._events = {
                key: events
                for key, events in self._events.items()
                if key in self._rows
            }
            self._row_log = PartialLog(
                output_dir / partial_name(_ROWS_STEM, shard)
            )
            self._vector_log = PartialLog(
                output_dir / partial_name(_VECTORS_STEM, shard)
            )

    @staticmethod
    def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)

    def _attach(self, key: str, row: dict[str, Any]) -> dict[str, Any]:
        copy = json.loads(json.dumps(row, ensure_ascii=False))
        events = sorted(self._events.get(key, []))
        if events:
            copy["_query_embeddings"] = [
                {"event_index": index, "vector": vector}
                for index, vector in events
            ]
        return copy

    def get(self, key: str) -> dict[str, Any] | None:
        row = self._rows.get(key)
        if row is None:
            return None
        if not self._events_available:
            raise AuditIntegrationError(
                f"{self.output_dir} is a complete FULL pass from before "
                "full_all_query_embeddings.npz existed; regenerate it into a "
                "fresh full dir to serve the standard audit."
            )
        self.hits += 1
        return self._attach(key, row)

    def generate(self, key: str, example: AuditExample) -> dict[str, Any]:
        if self.complete:
            raise AuditIntegrationError(
                f"FULL pass in {self.output_dir} is complete; "
                f"{key!r} is not part of it (prompt/limit drift?)."
            )
        row = audit_example(
            self._backend,
            example,
            DatabaseState.FULL,
            max_new_tokens=self._max_new_tokens,
        )
        embeddings = row.pop("_query_embeddings", None) or []
        events = [
            (int(item["event_index"]), np.asarray(item["vector"], dtype=np.float32))
            for item in embeddings
        ]
        for index, vector in events:
            self._vector_log.write(
                {"key": key, "event_index": index, "vector": vector.tolist()}
            )
        self._row_log.write(row)
        self._rows[key] = row
        if events:
            self._events[key] = events
        self.generated_count += 1
        return self._attach(key, row)

    def get_or_generate(self, key: str, example: AuditExample) -> dict[str, Any]:
        return self.get(key) or self.generate(key, example)

    def missing_keys(self) -> list[str]:
        return [key for key in self._examples if key not in self._rows]

    def selected_vector(self, key: str) -> np.ndarray | None:
        """The vector of the row's audited retrieval event (legacy flat view)."""
        row = self._rows.get(key)
        if row is None:
            return None
        trace = row.get("retrieval_trace") or {}
        selected = trace.get("selected_event_index")
        if not isinstance(selected, int):
            return None
        for index, vector in self._events.get(key, []):
            if index == selected:
                return vector
        return None

    def rows_and_vectors(
        self,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray]]:
        """The legacy `_full_pass` return contract: rows without embeddings,
        flat selected-event vectors."""
        rows = {key: self._rows[key] for key in self._examples if key in self._rows}
        vectors = {}
        for key in rows:
            vector = self.selected_vector(key)
            if vector is not None:
                vectors[key] = vector
        return rows, vectors

    def close(self) -> None:
        if self._row_log is not None:
            self._row_log.close()
        if self._vector_log is not None:
            self._vector_log.close()

    def finalize(self, *, progress_desc: str = "FULL pass") -> None:
        """Generate anything missing, then write the canonical artifacts and
        drop the partials. Not valid for shard runs (the finalize invocation
        is the unsharded one that sees every shard's partials)."""
        if self.complete:
            return
        if self._shard is not None:
            raise AuditIntegrationError(
                "A shard run cannot finalize the FULL pass; run the unsharded "
                "invocation to merge and canonicalize."
            )
        missing = self.missing_keys()
        if missing:
            for key in tqdm(missing, desc=progress_desc, unit="prompt"):
                self.generate(key, self._examples[key])
        self.close()
        rows, vectors = self.rows_and_vectors()
        with self._rows_path.open("w", encoding="utf-8") as handle:
            for row in rows.values():
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        if vectors:
            np.savez_compressed(self._flat_npz_path, **vectors)
        np.savez_compressed(
            self._all_npz_path,
            **{
                _all_events_key(key, index): vector
                for key, events in self._events.items()
                for index, vector in events
            },
        )
        for stem in (_ROWS_STEM, _VECTORS_STEM):
            for path in partial_paths(self.output_dir, stem):
                path.unlink()
        self.complete = True


def run_full_pass(
    backend: AuditBackend,
    examples: dict[str, AuditExample],
    prompt_path: Path,
    limit: int | None,
    output_dir: Path,
    max_new_tokens: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray]]:
    """FULL over every prompt, resumable per row. Drop-in replacement for the
    former wholesale `_full_pass`."""
    store = FullPassStore(
        backend=backend,
        examples=examples,
        prompt_path=prompt_path,
        limit=limit,
        output_dir=output_dir,
        max_new_tokens=max_new_tokens,
    )
    store.finalize()
    return store.rows_and_vectors()
