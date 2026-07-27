"""Incremental persistence primitives for resumable audit runs.

Canonical artifacts (results JSONL, npz sidecars, CSVs) are only ever written
whole, at completion, exactly as a non-resumable run would write them.
Everything incremental lives in append-only ``*.partial.jsonl`` files that are
deleted after the canonical artifacts are materialized, so the existence of a
canonical file always means "complete".
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, IO

from halo.interventions.errors import AuditIntegrationError

AUDIT_SCHEMA_VERSION = 2


def atomic_write_text(path: Path, text: str) -> None:
    """Write via a same-directory temp file + rename so concurrent readers
    never observe a torn file and concurrent identical writers converge."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: Path, payload: Any, *, indent: int = 2) -> None:
    atomic_write_text(
        path, json.dumps(payload, indent=indent, sort_keys=True, default=str)
    )


def parse_shard(spec: str) -> tuple[int, int]:
    """Parse an ``I/N`` shard spec into (index, count)."""
    parts = spec.split("/")
    if len(parts) != 2:
        raise ValueError(f"Shard spec must be 'I/N', got {spec!r}.")
    try:
        index, count = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise ValueError(f"Shard spec must be 'I/N' integers, got {spec!r}.") from exc
    if count < 1 or not 0 <= index < count:
        raise ValueError(f"Shard spec needs 0 <= I < N, got {spec!r}.")
    return index, count


def partial_name(stem: str, shard: tuple[int, int] | None) -> str:
    if shard is None:
        return f"{stem}.partial.jsonl"
    index, count = shard
    return f"{stem}.shard{index}of{count}.partial.jsonl"


def partial_paths(directory: Path, stem: str) -> list[Path]:
    """Every partial for ``stem`` (unsharded first, shards in sorted order)."""
    return sorted(
        set(directory.glob(f"{stem}.partial.jsonl"))
        | set(directory.glob(f"{stem}.shard*.partial.jsonl"))
    )


def read_partial_jsonl(path: Path) -> list[dict[str, Any]]:
    """Rows from an append-only partial file.

    A crash can tear only the final line, so a decode error there drops the
    line; a decode error anywhere else means real corruption and raises.
    """
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()
    for line_number, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            rows.append(json.loads(stripped))
        except json.JSONDecodeError:
            if line_number == len(lines) - 1:
                break
            raise AuditIntegrationError(
                f"Corrupt partial file {path} at line {line_number + 1}; "
                "only the final line of a partial may be torn."
            )
    return rows


class PartialLog:
    """Lazily opened append-only JSONL log, flushed per line so a crash
    loses at most the line being written."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle: IO[str] | None = None

    def write(self, obj: Any) -> None:
        if self._handle is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open("a", encoding="utf-8")
        self._handle.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self._handle.flush()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def prompt_digest(prompt_path: Path) -> str:
    digest = hashlib.sha256()
    with prompt_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def backend_resume_identity(backend: Any) -> dict[str, Any]:
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


def ensure_resume_config(
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
    atomic_write_text(
        path, json.dumps(normalized, indent=2, sort_keys=True)
    )
