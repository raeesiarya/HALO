from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Callable

from halo.core.backend import AuditBackend


def _no_validate(args: argparse.Namespace) -> None:
    return None


def _no_arguments(parser: argparse.ArgumentParser) -> None:
    return None


@dataclass(frozen=True)
class BackendSpec:
    """How the audit application builds and drives one model backend.

    The audit core (``core``/``interventions``) knows nothing about concrete
    models; each model package registers a spec, and the CLI dispatches
    through the registry. Adding a model means dropping a package under
    ``src/models`` that registers a spec — no edits to the audit core.
    """

    name: str
    # Build the backend for one job group. `group_key` is whatever
    # `group_key(args, job)` returned (e.g. a database path for rel-LMLM, the
    # index path for Co-LMLM).
    build_backend: Callable[[argparse.Namespace, Any], AuditBackend]
    # The search index the closure/sweep/adversarial machinery drives.
    build_search_index: Callable[[AuditBackend], Any]
    # Jobs sharing a group_key reuse one backend instance.
    group_key: Callable[[argparse.Namespace, Any], Any]
    # Registers this backend's own CLI arguments (artifact paths and genuine
    # research knobs — anything with no universal default). The CLI adds only
    # model-agnostic flags; each model contributes the rest here.
    add_arguments: Callable[[argparse.ArgumentParser], None] = _no_arguments
    # Backend-specific argument validation (missing paths, unsupported
    # closure predicates, ...). Generic audit-flag validation stays in the CLI.
    validate: Callable[[argparse.Namespace], None] = _no_validate


_REGISTRY: dict[str, BackendSpec] = {}


def register_backend(spec: BackendSpec) -> None:
    _REGISTRY[spec.name] = spec


def get_backend_spec(name: str) -> BackendSpec:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"No audit backend registered under {name!r}. "
            f"Available: {available_backends()}."
        ) from None


def available_backends() -> list[str]:
    return sorted(_REGISTRY)
