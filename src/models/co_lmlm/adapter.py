"""How Co-LMLM plugs into the audit's deletion/search machinery."""

from __future__ import annotations

from typing import Any


def build_search_index(backend: Any) -> Any:
    # The public generator's index already implements the search interface
    # the closure/sweep/adversarial machinery expects.
    return backend.generator.index
