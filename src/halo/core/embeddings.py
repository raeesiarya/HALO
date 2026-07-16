from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np


def result_example_key(result_row: Mapping[str, Any], row_index: int) -> str:
    for field_name in ("prompt_id", "fact_id"):
        value = result_row.get(field_name)
        if value is not None:
            return str(value)
    return f"row{row_index}"


class QueryEmbeddingSink:
    """Accumulates raw per-retrieval-event query vectors and writes one
    compressed .npz sidecar per prompt file.

    Keys have the form ``{example_key}/{state}/event{n}``. Vectors are stored
    as emitted by the model (unnormalized); the index applies L2
    normalization at search time, so consumers that need cosine geometry
    should normalize on load.
    """

    def __init__(self) -> None:
        self._vectors: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self._vectors)

    def add(
        self,
        *,
        example_key: str,
        state: str,
        event_index: int,
        vector: Any,
    ) -> None:
        key = f"{example_key}/{state}/event{event_index}"
        if key in self._vectors:
            raise ValueError(
                f"Duplicate query-embedding key {key!r}; prompt rows must have "
                "unique prompt_id/fact_id values."
            )
        self._vectors[key] = np.asarray(vector, dtype=np.float32).reshape(-1)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **self._vectors)
