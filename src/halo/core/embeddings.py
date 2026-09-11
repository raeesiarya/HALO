from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np


def normalize_rows_inplace(matrix: np.ndarray, chunk_rows: int = 500_000) -> np.ndarray:
    """L2-normalize the rows of ``matrix`` without a second full-size copy.

    Source-embedding artifacts (one row per Wikipedia article, 6.4M rows for
    the NULLs release) are 10-20 GB; the naive ``(m / norms).astype(f32)``
    materializes two more copies of that. This converts to a writable
    float32 C array only when the input is not one already — the caller's
    array is then normalized in place — and walks the rows in chunks so the
    temporaries stay at chunk size. Zero rows are left as zeros.
    """
    if (
        matrix.dtype != np.float32
        or not matrix.flags.writeable
        or not matrix.flags.c_contiguous
    ):
        matrix = np.ascontiguousarray(matrix, dtype=np.float32)
    for start in range(0, matrix.shape[0], chunk_rows):
        block = matrix[start : start + chunk_rows]
        norms = np.linalg.norm(block, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        block /= norms
    return matrix


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
