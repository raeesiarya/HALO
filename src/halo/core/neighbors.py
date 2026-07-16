from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

VALID_NEIGHBOR_MODES = ("cosine", "same-source")


@dataclass(frozen=True)
class NeighborConfig:
    mode: str = "cosine"
    ball: float = 0.5
    cap: int = 20

    def __post_init__(self) -> None:
        if self.mode not in VALID_NEIGHBOR_MODES:
            raise ValueError(
                f"Unknown neighbor mode {self.mode!r}; valid modes are "
                f"{list(VALID_NEIGHBOR_MODES)!r}."
            )
        if not -1.0 <= self.ball <= 1.0:
            raise ValueError("ball must be a cosine similarity in [-1, 1].")
        if self.cap < 1:
            raise ValueError("cap must be at least 1.")


def compute_cosine_neighbors(
    embeddings: Mapping[str, Any],
    config: NeighborConfig,
) -> dict[str, list[dict[str, Any]]]:
    keys = sorted(embeddings)
    vectors = {}
    for key in keys:
        vector = np.asarray(embeddings[key], dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if norm > 0.0:
            vectors[key] = vector / norm

    neighbors: dict[str, list[dict[str, Any]]] = {key: [] for key in keys}
    for key in keys:
        if key not in vectors:
            # No usable embedding: an explicit empty neighbor set, never a
            # silent skip.
            continue
        scored = []
        for other in keys:
            if other == key or other not in vectors:
                continue
            cosine = float(np.dot(vectors[key], vectors[other]))
            if cosine >= config.ball:
                scored.append({"neighbor": other, "cosine": cosine})
        scored.sort(key=lambda item: (-item["cosine"], item["neighbor"]))
        neighbors[key] = scored[: config.cap]
    return neighbors


def compute_same_source_neighbors(
    sources: Mapping[str, str | None],
    config: NeighborConfig,
) -> dict[str, list[dict[str, Any]]]:
    keys = sorted(sources)
    neighbors: dict[str, list[dict[str, Any]]] = {key: [] for key in keys}
    for key in keys:
        source = sources[key]
        if source is None:
            continue
        matches = [
            {"neighbor": other, "cosine": None}
            for other in keys
            if other != key and sources[other] == source
        ]
        neighbors[key] = matches[: config.cap]
    return neighbors


def neighbor_keys(
    neighbors: Mapping[str, list[dict[str, Any]]],
) -> dict[str, list[str]]:
    return {
        key: [item["neighbor"] for item in items]
        for key, items in neighbors.items()
    }


def write_neighbors_file(
    neighbors: Mapping[str, list[dict[str, Any]]],
    config: NeighborConfig,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "mode": config.mode,
            "ball": config.ball,
            "cap": config.cap,
        },
        "neighbors": {key: list(items) for key, items in neighbors.items()},
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
