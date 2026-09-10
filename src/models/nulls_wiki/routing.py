"""Sink identity and routing for the NULLs backend.

The registry owns the two released-artifact contracts:

- ``title_to_index``: the training-time mapping from Wikipedia article title
  to integer index. A source's sink seq id is ``index + vocab_size``,
  matching upstream's tokenizer-side assignment (`tokenize_wikipedia_topic.py`)
  and eval-side lookup (`compute_truth_ratio.py`). Seq-id tensors must be
  int32: upstream multiplies ``seq_id * 2**16`` in int32, and the wraparound
  for high article indices is part of the trained mask contract.
- ``title embeddings``: one vector per title, row-aligned with the insertion
  order of ``title_to_index`` (upstream aligns rows with
  ``list(title_to_index.keys())``). Used for next-closest routing (DEL-ON)
  and the placebo-sink control (DEL-OFF sensitivity).

Neither artifact is derivable from the checkpoint. The embeddings are
regenerated from the title list by ``scripts/build_nulls_title_embeddings.py``;
the mapping must be the authors' own artifact, and the verification gate
(``scripts/nulls_verification_gate.py``) checks per fact that it actually
addresses the expected sink.
"""

from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from halo.core.embeddings import normalize_rows_inplace


DEFAULT_PLACEBO_SIMILARITY_CEILING = 0.5
_PLACEBO_MAX_ATTEMPTS = 10_000


@dataclass
class SinkRegistry:
    """Title -> sink identity, plus embedding-space neighbor queries."""

    title_to_index: Mapping[str, int]
    vocab_size: int
    # (num_titles, dim) float32, L2-normalized rows, aligned with
    # ``list(title_to_index)`` order. None disables neighbor/placebo queries.
    embeddings: np.ndarray | None = None
    embedding_encoder: str | None = None
    _titles: tuple[str, ...] = field(init=False, repr=False)
    _title_pos: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._titles = tuple(self.title_to_index)
        self._title_pos = {title: pos for pos, title in enumerate(self._titles)}
        if self.embeddings is not None:
            if self.embeddings.shape[0] != len(self._titles):
                raise ValueError(
                    f"Embedding rows ({self.embeddings.shape[0]}) do not match "
                    f"title count ({len(self._titles)}); the artifact must be "
                    "row-aligned with title_to_index insertion order."
                )
            # The released mapping has 6.4M sources, so the routing matrix is
            # ~10 GB per process (one process per shard job): normalize in
            # place, chunked, instead of materializing two full copies.
            self.embeddings = normalize_rows_inplace(self.embeddings)

    @classmethod
    def load(
        cls,
        *,
        title_to_index_path: str | Path,
        vocab_size: int,
        embeddings_path: str | Path | None = None,
    ) -> "SinkRegistry":
        with open(title_to_index_path, "rb") as handle:
            title_to_index = pickle.load(handle)
        if not isinstance(title_to_index, Mapping):
            raise TypeError(
                f"{title_to_index_path} must unpickle to a title->index mapping, "
                f"got {type(title_to_index).__name__}."
            )
        embeddings = None
        encoder = None
        if embeddings_path is not None:
            embeddings, encoder = _load_embeddings(embeddings_path)
        return cls(
            title_to_index=title_to_index,
            vocab_size=vocab_size,
            embeddings=embeddings,
            embedding_encoder=encoder,
        )

    def __contains__(self, title: str) -> bool:
        return title in self.title_to_index

    def __len__(self) -> int:
        return len(self._titles)

    def seq_id(self, title: str) -> int:
        try:
            index = self.title_to_index[title]
        except KeyError:
            raise KeyError(
                f"Title {title!r} is not in title_to_index; the fact should "
                "have been excluded by the verification gate "
                "(scripts/nulls_verification_gate.py)."
            ) from None
        return int(index) + int(self.vocab_size)

    def _require_embeddings(self) -> np.ndarray:
        if self.embeddings is None:
            raise ValueError(
                "This SinkRegistry has no title embeddings; pass "
                "--nulls-title-embeddings (build one with "
                "scripts/build_nulls_title_embeddings.py)."
            )
        return self.embeddings

    def similarity(self, title_a: str, title_b: str) -> float:
        embeddings = self._require_embeddings()
        return float(
            embeddings[self._title_pos[title_a]] @ embeddings[self._title_pos[title_b]]
        )

    def nearest(
        self,
        title: str,
        *,
        exclude: Sequence[str] = (),
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Surviving nearest neighbors of ``title``, most similar first.

        ``title`` itself and everything in ``exclude`` are never returned, so
        with ``exclude = deletion manifest`` the first element is the DEL-ON
        routing target (the paper's next-closest surviving source).
        """
        embeddings = self._require_embeddings()
        scores = embeddings @ embeddings[self._title_pos[title]]
        excluded_positions = {self._title_pos[title]}
        for excluded_title in exclude:
            position = self._title_pos.get(excluded_title)
            if position is not None:
                excluded_positions.add(position)
        # Over-fetch so the excluded entries never starve the result.
        fetch = min(len(self._titles), top_k + len(excluded_positions) + 1)
        candidate_positions = np.argpartition(-scores, fetch - 1)[:fetch]
        ranked = candidate_positions[np.argsort(-scores[candidate_positions])]
        neighbors = [
            (self._titles[position], float(scores[position]))
            for position in ranked
            if position not in excluded_positions
        ]
        return neighbors[:top_k]

    def placebo(
        self,
        fact_key: str,
        *,
        avoid: Sequence[str] = (),
        anchor_title: str | None = None,
        similarity_ceiling: float = DEFAULT_PLACEBO_SIMILARITY_CEILING,
    ) -> str:
        """A deterministic, unrelated placebo title for ``fact_key``.

        The choice is a pure function of (fact_key, registry contents): a
        sha256 stream over ``fact_key`` indexes into the title list, skipping
        titles in ``avoid`` and, when an anchor and embeddings are available,
        titles whose cosine similarity to the anchor exceeds
        ``similarity_ceiling`` (the placebo must be far from the audited
        source, or it is not a placebo).
        """
        avoided = set(avoid)
        if anchor_title is not None:
            avoided.add(anchor_title)
        use_similarity = (
            self.embeddings is not None
            and anchor_title is not None
            and anchor_title in self._title_pos
        )
        if use_similarity:
            anchor_scores = self._require_embeddings() @ self._require_embeddings()[
                self._title_pos[anchor_title]
            ]
        for attempt in range(_PLACEBO_MAX_ATTEMPTS):
            digest = hashlib.sha256(f"{fact_key}::{attempt}".encode()).digest()
            position = int.from_bytes(digest[:8], "big") % len(self._titles)
            candidate = self._titles[position]
            if candidate in avoided:
                continue
            if use_similarity and float(anchor_scores[position]) > similarity_ceiling:
                continue
            return candidate
        raise RuntimeError(
            f"No placebo title found for {fact_key!r} within "
            f"{_PLACEBO_MAX_ATTEMPTS} attempts; similarity ceiling "
            f"{similarity_ceiling} may be too strict for this registry."
        )


def _load_embeddings(path: str | Path) -> tuple[np.ndarray, str | None]:
    """Load a title-embedding artifact (.npz with 'embeddings' [+ 'encoder',
    'mode'], or upstream's raw-pickled matrix).

    Rejects closure-space artifacts: routing (titles, MiniLM) and closure
    (article texts, shared E) are distinct spaces by design (§4) and must
    never be swapped, in either direction.
    """
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if "mode" in archive.files and str(archive["mode"]) != "routing":
                raise ValueError(
                    f"{path} is a {str(archive['mode'])!r}-space artifact; the "
                    "sink registry needs the routing space "
                    "(scripts/build_nulls_title_embeddings.py --mode routing)."
                )
            embeddings = archive["embeddings"]
            if embeddings.dtype != np.float32:
                embeddings = embeddings.astype(np.float32)
            encoder = None
            if "encoder" in archive.files:
                encoder = str(archive["encoder"])
        return embeddings, encoder
    with open(path, "rb") as handle:
        raw = pickle.load(handle)
    if hasattr(raw, "numpy"):  # torch tensor from upstream's title_embeddings.pkl
        raw = raw.numpy()
    return np.asarray(raw, dtype=np.float32), None
