"""Streaming access to the NULLs training corpus (``gauravrghosal/
wiki_nulls_corpus``: the unmodified ``wikimedia/wikipedia`` ``20231101.en``
snapshot, 41 parquet shards with ``title``/``text`` columns, 11.6 GB).

Every consumer streams it — nothing here ever holds the corpus in memory:
the closure-embedding builder encodes a row batch at a time, and the
value-source closure builder checks answer mentions per row and keeps only
the hits. ``iter_text_batches`` accepts a directory of parquet shards, one
parquet file, or a jsonl file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

ROWS_PER_BATCH = 8192


def corpus_files(path: Path) -> list[Path]:
    path = Path(path)
    if path.is_dir():
        files = sorted(path.rglob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No .parquet shards under {path}.")
        return files
    if not path.is_file():
        raise FileNotFoundError(f"{path} is neither a corpus directory nor a file.")
    return [path]


def iter_text_batches(
    path: Path,
    *,
    max_chars: int | None = None,
    rows_per_batch: int = ROWS_PER_BATCH,
) -> Iterator[tuple[list[str], list[str]]]:
    """Yield (titles, texts) batches; ``max_chars`` cuts each text as it is
    read so long articles never sit in memory whole."""

    def cut(text: object) -> str:
        text = str(text or "")
        return text if max_chars is None else text[:max_chars]

    for file in corpus_files(path):
        if file.suffix == ".parquet":
            import pyarrow.parquet as pq

            reader = pq.ParquetFile(file)
            for batch in reader.iter_batches(
                columns=["title", "text"], batch_size=rows_per_batch
            ):
                yield (
                    batch.column("title").to_pylist(),
                    [cut(text) for text in batch.column("text").to_pylist()],
                )
            continue
        titles: list[str] = []
        texts: list[str] = []
        with open(file, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                titles.append(str(row["title"]))
                texts.append(cut(row.get("text")))
                if len(titles) >= rows_per_batch:
                    yield titles, texts
                    titles, texts = [], []
        if titles:
            yield titles, texts
