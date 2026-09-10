"""Artifact plumbing for the NULLs audit: the embedding builder's corpus
streaming and the sink registry's memory-lean artifact loading."""

from __future__ import annotations

import importlib.util
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from models.nulls_wiki.routing import SinkRegistry, _load_embeddings  # noqa: E402


def _load_builder():
    spec = importlib.util.spec_from_file_location(
        "build_nulls_title_embeddings",
        REPO / "scripts" / "build_nulls_title_embeddings.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _StubEncoder:
    """Deterministic stand-in for SentenceTransformer: dim 4, vector = text
    length one-hot-ish so rows are distinguishable."""

    def get_sentence_embedding_dimension(self):
        return 4

    def encode(self, texts, batch_size, show_progress_bar, convert_to_numpy, normalize_embeddings):
        out = np.zeros((len(texts), 4), dtype=np.float32)
        for i, text in enumerate(texts):
            out[i, len(text) % 4] = 1.0
        return out


class TestBuilderCorpusStreaming:
    def test_title_positions_follow_insertion_order_not_values(self, tmp_path):
        builder = _load_builder()
        mapping = tmp_path / "t2i.pkl"
        # Values deliberately not equal to positions: the contract is insertion order.
        mapping.write_bytes(pickle.dumps({"B": 10, "A": 20, "C": 5}))
        assert builder.load_title_positions(mapping) == {"B": 0, "A": 1, "C": 2}

    def test_jsonl_batches_truncate_per_row(self, tmp_path):
        builder = _load_builder()
        corpus = tmp_path / "corpus.jsonl"
        corpus.write_text(
            json.dumps({"title": "A", "text": "x" * 50}) + "\n"
            + json.dumps({"title": "B", "text": "y" * 3}) + "\n"
        )
        batches = list(builder.iter_text_batches(corpus, max_chars=10))
        assert batches == [(["A", "B"], ["x" * 10, "y" * 3])]

    def test_parquet_directory_streams_all_shards(self, tmp_path):
        pq = pytest.importorskip("pyarrow.parquet")
        pa = pytest.importorskip("pyarrow")
        builder = _load_builder()
        shards = tmp_path / "train_raw" / "en"
        shards.mkdir(parents=True)
        for index, (title, text) in enumerate([("A", "aaaa"), ("B", "bb")]):
            table = pa.table({"id": [str(index)], "url": ["u"], "title": [title], "text": [text]})
            pq.write_table(table, shards / f"train-0000{index}-of-00002.parquet")
        batches = list(builder.iter_text_batches(tmp_path / "train_raw", max_chars=3))
        assert [b[0] for b in batches] == [["A"], ["B"]]
        assert [b[1] for b in batches] == [["aaa"], ["bb"]]

    def test_encode_closure_places_rows_by_position_and_reports_coverage(self):
        builder = _load_builder()
        positions = {"A": 0, "B": 1, "C": 2}
        batches = iter([(["B", "Z"], ["bb", "ignored: not a source"]), (["A"], ["a"])])
        embeddings, covered = builder.encode_closure(
            positions, batches, encoder=_StubEncoder(), batch_size=8
        )
        assert embeddings.shape == (3, 4)
        assert covered.tolist() == [True, True, False]
        assert embeddings[0].tolist() == [0.0, 1.0, 0.0, 0.0]  # len("a") == 1
        assert embeddings[1].tolist() == [0.0, 0.0, 1.0, 0.0]  # len("bb") == 2
        assert not embeddings[2].any()


class TestRegistryArtifactLoading:
    def test_npz_rows_are_normalized_in_place_as_float32(self, tmp_path):
        artifact = tmp_path / "routing.npz"
        raw = np.array([[3.0, 4.0], [0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
        np.savez(artifact, embeddings=raw, encoder=np.str_("stub"), mode=np.str_("routing"))
        registry = SinkRegistry.load(
            title_to_index_path=_mapping(tmp_path, ["A", "B", "C"]),
            vocab_size=49152,
            embeddings_path=artifact,
        )
        assert registry.embeddings.dtype == np.float32
        assert registry.embeddings.flags.c_contiguous
        np.testing.assert_allclose(registry.embeddings[0], [0.6, 0.8], rtol=1e-6)
        assert registry.embeddings[1].tolist() == [0.0, 0.0]  # zero row left alone
        assert registry.embedding_encoder == "stub"
        assert registry.seq_id("C") == 2 + 49152

    def test_closure_space_artifact_rejected_for_routing(self, tmp_path):
        artifact = tmp_path / "closure.npz"
        np.savez(artifact, embeddings=np.eye(2, dtype=np.float32), mode=np.str_("closure"))
        with pytest.raises(ValueError, match="closure"):
            _load_embeddings(artifact)

    def test_legacy_npz_without_mode_still_loads(self, tmp_path):
        artifact = tmp_path / "legacy.npz"
        np.savez(artifact, embeddings=np.eye(2, dtype=np.float32))
        embeddings, encoder = _load_embeddings(artifact)
        assert embeddings.shape == (2, 2) and encoder is None


def _mapping(tmp_path: Path, titles: list[str]) -> Path:
    path = tmp_path / "t2i.pkl"
    path.write_bytes(pickle.dumps({title: i for i, title in enumerate(titles)}))
    return path
