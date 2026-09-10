"""NULLs backend, sink routing, source closures, and scheduler wiring."""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from halo.core.backend import audit_example, validate_intervention_results
from halo.core.examples import AuditExample
from halo.core.states import DatabaseState
from halo.interventions.source_closure import (
    SourceSpace,
    build_closure_prompt_file,
    expand_row,
)
from halo.registry import available_backends, get_backend_spec
from models.nulls_wiki.backend import NullsWikiAuditBackend
from models.nulls_wiki.routing import SinkRegistry
from models.nulls_wiki.vendor.masking import (
    batch_seqtied_mask_mult,
    union_exclusion_mask,
)

# ---------------------------------------------------------------------------
# Vendored mask arithmetic


class TestMasking:
    def test_deterministic(self):
        ids = torch.full((1, 4), 49_152 + 123, dtype=torch.int32)
        first = batch_seqtied_mask_mult(ids, 512, 0.013)
        second = batch_seqtied_mask_mult(ids, 512, 0.013)
        assert torch.equal(first, second)

    def test_density_tracks_p_active(self):
        ids = torch.full((1, 1), 49_152 + 777, dtype=torch.int32)
        mask = batch_seqtied_mask_mult(ids, 8000, 0.013)
        density = float(mask.float().mean())
        assert 0.004 < density < 0.03

    def test_distinct_sources_get_distinct_masks(self):
        mask_a = batch_seqtied_mask_mult(
            torch.full((1, 1), 49_152 + 1, dtype=torch.int32), 8000, 0.013
        )
        mask_b = batch_seqtied_mask_mult(
            torch.full((1, 1), 49_152 + 2, dtype=torch.int32), 8000, 0.013
        )
        assert not torch.equal(mask_a, mask_b)

    def test_int32_wraparound_is_part_of_the_contract(self):
        # High article indices overflow int32 under `seq_id * 2**16`; the
        # trained masks rely on that exact wrap, so int32 and int64 inputs
        # must NOT be interchangeable for large ids.
        big = 49_152 + 5_000_000
        mask_int32 = batch_seqtied_mask_mult(
            torch.full((1, 1), big, dtype=torch.int32), 2048, 0.1
        )
        mask_int64 = batch_seqtied_mask_mult(
            torch.full((1, 1), big, dtype=torch.int64), 2048, 0.1
        )
        assert not torch.equal(mask_int32, mask_int64)

    def test_union_single_equals_upstream_path(self):
        ids = torch.full((2, 3), 49_152 + 42, dtype=torch.int32)
        assert torch.equal(
            union_exclusion_mask([ids], 512, 0.05),
            batch_seqtied_mask_mult(ids, 512, 0.05),
        )

    def test_union_is_elementwise_or(self):
        ids_a = torch.full((1, 2), 49_152 + 7, dtype=torch.int32)
        ids_b = torch.full((1, 2), 49_152 + 8, dtype=torch.int32)
        union = union_exclusion_mask([ids_a, ids_b], 512, 0.05)
        expected = torch.logical_or(
            batch_seqtied_mask_mult(ids_a, 512, 0.05),
            batch_seqtied_mask_mult(ids_b, 512, 0.05),
        )
        assert torch.equal(union, expected)


# ---------------------------------------------------------------------------
# Sink registry


def _registry(embeddings: bool = True) -> SinkRegistry:
    titles = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"]
    vectors = None
    if embeddings:
        # Alpha ~ Beta ~ Gamma cluster; Delta/Epsilon far away.
        vectors = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.8, 0.2, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    return SinkRegistry(
        title_to_index={title: i for i, title in enumerate(titles)},
        vocab_size=100,
        embeddings=vectors,
    )


class TestSinkRegistry:
    def test_seq_id_adds_vocab_size(self):
        assert _registry().seq_id("Alpha") == 100
        assert _registry().seq_id("Gamma") == 102

    def test_unknown_title_raises(self):
        with pytest.raises(KeyError, match="verification gate"):
            _registry().seq_id("Zeta")

    def test_nearest_excludes_self_and_manifest(self):
        neighbors = _registry().nearest("Alpha", exclude=["Beta"], top_k=2)
        titles = [title for title, _ in neighbors]
        assert "Alpha" not in titles and "Beta" not in titles
        assert titles[0] == "Gamma"

    def test_placebo_is_deterministic_and_far(self):
        registry = _registry()
        first = registry.placebo("fact-1", anchor_title="Alpha", similarity_ceiling=0.5)
        second = registry.placebo("fact-1", anchor_title="Alpha", similarity_ceiling=0.5)
        assert first == second
        assert first in ("Delta", "Epsilon")  # the cluster is above the ceiling

    def test_row_mismatch_rejected(self):
        with pytest.raises(ValueError, match="row-aligned"):
            SinkRegistry(
                title_to_index={"A": 0, "B": 1},
                vocab_size=10,
                embeddings=np.zeros((3, 2), dtype=np.float32),
            )


# ---------------------------------------------------------------------------
# Backend state mapping


class FakeTokenizer:
    eos_token_id = 0

    def __call__(self, text: str, return_tensors: str | None = None):
        ids = [3 + (len(word) % 4) for word in str(text).split()] or [3]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return " ".join(f"tok{int(i)}" for i in ids)


class FakeSeqTDModel:
    """Records every forward's sink configuration; output depends on it."""

    vocab = 997

    def __init__(self):
        self.calls: list[dict[str, Any]] = []
        self._param = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        return iter([self._param])

    def __call__(self, idx, seq_ids=None, exclude_seq_ids=None, eval_mode="all"):
        self.calls.append(
            {
                "eval_mode": eval_mode,
                "active": None if seq_ids is None else int(seq_ids[0, 0]),
                "excluded": (
                    []
                    if exclude_seq_ids is None
                    else sorted(int(ids[0, 0]) for ids in exclude_seq_ids)
                ),
            }
        )
        batch, length = idx.shape
        logits = torch.zeros(batch, length, self.vocab)
        marker = 7 if seq_ids is None else 1 + int(seq_ids[0, 0]) % (self.vocab - 1)
        logits[:, -1, marker] = 1.0
        return logits


def _backend(del_off_mode: str = "sinks-zero") -> NullsWikiAuditBackend:
    return NullsWikiAuditBackend(
        model=FakeSeqTDModel(),
        tokenizer=FakeTokenizer(),
        registry=_registry(),
        del_off_mode=del_off_mode,
        checkpoint_dir="fake-ckpt",
    )


def _example(manifest_sources=("Alpha",)) -> AuditExample:
    return AuditExample.from_prompt_row(
        {
            "prompt_text": "Alpha's capital is",
            "gold_object": "Beta City",
            "fact_id": "f1",
            "prompt_id": "p1",
            "subject": "Alpha",
            "source_title": "Alpha",
            "deletion_manifest": {
                "source_ids": list(manifest_sources),
                "strategy": "provenance-source",
            },
        }
    )


class TestBackendStateMapping:
    def test_full_activates_gold_sink(self):
        backend = _backend()
        backend.generate(_example(), DatabaseState.FULL, max_new_tokens=2)
        call = backend.model.calls[0]
        assert call["eval_mode"] == "activate_seq"
        assert call["active"] == 100  # Alpha
        assert call["excluded"] == []

    def test_del_on_excludes_manifest_and_routes_to_survivor(self):
        backend = _backend()
        observation = backend.generate(
            _example(("Alpha", "Beta")), DatabaseState.DEL_ON, max_new_tokens=2
        )
        call = backend.model.calls[0]
        assert call["eval_mode"] == "activate_seq"
        assert call["active"] == 102  # Gamma, nearest survivor
        assert call["excluded"] == [100, 101]
        selected = observation.retrieval_trace["selected_candidate"]
        assert selected["source_id"] == "Gamma"

    def test_del_on_empty_manifest_raises(self):
        with pytest.raises(ValueError, match="non-empty deletion manifest"):
            _backend().generate(
                _example(()), DatabaseState.DEL_ON, max_new_tokens=2
            )

    def test_del_on_entry_ids_rejected(self):
        example = AuditExample.from_prompt_row(
            {
                "prompt_text": "Alpha's capital is",
                "gold_object": "Beta City",
                "source_title": "Alpha",
                "deletion_manifest": {"entry_ids": ["e1"]},
            }
        )
        with pytest.raises(ValueError, match="source_ids"):
            _backend().generate(example, DatabaseState.DEL_ON, max_new_tokens=2)

    def test_del_off_sinks_zero_is_backbone_only(self):
        backend = _backend()
        observation = backend.generate(
            _example(), DatabaseState.DEL_OFF, max_new_tokens=2
        )
        call = backend.model.calls[0]
        assert call["eval_mode"] == "dropout"
        assert call["active"] is None
        assert observation.retrieval_trace["retrieval_enabled"] is False
        assert observation.retrieval_trace["del_off_mode"] == "sinks-zero"

    def test_del_off_placebo_activates_far_mask_and_excludes_gold(self):
        backend = _backend(del_off_mode="placebo-sink")
        backend.generate(_example(), DatabaseState.DEL_OFF, max_new_tokens=2)
        call = backend.model.calls[0]
        assert call["eval_mode"] == "activate_seq"
        assert call["active"] in (103, 104)  # Delta or Epsilon
        assert 100 in call["excluded"]  # gold is excluded under placebo

    def test_three_states_pass_core_validation(self):
        backend = _backend()
        example = _example()
        states = [DatabaseState.FULL, DatabaseState.DEL_ON, DatabaseState.DEL_OFF]
        rows = [
            audit_example(backend, example, state, max_new_tokens=2)
            for state in states
        ]
        validate_intervention_results(rows, expected_states=states)

    def test_unknown_gold_title_names_the_gate(self):
        example = AuditExample.from_prompt_row(
            {
                "prompt_text": "Zeta is",
                "gold_object": "x",
                "source_title": "Zeta",
                "deletion_manifest": {"source_ids": ["Zeta"]},
            }
        )
        with pytest.raises(ValueError, match="verification gate"):
            _backend().generate(example, DatabaseState.FULL, max_new_tokens=1)

    def test_fingerprints_reflect_state_dependence(self):
        backend = _backend()
        manifest = _example(("Alpha",)).deletion_manifest
        other = _example(("Alpha", "Beta")).deletion_manifest
        full_fp = backend.cross_phase_fingerprint(DatabaseState.FULL, manifest)
        assert full_fp == backend.cross_phase_fingerprint(DatabaseState.FULL, other)
        del_on_fp = backend.cross_phase_fingerprint(DatabaseState.DEL_ON, manifest)
        assert del_on_fp != backend.cross_phase_fingerprint(
            DatabaseState.DEL_ON, other
        )
        # sinks-zero DEL-OFF ignores the manifest entirely.
        assert backend.cross_phase_fingerprint(
            DatabaseState.DEL_OFF, manifest
        ) == backend.cross_phase_fingerprint(DatabaseState.DEL_OFF, other)

    def test_answer_logprob_runs_under_explicit_sink_config(self):
        backend = _backend()
        value = backend.answer_logprob(
            "Alpha's capital is", " Beta City", eval_mode="activate_seq",
            active_title="Alpha",
        )
        assert isinstance(value, float)


# ---------------------------------------------------------------------------
# Registration


def _args(**overrides: Any) -> argparse.Namespace:
    defaults: dict[str, Any] = {
        "prompt_files": ["prompts.jsonl"],
        "adversarial": False,
        "bootstrap_oracle_from_full": False,
        "closure": None,
        "radius_grid": None,
        "nulls_checkpoint_dir": "missing-ckpt",
        "nulls_title_to_index": "missing.pkl",
        "nulls_title_embeddings": None,
        "nulls_del_off_mode": "sinks-zero",
        "nulls_placebo_ceiling": 0.5,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestRegistration:
    def test_backend_is_registered(self):
        assert "nulls-wiki-1b" in available_backends()

    def test_validate_rejects_entry_level_machinery(self):
        spec = get_backend_spec("nulls-wiki-1b")
        with pytest.raises(ValueError, match="Not supported for nulls-wiki-1b"):
            spec.validate(_args(closure="geometric"))
        with pytest.raises(ValueError, match="Not supported for nulls-wiki-1b"):
            spec.validate(_args(adversarial=True))

    def test_validate_requires_artifacts(self, tmp_path):
        spec = get_backend_spec("nulls-wiki-1b")
        with pytest.raises(ValueError, match="nulls-checkpoint-dir"):
            spec.validate(_args())
        checkpoint = tmp_path / "ckpt"
        checkpoint.mkdir()
        with pytest.raises(ValueError, match="title-to-index"):
            spec.validate(_args(nulls_checkpoint_dir=str(checkpoint)))

    def test_placebo_mode_requires_embeddings(self, tmp_path):
        spec = get_backend_spec("nulls-wiki-1b")
        checkpoint = tmp_path / "ckpt"
        checkpoint.mkdir()
        mapping = tmp_path / "map.pkl"
        mapping.write_bytes(pickle.dumps({"Alpha": 0}))
        with pytest.raises(ValueError, match="placebo-sink"):
            spec.validate(
                _args(
                    nulls_checkpoint_dir=str(checkpoint),
                    nulls_title_to_index=str(mapping),
                    nulls_del_off_mode="placebo-sink",
                )
            )


# ---------------------------------------------------------------------------
# Source-level closures


def _space() -> SourceSpace:
    return SourceSpace(
        titles=("Alpha", "Beta", "Gamma", "Delta"),
        embeddings=np.array(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.8, 0.2],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )


def _closure_row() -> dict:
    return {
        "prompt_text": "Alpha's capital is",
        "gold_object": "Beta City",
        "source_title": "Alpha",
        "fact_id": "f1",
    }


class TestSourceClosure:
    def test_geometric_k_manifest(self):
        expanded = expand_row(_closure_row(), space=_space(), predicate="geometric", k=2)
        manifest = expanded["deletion_manifest"]
        assert manifest["source_ids"] == ["Alpha", "Beta", "Gamma"]
        assert manifest["strategy"] == "geometric-source"
        assert manifest["metadata"]["k"] == 2

    def test_provenance_is_gold_only(self):
        expanded = expand_row(
            _closure_row(), space=_space(), predicate="provenance", k=0
        )
        assert expanded["deletion_manifest"]["source_ids"] == ["Alpha"]

    def test_value_uses_source_texts(self):
        texts = {
            "Beta": "Beta City is mentioned here.",
            "Gamma": "Nothing relevant.",
            "Delta": "Beta City appears but Delta is far away.",
        }
        expanded = expand_row(
            _closure_row(),
            space=_space(),
            predicate="value",
            k=0,
            envelope_k=2,  # envelope = Beta, Gamma only
            source_texts=texts.get,
        )
        assert expanded["deletion_manifest"]["source_ids"] == ["Alpha", "Beta"]

    def test_nearest_many_matches_nearest(self):
        rng = np.random.default_rng(0)
        titles = tuple(f"T{i}" for i in range(60))
        space = SourceSpace(titles=titles, embeddings=rng.normal(size=(60, 8)).astype(np.float32))
        bulk = space.nearest_many(["T3", "T17", "T3", "T59"], 5, query_chunk=2, row_chunk=7)
        assert set(bulk) == {"T3", "T17", "T59"}
        for title, neighbors in bulk.items():
            assert [t for t, _ in neighbors] == [t for t, _ in space.nearest(title, 5)]
            assert title not in {t for t, _ in neighbors}
        assert space.nearest_many(["T3"], 0) == {"T3": []}

    def test_expand_row_with_precomputed_neighbors_and_hits(self):
        space = _space()
        neighbors = space.nearest("Alpha", 3)
        expanded = expand_row(
            _closure_row(), space=space, predicate="hybrid", k=1, envelope_k=3,
            neighbors=neighbors, value_hits={"Beta", "Delta", "NotInEnvelope"},
        )
        # hybrid = geometric-1 (Beta) ∩ value hits within the envelope
        assert expanded["deletion_manifest"]["source_ids"] == ["Alpha", "Beta"]
        expanded = expand_row(
            _closure_row(), space=space, predicate="value", k=0, envelope_k=3,
            neighbors=neighbors, value_hits={"Beta", "Delta", "NotInEnvelope"},
        )
        assert expanded["deletion_manifest"]["source_ids"] == ["Alpha", "Beta", "Delta"]
        with pytest.raises(ValueError, match="needed"):
            expand_row(_closure_row(), space=space, predicate="geometric", k=3, neighbors=neighbors[:1])

    def test_answer_mentioned_is_whole_phrase_and_alias_aware(self):
        from halo.interventions.source_closure import answer_mentioned, normalized_answer_aliases
        from halo.core.equivalence import normalize_text
        row = {"gold_object": "Beta City", "answer_aliases": ["Betaville"]}
        aliases = normalized_answer_aliases(row)
        assert answer_mentioned(normalize_text("Welcome to Betaville."), aliases)
        assert answer_mentioned(normalize_text("The Beta City council"), aliases)
        assert not answer_mentioned(normalize_text("Beta Cityscape"), aliases)  # not whole phrase
        assert not answer_mentioned(normalize_text("Beta"), aliases)

    def test_routing_space_artifact_rejected(self, tmp_path):
        artifact = tmp_path / "routing.npz"
        np.savez(
            artifact,
            embeddings=np.zeros((2, 2), dtype=np.float32),
            mode=np.str_("routing"),
        )
        with pytest.raises(ValueError, match="routing-space"):
            SourceSpace.load(titles=["A", "B"], embeddings_path=artifact)

    def test_build_closure_prompt_file(self, tmp_path):
        prompts = tmp_path / "prompts.jsonl"
        prompts.write_text(json.dumps(_closure_row()) + "\n")
        output = tmp_path / "prompts_k2.jsonl"
        report = build_closure_prompt_file(
            prompts, output, space=_space(), predicate="geometric", k=2
        )
        assert report["kept"] == 1
        row = json.loads(output.read_text())
        assert row["deletion_manifest"]["source_ids"] == ["Alpha", "Beta", "Gamma"]


# ---------------------------------------------------------------------------
# Prompt augmentation (models.nulls_wiki.titles)


class TestAugmentSourceTitles:
    def test_resolution_and_exclusion_report(self, tmp_path):
        from models.nulls_wiki import titles as module

        prompts = tmp_path / "prompts.jsonl"
        rows = [
            {"prompt_text": "a", "gold_object": "x", "subject": "Alpha"},
            {"prompt_text": "b", "gold_object": "y", "subject": "beta"},
            {"prompt_text": "c", "gold_object": "z", "subject": "Missing"},
        ]
        prompts.write_text("".join(json.dumps(row) + "\n" for row in rows))
        mapping = tmp_path / "map.pkl"
        mapping.write_bytes(pickle.dumps({"Alpha": 0, "Beta": 1}))
        output = tmp_path / "prompts_nulls.jsonl"

        report = module.augment(prompts, mapping, output)
        assert report["kept"] == 2 and report["dropped"] == 1
        augmented = [json.loads(line) for line in output.read_text().splitlines()]
        assert augmented[0]["source_title"] == "Alpha"
        assert augmented[1]["source_title"] == "Beta"  # normalized match
        assert augmented[0]["deletion_manifest"] == {
            "source_ids": ["Alpha"],
            "strategy": "provenance-source",
        }

    def test_normalization_is_mediawiki_rule_only(self):
        from models.nulls_wiki import titles as module

        resolve = module.build_title_resolver(["Foo bar"])
        # First-character case and underscore/space ARE the wiki's identity rule.
        assert resolve(["foo bar"]) == ("Foo bar", "mediawiki-normalized")
        assert resolve(["Foo_bar"]) == ("Foo bar", "mediawiki-normalized")
        # Full case-folding is NOT — "FOO BAR" could be a different article.
        assert resolve(["FOO BAR"]) is None

    def test_ambiguous_normalized_forms_are_not_guessed(self):
        from models.nulls_wiki import titles as module

        resolve = module.build_title_resolver(["Foo bar", "Foo_bar"])
        assert resolve(["foo bar"]) is None
        assert resolve(["Foo_bar"]) == ("Foo_bar", "exact")

    def test_aliases_are_not_consulted(self, tmp_path):
        from models.nulls_wiki import titles as module

        prompts = tmp_path / "prompts.jsonl"
        prompts.write_text(
            json.dumps(
                {
                    "prompt_text": "a",
                    "gold_object": "x",
                    "subject": "Nope",
                    "subject_aliases": ["Alpha"],
                }
            )
            + "\n"
        )
        mapping = tmp_path / "map.pkl"
        mapping.write_bytes(pickle.dumps({"Alpha": 0}))
        output = tmp_path / "out.jsonl"
        report = module.augment(prompts, mapping, output)
        assert report["kept"] == 0 and report["dropped"] == 1


# ---------------------------------------------------------------------------
# Verification-gate merge path (no model load)


class TestGateMerge:
    def test_merge_writes_summary_and_gated_prompts(self, tmp_path):
        from models.nulls_wiki import gate

        prompts = tmp_path / "prompts.jsonl"
        rows = [
            {"prompt_text": "a", "gold_object": "x", "fact_id": "f0"},
            {"prompt_text": "b", "gold_object": "y", "fact_id": "f1"},
            {"prompt_text": "c", "gold_object": "z", "fact_id": "f2"},
        ]
        prompts.write_text("".join(json.dumps(row) + "\n" for row in rows))
        output = tmp_path / "gate.jsonl"
        shard_records = [
            [
                {"row_index": 0, "fact_id": "f0", "status": "pass", "margin": 2.0},
                {"row_index": 2, "fact_id": "f2", "status": "fail", "margin": -0.5},
            ],
            [
                {"row_index": 1, "fact_id": "f1", "status": "pass", "margin": 1.0},
            ],
        ]
        for index, records in enumerate(shard_records):
            gate.write_shard(records, output, index, 2)

        gated = tmp_path / "gated.jsonl"
        summary = gate.finalize(
            gate.read_shards(output, 2),
            output=output,
            prompts_path=prompts,
            margin=0.0,
            emit_passed_prompts=gated,
        )

        merged = [json.loads(line) for line in output.read_text().splitlines()]
        assert [record["row_index"] for record in merged] == [0, 1, 2]
        assert summary["passed"] == 2 and summary["failed"] == 1
        written = json.loads(output.with_suffix(".summary.json").read_text())
        assert written["exclusion_rate"] == pytest.approx(1 / 3)
        gated_rows = [json.loads(line) for line in gated.read_text().splitlines()]
        assert [row["fact_id"] for row in gated_rows] == ["f0", "f1"]

    def test_missing_shard_raises(self, tmp_path):
        from models.nulls_wiki import gate

        with pytest.raises(FileNotFoundError, match="shard"):
            gate.read_shards(tmp_path / "gate.jsonl", 2)


# ---------------------------------------------------------------------------
# Corpus filter (matched-corpus Co-LMLM configuration)


@dataclass
class FakeSearchResult:
    id: str
    score: float
    text_value: str
    text_key: str = ""
    metadata: dict = field(default_factory=dict)
    vector: Any = None


class FakeIndex:
    def __init__(self, candidates):
        self.candidates = candidates

    def search(self, query_vector, top_k=1, similarity_threshold=None):
        return list(self.candidates[: max(top_k, 1)])


class TestCorpusFilter:
    def test_non_matching_sources_are_excluded_in_every_state(self):
        from halo.interventions.filtering import _FilteringSearchIndex

        candidates = [
            FakeSearchResult("e1", 0.9, "v1", metadata={"source_id": "fineweb::123"}),
            FakeSearchResult("e2", 0.8, "v2", metadata={"source_id": "wikipedia::Q1"}),
        ]
        index = _FilteringSearchIndex(
            base_index=FakeIndex(candidates),
            example=_example(),
            excluded_entry_ids=frozenset(),
            excluded_source_ids=frozenset(),
            support_judge=lambda candidate, example: {"supports_target": False},
            corpus_exclude=lambda candidate: "wikipedia" not in str(
                candidate.metadata.get("source_id", "")
            ),
        )
        retained = index.search(np.ones(3, dtype=np.float32), top_k=1)
        assert [candidate.id for candidate in retained] == ["e2"]


# ---------------------------------------------------------------------------
# Scheduler wiring


def _scheduler_repo(
    tmp_path: Path,
    *,
    closure_artifact: bool = False,
    routing_artifact: bool = False,
    checkpoint: bool = True,
    title_to_index: bool = True,
 corpus=False) -> Path:
    repo = tmp_path / "repo"
    (repo / "data").mkdir(parents=True)
    (repo / "scripts").mkdir()
    (repo / "scripts" / "run_audit_suite_co_lmlm.sh").write_text("#!/bin/bash\n")
    row = {"prompt_text": "p", "gold_object": "o", "subject": "Alpha"}
    (repo / "data" / "prompts_trex.jsonl").write_text(json.dumps(row) + "\n")
    if checkpoint:
        checkpoint_dir = repo / "data" / "nulls-wikipedia-full"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "lit_model.pth").write_bytes(b"stub")
    if title_to_index:
        (repo / "data" / "nulls-title-to-index.pkl").write_bytes(
            pickle.dumps({"Alpha": 0})
        )
    if routing_artifact:
        (repo / "data" / "nulls-title-embeddings.npz").write_bytes(b"stub")
    if closure_artifact:
        (repo / "data" / "nulls-closure-embeddings.npz").write_bytes(b"stub")
    if corpus:
        shards = repo / "data" / "nulls-wiki-corpus" / "train_raw" / "en"
        shards.mkdir(parents=True)
        (shards / "train-00000-of-00001.parquet").write_bytes(b"stub")
    return repo


def _build_nulls_jobs(repo: Path, env: dict[str, str], shards: int = 1):
    from halo.scheduler import build_jobs

    return build_jobs(
        repo_root=repo,
        out_root=repo / "out",
        co_lmlm_dir=repo / "co",
        index_dir=repo / "index",
        datasets=("trex",),
        models=("nulls-wiki-1b",),
        inherited_env=env,
        shards_per_phase=shards,
    )


class TestSchedulerNulls:
    def test_default_graph_runs_prep_then_audits(self, tmp_path):
        jobs = _build_nulls_jobs(_scheduler_repo(tmp_path), env={})
        phases = sorted(job.phase for job in jobs)
        assert phases == [
            "del-off",
            "prep-augment",
            "prep-embeddings",
            "prep-gate",
            "standard",
        ]
        by_phase = {job.phase: job for job in jobs}
        gate = by_phase["prep-gate"]
        assert by_phase["prep-augment"].key in gate.dependencies
        assert by_phase["prep-embeddings"].key in gate.dependencies
        assert gate.key in by_phase["standard"].dependencies
        assert by_phase["standard"].key in by_phase["del-off"].dependencies
        # The audits consume the GATED prompt set the gate emits.
        assert "--emit-passed-prompts" in gate.command
        gated = gate.command[gate.command.index("--emit-passed-prompts") + 1]
        assert gated in by_phase["standard"].command
        del_off = by_phase["del-off"]
        assert del_off.command[del_off.command.index("--nulls-del-off-mode") + 1] == (
            "placebo-sink"
        )

    def test_prebuilt_routing_artifact_skips_embeddings_job(self, tmp_path):
        repo = _scheduler_repo(tmp_path, routing_artifact=True)
        jobs = _build_nulls_jobs(repo, env={})
        assert not any(job.phase == "prep-embeddings" for job in jobs)

    def test_closure_artifact_enables_closures_and_sweep(self, tmp_path):
        repo = _scheduler_repo(tmp_path, closure_artifact=True)
        jobs = _build_nulls_jobs(repo, env={"NULLS_SWEEP_K_GRID": "1,2"})
        by_phase = {job.phase: job for job in jobs}
        assert "prep-closures" in by_phase
        sweep_phases = sorted(
            job.phase for job in jobs if job.phase.startswith("sweep-")
        )
        assert sweep_phases == ["sweep-k1", "sweep-k2"]
        sweep = by_phase["sweep-k1"]
        assert by_phase["prep-closures"].key in sweep.dependencies
        assert by_phase["standard"].key in sweep.dependencies

    def test_closure_artifact_and_corpus_enable_policy_rows(self, tmp_path):
        repo = _scheduler_repo(tmp_path, closure_artifact=True, corpus=True)
        jobs = _build_nulls_jobs(repo, env={"NULLS_POLICY_K": "8"})
        by_phase = {job.phase: job for job in jobs}
        assert {"prep-policy-closures", "policy-value", "policy-hybrid"} <= set(by_phase)
        closures = by_phase["prep-policy-closures"]
        cmd = closures.command
        assert cmd[cmd.index("--predicate") + 1] == "value,hybrid"
        assert cmd[cmd.index("--k-grid") + 1] == "8"
        assert cmd[cmd.index("--texts") + 1].endswith("nulls-wiki-corpus")
        assert by_phase["prep-gate"].key in closures.dependencies
        hybrid = by_phase["policy-hybrid"]
        assert closures.key in hybrid.dependencies
        assert by_phase["standard"].key in hybrid.dependencies
        prompt_file = hybrid.command[hybrid.command.index("--prompt-files") + 1]
        assert prompt_file.endswith("_hybrid_k8.jsonl")
        assert str(hybrid.expected_outputs[0]).endswith("policy_matrix/hybrid/cross_state_metrics.csv")
        # provenance/geometric rows are not re-run: standard and the sweep carry them.
        assert not any(job.phase in ("policy-provenance", "policy-geometric") for job in jobs)

    def test_policy_rows_absent_without_corpus(self, tmp_path):
        repo = _scheduler_repo(tmp_path, closure_artifact=True)
        jobs = _build_nulls_jobs(repo, env={})
        assert not any(job.phase.startswith("policy") for job in jobs)

    def test_explicit_policy_without_corpus_is_an_error(self, tmp_path):
        with pytest.raises(ValueError, match="training corpus"):
            _build_nulls_jobs(
                _scheduler_repo(tmp_path, closure_artifact=True),
                env={"NULLS_PHASES": "standard,policy"},
            )

    def test_bad_policy_k_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="NULLS_POLICY_K"):
            _build_nulls_jobs(_scheduler_repo(tmp_path), env={"NULLS_POLICY_K": "x"})

    def test_explicit_sweep_without_artifact_is_an_error(self, tmp_path):
        with pytest.raises(ValueError, match="shared-encoder artifact"):
            _build_nulls_jobs(
                _scheduler_repo(tmp_path), env={"NULLS_PHASES": "standard,sweep"}
            )

    def test_striping_shards_gate_and_standard(self, tmp_path):
        jobs = _build_nulls_jobs(
            _scheduler_repo(tmp_path), env={"NULLS_PHASES": "standard"}, shards=3
        )
        standard_keys = sorted(
            job.key for job in jobs if job.phase == "standard"
        )
        assert standard_keys == [
            "nulls-wiki-1b.trex.standard.finalize",
            "nulls-wiki-1b.trex.standard.shard0",
            "nulls-wiki-1b.trex.standard.shard1",
            "nulls-wiki-1b.trex.standard.shard2",
        ]
        shard = next(job for job in jobs if job.key.endswith("standard.shard1"))
        assert shard.command[shard.command.index("--shard") + 1] == "1/3"
        gate_finalize = next(
            job for job in jobs if job.key.endswith("prep-gate.finalize")
        )
        assert "--merge" in gate_finalize.command
        assert (
            sum(1 for job in jobs if job.phase == "prep-gate") == 4
        )  # 3 shards + finalize

    def test_missing_checkpoint_points_to_setup(self, tmp_path):
        with pytest.raises(ValueError, match="setup_data.sh"):
            _build_nulls_jobs(_scheduler_repo(tmp_path, checkpoint=False), env={})

    def test_missing_title_to_index_points_to_authors(self, tmp_path):
        with pytest.raises(ValueError, match="NULLs authors"):
            _build_nulls_jobs(_scheduler_repo(tmp_path, title_to_index=False), env={})

    def test_bad_del_off_mode_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="NULLS_DEL_OFF_MODE"):
            _build_nulls_jobs(
                _scheduler_repo(tmp_path), env={"NULLS_DEL_OFF_MODE": "off"}
            )

    def test_bad_k_grid_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="NULLS_SWEEP_K_GRID"):
            _build_nulls_jobs(
                _scheduler_repo(tmp_path), env={"NULLS_SWEEP_K_GRID": "1,x"}
            )
