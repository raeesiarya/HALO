import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lmlm_audit.interventions.adversary import AdversarialConfig, InjectedEntry
from lmlm_audit.interventions.closure import ClosureConfig
from lmlm_audit.core.backend import audit_example
from lmlm_audit.core.examples import AuditExample, DeletionManifest
from lmlm_audit.core.neighbors import NeighborConfig
from lmlm_audit.core.states import DatabaseState
from models.rel_lmlm.backend import RelLMLMAuditBackend
from models.rel_lmlm.database import AuditDatabaseManager, triple_id
from models.rel_lmlm.adapter import (
    TripleSearchIndex,
    rel_support_judge,
)
from lmlm_audit.cli.runner import run_adversarial_eval, run_entanglement_sweep

LOOKUP_PROMPT = "<|db_entity|>Hexol<|db_relationship|>First Described By<|db_return|>"


def test_triple_id_is_stable_and_unambiguous() -> None:
    assert triple_id("a", "b", "c") == triple_id("a", "b", "c")
    assert triple_id("a", "b|c", "d") != triple_id("a", "b", "c|d")
    assert json.loads(triple_id("s", "r", "o")) == ["s", "r", "o"]


# --- rel support judge -----------------------------------------------------


def _example_hexol() -> AuditExample:
    return AuditExample.from_prompt_row(
        {
            "prompt_id": "hexol",
            "prompt_text": "Who first described Hexol?",
            "gold_object": "Jørgensen",
            "object_aliases": ["Jorgensen"],
            "subject": "Hexol",
            "relation": "First Described By",
        }
    )


def test_rel_support_judge_uses_full_triple_when_available() -> None:
    supporting = SimpleNamespace(
        id="x",
        score=0.9,
        text_value="Jorgensen",
        metadata={
            "subject": "Hexol",
            "relation": "First Described By",
            "object": "Jorgensen",
        },
    )
    wrong_relation = SimpleNamespace(
        id="y",
        score=0.9,
        text_value="Jorgensen",
        metadata={
            "subject": "Hexol",
            "relation": "Structure Recognized By",
            "object": "Jorgensen",
        },
    )
    example = _example_hexol()
    assert rel_support_judge(supporting, example)["supports_target"] is True
    judged = rel_support_judge(wrong_relation, example)
    assert judged["supports_target"] is False
    assert judged["support_method"] == "triple-equivalence"


def test_rel_support_judge_falls_back_to_value_equivalence() -> None:
    candidate = {"entry_id": "x", "value": "Jorgensen", "score": 0.5}
    example = AuditExample(
        prompt="Q?", ground_truth="Jørgensen", object_aliases=("Jorgensen",)
    )
    judged = rel_support_judge(candidate, example)
    assert judged["supports_target"] is True
    assert judged["support_method"] == "value-equivalence"


# --- index adapter over a vector retriever ----------------------------------


class VectorIndex:
    def __init__(self, vectors: np.ndarray) -> None:
        self.vectors = vectors

    def search(self, query, top_k):
        scores = (np.asarray(query, dtype=np.float32) @ self.vectors.T)[0]
        order = np.argsort(-scores)[:top_k]
        return [scores[order]], [order]


class VectorRetriever:
    """Sentence-transformer + FAISS stand-in with real cosine geometry."""

    def __init__(self, triples, query_map):
        self.top_k = 5
        self.default_threshold = 0.6
        self.id_to_triplet = {i: (s, r, o) for i, (s, r, o, _vec) in enumerate(triples)}
        vectors = np.stack([np.asarray(vec, dtype=np.float32) for *_t, vec in triples])
        self.index = VectorIndex(
            vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        )
        self._query_map = {
            key: np.asarray(vec, dtype=np.float32) for key, vec in query_map.items()
        }
        self.model = SimpleNamespace(encode=self._encode)

    def _encode(self, texts, **_kwargs):
        return np.stack([self._query_map[text] for text in texts])

    @staticmethod
    def _normalize_text(text):
        return text.lower().replace("_", " ").strip()


def _two_fact_retriever() -> VectorRetriever:
    return VectorRetriever(
        triples=[
            ("France", "Capital", "Paris", [1.0, 0.0]),
            ("Poland", "Capital", "Warsaw", [0.6, 0.8]),
        ],
        query_map={
            "france capital": [1.0, 0.0],
            "poland capital": [0.6, 0.8],
        },
    )


def test_triple_search_index_adapts_the_retriever() -> None:
    retriever = _two_fact_retriever()
    adapter = TripleSearchIndex(SimpleNamespace(topk_retriever=retriever))

    results = adapter.search([1.0, 0.0], top_k=5, similarity_threshold=0.5)
    assert [candidate.metadata["object"] for candidate in results] == [
        "Paris",
        "Warsaw",
    ]
    assert results[0].id == triple_id("France", "Capital", "Paris")
    assert results[0].score == pytest.approx(1.0)
    assert results[1].score == pytest.approx(0.6)

    tight = adapter.search([1.0, 0.0], top_k=5, similarity_threshold=0.9)
    assert [candidate.text_value for candidate in tight] == ["Paris"]


# --- closure manifests and injections in the database manager ---------------


def _closure_manifest(entry_ids, *, semantic_target=None):
    metadata = {"predicates_active": ["geometric"]}
    if semantic_target is not None:
        metadata = {
            "predicates_active": ["geometric", "semantic"],
            "semantic_target": {
                "ground_truth": semantic_target,
                "object_aliases": [],
            },
        }
    return DeletionManifest(entry_ids=entry_ids, strategy="closure", metadata=metadata)


def test_manager_deletes_by_closure_entry_ids(fake_base_manager) -> None:
    manifest = _closure_manifest(
        [triple_id("Hexol", "First Described By", "Jorgensen")]
    )
    manager = AuditDatabaseManager(
        fake_base_manager,
        DatabaseState.DEL_ON,
        deletion_manifest=manifest,
    )
    value = manager.retrieve_from_database(LOOKUP_PROMPT)
    assert value == "Werner"
    assert [entry["object"] for entry in manager.last_trace["deleted_candidates"]] == [
        "Jorgensen"
    ]
    assert len(manager.captured_query_embeddings) == 1


def test_manager_semantic_backstop_judges_target_answer(
    fake_base_manager,
) -> None:
    # Closure lists only the Jorgensen triple, but the backstop must also
    # null the Werner triple because it expresses the target answer.
    manifest = _closure_manifest(
        [triple_id("Hexol", "First Described By", "Jorgensen")],
        semantic_target="Werner",
    )
    manager = AuditDatabaseManager(
        fake_base_manager,
        DatabaseState.DEL_ON,
        deletion_manifest=manifest,
    )
    value = manager.retrieve_from_database(LOOKUP_PROMPT)
    assert value == "2004"
    deleted = {entry["object"] for entry in manager.last_trace["deleted_candidates"]}
    assert deleted == {"Jorgensen", "Werner"}


def test_manager_splices_injected_survivors(fake_base_manager) -> None:
    manifest = _closure_manifest(
        [
            triple_id("Hexol", "First Described By", "Jorgensen"),
            triple_id("Hexol", "Structure Recognized By", "Werner"),
            triple_id("Jocelyne Girard-Bujold", "Term End", "2004"),
        ]
    )
    injection = InjectedEntry(
        entry_id="adv-0",
        vector=np.asarray([1.0], dtype=np.float32),
        value="The answer is Jorgensen.",
        template="verbatim",
        target_cosine=0.8,
    )
    manager = AuditDatabaseManager(
        fake_base_manager,
        DatabaseState.DEL_ON,
        deletion_manifest=manifest,
        injections=(injection,),
    )
    value = manager.retrieve_from_database(LOOKUP_PROMPT)
    assert value == "The answer is Jorgensen."
    selected = manager.last_trace["selected_candidate"]
    assert selected["entry_id"] == "adv-0"
    assert selected["synthetic"] is True


def test_backstop_catches_injected_verbatim_value(fake_base_manager) -> None:
    manifest = _closure_manifest(
        [
            triple_id("Hexol", "First Described By", "Jorgensen"),
            triple_id("Hexol", "Structure Recognized By", "Werner"),
            triple_id("Jocelyne Girard-Bujold", "Term End", "2004"),
        ],
        semantic_target="The answer is Jorgensen.",
    )
    injection = InjectedEntry(
        entry_id="adv-0",
        vector=np.asarray([1.0], dtype=np.float32),
        value="The answer is Jorgensen.",
        template="verbatim",
        target_cosine=0.8,
    )
    manager = AuditDatabaseManager(
        fake_base_manager,
        DatabaseState.DEL_ON,
        deletion_manifest=manifest,
        injections=(injection,),
    )
    with pytest.raises(ValueError, match="No retrieval results"):
        manager.retrieve_from_database(LOOKUP_PROMPT)
    deleted_ids = {
        entry["entry_id"] for entry in manager.last_trace["deleted_candidates"]
    }
    assert "adv-0" in deleted_ids


# --- full rel backend with a fake model --------------------------------------


PROMPT_A = "What is the capital of France?"
PROMPT_B = "What is the capital of Poland?"


class FakeRelModel:
    def __init__(self, lookups):
        self.lookups = dict(lookups)
        self.db_manager = None
        self.logits_processor = None

    def eval(self):
        pass

    def parameters(self):
        return iter([SimpleNamespace(device="cpu")])

    def set_logits_bias(self, _tokenizer):
        pass

    def generate(self, **_kwargs):
        return "generated-ids"

    def _decode_with_special_tokens(self, _outputs, _tok, _len, prompt):
        entity, relation = self.lookups[prompt]
        return f"<|db_entity|>{entity}<|db_relationship|>{relation}<|db_return|>"

    def generate_with_lookup(self, prompt, **_kwargs):
        return "unknown"

    def post_process(self, raw_output, _tokenizer):
        return raw_output


def _fake_tokenizer():
    tokenizer = MagicMock()
    tokenizer.encode.return_value = list(range(8))
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 2
    tokenizer.unk_token_id = 3
    tokenizer.convert_tokens_to_ids.return_value = 42
    inputs = MagicMock()

    def getitem(_key):
        item = MagicMock()
        item.shape = (1, 8)
        return item

    inputs.__getitem__.side_effect = getitem
    call_result = MagicMock()
    call_result.to.return_value = inputs
    tokenizer.side_effect = lambda *a, **kw: call_result
    return tokenizer


class VectorBaseManager:
    def __init__(self, retriever):
        self.database_name = "vector"
        self.database_org_file = []
        self.database = {}
        self.topk_retriever = retriever

    def init_topk_retriever(self, *args, **kwargs):
        pass

    def retrieve_from_database(self, prompt, threshold=None):
        from models.rel_lmlm.database import (
            extract_lookup_query,
            retrieve_triplet_candidates,
        )

        entity, relation = extract_lookup_query(prompt)
        results, _ = retrieve_triplet_candidates(
            self.topk_retriever, entity, relation, threshold=threshold
        )
        if not results:
            raise ValueError("no results")
        return results[0][2]


def _rel_backend():
    retriever = _two_fact_retriever()
    base = VectorBaseManager(retriever)
    model = FakeRelModel(
        {
            PROMPT_A: ("France", "Capital"),
            PROMPT_B: ("Poland", "Capital"),
        }
    )
    backend = RelLMLMAuditBackend(
        base_db_manager=base, model=model, tokenizer=_fake_tokenizer()
    )
    return backend, base


def _rel_prompt_file(tmp_path: Path) -> Path:
    prompt_path = tmp_path / "prompts.jsonl"
    rows = [
        {
            "prompt_id": "pA",
            "prompt_text": PROMPT_A,
            "gold_object": "Paris",
            "subject": "France",
            "relation": "Capital",
        },
        {
            "prompt_id": "pB",
            "prompt_text": PROMPT_B,
            "gold_object": "Warsaw",
            "subject": "Poland",
            "relation": "Capital",
        },
    ]
    prompt_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    return prompt_path


def test_rel_backend_emits_query_embeddings_and_events() -> None:
    backend, _ = _rel_backend()
    example = AuditExample.from_prompt_row(
        {
            "prompt_id": "pA",
            "prompt_text": PROMPT_A,
            "gold_object": "Paris",
            "subject": "France",
            "relation": "Capital",
        }
    )
    result = audit_example(backend, example, DatabaseState.FULL)

    assert result["model_output"] == "Paris"
    embeddings = result["_query_embeddings"]
    np.testing.assert_allclose(
        embeddings[0]["vector"], np.asarray([1.0, 0.0], dtype=np.float32)
    )
    events = result["retrieval_trace"]["retrieval_events"]
    assert events[0]["selected_candidate"]["entry_id"] == triple_id(
        "France", "Capital", "Paris"
    )


def test_rel_entanglement_sweep_end_to_end(tmp_path) -> None:
    backend, base = _rel_backend()
    prompt_path = _rel_prompt_file(tmp_path)
    output_dir = tmp_path / "sweep"

    summary = run_entanglement_sweep(
        prompt_path,
        backend,
        index=TripleSearchIndex(base),
        radii=(0.9, 0.5),
        closure_config=ClosureConfig(predicates=("geometric", "semantic")),
        neighbor_config=NeighborConfig(mode="cosine", ball=0.5, cap=20),
        output_dir=output_dir,
    )

    assert summary["swept_facts"] == 2
    assert summary["skipped_facts"] == []
    for fact in ("pA", "pB"):
        curve = summary["entanglement"][fact]["curve"]
        assert [point["rho"] for point in curve] == [0.9, 0.5]
        # Tight radius forgets the fact without collateral; the loose radius
        # swallows the neighbor's triple.
        assert curve[0]["efficacy"] == 1.0
        assert curve[0]["collateral"] == 0.0
        assert curve[1]["collateral"] == 1.0
        assert summary["entanglement"][fact]["gap"] == 0.0


def test_rel_adversarial_eval_end_to_end(tmp_path) -> None:
    backend, base = _rel_backend()
    prompt_path = _rel_prompt_file(tmp_path)

    summary = run_adversarial_eval(
        prompt_path,
        backend,
        index=TripleSearchIndex(base),
        closure_config=ClosureConfig(predicates=("geometric",), radius=0.9),
        adversarial_config=AdversarialConfig(
            rho=0.9,
            epsilons=(0.05,),
            templates=("verbatim", "hyphenated"),
        ),
        output_dir=tmp_path / "adversarial",
    )

    assert summary["attacked_facts"] == 2
    evasion = {row["template"]: row["evasion_rate"] for row in summary["evasion"]}
    # The survivor value is spliced verbatim into the answer, so the control
    # template restores the fact; the hyphenated paraphrase does not surface
    # the exact answer string.
    assert evasion["verbatim"] == 1.0
    assert evasion["hyphenated"] == 0.0
    assert backend.injections == ()
    for row in summary["margins"]:
        assert row["s_del"] == pytest.approx(1.0)
        # Geometric-only closure searches at the radius, so sub-radius
        # survivors are never observed and the margin is unobservable
        # without the semantic envelope.
        assert row["s_surv"] is None
        assert row["margin"] is None


def test_rel_adversarial_margin_is_observable_with_semantic_envelope(
    tmp_path,
) -> None:
    backend, base = _rel_backend()
    prompt_path = _rel_prompt_file(tmp_path)

    summary = run_adversarial_eval(
        prompt_path,
        backend,
        index=TripleSearchIndex(base),
        closure_config=ClosureConfig(predicates=("geometric", "semantic"), radius=0.9),
        adversarial_config=AdversarialConfig(
            rho=0.9, epsilons=(0.05,), templates=("hyphenated",)
        ),
        output_dir=tmp_path / "adversarial",
    )
    # The semantic envelope surfaces the neighbor triple as a survivor, so
    # the margin geometry is populated.
    for row in summary["margins"]:
        assert row["s_del"] == pytest.approx(1.0)
        assert row["s_surv"] == pytest.approx(0.6)
        assert row["margin"] == pytest.approx(0.4)
