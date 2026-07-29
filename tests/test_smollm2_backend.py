import argparse
import json
import re
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from halo.cli.persistence import backend_resume_identity
from halo.cli.runner import run_backend_audit
from halo.core.backend import audit_example, validate_intervention_results
from halo.core.examples import AuditExample, DeletionManifest
from halo.core.metrics import metrics_total
from halo.core.states import DatabaseState
from halo.registry import available_backends, get_backend_spec
from models.smollm2_360m import MODEL
from models.smollm2_360m.backend import (
    SmolLM2AuditBackend,
    extract_smollm2_answer,
)

STATES = [DatabaseState.FULL, DatabaseState.DEL_ON, DatabaseState.DEL_OFF]


class FakeTokenizer:
    """Word-level tokenizer over a growing vocabulary; id 0 is <eos>."""

    eos_token_id = 0
    pad_token_id = None

    def __init__(self):
        self._vocab: dict[str, int] = {"<eos>": 0}
        self._words: dict[int, str] = {0: "<eos>"}

    def _id(self, word: str) -> int:
        if word not in self._vocab:
            index = len(self._vocab)
            self._vocab[word] = index
            self._words[index] = word
        return self._vocab[word]

    def __call__(self, text: str, return_tensors: str | None = None):
        assert return_tensors == "pt"
        ids = [self._id(word) for word in text.split()]
        return {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
        }

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        words = [self._words[int(item)] for item in ids.tolist()]
        if skip_special_tokens:
            words = [word for word in words if word != "<eos>"]
        return " ".join(words)


class FakeModel:
    """Deterministic prompt->completion lookup behind the HF generate API."""

    device = torch.device("cpu")

    def __init__(self, tokenizer: FakeTokenizer, completions: dict[str, str]):
        self.tokenizer = tokenizer
        self.completions = completions
        self.generate_calls = 0
        self.do_sample_flags: list[bool | None] = []
        self.max_new_tokens_seen: list[int] = []

    def generate(
        self,
        *,
        input_ids,
        attention_mask=None,
        max_new_tokens: int = 12,
        do_sample=None,
        pad_token_id=None,
    ):
        self.generate_calls += 1
        self.do_sample_flags.append(do_sample)
        self.max_new_tokens_seen.append(max_new_tokens)
        prompt = self.tokenizer.decode(input_ids[0])
        completion = self.completions[prompt]
        completion_ids = [
            self.tokenizer._id(word) for word in completion.split()
        ][:max_new_tokens]
        return torch.cat(
            [input_ids, torch.tensor([completion_ids], dtype=torch.long)],
            dim=1,
        )


def _make_backend(completions: dict[str, str]) -> tuple[SmolLM2AuditBackend, FakeModel]:
    tokenizer = FakeTokenizer()
    model = FakeModel(tokenizer, completions)
    backend = SmolLM2AuditBackend(
        model=model, tokenizer=tokenizer, model_path=MODEL
    )
    return backend, model


def _example(prompt: str, gold: str, fact_id: str = "f1") -> AuditExample:
    return AuditExample(
        prompt=prompt,
        ground_truth=gold,
        fact_id=fact_id,
        prompt_id=fact_id,
    )


class TestAnswerExtraction:
    def test_first_sentence_only(self):
        assert extract_smollm2_answer("Paris. The city is large.") == "Paris"

    def test_first_line_only(self):
        assert extract_smollm2_answer("Paris\nThe capital of Italy is") == "Paris"

    def test_prefix_stripped(self):
        assert extract_smollm2_answer("the answer is Paris") == "Paris"

    def test_quotes_and_punctuation_stripped(self):
        assert extract_smollm2_answer(' "Paris", ') == "Paris"

    def test_whitespace_normalized(self):
        assert extract_smollm2_answer("  Paris   France  ") == "Paris France"


class TestGenerate:
    def test_three_states_identical_and_valid(self):
        backend, model = _make_backend(
            {"France capital is": "Paris . more text"}
        )
        example = _example("France capital is", "Paris")
        rows = [
            audit_example(backend, example, state, max_new_tokens=12)
            for state in STATES
        ]
        validate_intervention_results(rows, expected_states=STATES)
        outputs = {row["model_output"] for row in rows}
        assert outputs == {"Paris"}
        assert model.generate_calls == 3  # three honest passes, no cache
        assert model.do_sample_flags == [False, False, False]  # greedy

    def test_empty_manifest_non_full_states_allowed(self):
        backend, _ = _make_backend({"France capital is": "Paris"})
        example = _example("France capital is", "Paris")
        assert example.deletion_manifest.is_empty
        observation = backend.generate(
            example, DatabaseState.DEL_ON, max_new_tokens=4
        )
        assert observation.model_output == "Paris"

    def test_traces_are_state_correct(self):
        backend, _ = _make_backend({"France capital is": "Paris"})
        example = _example("France capital is", "Paris")
        rows = {
            state: audit_example(backend, example, state)
            for state in STATES
        }
        for state, row in rows.items():
            trace = row["retrieval_trace"]
            assert trace["state"] == state.value
            assert trace["trace_available"] is False
            assert trace["retrieval_events"] == []
        assert rows[DatabaseState.DEL_OFF]["retrieval_trace"][
            "retrieval_enabled"
        ] is False

    def test_max_new_tokens_forwarded(self):
        backend, model = _make_backend(
            {"France capital is": "Paris and much more text here"}
        )
        backend.generate(
            _example("France capital is", "Paris"),
            DatabaseState.FULL,
            max_new_tokens=2,
        )
        assert model.max_new_tokens_seen == [2]

    def test_generation_metadata(self):
        backend, _ = _make_backend({"France capital is": "Paris"})
        observation = backend.generate(
            _example("France capital is", "Paris"), DatabaseState.FULL
        )
        metadata = observation.generation_metadata
        assert metadata["raw_completion"] == "Paris"
        assert metadata["model_path"] == MODEL
        assert metadata["gen_decoded_tokens"] == 1
        assert metadata["t_generate_s"] >= 0.0


class TestCapabilityHooks:
    def test_manifest_fingerprint_ignores_manifest(self):
        backend, _ = _make_backend({})
        empty = DeletionManifest()
        full = DeletionManifest(entry_ids=("e1", "e2"), source_ids=("s1",))
        assert backend.manifest_fingerprint(empty) is not None
        assert backend.manifest_fingerprint(empty) == backend.manifest_fingerprint(
            full
        )

    def test_cross_phase_fingerprint_per_state_not_manifest(self):
        backend, _ = _make_backend({})
        empty = DeletionManifest()
        full = DeletionManifest(entry_ids=("e1",))
        fingerprints = {
            state: backend.cross_phase_fingerprint(state, empty)
            for state in STATES
        }
        assert len(set(fingerprints.values())) == len(STATES)
        for state in STATES:
            assert fingerprints[state] == backend.cross_phase_fingerprint(
                state, full
            )

    def test_full_row_unaffected_always(self):
        backend, _ = _make_backend({})
        assert backend.full_row_unaffected(
            {"retrieval_trace": {}}, DeletionManifest(entry_ids=("e1",))
        )


class TestEndToEnd:
    def test_standard_audit_metrics(self, tmp_path):
        completions = {
            "France capital is": "Paris . trailing",
            "Italy capital is": "Berlin . trailing",
        }
        backend, model = _make_backend(completions)
        prompt_path = tmp_path / "prompts.jsonl"
        rows = [
            {
                "prompt_id": "f1",
                "fact_id": "f1",
                "prompt_text": "France capital is",
                "gold_object": "Paris",
            },
            {
                "prompt_id": "f2",
                "fact_id": "f2",
                "prompt_text": "Italy capital is",
                "gold_object": "Rome",
            },
        ]
        prompt_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

        results = run_backend_audit(
            prompt_path=prompt_path,
            backend=backend,
            states=STATES,
            max_new_tokens=12,
        )

        assert len(results) == len(rows) * len(STATES)
        assert model.generate_calls == len(rows) * len(STATES)
        metrics = metrics_total(results)
        assert metrics["paired_count"] == 2
        # The parametric readout: L(f) = closed-book correctness, and the
        # retrieval-mediated metrics are identically zero.
        assert metrics["parametric_leakage"] == 0.5
        assert metrics["retrieval_mediated_correctness"] == 0.0
        assert metrics["retrieval_interference"] == 0.0
        assert metrics["post_deletion_survival_given_full"] == 1.0
        assert metrics["retrieval_artifact_eligible_count"] == 0


class TestRegistration:
    def test_backend_registered(self):
        assert "smollm2-360m" in available_backends()

    def test_resume_identity_pins_the_model(self):
        backend, _ = _make_backend({})
        identity = backend_resume_identity(backend)
        assert identity["model_path"] == MODEL
        assert identity["class"].endswith("SmolLM2AuditBackend")
        assert identity["similarity_threshold"] is None


def _args(**overrides) -> argparse.Namespace:
    defaults = {
        "prompt_files": [Path("prompts.jsonl")],
        "closure": None,
        "radius_grid": None,
        "adversarial": False,
        "bootstrap_oracle_from_full": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestValidate:
    def test_accepts_plain_standard_audit(self):
        get_backend_spec("smollm2-360m").validate(_args())

    def test_requires_prompt_files(self):
        with pytest.raises(ValueError, match="--prompt-files"):
            get_backend_spec("smollm2-360m").validate(_args(prompt_files=None))

    @pytest.mark.parametrize(
        "overrides, flag",
        [
            ({"closure": "geometric"}, "--closure"),
            ({"radius_grid": "0.95:0.70:0.05"}, "--radius-grid"),
            ({"adversarial": True}, "--adversarial"),
            (
                {"bootstrap_oracle_from_full": True},
                "--bootstrap-oracle-from-full",
            ),
        ],
    )
    def test_rejects_retrieval_modes(self, overrides, flag):
        with pytest.raises(ValueError, match=re.escape(flag)):
            get_backend_spec("smollm2-360m").validate(_args(**overrides))
