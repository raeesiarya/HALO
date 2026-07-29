import argparse
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from test_smollm2_backend import FakeModel, FakeTokenizer

from halo.cli.persistence import backend_resume_identity
from halo.core.backend import audit_example, validate_intervention_results
from halo.core.states import DatabaseState
from halo.registry import available_backends, get_backend_spec
from models.smollm2_360m.backend import SmolLM2AuditBackend
from models.standard_lm_360m_fw import MODEL
from models.standard_lm_360m_fw.backend import StandardLMAuditBackend

STATES = [DatabaseState.FULL, DatabaseState.DEL_ON, DatabaseState.DEL_OFF]


def _make_backend(completions):
    tokenizer = FakeTokenizer()
    model = FakeModel(tokenizer, completions)
    backend = StandardLMAuditBackend(
        model=model, tokenizer=tokenizer, model_path=MODEL
    )
    return backend, model


class TestBackend:
    def test_shares_the_parametric_implementation(self):
        assert issubclass(StandardLMAuditBackend, SmolLM2AuditBackend)

    def test_three_states_identical_and_valid(self):
        backend, model = _make_backend(
            {"France capital is": "Paris . more text"}
        )
        from halo.core.examples import AuditExample

        example = AuditExample(
            prompt="France capital is",
            ground_truth="Paris",
            fact_id="f1",
            prompt_id="f1",
        )
        rows = [audit_example(backend, example, state) for state in STATES]
        validate_intervention_results(rows, expected_states=STATES)
        assert {row["model_output"] for row in rows} == {"Paris"}
        assert model.generate_calls == 3

    def test_resume_identity_names_this_model_and_class(self):
        backend, _ = _make_backend({})
        identity = backend_resume_identity(backend)
        assert identity["model_path"] == MODEL
        assert identity["class"].endswith("StandardLMAuditBackend")
        # Distinct from the smollm2 backend's identity even at equal prompts.
        assert "standard_lm_360m_fw" in identity["class"]


class TestRegistration:
    def test_backend_registered(self):
        assert "standard-lm-360m-fw" in available_backends()

    def test_no_oracle_bootstrap_capability(self):
        assert not get_backend_spec(
            "standard-lm-360m-fw"
        ).supports_oracle_bootstrap

    def test_model_is_the_matched_baseline(self):
        assert MODEL == "lil-lab/CoLMLM-Standard-LM-Baseline-360M-FW"


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
        get_backend_spec("standard-lm-360m-fw").validate(_args())

    def test_requires_prompt_files(self):
        with pytest.raises(ValueError, match="--prompt-files"):
            get_backend_spec("standard-lm-360m-fw").validate(
                _args(prompt_files=None)
            )

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
            get_backend_spec("standard-lm-360m-fw").validate(
                _args(**overrides)
            )
