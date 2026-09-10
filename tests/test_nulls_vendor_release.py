"""The vendored SeqTD model must import and construct from the *released*
NULLs configuration (gauravrghosal/NULLS-Wikipedia-Full, final/model_config.yaml,
copied here verbatim on 2026-09-11).

This is the link the stub-based backend tests cannot cover: the lazy import
of ``vendor.seqtd_model`` against the litgpt version the repo's torch pin
allows. Skipped where litgpt is not installed (CI's pip environment)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

litgpt = pytest.importorskip("litgpt")
torch = pytest.importorskip("torch")

RELEASED_MODEL_CONFIG = """\
attention_logit_softcapping: null
attention_scores_scalar: null
attn_bias: false
bias: false
block_size: 8192
final_logit_softcapping: null
gelu_approximate: none
head_size: 128
hf_config: {}
intermediate_size: 8192
lm_head_bias: false
mlp_class_name: LLaMAMLPSeqTD
n_embd: 960
n_expert: 0
n_expert_per_token: 0
n_head: 15
n_layer: 32
n_query_groups: 5
name: ''
norm_class_name: RMSNorm
norm_eps: 1.0e-05
norm_qk: false
p_gen: 0.06103515625
p_mem: 0.013
padded_vocab_size: 49152
padding_multiple: 512
parallel_residual: false
post_attention_norm: false
post_mlp_norm: false
rope_adjustments: null
rope_base: 100000
rope_condense_ratio: 1
rotary_percentage: 1.0
scale_embeddings: false
shared_attention_norm: false
sliding_window_layer_placing: null
sliding_window_size: null
use_seqtd: true
vocab_size: 49152
"""


@pytest.fixture
def released_checkpoint_dir(tmp_path: Path) -> Path:
    (tmp_path / "model_config.yaml").write_text(RELEASED_MODEL_CONFIG)
    return tmp_path


def test_vendored_model_imports_under_pinned_litgpt():
    from models.nulls_wiki.vendor import seqtd_model

    # Present either from litgpt (>= 0.5.5) or from the vendored fallback.
    assert float(seqtd_model.do_softcapping(torch.tensor(2.0), 1.0)) == pytest.approx(
        float(torch.tanh(torch.tensor(2.0)))
    )


def test_released_config_parses_and_constructs(released_checkpoint_dir: Path):
    from models.nulls_wiki.vendor.seqtd_config import SeqTDConfig
    from models.nulls_wiki.vendor.seqtd_model import GPTSeqTD

    config = SeqTDConfig.from_checkpoint(released_checkpoint_dir)
    assert config.use_seqtd and config.mlp_class_name == "LLaMAMLPSeqTD"
    # The seq-id offset the sink registry adds: vocab == padded vocab == 49152.
    assert config.vocab_size == config.padded_vocab_size == 49152
    assert config.attention_logit_softcapping is None
    assert config.final_logit_softcapping is None

    with torch.device("meta"):
        model = GPTSeqTD(config)
    params = sum(p.numel() for p in model.parameters())
    assert 1.0e9 < params < 1.2e9  # the ~1B release
    keys = model.state_dict().keys()
    assert "transformer.wte.weight" in keys and "lm_head.weight" in keys
    assert "transformer.h.31.mlp.proj.weight" in keys  # 32 SeqTD blocks


TINY_MODEL_CONFIG = RELEASED_MODEL_CONFIG.replace("block_size: 8192", "block_size: 32") \
    .replace("head_size: 128", "head_size: 8").replace("intermediate_size: 8192", "intermediate_size: 64") \
    .replace("n_embd: 960", "n_embd: 32").replace("n_head: 15", "n_head: 4").replace("n_layer: 32", "n_layer: 2") \
    .replace("n_query_groups: 5", "n_query_groups: 2").replace("padded_vocab_size: 49152", "padded_vocab_size: 64") \
    .replace("padding_multiple: 512", "padding_multiple: 64").replace("vocab_size: 49152", "vocab_size: 64")


def test_backend_load_recipe_survives_to_device_and_runs_all_three_states(tmp_path: Path):
    """The loading recipe in models.nulls_wiki.backend.load_seqtd_checkpoint,
    on a tiny SeqTD model: real-device init -> strict assign-load ->
    max_seq_length -> .to(dtype, device) -> forward in every eval mode.

    Regression for a meta-device init: the RoPE buffers are not in the state
    dict, so they stayed on meta and .to(device) raised."""
    from models.nulls_wiki.vendor.seqtd_config import SeqTDConfig
    from models.nulls_wiki.vendor.seqtd_model import GPTSeqTD

    (tmp_path / "model_config.yaml").write_text(TINY_MODEL_CONFIG)
    config = SeqTDConfig.from_checkpoint(tmp_path)
    torch.manual_seed(0)
    source = GPTSeqTD(config)
    state = {key: value.clone() for key, value in source.state_dict().items()}

    model = GPTSeqTD(config)
    model.load_state_dict(state, strict=True, assign=True)
    model.max_seq_length = 16
    model = model.to(dtype=torch.float32, device="cpu")
    model.eval()
    assert model.cos.device.type == "cpu" and tuple(model.cos.shape)[0] == 16

    idx = torch.tensor([[1, 2, 3, 4]])
    sink = lambda source_index: torch.full((1, 4), source_index + config.vocab_size, dtype=torch.int32)
    with torch.no_grad():
        full = model(idx=idx, seq_ids=sink(7), exclude_seq_ids=None, eval_mode="activate_seq")
        del_on = model(idx=idx, seq_ids=sink(3), exclude_seq_ids=[sink(7)], eval_mode="activate_seq")
        del_off = model(idx=idx, seq_ids=None, exclude_seq_ids=None, eval_mode="dropout")
    for logits in (full, del_on, del_off):
        assert tuple(logits.shape) == (1, 4, config.padded_vocab_size)
        assert torch.isfinite(logits).all()
    # The sink states are genuinely different computations.
    assert not torch.allclose(full, del_off)
