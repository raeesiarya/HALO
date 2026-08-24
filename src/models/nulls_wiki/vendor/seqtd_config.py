"""NULLs ``SeqTDConfig`` (vendored, see NOTICE.md).

Field set and ``__post_init__`` semantics are verbatim from upstream
``MemSinks/src/src/SeqTDConfig.py``; the litgpt config-catalog fallbacks
(``from_name``) are dropped because the released checkpoints always ship a
``model_config.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, Type, Union

import yaml

from models.nulls_wiki.vendor.masking import find_multiple


@dataclass
class SeqTDConfig:
    name: str = ""
    hf_config: dict = field(default_factory=dict)
    # General size parameters
    block_size: int = 4096
    n_layer: int = 16
    n_embd: int = 4096
    vocab_size: int = 50254
    padding_multiple: int = 512
    padded_vocab_size: Optional[int] = None
    # Transformer block (structure, normalizations)
    norm_class_name: Literal["LayerNorm", "RMSNorm"] = "LayerNorm"
    norm_eps: float = 1e-5
    norm_qk: bool = False
    post_attention_norm: bool = False
    post_mlp_norm: bool = False
    p_mem: float = 0.0
    p_gen: float = 1.0
    parallel_residual: bool = True
    shared_attention_norm: bool = False
    # Transformer block (self-attention)
    n_head: int = 32
    head_size: Optional[int] = None
    n_query_groups: Optional[int] = None
    attn_bias: bool = False
    attention_scores_scalar: Optional[int] = None
    sliding_window_size: Optional[int] = None
    sliding_window_layer_placing: Optional[Literal["all", "interleaved"]] = None
    attention_logit_softcapping: Optional[float] = None
    # Rotary position embedding (RoPE)
    rope_base: int = 10000
    rotary_percentage: float = 0.25
    rope_condense_ratio: int = 1
    rope_adjustments: Optional[dict] = None
    # Transformer block (MLP)
    intermediate_size: Optional[int] = None
    bias: bool = True
    use_seqtd: bool = True
    mlp_class_name: Literal[
        "GptNeoxMLP", "LLaMAMLP", "GemmaMLP", "LLaMAMoE", "LLaMAMLPSeqTD"
    ] = "LLaMAMLPSeqTD"
    gelu_approximate: str = "none"
    n_expert: int = 0
    n_expert_per_token: int = 0
    # GPT before/after blocks
    scale_embeddings: bool = False
    lm_head_bias: bool = False
    final_logit_softcapping: Optional[float] = None
    # Accessed by CausalSelfAttention behind `sliding_window_size is not None`
    # and `norm_qk` guards; kept explicit so attribute lookups never fail.
    sliding_window_indices: Optional[list] = None
    norm_qk_type: str = "default"

    def __post_init__(self) -> None:
        if not self.name:
            self.name = self.hf_config.get("name", self.name)

        if self.head_size is None:
            assert self.n_embd % self.n_head == 0
            self.head_size = self.n_embd // self.n_head

        # vocab size should be a power of 2 to be optimal on hardware. compute the closest value
        if self.padded_vocab_size is None:
            self.padded_vocab_size = find_multiple(self.vocab_size, self.padding_multiple)
        else:
            # vocab size shouldn't be larger than padded vocab size
            self.vocab_size = min(self.vocab_size, self.padded_vocab_size)

        # compute the number of query groups
        if self.n_query_groups is not None:
            assert self.n_head % self.n_query_groups == 0
        else:
            self.n_query_groups = self.n_head

        # compute the intermediate size for MLP if not set
        if self.intermediate_size is None:
            if self.mlp_class_name == "LLaMAMLP":
                raise ValueError(f"The config {self.name!r}, needs to set the `intermediate_size`")
            self.intermediate_size = 4 * self.n_embd

        self.rope_n_elem = int(self.rotary_percentage * self.head_size)

        if self.sliding_window_size is not None:
            self.sliding_window_layer_stride = (
                1
                if (self.sliding_window_layer_placing is None or self.sliding_window_layer_placing == "all")
                else 2
            )

    @classmethod
    def from_file(cls, path: Union[str, Path], **kwargs: Any) -> "SeqTDConfig":
        with open(path, encoding="utf-8") as fp:
            file_kwargs = yaml.safe_load(fp)
            if file_kwargs is None:
                raise ValueError(f"{path} is empty which is likely unexpected.")
        file_kwargs.update(kwargs)
        return cls(**file_kwargs)

    @classmethod
    def from_checkpoint(cls, path: Path, **kwargs: Any) -> "SeqTDConfig":
        """Load ``model_config.yaml`` from a checkpoint directory."""
        if (config_path := Path(path) / "model_config.yaml").is_file():
            return cls.from_file(config_path, **kwargs)
        raise FileNotFoundError(f"No 'model_config.yaml' under {str(path)!r}.")

    @property
    def mlp_class(self) -> Type:
        # `self.mlp_class_name` cannot be the type to keep the config serializable
        if self.mlp_class_name == "LLaMAMLPSeqTD":
            from models.nulls_wiki.vendor.seqtd_model import LLaMAMLPSeqTD

            return LLaMAMLPSeqTD
        import litgpt.model

        return getattr(litgpt.model, self.mlp_class_name)

    @property
    def norm_class(self) -> Type:
        # `self.norm_class_name` cannot be the type to keep the config serializable
        from functools import partial

        if self.norm_class_name == "RMSNorm":
            from litgpt.model import RMSNorm

            return partial(RMSNorm, add_unit_offset="Gemma" in self.name)

        import torch

        if self.norm_class_name == "LayerNorm" and "OLMo" in self.name:
            return partial(torch.nn.LayerNorm, elementwise_affine=False)

        return getattr(torch.nn, self.norm_class_name)
