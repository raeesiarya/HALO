"""NULLs SeqTD transformer (vendored, see NOTICE.md).

Verbatim from upstream ``MemSinks/src/src/SeqTDModel.py`` except for the
documented edits: import paths, debug prints and per-forward
``torch.cuda.empty_cache()`` removed, ``exclude_seq_ids`` may be a
sequence of id tensors whose sink masks are unioned before exclusion (the
single-tensor path is byte-identical to upstream), and the ``all`` and
``dropout`` neuron masks are built in the activation dtype instead of
float32. Upstream runs the model in fp32, so its masks match by accident;
we load the released checkpoint in bfloat16 on CUDA, where a float32 mask
promotes ``x * mask`` to float32 and the next bf16 Linear raises
"mat1 and mat2 to have the same dtype". The mask values are unchanged.
"""

from __future__ import annotations

import math
from functools import partial
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from litgpt.model import (
    KVCache,
    apply_rope,
    batched_index_select,
    build_mask_cache,
    build_rope_cache,
)
from litgpt.scripts.convert_hf_checkpoint import qkv_reassemble

try:  # litgpt >= 0.5.5
    from litgpt.model import do_softcapping
except ImportError:
    # The pinned litgpt (>=0.5.12) ships do_softcapping; this fallback only
    # serves an older litgpt. Upstream's definition verbatim (litgpt 0.5.5
    # model.py). The released NULLs checkpoint sets both
    # attention_logit_softcapping and final_logit_softcapping to null, so it
    # is never invoked for it.
    def do_softcapping(x: torch.Tensor, thresh: float) -> torch.Tensor:
        return torch.tanh(x / thresh) * thresh

from models.nulls_wiki.vendor.masking import (
    batch_seqtied_mask_mult,
    union_exclusion_mask as _union_exclusion_mask,
)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: Any, block_idx: int) -> None:
        super().__init__()
        # key, query and value projections for all heads, but in a batch
        self.qkv = nn.Linear(
            config.n_embd,
            (config.n_head + 2 * config.n_query_groups)
            * config.head_size,  # support for grouped/multi queries
            bias=config.bias or config.attn_bias,
        )
        # output projection
        self.proj = nn.Linear(
            config.head_size * config.n_head, config.n_embd, bias=config.bias
        )
        # disabled by default
        self.kv_cache: Optional[KVCache] = None
        self.apply_sliding_window_attention = False
        if (
            config.sliding_window_size is not None
            and config.sliding_window_indices is not None
        ):
            self.apply_sliding_window_attention = config.sliding_window_indices[
                block_idx
            ]

        if config.norm_qk:
            norm_q_size = (
                config.n_head * config.head_size
                if config.norm_qk_type == "olmo2"
                else config.head_size
            )
            norm_k_size = (
                config.n_query_groups * config.head_size
                if config.norm_qk_type == "olmo2"
                else config.head_size
            )
            self.norm_q = config.norm_class(norm_q_size, eps=config.norm_eps)
            self.norm_k = config.norm_class(norm_k_size, eps=config.norm_eps)
        else:
            self.norm_q = self.norm_k = None

        self.config = config
        self.block_idx = block_idx

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        input_pos_maxp1: Optional[int] = None,
        seq_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        head_size = self.config.head_size
        n_head = self.config.n_head
        n_query_groups = self.config.n_query_groups
        rope_n_elem = self.config.rope_n_elem
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        qkv = self.qkv(x)  # (B, T, 3xC*)

        query_size = n_head * head_size
        key_size = value_size = n_query_groups * head_size
        q, k, v = qkv.split((query_size, key_size, value_size), dim=-1)  # 3x(B, T, C*)

        if self.config.norm_qk and self.config.norm_qk_type == "olmo2":
            q = self.norm_q(q)
            k = self.norm_k(k)

        q = q.view(B, T, n_head, head_size)  # (B, T, nh_q, hs)
        k = k.view(B, T, n_query_groups, head_size)  # (B, T, n_query_groups, hs)
        v = v.view(B, T, n_query_groups, head_size)  # (B, T, n_query_groups, hs)

        q = q.transpose(1, 2)  # (B, nh_q, T, hs)
        k = k.transpose(1, 2)  # (B, nh_k, T, hs)
        v = v.transpose(1, 2)  # (B, nh_v, T, hs)

        if self.config.norm_qk and self.config.norm_qk_type == "default":
            q = self.norm_q(q)
            k = self.norm_k(k)

        q_roped = apply_rope(q[..., :rope_n_elem], cos, sin)
        k_roped = apply_rope(k[..., :rope_n_elem], cos, sin)
        q = torch.cat((q_roped, q[..., rope_n_elem:]), dim=-1)  # (B, nh_q, T, hs)
        k = torch.cat((k_roped, k[..., rope_n_elem:]), dim=-1)  # (B, nh_k, T, hs)

        if input_pos is not None:
            if not isinstance(self.kv_cache, KVCache):
                raise TypeError("You need to call `gpt.set_kv_cache()`")
            k, v = self.kv_cache(input_pos, k, v)
            if input_pos_maxp1 is not None:
                k = k[..., :input_pos_maxp1, :]
                v = v[..., :input_pos_maxp1, :]

        if n_query_groups != n_head and (input_pos is None or n_query_groups != 1):
            q_per_kv = n_head // n_query_groups
            k = k.repeat_interleave(q_per_kv, dim=1)  # (B, nh_q, T, hs)
            v = v.repeat_interleave(q_per_kv, dim=1)  # (B, nh_q, T, hs)

        if self.apply_sliding_window_attention:
            if mask is None:
                mask = torch.ones(T, T, dtype=q.dtype, device=q.device).triu(diagonal=1)
                mask.masked_fill_(mask.bool(), float("-inf"))
                mask = mask.view(1, 1, *mask.shape)
            sliding_window_bias = torch.ones_like(mask).tril(
                diagonal=-self.config.sliding_window_size
            )
            sliding_window_bias.masked_fill_(sliding_window_bias.bool(), float("-inf"))
            mask += sliding_window_bias

        y = self.scaled_dot_product_attention(q, k, v, mask, seq_ids=seq_ids)

        y = y.reshape(B, T, head_size * n_head)

        return self.proj(y)  # (B, T, C)

    def scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        seq_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        scale = 1.0 / math.sqrt(
            self.config.attention_scores_scalar or self.config.head_size
        )
        if mask is None:
            if seq_ids is not None:
                doc_attention_mask = (
                    (seq_ids[:, None, :, None] == seq_ids[:, None, None, :])
                    .to(q.dtype)
                    .to(q.device)
                )
                mask = (
                    torch.ones(
                        q.size(2), q.size(2), dtype=q.dtype, device=q.device
                    ).triu(diagonal=1)
                    * doc_attention_mask
                )
            else:
                mask = torch.ones(
                    q.size(2), q.size(2), dtype=q.dtype, device=q.device
                ).triu(diagonal=1)
            mask.masked_fill_(mask.bool(), torch.finfo(q.dtype).min)
        # with softcapping we cannot use SDPA
        if self.config.attention_logit_softcapping is not None:
            scores = q @ k.mT * scale
            scores = do_softcapping(scores, self.config.attention_logit_softcapping)
            scores = scores + mask
            scores = F.softmax(scores, dim=-1, dtype=torch.float).to(dtype=q.dtype)
            y = scores @ v
        else:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=0.0,
                scale=scale,
                is_causal=mask is None,
            )
        return y.transpose(1, 2)

    def build_kv_cache(
        self,
        batch_size: int,
        max_seq_length: int,
        rope_cache_length: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "KVCache":
        v_shape = (
            batch_size,
            self.config.n_query_groups,
            max_seq_length,
            self.config.head_size,
        )
        if rope_cache_length is None:
            if self.config.rotary_percentage != 1.0:
                raise TypeError(
                    "Please pass the `rope_cache_length=gpt.cos.size(-1)` value"
                )
            k_shape = v_shape
        else:
            k_shape = (
                batch_size,
                self.config.n_query_groups,
                max_seq_length,
                rope_cache_length + self.config.head_size - self.config.rope_n_elem,
            )
        return KVCache(k_shape, v_shape, device=device, dtype=dtype)

    def _load_from_state_dict(
        self, state_dict: dict, prefix: str, *args: Any, **kwargs: Any
    ) -> None:
        """For compatibility with legacy checkpoints."""

        for attr in ("weight", "bias"):
            legacy_key = f"{prefix}attn.{attr}"
            current_key = f"{prefix}qkv.{attr}"
            if legacy_key in state_dict:
                state_dict[current_key] = qkv_reassemble(
                    state_dict.pop(legacy_key), self.config
                )

        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class LLaMAMLPSeqTD(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.fc_1 = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.fc_2 = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)
        self.p_mem = config.p_mem
        self.p_gen = config.p_gen
        self.num_gen_neurons = int(config.intermediate_size * config.p_gen)
        self.num_mem_neurons = config.intermediate_size - self.num_gen_neurons
        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        seq_ids: torch.Tensor,
        exclude_seq_ids=None,
        eval_mode: str = "all",
    ) -> torch.Tensor:
        if self.training:
            assert seq_ids is not None
        if self.training and seq_ids is not None:
            assert seq_ids.shape == x.shape[:-1]
            mask = batch_seqtied_mask_mult(seq_ids, self.num_mem_neurons, self.p_mem)
            gen_neuron_block = torch.ones(
                (mask.shape[0], mask.shape[1], self.num_gen_neurons)
            ).to(mask.device)
            gen_neuron_block.requires_grad_(False)
            mask = torch.cat((gen_neuron_block, mask), dim=-1)
            mask = mask.to(torch.bool)
        else:
            if eval_mode == "all":
                mask_gen = torch.ones(
                    (x.shape[0], x.shape[1], self.num_gen_neurons),
                    dtype=x.dtype,
                    device=x.device,
                )
                mask_mem = (
                    torch.ones(
                        (x.shape[0], x.shape[1], self.num_mem_neurons),
                        dtype=x.dtype,
                        device=x.device,
                    )
                    * self.p_mem
                )
                if exclude_seq_ids is not None:
                    mask_mem_exclude = _union_exclusion_mask(
                        exclude_seq_ids, self.num_mem_neurons, self.p_mem
                    ).int()
                    mask_mem = torch.mul(mask_mem, 1 - mask_mem_exclude)
                mask = torch.cat((mask_gen, mask_mem), dim=-1)
                mask = mask.to(x.device)
                mask.requires_grad_(False)
            elif eval_mode == "dropout":
                mask_gen = torch.ones(
                    (x.shape[0], x.shape[1], self.num_gen_neurons),
                    dtype=x.dtype,
                    device=x.device,
                ) * ((self.p_gen + self.p_mem) / self.p_gen)
                mask_mem = torch.zeros(
                    (x.shape[0], x.shape[1], self.num_mem_neurons),
                    dtype=x.dtype,
                    device=x.device,
                )
                mask = torch.cat((mask_gen, mask_mem), dim=-1)
                mask = mask.to(x.device)
                mask.requires_grad_(False)
            elif eval_mode == "activate_seq":
                assert seq_ids.shape == x.shape[:-1]
                mask = batch_seqtied_mask_mult(
                    seq_ids, self.num_mem_neurons, self.p_mem
                )
                if exclude_seq_ids is not None:
                    mask_exclude = _union_exclusion_mask(
                        exclude_seq_ids, self.num_mem_neurons, self.p_mem
                    )
                    mask = torch.logical_and(
                        mask, torch.logical_not(mask_exclude)
                    ).int()
                gen_neuron_block = torch.ones(
                    (mask.shape[0], mask.shape[1], self.num_gen_neurons)
                ).to(mask.device)
                mask = torch.cat((gen_neuron_block, mask), dim=-1)
                mask = mask.to(torch.bool)
                mask = mask.to(x.device)
                mask.requires_grad_(False)
            else:
                raise ValueError(f"Unknown eval_mode {eval_mode!r}")

        x_fc_1 = self.fc_1(x)
        x_fc_2 = self.fc_2(x)
        x = F.silu(x_fc_1) * x_fc_2
        x = x * mask
        return self.proj(x)


class BlockSeqTD(nn.Module):
    def __init__(self, config: Any, block_idx: int) -> None:
        super().__init__()
        if not config.parallel_residual and config.shared_attention_norm:
            raise NotImplementedError(
                "No checkpoint amongst the ones we support uses this configuration"
                " (non-parallel residual and shared attention norm)."
            )

        self.norm_1 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config, block_idx)
        self.post_attention_norm = (
            config.norm_class(config.n_embd, eps=config.norm_eps)
            if config.post_attention_norm
            else nn.Identity()
        )
        self.norm_2 = (
            None
            if config.shared_attention_norm
            else config.norm_class(config.n_embd, eps=config.norm_eps)
        )
        self.mlp = config.mlp_class(config)
        self.post_mlp_norm = (
            config.norm_class(config.n_embd, eps=config.norm_eps)
            if config.post_mlp_norm
            else nn.Identity()
        )
        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        seq_ids: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        input_pos_maxp1: Optional[torch.Tensor] = None,
        exclude_seq_ids=None,
        eval_mode: str = "all",
    ) -> torch.Tensor:
        x_normed = self.norm_1(x)
        attention_output = self.attn(
            x_normed, cos, sin, mask, input_pos, input_pos_maxp1, seq_ids=seq_ids
        )
        attention_output = self.post_attention_norm(attention_output)
        if self.config.parallel_residual:
            if not self.config.shared_attention_norm:
                x_normed = self.norm_2(x)
            x = attention_output + x
        else:
            x = attention_output + x
            x_normed = self.norm_2(x)
        return (
            self.post_mlp_norm(self.mlp(x_normed, seq_ids, exclude_seq_ids, eval_mode))
            + x
        )


class GPTSeqTD(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        assert config.padded_vocab_size is not None
        self.config = config

        self.lm_head = nn.Linear(
            config.n_embd, config.padded_vocab_size, bias=config.lm_head_bias
        )
        from litgpt.model import Block  # only needed for the non-SeqTD fallback

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
                h=nn.ModuleList(
                    BlockSeqTD(config, block_idx)
                    if config.use_seqtd
                    else Block(config, block_idx)
                    for block_idx in range(config.n_layer)
                ),
                ln_f=config.norm_class(config.n_embd, eps=config.norm_eps),
            )
        )
        self.mask_cache: Optional[torch.Tensor] = None
        self.max_seq_length = self.config.block_size

    @property
    def max_seq_length(self) -> int:
        return self._max_seq_length

    @max_seq_length.setter
    def max_seq_length(self, value: int) -> None:
        if value > self.config.block_size:
            raise ValueError(
                f"Cannot attend to {value}, block size is only {self.config.block_size}."
                " This is likely because the input text exceeds the supported context length of this model."
            )
        self._max_seq_length = value
        if not hasattr(self, "cos"):
            # first call
            cos, sin = self.rope_cache()
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)
        # override
        elif value != self.cos.size(0):
            self.cos, self.sin = self.rope_cache(device=self.cos.device)

    def reset_parameters(self) -> None:
        # Trigger resetting the rope-cache
        self.cos, self.sin = self.rope_cache(device=self.cos.device)

    def forward(
        self,
        idx: torch.Tensor,
        seq_ids: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
        input_pos_maxp1: Optional[torch.Tensor] = None,
        lm_head_chunk_size: int = 0,
        exclude_seq_ids=None,
        eval_mode: str = "all",
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        T = idx.size(1)
        if self.max_seq_length < T:
            raise ValueError(
                f"Cannot forward sequence of length {T}, max seq length is only {self.max_seq_length}."
            )
        if input_pos is not None:  # use the kv cache
            if input_pos.dim() > 2:
                raise ValueError(
                    f"input_pos must have 1 or 2 dimensions, input_pos.shape = {input_pos.shape}"
                )
            if input_pos.shape[-1] != T:
                raise ValueError(
                    f"input_pos.shape[-1] = {input_pos.shape[-1]} != {T} = idx.shape[1], must be the same"
                )
            cos = batched_index_select(self.cos, 0, input_pos)
            sin = batched_index_select(self.sin, 0, input_pos)
            if input_pos.dim() == 1:
                cos = cos.unsqueeze(0)
                sin = sin.unsqueeze(0)
            if self.mask_cache is None:
                raise TypeError("You need to call `gpt.set_kv_cache()`")
            mask = batched_index_select(self.mask_cache, 2, input_pos)
            if mask.dim() > 4:
                mask = mask.view(*(mask.shape[0:1] + mask.shape[2:]))
            if input_pos_maxp1 is not None:
                if input_pos_maxp1 > self.max_seq_length:
                    raise ValueError(
                        f"Positions in 'input_pos' must be in [0,{self.max_seq_length})"
                    )
                mask = mask[..., :input_pos_maxp1]
        else:
            cos = self.cos[:T].unsqueeze(0)
            sin = self.sin[:T].unsqueeze(0)
            mask = None  # defaults to causal mask
            input_pos_maxp1 = None

        x = self.transformer.wte(idx)  # token embeddings of shape (B, T, n_embd)
        if self.config.scale_embeddings:
            x = x * torch.tensor(self.config.n_embd**0.5, dtype=x.dtype)

        for block in self.transformer.h:
            x = (
                block(
                    x,
                    seq_ids,
                    cos,
                    sin,
                    mask,
                    input_pos,
                    input_pos_maxp1,
                    exclude_seq_ids,
                    eval_mode,
                )
                if self.config.use_seqtd
                else block(x, cos, sin, mask, input_pos, input_pos_maxp1)
            )
        x = self.transformer.ln_f(x)
        clamp_head = (
            partial(do_softcapping, thresh=self.config.final_logit_softcapping)
            if self.config.final_logit_softcapping is not None
            else nn.Identity()
        )
        if lm_head_chunk_size > 0:
            return [
                clamp_head(self.lm_head(x_i))
                for x_i in x.split(lm_head_chunk_size, dim=1)
            ]
        return clamp_head(self.lm_head(x))  # (B, T, padded_vocab_size)

    def rope_cache(
        self, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.config.rope_adjustments is None:
            extra_config = None
        else:
            adjusted_params_required = [
                "factor",
                "low_freq_factor",
                "high_freq_factor",
                "original_max_seq_len",
            ]
            params_present = [
                param in self.config.rope_adjustments
                for param in adjusted_params_required
            ]
            num_params_present = sum(params_present)

            if num_params_present == 0:
                extra_config = None  # uses standard RoPE
            elif num_params_present == 4:
                extra_config = {
                    name: self.config.rope_adjustments[name]
                    for name in adjusted_params_required
                }
            else:
                missing_params = [
                    param
                    for param, present in zip(adjusted_params_required, params_present)
                    if not present
                ]
                raise ValueError(
                    f"The following adjusted RoPE parameters are missing in rope_adjustments: {', '.join(missing_params)}. "
                    "All adjusted RoPE parameters must be specified together."
                )
        return build_rope_cache(
            seq_len=self.max_seq_length,
            n_elem=self.config.rope_n_elem,
            device=device,
            condense_ratio=self.config.rope_condense_ratio,
            base=self.config.rope_base,
            extra_config=extra_config,
        )

    def set_kv_cache(
        self,
        batch_size: int,
        max_seq_length: Optional[int] = None,
        rope_cache_length: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        if rope_cache_length is None:
            rope_cache_length = self.cos.size(-1)

        if max_seq_length is None:
            max_seq_length = self.max_seq_length

        for block in self.transformer.h:
            block.attn.kv_cache = block.attn.build_kv_cache(
                batch_size,
                max_seq_length,
                rope_cache_length,
                device,
                dtype,
            )

        if self.mask_cache is None or self.mask_cache.size(3) != max_seq_length:
            self.mask_cache = build_mask_cache(max_seq_length, device)

    def clear_kv_cache(self) -> None:
        self.mask_cache = None
        for block in self.transformer.h:
            block.attn.kv_cache = None
