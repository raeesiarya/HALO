"""FlashAttention-2-equivalent attention for machines without flash-attn.

Co-LMLM's generator rolls back a failed ``<FACT>`` lookup by cropping the KV
cache to the full kept sequence. Its next ``model.generate()`` call then has
no uncached token, and transformers (5.x) slices the input as
``input_ids[:, -0:]`` — the *whole* sequence — so the kept context is fed a
second time, with position ids restarting at 0, on top of the cached copy.

What the model computes from that point depends on the attention kernel:

* flash_attention_2 (the x86 runs that produced the canonical results)
  ignores the all-ones 2D mask and applies causal attention aligned to the
  bottom-right of the key axis, so every query sees the whole cache: the
  model continues coherently from "context, context";
* sdpa / eager build a mask from the 2D attention mask and position ids,
  which no longer line up with the cache, and the continuation degenerates
  ("Ven is Ven Ven is Ven ...").

This affects every row with a failed lookup — all DEL-OFF rows under
null-retrieval and a minority of DEL-ON/FULL rows. To reproduce the
canonical results where flash-attn cannot be installed (Linux/aarch64),
``fa2_compat_attention`` implements FA2's semantics with SDPA: causal
attention aligned to the bottom-right of the key axis, ignoring the mask.
That equivalence holds for unpadded batch-size-1 decoding, which is the only
way the audit calls the model; padded batches are refused.
"""

from __future__ import annotations

from typing import Any

FA2_COMPAT_ATTENTION = "halo_fa2_compat"


def fa2_compat_attention(
    module: Any,
    query: Any,
    key: Any,
    value: Any,
    attention_mask: Any,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    **kwargs: Any,
) -> tuple[Any, None]:
    import torch
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    if query.shape[0] != 1:
        raise ValueError(
            "fa2_compat_attention reproduces flash-attention-2 only for "
            "unpadded batch-size-1 inputs; got batch size "
            f"{query.shape[0]}."
        )
    q_len, k_len = query.shape[2], key.shape[2]
    if q_len == 1:
        # A single query attends to the whole cache under FA2.
        mask = None
    else:
        # Bottom-right-aligned causal mask: query i sees keys
        # 0 .. i + (k_len - q_len), exactly flash-attn's causal=True.
        mask = torch.ones(q_len, k_len, dtype=torch.bool, device=query.device).tril(
            diagonal=k_len - q_len
        )[None, None]
    return sdpa_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask=mask,
        dropout=dropout,
        scaling=scaling,
        is_causal=False,
        **kwargs,
    )


def register_fa2_compat_attention() -> str:
    """Register the attention under its name; returns that name for
    ``attn_implementation=``. Safe to call more than once."""
    from transformers import AttentionInterface, AttentionMaskInterface
    from transformers.masking_utils import sdpa_mask

    AttentionInterface.register(FA2_COMPAT_ATTENTION, fa2_compat_attention)
    # The mask transformers builds is ignored by the attention above; SDPA's
    # builder is registered only so mask construction has a known function.
    AttentionMaskInterface.register(FA2_COMPAT_ATTENTION, sdpa_mask)
    return FA2_COMPAT_ATTENTION


def flash_attention_2_available() -> bool:
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        return False
    return True
