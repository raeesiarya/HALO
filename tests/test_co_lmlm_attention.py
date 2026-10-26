from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers.integrations.sdpa_attention")

from models.co_lmlm import attention as fa2  # noqa: E402
from models.co_lmlm import backend  # noqa: E402


class _Module:
    num_key_value_groups = 1
    is_causal = True
    training = False


def _reference(query, key, value, scaling):
    """flash-attn causal=True semantics, written out: query i attends to
    keys 0 .. i + (k_len - q_len)."""
    q_len, k_len = query.shape[2], key.shape[2]
    scores = (query @ key.transpose(-1, -2)) * scaling
    allowed = torch.ones(q_len, k_len, dtype=torch.bool).tril(diagonal=k_len - q_len)
    scores = scores.masked_fill(~allowed, float("-inf"))
    return (scores.softmax(-1) @ value).transpose(1, 2)


@pytest.mark.parametrize("q_len,k_len", [(1, 7), (3, 3), (4, 9), (5, 10)])
def test_fa2_compat_matches_bottom_right_causal_attention(q_len, k_len) -> None:
    generator = torch.Generator().manual_seed(q_len * 31 + k_len)
    query = torch.randn(1, 2, q_len, 8, generator=generator)
    key = torch.randn(1, 2, k_len, 8, generator=generator)
    value = torch.randn(1, 2, k_len, 8, generator=generator)
    # A mask transformers built for the wrong lengths must be ignored, as
    # flash-attn ignores an all-ones 2D mask.
    bogus_mask = torch.zeros(1, 1, q_len, 2, dtype=torch.bool)
    out, _ = fa2.fa2_compat_attention(
        _Module(), query, key, value, bogus_mask, scaling=8**-0.5
    )
    torch.testing.assert_close(out, _reference(query, key, value, 8**-0.5))


def test_fa2_compat_refuses_padded_batches() -> None:
    query = torch.randn(2, 1, 3, 4)
    key = value = torch.randn(2, 1, 3, 4)
    with pytest.raises(ValueError, match="batch-size-1"):
        fa2.fa2_compat_attention(_Module(), query, key, value, None, scaling=0.5)


def test_attention_choice(monkeypatch) -> None:
    monkeypatch.setattr(fa2, "register_fa2_compat_attention", lambda: fa2.FA2_COMPAT_ATTENTION)
    monkeypatch.delenv("CO_LMLM_ATTN", raising=False)
    monkeypatch.setattr(fa2, "flash_attention_2_available", lambda: True)
    assert backend._co_lmlm_attention() == "flash_attention_2"
    # Without flash-attn the default reproduces FA2 instead of plain sdpa.
    monkeypatch.setattr(fa2, "flash_attention_2_available", lambda: False)
    assert backend._co_lmlm_attention() == fa2.FA2_COMPAT_ATTENTION
    monkeypatch.setenv("CO_LMLM_ATTN", "sdpa")
    assert backend._co_lmlm_attention() == "sdpa"
    monkeypatch.setenv("CO_LMLM_ATTN", "bogus")
    with pytest.raises(ValueError, match="CO_LMLM_ATTN"):
        backend._co_lmlm_attention()
