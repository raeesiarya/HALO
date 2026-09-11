"""Verbatim NULLs sink-mask arithmetic. See NOTICE.md for provenance.

``batch_seqtied_mask_mult`` is the contract between a source id and its sink
neurons. Do not reformulate it: ``torch.pow`` on int64 overflows and wraps for
exponents >= 3, and the released checkpoints were trained against exactly that
wrapped arithmetic, so any mathematically "equivalent" rewrite (e.g. iterative
modular exponentiation) produces different masks.
"""

from __future__ import annotations

import torch


def find_multiple(n: int, k: int) -> int:
    """Utility function for finding the nearest value to n which is a multiple of k.

    NOTE: We define this function in this module rather than `litgpt.utils` so that users can import
    this file to do configuration manipulations in Python environments which do not include all the dependencies
    demanded by `litgpt.utils`.
    """
    assert k > 0
    if n % k == 0:
        return n
    return n + k - (n % k)


def union_exclusion_mask(exclude_seq_ids, neuron_dim: int, p_active: float):
    """Union of sink masks for one id tensor or a sequence of id tensors.

    HALO addition: upstream accepts a single ``exclude_seq_ids`` tensor.
    Deletion manifests can name several sources, whose exclusion is the
    union of their masks; a neuron shared by an excluded and a surviving
    source is still excluded (deletion wins), matching upstream's
    single-source semantics. The single-tensor path is upstream's exactly.
    """
    if isinstance(exclude_seq_ids, (list, tuple)):
        combined = None
        for ids in exclude_seq_ids:
            single = batch_seqtied_mask_mult(ids, neuron_dim, p_active)
            combined = (
                single if combined is None else torch.logical_or(combined, single)
            )
        return combined
    return batch_seqtied_mask_mult(exclude_seq_ids, neuron_dim, p_active)


def batch_seqtied_mask_mult(seq_ids, neuron_dim, p_active, a=1588635695, M=4294967291, base=pow(2, 16)):
    #seq_ids has shape (batch_size, seq_len )
    seq_ids = torch.mul(seq_ids, base)
    powers_of_a = torch.remainder(torch.pow(a, torch.arange(neuron_dim)), M).to(seq_ids.device)
    multiplied = torch.remainder(torch.mul(seq_ids.unsqueeze(-1), powers_of_a.unsqueeze(0).unsqueeze(0)), M) #seq-ids would have shape [batch_size, seq_len, 1], powers of a would have the shape [1,1, neuron_dim]-> multiplied should have shape [batch, seq, neurons]
    assert multiplied.shape == torch.Size([seq_ids.shape[0], seq_ids.shape[1], neuron_dim])
    added = torch.remainder(multiplied, M)
    mask = ((added / M) < p_active) ###Not sure we need to cast mask into an integer here
    mask.requires_grad_(False)
    return mask
