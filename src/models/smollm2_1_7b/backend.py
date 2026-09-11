from __future__ import annotations

# SmolLM2-1.7B is a plain parametric causal LM; its audit behavior is
# identical to the 360M backend's (greedy decoding, no retrieval, vacuous
# traces), so the implementation is shared. The subclass exists so resume
# identities and traces name the actual model under audit.
from models.smollm2_360m.backend import SmolLM2AuditBackend


class SmolLM2LargeAuditBackend(SmolLM2AuditBackend):
    """SmolLM2-1.7B under audit: the off-the-shelf reference for NULLs.

    NULLs-Wikipedia is ~1B on SmolLM2 geometry, and 1.7B is the nearest
    released SmolLM2 size, so this backend is to `nulls-wiki-1b` what
    `smollm2-360m` is to `co-lmlm`: same family, nearest scale, knowledge
    entirely in the weights. Its L(f) brackets NULLs' Sink-On correctness
    from above, but the bracket is loose by training corpus as well as by
    scale — see the package docstring.
    """
