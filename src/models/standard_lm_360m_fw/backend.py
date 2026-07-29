from __future__ import annotations

# The Standard LM baseline is a plain parametric causal LM; its audit
# behavior is identical to the SmolLM2 backend's (greedy decoding, no
# retrieval, vacuous traces), so the implementation is shared. The subclass
# exists so resume identities and traces name the actual model under audit.
from models.smollm2_360m.backend import SmolLM2AuditBackend


class StandardLMAuditBackend(SmolLM2AuditBackend):
    """The Co-LMLM paper's Standard LM baseline under audit.

    Same SmolLM2-360M architecture and the same FineWeb-Edu corpus as
    CoLMLM-360M-FW, trained from scratch on the unannotated text (fact
    tags stripped) — the matched-data parametric control. Comparing this
    backend's L(f) against Co-LMLM's isolates what the externalization
    recipe moved out of the weights, with training data held constant.
    """
