from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Hashable, Mapping

import torch

from halo.core.backend import AuditObservation
from halo.core.examples import AuditExample, DeletionManifest
from halo.core.states import DatabaseState


def _auto_device_dtype() -> tuple[str, torch.dtype]:
    """Best available device and a sane dtype for it — no user flags needed."""
    if torch.cuda.is_available():
        return "cuda:0", torch.bfloat16
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


def _clean_completion(completion: str) -> str:
    # A base LM often continues past the answer onto a new line or sentence;
    # the answer is the first line's first sentence.
    completion = str(completion).strip().split("\n", maxsplit=1)[0]
    completion = re.sub(r"\s+", " ", completion).strip()
    for prefix in ("answer:", "the answer is", "it is", "it's"):
        if completion.casefold().startswith(prefix):
            completion = completion[len(prefix) :].strip()
            break
    completion = re.split(r"(?<=[.!?])\s+", completion, maxsplit=1)[0]
    return completion.strip(" \t\n\r\"'`,;:.")


def extract_smollm2_answer(raw_completion: str) -> str:
    return _clean_completion(raw_completion)


@dataclass
class SmolLM2AuditBackend:
    """A parametric (closed-book) causal LM under audit.

    There is no retrieval index and no external memory, so no database state
    consults a deletion manifest: FULL, DEL-ON, and DEL-OFF are the same
    computation by construction. The audit's cross-state metrics then read
    out parametric knowledge alone — L(f) equals closed-book correctness and
    R(f)/I(f) are identically zero. Decoding is greedy, which the capability
    hooks below and the harness's reuse canaries rely on.
    """

    model: Any
    tokenizer: Any
    model_path: str | None = None
    answer_extractor: Callable[[str], str] = extract_smollm2_answer

    @classmethod
    def from_pretrained(cls, *, model_path: str) -> "SmolLM2AuditBackend":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device, torch_dtype = _auto_device_dtype()
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch_dtype
        )
        model.to(device)
        model.eval()
        return cls(model=model, tokenizer=tokenizer, model_path=str(model_path))

    def manifest_fingerprint(self, manifest: DeletionManifest) -> Hashable:
        """Capability hook: generation never consults a deletion manifest —
        there is no retrieval — so every manifest shares one fingerprint."""
        return ("parametric",)

    def cross_phase_fingerprint(
        self, state: DatabaseState, manifest: DeletionManifest
    ) -> Hashable:
        """Capability hook: with greedy decoding and no retrieval, output is a
        pure function of (prompt, max_new_tokens) in every state; neither the
        manifest nor a database is ever consulted."""
        return ("parametric", state.value)

    def full_row_unaffected(
        self, full_row: Mapping[str, Any], manifest: DeletionManifest
    ) -> bool:
        """Capability hook: no manifest can change a generation that never
        performs retrieval."""
        return True

    def generate(
        self,
        example: AuditExample,
        state: DatabaseState,
        *,
        max_new_tokens: int = 12,
    ) -> AuditObservation:
        # Deliberately no empty-manifest guard for non-FULL states: deletion
        # is a no-op by construction, and each state still runs as its own
        # honest forward pass (mirroring Co-LMLM: reuse happens through the
        # capability hooks, never inside generate()).
        encoded = self.tokenizer(example.prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self.model.device)
        attention_mask = encoded["attention_mask"].to(self.model.device)
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        start = time.perf_counter()
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
            )
        t_generate_s = time.perf_counter() - start

        completion_ids = output_ids[0][input_ids.shape[1] :]
        raw_completion = self.tokenizer.decode(
            completion_ids, skip_special_tokens=True
        )
        return AuditObservation(
            model_output=self.answer_extractor(raw_completion),
            # None lets the core stamp the state-correct empty trace: no
            # retrieval fired, and DEL-OFF reports retrieval disabled.
            retrieval_trace=None,
            generation_metadata={
                "raw_text": example.prompt + raw_completion,
                "raw_completion": raw_completion,
                "t_generate_s": float(t_generate_s),
                "gen_decoded_tokens": int(completion_ids.shape[0]),
                "model_path": self.model_path,
            },
        )
