"""Audit backend for NULLs (natively unlearnable LM), Wikipedia release.

A parametric model whose memory is a pool of source-keyed sink neurons
(arXiv:2606.13873). Unlike the closed-book baselines, the three database
states are genuinely different computations (docs/NULLS_AUDIT_DESIGN.md §3):

- FULL: the gold article's sink mask is active (the paper's Sink-On).
- DEL-ON: every manifest source's sink mask is excluded (union), and the
  next-closest *surviving* source's mask is activated instead — the memory
  system keeps operating over what survives, mirroring Co-LMLM's DEL-ON.
- DEL-OFF: the sink memory is disabled. Two modes form the sensitivity pair
  (design §6): ``sinks-zero`` (backbone only, upstream's ``dropout`` mode)
  and ``placebo-sink`` (a deterministic far-away source's mask is activated,
  keeping activation statistics on-distribution while providing no
  fact-relevant memory; manifest and gold masks are still excluded).

Deletion manifests carry Wikipedia article titles in ``source_ids``; the
``entry_ids`` field has no referent here and must be empty.

Decoding is greedy with a full-context re-forward per token: upstream's
generator is not a faithful decoder (see vendor/NOTICE.md) and correctness
outranks speed at audit completion lengths.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Hashable, Sequence

import torch

from halo.core.backend import AuditObservation
from halo.core.examples import AuditExample, DeletionManifest
from halo.core.states import DatabaseState
from models.nulls_wiki.routing import SinkRegistry

DEL_OFF_MODES = ("sinks-zero", "placebo-sink")
SOURCE_TITLE_FIELD = "source_title"
# Trained context length (hyperparameters.yaml max_seq_length), not the
# architectural block_size: activations beyond it are off-distribution.
TRAINED_CONTEXT = 1024


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


def extract_nulls_answer(raw_completion: str) -> str:
    return _clean_completion(raw_completion)


def load_seqtd_checkpoint(
    checkpoint_dir: str | Path, *, device: str, dtype: torch.dtype
) -> tuple[Any, Any]:
    """Load a released NULLs checkpoint (litgpt SeqTD layout) and tokenizer.

    The plain-``transformers`` ``config.json`` in the release does NOT match
    the weights (vendor/NOTICE.md); only ``model_config.yaml`` +
    ``lit_model.pth`` are authoritative.
    """
    from transformers import AutoTokenizer

    from models.nulls_wiki.vendor.seqtd_config import SeqTDConfig
    from models.nulls_wiki.vendor.seqtd_model import GPTSeqTD

    checkpoint_dir = Path(checkpoint_dir)
    config = SeqTDConfig.from_checkpoint(checkpoint_dir)
    if not config.use_seqtd or config.mlp_class_name != "LLaMAMLPSeqTD":
        raise ValueError(
            f"{checkpoint_dir} is not a SeqTD checkpoint "
            f"(use_seqtd={config.use_seqtd}, mlp={config.mlp_class_name})."
        )

    state = torch.load(
        checkpoint_dir / "lit_model.pth", map_location="cpu", mmap=True, weights_only=False
    )
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    state = {
        key.removeprefix("_orig_mod."): value
        for key, value in state.items()
        if isinstance(value, torch.Tensor)
    }

    with torch.device("meta"):
        model = GPTSeqTD(config)
    model.load_state_dict(state, strict=True, assign=True)
    model.max_seq_length = TRAINED_CONTEXT
    model = model.to(dtype=dtype, device=device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir))
    return model, tokenizer


@dataclass
class NullsWikiAuditBackend:
    """NULLs under audit: sink masking is the deletion operator."""

    model: Any
    tokenizer: Any
    registry: SinkRegistry
    del_off_mode: str = "sinks-zero"
    checkpoint_dir: str | None = None
    placebo_similarity_ceiling: float = 0.5
    answer_extractor: Any = field(default=extract_nulls_answer)

    def __post_init__(self) -> None:
        if self.del_off_mode not in DEL_OFF_MODES:
            raise ValueError(
                f"del_off_mode must be one of {', '.join(DEL_OFF_MODES)}, "
                f"got {self.del_off_mode!r}."
            )

    @classmethod
    def from_release(
        cls,
        *,
        checkpoint_dir: str | Path,
        title_to_index_path: str | Path,
        title_embeddings_path: str | Path | None,
        del_off_mode: str = "sinks-zero",
        placebo_similarity_ceiling: float = 0.5,
    ) -> "NullsWikiAuditBackend":
        device, dtype = _auto_device_dtype()
        model, tokenizer = load_seqtd_checkpoint(
            checkpoint_dir, device=device, dtype=dtype
        )
        registry = SinkRegistry.load(
            title_to_index_path=title_to_index_path,
            vocab_size=int(model.config.vocab_size),
            embeddings_path=title_embeddings_path,
        )
        return cls(
            model=model,
            tokenizer=tokenizer,
            registry=registry,
            del_off_mode=del_off_mode,
            checkpoint_dir=str(checkpoint_dir),
            placebo_similarity_ceiling=placebo_similarity_ceiling,
        )

    # ----- sink resolution -------------------------------------------------

    def _gold_title(self, example: AuditExample) -> str:
        title = example.source_row.get(SOURCE_TITLE_FIELD) or example.subject
        if not title:
            raise ValueError(
                f"Fact {example.fact_id!r} has no {SOURCE_TITLE_FIELD!r} field and "
                "no subject; run scripts/augment_source_titles.py first."
            )
        title = str(title)
        if title not in self.registry:
            raise ValueError(
                f"Fact {example.fact_id!r}: title {title!r} is not in "
                "title_to_index. Unmappable facts must be excluded by the "
                "verification gate (scripts/nulls_verification_gate.py)."
            )
        return title

    def _manifest_titles(self, manifest: DeletionManifest) -> list[str]:
        if manifest.entry_ids:
            raise ValueError(
                "NULLs deletion manifests must use source_ids (article "
                f"titles); got entry_ids={manifest.entry_ids!r}."
            )
        for title in manifest.source_ids:
            if title not in self.registry:
                raise ValueError(
                    f"Manifest names unknown title {title!r}; its sink mask "
                    "cannot be computed, so the deletion would silently not "
                    "happen. Fix the manifest or the title mapping."
                )
        return list(manifest.source_ids)

    def _fact_key(self, example: AuditExample) -> str:
        return f"{example.fact_id}::{example.prompt_id}"

    # ----- sink plan per state --------------------------------------------

    def _sink_plan(
        self, example: AuditExample, state: DatabaseState
    ) -> dict[str, Any]:
        """Resolve (eval_mode, active title, excluded titles) for a state."""
        manifest = example.deletion_manifest
        if state is DatabaseState.FULL:
            gold = self._gold_title(example)
            return {
                "eval_mode": "activate_seq",
                "active_title": gold,
                "excluded_titles": [],
                "routing": None,
            }
        if state is DatabaseState.DEL_ON:
            if manifest.is_empty:
                raise ValueError(
                    "DEL-ON requires a non-empty deletion manifest for "
                    "nulls-wiki-1b; deletion without a manifest has no meaning."
                )
            gold = self._gold_title(example)
            excluded = self._manifest_titles(manifest)
            if gold not in excluded:
                active, routing = gold, None
            else:
                neighbors = self.registry.nearest(gold, exclude=excluded, top_k=5)
                if not neighbors:
                    raise RuntimeError(
                        f"No surviving routing target for fact {example.fact_id!r}."
                    )
                active = neighbors[0][0]
                routing = neighbors
            return {
                "eval_mode": "activate_seq",
                "active_title": active,
                "excluded_titles": excluded,
                "routing": routing,
            }
        # DEL-OFF
        gold = self._gold_title(example)
        excluded = sorted(set(self._manifest_titles(manifest)) | {gold})
        if self.del_off_mode == "sinks-zero":
            return {
                "eval_mode": "dropout",
                "active_title": None,
                "excluded_titles": excluded,
                "routing": None,
            }
        placebo = self.registry.placebo(
            self._fact_key(example),
            avoid=excluded,
            anchor_title=gold,
            similarity_ceiling=self.placebo_similarity_ceiling,
        )
        return {
            "eval_mode": "activate_seq",
            "active_title": placebo,
            "excluded_titles": excluded,
            "routing": None,
        }

    # ----- forward / decoding ---------------------------------------------

    def _sink_tensors(
        self, plan: dict[str, Any], shape: tuple[int, int], device: torch.device
    ) -> tuple[torch.Tensor | None, list[torch.Tensor] | None]:
        seq_ids = None
        if plan["active_title"] is not None:
            active_id = self.registry.seq_id(plan["active_title"])
            seq_ids = torch.full(shape, active_id, dtype=torch.int32, device=device)
        exclude = None
        if plan["eval_mode"] == "activate_seq" and plan["excluded_titles"]:
            exclude = [
                torch.full(
                    shape,
                    self.registry.seq_id(title),
                    dtype=torch.int32,
                    device=device,
                )
                for title in plan["excluded_titles"]
            ]
        return seq_ids, exclude

    def _logits(self, input_ids: torch.Tensor, plan: dict[str, Any]) -> torch.Tensor:
        seq_ids, exclude = self._sink_tensors(
            plan, tuple(input_ids.shape), input_ids.device
        )
        return self.model(
            idx=input_ids,
            seq_ids=seq_ids,
            exclude_seq_ids=exclude,
            eval_mode=plan["eval_mode"],
        )

    def _encode_prompt(self, prompt: str, *, reserve: int) -> torch.Tensor:
        device = next(self.model.parameters()).device
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
        budget = TRAINED_CONTEXT - reserve
        if input_ids.shape[1] > budget:
            input_ids = input_ids[:, -budget:]
        return input_ids.to(device)

    def _greedy_decode(
        self, prompt: str, plan: dict[str, Any], *, max_new_tokens: int
    ) -> tuple[str, int]:
        input_ids = self._encode_prompt(prompt, reserve=max_new_tokens)
        eos_id = self.tokenizer.eos_token_id
        generated: list[int] = []
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self._logits(input_ids, plan)
                next_id = int(torch.argmax(logits[0, -1]).item())
                generated.append(next_id)
                if eos_id is not None and next_id == eos_id:
                    break
                input_ids = torch.cat(
                    [
                        input_ids,
                        torch.tensor(
                            [[next_id]], dtype=input_ids.dtype, device=input_ids.device
                        ),
                    ],
                    dim=1,
                )
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return text, len(generated)

    def answer_logprob(
        self,
        prompt: str,
        answer: str,
        *,
        eval_mode: str,
        active_title: str | None = None,
        excluded_titles: Sequence[str] = (),
    ) -> float:
        """Sum log-probability of ``answer`` tokens given ``prompt`` under an
        explicit sink configuration. Used by the verification gate and the
        secondary truth-ratio readout — not by the audit path."""
        plan = {
            "eval_mode": eval_mode,
            "active_title": active_title,
            "excluded_titles": list(excluded_titles),
            "routing": None,
        }
        device = next(self.model.parameters()).device
        prompt_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
        answer_ids = self.tokenizer(answer, return_tensors="pt")["input_ids"]
        input_ids = torch.cat([prompt_ids, answer_ids], dim=1).to(device)
        if input_ids.shape[1] > TRAINED_CONTEXT:
            raise ValueError("prompt+answer exceeds the trained context length.")
        with torch.no_grad():
            logits = self._logits(input_ids, plan)
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        total = 0.0
        prompt_len = prompt_ids.shape[1]
        for position in range(answer_ids.shape[1]):
            token_id = int(input_ids[0, prompt_len + position])
            total += float(log_probs[0, prompt_len + position - 1, token_id])
        return total

    # ----- capability hooks ------------------------------------------------

    def manifest_fingerprint(self, manifest: DeletionManifest) -> Hashable:
        """For a fixed prompt and state, generation depends on the manifest
        only through its excluded titles (masks are pure functions of them)."""
        return ("nulls-sources", tuple(sorted(manifest.source_ids)))

    def cross_phase_fingerprint(
        self, state: DatabaseState, manifest: DeletionManifest
    ) -> Hashable:
        """Greedy decoding over a fixed checkpoint: output is a pure function
        of (prompt, state, and the state's effective sink configuration).
        FULL activates the gold sink regardless of the manifest, and the
        sinks-zero DEL-OFF ignores exclusions entirely, so neither includes
        manifest sources — this is what lets sweep and del-off phases reuse
        the standard phase's FULL rows across differing manifests."""
        base = ("nulls-wiki", self.checkpoint_dir, state.value)
        if state is DatabaseState.FULL:
            return base
        if state is DatabaseState.DEL_OFF:
            if self.del_off_mode == "sinks-zero":
                return (*base, "sinks-zero")
            return (*base, "placebo-sink", tuple(sorted(manifest.source_ids)))
        return (*base, tuple(sorted(manifest.source_ids)))

    # Deletion genuinely changes generations: never claim full_row_unaffected.

    # ----- audit entry point ----------------------------------------------

    def generate(
        self,
        example: AuditExample,
        state: DatabaseState,
        *,
        max_new_tokens: int = 12,
    ) -> AuditObservation:
        plan = self._sink_plan(example, state)
        start = time.perf_counter()
        raw_completion, decoded_tokens = self._greedy_decode(
            example.prompt, plan, max_new_tokens=max_new_tokens
        )
        t_generate_s = time.perf_counter() - start

        trace = self._trace(example, state, plan)
        metadata = {
            "raw_completion": raw_completion,
            "t_generate_s": float(t_generate_s),
            "gen_decoded_tokens": decoded_tokens,
            "checkpoint_dir": self.checkpoint_dir,
            "eval_mode": plan["eval_mode"],
            "active_sink_title": plan["active_title"],
            "active_sink_seq_id": (
                self.registry.seq_id(plan["active_title"])
                if plan["active_title"] is not None
                else None
            ),
            "excluded_sink_titles": list(plan["excluded_titles"]),
        }
        return AuditObservation(
            model_output=self.answer_extractor(raw_completion),
            retrieval_trace=trace,
            generation_metadata=metadata,
        )

    def _trace(
        self, example: AuditExample, state: DatabaseState, plan: dict[str, Any]
    ) -> dict[str, Any]:
        def sink_candidate(title: str, score: float | None = None) -> dict[str, Any]:
            return {
                "entry_id": f"sink::{title}",
                "source_id": title,
                "score": score,
            }

        if state is DatabaseState.DEL_OFF:
            return {
                "state": state.value,
                "retrieval_enabled": False,
                "retrieval_triggered": False,
                "del_off_mode": self.del_off_mode,
                "selected_candidate": None,
                "retained_candidates": [],
                "deleted_candidates": [
                    sink_candidate(title) for title in plan["excluded_titles"]
                ],
                "retrieval_events": [],
            }
        selected = sink_candidate(plan["active_title"])
        retained = [selected]
        if plan["routing"]:
            selected = sink_candidate(*plan["routing"][0])
            retained = [sink_candidate(title, score) for title, score in plan["routing"]]
        return {
            "state": state.value,
            "retrieval_enabled": True,
            "retrieval_triggered": True,
            "del_off_mode": None,
            "selected_candidate": selected,
            "retained_candidates": retained,
            "deleted_candidates": [
                sink_candidate(title) for title in plan["excluded_titles"]
            ],
            "retrieval_events": [],
        }
