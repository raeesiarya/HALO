from __future__ import annotations

import importlib
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from halo.core.backend import AuditObservation
from halo.interventions.judge import default_support_judge
from halo.interventions.errors import AuditIntegrationError
from halo.interventions.filtering import _FilteringSearchIndex
from halo.core.examples import AuditExample
from halo.core.states import DatabaseState

_FACT_BLOCK_PATTERN = re.compile(r"<FACT>.*?</FACT>", re.DOTALL)
_SPECIAL_TOKEN_PATTERN = re.compile(r"</?[A-Z_]+>")

_SQLITE_MAPPING_NAMES = ("faiss_id_to_entry_id.db", "faiss_id_to_entry_id.sqlite")


def _auto_device_dtype() -> tuple[str, str]:
    """Best available device and a sane dtype for it — no user flags needed."""
    try:
        import torch
    except ImportError:
        return "cpu", "float32"
    if torch.cuda.is_available():
        return "cuda:0", "bfloat16"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps", "float32"
    return "cpu", "float32"


def _has_sqlite_mapping(index_path: Path) -> bool:
    return any((index_path / name).exists() for name in _SQLITE_MAPPING_NAMES)


def extract_colmlm_answer(raw_text: str, prompt: str) -> str:
    completion = str(raw_text)
    if prompt and completion.startswith(prompt):
        completion = completion[len(prompt) :]
    completion = _FACT_BLOCK_PATTERN.sub(" ", completion)
    completion = _SPECIAL_TOKEN_PATTERN.sub(" ", completion)
    completion = re.sub(r"\s+", " ", completion).strip()
    for prefix in ("answer:", "the answer is", "it is", "it's"):
        if completion.casefold().startswith(prefix):
            completion = completion[len(prefix) :].strip()
            break
    completion = re.split(r"(?<=[.!?])\s+", completion, maxsplit=1)[0]
    return completion.strip(" \t\n\r\"'`,;:.")


@dataclass
class CoLMLMAuditBackend:
    generator: Any
    support_judge: Callable[[Any, AuditExample], Mapping[str, Any]] = (
        default_support_judge
    )
    answer_extractor: Callable[[str, str], str] = extract_colmlm_answer
    max_filter_overfetch: int = 4096
    del_off_mode: str = "null-retrieval"
    release_source: str | None = None
    # Synthetic index entries (adversarial survivors) active for subsequent
    # generate() calls; set/cleared by the adversarial runner.
    injections: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        if self.max_filter_overfetch < 0:
            raise ValueError("max_filter_overfetch cannot be negative.")
        if self.del_off_mode not in ("null-retrieval", "forbid-token"):
            raise ValueError(
                "del_off_mode must be 'null-retrieval' or 'forbid-token', "
                f"got {self.del_off_mode!r}."
            )

    @classmethod
    def from_public_release(
        cls,
        *,
        model_path: str | Path,
        index_path: str | Path,
        db_path: str | Path | None = None,
        source_path: str | Path | None = None,
        similarity_threshold: float | None = None,
        nprobe: int | None = None,
        max_new_tokens: int = 12,
        del_off_mode: str = "null-retrieval",
    ) -> "CoLMLMAuditBackend":
        release_source = None
        if source_path is not None:
            source_root = Path(source_path).expanduser().resolve()
            source_src = (
                source_root / "src" if (source_root / "src").is_dir() else source_root
            )
            if not (source_src / "lmlm" / "eval" / "hf_generate.py").is_file():
                raise FileNotFoundError(
                    f"No Co-LMLM public source found below {source_src}."
                )
            loaded_lmlm = sys.modules.get("lmlm")
            loaded_file = getattr(loaded_lmlm, "__file__", None)
            if (
                loaded_file is not None
                and source_src not in Path(loaded_file).resolve().parents
            ):
                raise AuditIntegrationError(
                    "A different `lmlm` package is already imported. Run Co-LMLM "
                    "in its own process/environment to avoid a namespace collision."
                )
            sys.path.insert(0, str(source_src))
            release_source = str(source_root)

        try:
            module = importlib.import_module("lmlm.eval.hf_generate")
            loader = module.load_retriever_generator
        except (ImportError, AttributeError) as exc:
            raise ImportError(
                "The public Co-LMLM release is required. Run this command from the "
                "public Co-LMLM checkout in its Python 3.12 environment."
            ) from exc

        # Everything below is auto-resolved — no user-facing device/dtype/attn/
        # sqlite/mmap flags. Memory-map the (large) FAISS file by default.
        os.environ.setdefault("LMLM_FAISS_MMAP", "1")
        device, torch_dtype = _auto_device_dtype()
        loader_kwargs: dict[str, Any] = dict(
            model_path=str(model_path),
            index_path=Path(index_path),
            db_path=Path(db_path) if db_path is not None else None,
            use_sqlite_id_mapping=_has_sqlite_mapping(Path(index_path)),
            device=device,
            torch_dtype=torch_dtype,
            similarity_threshold=similarity_threshold,
            retrieval_top_k=1,
            max_new_tokens=max_new_tokens,
        )
        if nprobe is not None:
            loader_kwargs["nprobe"] = nprobe

        try:
            generator = loader(attn_implementation="flash_attention_2", **loader_kwargs)
        except Exception:
            # flash-attention-2 is often unavailable; fall back to eager.
            generator = loader(attn_implementation=None, **loader_kwargs)
        return cls(
            generator=generator,
            del_off_mode=del_off_mode,
            release_source=release_source,
        )

    def generate(
        self,
        example: AuditExample,
        state: DatabaseState,
        *,
        max_new_tokens: int = 12,
    ) -> AuditObservation:
        manifest = example.deletion_manifest
        if state is not DatabaseState.FULL and manifest.is_empty:
            raise ValueError(
                f"{state.value} requires deletion_entry_ids, oracle_entry_ids, "
                "source_ids, or an explicit deletion_manifest."
            )

        generation_config = getattr(self.generator, "generation_config", None)
        previous_max_tokens = getattr(generation_config, "max_new_tokens", None)
        if generation_config is not None:
            generation_config.max_new_tokens = max_new_tokens

        original_index = getattr(self.generator, "index", None)
        filtered_index: _FilteringSearchIndex | None = None
        try:
            if state is DatabaseState.DEL_OFF and self.del_off_mode == "forbid-token":
                no_retrieval = getattr(self.generator, "generate_no_retrieval", None)
                if no_retrieval is None:
                    raise AuditIntegrationError(
                        "This Co-LMLM generator has no generate_no_retrieval() method."
                    )
                result = no_retrieval(example.prompt)
            else:
                if original_index is None:
                    raise AuditIntegrationError(
                        "The Co-LMLM generator does not expose its search index."
                    )
                manifest_metadata = (
                    manifest.metadata if isinstance(manifest.metadata, Mapping) else {}
                )
                manifest_predicates = manifest_metadata.get("predicates_active")
                semantic_backstop = (
                    state is DatabaseState.DEL_ON
                    and isinstance(manifest_predicates, (list, tuple))
                    and "semantic" in manifest_predicates
                )
                semantic_target = manifest_metadata.get("semantic_target")
                backstop_example = None
                if semantic_backstop and isinstance(semantic_target, Mapping):
                    # Judge against the deleted fact's answer, which is not
                    # necessarily this prompt's answer (neighbor prompts in a
                    # sweep run under the target fact's manifest).
                    backstop_example = AuditExample(
                        prompt="",
                        ground_truth=str(semantic_target.get("ground_truth", "")),
                        object_aliases=tuple(
                            str(alias)
                            for alias in (semantic_target.get("object_aliases") or ())
                        ),
                    )
                filtered_index = _FilteringSearchIndex(
                    base_index=original_index,
                    example=example,
                    excluded_entry_ids=frozenset(
                        manifest.entry_ids if state is DatabaseState.DEL_ON else ()
                    ),
                    excluded_source_ids=frozenset(
                        manifest.source_ids if state is DatabaseState.DEL_ON else ()
                    ),
                    exclude_all=state is DatabaseState.DEL_OFF,
                    exclude_supporting=semantic_backstop,
                    backstop_example=backstop_example,
                    injections=tuple(self.injections),
                    support_judge=self.support_judge,
                    max_filter_overfetch=self.max_filter_overfetch,
                )
                self.generator.index = filtered_index
                result = self.generator.generate(example.prompt)
        finally:
            if original_index is not None:
                self.generator.index = original_index
            if generation_config is not None and previous_max_tokens is not None:
                generation_config.max_new_tokens = previous_max_tokens

        raw_text = str(getattr(result, "text", ""))
        events = filtered_index.events if filtered_index is not None else []
        all_candidates = [
            candidate for event in events for candidate in event["all_candidates"]
        ]
        deleted_candidates = [
            candidate for event in events for candidate in event["deleted_candidates"]
        ]
        retained_candidates = [
            candidate for event in events for candidate in event["retained_candidates"]
        ]
        num_retrievals = int(getattr(result, "num_retrievals", 0) or 0)
        failed_retrievals = int(getattr(result, "failed_retrievals", 0) or 0)
        selected_candidate = next(
            (
                event["selected_candidate"]
                for event in events
                if event["selected_candidate"]
            ),
            None,
        )
        retrieval_trace = {
            "state": state.value,
            "trace_available": True,
            "trace_complete": True,
            "retrieval_enabled": state is not DatabaseState.DEL_OFF,
            "del_off_mode": (
                self.del_off_mode if state is DatabaseState.DEL_OFF else None
            ),
            "retrieval_triggered": bool(events or num_retrievals or failed_retrievals),
            "threshold_fallback": failed_retrievals > 0,
            "lookup_query": None,
            "threshold": getattr(
                getattr(self.generator, "retrieval_config", None),
                "similarity_threshold",
                None,
            ),
            "all_candidates": all_candidates,
            "deleted_candidates": deleted_candidates,
            "retained_candidates": retained_candidates,
            "selected_candidate": selected_candidate,
            "selected_value": (
                selected_candidate.get("value") if selected_candidate else None
            ),
            "retrieval_events": events,
            "num_retrievals": num_retrievals,
            "failed_retrievals": failed_retrievals,
            "deletion_manifest_id": manifest.manifest_id,
            "error": None,
        }
        generation_metadata = {
            "raw_text": raw_text,
            "num_retrievals": num_retrievals,
            "failed_retrievals": failed_retrievals,
            "t_generate_s": float(getattr(result, "t_generate_s", 0.0) or 0.0),
            "t_encode_s": float(getattr(result, "t_encode_s", 0.0) or 0.0),
            "t_search_s": float(getattr(result, "t_search_s", 0.0) or 0.0),
            "gen_decoded_tokens": int(getattr(result, "gen_decoded_tokens", 0) or 0),
            "release_source": self.release_source,
        }
        query_embeddings = tuple(
            {"event_index": index, "vector": vector}
            for index, vector in enumerate(
                filtered_index.query_embeddings if filtered_index is not None else []
            )
            if vector is not None
        )
        return AuditObservation(
            model_output=self.answer_extractor(raw_text, example.prompt),
            retrieval_trace=retrieval_trace,
            generation_metadata=generation_metadata,
            query_embeddings=query_embeddings,
        )
