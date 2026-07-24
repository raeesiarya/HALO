from __future__ import annotations

import importlib
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Hashable, Mapping

from halo.core.backend import AuditObservation
from halo.interventions.judge import default_support_judge
from halo.interventions.errors import AuditIntegrationError
from halo.interventions.filtering import _FilteringSearchIndex
from halo.core.examples import AuditExample, DeletionManifest
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


def _clean_completion(completion: str) -> str:
    completion = _FACT_BLOCK_PATTERN.sub(" ", completion)
    completion = _SPECIAL_TOKEN_PATTERN.sub(" ", completion)
    completion = re.sub(r"\s+", " ", completion).strip()
    for prefix in ("answer:", "the answer is", "it is", "it's"):
        if completion.casefold().startswith(prefix):
            completion = completion[len(prefix) :].strip()
            break
    completion = re.split(r"(?<=[.!?])\s+", completion, maxsplit=1)[0]
    return completion.strip(" \t\n\r\"'`,;:.")


def extract_colmlm_answer(raw_text: str, prompt: str) -> str:
    completion = str(raw_text)
    if prompt and completion.startswith(prompt):
        completion = completion[len(prompt) :]
    return _clean_completion(completion)


def _manifest_value_backstop(
    manifest: DeletionManifest,
) -> tuple[bool, AuditExample | None]:
    """Whether the manifest activates the run-time value backstop, and the
    example whose answer it judges against. The example is None when the
    manifest carries no explicit target (the filter would then fall back
    to judging against whichever prompt is running)."""
    metadata = manifest.metadata if isinstance(manifest.metadata, Mapping) else {}
    predicates = metadata.get("predicates_active")
    # Compatibility with manifests written before the value rename.
    active = isinstance(predicates, (list, tuple)) and bool(
        {"value", "semantic"}.intersection(predicates)
    )
    if not active:
        return False, None
    value_target = metadata.get("value_target", metadata.get("semantic_target"))
    if not isinstance(value_target, Mapping):
        return True, None
    return True, AuditExample(
        prompt="",
        ground_truth=str(value_target.get("ground_truth", "")),
        object_aliases=tuple(
            str(alias) for alias in (value_target.get("object_aliases") or ())
        ),
    )


def manifest_reuse_fingerprint(manifest: DeletionManifest) -> Hashable | None:
    """Everything a DEL-ON generation can observe of a deletion manifest.

    Generation depends on the manifest only through the retrieval filters:
    the excluded entry/source IDs and the value-backstop target. Radius,
    entry counts, and the rest of the metadata are bookkeeping. Declines
    (None) when the backstop is active without an explicit target, because
    the filter would then depend implicitly on the running prompt.
    """
    backstop_active, backstop_example = _manifest_value_backstop(manifest)
    if backstop_active and backstop_example is None:
        return None
    return (
        frozenset(manifest.entry_ids),
        frozenset(manifest.source_ids),
        backstop_active,
        (
            (backstop_example.ground_truth, backstop_example.object_aliases)
            if backstop_example is not None
            else None
        ),
    )


def full_trace_unaffected(
    full_row: Mapping[str, Any],
    manifest: DeletionManifest,
    support_judge: Callable[[Any, AuditExample], Mapping[str, Any]],
) -> bool:
    """Whether ``manifest`` provably cannot change the generation recorded in
    ``full_row``.

    Co-LMLM decodes greedily and its deletion filter only *removes*
    candidates, so retrieval results — and therefore the whole generation —
    are unchanged whenever every candidate the FULL pass actually saw
    survives the manifest's three filters (entry ID, source ID, value
    backstop). The check walks the FULL trace's retrieval events; events
    that returned nothing stay empty under any deletion.
    """
    trace = (full_row or {}).get("retrieval_trace") or {}
    if trace.get("state") != DatabaseState.FULL.value:
        return False
    if not trace.get("trace_available") or not trace.get("trace_complete"):
        return False
    backstop_active, backstop_example = _manifest_value_backstop(manifest)
    if backstop_active and backstop_example is None:
        return False
    excluded_entry_ids = set(manifest.entry_ids)
    excluded_source_ids = set(manifest.source_ids)
    for event in trace.get("retrieval_events") or []:
        if event.get("injected_candidates_count"):
            return False
        for candidate in event.get("all_candidates") or []:
            if str(candidate.get("entry_id")) in excluded_entry_ids:
                return False
            source_id = candidate.get("source_id")
            if source_id is not None and str(source_id) in excluded_source_ids:
                return False
            if backstop_active and dict(
                support_judge(candidate, backstop_example)
            ).get("supports_target"):
                return False
    return True


@dataclass
class CoLMLMAuditBackend:
    generator: Any
    support_judge: Callable[[Any, AuditExample], Mapping[str, Any]] = (
        default_support_judge
    )
    answer_extractor: Callable[[str, str], str] = extract_colmlm_answer
    max_filter_overfetch: int = 4096
    max_filter_search_k: int = 131072
    del_off_mode: str = "null-retrieval"
    # full_row_unaffected reuse is only sound when the index's top-1 is
    # independent of the requested k. DEL-ON over-fetches (k = 1 + |entry_ids|)
    # while the FULL pass searches at k=1, so on an *approximate* index
    # (IVF/PQ/OPQ) a deeper search can return a different top-1 even when the
    # manifest deletes nothing the FULL pass saw — the reuse canary catches this
    # as a full_row_unaffected failure. Default off; opt in only for an exact
    # (Flat) index. The fingerprint hook stays sound regardless (same manifest
    # => same search_k => same result under greedy decoding).
    assume_exact_index: bool = False
    release_source: str | None = None
    model_path: str | None = None
    index_path: str | None = None
    # Synthetic index entries (adversarial survivors) active for subsequent
    # generate() calls; set/cleared by the adversarial runner.
    injections: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        if self.max_filter_overfetch < 0:
            raise ValueError("max_filter_overfetch cannot be negative.")
        if self.max_filter_search_k < self.max_filter_overfetch:
            raise ValueError(
                "max_filter_search_k cannot be smaller than max_filter_overfetch."
            )
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
        assume_exact_index: bool = False,
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
            assume_exact_index=assume_exact_index,
            release_source=release_source,
            model_path=str(model_path),
            index_path=str(Path(index_path).expanduser().resolve()),
        )

    def manifest_fingerprint(self, manifest: DeletionManifest) -> Hashable | None:
        """Capability hook: DEL-ON output depends on the manifest only through
        its retrieval filters (decoding is greedy). No claim while adversarial
        injections are active — they change retrieval independently of the
        manifest."""
        if self.injections:
            return None
        return manifest_reuse_fingerprint(manifest)

    def full_row_unaffected(
        self, full_row: Mapping[str, Any], manifest: DeletionManifest
    ) -> bool:
        """Capability hook: with greedy decoding and removal-only filtering,
        a manifest that catches nothing the FULL pass retrieved cannot change
        the generation — but only if the index's top-1 does not depend on the
        search depth (see ``assume_exact_index``). DEL-ON over-fetches relative
        to the FULL pass, so on an approximate index this claim is unsound and
        the hook makes none."""
        if self.injections or not self.assume_exact_index:
            return False
        return full_trace_unaffected(full_row, manifest, self.support_judge)

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
                backstop_active, backstop_target = _manifest_value_backstop(
                    manifest
                )
                value_backstop = (
                    state is DatabaseState.DEL_ON and backstop_active
                )
                backstop_example = (
                    backstop_target if value_backstop else None
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
                    exclude_supporting=value_backstop,
                    backstop_example=backstop_example,
                    injections=tuple(self.injections),
                    support_judge=self.support_judge,
                    max_filter_overfetch=self.max_filter_overfetch,
                    max_filter_search_k=self.max_filter_search_k,
                    slim_trace=state is not DatabaseState.FULL,
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
        selected_candidates = [
            event["selected_candidate"]
            for event in events
            if event["selected_candidate"]
        ]
        # Prefer the selection that supports the target: the generation may
        # look up other attributes (nationality, birth year) before the one
        # the prompt asks about, and the oracle bootstrap judges only this
        # candidate.
        selected_candidate = next(
            (
                candidate
                for candidate in selected_candidates
                if candidate.get("supports_target") is True
            ),
            selected_candidates[0] if selected_candidates else None,
        )
        all_candidates_count = sum(
            event.get("all_candidates_count", len(event["all_candidates"]))
            for event in events
        )
        deleted_candidates_count = sum(
            event.get("deleted_candidates_count", len(event["deleted_candidates"]))
            for event in events
        )
        retained_candidates_count = sum(
            event.get("retained_candidates_count", len(event["retained_candidates"]))
            for event in events
        )
        # In slim mode the trace-level aggregates above already carry the
        # candidate records; keeping them inside each event too would double
        # the row for nothing (only FULL events are ever walked downstream).
        if state is not DatabaseState.FULL:
            events = [
                {
                    key: value
                    for key, value in event.items()
                    if key
                    not in (
                        "all_candidates",
                        "deleted_candidates",
                        "retained_candidates",
                        "selected_candidate",
                    )
                }
                for event in events
            ]
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
            "candidates_slim": state is not DatabaseState.FULL,
            "all_candidates_count": all_candidates_count,
            "deleted_candidates_count": deleted_candidates_count,
            "retained_candidates_count": retained_candidates_count,
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
