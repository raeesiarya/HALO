from __future__ import annotations

import importlib
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from audit_backend import AuditExample, AuditObservation
from database_states import DatabaseState
from equivalence import normalize_text


class CoLMLMIntegrationError(RuntimeError):
    pass


class ExclusionSearchExhaustedError(CoLMLMIntegrationError):
    pass


def _candidate_id(candidate: Any) -> str:
    value = getattr(candidate, "id", None)
    if value is None and isinstance(candidate, Mapping):
        value = candidate.get("id")
        if value is None:
            value = candidate.get("entry_id")
    return "" if value is None else str(value)


def _candidate_metadata(candidate: Any) -> dict[str, Any]:
    value = getattr(candidate, "metadata", None)
    if value is None and isinstance(candidate, Mapping):
        value = candidate.get("metadata")
    return dict(value) if isinstance(value, Mapping) else {}


def _candidate_source_id(candidate: Any) -> str | None:
    metadata = _candidate_metadata(candidate)
    for key in ("source_id", "source", "document_id", "sample_id", "url"):
        value = metadata.get(key)
        if value is not None:
            return str(value)
    return None


def _candidate_text(candidate: Any) -> str:
    value = getattr(candidate, "text_value", None)
    if value is None and isinstance(candidate, Mapping):
        value = candidate.get("text_value")
        if value is None:
            value = candidate.get("value")
    return "" if value is None else str(value)


def _candidate_score(candidate: Any) -> float | None:
    value = getattr(candidate, "score", None)
    if value is None and isinstance(candidate, Mapping):
        value = candidate.get("score")
    return None if value is None else float(value)


def _default_support_judge(candidate: Any, example: AuditExample) -> dict[str, Any]:
    text = normalize_text(_candidate_text(candidate))
    answers = (example.ground_truth, *example.object_aliases)
    padded_text = f" {text} "
    supports = any(
        normalized and f" {normalized} " in padded_text
        for answer in answers
        if (normalized := normalize_text(answer))
    )
    return {
        "supports_target": supports,
        "support_method": "normalized-answer-mention",
        "support_confidence": 1.0 if supports else 0.0,
    }


def _serialize_candidate(
    candidate: Any,
    example: AuditExample,
    support_judge: Callable[[Any, AuditExample], Mapping[str, Any]],
) -> dict[str, Any]:
    metadata = _candidate_metadata(candidate)
    text_key = getattr(candidate, "text_key", None)
    if text_key is None and isinstance(candidate, Mapping):
        text_key = candidate.get("text_key")
    result = {
        "entry_id": _candidate_id(candidate),
        "source_id": _candidate_source_id(candidate),
        "value": _candidate_text(candidate),
        "text_key": text_key,
        "score": _candidate_score(candidate),
        "metadata": metadata,
    }
    result.update(dict(support_judge(candidate, example)))
    return result


@dataclass
class _FilteringSearchIndex:
    base_index: Any
    example: AuditExample
    excluded_entry_ids: frozenset[str]
    excluded_source_ids: frozenset[str]
    support_judge: Callable[[Any, AuditExample], Mapping[str, Any]]
    max_filter_overfetch: int = 4096
    events: list[dict[str, Any]] = field(default_factory=list)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_index, name)

    def _is_excluded(self, candidate: Any) -> bool:
        entry_id = _candidate_id(candidate)
        source_id = _candidate_source_id(candidate)
        return entry_id in self.excluded_entry_ids or (
            source_id is not None and source_id in self.excluded_source_ids
        )

    def search(
        self,
        query_vector: Any,
        top_k: int = 1,
        similarity_threshold: float | None = None,
    ) -> list[Any]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        entry_exclusion_count = len(self.excluded_entry_ids)
        source_exclusions_are_unbounded = bool(self.excluded_source_ids)
        if source_exclusions_are_unbounded:
            extra = self.max_filter_overfetch
        else:
            extra = min(entry_exclusion_count, self.max_filter_overfetch)
        search_k = top_k + extra
        raw = self.base_index.search(
            query_vector,
            top_k=search_k,
            similarity_threshold=similarity_threshold,
        )
        if raw and isinstance(raw[0], list):
            if len(raw) != 1:
                raise CoLMLMIntegrationError(
                    "Co-LMLM generator issued a single query but the index returned "
                    f"{len(raw)} result lists."
                )
            raw = raw[0]
        candidates = list(raw or [])
        deleted: list[Any] = []
        retained: list[Any] = []
        for candidate in candidates:
            (deleted if self._is_excluded(candidate) else retained).append(candidate)
        selected = retained[:top_k]

        event = {
            "event_index": len(self.events),
            "threshold": similarity_threshold,
            "requested_top_k": top_k,
            "searched_top_k": search_k,
            "all_candidates": [
                _serialize_candidate(candidate, self.example, self.support_judge)
                for candidate in candidates
            ],
            "deleted_candidates": [
                _serialize_candidate(candidate, self.example, self.support_judge)
                for candidate in deleted
            ],
            "retained_candidates": [
                _serialize_candidate(candidate, self.example, self.support_judge)
                for candidate in retained
            ],
            "selected_candidate": (
                _serialize_candidate(selected[0], self.example, self.support_judge)
                if selected
                else None
            ),
        }
        self.events.append(event)

        bounded_out = (
            (
                source_exclusions_are_unbounded
                or entry_exclusion_count > self.max_filter_overfetch
            )
            and len(candidates) == search_k
            and len(selected) < top_k
        )
        if bounded_out:
            raise ExclusionSearchExhaustedError(
                "The exclusion filter exhausted its over-retrieval budget before "
                "finding enough retained candidates. Increase max_filter_overfetch "
                "or use a native FAISS ID selector."
            )
        return selected


_FACT_BLOCK_PATTERN = re.compile(r"<FACT>.*?</FACT>", re.DOTALL)
_SPECIAL_TOKEN_PATTERN = re.compile(r"</?[A-Z_]+>")


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
        _default_support_judge
    )
    answer_extractor: Callable[[str, str], str] = extract_colmlm_answer
    max_filter_overfetch: int = 4096
    release_source: str | None = None

    def __post_init__(self) -> None:
        if self.max_filter_overfetch < 0:
            raise ValueError("max_filter_overfetch cannot be negative.")

    @classmethod
    def from_public_release(
        cls,
        *,
        model_path: str | Path,
        index_path: str | Path,
        db_path: str | Path | None = None,
        source_path: str | Path | None = None,
        use_sqlite_id_mapping: bool = False,
        **loader_kwargs: Any,
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
                raise CoLMLMIntegrationError(
                    "A different `lmlm` package is already imported. Run Co-LMLM "
                    "in its own process/environment to avoid the rel-LMLM namespace "
                    "collision."
                )
            sys.path.insert(0, str(source_src))
            release_source = str(source_root)

        try:
            module = importlib.import_module("lmlm.eval.hf_generate")
            loader = module.load_retriever_generator
        except (ImportError, AttributeError) as exc:
            raise ImportError(
                "The public Co-LMLM release is required. Run this command from its "
                "Python 3.12 environment or pass --colmlm-source-path."
            ) from exc

        generator = loader(
            model_path=str(model_path),
            index_path=Path(index_path),
            db_path=Path(db_path) if db_path is not None else None,
            use_sqlite_id_mapping=use_sqlite_id_mapping,
            **loader_kwargs,
        )
        return cls(generator=generator, release_source=release_source)

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
            if state is DatabaseState.DEL_OFF:
                no_retrieval = getattr(self.generator, "generate_no_retrieval", None)
                if no_retrieval is None:
                    raise CoLMLMIntegrationError(
                        "This Co-LMLM generator has no generate_no_retrieval() method."
                    )
                result = no_retrieval(example.prompt)
            else:
                if original_index is None:
                    raise CoLMLMIntegrationError(
                        "The Co-LMLM generator does not expose its search index."
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
            (event["selected_candidate"] for event in events if event["selected_candidate"]),
            None,
        )
        retrieval_trace = {
            "state": state.value,
            "trace_available": True,
            "trace_complete": True,
            "retrieval_enabled": state is not DatabaseState.DEL_OFF,
            "retrieval_triggered": bool(
                events or num_retrievals or failed_retrievals
            ),
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
            "gen_decoded_tokens": int(
                getattr(result, "gen_decoded_tokens", 0) or 0
            ),
            "release_source": self.release_source,
        }
        return AuditObservation(
            model_output=self.answer_extractor(raw_text, example.prompt),
            retrieval_trace=retrieval_trace,
            generation_metadata=generation_metadata,
        )
