from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from lmlm_audit.core.backend import AuditObservation
from lmlm_audit.colmlm.answers import _default_support_judge, extract_colmlm_answer
from lmlm_audit.colmlm.errors import CoLMLMIntegrationError
from lmlm_audit.colmlm.index_filter import _FilteringSearchIndex
from lmlm_audit.core.examples import AuditExample
from lmlm_audit.core.states import DatabaseState


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
