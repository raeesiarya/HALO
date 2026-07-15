import dataclasses
import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from lmlm_audit.backend import (
    AuditBackend,
    audit_example,
    validate_intervention_results,
)
from lmlm_audit.examples import AuditExample, DeletionManifest
from lmlm_audit.rel_lmlm.backend import RelLMLMAuditBackend
from lmlm_audit.states import DatabaseState


def load_prompts(prompts_path: Path) -> list[dict[str, Any]]:
    with prompts_path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_backend_prompt_audit(
    backend: AuditBackend,
    prompt_row: dict[str, Any],
    state: DatabaseState,
    max_new_tokens: int = 12,
) -> dict[str, Any]:
    return audit_example(
        backend=backend,
        example=AuditExample.from_prompt_row(prompt_row),
        state=state,
        max_new_tokens=max_new_tokens,
    )


def run_prompt_audit(
    base_db_manager: Any,
    model: Any,
    tokenizer: Any,
    prompt_row: dict[str, Any],
    state: DatabaseState,
    max_new_tokens: int = 12,
) -> dict[str, Any]:
    backend = RelLMLMAuditBackend(
        base_db_manager=base_db_manager,
        model=model,
        tokenizer=tokenizer,
    )
    return run_backend_prompt_audit(
        backend=backend,
        prompt_row=prompt_row,
        state=state,
        max_new_tokens=max_new_tokens,
    )


def run_backend_audit(
    prompt_path: Path,
    backend: AuditBackend,
    states: list[DatabaseState],
    max_new_tokens: int = 12,
    limit: int | None = None,
    bootstrap_oracle_from_full: bool = False,
) -> list[dict[str, Any]]:
    if not states:
        raise ValueError("At least one audit state is required.")
    if len(states) != len(set(states)):
        raise ValueError("Audit states must not contain duplicates.")

    prompts = load_prompts(prompt_path)
    if limit is not None:
        prompts = prompts[:limit]

    results: list[dict[str, Any]] = []
    for prompt in tqdm(
        prompts,
        desc=f"Auditing {prompt_path.stem}",
        unit="prompt",
    ):
        example = AuditExample.from_prompt_row(prompt)
        prompt_results: list[dict[str, Any]] = []
        remaining_states = list(states)

        if bootstrap_oracle_from_full and example.deletion_manifest.is_empty:
            if not remaining_states or remaining_states[0] is not DatabaseState.FULL:
                raise ValueError(
                    "Oracle bootstrapping requires FULL to be the first requested state."
                )
            full_result = audit_example(
                backend=backend,
                example=example,
                state=DatabaseState.FULL,
                max_new_tokens=max_new_tokens,
            )
            selected = (full_result.get("retrieval_trace") or {}).get(
                "selected_candidate"
            ) or {}
            entry_id = selected.get("entry_id")
            if not entry_id:
                raise ValueError(
                    "FULL produced no selected entry ID; cannot bootstrap an oracle manifest."
                )
            if selected.get("supports_target") is not True:
                raise ValueError(
                    "FULL's selected entry did not pass the configured target-support "
                    "judge; supply a reviewed deletion manifest manually."
                )
            manifest = DeletionManifest(
                entry_ids=(str(entry_id),),
                strategy="oracle-from-full",
                metadata={"bootstrap": "FULL.selected_candidate"},
            )
            example = dataclasses.replace(example, deletion_manifest=manifest)
            full_result["deletion_manifest"] = manifest.as_dict()
            full_result["retrieval_trace"]["deletion_manifest_id"] = (
                manifest.manifest_id
            )
            prompt_results.append(full_result)
            remaining_states = remaining_states[1:]

        for state in remaining_states:
            prompt_results.append(
                audit_example(
                    backend=backend,
                    example=example,
                    state=state,
                    max_new_tokens=max_new_tokens,
                )
            )
        validate_intervention_results(prompt_results, expected_states=states)
        results.extend(prompt_results)

    return results


def run_audit(
    prompt_path: Path,
    base_db_manager: Any,
    model: Any,
    tokenizer: Any,
    states: list[DatabaseState],
    max_new_tokens: int = 12,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    backend = RelLMLMAuditBackend(
        base_db_manager=base_db_manager,
        model=model,
        tokenizer=tokenizer,
    )
    return run_backend_audit(
        prompt_path=prompt_path,
        backend=backend,
        states=states,
        max_new_tokens=max_new_tokens,
        limit=limit,
    )
