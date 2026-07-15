import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping

from lmlm_audit.equivalence import prompt_row_aliases


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values = (value,)
    elif isinstance(value, (list, tuple, set)):
        values = tuple(str(item) for item in value if item is not None)
    else:
        values = (str(value),)
    return tuple(sorted({value.strip() for value in values if value.strip()}))


_MISSING = object()


def _first_present(*candidates: tuple[Mapping[str, Any], str]) -> Any:
    for mapping, key in candidates:
        value = mapping.get(key, _MISSING)
        if value is not _MISSING:
            return value
    return None


@dataclass(frozen=True)
class DeletionManifest:
    entry_ids: tuple[str, ...] = ()
    source_ids: tuple[str, ...] = ()
    strategy: str = "oracle-entry"
    metadata: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "entry_ids", _string_tuple(self.entry_ids))
        object.__setattr__(self, "source_ids", _string_tuple(self.source_ids))
        object.__setattr__(self, "strategy", str(self.strategy))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_prompt_row(cls, prompt_row: Mapping[str, Any]) -> "DeletionManifest":
        embedded = prompt_row.get("deletion_manifest")
        manifest_row = dict(embedded) if isinstance(embedded, Mapping) else {}

        entry_ids = _first_present(
            (manifest_row, "entry_ids"),
            (manifest_row, "deletion_entry_ids"),
            (prompt_row, "deletion_entry_ids"),
            (prompt_row, "oracle_entry_ids"),
            (prompt_row, "entry_ids"),
        )
        source_ids = _first_present(
            (manifest_row, "source_ids"),
            (prompt_row, "source_ids"),
        )
        strategy = str(
            manifest_row.get("strategy")
            or prompt_row.get("deletion_strategy")
            or "oracle-entry"
        )
        metadata = manifest_row.get("metadata")
        return cls(
            entry_ids=_string_tuple(entry_ids),
            source_ids=_string_tuple(source_ids),
            strategy=strategy,
            metadata=dict(metadata) if isinstance(metadata, Mapping) else {},
        )

    @property
    def manifest_id(self) -> str:
        payload = json.dumps(self.as_dict(include_id=False), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    @property
    def is_empty(self) -> bool:
        return not self.entry_ids and not self.source_ids

    def as_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result = {
            "entry_ids": list(self.entry_ids),
            "source_ids": list(self.source_ids),
            "strategy": self.strategy,
            "metadata": dict(self.metadata),
        }
        if include_id:
            result["manifest_id"] = self.manifest_id
        return result


@dataclass(frozen=True)
class AuditExample:
    prompt: str
    ground_truth: str
    fact_id: str | int | None = None
    prompt_id: str | int | None = None
    object_aliases: tuple[str, ...] = ()
    subject: str | None = None
    subject_aliases: tuple[str, ...] = ()
    relation: str | None = None
    relation_aliases: tuple[str, ...] = ()
    deletion_manifest: DeletionManifest = field(default_factory=DeletionManifest)
    source_row: Mapping[str, Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    @classmethod
    def from_prompt_row(cls, prompt_row: Mapping[str, Any]) -> "AuditExample":
        if "prompt_text" not in prompt_row:
            raise ValueError("Audit prompt row is missing required field 'prompt_text'.")
        if "gold_object" not in prompt_row:
            raise ValueError("Audit prompt row is missing required field 'gold_object'.")

        row = dict(prompt_row)
        return cls(
            fact_id=row.get("fact_id"),
            prompt_id=row.get("prompt_id"),
            prompt=str(row["prompt_text"]),
            ground_truth=str(row["gold_object"]),
            object_aliases=prompt_row_aliases(row, "object"),
            subject=(str(row["subject"]) if row.get("subject") is not None else None),
            subject_aliases=prompt_row_aliases(row, "subject"),
            relation=(
                str(row["relation"]) if row.get("relation") is not None else None
            ),
            relation_aliases=prompt_row_aliases(row, "relation"),
            deletion_manifest=DeletionManifest.from_prompt_row(row),
            source_row=row,
        )
