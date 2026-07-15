import re
from typing import Any


TOKEN_PATTERN = re.compile(r"\d+\.\d+|\w+(?:[-']\w+)*", re.UNICODE)

PROMPT_ROW_ALIAS_KEYS = {
    "subject": ("subject_aliases",),
    "relation": ("relation_aliases",),
    "object": ("object_aliases", "gold_object_aliases", "answer_aliases"),
}


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(str(text).casefold())


def normalize_text(text: str) -> str:
    return " ".join(tokenize(text))


def _flatten_alias_values(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, str):
        return [value]

    if isinstance(value, (list, tuple, set)):
        values: list[str] = []
        for item in value:
            values.extend(_flatten_alias_values(item))
        return values

    return [str(value)]


def _unique_preserving_order(values: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    unique_values: list[str] = []
    for value in values:
        normalized = normalize_text(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique_values.append(value)
    return tuple(unique_values)


def build_alias_set(canonical: str, aliases: Any = None) -> tuple[str, ...]:
    return _unique_preserving_order([canonical, *_flatten_alias_values(aliases)])


def prompt_row_aliases(prompt_row: dict[str, Any], field_name: str) -> tuple[str, ...]:
    return _unique_preserving_order(
        [
            alias
            for key in PROMPT_ROW_ALIAS_KEYS.get(field_name, ())
            for alias in _flatten_alias_values(prompt_row.get(key))
        ]
    )


def values_equivalent(
    left: str,
    right: str,
    left_aliases: Any = None,
    right_aliases: Any = None,
) -> bool:
    left_values = build_alias_set(left, left_aliases)
    right_values = build_alias_set(right, right_aliases)

    left_normalized = {normalize_text(value) for value in left_values}
    right_normalized = {normalize_text(value) for value in right_values}
    return bool(left_normalized & right_normalized)
