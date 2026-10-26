"""ParaRel relation paraphrase templates (Elazar et al., 2021, TACL).

ParaRel gives human-written paraphrase patterns ("[X] was born in [Y].",
"[X] is originally from [Y].", ...) for 38 T-REx relations, including the
Google-RE relations P19 (place of birth) and P20 (place of death). The
prepare scripts use them as paraphrase variants of slot-filling prompts.

Patterns are fetched from a pinned commit and cached under data/pararel/,
so a rebuild is reproducible. Set PARAREL_DIR to use a local copy.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

PARAREL_COMMIT = "7cdb6a8b3c54b9d8a3dee92477173dc94ec2da84"
PARAREL_URL = (
    "https://raw.githubusercontent.com/yanaiela/pararel/"
    f"{PARAREL_COMMIT}/data/pattern_data/graphs_json/{{relation}}.jsonl"
)
CACHE_DIR = Path(__file__).resolve().parent / "pararel"


def _pattern_path(relation: str) -> Path:
    override = os.environ.get("PARAREL_DIR")
    return (Path(override) if override else CACHE_DIR) / f"{relation}.jsonl"


def load_patterns(relation: str) -> list[str]:
    """ParaRel patterns for one relation id, [] when ParaRel lacks it."""
    path = _pattern_path(relation)
    if not path.is_file():
        if os.environ.get("PARAREL_DIR"):
            return []
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(PARAREL_URL.format(relation=relation), path)
        except urllib.error.HTTPError as error:
            if error.code == 404:
                path.write_text("", encoding="utf-8")  # cache the miss
                return []
            raise
    patterns = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                pattern = str(json.loads(line).get("pattern") or "").strip()
                if pattern:
                    patterns.append(pattern)
    return patterns


def prefix_patterns(relation: str) -> list[str]:
    """Patterns usable as a left-to-right prompt: [X] before a final [Y]."""
    usable = []
    for pattern in load_patterns(relation):
        if pattern.count("[X]") != 1 or pattern.count("[Y]") != 1:
            continue
        head, tail = pattern.split("[Y]", 1)
        if "[X]" not in head or tail.strip(" .") != "":
            continue
        usable.append(pattern)
    return usable


def fill_prefix(pattern: str, subject: str) -> str:
    return pattern.split("[Y]", 1)[0].replace("[X]", subject).strip()


def _normalized(text: str) -> str:
    return " ".join(text.replace(" .", ".").split()).rstrip(" .").casefold()


def paraphrase_prefixes(
    relation: str,
    subject: str,
    *,
    fact_id: str,
    exclude: tuple[str, ...] = (),
    limit: int = 4,
) -> list[str]:
    """Up to ``limit`` filled ParaRel prefixes for one fact.

    Patterns equal to an excluded template (e.g. the canonical LAMA
    template) are dropped. The order is a per-fact deterministic shuffle so
    the variants spread over a relation's patterns instead of every fact
    getting the same two.
    """
    excluded = {_normalized(item) for item in exclude}
    candidates = [
        pattern for pattern in prefix_patterns(relation) if _normalized(pattern) not in excluded
    ]
    candidates.sort(
        key=lambda pattern: hashlib.sha256(f"{fact_id}|{pattern}".encode()).hexdigest()
    )
    prefixes: list[str] = []
    for pattern in candidates:
        prefix = fill_prefix(pattern, subject)
        if prefix and prefix not in prefixes:
            prefixes.append(prefix)
        if len(prefixes) >= limit:
            break
    return prefixes
