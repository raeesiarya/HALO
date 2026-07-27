from __future__ import annotations

import json
import os
import urllib.request
import zipfile
from pathlib import Path

LAMA_URL = "https://dl.fbaipublicfiles.com/LAMA/data.zip"

# One prompt per fact. Cap on the number of facts (None = all).
LIMIT: int | None = None
# Facts kept per relation, so place_of_birth does not fill the whole slice
# before place_of_death is reached.
MAX_FACTS_PER_RELATION: int | None = 2000

OUTPUT_DIR = Path(__file__).resolve().parent
PROMPTS_PATH = OUTPUT_DIR / "prompts_googlere.jsonl"
LAMA_DIR = OUTPUT_DIR / "lama"

# Google-RE test shard -> (predicate id, cloze template). [X] is the subject,
# [Y] is the masked object; we keep only the prefix that precedes [Y].
# date_of_birth is intentionally excluded: its answer is a bare 4-digit year, so
# answer-mention containment and value-closure become over-broad (every span
# with that year matches), which is a poor fit for the identity analysis.
RELATIONS: dict[str, tuple[str, str]] = {
    "place_of_birth_test.jsonl": ("place_of_birth", "[X] was born in [Y] ."),
    "place_of_death_test.jsonl": ("place_of_death", "[X] died in [Y] ."),
}


def _google_re_dir() -> Path:
    override = os.environ.get("GOOGLE_RE_DIR")
    if override:
        return Path(override)
    gre_dir = LAMA_DIR / "Google_RE"
    if not gre_dir.is_dir():
        LAMA_DIR.mkdir(parents=True, exist_ok=True)
        zip_path = LAMA_DIR / "data.zip"
        print(f"Downloading LAMA data ({LAMA_URL}) ...")
        urllib.request.urlretrieve(LAMA_URL, zip_path)
        with zipfile.ZipFile(zip_path) as archive:
            members = [
                name
                for name in archive.namelist()
                if name.startswith("data/Google_RE/")
            ]
            archive.extractall(LAMA_DIR, members=members)
        (LAMA_DIR / "data" / "Google_RE").rename(gre_dir)
        (LAMA_DIR / "data").rmdir()
        zip_path.unlink()
    return gre_dir


def _prefix(template: str, subject: str) -> str | None:
    """Fill [X] with the subject and return the prefix before the [Y] slot."""
    if "[Y]" not in template:
        return None
    filled = template.replace("[X]", subject)
    return filled.split("[Y]", 1)[0].strip() or None


def main() -> None:
    gre_dir = _google_re_dir()
    prompt_rows: list[dict] = []
    seen_facts: set[str] = set()
    for shard, (predicate_id, template) in RELATIONS.items():
        shard_path = gre_dir / shard
        if not shard_path.is_file():
            print(f"  (skipping missing shard {shard})")
            continue
        rel_facts = 0
        with shard_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                subject = str(record.get("sub_label") or "").strip()
                obj_label = str(record.get("obj_label") or "").strip()
                fact_id = str(record.get("uuid") or "")
                if not subject or not obj_label or not fact_id:
                    continue
                if fact_id in seen_facts:
                    continue
                prefix = _prefix(template, subject)
                if prefix is None:
                    continue
                seen_facts.add(fact_id)
                rel_facts += 1
                prompt_rows.append(
                    {
                        "prompt_id": fact_id,
                        "fact_id": fact_id,
                        "prompt_text": prefix,
                        "gold_object": obj_label,
                        "answer_aliases": [
                            alias
                            for alias in (record.get("obj_aliases") or [])
                            if alias and alias != obj_label
                        ],
                        "subject": subject,
                        "predicate_id": predicate_id,
                    }
                )
                if LIMIT is not None and len(prompt_rows) >= LIMIT:
                    break
                if (
                    MAX_FACTS_PER_RELATION is not None
                    and rel_facts >= MAX_FACTS_PER_RELATION
                ):
                    break
        if LIMIT is not None and len(prompt_rows) >= LIMIT:
            break

    with PROMPTS_PATH.open("w", encoding="utf-8") as handle:
        for row in prompt_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(prompt_rows)} audit prompts to {PROMPTS_PATH}")


if __name__ == "__main__":
    main()
