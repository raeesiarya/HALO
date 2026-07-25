from __future__ import annotations

import json
import os
import random
import urllib.request
import zipfile
from pathlib import Path

LAMA_URL = "https://dl.fbaipublicfiles.com/LAMA/data.zip"

# One prompt per fact (uuid). A seeded shuffle picks a stable slice.
LIMIT: int | None = 1500
SEED = 0
# The prefix must be long enough to carry context and must mention the
# subject, so the model knows which entity the sentence is about.
MIN_PREFIX_CHARS = 30

OUTPUT_DIR = Path(__file__).resolve().parent
PROMPTS_PATH = OUTPUT_DIR / "prompts_trex.jsonl"
LAMA_DIR = OUTPUT_DIR / "lama"
MASK = "[MASK]"


def _trex_dir() -> Path:
    override = os.environ.get("TREX_DIR")
    if override:
        return Path(override)
    trex_dir = LAMA_DIR / "TREx"
    if not trex_dir.is_dir():
        LAMA_DIR.mkdir(parents=True, exist_ok=True)
        zip_path = LAMA_DIR / "data.zip"
        print(f"Downloading LAMA data ({LAMA_URL}) ...")
        urllib.request.urlretrieve(LAMA_URL, zip_path)
        with zipfile.ZipFile(zip_path) as archive:
            members = [
                name for name in archive.namelist() if name.startswith("data/TREx/")
            ]
            archive.extractall(LAMA_DIR, members=members)
        (LAMA_DIR / "data" / "TREx").rename(trex_dir)
        (LAMA_DIR / "data").rmdir()
        zip_path.unlink()
    return trex_dir


def _prompt_prefix(record: dict) -> tuple[str, str] | None:
    """Best (prefix, obj_surface) among the fact's evidences, or None."""
    best: tuple[str, str] | None = None
    for evidence in record.get("evidences") or []:
        sentence = evidence.get("masked_sentence") or ""
        subject = (evidence.get("sub_surface") or "").strip()
        obj_surface = (evidence.get("obj_surface") or "").strip()
        if MASK not in sentence or not subject or not obj_surface:
            continue
        prefix = sentence.split(MASK, 1)[0].strip()
        if len(prefix) < MIN_PREFIX_CHARS or subject not in prefix:
            continue
        # Prefer the shortest qualifying prefix: it keeps the mask close to
        # the subject mention and leaves fewer distracting clauses.
        if best is None or len(prefix) < len(best[0]):
            best = (prefix, obj_surface)
    return best


def main() -> None:
    trex_dir = _trex_dir()
    records: list[dict] = []
    for shard in sorted(trex_dir.glob("*.jsonl")):
        with shard.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))

    random.Random(SEED).shuffle(records)

    prompt_rows: list[dict] = []
    seen_facts: set[str] = set()
    for record in records:
        fact_id = str(record.get("uuid") or "")
        if not fact_id or fact_id in seen_facts:
            continue
        selected = _prompt_prefix(record)
        if selected is None:
            continue
        prefix, obj_surface = selected
        obj_label = str(record.get("obj_label") or "").strip()
        aliases = [obj_label] if obj_label and obj_label != obj_surface else []
        seen_facts.add(fact_id)
        prompt_rows.append(
            {
                "prompt_id": fact_id,
                "fact_id": fact_id,
                "prompt_text": prefix,
                "gold_object": obj_surface,
                "answer_aliases": aliases,
                "subject": record.get("sub_label"),
                "predicate_id": record.get("predicate_id"),
            }
        )
        if LIMIT is not None and len(prompt_rows) >= LIMIT:
            break

    with PROMPTS_PATH.open("w", encoding="utf-8") as handle:
        for row in prompt_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(prompt_rows)} audit prompts to {PROMPTS_PATH}")


if __name__ == "__main__":
    main()
