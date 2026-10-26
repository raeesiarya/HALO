from __future__ import annotations

import json
import os
import random
import urllib.request
import zipfile
from pathlib import Path

LAMA_URL = "https://dl.fbaipublicfiles.com/LAMA/data.zip"

# One prompt per fact (uuid). A seeded shuffle picks a stable slice.
LIMIT: int | None = None
SEED = 0
# The prefix must be long enough to carry context and must mention the
# subject, so the model knows which entity the sentence is about.
MIN_PREFIX_CHARS = 30
# Paraphrase variants (data/prompts_trex_variants.jsonl) for the
# paraphrase-robustness phase: the fact's *other* LAMA evidence sentences
# first (natural Wikipedia paraphrases), then ParaRel relation templates.
# The main prompt file is unaffected; VARIANTS_ONLY=1 writes only the
# variants, leaving an existing prompt file (and its fingerprint) untouched.
EMIT_VARIANTS = os.environ.get("EMIT_VARIANTS", "1") not in ("", "0", "false")
VARIANTS_ONLY = os.environ.get("VARIANTS_ONLY", "") not in ("", "0", "false")
MAX_VARIANT_CANDIDATES = 4

OUTPUT_DIR = Path(__file__).resolve().parent
PROMPTS_PATH = OUTPUT_DIR / "prompts_trex.jsonl"
VARIANTS_PATH = OUTPUT_DIR / "prompts_trex_variants.jsonl"
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


def _qualifying_prefixes(record: dict) -> list[tuple[str, str]]:
    """Every (prefix, obj_surface) evidence that qualifies as a prompt."""
    qualifying: list[tuple[str, str]] = []
    for evidence in record.get("evidences") or []:
        sentence = evidence.get("masked_sentence") or ""
        subject = (evidence.get("sub_surface") or "").strip()
        obj_surface = (evidence.get("obj_surface") or "").strip()
        if MASK not in sentence or not subject or not obj_surface:
            continue
        prefix = sentence.split(MASK, 1)[0].strip()
        if len(prefix) < MIN_PREFIX_CHARS or subject not in prefix:
            continue
        qualifying.append((prefix, obj_surface))
    return qualifying


def _prompt_prefix(record: dict) -> tuple[str, str] | None:
    """Best (prefix, obj_surface) among the fact's evidences, or None."""
    best: tuple[str, str] | None = None
    for prefix, obj_surface in _qualifying_prefixes(record):
        # Prefer the shortest qualifying prefix: it keeps the mask close to
        # the subject mention and leaves fewer distracting clauses.
        if best is None or len(prefix) < len(best[0]):
            best = (prefix, obj_surface)
    return best


def _variant_rows(record: dict, row: dict) -> list[dict]:
    """Paraphrase candidates for one selected fact (the builder filters)."""
    from _pararel import paraphrase_prefixes

    canonical = row["prompt_text"]
    gold = row["gold_object"]
    candidates: list[tuple[str, str, list[str]]] = []
    seen = {canonical}
    # Other evidence sentences, shortest first (the canonical rule's order).
    for prefix, obj_surface in sorted(
        _qualifying_prefixes(record), key=lambda item: (len(item[0]), item[0])
    ):
        if prefix in seen:
            continue
        seen.add(prefix)
        extra = [obj_surface] if obj_surface != gold else []
        candidates.append((prefix, "lama-evidence", extra))
    subject = str(record.get("sub_label") or "").strip()
    relation = str(record.get("predicate_id") or "")
    if subject and relation:
        for prefix in paraphrase_prefixes(
            relation, subject, fact_id=row["fact_id"], limit=MAX_VARIANT_CANDIDATES
        ):
            if prefix not in seen:
                seen.add(prefix)
                candidates.append((prefix, "pararel", []))
    return [
        {
            "prompt_id": f"{row['fact_id']}-para{index}",
            "fact_id": row["fact_id"],
            "prompt_text": text,
            "gold_object": gold,
            "variant": "paraphrase",
            "variant_source": source,
            "extra_answer_aliases": extra,
        }
        for index, (text, source, extra) in enumerate(
            candidates[:MAX_VARIANT_CANDIDATES]
        )
    ]


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
    variant_rows: list[dict] = []
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
        if EMIT_VARIANTS or VARIANTS_ONLY:
            variant_rows.extend(_variant_rows(record, prompt_rows[-1]))
        if LIMIT is not None and len(prompt_rows) >= LIMIT:
            break

    if not VARIANTS_ONLY:
        with PROMPTS_PATH.open("w", encoding="utf-8") as handle:
            for row in prompt_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Wrote {len(prompt_rows)} audit prompts to {PROMPTS_PATH}")

    if EMIT_VARIANTS or VARIANTS_ONLY:
        with VARIANTS_PATH.open("w", encoding="utf-8") as handle:
            for row in variant_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(
            f"Wrote {len(variant_rows)} paraphrase candidates to {VARIANTS_PATH} "
            "(not for the standard audit)"
        )


if __name__ == "__main__":
    main()
