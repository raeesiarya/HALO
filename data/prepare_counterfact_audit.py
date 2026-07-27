"""Build CounterFact audit prompts (data/prompts_counterfact.jsonl).

CounterFact (Meng et al., ROME/MEMIT) ships one record per fact. We keep the
*true* answer (`target_true`, never the counterfactual `target_new`) and emit
the requested-rewrite prompt, filled on the subject, as a single fact-level row.
That plugs straight into the fact-level three-state audit, one row per fact,
exactly like the T-REx set.

The paraphrase and neighborhood prompts are NOT written to the main file: the
audit keys every row by its own prompt_id and has no notion of grouping rows
under a shared fact, so extra rows per fact would silently inflate the cohort
and correlate the samples (breaking the per-fact Wilson/McNemar assumptions).
Set EMIT_VARIANTS=1 to additionally dump those rows to
data/prompts_counterfact_variants.jsonl for the (not-yet-implemented)
identity / paraphrase-robustness analysis; that file must not be fed to the
standard audit as-is.

The canonical release (~40 MB JSON) is downloaded to data/counterfact.json on
first run; set COUNTERFACT_JSON to point at an existing file instead.
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

COUNTERFACT_URL = "https://rome.baulab.info/data/dsets/counterfact.json"

# Cap on the number of facts (None = all): one fact-level row each.
LIMIT: int | None = None
# Also write paraphrase + neighborhood rows to a separate *_variants.jsonl.
EMIT_VARIANTS = os.environ.get("EMIT_VARIANTS", "") not in ("", "0", "false")

OUTPUT_DIR = Path(__file__).resolve().parent
PROMPTS_PATH = OUTPUT_DIR / "prompts_counterfact.jsonl"
VARIANTS_PATH = OUTPUT_DIR / "prompts_counterfact_variants.jsonl"
COUNTERFACT_PATH = OUTPUT_DIR / "counterfact.json"


def _counterfact_records() -> list[dict]:
    override = os.environ.get("COUNTERFACT_JSON")
    path = Path(override) if override else COUNTERFACT_PATH
    if not path.is_file():
        print(f"Downloading CounterFact ({COUNTERFACT_URL}) ...")
        urllib.request.urlretrieve(COUNTERFACT_URL, path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    records = _counterfact_records()

    prompt_rows: list[dict] = []
    variant_rows: list[dict] = []
    facts = 0
    for record in records:
        rewrite = record.get("requested_rewrite") or {}
        template = rewrite.get("prompt") or ""
        subject = str(rewrite.get("subject") or "").strip()
        gold = str((rewrite.get("target_true") or {}).get("str") or "").strip()
        case_id = record.get("case_id")
        if "{}" not in template or not subject or not gold or case_id is None:
            continue
        fact_id = f"cf-{case_id}"
        relation_id = rewrite.get("relation_id")

        # One fact-level row per fact (prompt_id == fact_id), like T-REx.
        prompt_rows.append(
            {
                "prompt_id": fact_id,
                "fact_id": fact_id,
                "prompt_text": template.format(subject).strip(),
                "gold_object": gold,
                "answer_aliases": [],
                "subject": subject,
                "predicate_id": relation_id,
            }
        )

        if EMIT_VARIANTS:
            for i, paraphrase in enumerate(record.get("paraphrase_prompts") or []):
                text = str(paraphrase).strip()
                if not text:
                    continue
                variant_rows.append(
                    {
                        "prompt_id": f"{fact_id}-para{i}",
                        "fact_id": fact_id,
                        "prompt_text": text,
                        "gold_object": gold,
                        "answer_aliases": [],
                        "subject": subject,
                        "predicate_id": relation_id,
                        "variant": "paraphrase",
                    }
                )
            for i, neighbor in enumerate(record.get("neighborhood_prompts") or []):
                text = str(neighbor).strip()
                if not text:
                    continue
                variant_rows.append(
                    {
                        "prompt_id": f"{fact_id}-nbr{i}",
                        "fact_id": f"{fact_id}-nbr{i}",
                        "prompt_text": text,
                        # Neighborhood prompts complete to the same true answer,
                        # but about a *different* subject (same-answer negative).
                        "gold_object": gold,
                        "answer_aliases": [],
                        "predicate_id": relation_id,
                        "variant": "neighborhood",
                        "shared_answer_group": fact_id,
                    }
                )

        facts += 1
        if LIMIT is not None and facts >= LIMIT:
            break

    with PROMPTS_PATH.open("w", encoding="utf-8") as handle:
        for row in prompt_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(prompt_rows)} fact-level prompts to {PROMPTS_PATH}")

    if EMIT_VARIANTS:
        with VARIANTS_PATH.open("w", encoding="utf-8") as handle:
            for row in variant_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(
            f"Wrote {len(variant_rows)} paraphrase/neighborhood rows to "
            f"{VARIANTS_PATH} (not for the standard audit)"
        )


if __name__ == "__main__":
    main()
