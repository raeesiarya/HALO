"""Build ZsRE audit prompts (data/prompts_zsre.jsonl).

ZsRE (zero-shot relation extraction) is the slot-filling QA set used across the
model-editing literature. We use the MEND evaluation release. Each record has a
question (`src`), a reworded question (`rephrase`), and gold `answers`. We emit
one fact-level row per record: the question wrapped with the same answer stub as
the PopQA prompts (prepare_popqa_audit.py), following Co-LMLM's own QA eval
(lmlm/eval/append_answer_stub.py).

The rephrase is NOT written to the main file: the audit keys every row by its own
prompt_id with no grouping, so a second row per fact would inflate the cohort and
correlate the samples. Set EMIT_VARIANTS=1 to dump rephrase rows to
data/prompts_zsre_variants.jsonl for the (not-yet-implemented) paraphrase
analysis; that file must not be fed to the standard audit as-is.

Note: question-format prompts are more prone to Co-LMLM's <FACT>-timing problem
(paper Table 5 / Sec 4.7) than slot-filling, so expect lower FULL-retrieval
cohort coverage here than on T-REx.

The eval JSON (~20 MB) is downloaded to data/zsre_mend_eval.json on first run;
set ZSRE_JSON to point at an existing file instead.
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

ZSRE_URL = "https://rome.baulab.info/data/dsets/zsre_mend_eval.json"

# Cap on the number of facts (None = all): one fact-level row each.
LIMIT: int | None = None
# Also write rephrase rows to a separate *_variants.jsonl.
EMIT_VARIANTS = os.environ.get("EMIT_VARIANTS", "") not in ("", "0", "false")
ANSWER_STUB = "\nThe answer is"

OUTPUT_DIR = Path(__file__).resolve().parent
PROMPTS_PATH = OUTPUT_DIR / "prompts_zsre.jsonl"
VARIANTS_PATH = OUTPUT_DIR / "prompts_zsre_variants.jsonl"
ZSRE_PATH = OUTPUT_DIR / "zsre_mend_eval.json"


def _zsre_records() -> list[dict]:
    override = os.environ.get("ZSRE_JSON")
    path = Path(override) if override else ZSRE_PATH
    if not path.is_file():
        print(f"Downloading ZsRE ({ZSRE_URL}) ...")
        urllib.request.urlretrieve(ZSRE_URL, path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _question_row(
    prompt_id: str,
    fact_id: str,
    question: str,
    answers: list[str],
    subject: str,
    variant: str | None,
) -> dict:
    row = {
        "prompt_id": prompt_id,
        "fact_id": fact_id,
        "prompt_text": f"{question.strip()}{ANSWER_STUB}",
        "gold_object": answers[0],
        "answer_aliases": answers[1:],
        "subject": subject,
    }
    if variant is not None:
        row["variant"] = variant
    return row


def main() -> None:
    records = _zsre_records()

    prompt_rows: list[dict] = []
    variant_rows: list[dict] = []
    facts = 0
    for index, record in enumerate(records):
        question = str(record.get("src") or "").strip()
        answers = [
            str(a).strip() for a in (record.get("answers") or []) if str(a).strip()
        ]
        subject = str(record.get("subject") or "").strip()
        if not question or not answers:
            continue
        fact_id = f"zsre-{index}"

        # One fact-level row per fact (prompt_id == fact_id), like T-REx.
        prompt_rows.append(
            _question_row(fact_id, fact_id, question, answers, subject, None)
        )

        if EMIT_VARIANTS:
            rephrase = str(record.get("rephrase") or "").strip()
            if rephrase:
                variant_rows.append(
                    _question_row(
                        f"{fact_id}-para",
                        fact_id,
                        rephrase,
                        answers,
                        subject,
                        "paraphrase",
                    )
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
            f"Wrote {len(variant_rows)} rephrase rows to {VARIANTS_PATH} "
            "(not for the standard audit)"
        )


if __name__ == "__main__":
    main()
