from __future__ import annotations

import ast
import json
import os
from pathlib import Path

POPQA_DATASET = "akariasai/PopQA"
POPQA_SPLIT = "test"
# Keep entities whose subject popularity (PopQA `s_pop`, Wikipedia page views)
# is at least this. Higher values retain only well-known entities, which the
# wiki index is more likely to hold — the oracle bootstrap needs the retrieved
# entry to literally contain the answer. Set to None to keep every entity.
MIN_POPULARITY: float | None = 10_000.0
# Cap on the number of facts (None = all). A seeded shuffle picks a stable slice.
LIMIT: int | None = None
SEED = 0

OUTPUT_DIR = Path(__file__).resolve().parent
PROMPTS_PATH = OUTPUT_DIR / "prompts.jsonl"
VARIANTS_PATH = OUTPUT_DIR / "prompts_popqa_variants.jsonl"
# Paraphrase variants for the paraphrase-robustness phase. PopQA builds each
# question from one template per relation (`prop`), so the variants are
# hand-written rewordings of that template, filled with the same subject
# surface form. VARIANTS_ONLY=1 leaves an existing prompt file (and its
# fingerprint) untouched.
EMIT_VARIANTS = os.environ.get("EMIT_VARIANTS", "1") not in ("", "0", "false")
VARIANTS_ONLY = os.environ.get("VARIANTS_ONLY", "") not in ("", "0", "false")
ANSWER_STUB = "\nThe answer is"
PARAPHRASE_TEMPLATES: dict[str, tuple[str, ...]] = {
    "occupation": (
        "What does {subj} do for a living?",
        "What is the profession of {subj}?",
        "What kind of work does {subj} do?",
    ),
    "place of birth": (
        "Where was {subj} born?",
        "What is the birthplace of {subj}?",
        "In which city was {subj} born?",
    ),
    "genre": (
        "What is the genre of {subj}?",
        "Which genre does {subj} belong to?",
        "What kind of genre is {subj} associated with?",
    ),
    "father": (
        "Who was {subj}'s father?",
        "Who is the dad of {subj}?",
        "Name the father of {subj}.",
    ),
    "country": (
        "Which country is {subj} located in?",
        "{subj} is in which country?",
        "What country is {subj} part of?",
    ),
    "producer": (
        "Who produced {subj}?",
        "Who is credited as the producer of {subj}?",
        "Which person produced {subj}?",
    ),
    "director": (
        "Who directed {subj}?",
        "Which filmmaker directed {subj}?",
        "Who is credited as the director of {subj}?",
    ),
    "capital of": (
        "{subj} is the capital of what?",
        "Of which place is {subj} the capital?",
        "Which country or region has {subj} as its capital?",
    ),
    "screenwriter": (
        "Who wrote the screenplay for {subj}?",
        "Who wrote the script of {subj}?",
        "Who is credited as the screenwriter of {subj}?",
    ),
    "composer": (
        "Who composed {subj}?",
        "Who wrote the music for {subj}?",
        "Which composer created {subj}?",
    ),
    "color": (
        "What is the color of {subj}?",
        "Which color is {subj}?",
        "What colour is {subj}?",
    ),
    "religion": (
        "Which religion does {subj} follow?",
        "What faith is {subj} associated with?",
        "What religion is {subj} affiliated with?",
    ),
    "sport": (
        "Which sport is {subj} associated with?",
        "What sport is played by {subj}?",
        "In which sport does {subj} compete?",
    ),
    "author": (
        "Who wrote {subj}?",
        "Who is the writer of {subj}?",
        "Which author wrote {subj}?",
    ),
    "mother": (
        "Who was {subj}'s mother?",
        "Who is the mom of {subj}?",
        "Name the mother of {subj}.",
    ),
    "capital": (
        "Which city is the capital of {subj}?",
        "What city serves as the capital of {subj}?",
        "Where is the capital of {subj}?",
    ),
}


def _acceptable_answers(possible_answers: object) -> list[str]:
    """PopQA `possible_answers` (a list, sometimes stringified), deduplicated
    and order-preserved."""
    if isinstance(possible_answers, str):
        possible_answers = ast.literal_eval(possible_answers)
    seen: set[str] = set()
    answers: list[str] = []
    for answer in possible_answers or []:
        text = str(answer).strip()
        if text and text not in seen:
            seen.add(text)
            answers.append(text)
    return answers


def main() -> None:
    from datasets import load_dataset

    dataset = load_dataset(POPQA_DATASET, split=POPQA_SPLIT)
    if MIN_POPULARITY is not None:
        dataset = dataset.filter(
            lambda row: row["s_pop"] is not None and row["s_pop"] >= MIN_POPULARITY
        )
    dataset = dataset.shuffle(seed=SEED)
    if LIMIT is not None:
        dataset = dataset.select(range(min(LIMIT, len(dataset))))

    prompt_rows: list[dict] = []
    variant_rows: list[dict] = []
    unknown_props: set[str] = set()
    for record in dataset:
        answers = _acceptable_answers(record["possible_answers"])
        if not answers:
            continue
        # PopQA's `s_wiki_title` is the exact Wikipedia article title of the
        # subject: it is the provenance source the NULLs audit resolves sink
        # masks on (models.nulls_wiki.titles), and it gives the analysis a
        # real (subject, relation, answer) proposition key. `subj` is the
        # surface form, kept as an alias when it differs.
        subject = str(record.get("s_wiki_title") or record.get("subj") or "").strip() or None
        surface = str(record.get("subj") or "").strip()
        prompt_rows.append(
            {
                "prompt_id": record.get("id"),
                "fact_id": record.get("id"),
                # Question + answer stub, following Co-LMLM's own QA evals
                # (lmlm/eval/append_answer_stub.py): a base LM reads the cue
                # and emits the answer instead of continuing prose.
                "prompt_text": f"{record['question']}\nThe answer is",
                "gold_object": answers[0],
                "answer_aliases": answers[1:],
                "subject": subject,
                "subject_aliases": [surface] if surface and surface != subject else [],
                "predicate_id": record.get("prop"),
            }
        )
        if EMIT_VARIANTS or VARIANTS_ONLY:
            prop = str(record.get("prop") or "")
            templates = PARAPHRASE_TEMPLATES.get(prop)
            if templates is None:
                unknown_props.add(prop)
                continue
            for index, template in enumerate(templates):
                variant_rows.append(
                    {
                        "prompt_id": f"{record.get('id')}-para{index}",
                        "fact_id": record.get("id"),
                        "prompt_text": template.format(subj=surface or subject)
                        + ANSWER_STUB,
                        "gold_object": answers[0],
                        "variant": "paraphrase",
                        "variant_source": "template",
                    }
                )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
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
        if unknown_props:
            print(
                "  no paraphrase templates for relations: "
                + ", ".join(sorted(unknown_props))
            )


if __name__ == "__main__":
    main()
