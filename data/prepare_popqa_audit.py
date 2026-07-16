from __future__ import annotations

import ast
import json
from pathlib import Path

POPQA_DATASET = "akariasai/PopQA"
POPQA_SPLIT = "test"
# Keep entities whose subject popularity (PopQA `s_pop`) is below this. Lower
# values retain only well-known entities, which the wiki index is more likely
# to hold. Set to None to keep every entity.
MAX_POPULARITY: float | None = 100.0
# Cap on the number of facts (None = all). A seeded shuffle picks a stable slice.
LIMIT: int | None = None
SEED = 0

OUTPUT_DIR = Path(__file__).resolve().parent
PROMPTS_PATH = OUTPUT_DIR / "prompts.jsonl"


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
    if MAX_POPULARITY is not None:
        dataset = dataset.filter(
            lambda row: row["s_pop"] is not None and row["s_pop"] < MAX_POPULARITY
        )
    dataset = dataset.shuffle(seed=SEED)
    if LIMIT is not None:
        dataset = dataset.select(range(min(LIMIT, len(dataset))))

    prompt_rows: list[dict] = []
    for record in dataset:
        answers = _acceptable_answers(record["possible_answers"])
        if not answers:
            continue
        prompt_rows.append(
            {
                "prompt_id": record.get("id"),
                "fact_id": record.get("id"),
                "prompt_text": record["question"],
                "gold_object": answers[0],
                "answer_aliases": answers[1:],
            }
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with PROMPTS_PATH.open("w", encoding="utf-8") as handle:
        for row in prompt_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(prompt_rows)} audit prompts to {PROMPTS_PATH}")


if __name__ == "__main__":
    main()
