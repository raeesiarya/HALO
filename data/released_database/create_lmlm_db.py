import json
import os
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm


def create_lmlm_database() -> None:
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")

    ds = load_dataset("kilian-group/LMLM-database", token=hf_token)

    triplets = ds["test"][0]["triplets"]

    jsonl_path = Path("data/lmlm_database.jsonl")
    json_path = Path("data/lmlm_database.json")

    with jsonl_path.open("w", encoding="utf-8") as f:
        for i, (s, r, o) in enumerate(tqdm(triplets, desc="Saving triplets")):
            json.dump({"id": i, "s": s, "r": r, "o": o}, f, ensure_ascii=False)
            f.write("\n")

    database = {
        "entities": sorted({s for s, _, _ in triplets}),
        "relationships": sorted({r for _, r, _ in triplets}),
        "return_values": sorted({o for _, _, o in triplets}),
        "triplets": [list(triplet) for triplet in triplets],
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(database, f, ensure_ascii=False)


if __name__ == "__main__":
    create_lmlm_database()
