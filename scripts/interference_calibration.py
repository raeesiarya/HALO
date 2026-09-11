"""Where do interference cases sit relative to the retrieval threshold?

Retrieval interference I(f) counts facts answered correctly with retrieval
disabled after deletion but incorrectly with retrieval enabled: deletion
changed which entry wins retrieval, and the surviving winner overrode a
correct parametric answer. One reading is a calibration failure: the harmful
splice would then sit just above the retrieval threshold, where the model
trusts low-confidence entries. This script tests that reading against the
raw DEL-ON retrieval traces of the standard audit.

Among FULL-correct facts that are also DEL-OFF-correct (the facts where
interference is possible at all), it splits outcomes into benign (DEL-ON
correct) and interference (DEL-ON incorrect) and compares the cosine score
of the DEL-ON selected entry between the two, including the share of each
distribution inside a band just above the threshold.

Reads results/co-lmlm/<dataset>/prompts*_results.jsonl; writes
results/status_update_2/interference_calibration.json and prints a summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from halo.core.metrics import _group_results_by_fact, _result_is_correct  # noqa: E402

RESULTS = ROOT / "results"
OUT = RESULTS / "status_update_2"

DATASETS = {
    "trex": "trex/prompts_trex_results.jsonl",
    "popqa": "popqa/prompts_results.jsonl",
    "googlere": "googlere/prompts_googlere_results.jsonl",
    "counterfact": "counterfact/prompts_counterfact_results.jsonl",
    "zsre": "zsre/prompts_zsre_results.jsonl",
}

# Only these fields are needed downstream; dropping the rest keeps the
# grouped rows small (the raw rows carry full candidate lists).
SLIM_FIELDS = (
    "fact_id",
    "prompt_id",
    "prompt",
    "ground_truth",
    "deletion_manifest",
    "state",
    "model_output",
    "object_aliases",
)


def load_slim_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            slim = {field: row.get(field) for field in SLIM_FIELDS}
            trace = row.get("retrieval_trace") or {}
            selected = trace.get("selected_candidate") or {}
            slim["_selected_score"] = selected.get("score")
            slim["_threshold"] = trace.get("threshold")
            rows.append(slim)
    return rows


def score_stats(scores: list[float], threshold: float, band: float) -> dict[str, Any]:
    if not scores:
        return {"n_spliced": 0}
    arr = np.asarray(scores, dtype=float)
    return {
        "n_spliced": int(arr.size),
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "share_in_band": float(np.mean((arr >= threshold) & (arr < threshold + band))),
    }


def analyze_dataset(path: Path, band: float) -> dict[str, Any]:
    grouped = _group_results_by_fact(load_slim_rows(path))
    thresholds: set[float] = set()
    outcomes: dict[str, dict[str, Any]] = {
        "benign": {"scores": [], "n": 0, "n_no_splice": 0},
        "interference": {"scores": [], "n": 0, "n_no_splice": 0},
    }
    for states in grouped.values():
        if not {"FULL", "DEL-ON", "DEL-OFF"} <= states.keys():
            continue
        # Interference is only defined where the parametric answer is
        # correct to begin with: FULL-correct and DEL-OFF-correct.
        if not (
            _result_is_correct(states["FULL"])
            and _result_is_correct(states["DEL-OFF"])
        ):
            continue
        del_on = states["DEL-ON"]
        label = "benign" if _result_is_correct(del_on) else "interference"
        bucket = outcomes[label]
        bucket["n"] += 1
        score = del_on.get("_selected_score")
        if score is None:
            bucket["n_no_splice"] += 1
        else:
            bucket["scores"].append(float(score))
        if del_on.get("_threshold") is not None:
            thresholds.add(float(del_on["_threshold"]))
    threshold = max(thresholds) if thresholds else 0.7
    report: dict[str, Any] = {"threshold": threshold, "band": band}
    for label, bucket in outcomes.items():
        report[label] = {
            "n": bucket["n"],
            "n_no_splice": bucket["n_no_splice"],
            **score_stats(bucket["scores"], threshold, band),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--band",
        type=float,
        default=0.05,
        help="width of the near-threshold band [tau, tau + band)",
    )
    parser.add_argument("--out", type=Path, default=OUT / "interference_calibration.json")
    args = parser.parse_args()

    report: dict[str, Any] = {}
    for dataset, rel_path in DATASETS.items():
        path = RESULTS / "co-lmlm" / rel_path
        if not path.exists():
            print(f"{dataset}: missing {path}, skipped", file=sys.stderr)
            continue
        report[dataset] = analyze_dataset(path, args.band)
        b, i = report[dataset]["benign"], report[dataset]["interference"]
        print(
            f"{dataset:12s} tau={report[dataset]['threshold']:.2f}  "
            f"benign n={b['n']} med={b.get('median', float('nan')):.3f} "
            f"band={b.get('share_in_band', float('nan')):.1%}  |  "
            f"interference n={i['n']} no-splice={i['n_no_splice']} "
            f"med={i.get('median', float('nan')):.3f} "
            f"band={i.get('share_in_band', float('nan')):.1%}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
