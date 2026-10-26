"""Compare two standard-audit results files row by row.

Used to validate a port of the audit runtime (scripts/verify_co_lmlm_port.sh):
re-audit a canonical slice on the new platform and check that it reproduces
the reference run. Rows are matched on (prompt key, state); only keys
present in both files are compared, and coverage differences (facts one run
audited and the other skipped) are reported separately.

Per state it reports agreement on the generated text, on answer
correctness, and on the retrieved entry (the selected candidate's id), and
lists a few disagreements. Exit status is 1 when correctness agreement on
any state falls below --min-agreement.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from halo.core.entanglement import fact_key  # noqa: E402
from halo.core.metrics import _result_is_correct  # noqa: E402


def _rows(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[(fact_key(row), str(row.get("state")))] = row
    return rows


def _selected_entry(row: dict[str, Any]) -> str | None:
    trace = row.get("retrieval_trace") or {}
    selected = trace.get("selected_candidate") or {}
    return selected.get("entry_id")


def _skipped_keys(candidate: Path) -> set[str]:
    """Facts the candidate run attempted but skipped (its skip log)."""
    stem = candidate.name.removesuffix("_results.jsonl")
    path = candidate.with_name(f"{stem}_skipped_facts.jsonl")
    if not path.is_file():
        return set()
    with path.open(encoding="utf-8") as handle:
        return {
            str(json.loads(line).get("fact_id"))
            for line in handle
            if line.strip()
        }


def compare(reference: Path, candidate: Path, *, examples: int = 5) -> dict[str, Any]:
    ref = _rows(reference)
    cand = _rows(candidate)
    candidate_skipped = _skipped_keys(candidate)
    ref_keys = {key for key, _ in ref}
    cand_keys = {key for key, _ in cand}
    shared = sorted(set(ref) & set(cand))
    per_state: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"rows": 0, "text": 0, "correct": 0, "entry": 0, "mismatches": []}
    )
    for key in shared:
        a, b = ref[key], cand[key]
        stats = per_state[key[1]]
        stats["rows"] += 1
        same_text = a.get("model_output") == b.get("model_output")
        same_correct = _result_is_correct(a) == _result_is_correct(b)
        stats["text"] += same_text
        stats["correct"] += same_correct
        stats["entry"] += _selected_entry(a) == _selected_entry(b)
        if not same_text and len(stats["mismatches"]) < examples:
            stats["mismatches"].append(
                {
                    "key": key[0],
                    "reference": a.get("model_output"),
                    "candidate": b.get("model_output"),
                    "correctness_changed": not same_correct,
                }
            )
    return {
        "reference_facts": len(ref_keys),
        "candidate_facts": len(cand_keys),
        "shared_facts": len(ref_keys & cand_keys),
        "only_candidate": sorted(cand_keys - ref_keys)[:examples],
        "only_candidate_count": len(cand_keys - ref_keys),
        # Facts the reference audited that the candidate attempted but
        # skipped (its FULL pass found no answer-bearing entry).
        "only_reference_attempted": sorted(ref_keys & candidate_skipped)[:examples],
        "only_reference_attempted_count": len(ref_keys & candidate_skipped),
        "states": {
            state: {
                "rows": s["rows"],
                "text_agreement": s["text"] / s["rows"] if s["rows"] else None,
                "correctness_agreement": s["correct"] / s["rows"] if s["rows"] else None,
                "selected_entry_agreement": s["entry"] / s["rows"] if s["rows"] else None,
                "mismatches": s["mismatches"],
            }
            for state, s in sorted(per_state.items())
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--min-agreement", type=float, default=0.98)
    parser.add_argument("--json", type=Path, help="also write the report here")
    args = parser.parse_args(argv)

    report = compare(args.reference, args.candidate)
    print(
        f"facts: reference {report['reference_facts']}, candidate "
        f"{report['candidate_facts']}, shared {report['shared_facts']}"
    )
    if report["only_candidate_count"]:
        print(
            f"  {report['only_candidate_count']} facts audited only by the "
            f"candidate (reference skipped them), e.g. {report['only_candidate']}"
        )
    if report["only_reference_attempted_count"]:
        print(
            f"  {report['only_reference_attempted_count']} facts audited by the "
            "reference but skipped by the candidate, e.g. "
            f"{report['only_reference_attempted']}"
        )
    ok = report["shared_facts"] > 0
    for state, stats in report["states"].items():
        print(
            f"  {state:>7}: rows={stats['rows']} text={stats['text_agreement']:.3f} "
            f"correct={stats['correctness_agreement']:.3f} "
            f"entry={stats['selected_entry_agreement']:.3f}"
        )
        for item in stats["mismatches"]:
            flag = " (correctness changed)" if item["correctness_changed"] else ""
            print(
                f"           {item['key']}: {item['reference']!r} -> "
                f"{item['candidate']!r}{flag}"
            )
        if stats["correctness_agreement"] < args.min_agreement:
            ok = False
    if args.json:
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("PASS" if ok else f"FAIL (correctness agreement below {args.min_agreement})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
