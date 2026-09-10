#!/usr/bin/env python3
"""Robustness numbers for the paper: CIs, backfill sensitivity, factq subcohort.

Three stages, all pure re-analysis of the released per-fact dumps (no GPU,
no model). Each recomputed aggregate is asserted against the released
aggregates (numbers.json / entanglement_curves.csv / policy_full.json /
evasion.csv) before the new numbers are trusted. Writes
results/status_update_2/robustness.json.

  ci        Wilson 95% CIs for the per-fact channels S/R/I/L on the
            FULL-correct cohort, for baseline closed-book correctness on the
            identical facts, and for the adversarial attack-gain rates.
            Paired McNemar exact tests for (a) Co-LMLM L vs. each baseline
            on shared facts and (b) the null-retrieval vs. forbid-token
            DEL-OFF controls on the audited cohort.
  backfill  Entanglement sensitivity: recompute collateral X, G(f)=0 and
            never-forgotten shares restricting N(f) to strictly within-ball
            neighbors (cosine >= ball), dropping the backfilled fill-ins.
            The all-neighbor recomputation is validated against
            entanglement_curves.csv first.
  factq     Policy comparison on the factq-covered subcohort: S/R/I per
            policy restricted to facts with generated questions, so the
            factq row is compared with every other policy on the identical
            fact set. Validated against policy_full.json on the full cohort.
  sweep-channels
            Reconcile the policy matrix with the radius sweep: the DEL-OFF
            outcome is closure-independent, so combining each sweep radius's
            DEL-ON outcome with the standard run's DEL-OFF outcome yields
            S/R/I for pure geometric deletion at every swept radius, on the
            FULL-correct cohort. Validated at rho=0.85 against the geometric
            policy row and against the released efficacy curve at every rho.
  sweep-sizes
            Deletion cost of pure geometric closures: entries deleted per
            fact at every swept radius (median/mean/p90/max, truncation
            share), read from the sweep manifests.
  artifact-split
            Decompose the artifact rate (DEL-ON-correct facts with no
            answer-mentioning retained candidate) into its parametric part
            (also DEL-OFF-correct) and its retrieval-only remainder.
            Validated against the released artifact rate.

Run from the repo root:

  uv run python scripts/robustness_analysis.py --stage all
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

from halo.core.entanglement import fact_key
from halo.core.metrics import _result_is_correct

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "status_update_2"
OUT_PATH = OUT / "robustness.json"

DATASETS = ["trex", "popqa", "googlere", "counterfact", "zsre"]
BASELINES = ["smollm2-360m", "standard-lm-360m-fw"]
POLICIES = ["oracle", "geometric", "provenance", "factq", "value", "hybrid"]
RHOS = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70]
Z95 = 1.959963984540054
TOL = 5e-4


# --------------------------------------------------------------- utilities
def one(pattern: str) -> str:
    hits = sorted(glob.glob(pattern))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one match for {pattern}, got {hits}")
    return hits[0]


def check(
    name: str, computed: float | None, expected: float | None, tol: float = TOL
) -> None:
    if expected is None:
        return
    if computed is None or abs(computed - expected) > tol:
        raise AssertionError(f"{name}: computed {computed} vs released {expected}")


def wilson(successes: int, n: int, z: float = Z95) -> dict:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return {"rate": None, "lo": None, "hi": None, "n": 0, "k": 0}
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return {
        "rate": p,
        "lo": max(0.0, center - half),
        "hi": min(1.0, center + half),
        "n": n,
        "k": successes,
    }


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p from the discordant counts (binomial)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1))
    p = 2.0 * float(Fraction(tail, 2**n))
    return min(1.0, p)


def stream_facts(path: str):
    """One pass over a results.jsonl: fact -> {state: correct}."""
    per_fact: dict[str, dict] = defaultdict(dict)
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            k, state = fact_key(row), row.get("state")
            if not k or not state:
                continue
            if state not in per_fact[k]:  # one prompt per fact in these runs
                per_fact[k][state] = bool(_result_is_correct(row))
    return per_fact


def cohorts(per_fact: dict[str, dict]) -> tuple[set[str], set[str]]:
    eligible = {k for k, st in per_fact.items() if "DEL-ON" in st and "DEL-OFF" in st}
    return eligible, {k for k in eligible if per_fact[k].get("FULL")}


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def save_block(name: str, block: dict) -> None:
    out = load_json(OUT_PATH) if OUT_PATH.exists() else {}
    out[name] = block
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=1)
    print(f"{name}: wrote block -> {OUT_PATH}")


# ---------------------------------------------------------------- stage: ci
def stage_ci() -> None:
    numbers = load_json(OUT / "numbers.json")
    out = {}
    for ds in DATASETS:
        base = f"{RESULTS}/co-lmlm/{ds}"
        per_fact = stream_facts(one(f"{base}/prompts*_results.jsonl"))
        eligible, full_c = cohorts(per_fact)
        released = numbers["datasets"][ds]
        check(f"{ds} n", len(full_c), released["full_correct"])

        n = len(full_c)
        ind = {
            "S": [per_fact[k]["DEL-ON"] for k in full_c],
            "L": [per_fact[k]["DEL-OFF"] for k in full_c],
            "R": [per_fact[k]["DEL-ON"] and not per_fact[k]["DEL-OFF"] for k in full_c],
            "I": [not per_fact[k]["DEL-ON"] and per_fact[k]["DEL-OFF"] for k in full_c],
        }
        d = {"n_full_correct": n, "channels": {}}
        for ch, xs in ind.items():
            ci = wilson(sum(xs), n)
            check(f"{ds} {ch}", ci["rate"], released["over_full_correct"][ch])
            d["channels"][ch] = ci

        # baselines on the identical FULL-correct facts, paired against L
        d["baselines"] = {}
        for bl in BASELINES:
            bfacts = stream_facts(one(f"{RESULTS}/{bl}/{ds}/prompts*_results.jsonl"))
            bcorrect = {k: st["FULL"] for k, st in bfacts.items() if "FULL" in st}
            common = sorted(full_c & set(bcorrect))
            ci = wilson(sum(bcorrect[k] for k in common), len(common))
            check(
                f"{ds} {bl}",
                ci["rate"],
                released["baselines"][bl]["rate_on_full_correct"],
            )
            b = sum(1 for k in common if per_fact[k]["DEL-OFF"] and not bcorrect[k])
            c = sum(1 for k in common if not per_fact[k]["DEL-OFF"] and bcorrect[k])
            d["baselines"][bl] = {
                **ci,
                "mcnemar_vs_L": {
                    "colmlm_only": b,
                    "baseline_only": c,
                    "p": mcnemar_exact(b, c),
                },
            }

        # DEL-OFF mechanism: null-retrieval vs forbid-token on the cohort
        forbid = stream_facts(
            one(f"{base}/del_off_sensitivity/forbid-token/prompts*_results.jsonl")
        )
        common = sorted(k for k in eligible if "DEL-OFF" in forbid.get(k, {}))
        b = sum(
            1 for k in common if per_fact[k]["DEL-OFF"] and not forbid[k]["DEL-OFF"]
        )
        c = sum(
            1 for k in common if not per_fact[k]["DEL-OFF"] and forbid[k]["DEL-OFF"]
        )
        d["del_off_mechanism"] = {
            "n": len(common),
            "null_only": b,
            "forbid_only": c,
            "delta": (c - b) / len(common) if common else None,
            "p": mcnemar_exact(b, c),
        }

        # adversarial attack gain per (template, epsilon) on FULL-correct
        adv_dir = glob.glob(f"{base}/prompts*adversarial")
        if adv_dir:
            baseline_c: dict[str, bool] = {}
            attack_c: dict[tuple, dict[str, bool]] = defaultdict(dict)
            with open(f"{adv_dir[0]}/adversarial_results.jsonl") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    tag = row.get("adversarial") or {}
                    k = str(tag.get("target_key", ""))
                    if not k:
                        continue
                    if tag.get("role") == "baseline":
                        baseline_c[k] = bool(_result_is_correct(row))
                    elif tag.get("role") == "attack":
                        key = (str(tag.get("template")), float(tag.get("epsilon")))
                        attack_c[key][k] = bool(_result_is_correct(row))
            ev = {
                (r["template"], float(r["epsilon"])): r
                for r in csv.DictReader(open(f"{adv_dir[0]}/evasion.csv"))
                if r["topology"] == "single"
            }
            d["adversarial_gain"] = {}
            for (tpl, eps), correct in sorted(attack_c.items()):
                facts = sorted(full_c & set(correct) & set(baseline_c))
                gains = sum(1 for k in facts if correct[k] and not baseline_c[k])
                ci = wilson(gains, len(facts))
                check(
                    f"{ds} adv {tpl} {eps}",
                    ci["rate"],
                    float(ev[(tpl, eps)]["attack_gain_rate"]),
                )
                d["adversarial_gain"][f"{tpl}@{eps}"] = ci

        out[ds] = d
        fmt = {
            ch: f"{100 * v['rate']:.1f} [{100 * v['lo']:.1f}, {100 * v['hi']:.1f}]"
            for ch, v in d["channels"].items()
        }
        print(f"ci: {ds} n={n} {fmt} mech p={d['del_off_mechanism']['p']:.3g}")
    save_block("ci", out)


# ---------------------------------------------------------- stage: backfill
def stage_backfill() -> None:
    out = {}
    for ds in DATASETS:
        base = f"{RESULTS}/co-lmlm/{ds}"
        sweep = one(f"{base}/prompts*sweep")
        nb = load_json(Path(sweep) / "neighbors.json")["neighbors"]

        # FULL-correctness for neighbor breakage, from the main run
        full_correct: set[str] = set()
        with open(one(f"{base}/prompts*_results.jsonl")) as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("state") == "FULL" and _result_is_correct(row):
                    full_correct.add(fact_key(row))

        # stream the sweep: target efficacy + per-neighbor breakage per rho
        target_forgot: dict[str, dict[float, bool]] = defaultdict(dict)
        neighbor_ok: dict[tuple, dict[str, bool]] = defaultdict(dict)
        for rho in RHOS:
            with open(f"{sweep}/sweep_rho_{rho:.4f}.jsonl") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    tag = row.get("sweep") or {}
                    t = str(tag.get("target_key", ""))
                    if not t:
                        continue
                    if tag.get("role") == "target":
                        target_forgot[t][rho] = not _result_is_correct(row)
                    elif tag.get("role") == "neighbor":
                        neighbor_ok[(t, rho)][fact_key(row)] = bool(
                            _result_is_correct(row)
                        )

        # released aggregates to validate the all-neighbor recomputation
        by_rho_released = defaultdict(list)
        for r in csv.DictReader(open(f"{sweep}/entanglement_curves.csv")):
            by_rho_released[float(r["rho"])].append(float(r["collateral"]))
        released_gaps = [
            float(r["gap"])
            for r in csv.DictReader(open(f"{sweep}/entanglement_gaps.csv"))
            if r["gap_eligible"] == "True"
        ]

        def analyze(within_only: bool) -> dict:
            coll_by_rho = defaultdict(list)
            gaps = []
            n_eligible = 0
            for t, forgot in target_forgot.items():
                if t not in full_correct:
                    continue
                neigh = [
                    e["neighbor"]
                    for e in nb.get(t, [])
                    if not within_only or e["within_ball"]
                ]
                if not neigh:
                    continue
                n_eligible += 1
                gap = None
                for rho in RHOS:
                    broken = sum(
                        1
                        for nk in neigh
                        if nk in full_correct
                        and not neighbor_ok.get((t, rho), {}).get(nk, True)
                    )
                    x = broken / len(neigh)
                    coll_by_rho[rho].append(x)
                    term = (0.0 if forgot.get(rho) else 1.0) + x
                    gap = term if gap is None else min(gap, term)
                gaps.append(gap)
            return {
                "n_targets": n_eligible,
                "collateral_by_rho": {
                    f"{rho:.2f}": sum(v) / len(v) if v else None
                    for rho, v in sorted(coll_by_rho.items())
                },
                "share_gap_zero": sum(g == 0.0 for g in gaps) / len(gaps)
                if gaps
                else None,
                "share_never_forgotten": sum(g >= 1.0 for g in gaps) / len(gaps)
                if gaps
                else None,
            }

        alln, within = analyze(False), analyze(True)
        for rho, vals in by_rho_released.items():
            check(
                f"{ds} collateral rho={rho}",
                alln["collateral_by_rho"][f"{rho:.2f}"],
                sum(vals) / len(vals),
            )
        check(
            f"{ds} share_gap_zero",
            alln["share_gap_zero"],
            sum(g == 0.0 for g in released_gaps) / len(released_gaps),
        )

        counts = [
            sum(e["within_ball"] for e in nb[t])
            for t in target_forgot
            if t in full_correct and nb.get(t)
        ]
        out[ds] = {
            "all_neighbors": alln,
            "within_ball_only": within,
            "targets_with_zero_within_ball": sum(c == 0 for c in counts),
            "targets_with_any_backfill": sum(
                1
                for t in target_forgot
                if t in full_correct
                and nb.get(t)
                and any(not e["within_ball"] for e in nb[t])
            ),
        }
        print(
            f"backfill: {ds} all(n={alln['n_targets']}, "
            f"G0={alln['share_gap_zero']:.3f}) "
            f"within(n={within['n_targets']}, "
            f"G0={within['share_gap_zero']:.3f}, "
            f"never={within['share_never_forgotten']:.3f}) "
            f"zero-within={out[ds]['targets_with_zero_within_ball']}"
        )
    save_block("backfill", out)


# ------------------------------------------------------------- stage: factq
def stage_factq() -> None:
    policy_full = load_json(OUT / "policy_full.json")
    out = {}
    for ds in DATASETS:
        base = f"{RESULTS}/co-lmlm/{ds}"
        covered = set()
        with open(one(f"{base}/prompts*factq.jsonl")) as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("factq_questions"):
                    covered.add(str(row.get("prompt_id") or row.get("fact_id")))

        d = {"covered_prompts": len(covered), "policies": {}}
        for pol in POLICIES:
            per_fact = stream_facts(
                one(f"{base}/policy_matrix/{pol}/prompts*_results.jsonl")
            )
            _, full_c = cohorts(per_fact)
            check(
                f"{ds} {pol} R (full cohort)",
                sum(
                    per_fact[k]["DEL-ON"] and not per_fact[k]["DEL-OFF"] for k in full_c
                )
                / len(full_c),
                policy_full[ds].get(pol, {}).get("R_full"),
            )
            sub = sorted(full_c & covered)
            n = len(sub)
            d["policies"][pol] = {
                "n": n,
                "S": wilson(sum(per_fact[k]["DEL-ON"] for k in sub), n),
                "R": wilson(
                    sum(
                        per_fact[k]["DEL-ON"] and not per_fact[k]["DEL-OFF"]
                        for k in sub
                    ),
                    n,
                ),
                "I": wilson(
                    sum(
                        not per_fact[k]["DEL-ON"] and per_fact[k]["DEL-OFF"]
                        for k in sub
                    ),
                    n,
                ),
            }
        out[ds] = d
        fmt = {pol: f"{100 * v['R']['rate']:.1f}" for pol, v in d["policies"].items()}
        print(f"factq: {ds} n_subcohort={d['policies']['factq']['n']} R% {fmt}")
    save_block("factq_subcohort", out)


# ---------------------------------------------------- stage: sweep-channels
def stage_sweep_channels() -> None:
    numbers = load_json(OUT / "numbers.json")
    policy_full = load_json(OUT / "policy_full.json")
    out = {}
    for ds in DATASETS:
        base = f"{RESULTS}/co-lmlm/{ds}"
        per_fact = stream_facts(one(f"{base}/prompts*_results.jsonl"))
        _, full_c = cohorts(per_fact)

        sweep = one(f"{base}/prompts*sweep")
        delon: dict[float, dict[str, bool]] = defaultdict(dict)
        for rho in RHOS:
            with open(f"{sweep}/sweep_rho_{rho:.4f}.jsonl") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    tag = row.get("sweep") or {}
                    if tag.get("role") == "target":
                        delon[rho][str(tag.get("target_key", ""))] = bool(
                            _result_is_correct(row)
                        )

        d = {}
        for rho in RHOS:
            facts = sorted(full_c & set(delon[rho]))
            n = len(facts)
            s = sum(delon[rho][k] for k in facts) / n
            check(
                f"{ds} sweep rho={rho} efficacy",
                1.0 - s,
                numbers["entanglement"][ds]["curve"][str(rho)]["efficacy"],
            )
            d[f"{rho:.2f}"] = {
                "n": n,
                "S": s,
                "R": sum(delon[rho][k] and not per_fact[k]["DEL-OFF"] for k in facts)
                / n,
                "I": sum(not delon[rho][k] and per_fact[k]["DEL-OFF"] for k in facts)
                / n,
            }
        check(
            f"{ds} sweep rho=0.85 vs geometric policy S",
            d["0.85"]["S"],
            policy_full[ds]["geometric"]["S_full"],
        )
        out[ds] = {"by_rho": d, "L": numbers["datasets"][ds]["over_full_correct"]["L"]}
        fmt = {
            rho: f"S={100 * v['S']:.1f} R={100 * v['R']:.1f} I={100 * v['I']:.1f}"
            for rho, v in d.items()
        }
        print(f"sweep-channels: {ds} L={100 * out[ds]['L']:.1f} {fmt}")
    save_block("sweep_channels", out)


# ------------------------------------------------------ stage: sweep-sizes
def stage_sweep_sizes() -> None:
    """Deletion cost of pure geometric closures: entries deleted per fact at
    every swept radius (median / mean / max over targets, plus the share of
    closures truncated at the size cap), read from the sweep manifests."""
    out = {}
    for ds in DATASETS:
        sweep = one(f"{RESULTS}/co-lmlm/{ds}/prompts*sweep")
        d = {}
        for rho in RHOS:
            sizes, truncated = [], 0
            with open(f"{sweep}/sweep_rho_{rho:.4f}.jsonl") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if (row.get("sweep") or {}).get("role") != "target":
                        continue
                    manifest = row.get("deletion_manifest") or {}
                    sizes.append(len(manifest.get("entry_ids") or []))
                    truncated += bool((manifest.get("metadata") or {}).get("truncated"))
            sizes.sort()
            n = len(sizes)
            d[f"{rho:.2f}"] = {
                "n": n,
                "median": sizes[n // 2] if n else None,
                "mean": sum(sizes) / n if n else None,
                "p90": sizes[int(0.9 * n)] if n else None,
                "max": sizes[-1] if n else None,
                "truncated_share": truncated / n if n else None,
            }
        out[ds] = d
        fmt = {
            rho: f"med={v['median']} p90={v['p90']} max={v['max']} "
            f"trunc={100 * v['truncated_share']:.1f}%"
            for rho, v in d.items()
        }
        print(f"sweep-sizes: {ds} {fmt}")
    save_block("sweep_sizes", out)


# ----------------------------------------------------- stage: artifact-split
def stage_artifact_split() -> None:
    from halo.core.metrics import trace_has_gold_equivalent

    numbers = load_json(OUT / "numbers.json")
    out = {}
    for ds in DATASETS:
        base = f"{RESULTS}/co-lmlm/{ds}"
        per_fact: dict[str, dict] = defaultdict(dict)
        artifact: dict[str, bool] = {}
        with open(one(f"{base}/prompts*_results.jsonl")) as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                k, state = fact_key(row), row.get("state")
                if not k or not state or state in per_fact[k]:
                    continue
                per_fact[k][state] = bool(_result_is_correct(row))
                if state == "DEL-ON":
                    artifact[k] = per_fact[k][state] and not trace_has_gold_equivalent(
                        row
                    )
        eligible, full_c = cohorts(per_fact)
        rate = sum(artifact[k] for k in eligible) / len(eligible)
        check(f"{ds} artifact", rate, numbers["datasets"][ds]["artifact"], tol=2e-3)
        both = sum(artifact[k] and per_fact[k]["DEL-OFF"] for k in eligible)
        retr = sum(artifact[k] and not per_fact[k]["DEL-OFF"] for k in eligible)
        out[ds] = {
            "n": len(eligible),
            "artifact_rate": rate,
            "parametric_share": both / len(eligible),
            "retrieval_only_share": retr / len(eligible),
            "artifact_rate_full_correct": sum(artifact[k] for k in full_c)
            / len(full_c),
            "parametric_share_full_correct": sum(
                artifact[k] and per_fact[k]["DEL-OFF"] for k in full_c
            )
            / len(full_c),
        }
        print(
            f"artifact-split: {ds} rate={100 * rate:.1f} "
            f"parametric={100 * out[ds]['parametric_share']:.1f} "
            f"retrieval-only={100 * out[ds]['retrieval_only_share']:.1f} "
            f"(full-correct: {100 * out[ds]['artifact_rate_full_correct']:.1f}"
            f"/{100 * out[ds]['parametric_share_full_correct']:.1f})"
        )
    save_block("artifact_split", out)


# ------------------------------------------------------------------- main
STAGES = {
    "ci": stage_ci,
    "backfill": stage_backfill,
    "factq": stage_factq,
    "sweep-channels": stage_sweep_channels,
    "sweep-sizes": stage_sweep_sizes,
    "artifact-split": stage_artifact_split,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", default="all", choices=[*STAGES, "all"])
    args = ap.parse_args()
    for name, fn in STAGES.items():
        if args.stage in (name, "all"):
            print(f"== {name} ==")
            fn()


if __name__ == "__main__":
    main()
