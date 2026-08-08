#!/usr/bin/env python3
"""Reproduce every number and figure in "Co-LMLM Deletion Audit: Status Update 2".

This script consolidates the full analysis pipeline behind the report: all
rates are recomputed from the raw per-fact result rows and asserted against
HALO's own aggregate CSVs, so the paper numbers cannot drift from the release.

Stages (each writes JSON/JSONL artifacts under results/status_update_2/):

  extract         Stream all results.jsonl files once. Produces:
                  - numbers.json: cross-state S/R/I/L (cohort and FULL-
                    conditioned), per-dataset cohort funnel, baseline
                    correctness on the matched cohorts, entanglement
                    aggregates and gap shares, probe summaries with the
                    per-fact phi(probe, behavior), adversarial gain/
                    regression rates from evasion.csv.
                  - cohort_<ds>.jsonl: per audited fact, prompt, gold,
                    aliases, FULL/DEL-OFF correctness (input to later stages).
                  - perfact_<baseline>.json: per-fact closed-book correctness.
  policies        FULL-conditioned S/R/I/L per deletion policy (T-REx policy
                  matrix). Produces policy_full.json.
  probe-controls  Prompt-only control probes through HALO's own run_probe
                  (identical ridge/CV/scoring protocol): hashed char-trigram
                  features of the prompt, and the mean of the prompt's input
                  token embeddings (SmolLM2-360M embedding table; the
                  Standard-LM tokenizer config does not load under the pinned
                  transformers version). Produces probe_controls.json.
  probe-grouped   Reruns the query-embedding probe AND both prompt-only
                  controls with folds grouped by canonicalized proposition
                  (normalized subject, relation, answer) instead of fact ID.
                  ZsRE and PopQA contain paraphrase prompts of the same
                  proposition; under fact-ID folds those straddle the
                  train/test split and inflate probe and controls alike
                  (ZsRE query probe 40.6% -> 7.4%). The report's probe
                  numbers and figure use this stage's output. Produces
                  probe_grouped.json.
  paraphrase      Value-policy (oracle answer filter) survivor audit on
                  T-REx: classifies DEL-ON survivors and dumps the
                  retrieval-mediated ones (correct only with retrieval on,
                  each with its spliced retained entry) for manual review.
                  Produces value_survivors.jsonl. The per-fact adjudication
                  of those rows (surface variants vs. associative cues)
                  lives in annotations/status_update_2/
                  value_survivor_labels.csv; this stage computes the split
                  from that file and warns if it diverges from the report's
                  numbers (REPORTED_SPLIT below).
  frequency       Answer density: per audited fact, the number of entries in
                  its top-500 retrieval envelope caught by the value
                  predicate, read from the closure tarballs under
                  results/co-lmlm. Quartile-binned
                  unaided answerability and Spearman rho for all three
                  models on identical facts. Produces frequency.json.
  figures         All five report figures (baselines, frequency,
                  entanglement, probe, and the adversarial numbers behind
                  Table 3 are printed rather than plotted since the report
                  uses a table). Written to --figdir.

Run from the repo root:

  uv run python scripts/status_update_2_analysis.py --stage all
  uv run python scripts/status_update_2_analysis.py --stage frequency
  uv run python scripts/status_update_2_analysis.py --stage figures \
      --figdir /path/to/paper/figures

Stage order matters on a fresh checkout: extract must run before policies /
probe-controls / probe-grouped / paraphrase / frequency, and figures needs
extract, probe-grouped, and frequency.
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

from halo.core.entanglement import fact_key
from halo.core.equivalence import normalize_text
from halo.core.metrics import _result_is_correct

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CLOSURES = RESULTS / "co-lmlm"
OUT = RESULTS / "status_update_2"

DATASETS = ["trex", "popqa", "googlere", "counterfact", "zsre"]
BASELINES = ["smollm2-360m", "standard-lm-360m-fw"]
POLICIES = ["oracle", "provenance", "geometric", "value", "hybrid"]
TOL = 5e-4  # recomputed rates must match HALO's CSVs within this

# Split of the 110 retrieval-mediated value-policy survivors as currently
# stated in the report (surface variants vs. associative cues). The source
# of truth is the per-fact re-adjudication in
# annotations/status_update_2/value_survivor_labels.csv (label + borderline
# flag + rationale per fact); the paraphrase stage computes the totals from
# that file and warns if they diverge from these numbers, in which case the
# REPORT should be updated to match the file, not the other way around.
REPORTED_SPLIT = {"surface_variants": 49, "associative_cues": 61}


# --------------------------------------------------------------- utilities
def one(pattern: str) -> str:
    hits = sorted(glob.glob(pattern))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one match for {pattern}, got {hits}")
    return hits[0]


def read_rows(path: str) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def numeric_cols(row: dict) -> dict:
    out = {}
    for k, v in row.items():
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def rate(xs) -> float | None:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def check(name: str, computed: float | None, expected: float | None) -> None:
    if expected is None:
        return
    if computed is None or abs(computed - expected) > TOL:
        raise AssertionError(f"{name}: computed {computed} vs CSV {expected}")


def stream_facts(path: str, want_prompt: bool = False):
    """One pass over a results.jsonl: fact -> {state: correct}, plus labels."""
    per_fact: dict[str, dict] = defaultdict(dict)
    labels: dict[str, dict] = {}
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
            if state == "FULL" and k not in labels:
                labels[k] = {
                    "ground_truth": str(row.get("ground_truth", "")),
                    "aliases": list(row.get("object_aliases") or []),
                    "prompt": str(row.get("prompt", "")) if want_prompt else None,
                }
    return per_fact, labels


def spearman(x: list[float], y: list[float]) -> float:
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx) ** 0.5
    vy = sum((b - my) ** 2 for b in ry) ** 0.5
    return cov / (vx * vy) if vx and vy else float("nan")


# ---------------------------------------------------------- stage: extract
def stage_extract() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    numbers = {"datasets": {}, "adversarial": {}, "entanglement": {}, "probe": {}}
    baseline_perfact = {bl: {} for bl in BASELINES}

    for ds in DATASETS:
        base = f"{RESULTS}/co-lmlm/{ds}"
        cross = numeric_cols(read_rows(one(f"{base}/cross_state_metrics.csv"))[0])
        per_state = {r["state"]: numeric_cols(r)
                     for r in read_rows(one(f"{base}/per_state_metrics.csv"))}

        per_fact, labels = stream_facts(one(f"{base}/prompts*_results.jsonl"),
                                        want_prompt=True)
        eligible = {k for k, st in per_fact.items()
                    if "DEL-ON" in st and "DEL-OFF" in st}
        full_c = {k for k in eligible if per_fact[k].get("FULL")}

        def rates(cohort):
            return {
                "S": rate(per_fact[k]["DEL-ON"] for k in cohort),
                "L": rate(per_fact[k]["DEL-OFF"] for k in cohort),
                "R": rate(per_fact[k]["DEL-ON"] and not per_fact[k]["DEL-OFF"]
                          for k in cohort),
                "I": rate(not per_fact[k]["DEL-ON"] and per_fact[k]["DEL-OFF"]
                          for k in cohort),
            }
        over_cohort, over_full = rates(eligible), rates(full_c)
        check(f"{ds} L", over_cohort["L"], cross.get("parametric_leakage"))
        check(f"{ds} L|full", over_full["L"], cross.get("parametric_leakage_given_full"))
        check(f"{ds} R", over_cohort["R"], cross.get("retrieval_mediated_correctness"))
        check(f"{ds} I", over_cohort["I"], cross.get("retrieval_interference"))
        check(f"{ds} S|full", over_full["S"], cross.get("post_deletion_survival_given_full"))

        sens = glob.glob(f"{base}/del_off_sensitivity/forbid-token/cross_state_metrics.csv")
        d = {
            "cohort": len(eligible),
            "full_correct": len(full_c),
            "over_cohort": over_cohort,
            "over_full_correct": over_full,
            "artifact": cross.get("retrieval_artifact_rate"),
            "full_contains": per_state.get("FULL", {}).get("contains_match"),
            "L_forbid_token": (numeric_cols(read_rows(sens[0])[0])
                               .get("parametric_leakage") if sens else None),
            "baselines": {},
        }

        # baselines scored on exactly the matched fact sets
        for bl in BASELINES:
            bfacts, _ = stream_facts(one(f"{RESULTS}/{bl}/{ds}/prompts*_results.jsonl"))
            bcorrect = {k: st.get("FULL") for k, st in bfacts.items() if "FULL" in st}
            baseline_perfact[bl][ds] = {k: bool(v) for k, v in bcorrect.items()}
            common_cohort = eligible & set(bcorrect)
            common_full = full_c & set(bcorrect)
            d["baselines"][bl] = {
                "rate_all": rate(bcorrect.values()),
                "rate_on_cohort": rate(bcorrect[k] for k in common_cohort),
                "rate_on_full_correct": rate(bcorrect[k] for k in common_full),
                "n_common_cohort": len(common_cohort),
            }
            bcross = numeric_cols(read_rows(one(f"{RESULTS}/{bl}/{ds}/cross_state_metrics.csv"))[0])
            check(f"{ds} {bl} L", d["baselines"][bl]["rate_all"],
                  bcross.get("parametric_leakage"))
        numbers["datasets"][ds] = d

        # cohort dump for later stages
        with open(OUT / f"cohort_{ds}.jsonl", "w") as f:
            for k in sorted(eligible):
                lab = labels.get(k, {})
                f.write(json.dumps({
                    "fact": k, "prompt": lab.get("prompt"),
                    "ground_truth": lab.get("ground_truth"),
                    "aliases": lab.get("aliases"),
                    "full_correct": k in full_c,
                    "del_off_correct": per_fact[k]["DEL-OFF"],
                }) + "\n")

        # entanglement aggregates
        sweep = one(f"{base}/prompts*sweep")
        by_rho = defaultdict(lambda: {"eff": [], "coll": []})
        for r in read_rows(f"{sweep}/entanglement_curves.csv"):
            by_rho[float(r["rho"])]["eff"].append(float(r["efficacy"]))
            by_rho[float(r["rho"])]["coll"].append(float(r["collateral"]))
        gaps = [float(r["gap"]) for r in read_rows(f"{sweep}/entanglement_gaps.csv")
                if r["gap_eligible"] == "True"]
        numbers["entanglement"][ds] = {
            "curve": {str(rho): {"efficacy": rate(v["eff"]),
                                 "collateral": rate(v["coll"])}
                      for rho, v in sorted(by_rho.items())},
            "n_targets": len(gaps),
            "share_gap_zero": rate(g == 0.0 for g in gaps),
            "share_gap_one": rate(g >= 1.0 for g in gaps),
        }

        # probe summary + per-fact phi(probe, behavior) + gold-answer prior
        psum = read_rows(one(f"{base}/prompts*_probe_summary.csv"))[0]
        pf = read_rows(one(f"{base}/prompts*_probe_per_fact.csv"))
        n11 = sum(1 for r in pf if float(r["l_rep"]) > 0 and float(r["behavioral_l"]) > 0)
        n10 = sum(1 for r in pf if float(r["l_rep"]) > 0 and float(r["behavioral_l"]) == 0)
        n01 = sum(1 for r in pf if float(r["l_rep"]) == 0 and float(r["behavioral_l"]) > 0)
        n00 = sum(1 for r in pf if float(r["l_rep"]) == 0 and float(r["behavioral_l"]) == 0)
        denom = ((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)) ** 0.5
        counts = Counter(normalize_text(r["answer"]) for r in pf)
        numbers["probe"][ds] = {
            "facts": int(float(psum["facts"])),
            "candidates": int(float(psum["candidates"])),
            "l_rep_hat": float(psum["l_rep_hat"]),
            "l_hat": float(psum["l_hat"]),
            "phi_rep_behavioral": ((n11 * n00 - n10 * n01) / denom) if denom else None,
            "majority_gold_share": max(counts.values()) / len(pf),
        }

        # adversarial rates (single-survivor topology) straight off evasion.csv
        adv = glob.glob(f"{base}/prompts*adversarial")
        if adv:
            numbers["adversarial"][ds] = [
                {k: (float(r[k]) if k not in ("template", "topology") else r[k])
                 for k in ("epsilon", "template", "baseline_correct_rate",
                           "evasion_rate", "attack_gain_rate",
                           "attack_regression_rate")}
                for r in read_rows(f"{adv[0]}/evasion.csv")
                if r["topology"] == "single"
            ]

    with open(OUT / "numbers.json", "w") as f:
        json.dump(numbers, f, indent=1)
    for bl in BASELINES:
        with open(OUT / f"perfact_{bl}.json", "w") as f:
            json.dump(baseline_perfact[bl], f)
    print("extract: OK, all recomputed rates matched HALO CSVs within", TOL)


# --------------------------------------------------------- stage: policies
def stage_policies() -> None:
    out = {}
    for pol in POLICIES:
        per_fact, _ = stream_facts(
            one(f"{RESULTS}/co-lmlm/trex/policy_matrix/{pol}/prompts*_results.jsonl"))
        full_c = [k for k, st in per_fact.items()
                  if "DEL-ON" in st and "DEL-OFF" in st and st.get("FULL")]
        n = len(full_c)
        out[pol] = {
            "n_full_correct": n,
            "S_full": sum(per_fact[k]["DEL-ON"] for k in full_c) / n,
            "R_full": sum(per_fact[k]["DEL-ON"] and not per_fact[k]["DEL-OFF"]
                          for k in full_c) / n,
            "I_full": sum(not per_fact[k]["DEL-ON"] and per_fact[k]["DEL-OFF"]
                          for k in full_c) / n,
            "L_full": sum(per_fact[k]["DEL-OFF"] for k in full_c) / n,
        }
        print("policies:", pol, {k: round(v, 4) for k, v in out[pol].items()})
    with open(OUT / "policy_full.json", "w") as f:
        json.dump(out, f, indent=1)


# --------------------------------------------- stage: probe-controls
def trigram_features(text: str, dim: int = 2048):
    import numpy as np
    text = f"##{normalize_text(text)}##"
    v = np.zeros(dim, dtype=np.float32)
    for s in range(len(text) - 2):
        v[int(hashlib.md5(text[s:s + 3].encode()).hexdigest(), 16) % dim] += 1.0
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def load_embed_table():
    import numpy as np
    from safetensors import safe_open
    from transformers import AutoTokenizer
    snap = one(os.path.expanduser(
        "~/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-360M/snapshots/*"))
    tok = AutoTokenizer.from_pretrained(snap)
    for st_file in glob.glob(f"{snap}/*.safetensors"):
        with safe_open(st_file, framework="pt") as f:
            for name in f.keys():
                if "embed_tokens.weight" in name:
                    return tok, np.asarray(
                        f.get_tensor(name).to("cpu").float().numpy(),
                        dtype=np.float32)
    raise RuntimeError("no embedding table found; download SmolLM2-360M first")


def stage_probe_controls() -> None:
    import numpy as np
    from halo.core.probe import ProbeConfig, ProbeSample, run_probe
    tok, table = load_embed_table()

    def mean_embed(text):
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if not ids:
            return None
        v = table[ids].mean(axis=0)
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else None

    out = {}
    for ds in DATASETS:
        rows = [json.loads(l) for l in open(OUT / f"cohort_{ds}.jsonl")]
        rows = [r for r in rows if r["prompt"] and r["ground_truth"]]
        labels = {r["fact"]: {"ground_truth": r["ground_truth"],
                              "aliases": tuple(r["aliases"] or ())} for r in rows}
        res = {}
        for name, featfn in [("prompt_ngram", trigram_features),
                             ("prompt_embed", mean_embed)]:
            samples = [ProbeSample(sample_id=r["fact"], fact=r["fact"], vector=v)
                       for r in rows if (v := featfn(r["prompt"])) is not None]
            rep = run_probe(samples, labels, ProbeConfig())
            res[name] = {"l_rep_hat": rep.summary["l_rep_hat"],
                         "facts": rep.summary["facts"]}
            print(f"probe-controls: {ds} {name}: {rep.summary['l_rep_hat']:.4f}")
        out[ds] = res
    with open(OUT / "probe_controls.json", "w") as f:
        json.dump(out, f, indent=1)


# --------------------------------------------- stage: probe-grouped
def load_proposition_groups(ds: str):
    """fact -> canonical proposition key (norm subject, relation, answer),
    plus prompt text, from the FULL rows of the dataset's results.jsonl."""
    path = one(f"{RESULTS}/co-lmlm/{ds}/prompts*_results.jsonl")
    group_of, prompts, labels = {}, {}, {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("state") != "FULL":
                continue
            k = fact_key(row)
            if not k or k in group_of:
                continue
            group_of[k] = "|".join((
                normalize_text(str(row.get("subject", ""))),
                normalize_text(str(row.get("relation") or "")),
                normalize_text(str(row.get("ground_truth", ""))),
            ))
            prompts[k] = str(row.get("prompt", ""))
            labels[k] = {
                "ground_truth": str(row.get("ground_truth", "")),
                "aliases": tuple(row.get("object_aliases") or ()),
            }
    return group_of, prompts, labels


def stage_probe_grouped() -> None:
    import numpy as np
    from halo.core.probe import (ProbeConfig, ProbeSample, load_probe_samples,
                                 run_probe)
    tok, table = load_embed_table()

    def mean_embed(text):
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if not ids:
            return None
        v = table[ids].mean(axis=0)
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else None

    def phi(pf, behavioral):
        n = {(1, 1): 0, (1, 0): 0, (0, 1): 0, (0, 0): 0}
        for r in pf:
            if r["l_rep"] is None or r["fact"] not in behavioral:
                continue
            n[(int(r["l_rep"] > 0), int(behavioral[r["fact"]] > 0))] += 1
        den = ((n[1, 1] + n[1, 0]) * (n[0, 1] + n[0, 0])
               * (n[1, 1] + n[0, 1]) * (n[1, 0] + n[0, 0])) ** 0.5
        return ((n[1, 1] * n[0, 0] - n[1, 0] * n[0, 1]) / den) if den else None

    out = {}
    for ds in DATASETS:
        group_of, prompts, labels = load_proposition_groups(ds)
        behavioral = {r["fact"]: (1.0 if r["del_off_correct"] else 0.0)
                      for r in (json.loads(l)
                                for l in open(OUT / f"cohort_{ds}.jsonl"))}
        npz = Path(one(f"{RESULTS}/co-lmlm/{ds}/prompts*query_embeddings.npz"))
        qsamples = [s for s in load_probe_samples([npz], state="FULL")
                    if s.fact in labels]
        rep = run_probe(qsamples, labels, ProbeConfig(), fold_key=group_of)
        counts = Counter(normalize_text(labels[s.fact]["ground_truth"])
                         for s in qsamples)
        res = {
            "facts": rep.summary["facts"],
            "groups": len({group_of[s.fact] for s in qsamples}),
            "candidates": rep.summary["candidates"],
            "l_rep_hat": rep.summary["l_rep_hat"],
            "l_hat": rate(behavioral[s.fact] for s in qsamples
                          if s.fact in behavioral),
            "phi_rep_behavioral": phi(rep.per_fact, behavioral),
            "majority_gold_share": max(counts.values()) / len(qsamples),
        }
        for name, featfn in [("prompt_ngram", trigram_features),
                             ("prompt_embed", mean_embed)]:
            csamples = [ProbeSample(sample_id=s.fact, fact=s.fact,
                                    vector=v)
                        for s in qsamples
                        if (v := featfn(prompts[s.fact])) is not None]
            crep = run_probe(csamples, labels, ProbeConfig(),
                             fold_key=group_of)
            res[name] = {"l_rep_hat": crep.summary["l_rep_hat"],
                         "facts": crep.summary["facts"]}
        out[ds] = res
        print(f"probe-grouped: {ds} facts={res['facts']} "
              f"groups={res['groups']} probe={res['l_rep_hat']:.4f} "
              f"ngram={res['prompt_ngram']['l_rep_hat']:.4f} "
              f"embed={res['prompt_embed']['l_rep_hat']:.4f} "
              f"phi={res['phi_rep_behavioral']}")
    with open(OUT / "probe_grouped.json", "w") as f:
        json.dump(out, f, indent=1)


# -------------------------------------------------------- stage: paraphrase
def stage_paraphrase() -> None:
    per_fact_rows: dict[str, dict] = defaultdict(dict)
    path = one(f"{RESULTS}/co-lmlm/trex/policy_matrix/value/prompts*_results.jsonl")
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            k, state = fact_key(row), row.get("state")
            if not k or not state or state in per_fact_rows[k]:
                continue
            keep = {"correct": bool(_result_is_correct(row)),
                    "output": row.get("model_output"),
                    "gold": row.get("ground_truth"),
                    "aliases": row.get("object_aliases") or [],
                    "prompt": row.get("prompt")}
            if state == "DEL-ON":
                keep["trace"] = row.get("retrieval_trace")
            per_fact_rows[k][state] = keep

    eligible = {k: st for k, st in per_fact_rows.items()
                if "DEL-ON" in st and "DEL-OFF" in st}
    full_c = {k for k, st in eligible.items() if st.get("FULL", {}).get("correct")}
    survivors = {k for k in full_c if eligible[k]["DEL-ON"]["correct"]}
    r_cell = {k for k in survivors if not eligible[k]["DEL-OFF"]["correct"]}

    dump = []
    for k in sorted(r_cell):
        st = eligible[k]
        tr = st["DEL-ON"].get("trace") or {}
        dump.append({
            "fact": k, "gold": st["DEL-ON"]["gold"],
            "aliases": st["DEL-ON"]["aliases"],
            "prompt": (st["DEL-ON"]["prompt"] or "")[:300],
            "output": st["DEL-ON"]["output"],
            "selected_value": tr.get("selected_value"),
            "selected_candidate": tr.get("selected_candidate"),
            "retained_candidates_count": tr.get("retained_candidates_count"),
        })
    with open(OUT / "value_survivors.jsonl", "w") as f:
        for d in dump:
            f.write(json.dumps(d) + "\n")
    print(f"paraphrase: full_correct={len(full_c)} survivors={len(survivors)} "
          f"retrieval_mediated={len(r_cell)} -> value_survivors.jsonl")

    labels_path = ROOT / "annotations" / "status_update_2" / "value_survivor_labels.csv"
    if labels_path.exists():
        with open(labels_path) as f:
            label_rows = list(csv.DictReader(f))
        missing = set(r_cell) - {r["fact"] for r in label_rows}
        extra = {r["fact"] for r in label_rows} - set(r_cell)
        if missing or extra:
            raise AssertionError(
                f"value_survivor_labels.csv does not cover the survivor set: "
                f"{len(missing)} unlabeled, {len(extra)} unknown facts")
        split = Counter(r["label"] for r in label_rows)
        found = {"surface_variants": split.get("surface_variant", 0),
                 "associative_cues": split.get("associative_cue", 0)}
        n_borderline = sum(1 for r in label_rows if r["borderline"] == "1")
        print(f"paraphrase: adjudicated split from {labels_path.name}: "
              f"{found} ({n_borderline} borderline)")
        if found != REPORTED_SPLIT:
            print(f"paraphrase: WARNING adjudicated split differs from the "
                  f"report's {REPORTED_SPLIT} -- update the report numbers")
    else:
        print(f"paraphrase: {labels_path} missing; report split "
              f"{REPORTED_SPLIT} is unverified")


# -------------------------------------------------------- stage: frequency
def stage_frequency() -> None:
    """Answer density (the report's frequency analysis).

    dens(f) = number of entries in the fact's top-500 retrieval envelope
    caught by the value predicate (answer mention), read from the materialized
    closures. Quartile-binned unaided answerability + Spearman rho for all
    three models on identical facts. Right-censored at the envelope cap
    (material only on PopQA; the share at >=450 is reported).
    """
    base = {bl: json.load(open(OUT / f"perfact_{bl}.json")) for bl in BASELINES}
    report = {}
    for ds in DATASETS:
        cohort = {r["fact"]: r for r in
                  (json.loads(l) for l in open(OUT / f"cohort_{ds}.jsonl"))}
        tar_path = one(f"{CLOSURES}/{ds}/prompts*closures.tar.gz")
        counts = {}
        with tarfile.open(tar_path, "r:gz") as tf:
            for m in tf:
                fact = Path(m.name).stem
                if not m.isfile() or fact not in cohort:
                    continue
                d = json.load(tf.extractfile(m))
                counts[fact] = sum(1 for e in d["entries"]
                                   if "value" in e["caught_by"])
        facts = sorted(counts)
        cv_sorted = sorted(counts[f] for f in facts)
        n = len(facts)
        qs = [cv_sorted[int(q * n)] for q in (0.25, 0.5, 0.75)]
        models = {
            "co-lmlm": {f: 1.0 if cohort[f]["del_off_correct"] else 0.0 for f in facts},
            **{bl: {f: 1.0 if base[bl][ds].get(f) else 0.0 for f in facts}
               for bl in BASELINES},
        }
        stats = {
            "facts": n,
            "missing_closure": len(cohort) - n,
            "quartile_edges": qs,
            "near_cap_share": rate(c >= 450 for c in cv_sorted),
        }
        for name, l in models.items():
            bins = defaultdict(list)
            for f in facts:
                bins[sum(counts[f] > q for q in qs)].append(l[f])
            stats[name] = {
                "spearman": round(spearman([counts[f] for f in facts],
                                           [l[f] for f in facts]), 3),
                "L_by_quartile": [round(sum(b) / len(b), 4) if (b := bins.get(i))
                                  else None for i in range(4)],
            }
        report[ds] = stats
        print(f"frequency: {ds} n={n} cap-share={stats['near_cap_share']:.1%} "
              + " ".join(f"{m}:rho={stats[m]['spearman']:+.2f}" for m in models))
    with open(OUT / "frequency.json", "w") as f:
        json.dump(report, f, indent=1)


# ---------------------------------------------------------- stage: figures
DS_LABEL = {"trex": "T-REx", "popqa": "PopQA", "googlere": "Google-RE",
            "counterfact": "CounterFact", "zsre": "ZsRE"}
DS_COLOR = {"trex": "#e6a817", "popqa": "#e377c2", "googlere": "#1f77b4",
            "counterfact": "#2ca02c", "zsre": "#d62728"}
C_COLMLM, C_SMOL, C_STD, C_PROBE = "#1f77b4", "#8c8c8c", "#c4c4c4", "#d95f02"
SMOL, STD = BASELINES


def stage_figures(figdir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    figdir.mkdir(parents=True, exist_ok=True)
    NUM = json.load(open(OUT / "numbers.json"))
    GRP = json.load(open(OUT / "probe_grouped.json"))
    FREQ = json.load(open(OUT / "frequency.json"))
    pct = lambda x: 100.0 * x

    # -- baselines: L on identical audited facts, three models
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    x = np.arange(len(DATASETS))
    w = 0.26
    col = [pct(NUM["datasets"][d]["over_full_correct"]["L"]) for d in DATASETS]
    smol = [pct(NUM["datasets"][d]["baselines"][SMOL]["rate_on_full_correct"])
            for d in DATASETS]
    std = [pct(NUM["datasets"][d]["baselines"][STD]["rate_on_full_correct"])
           for d in DATASETS]
    ax.bar(x - w, col, w, color=C_COLMLM, label="Co-LMLM $L$ (retrieval off)")
    ax.bar(x, smol, w, color=C_SMOL, label="SmolLM2-360M (closed-book)")
    ax.bar(x + w, std, w, color=C_STD, edgecolor="#999999", linewidth=0.4,
           label="Standard-LM-360M-FW (closed-book)")
    for xi, vals in zip(x, zip(col, smol, std)):
        for off, v in zip((-w, 0, w), vals):
            ax.text(xi + off, v + 0.8, f"{v:.1f}", ha="center", fontsize=6.5)
    ax.set_xticks(x, [DS_LABEL[d] for d in DATASETS])
    ax.set_ylabel("Facts answered (%)")
    ax.legend(fontsize=7.5, frameon=False, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, 1.18))
    ax.set_ylim(0, 55)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(figdir / "baselines.pdf")
    plt.close(fig)

    # -- frequency: answer-density quartile curves, three models
    fig, axes = plt.subplots(1, 5, figsize=(7.2, 2.3), sharey=True)
    q = [1, 2, 3, 4]
    styles = [("co-lmlm", C_COLMLM, "o", "-", "Co-LMLM $L$"),
              (SMOL, C_SMOL, "^", "-", "SmolLM2-360M"),
              (STD, "#b0b0b0", "v", "--", "Standard-LM-360M-FW")]
    for ax, d in zip(axes, DATASETS):
        s = FREQ[d]
        for key, color, mk, ls, _ in styles:
            ax.plot(q, [pct(v) for v in s[key]["L_by_quartile"]],
                    marker=mk, color=color, linestyle=ls, markersize=3.5)
        rhos = " / ".join(f"{s[k]['spearman']:+.2f}" for k, *_ in styles)
        ax.set_title(f"{DS_LABEL[d]}\n$\\rho_s$: {rhos}", fontsize=7)
        ax.set_xticks(q, ["Q1", "Q2", "Q3", "Q4"], fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
    axes[0].set_ylabel("Answered unaided (%)", fontsize=8)
    axes[2].set_xlabel(
        "Answer-density quartile (answer-bearing entries in top-500 envelope)",
        fontsize=8)
    handles = [Line2D([], [], color=c, marker=mk, linestyle=ls, markersize=4,
                      label=lab) for _, c, mk, ls, lab in styles]
    fig.legend(handles=handles, fontsize=7.5, ncol=3, frameon=False,
               loc="upper center", bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(figdir / "frequency.pdf", bbox_inches="tight")
    plt.close(fig)

    # -- entanglement: operating curves + G(f) shares, baselines as references
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.9),
                                   gridspec_kw={"width_ratios": [1.25, 1]})
    for d in DATASETS:
        curve = NUM["entanglement"][d]["curve"]
        rhos = sorted(curve.keys(), key=float, reverse=True)
        xs = [max(pct(curve[r]["collateral"]), 0.04) for r in rhos]
        ys = [pct(curve[r]["efficacy"]) for r in rhos]
        ax1.plot(xs, ys, "o-", color=DS_COLOR[d], markersize=3.5, label=DS_LABEL[d])
    ax1.set_xscale("log")
    ax1.set_xlabel("Collateral $X$ among neighbors (%, log scale)")
    ax1.set_ylabel("Efficacy $E$ (%)")
    ax1.legend(fontsize=6.5, frameon=False, loc="lower right")
    ax1.set_title("Deletion operating curves ($\\rho$: 0.95 $\\to$ 0.70)", fontsize=8)
    x = np.arange(len(DATASETS))
    g0 = [pct(NUM["entanglement"][d]["share_gap_zero"]) for d in DATASETS]
    g1 = [pct(NUM["entanglement"][d]["share_gap_one"]) for d in DATASETS]
    ax2.bar(x, g0, 0.55, color=[DS_COLOR[d] for d in DATASETS], alpha=0.85)
    ax2.bar(x, g1, 0.55, bottom=[100 - v for v in g1], color="black", alpha=0.55)
    for xi, v in zip(x, g0):
        ax2.text(xi, v - 6, f"{v:.0f}", ha="center", fontsize=7, color="white")
    for bl, mk in [(SMOL, "^"), (STD, "v")]:
        ax2.plot(x, [pct(NUM["datasets"][d]["baselines"][bl]["rate_on_full_correct"])
                     for d in DATASETS],
                 marker=mk, color="black", markersize=5, fillstyle="none",
                 linestyle="none")
    ax2.set_xticks(x, [DS_LABEL[d] for d in DATASETS], fontsize=7, rotation=20)
    ax2.set_ylabel("Share of audited facts (%)")
    ax2.set_ylim(0, 100)
    ax2.set_title("Entanglement gap $G(f)$", fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "entanglement.pdf")
    plt.close(fig)

    # -- probe: behavioral vs. probe with prompt-only controls, three models
    # (proposition-grouped folds; see stage probe-grouped)
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    x = np.arange(len(DATASETS))
    w = 0.2
    smol = [pct(NUM["datasets"][d]["baselines"][SMOL]["rate_on_cohort"])
            for d in DATASETS]
    std = [pct(NUM["datasets"][d]["baselines"][STD]["rate_on_cohort"])
           for d in DATASETS]
    beh = [pct(GRP[d]["l_hat"]) for d in DATASETS]
    prb = [pct(GRP[d]["l_rep_hat"]) for d in DATASETS]
    ax.bar(x - 1.5 * w, smol, w, color=C_SMOL, label="SmolLM2-360M (closed-book)")
    ax.bar(x - 0.5 * w, std, w, color=C_STD, edgecolor="#999999", linewidth=0.4,
           label="Standard-LM-360M (closed-book)")
    ax.bar(x + 0.5 * w, beh, w, color=C_COLMLM, label="Co-LMLM behavioral $L$")
    ax.bar(x + 1.5 * w, prb, w, color=C_PROBE,
           label="Co-LMLM probe $L_{\\mathrm{rep}}$")
    for i, d in enumerate(DATASETS):
        ng = pct(GRP[d]["prompt_ngram"]["l_rep_hat"])
        em = pct(GRP[d]["prompt_embed"]["l_rep_hat"])
        pr = pct(GRP[d]["majority_gold_share"])
        for v, mk in [(ng, "_"), (em, "x"), (pr, ".")]:
            ax.plot([i + 1.5 * w], [v], marker=mk, color="black", markersize=6,
                    linestyle="none", zorder=5)
        ax.text(i + 1.5 * w, max(prb[i], ng, em) + 1.6, f"{prb[i]:.1f}",
                ha="center", fontsize=7)
        ax.text(i + 0.5 * w, beh[i] + 1.2, f"{beh[i]:.1f}", ha="center", fontsize=7)
    handles, labels_ = ax.get_legend_handles_labels()
    handles += [Line2D([], [], marker="_", color="black", linestyle="none"),
                Line2D([], [], marker="x", color="black", linestyle="none"),
                Line2D([], [], marker=".", color="black", linestyle="none")]
    labels_ += ["prompt $n$-gram probe", "prompt mean-embedding probe",
                "gold-answer prior"]
    ax.set_xticks(x, [DS_LABEL[d] for d in DATASETS])
    ax.set_ylabel("Answer recovered (%)")
    ax.legend(handles, labels_, fontsize=7, ncol=2, frameon=False, loc="upper right")
    ax.set_ylim(0, 70)
    fig.tight_layout()
    fig.savefig(figdir / "probe.pdf")
    plt.close(fig)

    # -- adversarial (Table 3 in the report): print, no plot
    print("figures: adversarial numbers (report Table 3), per dataset:")
    for d in DATASETS:
        rows = NUM["adversarial"].get(d, [])
        verb = {r["epsilon"]: pct(r["attack_gain_rate"])
                for r in rows if r["template"] == "verbatim"}
        ev_gain = max((pct(r["attack_gain_rate"]) for r in rows
                       if r["template"] != "verbatim"), default=None)
        ev_reg = max((pct(r["attack_regression_rate"]) for r in rows
                      if r["template"] != "verbatim"), default=None)
        print(f"  {DS_LABEL[d]}: verbatim gain "
              f"{verb.get(0.01):.1f}/{verb.get(0.02):.1f}/{verb.get(0.05):.1f} "
              f"(eps .01/.02/.05); evading max gain {ev_gain:.1f}, "
              f"max regression {ev_reg:.1f}")
    print(f"figures: wrote baselines/frequency/entanglement/probe .pdf -> {figdir}")


# ------------------------------------------------------------------- main
STAGES = {
    "extract": stage_extract,
    "policies": stage_policies,
    "probe-controls": stage_probe_controls,
    "probe-grouped": stage_probe_grouped,
    "paraphrase": stage_paraphrase,
    "frequency": stage_frequency,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", default="all",
                    choices=[*STAGES, "figures", "all"])
    ap.add_argument("--figdir", type=Path, default=OUT / "figures",
                    help="where figure PDFs go (point at the paper's figures/)")
    args = ap.parse_args()
    if args.stage == "all":
        for name, fn in STAGES.items():
            print(f"== {name} ==")
            fn()
        print("== figures ==")
        stage_figures(args.figdir)
    elif args.stage == "figures":
        stage_figures(args.figdir)
    else:
        STAGES[args.stage]()


if __name__ == "__main__":
    main()
