"""Radius sweep vs backbone recitation (paper Table: sweep split by BBR).
Among audited facts (FULL-correct), how many BBR=1 facts are made incorrect
(E=1) at some swept radius. Writes results/status_update_2/sweep_vs_bbr.json."""
import csv, glob, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from halo.core.metrics import _result_is_correct
from halo.core.entanglement import fact_key

DS = ["trex", "popqa", "googlere", "counterfact", "zsre"]
out = {}
for ds in DS:
    base = f"{ROOT}/results/co-lmlm/{ds}"
    res = glob.glob(f"{base}/prompts*_results.jsonl") + glob.glob(f"{base}/prompts_results.jsonl")
    res = sorted(set(res))[0]
    full, off = {}, {}
    for line in open(res):
        r = json.loads(line)
        k = fact_key(r)
        if r["state"] == "FULL":
            full[k] = _result_is_correct(r)
        elif r["state"] == "DEL-OFF":
            off[k] = _result_is_correct(r)
    sweep = sorted(glob.glob(f"{base}/prompts*sweep"))[0]
    ever = {}   # target -> E=1 at some rho
    n_rho = {}
    for r in csv.DictReader(open(f"{sweep}/entanglement_curves.csv")):
        k = r["target_key"]
        ever[k] = ever.get(k, False) or float(r["efficacy"]) >= 1.0
        n_rho[k] = n_rho.get(k, 0) + 1
    gaps = {r["target_key"]: r for r in csv.DictReader(open(f"{sweep}/entanglement_gaps.csv")) if r["gap_eligible"] == "True"}
    targets = [k for k in gaps if full.get(k)]
    bbr1 = [k for k in targets if off.get(k)]
    bbr0 = [k for k in targets if k in off and not off[k]]
    d = {
        "n_targets": len(targets),
        "ever_removed_all": sum(ever[k] for k in targets) / len(targets),
        "n_bbr1": len(bbr1),
        "ever_removed_bbr1": sum(ever[k] for k in bbr1) / max(1, len(bbr1)),
        "n_bbr0": len(bbr0),
        "ever_removed_bbr0": sum(ever[k] for k in bbr0) / max(1, len(bbr0)),
        "gap_zero_bbr1": sum(float(gaps[k]["gap"]) == 0.0 for k in bbr1) / max(1, len(bbr1)),
        "gap_zero_all": sum(float(gaps[k]["gap"]) == 0.0 for k in targets) / len(targets),
        "missing_off": sum(k not in off for k in targets),
        "rhos_per_target": sorted(set(n_rho[k] for k in targets)),
    }
    out[ds] = d
    print(ds, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()})
json.dump(out, open(ROOT / "results/status_update_2/sweep_vs_bbr.json", "w"), indent=2)
