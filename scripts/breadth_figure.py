"""Deletion-breadth figure (fig:breadth): target removal E and collateral X.

Writes two figures with the same layout: ``breadth`` (T-REx, Google-RE,
CounterFact; main text) and ``breadth_appendix`` (PopQA, ZsRE).

Left panel: Co-LMLM over the geometric radius rho (0.95 -> tau = 0.70), read
from each dataset's ``entanglement_curves.csv``. Right panel: NULLs over the
source-closure breadth k in {1, 2, 4, 8, 16}. Solid lines are E, dashed lines
are X, both in percent of F, with Wilson 95% bands.

NULLs has no collateral pass of its own; X is reconstructed exactly from the
runs that exist, using the backend's routing rule
(src/models/nulls_wiki/backend.py):

- A control fact f' whose gold source is outside the deletion set C_k(f)
  keeps its own sink, so its output equals its FULL output. Controls are
  FULL-correct, so it is not broken.
- If gold(f') is inside C_k(f), DEL-ON activates the first title in gold(f')'s
  title-space ranking that is not in C_k(f). That ranking (top 5, gold
  excluded) is the routing list of f''s provenance DEL-ON run. Greedy decoding
  with a fixed prompt and a fixed active sink is deterministic, so the outcome
  is read off whichever existing DEL-ON run of f' activated the same sink.
  Pairs with no such run are counted as unresolved and left out of X.

N(f) is the set of FULL-correct cohort facts, other than f, whose gold
source lies in C_16(f) and differs from gold(f). It is fixed across k, like
Co-LMLM's neighbor set, and like Co-LMLM's it contains only facts the model
answered correctly with memory intact (the Co-LMLM sweep draws neighbors from
the FULL-correct families only, so both systems use the same definition). X
is the share of N(f) that breaks under the target's deletion, so it is, like
E, a rate over known facts; it averages over targets with a non-empty N(f)
and is drawn only where at least ``MIN_X_TARGETS`` targets have one. Pairs
whose DEL-ON outcome cannot be resolved from an existing run are left out of
both numerator and denominator. Facts that share f's own source are left
out: removing the rest of the requested source is NULLs' intended behaviour,
not collateral.

The Wilson band for X treats the per-target mean as a proportion over the
number of targets.

Usage:
  uv run python scripts/breadth_figure.py
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from halo.core.metrics import _result_is_correct
from paper_figures import DATASETS, DS_COLOR, DS_LABEL, INK2, one, setup_style, ygrid

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RHOS = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70]
KS = [1, 2, 4, 8, 16]
MIN_X_TARGETS = 30
# One marker shape per prompt set, shared by both panels and the legend, so
# the lines stay distinguishable in grayscale and for color-blind readers.
DS_MARKER = {"trex": "o", "googlere": "s", "counterfact": "^", "popqa": "D", "zsre": "v"}
Z = 1.959963984540054


def wilson(p: float, n: int) -> tuple[float, float]:
    if n == 0:
        return math.nan, math.nan
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return c - h, c + h


def point(values: list[float]) -> dict[str, float]:
    n = len(values)
    mean = sum(values) / n if n else math.nan
    lo, hi = wilson(mean, n)
    return {"mean": 100 * mean, "lo": 100 * lo, "hi": 100 * hi, "n": n}


def read_jsonl(path: str):
    with open(path) as handle:
        for line in handle:
            yield json.loads(line)


# ------------------------------------------------------------------ Co-LMLM
def colmlm_curves(ds: str) -> dict[float, dict[str, dict[str, float]]]:
    path = one(str(RESULTS / "co-lmlm" / ds / "prompts*_sweep" / "entanglement_curves.csv"))
    eff: dict[float, list[float]] = defaultdict(list)
    col: dict[float, list[float]] = defaultdict(list)
    for row in csv.DictReader(open(path)):
        rho = round(float(row["rho"]), 2)
        eff[rho].append(float(row["efficacy"]))
        # Co-LMLM neighbors are FULL-correct by construction (the sweep
        # builds them from the FULL-correct families), so this equals the
        # neighbor_count > 0 filter on every existing curves file.
        if int(row["neighbors_full_correct"]) > 0:
            col[rho].append(float(row["collateral"]))
    return {rho: {"E": point(eff[rho]), "X": point(col[rho])} for rho in RHOS}


# -------------------------------------------------------------------- NULLs
def _active(row: dict) -> str | None:
    retained = row["retrieval_trace"].get("retained_candidates") or []
    return retained[0]["source_id"] if retained else None


def nulls_curves(ds: str) -> tuple[dict[int, dict[str, dict[str, float]]], dict[str, int]]:
    base = RESULTS / "nulls-wiki-1b" / ds
    prefix = "" if ds == "popqa" else f"{ds}_"
    stem = f"prompts_{prefix}nulls_gated"

    gold = {r["fact_id"]: r["source_title"] for r in read_jsonl(str(base / "prep" / f"{stem}.jsonl"))}
    closure: dict[int, dict[str, set[str]]] = {
        k: {
            r["fact_id"]: set(r["deletion_manifest"]["source_ids"])
            for r in read_jsonl(str(base / "prep" / "source_closures" / f"{stem}_geometric_k{k}.jsonl"))
        }
        for k in KS
    }

    # FULL correctness and the unconstrained routing list from the standard run.
    full_ok: dict[str, bool] = {}
    routing: dict[str, list[str]] = {}
    # (fact, active sink) -> DEL-ON correctness, pooled over every run.
    outcome: dict[tuple[str, str], bool] = {}

    # The standard run is required: it holds FULL and the routing lists. The
    # other runs only add (fact, sink) outcomes, so a run missing locally
    # shrinks coverage (reported as unresolved pairs) but biases nothing.
    runs = [one(str(base / f"{stem}_results.jsonl"))]
    runs += [str(base / "source_sweep" / f"k{k}" / f"{stem}_geometric_k{k}_results.jsonl") for k in KS]
    runs += sorted(glob.glob(str(base / "policy_matrix" / "*" / f"{stem}_*_results.jsonl")))
    for path in runs:
        if not Path(path).is_file():
            print(f"note: {path} not found locally; skipped")
            continue
        for row in read_jsonl(path):
            fid, state = row["fact_id"], row["state"]
            if state == "FULL":
                full_ok.setdefault(fid, _result_is_correct(row))
            elif state == "DEL-ON":
                ok = _result_is_correct(row)
                active = _active(row)
                if active is not None:
                    outcome.setdefault((fid, active), ok)
                if path == runs[0]:
                    routing[fid] = [c["source_id"] for c in row["retrieval_trace"]["retained_candidates"]]

    by_gold: dict[str, list[str]] = defaultdict(list)
    for fid, title in gold.items():
        by_gold[title].append(fid)

    targets = [fid for fid in gold if full_ok.get(fid)]
    stats = {"targets": len(targets), "pairs": 0, "unresolved": 0, "targets_with_any_neighbor": 0}
    # E(f, k) = 1 - C_DEL-ON(f), so its mean over F_full is 1 - S from the
    # sweep's own metrics file; this needs no per-fact output.
    efficacy = {}
    for k in KS:
        row = next(csv.DictReader(open(base / "source_sweep" / f"k{k}" / "cross_state_metrics.csv")))
        n = int(row["full_correct_paired_count"])
        e = 1 - float(row["post_deletion_survival_given_full"])
        lo, hi = wilson(e, n)
        efficacy[k] = {"mean": 100 * e, "lo": 100 * lo, "hi": 100 * hi, "n": n}
    col: dict[int, list[float]] = defaultdict(list)
    for fid in targets:
        widest = closure[16][fid]
        candidates = [
            other
            for title in widest
            if title != gold[fid]
            for other in by_gold.get(title, [])
            if other != fid
        ]
        stats["targets_with_any_neighbor"] += bool(candidates)
        # N(f): FULL-correct controls only. A control the model never answered
        # cannot be lost, and counting it would scale X by FULL accuracy.
        neighbors = [other for other in candidates if full_ok.get(other)]
        for k in KS:
            if not neighbors:
                continue
            deleted = closure[k][fid]
            broken = resolved = 0
            for other in neighbors:
                if gold[other] not in deleted:
                    # Controls outside C_k(f) keep their own sink and so
                    # their FULL output: known and not lost.
                    resolved += 1
                    continue
                sink = next((t for t in routing.get(other, []) if t not in deleted), None)
                ok = outcome.get((other, sink)) if sink is not None else None
                if k == 16:
                    stats["pairs"] += 1
                    stats["unresolved"] += ok is None
                if ok is None:
                    continue
                resolved += 1
                broken += not ok
            if resolved:
                col[k].append(broken / resolved)
    stats["targets_with_neighbors"] = len(col[16])
    return {k: {"E": efficacy[k], "X": point(col[k])} for k in KS}, stats


# ------------------------------------------------------------------- figure
# The main text reports T-REx, Google-RE and CounterFact; PopQA and ZsRE go to
# the appendix, matching the table split.
FIGURES = {
    "breadth": ["trex", "googlere", "counterfact"],
    "breadth_appendix": ["popqa", "zsre"],
}


def draw(ax, xs, curves, keys, datasets) -> None:
    for ds in datasets:
        color = DS_COLOR[ds]
        for metric, style in (("E", "-"), ("X", "--")):
            if metric == "X" and min(curves[ds][k]["X"]["n"] for k in keys) < MIN_X_TARGETS:
                continue
            ys = [curves[ds][k][metric]["mean"] for k in keys]
            lo = [curves[ds][k][metric]["lo"] for k in keys]
            hi = [curves[ds][k][metric]["hi"] for k in keys]
            ax.fill_between(xs, lo, hi, color=color, alpha=0.12, linewidth=0)
            ax.plot(xs, ys, style, color=color, linewidth=1.2, marker=DS_MARKER[ds], markersize=3)
    ax.set_ylim(0, 100)
    ygrid(ax)


def render(colmlm, nulls, datasets, out: Path) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(5.5, 2.3), sharey=True)
    draw(left, RHOS, colmlm, RHOS, datasets)
    left.set_xlim(0.955, 0.695)
    left.set_xticks(RHOS)
    left.set_xlabel(r"cosine threshold $\rho$ (broader $\rightarrow$)")
    left.set_ylabel(r"% of $\mathcal{F}$")
    left.set_title("Co-LMLM", fontsize=8, color=INK2)

    draw(right, KS, nulls, KS, datasets)
    right.set_xscale("log", base=2)
    right.set_xticks(KS)
    right.set_xticklabels([str(k) for k in KS])
    right.set_xlabel(r"sources removed $k$ (broader $\rightarrow$)")
    right.set_title("NULLs", fontsize=8, color=INK2)

    handles = [
        Line2D([], [], color=DS_COLOR[ds], linewidth=1.2, marker=DS_MARKER[ds], markersize=3, label=DS_LABEL[ds])
        for ds in datasets
    ]
    handles += [
        Line2D([], [], color=INK2, linestyle="-", linewidth=1.2, label="$E$ (target removed)"),
        Line2D([], [], color=INK2, linestyle="--", linewidth=1.2, label="$X$ (collateral)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 1.08), fontsize=6.5, handlelength=1.8, columnspacing=1.0)
    fig.tight_layout()

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out.with_suffix(".png"), dpi=250, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "paper" / "figures")
    args = parser.parse_args()

    colmlm = {ds: colmlm_curves(ds) for ds in DATASETS}
    nulls, nulls_stats = {}, {}
    for ds in DATASETS:
        nulls[ds], nulls_stats[ds] = nulls_curves(ds)

    setup_style()
    for name, datasets in FIGURES.items():
        render(colmlm, nulls, datasets, args.out_dir / name)

    data = {
        "co-lmlm": {ds: {str(k): v for k, v in c.items()} for ds, c in colmlm.items()},
        "nulls": {ds: {str(k): v for k, v in c.items()} for ds, c in nulls.items()},
        "nulls_collateral_coverage": nulls_stats,
    }
    (args.out_dir / "breadth.json").write_text(json.dumps(data, indent=2))
    for ds in DATASETS:
        s = nulls_stats[ds]
        print(f"NULLs {ds}: {s['targets']} targets, {s['targets_with_any_neighbor']} with any neighbor, "
              f"{s['targets_with_neighbors']} with FULL-correct neighbors, "
              f"{s['unresolved']}/{s['pairs']} in-closure pairs unresolved at k=16")


if __name__ == "__main__":
    main()
