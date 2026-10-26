"""NULLs-specific analyses (Appendix app:nulls): source attribution and
sibling re-activation. CPU only; reads existing audit outputs.

Source attribution. The verification gate scored every mappable fact under
the gold mask and under a placebo mask. delta = sink_on - placebo log-prob.
Acc_attr is the share with delta > 0 (Wilson 95%), tested against symmetry
around zero with a Wilcoxon signed-rank test; the median of delta carries a
percentile-bootstrap 95% interval. The sh(f) bins use the value-policy
closures, which exist only for gate-passing facts, so inside a bin delta is
conditioned on delta > 0 and Acc_attr is undefined. The binned table is
therefore the median of delta among gated facts and is labelled as such.

Sibling re-activation. Upstream holds seq ids as int32 and multiplies by
2**16 in that type, so a mask depends on the id only through id mod 2**16.
We count collision classes over the released title_to_index, compute every
one of the 2**16 possible masks with the vendored mask function to get the
number of distinct masks, and verify mask equality on sampled sibling pairs.
Ev_sib = C_FULL by identity, so Gain_sib is the share of F_full that DEL-ON
made incorrect; it is read off the existing standard run.

Writes ``nulls_attribution`` (figure), ``nulls_specific.json`` and
``nulls_specific_tables.tex`` to the output directory.

Usage:
  uv run python scripts/nulls_specific_analyses.py
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy import stats

from halo.core.metrics import _result_is_correct
from models.nulls_wiki.vendor.masking import batch_seqtied_mask_mult
from paper_figures import DATASETS, DS_COLOR, DS_LABEL, INK2, setup_style, ygrid

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "nulls-wiki-1b"
Z = 1.959963984540054
BASE = 2**16
SH_BINS = [(0, 9), (10, 49), (50, 199), (200, 500)]
BOOT = 2000
SEED = 0


def read_jsonl(path: Path):
    with open(path) as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return math.nan, math.nan
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def boot_median(x: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    if len(x) < 2:
        return math.nan, math.nan
    idx = rng.integers(0, len(x), size=(BOOT, len(x)))
    meds = np.median(x[idx], axis=1)
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def stem(ds: str) -> str:
    return "prompts_nulls_gated" if ds == "popqa" else f"prompts_{ds}_nulls_gated"


# -------------------------------------------------------- source attribution
def attribution(ds: str, rng: np.random.Generator) -> dict:
    base = RESULTS / ds
    rows = [r for r in read_jsonl(base / "prep" / "gate.jsonl") if r.get("margin") is not None]
    delta = np.array([r["margin"] for r in rows], dtype=np.float64)
    n, k = len(delta), int((delta > 0).sum())
    lo, hi = wilson(k, n)
    mlo, mhi = boot_median(delta, rng)
    out = {
        "n": n,
        "acc_attr": k / n,
        "acc_attr_lo": lo,
        "acc_attr_hi": hi,
        "median_delta": float(np.median(delta)),
        "median_delta_lo": mlo,
        "median_delta_hi": mhi,
        "wilcoxon_p": float(stats.wilcoxon(delta).pvalue),
        "delta": delta,
    }

    # sh(f): other sources in the value-policy envelope that mention the answer.
    # Available for gate-passing facts only.
    closure = base / "prep" / "policy_closures" / f"{stem(ds)}_value_k0.jsonl"
    margin = {r["fact_id"]: r["margin"] for r in rows}
    sh = {r["fact_id"]: len(set(r["deletion_manifest"]["source_ids"]) - {r["source_title"]}) for r in read_jsonl(closure)}
    bins = []
    for lo_b, hi_b in SH_BINS:
        vals = np.array([margin[f] for f, s in sh.items() if lo_b <= s <= hi_b and f in margin])
        blo, bhi = boot_median(vals, rng)
        bins.append({"lo": lo_b, "hi": hi_b, "n": len(vals), "median_delta": float(np.median(vals)) if len(vals) else math.nan, "ci_lo": blo, "ci_hi": bhi})
    out["sh_bins_gated"] = bins
    sh_arr = np.array([sh[f] for f in sh if f in margin])
    m_arr = np.array([margin[f] for f in sh if f in margin])
    rho = stats.spearmanr(sh_arr, m_arr)
    out["sh_spearman_gated"] = {"rho": float(rho.statistic), "p": float(rho.pvalue), "n": len(sh_arr)}
    return out


def fig_attribution(attr: dict[str, dict], out: Path) -> None:
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(5.5, 1.5), sharey=True)
    edges = np.linspace(-3, 3, 61)
    for ax, ds in zip(axes, DATASETS):
        d = np.clip(attr[ds]["delta"], -3, 3)
        weights = np.full(len(d), 100.0 / len(d))
        ax.hist(d, bins=edges, weights=weights, color=DS_COLOR[ds], linewidth=0)
        ax.axvline(0, color=INK2, linewidth=0.6)
        a = attr[ds]
        ax.set_title(f"{DS_LABEL[ds]}\n" + rf"$\mathrm{{Acc}}_{{\mathrm{{attr}}}}={100 * a['acc_attr']:.1f}\%$", fontsize=6.5, color=INK2)
        ax.set_xlim(-3, 3)
        ax.set_xticks([-2, 0, 2])
        ygrid(ax)
    axes[0].set_ylabel(r"% of $\mathcal{F}_{\mathrm{map}}$")
    fig.supxlabel(r"$\delta(f)$: gold mask minus placebo mask, answer log-likelihood (nats)", fontsize=7, color=INK2, y=0.02)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out.with_suffix(".png"), dpi=250, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# ---------------------------------------------------- sibling re-activation
def masks(seq_ids: np.ndarray, n_pool: int, p_mem: float) -> np.ndarray:
    ids = torch.from_numpy(seq_ids.astype(np.int32)).unsqueeze(0)
    return batch_seqtied_mask_mult(ids, n_pool, p_mem)[0].numpy()


def sibling(title_to_index: dict[str, int], vocab: int, n_pool: int, p_mem: float, rng: np.random.Generator) -> tuple[dict, np.ndarray]:
    index = np.fromiter(title_to_index.values(), dtype=np.int64, count=len(title_to_index))
    seq = index + vocab
    residue = seq % BASE
    class_size = np.bincount(residue, minlength=BASE)
    siblings = class_size[residue] - 1

    # Every possible mask: one per residue. Hash rows to count distinct masks.
    seen: set[bytes] = set()
    sizes = np.zeros(BASE, dtype=np.int64)
    for start in range(0, BASE, 4096):
        block = masks(np.arange(start, min(start + 4096, BASE)), n_pool, p_mem)
        sizes[start : start + len(block)] = block.sum(1)
        seen.update(np.packbits(row).tobytes() for row in block)
    occupied = class_size > 0

    # Bit-for-bit check on real ids: a source against one sibling and one non-sibling.
    order = np.argsort(residue, kind="stable")
    sample = rng.choice(len(seq), size=2000, replace=False)
    sib_equal, other_overlap = [], []
    starts = np.concatenate([[0], np.cumsum(class_size)])
    for i in sample:
        members = order[starts[residue[i]] : starts[residue[i] + 1]]
        j = int(members[0] if members[0] != i else members[1])
        o = int(rng.integers(len(seq)))
        m = masks(np.array([seq[i], seq[j], seq[o]]), n_pool, p_mem)
        sib_equal.append(bool((m[0] == m[1]).all()))
        if residue[o] != residue[i] and m[0].sum():
            other_overlap.append((m[0] & m[2]).sum() / m[0].sum())

    full_pool = int((sizes[occupied] == n_pool).sum())
    out = {
        "n_sources": int(len(seq)),
        "vocab_size": vocab,
        "n_pool": n_pool,
        "p_mem": p_mem,
        "n_distinct_masks": len(seen),
        "n_occupied_classes": int(occupied.sum()),
        "siblings_min": int(siblings.min()),
        "siblings_median": float(np.median(siblings)),
        "siblings_max": int(siblings.max()),
        "share_with_sibling": float((siblings >= 1).mean()),
        "mask_size_mean": float(sizes[occupied].mean()),
        "mask_size_sd": float(sizes[occupied].std()),
        "classes_activating_whole_pool": full_pool,
        "sources_activating_whole_pool": int(class_size[sizes == n_pool].sum()),
        "verified_sibling_pairs": len(sib_equal),
        "verified_sibling_pairs_identical": int(sum(sib_equal)),
        "non_sibling_overlap_mean": float(np.mean(other_overlap)),
    }
    return out, siblings


def sibling_audit(ds: str, title_to_index: dict[str, int], vocab: int, siblings: np.ndarray, position: dict[str, int]) -> dict:
    by_fact: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in read_jsonl(RESULTS / ds / f"{stem(ds)}_results.jsonl"):
        by_fact[r["fact_id"]][r["state"]] = r
    full = [s for s in by_fact.values() if "FULL" in s and "DEL-ON" in s and _result_is_correct(s["FULL"])]
    n = len(full)
    flipped = sum(not _result_is_correct(s["DEL-ON"]) for s in full)
    lo, hi = wilson(flipped, n)

    sib = np.array([siblings[position[s["FULL"]["generation_metadata"]["active_sink_title"]]] for s in full])
    # A routed neighbor that is itself a sibling of gold would silently run the gold mask.
    res = lambda t: (title_to_index[t] + vocab) % BASE
    routed_sibling = sum(
        res(s["DEL-ON"]["generation_metadata"]["active_sink_title"]) == res(s["FULL"]["generation_metadata"]["active_sink_title"])
        for s in full
        if s["DEL-ON"]["generation_metadata"].get("active_sink_title")
    )
    return {
        "n_full": n,
        "made_incorrect": flipped,
        "made_incorrect_rate": flipped / n if n else math.nan,
        "made_incorrect_lo": lo,
        "made_incorrect_hi": hi,
        "gain_sib_star": 1.0,
        "reg_sib_star": 0.0,
        "share_with_sibling": float((sib >= 1).mean()) if n else math.nan,
        "siblings_min": int(sib.min()) if n else None,
        "siblings_median": float(np.median(sib)) if n else None,
        "del_on_routed_to_sibling": int(routed_sibling),
    }


# ------------------------------------------ gold sink and placebo channels
def rate(k: int, n: int) -> dict:
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": k / n if n else math.nan, "lo": lo, "hi": hi}


def mcnemar(n10: int, n01: int) -> float:
    return float(stats.binomtest(n10, n10 + n01, 0.5).pvalue) if n10 + n01 else math.nan


def channels(ds: str) -> dict:
    def load(path: Path) -> dict[str, dict[str, bool]]:
        out: dict[str, dict[str, bool]] = defaultdict(dict)
        for r in read_jsonl(path):
            out[r["fact_id"]][r["state"]] = _result_is_correct(r)
        return out

    std = load(RESULTS / ds / f"{stem(ds)}_results.jsonl")
    plc = load(RESULTS / ds / "del_off_sensitivity" / "placebo-sink" / f"{stem(ds)}_results.jsonl")
    facts = [f for f, s in std.items() if {"FULL", "DEL-ON", "DEL-OFF"} <= s.keys() and "DEL-OFF" in plc.get(f, {})]

    # Effect of the gold sink, over F.
    gain = sum(std[f]["FULL"] and not std[f]["DEL-OFF"] for f in facts)
    loss = sum(not std[f]["FULL"] and std[f]["DEL-OFF"] for f in facts)
    own = {"gain": rate(gain, len(facts)), "loss": rate(loss, len(facts)), "mcnemar_p": mcnemar(gain, loss)}

    # Placebo-calibrated channels, over F_full.
    full = [f for f in facts if std[f]["FULL"]]
    nb = [std[f]["DEL-ON"] for f in full]
    pl = [plc[f]["DEL-OFF"] for f in full]
    off = [std[f]["DEL-OFF"] for f in full]
    n = len(full)
    n10 = sum(a and not b for a, b in zip(nb, pl))
    n01 = sum(b and not a for a, b in zip(nb, pl))
    cal = {
        "R": rate(sum(a and not c for a, c in zip(nb, off)), n),
        "R_plc": rate(sum(b and not c for b, c in zip(pl, off)), n),
        "I": rate(sum(c and not a for a, c in zip(nb, off)), n),
        "I_plc": rate(sum(c and not b for b, c in zip(pl, off)), n),
        "mcnemar_p": mcnemar(n10, n01),
    }
    return {"own": own, "placebo": cal}


# -------------------------------------------------------------------- tables
def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def fmt_p(p: float) -> str:
    if p < 1e-3:
        e = int(math.floor(math.log10(p)))
        return rf"${p / 10**e:.1f}\times10^{{{e}}}$"
    return f"${p:.3f}$"


def longtable(caption: str, label: str, spec: str, header: str, rows: list[str]) -> list[str]:
    """Same longtable skeleton as the paper's other tables."""
    ncol = sum(c in "lrc" for c in spec)
    head = [r"\toprule", header + r" \\", r"\midrule"] if header else [r"\toprule"]
    return [
        r"\begingroup", r"\footnotesize", r"\setlength{\LTcapwidth}{\textwidth}",
        r"\setlength{\LTleft}{\fill}", r"\setlength{\LTright}{\fill}",
        rf"\begin{{longtable}}{{{spec}}}", rf"\caption{{{caption}}}", rf"\label{{{label}}}\\",
        *head, r"\endfirsthead", rf"\caption[]{{Table~\ref{{{label}}} (continued).}}\\",
        *head, r"\endhead", r"\midrule",
        rf"\multicolumn{{{ncol}}}{{r}}{{\emph{{continued on next page}}}}\\", r"\endfoot",
        r"\bottomrule", r"\endlastfoot", *rows, r"\end{longtable}", r"\endgroup", "",
    ]


def tables(attr: dict, chan: dict) -> str:
    num = lambda v: f"${v:,}$".replace(",", "{,}")
    L = [r"% Generated by scripts/nulls_specific_analyses.py. Do not edit by hand.", ""]

    rows = [
        f"{DS_LABEL[ds]} & {num(attr[ds]['n'])} & "
        rf"${pct(attr[ds]['acc_attr'])}$\,{{\scriptsize$[{pct(attr[ds]['acc_attr_lo'])},{pct(attr[ds]['acc_attr_hi'])}]$}} & "
        rf"${attr[ds]['median_delta']:.3f}$\,{{\scriptsize$[{attr[ds]['median_delta_lo']:.3f},{attr[ds]['median_delta_hi']:.3f}]$}} & "
        f"{fmt_p(attr[ds]['wilcoxon_p'])}" + r" \\"
        for ds in DATASETS
    ]
    L += longtable(
        r"Source attribution over $\mathcal{F}_{\mathrm{map}}$. $\mathrm{Acc}_{\mathrm{attr}}$ is in percent with Wilson $95\%$ intervals, the median of $\delta$ is in nats with percentile-bootstrap $95\%$ intervals, and $p$ is a two-sided Wilcoxon signed-rank test of symmetry around zero.",
        "tab:nulls-attr", "l rrrr", r" & $n$ & $\mathrm{Acc}_{\mathrm{attr}}$ & median $\delta$ & $p$", rows)

    rows = []
    for ds in DATASETS:
        if rows:
            rows.append(r"\addlinespace[2pt]")
        for i, (lo_b, hi_b) in enumerate(SH_BINS):
            b = attr[ds]["sh_bins_gated"][i]
            cell = "--" if b["n"] < 30 else rf"${b['median_delta']:.2f}$\,{{\scriptsize$[{b['ci_lo']:.2f},{b['ci_hi']:.2f}]$}}"
            rows.append(f"{DS_LABEL[ds] if i == 0 else ''} & ${lo_b}$--${hi_b}$ & {num(b['n'])} & {cell}" + r" \\")
    L += longtable(
        r"Median of $\delta$ in nats by $\mathrm{sh}(f)$, with percentile-bootstrap $95\%$ intervals. $\mathrm{sh}(f)$ is available only for facts that pass the verification gate, so every cell is conditioned on $\delta>0$.",
        "tab:nulls-attr-sh", "l l rr", r" & $\mathrm{sh}(f)$ & $n$ & median $\delta$", rows)

    ci = lambda r: rf"${pct(r['rate'])}$\,{{\scriptsize$[{pct(r['lo'])},{pct(r['hi'])}]$}}"
    rows = [
        f"{DS_LABEL[ds]} & {num(chan[ds]['own']['gain']['n'])} & {ci(chan[ds]['own']['gain'])} & {ci(chan[ds]['own']['loss'])} & {fmt_p(chan[ds]['own']['mcnemar_p'])}" + r" \\"
        for ds in DATASETS
    ]
    L += longtable(
        r"Effect of the gold sink over $\mathcal{F}$, in percent with Wilson $95\%$ intervals. $\mathrm{Gain}_{\mathrm{own}}$ is correct under \textsc{FULL} only and $\mathrm{Loss}_{\mathrm{own}}$ is correct under \textsc{DEL-OFF} only. $p$ is McNemar's exact test of $\mathrm{Gain}_{\mathrm{own}}=\mathrm{Loss}_{\mathrm{own}}$.",
        "tab:nulls-own", "l rrrr", r" & $n$ & $\mathrm{Gain}_{\mathrm{own}}$ & $\mathrm{Loss}_{\mathrm{own}}$ & $p$", rows)

    rows = []
    for ds in DATASETS:
        c = chan[ds]["placebo"]
        if rows:
            rows.append(r"\addlinespace[2pt]")
        rows.append(f"{DS_LABEL[ds]} & {num(c['R']['n'])} & $R$ & {ci(c['R'])} & {ci(c['R_plc'])} & {fmt_p(c['mcnemar_p'])}" + r" \\")
        rows.append(f" & & $I$ & {ci(c['I'])} & {ci(c['I_plc'])} & " + r" \\")
    L += longtable(
        r"Placebo-calibrated channels over $\mathcal{F}_{\mathrm{full}}$, in percent with Wilson $95\%$ intervals. ``Next-closest'' is the \textsc{DEL-ON} condition and ``placebo'' activates a source far from $\mathrm{gold}(f)$. Both are compared with sinks-zero \textsc{DEL-OFF}. $p$ is McNemar's exact test between the next-closest and placebo conditions.",
        "tab:nulls-placebo", "l r l rrr", r" & $n$ & & Next-closest & Placebo & $p$", rows)

    return "\n".join(L)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "paper" / "figures")
    parser.add_argument("--title-to-index", type=Path, default=ROOT / "data" / "nulls-title-to-index.pkl")
    parser.add_argument("--model-config", type=Path, default=ROOT / "data" / "nulls-wikipedia-full" / "final" / "model_config.yaml")
    args = parser.parse_args()
    rng = np.random.default_rng(SEED)

    cfg = yaml.safe_load(args.model_config.read_text())
    n_gen = int(cfg["intermediate_size"] * cfg["p_gen"])
    n_pool, p_mem, vocab = cfg["intermediate_size"] - n_gen, cfg["p_mem"], cfg["vocab_size"]
    with open(args.title_to_index, "rb") as handle:
        title_to_index = pickle.load(handle)
    position = {t: i for i, t in enumerate(title_to_index)}

    attr = {ds: attribution(ds, rng) for ds in DATASETS}
    sib, siblings = sibling(title_to_index, vocab, n_pool, p_mem, rng)
    chan = {ds: channels(ds) for ds in DATASETS}
    audit = {ds: sibling_audit(ds, title_to_index, vocab, siblings, position) for ds in DATASETS}

    setup_style()
    fig_attribution(attr, args.out_dir / "nulls_attribution")
    (args.out_dir / "nulls_specific_tables.tex").write_text(tables(attr, chan))
    for a in attr.values():
        a.pop("delta")
    (args.out_dir / "nulls_specific.json").write_text(json.dumps({"attribution": attr, "gold_sink": {ds: chan[ds]["own"] for ds in DATASETS}, "placebo_channels": {ds: chan[ds]["placebo"] for ds in DATASETS}, "sibling": sib, "sibling_audit": audit}, indent=2))

    print(json.dumps(sib, indent=1))
    for ds in DATASETS:
        a, u = attr[ds], audit[ds]
        print(f"{ds}: n={a['n']} Acc_attr={pct(a['acc_attr'])} [{pct(a['acc_attr_lo'])},{pct(a['acc_attr_hi'])}] "
              f"median={a['median_delta']:.3f} [{a['median_delta_lo']:.3f},{a['median_delta_hi']:.3f}] p={a['wilcoxon_p']:.2g} | "
              f"F_full={u['n_full']} made-incorrect={pct(u['made_incorrect_rate'])} routed-to-sibling={u['del_on_routed_to_sibling']} "
              f"| sh~delta (gated) rho={a['sh_spearman_gated']['rho']:.3f}")


if __name__ == "__main__":
    main()
