from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "status_update_2"

DATASETS = ["trex", "popqa", "googlere", "counterfact", "zsre"]
DS_LABEL = {
    "trex": "T-REx",
    "popqa": "PopQA",
    "googlere": "Google-RE",
    "counterfact": "CounterFact",
    "zsre": "ZsRE",
}
POLICIES = ["oracle", "provenance", "geometric", "value", "hybrid", "factq"]
SMOL, STD = "smollm2-360m", "standard-lm-360m-fw"

# NeurIPS text width is 5.5 in; every figure is designed at that width.
W = 5.5

# Palette: CVD-validated categorical slots (adjacent + three-slot all-pairs
# gates pass; sub-3:1 contrast slots carry direct labels as relief).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
YELLOW, MAGENTA, VIOLET = "#eda100", "#e87ba4", "#4a3aa7"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
GRAY_DARK, GRAY_LIGHT = "#6e6d69", "#c3c2b7"

POL_COLOR = {
    "oracle": YELLOW,
    "provenance": MAGENTA,
    "geometric": BLUE,
    "value": AQUA,
    "hybrid": VIOLET,
    "factq": ORANGE,
}
POL_SHORT = {
    "oracle": "oracle",
    "provenance": "provenance",
    "geometric": "geometric",
    "value": "value (oracle)",
    "hybrid": "hybrid",
    "factq": "$\\langle$FACT-q$\\rangle$",
}
DS_COLOR = dict(zip(DATASETS, [BLUE, ORANGE, AQUA, YELLOW, MAGENTA]))

pct = lambda x: 100.0 * x


def one(pattern: str) -> str:
    hits = sorted(glob.glob(pattern))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one match for {pattern}: {hits}")
    return hits[0]


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 7,
            "font.family": "sans-serif",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.6,
            "axes.labelsize": 7,
            "axes.labelcolor": INK,
            "xtick.color": INK2,
            "ytick.color": INK,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 7,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "xtick.major.pad": 2,
            "ytick.major.pad": 2,
            "axes.grid": False,
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "legend.frameon": False,
            "legend.fontsize": 6.5,
            "lines.solid_capstyle": "round",
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save(fig, figdir: Path, name: str) -> None:
    fig.savefig(figdir / f"{name}.png", dpi=250, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {name}.png")


def ygrid(ax) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def xgrid(ax) -> None:
    ax.grid(axis="x", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def top_legend(ax, handles, ncol, **kw):
    ax.legend(
        handles=handles,
        ncol=ncol,
        loc="lower left",
        bbox_to_anchor=(-0.02, 0.99),
        borderpad=0,
        borderaxespad=0,
        columnspacing=1.1,
        handletextpad=0.3,
        **kw,
    )


# ------------------------------------------------------------- 1. baselines
def fig_baselines(NUM, figdir):
    fig, ax = plt.subplots(figsize=(W, 1.75))
    y = np.arange(len(DATASETS))[::-1]
    for d, yi in zip(DATASETS, y):
        L = pct(NUM["datasets"][d]["over_full_correct"]["L"])
        smol = pct(NUM["datasets"][d]["baselines"][SMOL]["rate_on_full_correct"])
        std = pct(NUM["datasets"][d]["baselines"][STD]["rate_on_full_correct"])
        ax.plot(
            [min(L, smol, std), max(L, smol, std)],
            [yi, yi],
            color=GRID,
            linewidth=1.2,
            zorder=1,
        )
        ax.plot(smol, yi, "o", color=GRAY_DARK, markersize=5, zorder=3)
        ax.plot(
            std,
            yi,
            "o",
            markerfacecolor="white",
            markeredgecolor=GRAY_DARK,
            markeredgewidth=1.0,
            markersize=5,
            zorder=3,
        )
        ax.plot(
            L,
            yi,
            "o",
            color=BLUE,
            markersize=6,
            markeredgecolor="white",
            markeredgewidth=0.9,
            zorder=4,
        )
        ax.annotate(
            f"{L:.0f}",
            (L, yi),
            textcoords="offset points",
            xytext=(0, 5.5),
            ha="center",
            fontsize=6,
            color=BLUE,
        )
    ax.set_yticks(y, [DS_LABEL[d] for d in DATASETS])
    ax.set_ylim(-0.55, len(DATASETS) - 0.45 + 0.45)
    ax.set_xlim(0, 52)
    ax.set_xlabel("facts answered with retrieval disabled (%)")
    xgrid(ax)
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            color=BLUE,
            markeredgecolor="white",
            markersize=6,
            label="Co-LMLM (parametric leakage $L$)",
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            color=GRAY_DARK,
            markersize=5,
            label="SmolLM2-360M",
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor=GRAY_DARK,
            markersize=5,
            label="Standard-LM-360M-FW",
        ),
    ]
    top_legend(ax, handles, ncol=3)
    fig.tight_layout()
    save(fig, figdir, "baselines")


# ----------------------------------------------------------------- 2. probe
def fig_probe(GRP, figdir):
    fig, ax = plt.subplots(figsize=(W, 1.9))
    y = np.arange(len(DATASETS))[::-1]
    for d, yi in zip(DATASETS, y):
        beh = pct(GRP[d]["l_hat"])
        prb = pct(GRP[d]["l_rep_hat"])
        controls = [
            pct(GRP[d]["prompt_ngram"]["l_rep_hat"]),
            pct(GRP[d]["prompt_embed"]["l_rep_hat"]),
            pct(GRP[d]["majority_gold_share"]),
        ]
        for v in controls:
            ax.plot(
                v,
                yi,
                marker="|",
                color=MUTED,
                markersize=8,
                markeredgewidth=1.1,
                linestyle="none",
                zorder=2,
            )
        ax.plot([beh, prb], [yi, yi], color=GRID, linewidth=1.2, zorder=1)
        ax.plot(
            beh,
            yi,
            "o",
            color=BLUE,
            markersize=6,
            markeredgecolor="white",
            markeredgewidth=0.9,
            zorder=4,
        )
        ax.plot(
            prb,
            yi,
            "o",
            color=ORANGE,
            markersize=6,
            markeredgecolor="white",
            markeredgewidth=0.9,
            zorder=4,
        )
        ax.annotate(
            f"{beh:.0f}",
            (beh, yi),
            textcoords="offset points",
            xytext=(0, 5.5),
            ha="center",
            fontsize=6,
            color=BLUE,
        )
        ax.annotate(
            f"{prb:.0f}",
            (prb, yi),
            textcoords="offset points",
            xytext=(0, 5.5),
            ha="center",
            fontsize=6,
            color=ORANGE,
        )
    ax.set_yticks(y, [DS_LABEL[d] for d in DATASETS])
    ax.set_ylim(-0.55, len(DATASETS) - 0.45 + 0.45)
    ax.set_xlim(0, 52)
    ax.set_xlabel("gold answer recovered (%)")
    xgrid(ax)
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            color=BLUE,
            markeredgecolor="white",
            markersize=6,
            label="behavioral leakage $L$",
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            color=ORANGE,
            markeredgecolor="white",
            markersize=6,
            label="probe $L_{\\mathrm{rep}}$ (query embeddings)",
        ),
        Line2D(
            [],
            [],
            marker="|",
            linestyle="none",
            color=MUTED,
            markersize=7,
            markeredgewidth=1.1,
            label="prompt-only controls",
        ),
    ]
    top_legend(ax, handles, ncol=3)
    fig.tight_layout()
    save(fig, figdir, "probe")


# ------------------------------------------------------------- 3. frequency
def fig_frequency(FREQ, figdir):
    fig, axes = plt.subplots(1, 5, figsize=(W, 1.55), sharey=True)
    series = [
        ("co-lmlm", BLUE, "-", 1.4, "Co-LMLM ($L$)"),
        (SMOL, GRAY_DARK, "-", 1.1, "SmolLM2-360M"),
        (STD, GRAY_LIGHT, "--", 1.1, "Standard-LM-360M-FW"),
    ]
    q = [1, 2, 3, 4]
    for ax, d in zip(axes, DATASETS):
        s = FREQ[d]
        for key, color, ls, lw, _ in series:
            ax.plot(
                q,
                [pct(v) for v in s[key]["L_by_quartile"]],
                color=color,
                linestyle=ls,
                linewidth=lw,
                marker="o",
                markersize=2.2,
                zorder=3,
            )
        ax.set_title(DS_LABEL[d], fontsize=7, pad=3)
        for i, (key, color, *_) in enumerate(series):
            ax.text(
                0.06,
                0.95 - 0.15 * i,
                f"{s[key]['spearman']:+.2f}",
                transform=ax.transAxes,
                fontsize=5.5,
                color=color,
                va="top",
            )
        ax.set_xticks(q, ["Q1", "Q2", "Q3", "Q4"], fontsize=6)
        ax.set_xlim(0.7, 4.3)
        ax.set_ylim(0, 88)
        ygrid(ax)
    axes[0].set_ylabel("answered w/o\nretrieval (%)", fontsize=6.5)
    axes[2].set_xlabel("answer-frequency quartile (rare $\\to$ common)", fontsize=7)
    handles = [
        Line2D(
            [],
            [],
            color=c,
            linestyle=ls,
            linewidth=lw,
            marker="o",
            markersize=2.2,
            label=lab,
        )
        for _, c, ls, lw, lab in series
    ]
    fig.legend(
        handles=handles,
        ncol=3,
        loc="lower left",
        bbox_to_anchor=(0.075, 0.97),
        borderpad=0,
        columnspacing=1.1,
        handletextpad=0.4,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, figdir, "frequency")


# ---------------------------------------------------- 4. entanglement sweep
def fig_entanglement_sweep(NUM, figdir):
    fig, (ax_e, ax_x) = plt.subplots(1, 2, figsize=(W, 1.75))
    for d in DATASETS:
        curve = NUM["entanglement"][d]["curve"]
        keys = sorted(curve, key=float, reverse=True)
        rhos = [float(r) for r in keys]
        ax_e.plot(
            rhos,
            [pct(curve[r]["efficacy"]) for r in keys],
            "o-",
            color=DS_COLOR[d],
            markersize=2.5,
            linewidth=1.2,
            label=DS_LABEL[d],
        )
        ax_x.plot(
            rhos,
            [pct(curve[r]["collateral"]) for r in keys],
            "o-",
            color=DS_COLOR[d],
            markersize=2.5,
            linewidth=1.2,
        )
    for ax, ylab, ymax in (
        (ax_e, "target forgotten (%)", 105),
        (ax_x, "neighbors lost (%)", 11),
    ):
        ax.invert_xaxis()
        ax.set_ylabel(ylab)
        ax.set_ylim(0, ymax)
        ax.set_xticks([0.95, 0.85, 0.75])
        ygrid(ax)
        ax.set_xlabel("deletion radius $\\rho$", fontsize=7)
    fig.legend(
        *ax_e.get_legend_handles_labels(),
        ncol=5,
        loc="lower left",
        bbox_to_anchor=(0.06, 0.97),
        borderpad=0,
        columnspacing=1.1,
        handletextpad=0.4,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, figdir, "entanglement_sweep")


# ------------------------------------------------- 5. entanglement outcomes
def fig_entanglement_outcomes(NUM, figdir):
    fig, ax = plt.subplots(figsize=(W, 1.55))
    y = np.arange(len(DATASETS))[::-1]
    segs = [
        ("clean", AQUA, "forgotten cleanly", "white"),
        ("cost", GRID, "forgotten at a cost to neighbors", INK2),
        ("never", INK2, "never forgotten", "white"),
    ]
    for d, yi in zip(DATASETS, y):
        g0 = pct(NUM["entanglement"][d]["share_gap_zero"])
        g1 = pct(NUM["entanglement"][d]["share_gap_one"])
        vals = {"clean": g0, "cost": 100 - g0 - g1, "never": g1}
        left = 0.0
        for key, color, _, ink in segs:
            v = vals[key]
            ax.barh(
                yi, v, 0.62, left=left, color=color, edgecolor="white", linewidth=1.4
            )
            if v >= 6:
                ax.text(
                    left + v / 2,
                    yi,
                    f"{v:.0f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color=ink,
                )
            left += v
    ax.set_yticks(y, [DS_LABEL[d] for d in DATASETS])
    ax.set_xlim(0, 100)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(bottom=False, labelbottom=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c, _, _ in segs]
    ax.legend(
        handles,
        [lab for _, _, lab, _ in segs],
        ncol=3,
        loc="lower left",
        bbox_to_anchor=(-0.02, 0.99),
        borderpad=0,
        columnspacing=1.1,
        handletextpad=0.4,
    )
    fig.tight_layout()
    save(fig, figdir, "entanglement_outcomes")


# --------------------------------------------------------- 6/7. policy grid
def _policy_panels(PF, figdir, metric, name, xmax, fmt, xlabel):
    fig, axes = plt.subplots(1, 5, figsize=(W, 1.8), sharex=True)
    rows = np.arange(len(POLICIES))[::-1]
    for ax, d in zip(axes, DATASETS):
        for pol, yi in zip(POLICIES, rows):
            v = pct(PF[d][pol][metric])
            ax.barh(yi, v, 0.66, color=POL_COLOR[pol])
            ax.text(v + xmax * 0.035, yi, fmt(v), va="center", fontsize=5.5, color=INK2)
        n = PF[d]["oracle"]["n_full_correct"]
        ax.set_title(f"{DS_LABEL[d]}\n$n$={n:,}", fontsize=6.5, pad=2, linespacing=1.3)
        ax.set_xlim(0, xmax)
        ax.set_yticks(rows)
        ax.set_yticklabels([])
        ax.tick_params(labelsize=6)
        xgrid(ax)
    axes[0].set_yticklabels([POL_SHORT[p] for p in POLICIES], fontsize=6.5)
    axes[2].set_xlabel(xlabel, fontsize=7)
    fig.tight_layout()
    save(fig, figdir, name)


def fig_policies(PF, figdir):
    _policy_panels(
        PF,
        figdir,
        "R_full",
        "policies_survival",
        xmax=80,
        fmt=lambda v: f"{v:.0f}",
        xlabel="facts surviving deletion via retrieval, $R$ (%)",
    )
    _policy_panels(
        PF,
        figdir,
        "I_full",
        "policies_interference",
        xmax=25,
        fmt=lambda v: f"{v:.1f}",
        xlabel="collateral interference, $I$ (%)",
    )


# ------------------------------------------------------- 8. factq vs. value
def fig_factq_vs_value(PF, figdir):
    story = ["geometric", "factq", "value"]
    fig, ax = plt.subplots(figsize=(W, 1.75))
    y = np.arange(len(DATASETS))[::-1]
    for d, yi in zip(DATASETS, y):
        vals = {p: pct(PF[d][p]["R_full"]) for p in story}
        ax.plot(
            [min(vals.values()), max(vals.values())],
            [yi, yi],
            color=GRID,
            linewidth=1.2,
            zorder=1,
        )
        for p in story:
            ax.plot(
                vals[p],
                yi,
                "o",
                color=POL_COLOR[p],
                markersize=6,
                markeredgecolor="white",
                markeredgewidth=0.9,
                zorder=3,
            )
        ax.annotate(
            f"{vals['factq']:.0f}",
            (vals["factq"], yi),
            textcoords="offset points",
            xytext=(0, 5.5),
            ha="center",
            fontsize=6,
            color=POL_COLOR["factq"],
        )
    ax.set_yticks(y, [DS_LABEL[d] for d in DATASETS])
    ax.set_ylim(-0.55, len(DATASETS) - 0.45 + 0.45)
    ax.set_xlim(-1, 70)
    ax.set_xlabel("facts surviving deletion via retrieval, $R$ (%)")
    xgrid(ax)
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=6,
            markeredgecolor="white",
            color=POL_COLOR[p],
            label=POL_SHORT[p],
        )
        for p in story
    ]
    top_legend(ax, handles, ncol=3)
    fig.tight_layout()
    save(fig, figdir, "factq_vs_value")


# ------------------------------------------------------- 9. factq coverage
def fig_factq_coverage(figdir):
    fig, ax = plt.subplots(figsize=(W, 1.6))
    y = np.arange(len(DATASETS))[::-1]
    segs = [
        ("enriched", ORANGE, "questions generated", "white"),
        ("span_miss", GRID, "no answer span in entry", INK2),
        ("no_full_entry", INK2, "no FULL-pass entry", "white"),
    ]
    for d, yi in zip(DATASETS, y):
        st = json.load(
            open(one(str(RESULTS / "co-lmlm" / d / "prompts*_factq_stats.json")))
        )
        total = st["facts"]
        left = 0.0
        for key, color, _, ink in segs:
            share = pct(st[key] / total)
            ax.barh(
                yi,
                share,
                0.62,
                left=left,
                color=color,
                edgecolor="white",
                linewidth=1.4,
            )
            if share >= 8:
                ax.text(
                    left + share / 2,
                    yi,
                    f"{share:.0f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color=ink,
                )
            left += share
        ax.text(101.5, yi, f"$n$={total:,}", va="center", fontsize=5.5, color=MUTED)
    ax.set_yticks(y, [DS_LABEL[d] for d in DATASETS])
    ax.set_xlim(0, 112)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(bottom=False, labelbottom=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c, _, _ in segs]
    ax.legend(
        handles,
        [lab for _, _, lab, _ in segs],
        ncol=3,
        loc="lower left",
        bbox_to_anchor=(-0.02, 0.99),
        borderpad=0,
        columnspacing=1.1,
        handletextpad=0.4,
    )
    fig.tight_layout()
    save(fig, figdir, "factq_coverage")


# --------------------------------------------------------------- captions
CAPTIONS = r"""% Suggested captions for the figures_v2 set. Adjust notation macros
% (\eg $L$, $R$, $I$) to match the paper's definitions section.

\begin{figure}[t]
  \centering
  \includegraphics{figures/baselines.pdf}
  \caption{\textbf{Externalization empties the weights.} Share of audited
  facts answered with retrieval disabled, restricted to facts the intact
  Co-LMLM answers correctly. Co-LMLM's parametric leakage $L$ (blue) is far
  below the size-matched closed-book baselines (gray) on every prompt set.}
  \label{fig:baselines}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics{figures/probe.pdf}
  \caption{\textbf{Query embeddings encode more than behavior shows.} A
  ridge probe over Co-LMLM's retrieval-query embeddings recovers the gold
  answer ($L_{\mathrm{rep}}$, orange) more often than the model's own
  output does ($L$, blue), under fact-disjoint, proposition-grouped folds.
  Gray ticks mark prompt-only controls (prompt $n$-gram, prompt
  mean-embedding, and the gold-answer prior), bounding what surface
  features alone explain.}
  \label{fig:probe}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics{figures/frequency.pdf}
  \caption{\textbf{Common facts leak into the weights; rare ones do not.}
  Unaided answerability by answer-density quartile (answer-bearing entries
  in the top-500 retrieval envelope; Q1 rare $\to$ Q4 common). Numbers in
  each panel give Spearman $\rho_s$ per model. The association is present
  but consistently weaker for Co-LMLM (blue) than for the closed-book
  baselines (gray).}
  \label{fig:frequency}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics{figures/entanglement_sweep.pdf}
  \caption{\textbf{Widening the deletion radius trades collateral for
  efficacy.} Geometric deletion at cosine radius $\rho$ (right of each
  panel = wider deletion): share of target facts forgotten (left) and
  share of FULL-correct neighbors broken as collateral (right).}
  \label{fig:entanglement-sweep}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics{figures/entanglement_outcomes.pdf}
  \caption{\textbf{Most facts can be forgotten cleanly; ZsRE is the
  exception.} Best outcome per audited fact across the radius sweep:
  forgotten at some radius with zero collateral (green), forgotten only at
  a cost to neighbors (light gray), or never forgotten (dark gray). Values
  are percentages of audited facts.}
  \label{fig:entanglement-outcomes}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics{figures/policies_survival.pdf}
  \caption{\textbf{Deletion-policy matrix: survival.} Share of FULL-correct
  facts still answered through retrieval after deletion ($R$; lower is
  stronger forgetting), per deletion policy and prompt set. The value
  filter is an evaluation oracle rather than a deployable rule; hybrid
  includes it. $\langle$FACT-q$\rangle$ deletes the union of entries
  retrieved by generated questions about the fact.}
  \label{fig:policies-survival}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics{figures/policies_interference.pdf}
  \caption{\textbf{Deletion-policy matrix: collateral.} Share of
  FULL-correct facts the model still answers correctly with retrieval
  disabled, yet answers incorrectly after deletion ($I$; lower is better),
  per deletion policy and prompt set. The aggressive value and hybrid
  policies buy their low survival with an order of magnitude more
  interference.}
  \label{fig:policies-interference}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics{figures/factq_vs_value.pdf}
  \caption{\textbf{$\langle$FACT-q$\rangle$ tracks geometric deletion, not
  the value oracle.} Post-deletion survival via retrieval ($R$; further
  left = stronger forgetting) under three closures: a cosine-radius ball
  (geometric, blue), the query-ensemble rule (orange, labeled), and
  deleting every answer-bearing entry (value, green --- an evaluation
  oracle).}
  \label{fig:factq-vs-value}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics{figures/factq_coverage.pdf}
  \caption{\textbf{$\langle$FACT-q$\rangle$ reaches only a minority of
  prompts.} Questions are generated from each fact's FULL-pass retrieved
  entry; prompts with no such entry (dark gray), or where the gold answer
  span cannot be located in it (light gray), are outside the policy's
  reach. Values are percentages of all prompts; $n$ gives prompts per set.}
  \label{fig:factq-coverage}
\end{figure}
"""


# ------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description="Generate paper figures from results/status_update_2")
    ap.add_argument("--figdir", type=Path, default=ROOT / "figures")
    args = ap.parse_args()
    args.figdir.mkdir(parents=True, exist_ok=True)
    setup_style()
    NUM = json.load(open(OUT / "numbers.json"))
    GRP = json.load(open(OUT / "probe_grouped.json"))
    FREQ = json.load(open(OUT / "frequency.json"))
    PF = json.load(open(OUT / "policy_full.json"))
    fig_baselines(NUM, args.figdir)
    fig_probe(GRP, args.figdir)
    fig_frequency(FREQ, args.figdir)
    fig_entanglement_sweep(NUM, args.figdir)
    fig_entanglement_outcomes(NUM, args.figdir)
    fig_policies(PF, args.figdir)
    fig_factq_vs_value(PF, args.figdir)
    fig_factq_coverage(args.figdir)
    (args.figdir / "captions.tex").write_text(CAPTIONS)
    print(f"wrote captions.tex -> {args.figdir}")


if __name__ == "__main__":
    main()
