"""Generate report-ready plots from LMLM audit metrics CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_MPL_CONFIG_DIR = Path("/tmp/lmlm-audit-matplotlib")
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src" / "lmlm-audit"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import math

from metrics import metrics_total


# Two-sided z critical values for the alpha levels we expose; we avoid pulling
# in scipy or implementing a full inverse-normal and instead raise on
# unsupported alphas so a typo cannot silently misreport a CI.
NORMAL_QUANTILES = {
    0.10: 1.6448536269514722,
    0.05: 1.959963984540054,
    0.01: 2.5758293035489004,
}


def normal_quantile(alpha: float) -> float:
    if alpha not in NORMAL_QUANTILES:
        raise ValueError(
            f"Unsupported alpha {alpha}; choose from {sorted(NORMAL_QUANTILES)}"
        )
    return NORMAL_QUANTILES[alpha]


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def chi2_p_value_1df(stat: float) -> float:
    if stat <= 0.0:
        return 1.0
    return 2.0 * (1.0 - normal_cdf(math.sqrt(stat)))


def wilson_ci(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    z = normal_quantile(alpha)
    p = successes / n
    z_sq = z * z
    denom = 1.0 + z_sq / n
    center = (p + z_sq / (2.0 * n)) / denom
    half = (z * math.sqrt(p * (1.0 - p) / n + z_sq / (4.0 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def proportion_report(
    successes: int, n: int, alpha: float = 0.05
) -> dict[str, float]:
    rate = successes / n if n > 0 else 0.0
    low, high = wilson_ci(successes, n, alpha=alpha)
    return {
        "successes": int(successes),
        "n": int(n),
        "rate": rate,
        "ci_low": low,
        "ci_high": high,
        "alpha": alpha,
    }


def mcnemar_chi2(b: int, c: int, continuity: bool = True) -> dict[str, float]:
    if b < 0 or c < 0:
        raise ValueError(f"Discordant counts must be nonnegative: b={b}, c={c}")
    n_disc = b + c
    if n_disc == 0:
        return {
            "test": "mcnemar_chi2",
            "statistic": 0.0,
            "p_value": 1.0,
            "b": int(b),
            "c": int(c),
            "continuity": continuity,
        }
    diff = abs(b - c)
    adjusted = max(0.0, diff - 1.0) if continuity else float(diff)
    stat = (adjusted * adjusted) / n_disc
    return {
        "test": "mcnemar_chi2",
        "statistic": stat,
        "p_value": chi2_p_value_1df(stat),
        "b": int(b),
        "c": int(c),
        "continuity": continuity,
    }


def mcnemar_exact(b: int, c: int) -> dict[str, float]:
    if b < 0 or c < 0:
        raise ValueError(f"Discordant counts must be nonnegative: b={b}, c={c}")
    n_disc = b + c
    if n_disc == 0:
        return {
            "test": "mcnemar_exact",
            "statistic": 0.0,
            "p_value": 1.0,
            "b": int(b),
            "c": int(c),
        }
    k = min(b, c)
    cumulative = sum(math.comb(n_disc, i) for i in range(k + 1))
    p_value = min(1.0, 2.0 * cumulative / (1 << n_disc))
    return {
        "test": "mcnemar_exact",
        "statistic": float(k),
        "p_value": p_value,
        "b": int(b),
        "c": int(c),
    }


def mcnemar_report(
    b: int, c: int, exact_threshold: int = 25, continuity: bool = True
) -> dict[str, Any]:
    chi2 = mcnemar_chi2(b, c, continuity=continuity)
    exact = mcnemar_exact(b, c)
    use_exact = (b + c) <= exact_threshold
    return {
        "chi2": chi2,
        "exact": exact,
        "recommended": exact if use_exact else chi2,
        "exact_threshold": exact_threshold,
    }


def format_p_value(p: float) -> str:
    if p < 1e-3:
        return "< 0.001"
    if p < 1e-2:
        return f"{p:.4f}"
    return f"{p:.3f}"


WANDB_PROJECT = "lmlm-audit-analysis"

STATE_ORDER = ["FULL", "DEL-ON", "DEL-OFF"]
STATE_LABELS = {
    "FULL": "FULL",
    "DEL-ON": "DEL-ON",
    "DEL-OFF": "DEL-OFF",
}
STATE_COLORS = {
    "FULL": "#2563eb",
    "DEL-ON": "#16a34a",
    "DEL-OFF": "#dc2626",
}
VARIANT_ORDER = ["base", "alias", "collision", "noise"]
VARIANT_LABELS = {
    "base": "Base",
    "alias": "Alias",
    "collision": "Collision",
    "noise": "Noise",
    "released_lmlm": "Released LMLM",
}
PROMPT_ORDER = [
    "direct_questions",
    "contextual_questions",
    "paraphrased_questions",
    "cloze",
    "continuations",
    "fewshot_questions",
]
PROMPT_LABELS = {
    "direct_questions": "Direct",
    "contextual_questions": "Contextual",
    "paraphrased_questions": "Paraphrased",
    "cloze": "Cloze",
    "continuations": "Continuations",
    "fewshot_questions": "Few-shot",
}
METRIC_LABELS = {
    "precision": "Precision",
    "recall": "Recall",
    "f1": "Token F1",
    "parametric_leakage": "Parametric leakage",
    "retrieval_artifact_rate": "Retrieval artifact rate",
}
DECOMPOSITION_COLORS = {
    "parametric_leakage": "#dc2626",
    "retrieval_artifact_rate": "#f59e0b",
}
CUSTOM_DOMAINS = {"countries", "politicians", "sports"}
CUSTOM_VARIANTS = {"base", "alias", "collision", "noise"}
RELEASED_DOMAIN = "released_database"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_released_lmlm_metrics(paths: list[Path]) -> dict[str, float] | None:
    existing_paths = [path for path in paths if path.exists()]
    if not existing_paths:
        return None

    results: list[dict[str, object]] = []
    for path in existing_paths:
        results.extend(read_jsonl(path))
    if not results:
        return None
    return metrics_total(results)


def add_derived_columns(row: dict[str, str]) -> dict[str, str]:
    prompt_path = Path(row["prompt_file"])
    database_path = Path(row["database_path"])
    row = dict(row)
    row["domain"] = database_path.parent.name
    row["variant"] = database_path.stem
    row["prompt_type"] = prompt_path.stem.removeprefix("prompts_")
    return row


def is_custom_row(row: dict[str, str]) -> bool:
    return row["domain"] in CUSTOM_DOMAINS and row["variant"] in CUSTOM_VARIANTS


def is_released_row(row: dict[str, str]) -> bool:
    return row["domain"] == RELEASED_DOMAIN


def filter_custom(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if is_custom_row(row)]


def filter_released(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if is_released_row(row)]


def numeric(row: dict[str, str], column: str) -> float:
    return float(row[column])


def weight(row: dict[str, str]) -> float:
    return numeric(row, "count")


def weighted_average(rows: list[dict[str, str]], metric: str) -> float:
    total_weight = sum(weight(row) for row in rows)
    if total_weight == 0:
        return 0.0
    return sum(numeric(row, metric) * weight(row) for row in rows) / total_weight


def released_metrics_from_cross_state(
    cross_state_rows: list[dict[str, str]],
) -> dict[str, float] | None:
    released_rows = filter_released(cross_state_rows)
    if not released_rows:
        return None
    metric_cols = [
        "precision",
        "recall",
        "f1",
        "parametric_leakage",
        "retrieval_mediated_correctness",
        "retrieval_artifact_rate",
    ]
    return {metric: weighted_average(released_rows, metric) for metric in metric_cols}


def released_per_state_metrics(
    per_state_rows: list[dict[str, str]],
) -> dict[str, dict[str, float]] | None:
    released_rows = filter_released(per_state_rows)
    if not released_rows:
        return None
    metric_cols = ["exact_match", "precision", "recall", "f1"]
    grouped = group_rows(released_rows, ("state",))
    return {
        key[0]: {metric: weighted_average(rows, metric) for metric in metric_cols}
        for key, rows in grouped.items()
    }


def group_rows(
    rows: list[dict[str, str]],
    keys: tuple[str, ...],
) -> dict[tuple[str, ...], list[dict[str, str]]]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    return dict(grouped)


def ordered_values(values: set[str], preferred_order: list[str]) -> list[str]:
    known = [value for value in preferred_order if value in values]
    extra = sorted(value for value in values if value not in preferred_order)
    return known + extra


def pretty_prompt(prompt_type: str) -> str:
    return PROMPT_LABELS.get(prompt_type, prompt_type.replace("_", " ").title())


def pretty_variant(variant: str) -> str:
    return VARIANT_LABELS.get(variant, variant.replace("_", " ").title())


def percent_axis(ax: plt.Axes, upper: float = 1.0) -> None:
    ax.set_ylim(0, upper)
    ticks = np.linspace(0, upper, 6)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{int(round(value * 100))}%" for value in ticks])
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)


def save_figure(
    fig: plt.Figure, output_dir: Path, stem: str, tight: bool = True
) -> None:
    if tight:
        fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_cross_state_diagnostic_by_variant(
    cross_state_rows: list[dict[str, str]],
    output_dir: Path,
    metric: str,
    color: str,
    stem: str,
    released_lmlm_metrics: dict[str, float] | None = None,
) -> None:
    variants = ordered_values(
        {row["variant"] for row in cross_state_rows}, VARIANT_ORDER
    )
    grouped = group_rows(cross_state_rows, ("variant",))
    labels = [pretty_variant(variant) for variant in variants]
    values = [
        weighted_average(grouped.get((variant,), []), metric) for variant in variants
    ]

    if released_lmlm_metrics is not None:
        labels.append(pretty_variant("released_lmlm"))
        values.append(released_lmlm_metrics[metric])

    max_value = max(values) if values else 0.0
    upper = max(0.02, min(1.0, np.ceil((max_value + 0.01) * 20) / 20))

    x = np.arange(len(labels))
    colors = [color] * len(labels)
    if released_lmlm_metrics is not None:
        colors[-1] = "#64748b"

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    bars = ax.bar(x, values, color=colors, width=0.62)
    ax.set_title(f"Average {METRIC_LABELS[metric]} across databases")
    ax.set_ylabel("Weighted rate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    percent_axis(ax, upper=upper)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + upper * 0.02,
            f"{value * 100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    save_figure(fig, output_dir, stem)


def default_released_result_paths(output_dir: Path) -> list[Path]:
    return sorted(output_dir.glob("prompts_*_results.jsonl"))


def plot_exact_match_by_variant_state(
    per_state_rows: list[dict[str, str]],
    output_dir: Path,
    released_per_state: dict[str, dict[str, float]] | None = None,
) -> None:
    variants = ordered_values({row["variant"] for row in per_state_rows}, VARIANT_ORDER)
    grouped = group_rows(per_state_rows, ("variant", "state"))

    labels = [pretty_variant(variant) for variant in variants]
    if released_per_state is not None:
        labels.append(pretty_variant("released_lmlm"))

    x = np.arange(len(labels))
    width = 0.26

    fig, ax = plt.subplots(figsize=(9.8, 5.4))
    for idx, state in enumerate(STATE_ORDER):
        values = [
            weighted_average(grouped.get((variant, state), []), "exact_match")
            for variant in variants
        ]
        if released_per_state is not None:
            values.append(released_per_state.get(state, {}).get("exact_match", 0.0))
        ax.bar(
            x + (idx - 1) * width,
            values,
            width,
            color=STATE_COLORS[state],
            label=STATE_LABELS[state],
        )

    ax.set_title("Exact match by variant and intervention state")
    ax.set_ylabel("Weighted exact match")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    percent_axis(ax)
    ax.legend(frameon=False, loc="upper right")
    save_figure(fig, output_dir, "exact_match_by_variant_state")


def plot_del_on_attribution_by_variant(
    cross_state_rows: list[dict[str, str]],
    output_dir: Path,
    released_lmlm_metrics: dict[str, float] | None = None,
) -> None:
    variants = ordered_values(
        {row["variant"] for row in cross_state_rows}, VARIANT_ORDER
    )
    grouped = group_rows(cross_state_rows, ("variant",))

    leakage = [
        weighted_average(grouped.get((variant,), []), "parametric_leakage")
        for variant in variants
    ]
    rmc = [
        weighted_average(grouped.get((variant,), []), "retrieval_mediated_correctness")
        for variant in variants
    ]
    artifact = [
        weighted_average(grouped.get((variant,), []), "retrieval_artifact_rate")
        for variant in variants
    ]
    labels = [pretty_variant(variant) for variant in variants]

    if released_lmlm_metrics is not None:
        labels.append(pretty_variant("released_lmlm"))
        leakage.append(released_lmlm_metrics.get("parametric_leakage", 0.0))
        rmc.append(released_lmlm_metrics.get("retrieval_mediated_correctness", 0.0))
        artifact.append(released_lmlm_metrics.get("retrieval_artifact_rate", 0.0))

    leakage_arr = np.array(leakage)
    rmc_arr = np.array(rmc)
    artifact_arr = np.array(artifact)

    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.bar(
        x - width / 2,
        leakage_arr,
        width,
        color=DECOMPOSITION_COLORS["parametric_leakage"],
        label=r"Parametric leakage $L(f)$ ($\hat{L}$)",
    )
    ax.bar(
        x - width / 2,
        rmc_arr,
        width,
        bottom=leakage_arr,
        color="#16a34a",
        label=r"Retrieval-mediated correctness $R(f)$",
    )
    ax.bar(
        x + width / 2,
        artifact_arr,
        width,
        color=DECOMPOSITION_COLORS["retrieval_artifact_rate"],
        label="Retrieval artifact rate",
    )

    stacked = leakage_arr + rmc_arr
    for i, total in enumerate(stacked):
        if total > 0:
            ax.text(
                x[i] - width / 2,
                total + 0.005,
                f"{total * 100:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    stacked_max = float(stacked.max(initial=0.0))
    artifact_max = float(artifact_arr.max(initial=0.0))
    upper = max(
        0.05, min(1.0, np.ceil((max(stacked_max, artifact_max) + 0.02) * 20) / 20)
    )

    ax.set_title("Attribution of DEL-ON correctness by variant")
    ax.set_ylabel("Weighted rate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    percent_axis(ax, upper=upper)
    ax.legend(frameon=False, loc="upper right")
    save_figure(fig, output_dir, "del_on_correctness_attribution_by_variant")


def plot_del_on_attribution_by_prompt(
    cross_state_rows: list[dict[str, str]],
    output_dir: Path,
) -> None:
    prompts = ordered_values(
        {row["prompt_type"] for row in cross_state_rows}, PROMPT_ORDER
    )
    grouped = group_rows(cross_state_rows, ("prompt_type",))

    leakage = [
        weighted_average(grouped.get((prompt,), []), "parametric_leakage")
        for prompt in prompts
    ]
    rmc = [
        weighted_average(grouped.get((prompt,), []), "retrieval_mediated_correctness")
        for prompt in prompts
    ]
    artifact = [
        weighted_average(grouped.get((prompt,), []), "retrieval_artifact_rate")
        for prompt in prompts
    ]
    labels = [pretty_prompt(prompt) for prompt in prompts]

    leakage_arr = np.array(leakage)
    rmc_arr = np.array(rmc)
    artifact_arr = np.array(artifact)

    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.bar(
        x - width / 2,
        leakage_arr,
        width,
        color=DECOMPOSITION_COLORS["parametric_leakage"],
        label=r"Parametric leakage $L(f)$ ($\hat{L}$)",
    )
    ax.bar(
        x - width / 2,
        rmc_arr,
        width,
        bottom=leakage_arr,
        color="#16a34a",
        label=r"Retrieval-mediated correctness $R(f)$",
    )
    ax.bar(
        x + width / 2,
        artifact_arr,
        width,
        color=DECOMPOSITION_COLORS["retrieval_artifact_rate"],
        label="Retrieval artifact rate",
    )

    stacked = leakage_arr + rmc_arr
    for i, total in enumerate(stacked):
        if total > 0:
            ax.text(
                x[i] - width / 2,
                total + 0.005,
                f"{total * 100:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    stacked_max = float(stacked.max(initial=0.0))
    artifact_max = float(artifact_arr.max(initial=0.0))
    upper = max(
        0.05, min(1.0, np.ceil((max(stacked_max, artifact_max) + 0.02) * 20) / 20)
    )

    ax.set_title("Attribution of DEL-ON correctness by prompt style")
    ax.set_ylabel("Weighted rate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    percent_axis(ax, upper=upper)
    ax.legend(frameon=False, loc="upper right")
    save_figure(fig, output_dir, "del_on_correctness_attribution_by_prompt")


def plot_overlap_metrics_by_prompt_state(
    per_state_rows: list[dict[str, str]],
    output_dir: Path,
) -> None:
    prompts = ordered_values(
        {row["prompt_type"] for row in per_state_rows},
        PROMPT_ORDER,
    )
    metrics = ["precision", "recall", "f1"]
    grouped = group_rows(per_state_rows, ("prompt_type", "state"))
    x = np.arange(len(prompts))
    width = 0.24

    fig, axes = plt.subplots(
        1,
        len(metrics),
        figsize=(16, 5.2),
        sharey=True,
        constrained_layout=False,
    )
    for ax, metric in zip(axes, metrics, strict=True):
        for idx, state in enumerate(STATE_ORDER):
            values = [
                weighted_average(grouped.get((prompt, state), []), metric)
                for prompt in prompts
            ]
            ax.bar(
                x + (idx - 1) * width,
                values,
                width,
                color=STATE_COLORS[state],
                label=STATE_LABELS[state],
            )
        ax.set_title(METRIC_LABELS[metric])
        ax.set_xticks(x)
        ax.set_xticklabels(
            [pretty_prompt(prompt) for prompt in prompts], rotation=30, ha="right"
        )
        percent_axis(ax)

    axes[0].set_ylabel("Weighted rate")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.subplots_adjust(top=0.76, bottom=0.24, wspace=0.12)
    fig.suptitle(
        "Average precision, recall, and token F1 by prompt style and state", y=0.98
    )
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
    )
    save_figure(fig, output_dir, "precision_recall_f1_by_prompt_and_state", tight=False)


def plot_token_f1_by_prompt_state(
    per_state_rows: list[dict[str, str]],
    output_dir: Path,
) -> None:
    prompts = ordered_values(
        {row["prompt_type"] for row in per_state_rows},
        PROMPT_ORDER,
    )
    grouped = group_rows(per_state_rows, ("prompt_type", "state"))
    x = np.arange(len(prompts))
    width = 0.24

    fig, ax = plt.subplots(figsize=(9.8, 5.4))
    for idx, state in enumerate(STATE_ORDER):
        values = [
            weighted_average(grouped.get((prompt, state), []), "f1")
            for prompt in prompts
        ]
        ax.bar(
            x + (idx - 1) * width,
            values,
            width,
            color=STATE_COLORS[state],
            label=STATE_LABELS[state],
        )
    ax.set_title("Token F1 by prompt style and state")
    ax.set_ylabel("Weighted rate")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [pretty_prompt(prompt) for prompt in prompts], rotation=30, ha="right"
    )
    percent_axis(ax)
    ax.legend(frameon=False, loc="upper right")
    save_figure(fig, output_dir, "token_f1_by_prompt_and_state")


def overall_audit_summary(cross_state_rows: list[dict[str, str]]) -> dict[str, float]:
    """Aggregate cross-state metrics across every (prompt file, database) cell.

    Weights each cell by its paired_count, since parametric_leakage,
    retrieval_mediated_correctness, and retrieval_artifact_rate are themselves
    averaged over paired (DEL-ON, DEL-OFF) fact groups inside metrics.metrics_total.
    The returned paired_count is the global denominator that backs the
    "across N alias-closure deletions, leakage is X%" claim.
    """
    if not cross_state_rows:
        return {
            "cell_count": 0,
            "paired_count": 0,
            "parametric_leakage": 0.0,
            "retrieval_mediated_correctness": 0.0,
            "retrieval_artifact_rate": 0.0,
        }

    total_paired = sum(int(float(row["paired_count"])) for row in cross_state_rows)

    def weighted(metric: str) -> float:
        if total_paired == 0:
            return 0.0
        return sum(
            float(row[metric]) * float(row["paired_count"])
            for row in cross_state_rows
        ) / total_paired

    return {
        "cell_count": len(cross_state_rows),
        "paired_count": total_paired,
        "parametric_leakage": weighted("parametric_leakage"),
        "retrieval_mediated_correctness": weighted("retrieval_mediated_correctness"),
        "retrieval_artifact_rate": weighted("retrieval_artifact_rate"),
    }


def print_audit_summary(label: str, summary: dict[str, float]) -> None:
    print(f"\n[{label}]")
    print(f"  cells:          {summary['cell_count']}")
    print(f"  paired count:   {int(summary['paired_count']):,}")
    print(f"  L_hat (param):  {summary['parametric_leakage'] * 100:.3f}%")
    print(f"  R_hat:          {summary['retrieval_mediated_correctness'] * 100:.3f}%")
    print(f"  artifact rate:  {summary['retrieval_artifact_rate'] * 100:.3f}%")


def write_audit_summary(summaries: dict[str, dict[str, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)


STATE_PAIR = ("DEL-ON", "DEL-OFF")
HEADLINE_METRICS = (
    ("parametric_leakage", r"Parametric leakage $\hat{L}$", "#dc2626"),
    ("retrieval_mediated_correctness", r"Retrieval-mediated $\hat{R}$", "#16a34a"),
    ("retrieval_artifact_rate", "Retrieval artifact rate", "#f59e0b"),
)


def _coerced_int(value: Any) -> int:
    if value is None or value == "":
        return 0
    return int(round(float(value)))


def _cell_paired_count(cross_row: dict[str, str]) -> int:
    return _coerced_int(cross_row.get("paired_count", 0))


def _cell_exact_count(
    per_state_index: dict[tuple[str, str], dict[str, str]],
    cell_key: str,
    state: str,
    paired: int,
) -> int:
    """Number of facts correct in `state` for a given cell, scaled to paired_count."""
    state_row = per_state_index.get((cell_key, state))
    if state_row is None or paired <= 0:
        return 0
    return _coerced_int(float(state_row["exact_match"]) * paired)


def _cell_del_on_del_off_contingency(
    cross_row: dict[str, str],
    per_state_index: dict[tuple[str, str], dict[str, str]],
) -> dict[str, int]:
    """(DEL-ON, DEL-OFF) paired contingency reconstructed from aggregated rates.

    Exact when the audit ran all three states on every fact in the cell, since
    retrieval_mediated_correctness encodes b directly and parametric_leakage
    plus the per-state DEL-ON exact_match supply enough to recover a, c, d.
    """
    paired = _cell_paired_count(cross_row)
    if paired <= 0:
        return {"a": 0, "b": 0, "c": 0, "d": 0, "n": 0}

    cell_key = f"{cross_row['prompt_file']}::{cross_row['database_path']}"
    del_on_correct = _cell_exact_count(per_state_index, cell_key, "DEL-ON", paired)
    del_off_correct = _cell_exact_count(per_state_index, cell_key, "DEL-OFF", paired)

    b = _coerced_int(
        float(cross_row.get("retrieval_mediated_correctness", 0.0)) * paired
    )
    a = max(0, del_on_correct - b)
    c = max(0, del_off_correct - a)
    d = max(0, paired - a - b - c)
    return {"a": a, "b": b, "c": c, "d": d, "n": a + b + c + d}


def _aggregate_contingency(
    contingencies: list[dict[str, int]],
) -> dict[str, int]:
    aggregated = {"a": 0, "b": 0, "c": 0, "d": 0, "n": 0}
    for cell in contingencies:
        for key in aggregated:
            aggregated[key] += int(cell.get(key, 0))
    return aggregated


def _state_exact_proportion(
    per_state_rows_in_group: list[dict[str, str]],
    cross_paired_lookup: dict[str, int],
    state: str,
    alpha: float,
) -> dict[str, float]:
    total_n = 0
    total_successes = 0
    for row in per_state_rows_in_group:
        if row["state"] != state:
            continue
        cell_key = f"{row['prompt_file']}::{row['database_path']}"
        paired = cross_paired_lookup.get(cell_key, 0)
        if paired <= 0:
            continue
        total_n += paired
        total_successes += _coerced_int(float(row["exact_match"]) * paired)
    return proportion_report(total_successes, total_n, alpha=alpha)


def _cross_state_proportion(
    cross_rows_in_group: list[dict[str, str]],
    metric: str,
    alpha: float,
) -> dict[str, float]:
    total_n = 0
    total_successes = 0
    for row in cross_rows_in_group:
        paired = _cell_paired_count(row)
        if paired <= 0:
            continue
        total_n += paired
        total_successes += _coerced_int(float(row[metric]) * paired)
    return proportion_report(total_successes, total_n, alpha=alpha)


def slice_report(
    label: str,
    cross_rows_in_group: list[dict[str, str]],
    per_state_rows_in_group: list[dict[str, str]],
    per_state_index: dict[tuple[str, str], dict[str, str]],
    cross_paired_lookup: dict[str, int],
    alpha: float = 0.05,
) -> dict[str, Any]:
    proportions = {
        "parametric_leakage": _cross_state_proportion(
            cross_rows_in_group, "parametric_leakage", alpha
        ),
        "retrieval_mediated_correctness": _cross_state_proportion(
            cross_rows_in_group, "retrieval_mediated_correctness", alpha
        ),
        "retrieval_artifact_rate": _cross_state_proportion(
            cross_rows_in_group, "retrieval_artifact_rate", alpha
        ),
        "exact_match_FULL": _state_exact_proportion(
            per_state_rows_in_group, cross_paired_lookup, "FULL", alpha
        ),
        "exact_match_DEL-ON": _state_exact_proportion(
            per_state_rows_in_group, cross_paired_lookup, "DEL-ON", alpha
        ),
        "exact_match_DEL-OFF": _state_exact_proportion(
            per_state_rows_in_group, cross_paired_lookup, "DEL-OFF", alpha
        ),
    }
    contingency = _aggregate_contingency(
        [
            _cell_del_on_del_off_contingency(row, per_state_index)
            for row in cross_rows_in_group
        ]
    )
    test = mcnemar_report(contingency["b"], contingency["c"])
    return {
        "label": label,
        "cell_count": len(cross_rows_in_group),
        "paired_count": sum(_cell_paired_count(row) for row in cross_rows_in_group),
        "alpha": alpha,
        "proportions": proportions,
        "mcnemar_DEL-ON_vs_DEL-OFF": {
            "contingency": contingency,
            "chi2": test["chi2"],
            "exact": test["exact"],
            "recommended": test["recommended"],
        },
    }


def collect_slice_reports(
    raw_cross_state_rows: list[dict[str, str]],
    raw_per_state_rows: list[dict[str, str]],
    alpha: float = 0.05,
) -> dict[str, Any]:
    per_state_index = {
        (f"{row['prompt_file']}::{row['database_path']}", row["state"]): row
        for row in raw_per_state_rows
    }
    cross_paired_lookup = {
        f"{row['prompt_file']}::{row['database_path']}": _cell_paired_count(row)
        for row in raw_cross_state_rows
    }

    custom_cross = filter_custom(raw_cross_state_rows)
    custom_per_state = filter_custom(raw_per_state_rows)
    released_cross = filter_released(raw_cross_state_rows)
    released_per_state = filter_released(raw_per_state_rows)

    overall = slice_report(
        "Overall",
        raw_cross_state_rows,
        raw_per_state_rows,
        per_state_index,
        cross_paired_lookup,
        alpha,
    )
    custom = slice_report(
        "Custom",
        custom_cross,
        custom_per_state,
        per_state_index,
        cross_paired_lookup,
        alpha,
    )
    released = slice_report(
        "Released LMLM",
        released_cross,
        released_per_state,
        per_state_index,
        cross_paired_lookup,
        alpha,
    )

    by_variant: dict[str, dict[str, Any]] = {}
    for variant in ordered_values({row["variant"] for row in custom_cross}, VARIANT_ORDER):
        cross_rows = [row for row in custom_cross if row["variant"] == variant]
        per_state_rows = [row for row in custom_per_state if row["variant"] == variant]
        by_variant[variant] = slice_report(
            f"Variant: {pretty_variant(variant)}",
            cross_rows,
            per_state_rows,
            per_state_index,
            cross_paired_lookup,
            alpha,
        )
    if released_cross:
        by_variant["released_lmlm"] = slice_report(
            f"Variant: {pretty_variant('released_lmlm')}",
            released_cross,
            released_per_state,
            per_state_index,
            cross_paired_lookup,
            alpha,
        )

    by_prompt: dict[str, dict[str, Any]] = {}
    for prompt_type in ordered_values(
        {row["prompt_type"] for row in raw_cross_state_rows}, PROMPT_ORDER
    ):
        cross_rows = [
            row for row in raw_cross_state_rows if row["prompt_type"] == prompt_type
        ]
        per_state_rows = [
            row for row in raw_per_state_rows if row["prompt_type"] == prompt_type
        ]
        by_prompt[prompt_type] = slice_report(
            f"Prompt: {pretty_prompt(prompt_type)}",
            cross_rows,
            per_state_rows,
            per_state_index,
            cross_paired_lookup,
            alpha,
        )

    return {
        "alpha": alpha,
        "overall": overall,
        "custom": custom,
        "released": released,
        "by_variant": by_variant,
        "by_prompt": by_prompt,
    }


def _proportion_to_arrays(
    reports: list[dict[str, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (rates, lower_errs, upper_errs) suitable for ax.errorbar yerr.

    Clamps the lower-error component to zero so that floating-point noise in
    the Wilson computation cannot produce a slightly-negative yerr value.
    """
    rates = np.array([r["rate"] for r in reports])
    lows = np.array([r["ci_low"] for r in reports])
    highs = np.array([r["ci_high"] for r in reports])
    return rates, np.maximum(rates - lows, 0.0), np.maximum(highs - rates, 0.0)


def plot_headline_rates_ci(
    grouped_reports: dict[str, dict[str, Any]],
    label_resolver,
    output_dir: Path,
    stem: str,
    title_suffix: str,
) -> None:
    """Bar chart of L_hat, R_hat, artifact rate per group with 95% Wilson CI.

    grouped_reports is {group_key: slice_report}. label_resolver(group_key) returns
    the human-readable axis label.
    """
    if not grouped_reports:
        return
    keys = list(grouped_reports.keys())
    labels = [label_resolver(key) for key in keys]
    x = np.arange(len(labels))
    width = 0.26

    fig, ax = plt.subplots(figsize=(max(8.0, 1.6 * len(labels) + 3.0), 5.4))
    for offset, (metric, metric_label, color) in enumerate(HEADLINE_METRICS):
        proportion_reports = [
            grouped_reports[key]["proportions"][metric] for key in keys
        ]
        rates, low_err, high_err = _proportion_to_arrays(proportion_reports)
        positions = x + (offset - 1) * width
        ax.bar(
            positions,
            rates,
            width,
            color=color,
            label=metric_label,
        )
        ax.errorbar(
            positions,
            rates,
            yerr=np.vstack([low_err, high_err]),
            fmt="none",
            ecolor="#1f2937",
            elinewidth=0.9,
            capsize=3,
        )

    ax.set_title(f"Headline rates with 95% Wilson CI {title_suffix}")
    ax.set_ylabel("Weighted rate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    upper_candidates = []
    for metric, _, _ in HEADLINE_METRICS:
        for key in keys:
            upper_candidates.append(grouped_reports[key]["proportions"][metric]["ci_high"])
    upper_value = max(upper_candidates) if upper_candidates else 0.05
    upper = max(0.05, min(1.0, np.ceil((upper_value + 0.02) * 20) / 20))
    percent_axis(ax, upper=upper)
    ax.legend(frameon=False, loc="upper right")
    save_figure(fig, output_dir, stem)


def plot_mcnemar_by_variant(
    by_variant: dict[str, dict[str, Any]],
    output_dir: Path,
) -> None:
    """Discordant pair counts and recommended McNemar p-values per variant.

    Pairs the b and c counts on a log-y axis (so widely-imbalanced pairs remain
    legible) and annotates each variant's column with the recommended test's
    formatted p-value.
    """
    if not by_variant:
        return
    keys = list(by_variant.keys())
    labels = [pretty_variant(key) for key in keys]
    bs = np.array(
        [by_variant[k]["mcnemar_DEL-ON_vs_DEL-OFF"]["contingency"]["b"] for k in keys],
        dtype=float,
    )
    cs = np.array(
        [by_variant[k]["mcnemar_DEL-ON_vs_DEL-OFF"]["contingency"]["c"] for k in keys],
        dtype=float,
    )
    p_values = [
        by_variant[k]["mcnemar_DEL-ON_vs_DEL-OFF"]["recommended"]["p_value"]
        for k in keys
    ]
    test_names = [
        "exact"
        if by_variant[k]["mcnemar_DEL-ON_vs_DEL-OFF"]["recommended"]["test"]
        == "mcnemar_exact"
        else "chi2"
        for k in keys
    ]

    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(max(11.0, 2.0 * len(labels) + 4.0), 6.8))
    ax.bar(
        x - width / 2,
        np.maximum(bs, 0.5),
        width,
        color="#16a34a",
        label="$b$: DEL-ON correct, DEL-OFF wrong",
    )
    ax.bar(
        x + width / 2,
        np.maximum(cs, 0.5),
        width,
        color="#dc2626",
        label="$c$: DEL-OFF correct, DEL-ON wrong",
    )

    max_count = max(float(bs.max(initial=0.0)), float(cs.max(initial=0.0)), 1.0)

    for i, (b_count, c_count, p_value, test_name) in enumerate(
        zip(bs, cs, p_values, test_names, strict=True)
    ):
        annotation = f"p {format_p_value(p_value)}\n({test_name})"
        bar_top = max(b_count, c_count, 1.0)
        ax.text(
            x[i],
            bar_top * 3.0,
            annotation,
            ha="center",
            va="bottom",
            fontsize=9,
        )
        ax.text(
            x[i] - width / 2,
            max(b_count, 0.6) * 1.08,
            f"{int(b_count)}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#065f46",
        )
        ax.text(
            x[i] + width / 2,
            max(c_count, 0.6) * 1.08,
            f"{int(c_count)}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#7f1d1d",
        )

    ax.set_yscale("log")
    ax.set_ylim(0.5, max_count * 25.0)
    ax.set_title(
        "McNemar (DEL-ON vs DEL-OFF) discordant counts by variant", pad=16
    )
    ax.set_ylabel("Discordant pair count (log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
    )
    ax.grid(axis="y", which="both", color="#e5e7eb", linewidth=0.6)
    ax.set_axisbelow(True)
    save_figure(fig, output_dir, "mcnemar_del_on_vs_del_off_by_variant")


def write_audit_statistics(
    slice_reports: dict[str, Any],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(slice_reports, f, indent=2)


def _format_proportion(label: str, report: dict[str, float]) -> str:
    n = report["n"]
    rate = report["rate"] * 100.0
    low = report["ci_low"] * 100.0
    high = report["ci_high"] * 100.0
    return (
        f"  {label:<32s} {rate:7.3f}%  [95% CI: {low:6.3f}%, {high:6.3f}%]   "
        f"(n={n})"
    )


def _format_mcnemar(entry: dict[str, Any]) -> str:
    contingency = entry["contingency"]
    recommended = entry["recommended"]
    test_name = "exact" if recommended["test"] == "mcnemar_exact" else "chi2"
    if test_name == "exact":
        stat_str = f"k={int(recommended['statistic'])}"
    else:
        stat_str = f"chi2={recommended['statistic']:.3f}"
    p_str = format_p_value(recommended["p_value"])
    return (
        f"  DEL-ON vs DEL-OFF       b={contingency['b']:>5d}  c={contingency['c']:>5d}  "
        f"{stat_str:<14s} p {p_str}  ({test_name})"
    )


def print_slice_report(report: dict[str, Any]) -> None:
    print(
        f"\n[{report['label']}]  cells={report['cell_count']}  "
        f"paired_count={report['paired_count']:,}  alpha={report['alpha']}"
    )
    print(" Proportions (Wilson 95% CI):")
    for key in (
        "parametric_leakage",
        "retrieval_mediated_correctness",
        "retrieval_artifact_rate",
        "exact_match_FULL",
        "exact_match_DEL-ON",
        "exact_match_DEL-OFF",
    ):
        print(_format_proportion(key, report["proportions"][key]))
    print(" McNemar test:")
    print(_format_mcnemar(report["mcnemar_DEL-ON_vs_DEL-OFF"]))


def init_wandb_run(project: str) -> Any:
    from dotenv import load_dotenv

    env_path = PROJECT_ROOT / ".env"
    load_dotenv(env_path, override=True)

    api_key = os.getenv("WANDB_API_KEY")
    if not api_key:
        raise RuntimeError(f"WANDB_API_KEY was not found after loading {env_path}.")

    import wandb

    wandb.login(key=api_key, relogin=True)
    return wandb.init(project=project, name="audit_plots", reinit="finish_previous")


def log_plots_to_wandb(run: Any, output_dir: Path) -> None:
    import wandb

    images = {
        path.stem: wandb.Image(str(path)) for path in sorted(output_dir.glob("*.png"))
    }
    if images:
        run.log(images)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--per-state",
        type=Path,
        default=Path("outputs/audit/per_state_metrics.csv"),
        help="Path to per-state metrics CSV.",
    )
    parser.add_argument(
        "--cross-state",
        type=Path,
        default=Path("outputs/audit/cross_state_metrics.csv"),
        help="Path to cross-state metrics CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/audit/plots"),
        help="Directory for generated plots.",
    )
    parser.add_argument(
        "--released-result-files",
        type=Path,
        nargs="*",
        default=None,
        help=(
            "Optional raw result JSONL files generated from "
            "`data/released_database/lmlm_database.json` and its prompts. "
            "Defaults to `outputs/audit/prompts_*_results.jsonl`."
        ),
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help=f"Log generated plots to the `{WANDB_PROJECT}` wandb project.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        choices=[0.10, 0.05, 0.01],
        help="Significance level for Wilson confidence intervals.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_per_state_rows = [add_derived_columns(row) for row in read_csv(args.per_state)]
    raw_cross_state_rows = [
        add_derived_columns(row) for row in read_csv(args.cross_state)
    ]
    per_state_rows = filter_custom(raw_per_state_rows)
    cross_state_rows = filter_custom(raw_cross_state_rows)

    released_result_paths = (
        args.released_result_files
        if args.released_result_files is not None
        else default_released_result_paths(args.cross_state.parent)
    )
    released_lmlm_metrics = read_released_lmlm_metrics(released_result_paths)
    if released_lmlm_metrics is None:
        released_lmlm_metrics = released_metrics_from_cross_state(raw_cross_state_rows)
    released_per_state = released_per_state_metrics(raw_per_state_rows)

    plot_exact_match_by_variant_state(
        per_state_rows,
        args.output_dir,
        released_per_state=released_per_state,
    )
    plot_del_on_attribution_by_variant(
        cross_state_rows,
        args.output_dir,
        released_lmlm_metrics=released_lmlm_metrics,
    )
    plot_del_on_attribution_by_prompt(cross_state_rows, args.output_dir)
    plot_cross_state_diagnostic_by_variant(
        cross_state_rows,
        args.output_dir,
        metric="parametric_leakage",
        color=DECOMPOSITION_COLORS["parametric_leakage"],
        stem="average_parametric_leakage_by_domain_variant",
        released_lmlm_metrics=released_lmlm_metrics,
    )
    plot_cross_state_diagnostic_by_variant(
        cross_state_rows,
        args.output_dir,
        metric="retrieval_artifact_rate",
        color=DECOMPOSITION_COLORS["retrieval_artifact_rate"],
        stem="average_retrieval_artifact_rate_by_domain_variant",
        released_lmlm_metrics=released_lmlm_metrics,
    )
    plot_overlap_metrics_by_prompt_state(per_state_rows, args.output_dir)
    plot_token_f1_by_prompt_state(per_state_rows, args.output_dir)

    print(f"Wrote plots to {args.output_dir}")

    overall_summary = overall_audit_summary(raw_cross_state_rows)
    custom_summary = overall_audit_summary(filter_custom(raw_cross_state_rows))
    released_summary = overall_audit_summary(filter_released(raw_cross_state_rows))

    print_audit_summary("Overall (all databases)", overall_summary)
    print_audit_summary("Custom databases", custom_summary)
    print_audit_summary("Released LMLM database", released_summary)

    summary_path = args.output_dir / "audit_summary.json"
    write_audit_summary(
        {
            "overall": overall_summary,
            "custom": custom_summary,
            "released": released_summary,
        },
        summary_path,
    )
    print(f"Wrote summary to {summary_path}")

    slice_reports = collect_slice_reports(
        raw_cross_state_rows, raw_per_state_rows, alpha=args.alpha
    )

    plot_headline_rates_ci(
        slice_reports["by_variant"],
        pretty_variant,
        args.output_dir,
        stem="headline_rates_ci_by_variant",
        title_suffix="(by database variant)",
    )
    plot_headline_rates_ci(
        slice_reports["by_prompt"],
        pretty_prompt,
        args.output_dir,
        stem="headline_rates_ci_by_prompt",
        title_suffix="(by prompt style)",
    )
    plot_mcnemar_by_variant(slice_reports["by_variant"], args.output_dir)

    statistics_path = args.output_dir / "audit_statistics.json"
    write_audit_statistics(slice_reports, statistics_path)
    print(f"Wrote statistics to {statistics_path}")

    print_slice_report(slice_reports["overall"])
    print_slice_report(slice_reports["custom"])
    print_slice_report(slice_reports["released"])
    for variant_report in slice_reports["by_variant"].values():
        print_slice_report(variant_report)
    for prompt_report in slice_reports["by_prompt"].values():
        print_slice_report(prompt_report)

    if args.wandb:
        run = init_wandb_run(WANDB_PROJECT)
        try:
            log_plots_to_wandb(run, args.output_dir)
            run.summary.update(
                {
                    "overall/paired_count": overall_summary["paired_count"],
                    "overall/parametric_leakage": overall_summary["parametric_leakage"],
                    "overall/retrieval_mediated_correctness": overall_summary[
                        "retrieval_mediated_correctness"
                    ],
                    "overall/retrieval_artifact_rate": overall_summary[
                        "retrieval_artifact_rate"
                    ],
                    "custom/paired_count": custom_summary["paired_count"],
                    "custom/parametric_leakage": custom_summary["parametric_leakage"],
                    "released/paired_count": released_summary["paired_count"],
                    "released/parametric_leakage": released_summary["parametric_leakage"],
                }
            )
        finally:
            run.finish()
        print(f"Logged plots to wandb project '{WANDB_PROJECT}'.")


if __name__ == "__main__":
    main()
