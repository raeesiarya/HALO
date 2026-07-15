import csv
import json
import os
from pathlib import Path
from typing import Any

from lmlm_audit.core.states import DatabaseState


PROJECT_ROOT = Path(__file__).resolve().parents[3]
WANDB_PROJECT = "lmlm-audit"


def write_metrics_csvs(
    cross_state_rows: list[dict[str, Any]],
    per_state_rows: list[dict[str, Any]],
    cross_state_path: Path,
    per_state_path: Path,
) -> None:
    cross_state_path.parent.mkdir(parents=True, exist_ok=True)
    per_state_path.parent.mkdir(parents=True, exist_ok=True)

    if cross_state_rows:
        with cross_state_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(cross_state_rows[0].keys()))
            writer.writeheader()
            writer.writerows(cross_state_rows)

    if per_state_rows:
        with per_state_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_state_rows[0].keys()))
            writer.writeheader()
            writer.writerows(per_state_rows)


def write_entanglement_outputs(
    entanglement: dict[str, dict[str, Any]],
    output_dir: Path,
) -> dict[str, Path]:
    if not entanglement:
        return {}
    output_dir.mkdir(parents=True, exist_ok=True)
    curves_path = output_dir / "entanglement_curves.csv"
    gaps_path = output_dir / "entanglement_gaps.csv"
    figure_path = output_dir / "entanglement.png"

    curve_rows = [
        {"target_key": target_key, **point}
        for target_key, summary in sorted(entanglement.items())
        for point in summary["curve"]
    ]
    gap_rows = [
        {
            "target_key": target_key,
            "gap": summary["gap"],
            "gap_rho": summary["gap_rho"],
        }
        for target_key, summary in sorted(entanglement.items())
    ]
    with curves_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(curve_rows[0].keys()))
        writer.writeheader()
        writer.writerows(curve_rows)
    with gaps_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(gap_rows[0].keys()))
        writer.writeheader()
        writer.writerows(gap_rows)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (curve_axis, gap_axis) = plt.subplots(1, 2, figsize=(11, 4.5))
    for target_key, summary in sorted(entanglement.items()):
        xs = [point["collateral"] for point in summary["curve"]]
        ys = [point["efficacy"] for point in summary["curve"]]
        curve_axis.plot(xs, ys, marker="o", alpha=0.6, label=target_key)
    curve_axis.set_xlabel("Collateral X(f, ρ)")
    curve_axis.set_ylabel("Efficacy E(f, ρ)")
    curve_axis.set_title("Deletion operating curves")
    curve_axis.set_xlim(-0.05, 1.05)
    curve_axis.set_ylim(-0.05, 1.05)
    if len(entanglement) <= 10:
        curve_axis.legend(fontsize="small")

    gaps = [summary["gap"] for summary in entanglement.values()]
    gap_axis.hist(gaps, bins=min(20, max(5, len(gaps))), edgecolor="black")
    gap_axis.set_xlabel("Entanglement gap G(f)")
    gap_axis.set_ylabel("Facts")
    gap_axis.set_title("G(f) distribution")

    fig.tight_layout()
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)

    return {
        "curves": curves_path,
        "gaps": gaps_path,
        "figure": figure_path,
    }


def save_results(results: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False))
            f.write("\n")


class AuditLogger:
    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.log_path.open("a", encoding="utf-8")

    def print(self, *values: Any, sep: str = " ", end: str = "\n") -> None:
        message = sep.join(str(value) for value in values)
        print(message, end=end)
        self._handle.write(message)
        self._handle.write(end)
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


def setup_wandb() -> Any:
    from dotenv import load_dotenv

    env_path = PROJECT_ROOT / ".env"
    load_dotenv(env_path, override=True)

    api_key = os.getenv("WANDB_API_KEY")
    if not api_key:
        raise RuntimeError(f"WANDB_API_KEY was not found after loading {env_path}.")

    import wandb

    wandb.login(key=api_key, relogin=True)
    return wandb


def log_metrics_to_wandb(
    wandb_module: Any,
    prompt_path: Path,
    state: DatabaseState,
    state_metrics: dict[str, float | int],
    cross_state_metrics: dict[str, float | int],
    model_name: str,
    database_path: Path,
    max_new_tokens: int,
    limit: int | None,
) -> None:
    prompt_label = str(prompt_path.with_suffix("")).replace("/", "__")
    run_name = f"{prompt_label}_{state.value}"
    run = wandb_module.init(
        project=WANDB_PROJECT,
        name=run_name,
        config={
            "prompt_file": str(prompt_path),
            "state": state.value,
            "model_name": model_name,
            "database_path": str(database_path),
            "max_new_tokens": max_new_tokens,
            "limit": limit,
        },
        reinit="finish_previous",
    )
    metrics_payload = {
        **{f"state/{key}": value for key, value in state_metrics.items()},
        **{f"cross_state/{key}": value for key, value in cross_state_metrics.items()},
    }
    run.log(metrics_payload)
    run.summary.update(metrics_payload)
    run.finish()
