import csv
import json
import os
from pathlib import Path
from typing import Any

from lmlm_audit.states import DatabaseState


PROJECT_ROOT = Path(__file__).resolve().parents[2]
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
