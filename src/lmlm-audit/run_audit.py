import argparse
import csv
import dataclasses
import html
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from tqdm import tqdm


from metrics import metrics_total
from database_states import DatabaseState, build_state_db_manager, retrieval_enabled
from equivalence import prompt_row_aliases


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPT_DIR = Path("data/prompts")
DEFAULT_CUSTOM_DATABASE_DIR = Path("data/custom_databases")
DEFAULT_RELEASED_DATABASE_DIR = Path("data/released_database")
DEFAULT_OUTPUT_DIR = Path("outputs/audit")
DEFAULT_DATABASE_PATH = Path("data/lmlm_database.json")
WANDB_PROJECT = "lmlm-audit"
LOOKUP_VALUE_PATTERN = re.compile(
    r"<\|db_entity\|>.*?<\|db_relationship\|>.*?<\|db_return\|>\s*(.*?)\s*<\|db_end\|>",
    re.DOTALL,
)
DB_MARKUP_SPAN_PATTERN = re.compile(r"<\|db_[^|]+\|>.*?<\|db_end\|>", re.DOTALL)
DB_SPECIAL_TOKEN_PATTERN = re.compile(r"<\|db_[^|]+\|>")
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")


def load_prompts(prompts_path: Path) -> list[dict[str, Any]]:
    with prompts_path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def prepare_prompt(prompt_text: str) -> str:
    return prompt_text.strip()


def clean_answer(answer_text: str) -> str:
    answer_text = answer_text.strip()

    while answer_text.lower().startswith("answer:"):
        answer_text = answer_text[len("answer:") :].strip()

    for prefix in ("the answer is ", "it is ", "it's "):
        if answer_text.lower().startswith(prefix):
            answer_text = answer_text[len(prefix) :].strip()
            break

    stop_markers = [
        "\nQuestion:",
        "\nContext:",
        "\nFact:",
        "\nPrompt:",
        "\nAnswer:",
        "\n\n",
    ]
    for marker in stop_markers:
        if marker in answer_text:
            answer_text = answer_text.split(marker, 1)[0].strip()

    answer_text = html.unescape(answer_text)
    answer_text = DB_MARKUP_SPAN_PATTERN.sub(" ", answer_text)
    answer_text = DB_SPECIAL_TOKEN_PATTERN.sub(" ", answer_text)
    answer_text = HTML_TAG_PATTERN.sub(" ", answer_text)
    answer_text = re.sub(r"\s+", " ", answer_text).strip()
    answer_text = answer_text.strip(" \t\n\r\"'`")

    # Keep the first sentence when the model starts elaborating.
    answer_text = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'])", answer_text, maxsplit=1)[
        0
    ].strip()

    return answer_text.strip(" \t\n\r\"'`,;:.")


def extract_lookup_values(raw_output: str) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()

    for match in LOOKUP_VALUE_PATTERN.findall(raw_output):
        value = clean_answer(match)
        if value and value not in seen:
            values.append(value)
            seen.add(value)

    return values


def choose_answer(
    prompt_text: str,
    processed_output: str,
    lookup_values: list[str],
) -> tuple[str, str]:
    cleaned_output = clean_answer(processed_output)
    is_fact_query = prompt_text.strip().endswith("?") or "____" in prompt_text

    if lookup_values and is_fact_query:
        return lookup_values[0], "lookup_value"

    if cleaned_output:
        return cleaned_output, "postprocessed_text"

    if lookup_values:
        return lookup_values[0], "lookup_value"

    return "", "empty"


def compute_generation_budget(
    tokenizer: Any,
    prompt_text: str,
    target_answer_tokens: int,
) -> int:
    prompt_token_count = len(tokenizer.encode(prompt_text, add_special_tokens=False))

    # LMLM uses `max_new_tokens` both as the per-step generation cap and as an
    # overall stopping budget over prompt + decoded text, so we need extra slack
    # for lookup markup before the retrieved value appears.
    return max(32, prompt_token_count + target_answer_tokens + 16)


def retrieve_lookup_value(model: Any, lookup_query: str) -> str:
    db_manager = getattr(model, "db_manager", None)
    if db_manager is None:
        return "unknown"

    try:
        return db_manager.retrieve_from_database(lookup_query)
    except Exception:
        fallback_policy = getattr(model, "fallback_policy", "top1_anyway")
        if fallback_policy == "top1_anyway":
            try:
                return db_manager.retrieve_from_database(lookup_query, threshold=-1.0)
            except Exception:
                return "unknown"
        return "unknown"


def generate_answer(
    model: Any,
    tokenizer: Any,
    prompt_text: str,
    max_new_tokens: int = 12,
    enable_dblookup: bool = True,
) -> str:
    prepared_prompt = prepare_prompt(prompt_text)
    generation_budget = compute_generation_budget(
        tokenizer=tokenizer,
        prompt_text=prepared_prompt,
        target_answer_tokens=max_new_tokens,
    )

    if enable_dblookup:
        model.eval()
        device = next(model.parameters()).device
        model.set_logits_bias(tokenizer)

        stop_token_ids = [
            tokenizer.convert_tokens_to_ids("<|db_return|>"),
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|end_of_text|>"),
        ]
        stop_token_ids = [
            token_id
            for token_id in stop_token_ids
            if token_id is not None and token_id != tokenizer.unk_token_id
        ]

        inputs = tokenizer(prepared_prompt, return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[1]

        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            logits_processor=model.logits_processor,
            max_new_tokens=generation_budget,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            return_dict_in_generate=False,
            do_sample=False,
            eos_token_id=stop_token_ids,
        )

        raw_output = model._decode_with_special_tokens(
            outputs,
            tokenizer,
            input_len,
            prepared_prompt,
        )

        if "<|db_return|>" in raw_output:
            return clean_answer(retrieve_lookup_value(model, raw_output))
    else:
        raw_output = model.generate_with_lookup(
            prompt=prepared_prompt,
            tokenizer=tokenizer,
            max_new_tokens=generation_budget,
            enable_dblookup=False,
            enable_postprocess=False,
        )

    processed_output = str(model.post_process(raw_output, tokenizer)).strip()
    lookup_values = extract_lookup_values(raw_output)
    final_output, _ = choose_answer(
        prompt_text=prompt_text,
        processed_output=processed_output,
        lookup_values=lookup_values,
    )
    return final_output


def _default_retrieval_trace(state: DatabaseState) -> dict[str, Any]:
    return {
        "state": state.value,
        "retrieval_enabled": retrieval_enabled(state),
        "lookup_query": None,
        "threshold": None,
        "all_candidates": [],
        "deleted_candidates": [],
        "retained_candidates": [],
        "selected_candidate": None,
        "selected_value": None,
        "error": None,
    }


def run_prompt_audit(
    base_db_manager: Any,
    model: Any,
    tokenizer: Any,
    prompt_row: dict[str, Any],
    state: DatabaseState,
    max_new_tokens: int = 12,
) -> dict[str, Any]:
    model.db_manager = build_state_db_manager(
        base_db_manager=base_db_manager,
        prompt_row=prompt_row,
        state=state,
    )
    if hasattr(model.db_manager, "reset_trace"):
        model.db_manager.reset_trace()

    answer = generate_answer(
        model=model,
        tokenizer=tokenizer,
        prompt_text=prompt_row["prompt_text"],
        max_new_tokens=max_new_tokens,
        enable_dblookup=retrieval_enabled(state),
    )
    retrieval_trace = getattr(model.db_manager, "last_trace", None)

    return {
        "fact_id": prompt_row["fact_id"],
        "subject": prompt_row["subject"],
        "subject_aliases": list(prompt_row_aliases(prompt_row, "subject")),
        "relation": prompt_row["relation"],
        "relation_aliases": list(prompt_row_aliases(prompt_row, "relation")),
        "state": state.value,
        "prompt": prompt_row["prompt_text"],
        "ground_truth": prompt_row["gold_object"],
        "object_aliases": list(prompt_row_aliases(prompt_row, "object")),
        "model_output": answer,
        "retrieval_trace": retrieval_trace or _default_retrieval_trace(state),
    }


def run_audit(
    prompt_path: Path,
    base_db_manager: Any,
    model: Any,
    tokenizer: Any,
    states: list[DatabaseState],
    max_new_tokens: int = 12,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    prompts = load_prompts(prompt_path)
    if limit is not None:
        prompts = prompts[:limit]

    results: list[dict[str, Any]] = []
    for prompt in tqdm(
        prompts,
        desc=f"Auditing {prompt_path.stem}",
        unit="prompt",
    ):
        for state in states:
            results.append(
                run_prompt_audit(
                    base_db_manager=base_db_manager,
                    model=model,
                    tokenizer=tokenizer,
                    prompt_row=prompt,
                    state=state,
                    max_new_tokens=max_new_tokens,
                )
            )

    return results


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


@dataclass(frozen=True)
class AuditJob:
    prompt_path: Path
    database_path: Path
    output_path: Path


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


def discover_custom_audit_jobs(output_dir: Path) -> list[AuditJob]:
    jobs: list[AuditJob] = []
    if not DEFAULT_CUSTOM_DATABASE_DIR.exists():
        return jobs
    for domain_dir in sorted(
        path for path in DEFAULT_CUSTOM_DATABASE_DIR.iterdir() if path.is_dir()
    ):
        prompts_root = domain_dir / "prompts"
        if not prompts_root.exists():
            continue

        for variant_dir in sorted(
            path for path in prompts_root.iterdir() if path.is_dir()
        ):
            database_path = domain_dir / f"{variant_dir.name}.json"
            if not database_path.exists():
                continue

            for prompt_path in sorted(variant_dir.glob("*.jsonl")):
                jobs.append(
                    AuditJob(
                        prompt_path=prompt_path,
                        database_path=database_path,
                        output_path=output_dir
                        / domain_dir.name
                        / variant_dir.name
                        / f"{prompt_path.stem}_results.jsonl",
                    )
                )
    return jobs


def discover_released_audit_jobs(output_dir: Path) -> list[AuditJob]:
    jobs: list[AuditJob] = []
    if not DEFAULT_RELEASED_DATABASE_DIR.exists():
        return jobs

    database_path = DEFAULT_RELEASED_DATABASE_DIR / "lmlm_database.json"
    prompts_dir = DEFAULT_RELEASED_DATABASE_DIR / "prompts"
    if not database_path.exists() or not prompts_dir.exists():
        return jobs

    for prompt_path in sorted(prompts_dir.glob("*.jsonl")):
        jobs.append(
            AuditJob(
                prompt_path=prompt_path,
                database_path=database_path,
                output_path=output_dir
                / DEFAULT_RELEASED_DATABASE_DIR.name
                / database_path.stem
                / f"{prompt_path.stem}_results.jsonl",
            )
        )
    return jobs


def discover_all_audit_jobs(output_dir: Path) -> list[AuditJob]:
    return discover_custom_audit_jobs(output_dir) + discover_released_audit_jobs(
        output_dir
    )


def infer_prompt_paths_for_database(database_path: Path) -> list[Path]:
    variant_prompt_dir = database_path.parent / "prompts" / database_path.stem
    if variant_prompt_dir.exists():
        return sorted(variant_prompt_dir.glob("*.jsonl"))
    return []


def resolve_audit_jobs(args: argparse.Namespace) -> list[AuditJob]:
    if args.prompt_files is not None:
        return [
            AuditJob(
                prompt_path=prompt_path,
                database_path=args.database_path,
                output_path=args.output_dir / f"{prompt_path.stem}_results.jsonl",
            )
            for prompt_path in args.prompt_files
        ]

    if args.database_path != DEFAULT_DATABASE_PATH:
        inferred_prompt_paths = infer_prompt_paths_for_database(args.database_path)
        if inferred_prompt_paths:
            return [
                AuditJob(
                    prompt_path=prompt_path,
                    database_path=args.database_path,
                    output_path=args.output_dir
                    / args.database_path.parent.name
                    / args.database_path.stem
                    / f"{prompt_path.stem}_results.jsonl",
                )
                for prompt_path in inferred_prompt_paths
            ]

    return discover_all_audit_jobs(args.output_dir)


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
    state_metrics: dict[str, float],
    cross_state_metrics: dict[str, float],
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the prompt audit.")
    parser.add_argument(
        "--prompt-files",
        nargs="*",
        type=Path,
        default=None,
        help=(
            "Specific prompt JSONL files to audit. If omitted, run all prompt files for "
            "all custom databases under data/custom_databases."
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=12,
        help="Maximum number of tokens to generate per prompt.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of prompts to run per file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where JSONL audit results will be written.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help=(
            "Optional path for a run log file. Defaults to <output-dir>/run_audit.log "
            "when omitted."
        ),
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="kilian-group/LMLM-llama2-382M",
        help="Model checkpoint to load.",
    )
    parser.add_argument(
        "--database-path",
        type=Path,
        default=DEFAULT_DATABASE_PATH,
        help=(
            "Path to a specific database JSON file. When provided without --prompt-files, "
            "run all prompt files in the sibling prompts/<variant>/ directory if present."
        ),
    )
    parser.add_argument(
        "--disable-dblookup",
        action="store_true",
        help="Deprecated shortcut for running only the DEL-OFF state.",
    )
    parser.add_argument(
        "--states",
        nargs="*",
        default=[state.value for state in DatabaseState],
        choices=[state.value for state in DatabaseState],
        help="Database states to evaluate.",
    )
    parser.add_argument(
        "--wandb_activation",
        type=str,
        default="off",
        choices=["on", "off"],
        help="Enable or disable Weights & Biases logging.",
    )
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    log_path = args.log_file or (args.output_dir / "run_audit.log")
    logger = AuditLogger(log_path)

    try:
        logger.print(f"Logging run_audit output to {log_path}")

        jobs = resolve_audit_jobs(args)
        if not jobs:
            raise FileNotFoundError(
                "No audit jobs found. Add custom prompts under data/custom_databases or "
                "pass --prompt-files explicitly."
            )

        from model_loader import load_model_and_tokenizer

        state_values = [DatabaseState(state) for state in args.states]
        if args.disable_dblookup:
            state_values = [DatabaseState.DEL_OFF]
        states = state_values
        wandb_module = setup_wandb() if args.wandb_activation == "on" else None

        jobs_by_database: dict[Path, list[AuditJob]] = defaultdict(list)
        for job in jobs:
            jobs_by_database[job.database_path].append(job)

        cross_state_rows: list[dict[str, Any]] = []
        per_state_rows: list[dict[str, Any]] = []

        for database_path in sorted(jobs_by_database):
            model, tokenizer = load_model_and_tokenizer(
                model_name=args.model_name,
                database_path=database_path,
            )
            base_db_manager = model.db_manager

            for job in jobs_by_database[database_path]:
                logger.print(f"Prompt file: {job.prompt_path}")
                logger.print(f"Database used: {database_path}")
                logger.print("DB states: " + ", ".join(state.value for state in states))
                logger.print(
                    f"Running audit for {job.prompt_path} with database {database_path}"
                )
                results = run_audit(
                    prompt_path=job.prompt_path,
                    base_db_manager=base_db_manager,
                    model=model,
                    tokenizer=tokenizer,
                    states=states,
                    max_new_tokens=args.max_new_tokens,
                    limit=args.limit,
                )

                save_results(results, job.output_path)
                total_metrics = metrics_total(results)
                metrics_by_state = {
                    state.value: metrics_total(
                        [result for result in results if result["state"] == state.value]
                    )
                    for state in states
                }

                cross_state_rows.append(
                    {
                        "prompt_file": str(job.prompt_path),
                        "database_path": str(database_path),
                        **total_metrics,
                    }
                )
                for state in states:
                    per_state_rows.append(
                        {
                            "prompt_file": str(job.prompt_path),
                            "database_path": str(database_path),
                            "state": state.value,
                            **metrics_by_state[state.value],
                        }
                    )

                logger.print("Cross-state audit metrics:")
                logger.print(f"  Paired count: {total_metrics['paired_count']}")
                logger.print(
                    f"  Parametric leakage L(f): {total_metrics['parametric_leakage']:.3f}"
                )
                logger.print(
                    "  Retrieval-mediated correctness R(f): "
                    f"{total_metrics['retrieval_mediated_correctness']:.3f}"
                )
                logger.print(
                    f"  Retrieval artifact rate: {total_metrics['retrieval_artifact_rate']:.3f}"
                )
                logger.print("Metrics by state:")
                for state in states:
                    metrics = metrics_by_state[state.value]
                    logger.print(f"{state.value}:")
                    logger.print(f"  Count: {metrics['count']}")
                    logger.print(f"  Exact match: {metrics['exact_match']:.3f}")
                    logger.print(f"  Contains match: {metrics['contains_match']:.3f}")
                    logger.print(f"  Unknown rate: {metrics['unknown_rate']:.3f}")
                    logger.print(f"  Precision: {metrics['precision']:.3f}")
                    logger.print(f"  Recall: {metrics['recall']:.3f}")
                    logger.print(f"  F1: {metrics['f1']:.3f}")
                    logger.print(
                        f"  Parametric leakage L(f): {metrics['parametric_leakage']:.3f}"
                    )
                    logger.print(
                        f"  Retrieval-mediated correctness R(f): {metrics['retrieval_mediated_correctness']:.3f}"
                    )
                    logger.print(
                        f"  Retrieval artifact rate: {metrics['retrieval_artifact_rate']:.3f}"
                    )
                    if wandb_module is not None:
                        log_metrics_to_wandb(
                            wandb_module=wandb_module,
                            prompt_path=job.prompt_path,
                            state=state,
                            state_metrics=metrics,
                            cross_state_metrics=total_metrics,
                            model_name=args.model_name,
                            database_path=database_path,
                            max_new_tokens=args.max_new_tokens,
                            limit=args.limit,
                        )
                        logger.print(f"  W&B run: {job.prompt_path.stem}_{state.value}")

        cross_state_csv_path = args.output_dir / "cross_state_metrics.csv"
        per_state_csv_path = args.output_dir / "per_state_metrics.csv"
        write_metrics_csvs(
            cross_state_rows=cross_state_rows,
            per_state_rows=per_state_rows,
            cross_state_path=cross_state_csv_path,
            per_state_path=per_state_csv_path,
        )
        logger.print(f"Wrote cross-state metrics CSV to {cross_state_csv_path}")
        logger.print(f"Wrote per-state metrics CSV to {per_state_csv_path}")
    finally:
        logger.close()


if __name__ == "__main__":
    main()
