import argparse
from dataclasses import dataclass
from pathlib import Path


DEFAULT_PROMPT_DIR = Path("data/prompts")
DEFAULT_CUSTOM_DATABASE_DIR = Path("data/custom_databases")
DEFAULT_RELEASED_DATABASE_DIR = Path("data/released_database")
DEFAULT_OUTPUT_DIR = Path("outputs/audit")
DEFAULT_DATABASE_PATH = Path("data/lmlm_database.json")


@dataclass(frozen=True)
class AuditJob:
    prompt_path: Path
    database_path: Path
    output_path: Path


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
