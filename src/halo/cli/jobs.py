import argparse
from dataclasses import dataclass
from pathlib import Path

DEFAULT_OUTPUT_DIR = Path("outputs/audit")
DEFAULT_INDEX_DIR = Path("data/co-lmlm-wiki-index")


@dataclass(frozen=True)
class AuditJob:
    prompt_path: Path
    output_path: Path


def resolve_audit_jobs(args: argparse.Namespace) -> list[AuditJob]:
    return [
        AuditJob(
            prompt_path=prompt_path,
            output_path=args.output_dir / f"{prompt_path.stem}_results.jsonl",
        )
        for prompt_path in args.prompt_files
    ]
