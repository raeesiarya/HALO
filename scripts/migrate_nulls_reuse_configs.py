#!/usr/bin/env python3
"""Key finished NULLs standard runs on the manifest-blind facts digest.

NULLs sweep and policy phases audit pre-built closure prompt files that
differ from the gated prompts only in their deletion manifests. They share
the standard run's FULL pass and reuse its rows through
``prompt_facts_sha256``, which runs finished before that key existed do not
store. For every ``<out-root>/nulls-wiki-1b/<set>/`` this checks the stored
raw ``prompt_sha256`` against the gated prompt file, then:

- ``<stem>_audit_config.json``: adds ``prompt_facts_sha256`` (what reuse
  consumers match on);
- ``<stem>_full/full_config.json``: replaces ``prompt_sha256`` with
  ``prompt_facts_sha256`` (the FULL pass's identity).

A config whose raw digest does not match the gated prompts is left untouched
and makes the run exit 1. Dry run by default; ``--apply`` writes.

Usage:
  uv run python scripts/migrate_nulls_reuse_configs.py --out-root out-cross-model
  uv run python scripts/migrate_nulls_reuse_configs.py --out-root out-cross-model --apply
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Sequence

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from halo.cli.persistence import (  # noqa: E402
    atomic_write_text,
    prompt_digest,
    prompt_facts_digest,
)

NULLS_MODEL = "nulls-wiki-1b"


def migrate_set(
    set_dir: Path, *, apply: bool, log: Callable[[str], None] = print
) -> bool:
    """Migrate one prompt set's configs; False when any config was refused."""
    gated_files = sorted((set_dir / "prep").glob("*_nulls_gated.jsonl"))
    if len(gated_files) != 1:
        log(
            f"{set_dir.name}: expected one gated prompt file under prep/, "
            f"found {len(gated_files)}; skipped"
        )
        return True
    gated = gated_files[0]
    stem = gated.stem
    raw = prompt_digest(gated)
    facts = prompt_facts_digest(gated)

    ok = True
    targets = (
        (set_dir / f"{stem}_audit_config.json", False),
        (set_dir / f"{stem}_full" / "full_config.json", True),
    )
    for path, replace_raw in targets:
        label = f"{set_dir.name}/{path.relative_to(set_dir)}"
        if not path.is_file():
            log(f"{label}: missing; skipped")
            continue
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored.get("prompt_facts_sha256") == facts and (
            "prompt_sha256" not in stored
        ) == replace_raw:
            log(f"{label}: already migrated")
            continue
        if stored.get("prompt_sha256") != raw:
            log(
                f"{label}: stored prompt_sha256 does not match {gated.name}; "
                "refusing to migrate"
            )
            ok = False
            continue
        upgraded = {**stored, "prompt_facts_sha256": facts}
        if replace_raw:
            upgraded.pop("prompt_sha256")
        log(f"{label}: {'migrated' if apply else 'would migrate'}")
        if apply:
            atomic_write_text(path, json.dumps(upgraded, indent=2, sort_keys=True))
    return ok


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out-root",
        type=Path,
        required=True,
        help="Scheduler output root holding nulls-wiki-1b/<set>/ runs.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the upgraded configs (default: dry run).",
    )
    args = parser.parse_args(argv)

    root = args.out_root / NULLS_MODEL
    set_dirs = sorted(path for path in root.iterdir() if path.is_dir()) if root.is_dir() else []
    if not set_dirs:
        print(f"No NULLs runs under {root}.")
        return 1
    results = [migrate_set(set_dir, apply=args.apply) for set_dir in set_dirs]
    if not args.apply:
        print("Dry run; re-run with --apply to write.")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
