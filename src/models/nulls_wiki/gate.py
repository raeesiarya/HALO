"""Verification-gate logic: per-fact evidence that title->sink mapping is
real (docs/NULLS_AUDIT_DESIGN.md §9).

Sinks are keyed through a PRNG on the source id; a wrong ``title_to_index``
entry silently activates an unrelated mask and mimics deletion. The gate
requires, per fact, that the gold answer's log-likelihood under Sink-On
(gold title's mask active) exceeds it under a deterministic placebo mask by
at least a margin. Facts that fail are excluded as *unmappable*; the
exclusion rate is a substantive statistic and lands in the summary.

A globally wrong ``title_to_index`` shows up as a margin distribution
centered on zero — but the converse does not hold per fact, so the gate is
a check on the authors' artifact, not a license to substitute a guessed one.

The scoring path needs the model (``nulls`` dependency group + GPU); the
merge/finalize path is pure file work. ``scripts/nulls_verification_gate.py``
is the thin CLI the cross-model scheduler drives, striped via shards.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any


def shard_path(output: Path, index: int, count: int) -> Path:
    return output.with_suffix(f".shard{index}of{count}.jsonl")


def parse_shard(spec: str) -> tuple[int, int]:
    index_text, count_text = spec.split("/", 1)
    index, count = int(index_text), int(count_text)
    if count < 1 or not 0 <= index < count:
        raise ValueError(f"shard must be I/N with 0 <= I < N, got {spec!r}.")
    return index, count


def score_rows(
    backend: Any,
    prompts_path: Path,
    *,
    margin: float,
    placebo_ceiling: float,
    shard: tuple[int, int] | None = None,
    limit: int | None = None,
) -> list[dict]:
    """Score every (sharded) fact: Sink-On vs placebo log-likelihood."""
    records: list[dict] = []
    with open(prompts_path, encoding="utf-8") as source:
        for row_index, line in enumerate(source):
            if limit is not None and row_index >= limit:
                break
            if shard is not None and row_index % shard[1] != shard[0]:
                continue
            row = json.loads(line)
            title = row.get("source_title")
            record: dict[str, Any] = {
                "row_index": row_index,
                "fact_id": row.get("fact_id"),
                "prompt_id": row.get("prompt_id"),
                "source_title": title,
            }
            if not title or title not in backend.registry:
                record.update(status="unmappable", margin=None)
                records.append(record)
                continue
            placebo_title = backend.registry.placebo(
                f"gate::{row.get('fact_id')}::{row.get('prompt_id')}",
                anchor_title=title,
                similarity_ceiling=placebo_ceiling,
            )
            prompt = row["prompt_text"]
            answer = " " + str(row["gold_object"]).strip()
            sink_on = backend.answer_logprob(
                prompt, answer, eval_mode="activate_seq", active_title=title
            )
            placebo = backend.answer_logprob(
                prompt, answer, eval_mode="activate_seq", active_title=placebo_title
            )
            observed_margin = sink_on - placebo
            record.update(
                status="pass" if observed_margin >= margin else "fail",
                margin=observed_margin,
                sink_on_logprob=sink_on,
                placebo_logprob=placebo,
                placebo_title=placebo_title,
            )
            records.append(record)
    return records


def write_shard(records: list[dict], output: Path, index: int, count: int) -> Path:
    target = shard_path(output, index, count)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as sink:
        for record in records:
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
    return target


def read_shards(output: Path, count: int) -> list[dict]:
    records: list[dict] = []
    for index in range(count):
        shard_file = shard_path(output, index, count)
        if not shard_file.is_file():
            raise FileNotFoundError(f"Missing gate shard: {shard_file}")
        with open(shard_file, encoding="utf-8") as handle:
            records.extend(json.loads(line) for line in handle if line.strip())
    return records


def finalize(
    records: list[dict],
    *,
    output: Path,
    prompts_path: Path,
    margin: float,
    emit_passed_prompts: Path | None = None,
) -> dict:
    """Write the merged gate records, the summary, and (optionally) the
    gated prompt set the audits consume. Returns the summary."""
    records = sorted(records, key=lambda record: record["row_index"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as sink:
        for record in records:
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")

    margins = [r["margin"] for r in records if r["margin"] is not None]
    passed = sum(1 for r in records if r["status"] == "pass")
    failed = sum(1 for r in records if r["status"] == "fail")
    unmappable = sum(1 for r in records if r["status"] == "unmappable")
    total = len(records)
    summary = {
        "prompts": str(prompts_path),
        "margin_threshold": margin,
        "total": total,
        "passed": passed,
        "failed": failed,
        "unmappable": unmappable,
        "exclusion_rate": (failed + unmappable) / total if total else None,
        "margin_mean": statistics.fmean(margins) if margins else None,
        "margin_median": statistics.median(margins) if margins else None,
        "margin_stdev": statistics.stdev(margins) if len(margins) > 1 else None,
    }
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))

    if emit_passed_prompts is not None:
        passed_indices = {
            record["row_index"] for record in records if record["status"] == "pass"
        }
        kept = 0
        with open(prompts_path, encoding="utf-8") as source, open(
            emit_passed_prompts, "w", encoding="utf-8"
        ) as gated:
            for row_index, line in enumerate(source):
                if row_index in passed_indices:
                    gated.write(line)
                    kept += 1
        if kept != len(passed_indices):
            raise RuntimeError(
                f"Gated prompt emission wrote {kept} rows but {len(passed_indices)} "
                "facts passed; the gate records do not match the prompt file."
            )
        summary["gated_prompts"] = str(emit_passed_prompts)
        summary["gated_rows"] = kept
    return summary
