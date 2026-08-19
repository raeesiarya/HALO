"""Embed generated factq questions through Co-LMLM's <FACT-q> head.

Stage 2 of the factq closure predicate. Each question from stage 1
(``data/generate_factq_questions.py``) runs through the audited backend as
a FULL-state generation; the query vector the model emits for its first
retrieval event *is* the question's <FACT-q> embedding — the same capture
path the audit uses for eval prompts, no model internals touched. A
question whose generation never issues a search is a recorded miss, not a
silent gap.

Run from the public Co-LMLM checkout in its environment (same requirement
as run_audit):

    python -m halo.factq_embed \
        --questions data/custom_databases/prompts_trex_factq.jsonl \
        --index-path <index dir>

Writes ``<questions stem>_vectors.npz`` (keys ``{fact}/FACTQ/q{n}``) for
run_audit's ``--closure ...,factq --factq-vectors`` plus a per-question
``_embed_log.jsonl`` with capture diagnostics.

Sharding mirrors the audit CLI: ``--shard I/N`` embeds only facts with
row index ≡ I (mod N) and writes shard-suffixed artifacts; a later
invocation without ``--shard`` merges the shard files, embeds anything
still missing, writes the canonical outputs, and removes the shard
partials. Reruns reuse prior vectors only when the logged question text
still matches the questions file, so a regenerated stage-1 artifact can
never silently mix with stale embeddings.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np
from tqdm import tqdm

from halo.cli.jobs import DEFAULT_INDEX_DIR
from halo.cli.persistence import parse_shard
from halo.core.backend import audit_example
from halo.core.examples import AuditExample
from halo.core.states import DatabaseState
from halo.interventions.factq import (
    FACTQ_QUESTIONS_FIELD,
    load_factq_vector_map,
    prompt_row_key,
    save_factq_vectors,
)
from halo.registry import available_backends, get_backend_spec
import models  # noqa: F401  (imports register the bundled backends)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument(
        "--backend",
        choices=available_backends(),
        default="co-lmlm",
        help="Inference backend whose <FACT-q> head embeds the questions.",
    )
    backend = pre.parse_known_args(argv)[0].backend

    parser = argparse.ArgumentParser(
        description="Embed factq questions via the audited backend.",
        parents=[pre],
    )
    parser.add_argument(
        "--questions",
        type=Path,
        required=True,
        help="Enriched prompt JSONL from data/generate_factq_questions.py.",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=DEFAULT_INDEX_DIR,
        help="Directory of the model's retrieval index.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Vectors .npz path; defaults to <questions stem>_vectors.npz "
            "next to --questions."
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=12,
        help="Generation budget per question; only the first search matters.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on question rows, for smoke tests.",
    )
    parser.add_argument(
        "--shard",
        type=str,
        default=None,
        metavar="I/N",
        help=(
            "Embed only facts with row index I mod N, writing shard-suffixed "
            "artifacts; run without --shard afterwards to merge and finalize."
        ),
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional path that mirrors this command's progress output.",
    )
    get_backend_spec(backend).add_arguments(parser)
    return parser.parse_args(argv)


def first_query_embedding(result: dict[str, Any]) -> Any | None:
    """The vector of the generation's earliest retrieval event.

    The question is the whole prompt, so the first search the model issues
    while answering is its question->fact query; later events belong to
    continued decoding.
    """
    events = sorted(
        (
            item
            for item in result.get("_query_embeddings") or []
            if item.get("vector") is not None
        ),
        key=lambda item: item.get("event_index", 0),
    )
    return events[0]["vector"] if events else None


def question_example(row: dict[str, Any], question: str) -> AuditExample:
    # The question replaces the eval prompt; gold fields stay so the
    # retrieval trace's support diagnostics refer to the right fact. The
    # FULL state never consults the deletion manifest.
    return AuditExample.from_prompt_row({**row, "prompt_text": question})


def plan_embeddings(
    questions: list[tuple[str, int, str]],
    prior_vectors: dict[str, dict[int, Any]],
    prior_records: dict[tuple[str, int], dict[str, Any]],
) -> tuple[
    dict[str, dict[int, Any]],
    dict[tuple[str, int], dict[str, Any]],
    list[tuple[str, int, str]],
]:
    """Split questions into reusable prior work and what must be embedded.

    A prior record is reused only when its logged question text equals the
    current question — a captured record additionally needs its vector in
    the prior store, and a miss record (captured=False) is itself the
    result. Everything else is re-embedded, so a regenerated questions
    file invalidates stale vectors instead of mixing with them.
    """
    kept_vectors: dict[str, dict[int, Any]] = {}
    kept_records: dict[tuple[str, int], dict[str, Any]] = {}
    todo: list[tuple[str, int, str]] = []
    for key, question_index, question in questions:
        record = prior_records.get((key, question_index))
        if record is None or record.get("question") != question:
            todo.append((key, question_index, question))
            continue
        if record.get("captured"):
            vector = prior_vectors.get(key, {}).get(question_index)
            if vector is None:
                todo.append((key, question_index, question))
                continue
            kept_vectors.setdefault(key, {})[question_index] = vector
        kept_records[(key, question_index)] = record
    return kept_vectors, kept_records, todo


def _shard_artifact_paths(output_path: Path, index: int, count: int) -> tuple[Path, Path]:
    stem = output_path.stem
    return (
        output_path.with_name(f"{stem}.shard{index}-of-{count}.npz"),
        output_path.with_name(f"{stem}.shard{index}-of-{count}_embed_log.jsonl"),
    )


def _existing_shard_artifacts(output_path: Path) -> list[tuple[Path, Path]]:
    stem = output_path.stem
    pairs = []
    for npz_path in sorted(output_path.parent.glob(f"{stem}.shard*-of-*.npz")):
        log_path = npz_path.with_name(f"{npz_path.stem}_embed_log.jsonl")
        pairs.append((npz_path, log_path))
    return pairs


def _read_log(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    records: dict[tuple[str, int], dict[str, Any]] = {}
    if not path.is_file():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            records[(str(record["key"]), int(record["question_index"]))] = record
    return records


def _write_log(
    path: Path, records: dict[tuple[str, int], dict[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for _, record in sorted(records.items()):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_question_rows(
    path: Path, limit: int | None
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if limit is not None:
        rows = rows[:limit]

    question_rows = [row for row in rows if row.get(FACTQ_QUESTIONS_FIELD)]
    if not question_rows:
        raise ValueError(
            f"{path} has no rows with {FACTQ_QUESTIONS_FIELD!r}; "
            "run data/generate_factq_questions.py first."
        )
    row_by_key: dict[str, dict[str, Any]] = {}
    for row in question_rows:
        key = prompt_row_key(row)
        if not key:
            raise ValueError(
                "Every enriched row needs a prompt_id/fact_id to key its "
                f"vectors; offending row: {json.dumps(row)[:200]}"
            )
        if key in row_by_key:
            raise ValueError(
                f"Duplicate fact key {key!r} in {path}; vectors would collide."
            )
        row_by_key[key] = row
    return question_rows, row_by_key


def _embed_questions(
    backend: Any,
    row_by_key: dict[str, dict[str, Any]],
    todo: list[tuple[str, int, str]],
    *,
    max_new_tokens: int,
    progress_desc: str,
) -> tuple[dict[str, dict[int, Any]], dict[tuple[str, int], dict[str, Any]]]:
    vectors: dict[str, dict[int, Any]] = {}
    records: dict[tuple[str, int], dict[str, Any]] = {}
    for key, question_index, question in tqdm(
        todo, desc=progress_desc, unit="question"
    ):
        result = audit_example(
            backend,
            question_example(row_by_key[key], question),
            DatabaseState.FULL,
            max_new_tokens=max_new_tokens,
        )
        vector = first_query_embedding(result)
        trace = result.get("retrieval_trace") or {}
        events = trace.get("retrieval_events") or []
        first_event = events[0] if events else {}
        selected = first_event.get("selected_candidate") or {}
        if vector is not None:
            vectors.setdefault(key, {})[question_index] = vector
        records[(key, question_index)] = {
            "key": key,
            "question_index": question_index,
            "question": question,
            "captured": vector is not None,
            "retrieval_events": len(events),
            "selected_entry_id": selected.get("entry_id"),
            "selected_score": selected.get("score"),
        }
    return vectors, records


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_path = args.output or args.questions.with_name(
        f"{args.questions.stem}_vectors.npz"
    )
    canonical_log = output_path.with_name(f"{output_path.stem}_embed_log.jsonl")
    shard = parse_shard(args.shard) if args.shard is not None else None

    emit: Callable[[str], None] = print
    logger = None
    if args.log_file is not None:
        from halo.cli.reporting import AuditLogger

        logger = AuditLogger(args.log_file)
        emit = logger.print

    try:
        question_rows, row_by_key = _load_question_rows(args.questions, args.limit)
        questions = [
            (prompt_row_key(row), question_index, str(question))
            for row in question_rows
            for question_index, question in enumerate(row[FACTQ_QUESTIONS_FIELD])
        ]

        if shard is not None:
            index, count = shard
            shard_keys = {
                prompt_row_key(row) for row in question_rows[index::count]
            }
            target = [item for item in questions if item[0] in shard_keys]
            npz_path, log_path = _shard_artifact_paths(output_path, index, count)
            prior_vectors = (
                load_factq_vector_map(npz_path) if npz_path.is_file() else {}
            )
            prior_records = _read_log(log_path)
        else:
            target = questions
            npz_path, log_path = output_path, canonical_log
            prior_vectors = (
                load_factq_vector_map(output_path)
                if output_path.is_file()
                else {}
            )
            prior_records = _read_log(canonical_log)
            for shard_npz, shard_log in _existing_shard_artifacts(output_path):
                prior_vectors.update(load_factq_vector_map(shard_npz))
                prior_records.update(_read_log(shard_log))

        vectors, records, todo = plan_embeddings(
            target, prior_vectors, prior_records
        )
        stats: Counter[str] = Counter(
            reused_captured=sum(
                1 for record in records.values() if record.get("captured")
            ),
            reused_missed=sum(
                1 for record in records.values() if not record.get("captured")
            ),
        )

        if todo:
            spec = get_backend_spec(args.backend)
            backend = spec.build_backend(args, args.index_path)
            new_vectors, new_records = _embed_questions(
                backend,
                row_by_key,
                todo,
                max_new_tokens=args.max_new_tokens,
                progress_desc=(
                    f"factq embed shard {shard[0]}/{shard[1]}"
                    if shard is not None
                    else "factq embed"
                ),
            )
            for key, by_index in new_vectors.items():
                vectors.setdefault(key, {}).update(by_index)
            records.update(new_records)
            stats["embedded"] = len(new_records)
            stats["no_search_issued"] = sum(
                1 for record in new_records.values() if not record["captured"]
            )

        if shard is None and not any(vectors.values()):
            raise ValueError(
                "No question produced a retrieval event; nothing to save. "
                f"See {log_path}."
            )
        save_factq_vectors(npz_path, vectors)
        _write_log(log_path, records)
        if shard is None:
            # Canonical outputs subsume the shard partials; drop them so a
            # stale shard can never leak into a later finalize.
            for shard_npz, shard_log in _existing_shard_artifacts(output_path):
                shard_npz.unlink()
                shard_log.unlink(missing_ok=True)

        captured = sum(len(by_index) for by_index in vectors.values())
        scope = (
            f"shard {shard[0]}/{shard[1]}" if shard is not None else "finalize"
        )
        emit(
            f"factq embed {scope}: {captured}/{len(target)} question vectors "
            f"over {len({key for key, _, _ in target})} facts -> {npz_path} "
            f"({stats['embedded']} embedded, {stats['reused_captured']} reused, "
            f"{stats['reused_missed'] + stats['no_search_issued']} no-search "
            f"misses; log {log_path})"
        )
    finally:
        if logger is not None:
            logger.close()


if __name__ == "__main__":
    main()
