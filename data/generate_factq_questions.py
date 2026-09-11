"""Generate <FACT-q> deletion questions for an audit prompt file.

Stage 1 of the factq closure predicate. For each audited fact, the
document shown to the question generator is the DB entry the FULL pass
retrieved (never the eval prompt: deleting what the eval question
retrieves guarantees a successful forget without measuring anything). The
gold answer's mention in that entry is marked as fact span 1 and the
released generator adapter emits question/paraphrased-answer pairs.

Reads the prompt JSONL plus a completed FULL-pass dir; writes an enriched
prompt JSONL with ``factq_questions``/``factq_answers`` fields that
``python -m halo.factq_embed`` (stage 2) consumes. Runs anywhere with
transformers+peft — the Co-LMLM environment is only needed for stage 2.

Usage:
    python data/generate_factq_questions.py \
        --prompts data/custom_databases/prompts_trex.jsonl \
        --full-dir outputs/prompts_trex_full \
        --output data/custom_databases/prompts_trex_factq.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from halo.core.examples import AuditExample  # noqa: E402
from halo.interventions.factq import (  # noqa: E402
    FACTQ_ANSWERS_FIELD,
    FACTQ_QUESTIONS_FIELD,
    FACTQ_SOURCE_ENTRY_FIELD,
    build_generator_prompt,
    dedupe_questions,
    mark_fact_span,
    parse_generator_output,
    prompt_row_key,
)

DEFAULT_ADAPTER = "lil-lab/CoLMLM-Question-Generator"
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
# Extra sampled completions per requested question, to absorb duplicates
# and parse failures before the dedupe cut.
OVERSAMPLE_FACTOR = 2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--prompts", type=Path, required=True,
                        help="Audit prompt JSONL to enrich.")
    parser.add_argument("--full-dir", type=Path, required=True,
                        help="Completed FULL-pass dir (full_results.jsonl).")
    parser.add_argument("--output", type=Path, default=None,
                        help="Enriched JSONL path; defaults to "
                             "<prompts stem>_factq.jsonl next to --prompts.")
    parser.add_argument("--adapter", type=str, default=DEFAULT_ADAPTER,
                        help="Question-generator LoRA adapter (HF id or path).")
    parser.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL,
                        help="Base model the adapter was trained on.")
    parser.add_argument("--num-questions", type=int, default=5,
                        help="Questions to keep per fact after dedupe.")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional cap on prompt rows, for smoke tests.")
    return parser.parse_args(argv)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def full_entry_by_key(full_dir: Path) -> dict[str, dict]:
    """key -> the FULL pass's selected DB entry {value, entry_id}."""
    results_path = full_dir / "full_results.jsonl"
    if not results_path.is_file():
        raise FileNotFoundError(
            f"{results_path} not found; run the audit's FULL pass first "
            "(factq documents are the retrieved DB entries)."
        )
    entries: dict[str, dict] = {}
    for row in read_jsonl(results_path):
        key = prompt_row_key(row)
        selected = (row.get("retrieval_trace") or {}).get("selected_candidate")
        if key and isinstance(selected, dict):
            entries[key] = selected
    return entries


def load_generator(adapter: str, base_model: str):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(adapter)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, adapter).eval()
    return tokenizer, model


def generate_questions(
    tokenizer,
    model,
    marked_document: str,
    *,
    num_questions: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> tuple[list[tuple[str, str]], int]:
    """(question, paraphrased answer) pairs and the parse-miss count.

    One greedy completion (the generator's canonical question) plus an
    oversampled batch for paraphrase diversity, deduped on question text.
    """
    import torch

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": build_generator_prompt(marked_document)}],
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_length = inputs["input_ids"].shape[1]

    completions: list[str] = []
    with torch.no_grad():
        greedy = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
        completions.append(
            tokenizer.decode(greedy[0, prompt_length:], skip_special_tokens=False)
        )
        if num_questions > 1:
            sampled = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=num_questions * OVERSAMPLE_FACTOR,
            )
            completions.extend(
                tokenizer.decode(row[prompt_length:], skip_special_tokens=False)
                for row in sampled
            )

    parsed = [parse_generator_output(text) for text in completions]
    parse_misses = sum(1 for pair in parsed if pair is None)
    pairs = dedupe_questions([pair for pair in parsed if pair is not None])
    return pairs[:num_questions], parse_misses


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_path = args.output or args.prompts.with_name(
        f"{args.prompts.stem}_factq.jsonl"
    )
    if output_path.resolve() == args.prompts.resolve():
        raise ValueError("--output must not overwrite --prompts.")

    rows = read_jsonl(args.prompts)
    if args.limit is not None:
        rows = rows[: args.limit]
    entries = full_entry_by_key(args.full_dir)

    import torch

    torch.manual_seed(args.seed)
    tokenizer, model = load_generator(args.adapter, args.base_model)

    stats: Counter[str] = Counter()
    questions_total = 0
    enriched_rows: list[dict] = []
    for row in rows:
        stats["facts"] += 1
        enriched = dict(row)
        enriched_rows.append(enriched)
        key = prompt_row_key(row)
        example = AuditExample.from_prompt_row(row)
        selected = entries.get(key)
        document = str((selected or {}).get("value") or "")
        if not document:
            # No FULL retrieval for this fact — nothing to question. The
            # missing fields make the gap visible downstream instead of an
            # empty-but-present question list.
            enriched["factq_skip_reason"] = "no-full-entry"
            stats["no_full_entry"] += 1
            continue
        marked = mark_fact_span(
            document, example.ground_truth, example.object_aliases
        )
        if marked is None:
            enriched["factq_skip_reason"] = "answer-span-not-found"
            stats["span_miss"] += 1
            continue
        pairs, parse_misses = generate_questions(
            tokenizer,
            model,
            marked,
            num_questions=args.num_questions,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
        )
        stats["parse_misses"] += parse_misses
        if not pairs:
            enriched["factq_skip_reason"] = "generator-output-unparseable"
            stats["generator_miss"] += 1
            continue
        enriched[FACTQ_QUESTIONS_FIELD] = [question for question, _ in pairs]
        enriched[FACTQ_ANSWERS_FIELD] = [answer for _, answer in pairs]
        enriched[FACTQ_SOURCE_ENTRY_FIELD] = selected.get("entry_id")
        stats["enriched"] += 1
        questions_total += len(pairs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for enriched in enriched_rows:
            handle.write(json.dumps(enriched, ensure_ascii=False) + "\n")

    summary = {
        **dict(stats),
        "questions_total": questions_total,
        "num_questions_requested": args.num_questions,
        "adapter": args.adapter,
        "base_model": args.base_model,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "prompts": str(args.prompts),
        "full_dir": str(args.full_dir),
    }
    stats_path = output_path.with_name(f"{output_path.stem}_stats.json")
    stats_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"Enriched {stats['enriched']}/{stats['facts']} facts with "
        f"{questions_total} questions -> {output_path}"
    )
    print(
        f"Misses: {stats['no_full_entry']} no-full-entry, "
        f"{stats['span_miss']} answer-span-not-found, "
        f"{stats['generator_miss']} generator-output-unparseable "
        f"({stats['parse_misses']} unparseable completions). Stats: {stats_path}"
    )


if __name__ == "__main__":
    main()
