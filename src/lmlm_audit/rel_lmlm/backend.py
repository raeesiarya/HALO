import html
import re
from dataclasses import dataclass
from typing import Any

from lmlm_audit.core.backend import AuditObservation, default_retrieval_trace
from lmlm_audit.core.examples import AuditExample
from lmlm_audit.rel_lmlm.database import build_state_db_manager
from lmlm_audit.core.states import DatabaseState, retrieval_enabled


LOOKUP_VALUE_PATTERN = re.compile(
    r"<\|db_entity\|>.*?<\|db_relationship\|>.*?<\|db_return\|>\s*(.*?)\s*<\|db_end\|>",
    re.DOTALL,
)
DB_MARKUP_SPAN_PATTERN = re.compile(r"<\|db_[^|]+\|>.*?<\|db_end\|>", re.DOTALL)
DB_SPECIAL_TOKEN_PATTERN = re.compile(r"<\|db_[^|]+\|>")
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")


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
    return default_retrieval_trace(state)


@dataclass
class RelLMLMAuditBackend:
    base_db_manager: Any
    model: Any
    tokenizer: Any

    def generate(
        self,
        example: AuditExample,
        state: DatabaseState,
        *,
        max_new_tokens: int = 12,
    ) -> AuditObservation:
        prompt_row = dict(example.source_row)
        self.model.db_manager = build_state_db_manager(
            base_db_manager=self.base_db_manager,
            prompt_row=prompt_row,
            state=state,
        )
        if hasattr(self.model.db_manager, "reset_trace"):
            self.model.db_manager.reset_trace()

        answer = generate_answer(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt_text=example.prompt,
            max_new_tokens=max_new_tokens,
            enable_dblookup=retrieval_enabled(state),
        )
        retrieval_trace = getattr(self.model.db_manager, "last_trace", None)
        return AuditObservation(
            model_output=answer,
            retrieval_trace=retrieval_trace,
        )
