import os
from pathlib import Path
from typing import Any

import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer


def _get_best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_database_path(database_path: Path) -> Path:
    if database_path.exists():
        return database_path

    if database_path.suffix == ".jsonl":
        json_path = database_path.with_suffix(".json")
        if json_path.exists():
            return json_path

    return database_path


def load_model_and_tokenizer(
    model_name: str = "kilian-group/LMLM-llama2-382M",
    database_path: str | Path = "data/lmlm_database.json",
    threshold: float = 0.6,
    fallback_policy: str = "top1_anyway",
) -> tuple[Any, AutoTokenizer]:
    try:
        from lmlm.database import DatabaseManager
        from lmlm.modeling_lmlm import LlamaForLMLM
    except ImportError as exc:
        raise ImportError(
            "The upstream `lmlm` package is required for lookup-aware inference. "
            "Install the original LMLM repository in your environment before running this script."
        ) from exc

    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")

    device = _get_best_device()
    print("Using device:", device)

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    db_manager = DatabaseManager()
    database_path = _resolve_database_path(Path(database_path))
    if database_path.exists():
        db_manager.load_database(str(database_path))

    model = LlamaForLMLM.from_pretrained_with_db(
        model_name,
        db_manager=db_manager,
        use_special_tokens=True,
        threshold=threshold,
        fallback_policy=fallback_policy,
        token=hf_token,
    )
    model.to(device)
    model.eval()

    return model, tokenizer
