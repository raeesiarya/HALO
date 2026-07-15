import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/lmlm-audit"))

import model_loader
from model_loader import _get_best_device, _resolve_database_path



class TestGetBestDevice:
    def test_returns_cuda_when_available(self):
        with patch.object(torch.cuda, "is_available", return_value=True):
            device = _get_best_device()
        assert device.type == "cuda"

    def test_returns_mps_when_cuda_unavailable_and_mps_available(self):
        with (
            patch.object(torch.cuda, "is_available", return_value=False),
            patch.object(torch.backends.mps, "is_available", return_value=True),
        ):
            device = _get_best_device()
        assert device.type == "mps"

    def test_returns_cpu_when_nothing_available(self):
        with (
            patch.object(torch.cuda, "is_available", return_value=False),
            patch.object(torch.backends.mps, "is_available", return_value=False),
        ):
            device = _get_best_device()
        assert device.type == "cpu"

    def test_cuda_takes_priority_over_mps(self):
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.backends.mps, "is_available", return_value=True),
        ):
            device = _get_best_device()
        assert device.type == "cuda"

    def test_returns_torch_device_instance(self):
        with (
            patch.object(torch.cuda, "is_available", return_value=False),
            patch.object(torch.backends.mps, "is_available", return_value=False),
        ):
            device = _get_best_device()
        assert isinstance(device, torch.device)



class TestResolveDatabasePath:
    def test_existing_json_path_returned_as_is(self, tmp_path):
        db = tmp_path / "database.json"
        db.touch()
        result = _resolve_database_path(db)
        assert result == db

    def test_existing_jsonl_path_returned_as_is(self, tmp_path):
        db = tmp_path / "database.jsonl"
        db.touch()
        result = _resolve_database_path(db)
        assert result == db

    def test_missing_jsonl_falls_back_to_json(self, tmp_path):
        json_path = tmp_path / "database.json"
        json_path.touch()
        jsonl_path = tmp_path / "database.jsonl"
        result = _resolve_database_path(jsonl_path)
        assert result == json_path

    def test_missing_json_no_fallback_returns_original(self, tmp_path):
        db = tmp_path / "database.json"
        result = _resolve_database_path(db)
        assert result == db

    def test_missing_jsonl_and_missing_json_returns_original(self, tmp_path):
        jsonl_path = tmp_path / "missing.jsonl"
        result = _resolve_database_path(jsonl_path)
        assert result == jsonl_path

    def test_returns_path_object(self, tmp_path):
        db = tmp_path / "db.json"
        db.touch()
        result = _resolve_database_path(db)
        assert isinstance(result, Path)

    def test_nested_directory(self, tmp_path):
        nested = tmp_path / "a" / "b"
        nested.mkdir(parents=True)
        db = nested / "db.json"
        db.touch()
        result = _resolve_database_path(db)
        assert result == db

    def test_non_jsonl_suffix_no_fallback(self, tmp_path):
        csv_path = tmp_path / "data.csv"
        result = _resolve_database_path(csv_path)
        assert result == csv_path

    def test_path_with_spaces(self, tmp_path):
        db = tmp_path / "my database.json"
        db.touch()
        result = _resolve_database_path(db)
        assert result == db



class TestLoadModelAndTokenizer:
    def test_raises_import_error_when_lmlm_missing(self):
        """Without the lmlm package the function must raise a clear ImportError."""
        import builtins

        real_import = builtins.__import__

        def _mock_import(name, *args, **kwargs):
            if name.startswith("lmlm"):
                raise ImportError("lmlm not installed")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_mock_import):
            with pytest.raises(ImportError, match="upstream `lmlm` package"):
                model_loader.load_model_and_tokenizer()

    def test_calls_load_dotenv_for_token(self, tmp_path):
        """load_dotenv should be called so HF_TOKEN can be sourced from .env."""
        dotenv_mock = MagicMock()
        tokenizer_mock = MagicMock()
        tokenizer_mock.pad_token = "x"

        fake_lmlm = MagicMock()
        model_out = fake_lmlm.modeling_lmlm.LlamaForLMLM.from_pretrained_with_db.return_value
        model_out.to.return_value = model_out

        with (
            patch.dict(
                sys.modules,
                {
                    "lmlm": fake_lmlm,
                    "lmlm.database": fake_lmlm.database,
                    "lmlm.modeling_lmlm": fake_lmlm.modeling_lmlm,
                },
            ),
            patch("model_loader.load_dotenv", dotenv_mock),
            patch("model_loader.AutoTokenizer") as tok_cls,
            patch.object(torch.cuda, "is_available", return_value=False),
            patch.object(torch.backends.mps, "is_available", return_value=False),
        ):
            tok_cls.from_pretrained.return_value = tokenizer_mock
            model_loader.load_model_and_tokenizer(
                model_name="fake/model",
                database_path=str(tmp_path / "db.json"),
            )

        dotenv_mock.assert_called_once()

    def test_pad_token_set_when_none(self, tmp_path):
        """If tokenizer.pad_token is None it should be set to eos_token."""
        tokenizer_mock = MagicMock()
        tokenizer_mock.pad_token = None
        tokenizer_mock.eos_token = "<eos>"

        fake_lmlm = MagicMock()
        model_out = fake_lmlm.modeling_lmlm.LlamaForLMLM.from_pretrained_with_db.return_value
        model_out.to.return_value = model_out

        with (
            patch.dict(
                sys.modules,
                {
                    "lmlm": fake_lmlm,
                    "lmlm.database": fake_lmlm.database,
                    "lmlm.modeling_lmlm": fake_lmlm.modeling_lmlm,
                },
            ),
            patch("model_loader.load_dotenv"),
            patch("model_loader.AutoTokenizer") as tok_cls,
            patch.object(torch.cuda, "is_available", return_value=False),
            patch.object(torch.backends.mps, "is_available", return_value=False),
        ):
            tok_cls.from_pretrained.return_value = tokenizer_mock
            model_loader.load_model_and_tokenizer(
                model_name="fake/model",
                database_path=str(tmp_path / "db.json"),
            )

        assert tokenizer_mock.pad_token == "<eos>"

    def test_pad_token_unchanged_when_already_set(self, tmp_path):
        tokenizer_mock = MagicMock()
        tokenizer_mock.pad_token = "<pad>"
        original_pad = tokenizer_mock.pad_token

        fake_lmlm = MagicMock()
        model_out = fake_lmlm.modeling_lmlm.LlamaForLMLM.from_pretrained_with_db.return_value
        model_out.to.return_value = model_out

        with (
            patch.dict(
                sys.modules,
                {
                    "lmlm": fake_lmlm,
                    "lmlm.database": fake_lmlm.database,
                    "lmlm.modeling_lmlm": fake_lmlm.modeling_lmlm,
                },
            ),
            patch("model_loader.load_dotenv"),
            patch("model_loader.AutoTokenizer") as tok_cls,
            patch.object(torch.cuda, "is_available", return_value=False),
            patch.object(torch.backends.mps, "is_available", return_value=False),
        ):
            tok_cls.from_pretrained.return_value = tokenizer_mock
            model_loader.load_model_and_tokenizer(
                model_name="fake/model",
                database_path=str(tmp_path / "db.json"),
            )

        assert tokenizer_mock.pad_token == original_pad

    def test_model_moved_to_device(self, tmp_path):
        tokenizer_mock = MagicMock()
        tokenizer_mock.pad_token = "x"

        fake_lmlm = MagicMock()
        model_mock = MagicMock()
        model_mock.to.return_value = model_mock
        fake_lmlm.modeling_lmlm.LlamaForLMLM.from_pretrained_with_db.return_value = model_mock

        with (
            patch.dict(
                sys.modules,
                {
                    "lmlm": fake_lmlm,
                    "lmlm.database": fake_lmlm.database,
                    "lmlm.modeling_lmlm": fake_lmlm.modeling_lmlm,
                },
            ),
            patch("model_loader.load_dotenv"),
            patch("model_loader.AutoTokenizer") as tok_cls,
            patch.object(torch.cuda, "is_available", return_value=False),
            patch.object(torch.backends.mps, "is_available", return_value=False),
        ):
            tok_cls.from_pretrained.return_value = tokenizer_mock
            model_loader.load_model_and_tokenizer(
                model_name="fake/model",
                database_path=str(tmp_path / "db.json"),
            )

        model_mock.to.assert_called_once()
        model_mock.eval.assert_called_once()

    def test_returns_tuple_of_model_and_tokenizer(self, tmp_path):
        tokenizer_mock = MagicMock()
        tokenizer_mock.pad_token = "x"

        fake_lmlm = MagicMock()
        model_mock = MagicMock()
        model_mock.to.return_value = model_mock
        fake_lmlm.modeling_lmlm.LlamaForLMLM.from_pretrained_with_db.return_value = model_mock

        with (
            patch.dict(
                sys.modules,
                {
                    "lmlm": fake_lmlm,
                    "lmlm.database": fake_lmlm.database,
                    "lmlm.modeling_lmlm": fake_lmlm.modeling_lmlm,
                },
            ),
            patch("model_loader.load_dotenv"),
            patch("model_loader.AutoTokenizer") as tok_cls,
            patch.object(torch.cuda, "is_available", return_value=False),
            patch.object(torch.backends.mps, "is_available", return_value=False),
        ):
            tok_cls.from_pretrained.return_value = tokenizer_mock
            result = model_loader.load_model_and_tokenizer(
                model_name="fake/model",
                database_path=str(tmp_path / "db.json"),
            )

        assert isinstance(result, tuple)
        assert len(result) == 2
