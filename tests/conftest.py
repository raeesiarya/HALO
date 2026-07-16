import os
import sys
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

WANDB_TEST_PROJECT = "halo-tests"


def pytest_configure(config):
    """Force offline W&B mode unless the caller overrides it."""
    os.environ.setdefault("WANDB_MODE", "offline")


@pytest.fixture(scope="session")
def wandb_run():
    """
    A single W&B run shared across the whole test session.

    Tests that want to log plots or metrics should accept ``wandb_run`` as a
    parameter and guard against ``None`` (returned when wandb cannot be
    initialised).
    """
    try:
        import wandb  # noqa: PLC0415

        run = wandb.init(
            project=WANDB_TEST_PROJECT,
            name="pytest-unit-tests",
            mode=os.environ.get("WANDB_MODE", "offline"),
            tags=["unit-tests"],
            reinit=True,
        )
        yield run
        run.finish()
    except Exception:
        yield None
