import os
import sys
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/lmlm-audit"))

WANDB_TEST_PROJECT = "lmlm-audit-tests"



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



class FakeModel:
    """Minimal sentence-embedding model stub."""

    def encode(self, texts, **_kwargs):
        return [[1.0] * max(1, len(texts))]


class FakeIndex:
    """Stub for a FAISS-like index."""

    def __init__(self, indices, distances):
        self.indices = list(indices)
        self.distances = list(distances)

    def search(self, _query_embedding, _top_k):
        return [self.distances], [self.indices]


class FakeRetriever:
    """Stub retriever that returns configurable triplets."""

    def __init__(self, id_to_triplet=None, distances=None):
        self.top_k = 3
        self.default_threshold = 0.6
        self.model = FakeModel()

        if id_to_triplet is None:
            id_to_triplet = {
                0: ("Hexol", "First Described By", "Jorgensen"),
                1: ("Hexol", "Structure Recognized By", "Werner"),
                2: ("Jocelyne Girard-Bujold", "Term End", "2004"),
            }
        if distances is None:
            distances = [0.95, 0.90, 0.80]

        self.id_to_triplet = id_to_triplet
        indices = sorted(id_to_triplet.keys())
        self.index = FakeIndex(
            indices=indices,
            distances=distances[: len(indices)],
        )

    @staticmethod
    def _normalize_text(text):
        return text.lower().strip()


class FakeBaseManager:
    """Stub database manager."""

    def __init__(self, return_value="Jorgensen"):
        self.database_name = "fake"
        self.database_org_file = []
        self.database = {}
        self.topk_retriever = FakeRetriever()
        self._return_value = return_value

    def init_topk_retriever(self, *args, **kwargs):
        pass

    def retrieve_from_database(self, _prompt, threshold=None):
        return self._return_value



@pytest.fixture
def fake_base_manager():
    return FakeBaseManager()


@pytest.fixture
def basic_target_fact():
    from database_states import TargetFact

    return TargetFact(
        fact_id=10,
        subject="Hexol",
        subject_aliases=(),
        relation="First Described By",
        relation_aliases=(),
        object="Jørgensen",
        object_aliases=("Jorgensen",),
    )
