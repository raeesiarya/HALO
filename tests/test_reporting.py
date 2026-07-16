import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from halo.cli.reporting import (
    wandb_log_image,
    wandb_log_metrics,
    wandb_log_output_artifacts,
)


class FakeSummary:
    def __init__(self):
        self.data = {}

    def update(self, payload):
        self.data.update(payload)


class FakeArtifact:
    def __init__(self, name, type):
        self.name = name
        self.type = type
        self.files = []

    def add_file(self, local_path, name=None):
        self.files.append((local_path, name))


class FakeRun:
    def __init__(self):
        self.logged = []
        self.summary = FakeSummary()
        self.artifacts = []

    def log(self, payload):
        self.logged.append(payload)

    def log_artifact(self, artifact):
        self.artifacts.append(artifact)


class FakeWandb:
    def Artifact(self, name, type):
        return FakeArtifact(name, type)

    def Image(self, path):
        return ("IMAGE", path)


def test_wandb_log_metrics_filters_and_prefixes() -> None:
    run = FakeRun()
    wandb_log_metrics(
        run,
        {"l_rep_hat": 0.8, "count": 5, "mode": "standard", "flag": True, "gap": None},
        prefix="p/probe/",
    )
    assert run.logged == [{"p/probe/l_rep_hat": 0.8, "p/probe/count": 5}]
    assert run.summary.data == {"p/probe/l_rep_hat": 0.8, "p/probe/count": 5}


def test_wandb_log_metrics_noop_when_nothing_numeric() -> None:
    run = FakeRun()
    wandb_log_metrics(run, {"mode": "x", "note": None}, prefix="p/")
    assert run.logged == []


def test_wandb_log_image_only_when_present(tmp_path) -> None:
    run = FakeRun()
    wandb = FakeWandb()
    missing = tmp_path / "nope.png"
    wandb_log_image(run, wandb, missing, "fig")
    assert run.logged == []

    present = tmp_path / "curve.png"
    present.write_bytes(b"x")
    wandb_log_image(run, wandb, present, "fig")
    assert run.logged == [{"fig": ("IMAGE", str(present))}]


def test_wandb_log_output_artifacts_uploads_all_files(tmp_path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "results.jsonl").write_text("{}\n")
    (tmp_path / "cross_state_metrics.csv").write_text("a\n")
    (tmp_path / "sub" / "entanglement.png").write_bytes(b"x")
    (tmp_path / "sub" / "sidecar.npz").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("ignored")  # not an uploaded type

    run = FakeRun()
    wandb_log_output_artifacts(run, FakeWandb(), tmp_path)

    assert len(run.artifacts) == 1
    uploaded = {name for _, name in run.artifacts[0].files}
    assert uploaded == {
        "results.jsonl",
        "cross_state_metrics.csv",
        "sub/entanglement.png",
        "sub/sidecar.npz",
    }
    assert "notes.txt" not in uploaded


def test_wandb_log_output_artifacts_noop_when_empty(tmp_path) -> None:
    run = FakeRun()
    wandb_log_output_artifacts(run, FakeWandb(), tmp_path)
    assert run.artifacts == []
