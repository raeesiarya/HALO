from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from halo.scheduler import (
    DATASETS,
    DEFAULT_MODELS,
    Job,
    SchedulerAlreadyRunning,
    SchedulerLock,
    SchedulerSignal,
    _csv,
    _parse_args,
    _termination_signals,
    _validate_inputs,
    _validate_runtime_options,
    _warm_up,
    build_jobs,
    plan_fingerprint,
    ready_jobs,
    run_jobs,
)


def _job(
    tmp_path: Path,
    key: str,
    code: str,
    *,
    dependencies: frozenset[str] = frozenset(),
    priority: int = 1,
    expected_outputs: tuple[Path, ...] = (),
) -> Job:
    return Job(
        key=key,
        model="test",
        dataset="test",
        phase="test",
        command=(sys.executable, "-c", code),
        environment=(),
        dependencies=dependencies,
        priority=priority,
        log_path=tmp_path / "logs" / f"{key}.log",
        expected_outputs=expected_outputs,
    )


def test_default_plan_has_90_jobs_and_correct_dependencies(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    jobs = build_jobs(
        repo_root=repo_root,
        out_root=tmp_path / "out",
        co_lmlm_dir=tmp_path / "Co-LMLM",
        index_dir=tmp_path / "index",
        inherited_env={},
    )

    # Per set: standard 1, sweep 1+6+1, adversarial 1, del-off 1, policy 5.
    assert len(jobs) == 90
    assert len([job for job in jobs if job.model == "co-lmlm"]) == 80
    assert len([job for job in jobs if job.model != "co-lmlm"]) == 10
    assert {job.dataset for job in jobs} == set(DATASETS)
    assert {job.model for job in jobs} == set(DEFAULT_MODELS)

    by_key = {job.key: job for job in jobs}
    for dataset in DATASETS:
        standard = f"co-lmlm.{dataset}.standard"
        assert by_key[standard].dependencies == frozenset()
        for phase in ("adversarial", "del-off"):
            job = by_key[f"co-lmlm.{dataset}.{phase}"]
            assert job.dependencies == frozenset({standard})
            assert job.env_dict()["SUITE_PHASES"] == phase

        # The sweep decomposes into prep -> one job per radius -> finalize.
        prep = by_key[f"co-lmlm.{dataset}.sweep.prep"]
        assert prep.dependencies == frozenset({standard})
        assert ("--sweep-shard-radii", "none") in zip(
            prep.command, prep.command[1:]
        )
        radius_keys = [f"co-lmlm.{dataset}.sweep.radius{i}" for i in range(6)]
        for index, key in enumerate(radius_keys):
            radius_job = by_key[key]
            assert radius_job.dependencies == frozenset({prep.key})
            assert ("--sweep-shard-radii", f"{index}/6") in zip(
                radius_job.command, radius_job.command[1:]
            )
            assert len(radius_job.expected_outputs) == 1
        finalize = by_key[f"co-lmlm.{dataset}.sweep.finalize"]
        assert finalize.dependencies == frozenset({prep.key, *radius_keys})
        assert len(finalize.expected_outputs) == 6

        # The policy matrix runs oracle first, then four policies in parallel.
        oracle = by_key[f"co-lmlm.{dataset}.policy.oracle"]
        assert oracle.dependencies == frozenset({standard})
        assert oracle.env_dict()["POLICIES"] == "oracle"
        for policy in ("geometric", "value", "provenance", "hybrid"):
            job = by_key[f"co-lmlm.{dataset}.policy.{policy}"]
            assert job.dependencies == frozenset({oracle.key})
            assert job.env_dict()["POLICIES"] == policy

        for model in ("standard-lm-360m-fw", "smollm2-360m"):
            assert by_key[f"{model}.{dataset}.standard"].dependencies == frozenset()

    # Standard jobs rank first because they unlock later phases.
    co_standard_priorities = {
        job.priority
        for job in jobs
        if job.model == "co-lmlm" and job.phase == "standard"
    }
    baseline_priorities = {
        job.priority for job in jobs if job.model != "co-lmlm"
    }
    assert min(co_standard_priorities) > max(baseline_priorities)

    # Gate jobs (sweep prep, oracle) outrank their non-gating siblings.
    for dataset in DATASETS:
        prep = by_key[f"co-lmlm.{dataset}.sweep.prep"]
        finalize = by_key[f"co-lmlm.{dataset}.sweep.finalize"]
        assert prep.priority > finalize.priority
        oracle = by_key[f"co-lmlm.{dataset}.policy.oracle"]
        hybrid = by_key[f"co-lmlm.{dataset}.policy.hybrid"]
        assert oracle.priority > hybrid.priority


def test_workers_per_job_applies_only_to_unsharded_fact_striped_jobs(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    jobs = build_jobs(
        repo_root=repo_root,
        out_root=tmp_path / "out",
        co_lmlm_dir=tmp_path / "Co-LMLM",
        index_dir=tmp_path / "index",
        datasets=("trex",),
        models=("co-lmlm",),
        workers_per_job=16,
        inherited_env={},
    )
    by_key = {job.key: job for job in jobs}
    for key in (
        "co-lmlm.trex.standard",
        "co-lmlm.trex.adversarial",
        "co-lmlm.trex.del-off",
        "co-lmlm.trex.policy.oracle",
        "co-lmlm.trex.policy.hybrid",
    ):
        assert by_key[key].env_dict()["SUITE_WORKERS"] == "16"
    # Sweep sub-jobs are single processes: parallelism lives in the graph.
    for key, job in by_key.items():
        if ".sweep." in key:
            assert job.env_dict()["SUITE_WORKERS"] == "1"


def test_shards_decompose_fact_striped_phases_into_shard_and_finalize_jobs(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    jobs = build_jobs(
        repo_root=repo_root,
        out_root=tmp_path / "out",
        co_lmlm_dir=tmp_path / "Co-LMLM",
        index_dir=tmp_path / "index",
        datasets=("trex",),
        models=("co-lmlm",),
        workers_per_job=16,
        shards_per_phase=2,
        inherited_env={},
    )
    # standard 3, sweep 8, adversarial 3, del-off 3, policy 5*3.
    assert len(jobs) == 32
    by_key = {job.key: job for job in jobs}

    standard_finalize = by_key["co-lmlm.trex.standard.finalize"]
    shard_keys = {f"co-lmlm.trex.standard.shard{i}" for i in range(2)}
    assert standard_finalize.dependencies == frozenset(shard_keys)
    for index in range(2):
        shard = by_key[f"co-lmlm.trex.standard.shard{index}"]
        assert shard.dependencies == frozenset()
        assert ("--shard", f"{index}/2") in zip(shard.command, shard.command[1:])
        # Shard jobs are single workers on their own GPU; a second fan-out
        # inside the job would collide with the injected --shard.
        assert shard.env_dict()["SUITE_WORKERS"] == "1"
        assert shard.expected_outputs == ()
    assert standard_finalize.expected_outputs != ()

    # Later phases hang off the standard finalize, not the shards.
    for key in (
        "co-lmlm.trex.sweep.prep",
        "co-lmlm.trex.adversarial.shard0",
        "co-lmlm.trex.del-off.shard0",
        "co-lmlm.trex.policy.oracle.shard0",
    ):
        assert by_key[key].dependencies == frozenset({standard_finalize.key})

    # Non-oracle policies wait for the oracle finalize (REUSE_FROM chain).
    oracle_finalize = by_key["co-lmlm.trex.policy.oracle.finalize"]
    for policy in ("geometric", "value", "provenance", "hybrid"):
        shard = by_key[f"co-lmlm.trex.policy.{policy}.shard0"]
        assert shard.dependencies == frozenset({oracle_finalize.key})
        assert shard.env_dict()["POLICIES"] == policy

    with pytest.raises(ValueError, match="shards"):
        build_jobs(
            repo_root=repo_root,
            out_root=tmp_path / "out",
            co_lmlm_dir=tmp_path / "Co-LMLM",
            index_dir=tmp_path / "index",
            datasets=("trex",),
            models=("co-lmlm",),
            shards_per_phase=0,
            inherited_env={},
        )


def test_del_off_environment_controls_complementary_phase(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    jobs = build_jobs(
        repo_root=repo_root,
        out_root=tmp_path / "out",
        co_lmlm_dir=tmp_path / "Co-LMLM",
        index_dir=tmp_path / "index",
        datasets=("trex",),
        models=("co-lmlm",),
        phases=("standard", "del-off"),
        inherited_env={"DEL_OFF_MODE": "forbid-token"},
    )
    assert {job.env_dict()["DEL_OFF_MODE"] for job in jobs} == {"forbid-token"}
    del_off = next(job for job in jobs if job.phase == "del-off")
    assert any("null-retrieval" in str(path) for path in del_off.expected_outputs)

    with pytest.raises(ValueError, match="DEL_OFF_MODE"):
        build_jobs(
            repo_root=repo_root,
            out_root=tmp_path / "invalid-out",
            co_lmlm_dir=tmp_path / "Co-LMLM",
            index_dir=tmp_path / "index",
            datasets=("trex",),
            models=("co-lmlm",),
            inherited_env={"DEL_OFF_MODE": "not-a-mode"},
        )


def test_scheduler_rejects_phase_or_backend_specific_forwarded_flags(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    common = dict(
        repo_root=repo_root,
        out_root=tmp_path / "out",
        co_lmlm_dir=tmp_path / "Co-LMLM",
        index_dir=tmp_path / "index",
        datasets=("trex",),
        inherited_env={},
    )
    for arguments in (
        ("--radius-grid", "0.9:0.7:0.1"),
        ("--co-lmlm-del-off-mode", "forbid-token"),
        ("--output-dir", "/tmp/wrong"),
    ):
        with pytest.raises(ValueError, match="controlled"):
            build_jobs(**common, audit_args=arguments)

    # Forwarded args reach every job; scheduler-injected flags follow them
    # (and therefore win once the suite appends "$@" last).
    jobs = build_jobs(**common, audit_args=("--limit", "10"))
    assert all(
        ("--limit", "10") in zip(job.command, job.command[1:]) for job in jobs
    )


def test_phase_only_run_requires_completed_standard_outputs(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    prompt = repo_root / DATASETS["trex"]
    prompt.parent.mkdir(parents=True)
    prompt.write_text("{}\n", encoding="utf-8")
    index_dir = repo_root / "index"
    index_dir.mkdir()
    out_root = tmp_path / "out"

    with pytest.raises(FileNotFoundError, match="phase-only"):
        _validate_inputs(
            repo_root=repo_root,
            out_root=out_root,
            datasets=("trex",),
            models=("co-lmlm",),
            phases=("sweep", "adversarial"),
            index_dir=index_dir,
            del_off_mode="null-retrieval",
        )

    standard_dir = out_root / "co-lmlm" / "trex"
    (standard_dir / "prompts_trex_full").mkdir(parents=True)
    (standard_dir / "prompts_trex_results.jsonl").touch()
    (standard_dir / "prompts_trex_full" / "full_results.jsonl").touch()
    (standard_dir / "prompts_trex_audit_config.json").write_text(
        '{"del_off_mode": "null-retrieval"}',
        encoding="utf-8",
    )
    _validate_inputs(
        repo_root=repo_root,
        out_root=out_root,
        datasets=("trex",),
        models=("co-lmlm",),
        phases=("sweep", "adversarial"),
        index_dir=index_dir,
        del_off_mode="null-retrieval",
    )
    with pytest.raises(ValueError, match="DEL-OFF mode"):
        _validate_inputs(
            repo_root=repo_root,
            out_root=out_root,
            datasets=("trex",),
            models=("co-lmlm",),
            phases=("sweep",),
            index_dir=index_dir,
            del_off_mode="forbid-token",
        )


def test_ready_jobs_respects_success_and_failure(tmp_path: Path) -> None:
    root = _job(tmp_path, "root", "pass", priority=1)
    child = _job(
        tmp_path,
        "child",
        "pass",
        dependencies=frozenset({"root"}),
        priority=3,
    )
    independent = _job(tmp_path, "independent", "pass", priority=2)

    runnable, blocked = ready_jobs(
        (root, child, independent), succeeded=set(), failed=set()
    )
    assert [job.key for job in runnable] == ["independent", "root"]
    assert blocked == []

    runnable, blocked = ready_jobs(
        (child, independent), succeeded={"root"}, failed=set()
    )
    assert [job.key for job in runnable] == ["child", "independent"]
    assert blocked == []

    runnable, blocked = ready_jobs(
        (child, independent), succeeded=set(), failed={"root"}
    )
    assert [job.key for job in runnable] == ["independent"]
    assert [job.key for job in blocked] == ["child"]


def test_plan_fingerprint_changes_with_prompt_contents(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    prompt = repo_root / DATASETS["trex"]
    prompt.parent.mkdir(parents=True)
    prompt.write_text('{"prompt_id":"first"}\n', encoding="utf-8")
    kwargs = dict(
        repo_root=repo_root,
        out_root=tmp_path / "out",
        co_lmlm_dir=tmp_path / "Co-LMLM",
        index_dir=tmp_path / "index",
        datasets=("trex",),
        models=("smollm2-360m",),
        inherited_env={},
    )
    first = build_jobs(**kwargs)
    prompt.write_text('{"prompt_id":"second"}\n', encoding="utf-8")
    second = build_jobs(**kwargs)
    assert plan_fingerprint(first) != plan_fingerprint(second)


def test_plan_fingerprint_changes_with_index_metadata(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    prompt = repo_root / DATASETS["trex"]
    prompt.parent.mkdir(parents=True)
    prompt.write_text("{}\n", encoding="utf-8")
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    index_file = index_dir / "index.faiss"
    index_file.write_bytes(b"first")
    kwargs = dict(
        repo_root=repo_root,
        out_root=tmp_path / "out",
        co_lmlm_dir=tmp_path / "Co-LMLM",
        index_dir=index_dir,
        datasets=("trex",),
        models=("co-lmlm",),
        phases=("standard",),
        inherited_env={},
    )
    first = build_jobs(**kwargs)
    index_file.write_bytes(b"a different size")
    second = build_jobs(**kwargs)
    assert plan_fingerprint(first) != plan_fingerprint(second)


def test_plan_fingerprint_changes_with_co_lmlm_source(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    prompt = repo_root / DATASETS["trex"]
    prompt.parent.mkdir(parents=True)
    prompt.write_text("{}\n", encoding="utf-8")
    co_lmlm_dir = tmp_path / "Co-LMLM"
    source = co_lmlm_dir / "src" / "lmlm" / "model.py"
    source.parent.mkdir(parents=True)
    source.write_text("VERSION = 1\n", encoding="utf-8")
    kwargs = dict(
        repo_root=repo_root,
        out_root=tmp_path / "out",
        co_lmlm_dir=co_lmlm_dir,
        index_dir=tmp_path / "index",
        datasets=("trex",),
        models=("co-lmlm",),
        phases=("standard",),
        inherited_env={},
    )
    first = build_jobs(**kwargs)
    source.write_text("VERSION = 2\n", encoding="utf-8")
    second = build_jobs(**kwargs)
    assert plan_fingerprint(first) != plan_fingerprint(second)


def test_output_root_lock_rejects_second_scheduler(tmp_path: Path) -> None:
    out_root = tmp_path / "out"
    with SchedulerLock(out_root, "plan-a"):
        with pytest.raises(SchedulerAlreadyRunning, match="already owns"):
            with SchedulerLock(out_root, "plan-b"):
                pass


def test_termination_signal_is_converted_to_cleanup_exception() -> None:
    previous = signal.getsignal(signal.SIGTERM)
    with pytest.raises(SchedulerSignal) as raised:
        with _termination_signals():
            handler = signal.getsignal(signal.SIGTERM)
            assert callable(handler)
            handler(signal.SIGTERM, None)
    assert raised.value.signum == signal.SIGTERM
    assert signal.getsignal(signal.SIGTERM) == previous


def test_sigterm_stops_active_job_process_group(tmp_path: Path) -> None:
    child_pid_path = tmp_path / "child.pid"
    child_code = (
        "import os, time; from pathlib import Path; "
        f"Path({str(child_pid_path)!r}).write_text(str(os.getpid())); "
        "time.sleep(60)"
    )
    scheduler_code = (
        "import sys; from pathlib import Path; "
        "from halo.scheduler import Job, run_jobs; "
        "job=Job(key='long', model='test', dataset='test', phase='test', "
        f"command=({sys.executable!r}, '-c', {child_code!r}), "
        "environment=(), dependencies=frozenset(), priority=1, "
        f"log_path=Path({str(tmp_path / 'long.log')!r})); "
        "raise SystemExit(run_jobs((job,), "
        f"repo_root=Path({str(tmp_path)!r}), "
        f"out_root=Path({str(tmp_path / 'out')!r}), "
        "gpus=('0',), max_parallel=1, poll_seconds=0.01))"
    )
    scheduler = subprocess.Popen(
        (sys.executable, "-c", scheduler_code),
        cwd=tmp_path,
    )
    deadline = time.monotonic() + 5
    while not child_pid_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert child_pid_path.exists()
    child_pid = int(child_pid_path.read_text())

    os.kill(scheduler.pid, signal.SIGTERM)
    assert scheduler.wait(timeout=5) == 128 + signal.SIGTERM

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.01)
    else:
        pytest.fail(f"child process {child_pid} survived scheduler SIGTERM")


def test_runtime_options_reject_duplicate_gpus_and_invalid_counts() -> None:
    with pytest.raises(ValueError, match="unique"):
        _validate_runtime_options(
            gpus=("0", "0"),
            max_parallel=2,
            workers_per_job=1,
            poll_seconds=1,
        )
    with pytest.raises(ValueError, match="workers-per-job"):
        _validate_runtime_options(
            gpus=("0",),
            max_parallel=1,
            workers_per_job=0,
            poll_seconds=1,
        )
    with pytest.raises(ValueError, match="poll-seconds"):
        _validate_runtime_options(
            gpus=("0",),
            max_parallel=1,
            workers_per_job=1,
            poll_seconds=0,
        )


def test_cli_reads_suite_workers_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SUITE_WORKERS", "7")
    args = _parse_args(["--dry-run"])
    assert args.workers_per_job == 7


def test_scheduler_accepts_comma_or_space_separated_sets() -> None:
    assert _csv("trex,popqa") == ("trex", "popqa")
    assert _csv("trex popqa") == ("trex", "popqa")


def test_warmup_installs_openblas_once_before_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    co_lmlm_dir = tmp_path / "Co-LMLM"
    expected_source = co_lmlm_dir / "src" / "lmlm" / "eval" / "hf_generate.py"
    expected_source.parent.mkdir(parents=True)
    expected_source.touch()
    calls: list[tuple[str, ...]] = []

    def fake_run(command, **_kwargs):
        command = tuple(command)
        calls.append(command)
        return subprocess.CompletedProcess(
            command,
            1 if command[:2] == ("dpkg", "-s") else 0,
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        "halo.scheduler.shutil.which",
        lambda command: f"/usr/bin/{command}",
    )
    monkeypatch.setattr("halo.scheduler.os.geteuid", lambda: 0)

    _warm_up(
        repo_root=repo_root,
        co_lmlm_dir=co_lmlm_dir,
        need_co_lmlm=True,
        out_root=tmp_path / "out",
    )

    assert calls == [
        ("uv", "sync"),
        ("uv", "sync"),
        ("dpkg", "-s", "libopenblas0"),
        ("apt-get", "update"),
        ("apt-get", "install", "-y", "libopenblas0"),
    ]


def test_run_jobs_assigns_gpu_and_waits_for_dependency(tmp_path: Path) -> None:
    root_output = tmp_path / "root.txt"
    child_output = tmp_path / "child.txt"
    root = _job(
        tmp_path,
        "root",
        (
            "import os; from pathlib import Path; "
            f"Path({str(root_output)!r}).write_text("
            "os.environ['CUDA_VISIBLE_DEVICES'])"
        ),
        priority=2,
    )
    child = _job(
        tmp_path,
        "child",
        (
            "from pathlib import Path; "
            f"source=Path({str(root_output)!r}); "
            "assert source.exists(); "
            f"Path({str(child_output)!r}).write_text(source.read_text())"
        ),
        dependencies=frozenset({"root"}),
        priority=1,
    )

    code = run_jobs(
        (root, child),
        repo_root=tmp_path,
        out_root=tmp_path / "out",
        gpus=("3",),
        max_parallel=1,
        resume=True,
        poll_seconds=0.01,
    )

    assert code == 0
    assert root_output.read_text() == "3"
    assert child_output.read_text() == "3"
    state = (
        tmp_path
        / "out"
        / "_scheduler_state"
        / plan_fingerprint((root, child))
    )
    assert (state / "root.exit").read_text().strip() == "0"
    assert (state / "child.exit").read_text().strip() == "0"


def test_resume_marker_requires_expected_outputs(tmp_path: Path) -> None:
    counter = tmp_path / "counter.txt"
    artifact = tmp_path / "artifact.txt"
    code = (
        "from pathlib import Path; "
        f"counter=Path({str(counter)!r}); "
        "count=int(counter.read_text())+1 if counter.exists() else 1; "
        "counter.write_text(str(count)); "
        f"Path({str(artifact)!r}).touch()"
    )
    job = _job(
        tmp_path,
        "resumable",
        code,
        expected_outputs=(artifact,),
    )
    kwargs = dict(
        jobs=(job,),
        repo_root=tmp_path,
        out_root=tmp_path / "out",
        gpus=("0",),
        max_parallel=1,
        resume=True,
        poll_seconds=0.01,
    )

    assert run_jobs(**kwargs) == 0
    assert run_jobs(**kwargs) == 0
    assert counter.read_text() == "1"

    artifact.unlink()
    assert run_jobs(**kwargs) == 0
    assert counter.read_text() == "2"


def test_successful_process_without_contract_outputs_fails_job(tmp_path: Path) -> None:
    missing = tmp_path / "never-created.txt"
    job = _job(
        tmp_path,
        "incomplete",
        "pass",
        expected_outputs=(missing,),
    )
    assert (
        run_jobs(
            (job,),
            repo_root=tmp_path,
            out_root=tmp_path / "out",
            gpus=("0",),
            max_parallel=1,
            poll_seconds=0.01,
        )
        == 1
    )
    marker = (
        tmp_path
        / "out"
        / "_scheduler_state"
        / plan_fingerprint((job,))
        / "incomplete.exit"
    )
    assert marker.read_text().strip() == "97"


def test_run_jobs_blocks_only_dependents_after_failure(tmp_path: Path) -> None:
    failed = _job(tmp_path, "failed", "raise SystemExit(7)", priority=2)
    child = _job(
        tmp_path,
        "child",
        "pass",
        dependencies=frozenset({"failed"}),
    )
    grandchild = _job(
        tmp_path,
        "grandchild",
        "pass",
        dependencies=frozenset({"child"}),
    )
    independent_output = tmp_path / "independent.txt"
    independent = _job(
        tmp_path,
        "independent",
        f"from pathlib import Path; Path({str(independent_output)!r}).touch()",
        priority=1,
    )

    code = run_jobs(
        (failed, child, grandchild, independent),
        repo_root=tmp_path,
        out_root=tmp_path / "out",
        gpus=("0", "1"),
        max_parallel=2,
        resume=False,
        poll_seconds=0.01,
    )

    assert code == 1
    assert independent_output.exists()
    assert not (tmp_path / "logs" / "child.log").exists()
    assert not (tmp_path / "logs" / "grandchild.log").exists()
