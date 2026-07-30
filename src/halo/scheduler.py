"""Schedule the 35 cross-model HALO jobs across GPUs."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable, Iterator, Mapping, Sequence


DATASETS: dict[str, str] = {
    "trex": "data/prompts_trex.jsonl",
    "popqa": "data/prompts.jsonl",
    "googlere": "data/prompts_googlere.jsonl",
    "counterfact": "data/prompts_counterfact.jsonl",
    "zsre": "data/prompts_zsre.jsonl",
}
DEFAULT_MODELS = ("co-lmlm", "standard-lm-360m-fw", "smollm2-360m")
CO_LMLM_PHASES = ("standard", "sweep", "adversarial", "del-off", "policy")
PARAMETRIC_MODELS = ("standard-lm-360m-fw", "smollm2-360m")

# These estimates only order runnable jobs.
DATASET_SIZE_FALLBACK = {
    "trex": 27_630,
    "counterfact": 21_919,
    "zsre": 19_086,
    "popqa": 2_818,
    "googlere": 2_766,
}
PHASE_COST = {
    "standard": 3,
    "sweep": 18,
    "adversarial": 13,
    "del-off": 3,
    "policy": 12,
    "parametric": 3,
}

# Include semantic environment values in resume fingerprints.
PASSTHROUGH_ENV = (
    "STANDARD_CLOSURE",
    "SWEEP_CLOSURE",
    "ADVERSARIAL_CLOSURE",
    "RADIUS_GRID",
    "NEIGHBOR_MODE",
    "NEIGHBOR_MIN_COUNT",
    "DEL_OFF_MODE",
    "HF_HOME",
    "HF_HUB_CACHE",
    "TRANSFORMERS_CACHE",
)
POLICIES = ("oracle", "geometric", "value", "provenance", "hybrid")
DEL_OFF_MODES = ("null-retrieval", "forbid-token")
SCHEDULER_STATE_VERSION = 2
RESERVED_AUDIT_OPTIONS = {
    "--backend",
    "--prompt-files",
    "--index-path",
    "--output-dir",
    "--log-file",
    "--full-dir",
    "--reuse-from",
    "--shard",
    "--sweep-shard-radii",
    "--bootstrap-oracle-from-full",
    "--closure",
    "--radius-grid",
    "--adversarial",
}


@dataclass(frozen=True)
class Job:
    """One schedulable process pinned to one GPU."""

    key: str
    model: str
    dataset: str
    phase: str
    command: tuple[str, ...]
    environment: tuple[tuple[str, str], ...]
    dependencies: frozenset[str]
    priority: int
    log_path: Path
    input_fingerprint: str = ""
    expected_outputs: tuple[Path, ...] = ()

    def env_dict(self) -> dict[str, str]:
        return dict(self.environment)


@dataclass
class RunningJob:
    job: Job
    gpu: str
    process: subprocess.Popen[bytes]
    log_handle: BinaryIO
    started_at: float


class SchedulerAlreadyRunning(RuntimeError):
    """Raised when another scheduler owns this output root."""


class SchedulerSignal(Exception):
    """Convert a termination signal into normal scheduler cleanup."""

    def __init__(self, signum: int):
        self.signum = signum
        super().__init__(f"received signal {signum}")


def _csv(value: str) -> tuple[str, ...]:
    # Support comma- and space-separated values.
    return tuple(part for part in value.replace(",", " ").split() if part)


def _radius_values(spec: str) -> tuple[float, ...]:
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError(f"--radius-grid must be 'start:stop:step', got {spec!r}.")
    start, stop, step = (float(part) for part in parts)
    if step <= 0 or start < stop:
        raise ValueError(f"Invalid descending radius grid: {spec!r}.")
    radii: list[float] = []
    radius = start
    while radius >= stop - 1e-9:
        radii.append(round(radius, 6))
        radius -= step
    return tuple(radii)


def _line_count(path: Path, fallback: int) -> int:
    if not path.is_file():
        return fallback
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def _file_digest(path: Path) -> str:
    if not path.is_file():
        return "missing"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_digest(repo_root: Path) -> str:
    """Fingerprint executable audit code, including uncommitted edits."""

    paths = [
        *repo_root.glob("src/**/*.py"),
        *repo_root.glob("scripts/*.sh"),
        repo_root / "pyproject.toml",
        repo_root / "uv.lock",
    ]
    digest = hashlib.sha256()
    for path in sorted(
        (path for path in paths if path.is_file()),
        key=lambda item: str(item.relative_to(repo_root)),
    ):
        relative = str(path.relative_to(repo_root))
        digest.update(relative.encode())
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _co_lmlm_source_digest(co_lmlm_dir: Path) -> str:
    if not co_lmlm_dir.is_dir():
        return "missing"
    paths = [
        *co_lmlm_dir.glob("src/**/*.py"),
        co_lmlm_dir / "pyproject.toml",
        co_lmlm_dir / "uv.lock",
    ]
    digest = hashlib.sha256()
    for path in sorted(
        (path for path in paths if path.is_file()),
        key=lambda item: str(item.relative_to(co_lmlm_dir)),
    ):
        relative = str(path.relative_to(co_lmlm_dir))
        digest.update(relative.encode())
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _directory_identity(path: Path) -> str:
    """Fingerprint a large directory without reading its contents."""

    if not path.is_dir():
        return "missing"
    digest = hashlib.sha256()
    for child in sorted(path.iterdir(), key=lambda item: item.name):
        stat = child.stat()
        digest.update(child.name.encode())
        digest.update(
            f"\0{int(child.is_dir())}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode()
        )
    return digest.hexdigest()


def _job_key(model: str, dataset: str, phase: str) -> str:
    return f"{model}.{dataset}.{phase}"


def _validate_selection(
    datasets: Sequence[str], models: Sequence[str], phases: Sequence[str]
) -> None:
    if not datasets:
        raise ValueError("Select at least one prompt set.")
    if not models:
        raise ValueError("Select at least one model.")
    if "co-lmlm" in models and not phases:
        raise ValueError("Select at least one Co-LMLM phase.")
    unknown_sets = sorted(set(datasets) - DATASETS.keys())
    unknown_models = sorted(set(models) - set(DEFAULT_MODELS))
    unknown_phases = sorted(set(phases) - set(CO_LMLM_PHASES))
    if unknown_sets:
        raise ValueError(f"Unknown prompt sets: {', '.join(unknown_sets)}")
    if unknown_models:
        raise ValueError(f"Unknown models: {', '.join(unknown_models)}")
    if unknown_phases:
        raise ValueError(f"Unknown Co-LMLM phases: {', '.join(unknown_phases)}")


def _validate_audit_args(arguments: Sequence[str]) -> None:
    for argument in arguments:
        option = argument.split("=", 1)[0]
        if option in RESERVED_AUDIT_OPTIONS or option.startswith("--co-lmlm-"):
            raise ValueError(
                f"{option} is controlled by the cross-model scheduler and "
                "cannot be forwarded to every backend. Use the scheduler "
                "options or suite environment variables instead."
            )


def build_jobs(
    *,
    repo_root: Path,
    out_root: Path,
    co_lmlm_dir: Path,
    index_dir: Path,
    datasets: Sequence[str] = tuple(DATASETS),
    models: Sequence[str] = DEFAULT_MODELS,
    phases: Sequence[str] = CO_LMLM_PHASES,
    audit_args: Sequence[str] = (),
    inherited_env: Mapping[str, str] | None = None,
    workers_per_job: int = 1,
) -> list[Job]:
    """Build the cross-model dependency graph without starting processes."""

    _validate_selection(datasets, models, phases)
    _validate_audit_args(audit_args)
    if workers_per_job < 1:
        raise ValueError("--workers-per-job must be at least 1.")
    inherited_env = os.environ if inherited_env is None else inherited_env
    shared_env = {
        name: inherited_env[name]
        for name in PASSTHROUGH_ENV
        if inherited_env.get(name)
    }
    del_off_mode = inherited_env.get("DEL_OFF_MODE", "null-retrieval")
    if del_off_mode not in DEL_OFF_MODES:
        raise ValueError(
            f"DEL_OFF_MODE must be one of {', '.join(DEL_OFF_MODES)}, "
            f"got {del_off_mode!r}."
        )
    radii = (
        _radius_values(inherited_env.get("RADIUS_GRID", "0.95:0.70:0.05"))
        if "co-lmlm" in models
        else ()
    )
    source_fingerprint = _source_digest(repo_root)
    co_lmlm_source_fingerprint = (
        _co_lmlm_source_digest(co_lmlm_dir)
        if "co-lmlm" in models
        else "unused"
    )
    index_fingerprint = (
        _directory_identity(index_dir) if "co-lmlm" in models else "unused"
    )
    suite = repo_root / "scripts" / "run_audit_suite_co_lmlm.sh"
    log_root = out_root / "_scheduler_jobs"
    jobs: list[Job] = []

    for dataset in datasets:
        prompt_path = repo_root / DATASETS[dataset]
        rows = _line_count(prompt_path, DATASET_SIZE_FALLBACK[dataset])
        prompt_fingerprint = _file_digest(prompt_path)
        stem = prompt_path.stem
        co_output = out_root / "co-lmlm" / dataset
        shared_full = co_output / f"{stem}_full" / "full_results.jsonl"
        phase1_results = co_output / f"{stem}_results.jsonl"

        if "co-lmlm" in models:
            standard_key = _job_key("co-lmlm", dataset, "standard")
            for phase in phases:
                key = _job_key("co-lmlm", dataset, phase)
                dependencies = (
                    frozenset({standard_key})
                    if phase != "standard" and "standard" in phases
                    else frozenset()
                )
                # Prioritize standard jobs because they unlock later phases.
                unlock_bonus = 10**12 if phase == "standard" else 0
                priority = unlock_bonus + rows * PHASE_COST[phase]
                phase_workers = (
                    min(workers_per_job, len(radii))
                    if phase == "sweep"
                    else workers_per_job
                )
                environment = {
                    **shared_env,
                    "CO_LMLM_DIR": str(co_lmlm_dir),
                    "INDEX_DIR": str(index_dir),
                    "PROMPTS": str(prompt_path),
                    "OUTPUT_DIR": str(co_output),
                    "SUITE_PHASES": phase,
                    # Keep DEL-OFF behavior consistent across phases.
                    "DEL_OFF_MODE": del_off_mode,
                    # Sweep workers are capped at the number of radii.
                    "SUITE_WORKERS": str(phase_workers),
                }
                if phase == "standard":
                    expected_outputs = (
                        phase1_results,
                        shared_full,
                        co_output / "cross_state_metrics.csv",
                    )
                elif phase == "sweep":
                    sweep_dir = co_output / f"{stem}_sweep"
                    expected_outputs = tuple(
                        sweep_dir / f"sweep_rho_{rho:.4f}.jsonl"
                        for rho in radii
                    )
                elif phase == "adversarial":
                    expected_outputs = (
                        co_output
                        / f"{stem}_adversarial"
                        / "adversarial_results.jsonl",
                    )
                elif phase == "del-off":
                    complementary_mode = (
                        "forbid-token"
                        if del_off_mode == "null-retrieval"
                        else "null-retrieval"
                    )
                    expected_outputs = (
                        co_output
                        / "del_off_sensitivity"
                        / complementary_mode
                        / "cross_state_metrics.csv",
                    )
                else:
                    expected_outputs = tuple(
                        co_output
                        / "policy_matrix"
                        / policy
                        / "cross_state_metrics.csv"
                        for policy in POLICIES
                    )
                jobs.append(
                    Job(
                        key=key,
                        model="co-lmlm",
                        dataset=dataset,
                        phase=phase,
                        command=("bash", str(suite), *audit_args),
                        environment=tuple(sorted(environment.items())),
                        dependencies=dependencies,
                        priority=priority,
                        log_path=log_root / f"{key}.log",
                        input_fingerprint=(
                            f"source={source_fingerprint};"
                            f"co_source={co_lmlm_source_fingerprint};"
                            f"prompt={prompt_fingerprint};"
                            f"index={index_fingerprint}"
                        ),
                        expected_outputs=expected_outputs,
                    )
                )

        for model in PARAMETRIC_MODELS:
            if model not in models:
                continue
            phase = "standard"
            key = _job_key(model, dataset, phase)
            output_dir = out_root / model / dataset
            environment = {**shared_env}
            jobs.append(
                Job(
                    key=key,
                    model=model,
                    dataset=dataset,
                    phase=phase,
                    command=(
                        "uv",
                        "run",
                        "python",
                        "-m",
                        "halo.run_audit",
                        "--backend",
                        model,
                        "--prompt-files",
                        str(prompt_path),
                        "--output-dir",
                        str(output_dir),
                        "--full-dir",
                        str(output_dir / f"{stem}_full"),
                        "--log-file",
                        str(output_dir / "run_audit.log"),
                        *audit_args,
                    ),
                    environment=tuple(sorted(environment.items())),
                    dependencies=frozenset(),
                    priority=rows * PHASE_COST["parametric"],
                    log_path=log_root / f"{key}.log",
                    input_fingerprint=(
                        f"source={source_fingerprint};prompt={prompt_fingerprint}"
                    ),
                    expected_outputs=(
                        output_dir / f"{stem}_results.jsonl",
                        output_dir / "cross_state_metrics.csv",
                    ),
                )
            )

    _validate_graph(jobs)
    return jobs


def _validate_graph(jobs: Sequence[Job]) -> None:
    keys = [job.key for job in jobs]
    if len(keys) != len(set(keys)):
        raise ValueError("Scheduler plan contains duplicate job keys.")
    known = set(keys)
    for job in jobs:
        missing = job.dependencies - known
        if missing:
            raise ValueError(
                f"{job.key} has dependencies outside the plan: {sorted(missing)}"
            )

    # Reject dependency cycles before starting GPU work.
    remaining = {job.key: set(job.dependencies) for job in jobs}
    resolved: set[str] = set()
    while remaining:
        ready = {key for key, deps in remaining.items() if deps <= resolved}
        if not ready:
            raise ValueError(
                "Scheduler dependency cycle: " + ", ".join(sorted(remaining))
            )
        resolved.update(ready)
        for key in ready:
            remaining.pop(key)


def plan_fingerprint(jobs: Sequence[Job]) -> str:
    payload = {
        "scheduler_state_version": SCHEDULER_STATE_VERSION,
        "jobs": [
            {
                "key": job.key,
                "command": job.command,
                "environment": job.environment,
                "dependencies": sorted(job.dependencies),
                "input_fingerprint": job.input_fingerprint,
                "expected_outputs": [str(path) for path in job.expected_outputs],
            }
            for job in sorted(jobs, key=lambda item: item.key)
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:12]


def ready_jobs(
    jobs: Iterable[Job],
    succeeded: set[str],
    failed: set[str],
) -> tuple[list[Job], list[Job]]:
    """Return runnable and permanently blocked pending jobs."""

    runnable: list[Job] = []
    blocked: list[Job] = []
    for job in jobs:
        if job.dependencies & failed:
            blocked.append(job)
        elif job.dependencies <= succeeded:
            runnable.append(job)
    runnable.sort(key=lambda item: (-item.priority, item.key))
    blocked.sort(key=lambda item: item.key)
    return runnable, blocked


class DriverLog:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("a", encoding="utf-8", buffering=1)

    def write(self, message: str) -> None:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{stamp}] {message}"
        print(line, flush=True)
        self._handle.write(line + "\n")

    def close(self) -> None:
        self._handle.close()


def _write_exit_marker(path: Path, code: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(f"{code}\n", encoding="utf-8")
    temporary.replace(path)


def _successful_marker(path: Path, job: Job) -> bool:
    try:
        succeeded = int(path.read_text(encoding="utf-8").strip()) == 0
    except (FileNotFoundError, ValueError):
        return False
    return succeeded and all(output.is_file() for output in job.expected_outputs)


class SchedulerLock:
    """Exclusive per-output-root lock held for warmup and job execution."""

    def __init__(self, out_root: Path, fingerprint: str):
        self.path = out_root / "_scheduler.lock"
        self.fingerprint = fingerprint
        self._handle = None

    def __enter__(self) -> "SchedulerLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.seek(0)
            owner = handle.read().strip() or "unknown owner"
            handle.close()
            raise SchedulerAlreadyRunning(
                f"Another scheduler already owns {self.path} ({owner})."
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()} plan={self.fingerprint}\n")
        handle.flush()
        self._handle = handle
        return self

    def update_fingerprint(self, fingerprint: str) -> None:
        self.fingerprint = fingerprint
        if self._handle is not None:
            self._handle.seek(0)
            self._handle.truncate()
            self._handle.write(f"pid={os.getpid()} plan={self.fingerprint}\n")
            self._handle.flush()

    def __exit__(self, *_exc: object) -> None:
        if self._handle is not None:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            self._handle.close()
            self._handle = None


@contextmanager
def _termination_signals() -> Iterator[None]:
    """Turn TERM/HUP into exceptions so active process groups are cleaned up."""

    if threading.current_thread() is not threading.main_thread():
        yield
        return

    watched = (signal.SIGTERM, signal.SIGHUP)
    previous = {signum: signal.getsignal(signum) for signum in watched}

    def handle(signum: int, _frame: object) -> None:
        raise SchedulerSignal(signum)

    for signum in watched:
        signal.signal(signum, handle)
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _terminate_running(running: Mapping[str, RunningJob], log: DriverLog) -> None:
    for item in running.values():
        if item.process.poll() is None:
            log.write(f"Stopping {item.job.key} on GPU {item.gpu}")
            try:
                os.killpg(item.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 20
    survivors: list[RunningJob] = []
    for item in running.values():
        try:
            remaining = max(0.0, deadline - time.monotonic())
            item.process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            survivors.append(item)
    for item in survivors:
        if item.process.poll() is None:
            try:
                os.killpg(item.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            item.process.wait()
    for item in running.values():
        item.log_handle.close()


def _run_jobs_locked(
    jobs: Sequence[Job],
    *,
    repo_root: Path,
    out_root: Path,
    gpus: Sequence[str],
    max_parallel: int,
    resume: bool = True,
    poll_seconds: float = 2.0,
) -> int:
    """Run a plan to completion, continuing unrelated work after failures."""

    if not jobs:
        raise ValueError("The scheduler plan is empty.")
    if not gpus:
        raise ValueError("At least one GPU id is required.")
    if len(gpus) != len(set(gpus)):
        raise ValueError("GPU ids must be unique.")
    if max_parallel < 1:
        raise ValueError("--max-parallel must be at least 1.")
    if poll_seconds <= 0:
        raise ValueError("--poll-seconds must be greater than 0.")
    max_parallel = min(max_parallel, len(gpus))

    fingerprint = plan_fingerprint(jobs)
    state_root = out_root / "_scheduler_state" / fingerprint
    driver_log = DriverLog(out_root / "_scheduler.log")
    pending = {job.key: job for job in jobs}
    succeeded: set[str] = set()
    failed: set[str] = set()
    blocked: set[str] = set()
    running: dict[str, RunningJob] = {}
    free_gpus = list(gpus)

    try:
        driver_log.write(
            f"Plan {fingerprint}: {len(jobs)} jobs, GPUs {','.join(gpus)}, "
            f"concurrency {max_parallel}"
        )
        if resume:
            for key in list(pending):
                marker = state_root / f"{key}.exit"
                if _successful_marker(marker, pending[key]):
                    succeeded.add(key)
                    pending.pop(key)
                    driver_log.write(f"[skip] {key} already completed")

        while pending or running:
            # Return finished jobs' GPUs to the queue.
            for key, item in list(running.items()):
                code = item.process.poll()
                if code is None:
                    continue
                item.log_handle.close()
                elapsed = time.monotonic() - item.started_at
                missing_outputs = [
                    path for path in item.job.expected_outputs if not path.is_file()
                ]
                if code == 0 and missing_outputs:
                    code = 97
                    driver_log.write(
                        f"[FAIL] {key} exited successfully but did not produce: "
                        + ", ".join(str(path) for path in missing_outputs)
                    )
                _write_exit_marker(state_root / f"{key}.exit", code)
                if code == 0:
                    succeeded.add(key)
                    driver_log.write(
                        f"[ok] {key} on GPU {item.gpu} ({elapsed / 60:.1f} min)"
                    )
                else:
                    failed.add(key)
                    driver_log.write(
                        f"[FAIL] {key} on GPU {item.gpu}, exit {code}; "
                        f"see {item.job.log_path}"
                    )
                free_gpus.append(item.gpu)
                running.pop(key)

            while True:
                runnable, newly_blocked = ready_jobs(
                    pending.values(), succeeded=succeeded, failed=failed | blocked
                )
                if not newly_blocked:
                    break
                for job in newly_blocked:
                    dependency = sorted(job.dependencies & (failed | blocked))
                    blocked.add(job.key)
                    pending.pop(job.key)
                    driver_log.write(
                        f"[BLOCKED] {job.key}; failed dependency: "
                        f"{', '.join(dependency)}"
                    )

            running_keys = set(running)
            runnable = [job for job in runnable if job.key not in running_keys]
            while (
                runnable
                and free_gpus
                and len(running) < max_parallel
            ):
                job = runnable.pop(0)
                gpu = free_gpus.pop(0)
                job.log_path.parent.mkdir(parents=True, exist_ok=True)
                handle = job.log_path.open("ab", buffering=0)
                header = (
                    f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} "
                    f"{job.key} GPU {gpu} ===\n"
                )
                handle.write(header.encode())
                environment = os.environ.copy()
                environment.update(job.env_dict())
                environment["CUDA_VISIBLE_DEVICES"] = gpu
                environment["PYTHONUNBUFFERED"] = "1"
                process: subprocess.Popen[bytes] | None = None
                try:
                    process = subprocess.Popen(
                        job.command,
                        cwd=repo_root,
                        env=environment,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    pending.pop(job.key)
                    running[job.key] = RunningJob(
                        job=job,
                        gpu=gpu,
                        process=process,
                        log_handle=handle,
                        started_at=time.monotonic(),
                    )
                except BaseException:
                    # Clean up a process created just before an interrupt.
                    if process is not None and process.poll() is None:
                        try:
                            os.killpg(process.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            try:
                                os.killpg(process.pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                            process.wait()
                    handle.close()
                    free_gpus.insert(0, gpu)
                    raise
                driver_log.write(
                    f"[start] {job.key} on GPU {gpu} -> {job.log_path}"
                )

            if not running and pending:
                unresolved = ", ".join(sorted(pending))
                raise RuntimeError(f"No runnable jobs remain: {unresolved}")
            if running:
                time.sleep(poll_seconds)

        driver_log.write("=== Cross-model scheduler summary ===")
        driver_log.write(
            f"{len(succeeded)} succeeded, {len(failed)} failed, "
            f"{len(blocked)} blocked"
        )
        return 0 if not failed and not blocked else 1
    except SchedulerSignal as exc:
        driver_log.write(
            f"Scheduler received signal {exc.signum}; terminating active jobs."
        )
        _terminate_running(running, driver_log)
        return 128 + exc.signum
    except KeyboardInterrupt:
        driver_log.write("Scheduler interrupted; terminating active jobs.")
        _terminate_running(running, driver_log)
        return 130
    except BaseException:
        _terminate_running(running, driver_log)
        raise
    finally:
        driver_log.close()


def run_jobs(
    jobs: Sequence[Job],
    *,
    repo_root: Path,
    out_root: Path,
    gpus: Sequence[str],
    max_parallel: int,
    resume: bool = True,
    poll_seconds: float = 2.0,
) -> int:
    """Run jobs under the same lock and signal handling used by the CLI."""

    fingerprint = plan_fingerprint(jobs)
    with SchedulerLock(out_root, fingerprint), _termination_signals():
        return _run_jobs_locked(
            jobs,
            repo_root=repo_root,
            out_root=out_root,
            gpus=gpus,
            max_parallel=max_parallel,
            resume=resume,
            poll_seconds=poll_seconds,
        )


def _format_job(job: Job) -> str:
    dependencies = ",".join(sorted(job.dependencies)) or "-"
    command = " ".join(shlex.quote(part) for part in job.command)
    worker_note = ""
    if "SUITE_WORKERS" in job.env_dict():
        worker_note = f"\n  workers: {job.env_dict()['SUITE_WORKERS']}"
    return (
        f"{job.key}\n"
        f"  deps: {dependencies}\n"
        f"  log:  {job.log_path}\n"
        f"  cmd:  {command}"
        f"{worker_note}"
    )


def _validate_inputs(
    *,
    repo_root: Path,
    out_root: Path,
    datasets: Sequence[str],
    models: Sequence[str],
    phases: Sequence[str],
    index_dir: Path,
    del_off_mode: str,
) -> None:
    missing = [
        repo_root / DATASETS[dataset]
        for dataset in datasets
        if not (repo_root / DATASETS[dataset]).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing prompt files:\n  " + "\n  ".join(str(path) for path in missing)
        )
    need_co_lmlm = "co-lmlm" in models
    if need_co_lmlm and not index_dir.is_dir():
        raise FileNotFoundError(f"Co-LMLM index directory not found: {index_dir}")
    nonstandard = set(phases) - {"standard"}
    if need_co_lmlm and nonstandard and "standard" not in phases:
        missing_prerequisites: list[Path] = []
        for dataset in datasets:
            prompt_path = repo_root / DATASETS[dataset]
            output_dir = out_root / "co-lmlm" / dataset
            required = (
                output_dir / f"{prompt_path.stem}_results.jsonl",
                output_dir
                / f"{prompt_path.stem}_full"
                / "full_results.jsonl",
                output_dir / f"{prompt_path.stem}_audit_config.json",
            )
            missing_prerequisites.extend(path for path in required if not path.is_file())
        if missing_prerequisites:
            raise FileNotFoundError(
                "Co-LMLM phase-only runs require completed standard outputs. "
                "Include 'standard' in --phases or provide:\n  "
                + "\n  ".join(str(path) for path in missing_prerequisites)
            )
        for dataset in datasets:
            prompt_path = repo_root / DATASETS[dataset]
            config_path = (
                out_root
                / "co-lmlm"
                / dataset
                / f"{prompt_path.stem}_audit_config.json"
            )
            config = json.loads(config_path.read_text(encoding="utf-8"))
            stored_mode = config.get("del_off_mode")
            if stored_mode != del_off_mode:
                raise ValueError(
                    f"{config_path} uses DEL-OFF mode {stored_mode!r}, but this "
                    f"phase-only run requests {del_off_mode!r}. Include "
                    "'standard' in --phases and use a fresh output root."
                )


def _validate_runtime_options(
    *,
    gpus: Sequence[str],
    max_parallel: int,
    workers_per_job: int,
    poll_seconds: float,
) -> None:
    if not gpus:
        raise ValueError("At least one GPU id is required.")
    if len(gpus) != len(set(gpus)):
        raise ValueError("GPU ids must be unique.")
    if max_parallel < 1:
        raise ValueError("--max-parallel must be at least 1.")
    if workers_per_job < 1:
        raise ValueError("--workers-per-job must be at least 1.")
    if poll_seconds <= 0:
        raise ValueError("--poll-seconds must be greater than 0.")


def _warm_up(
    *, repo_root: Path, co_lmlm_dir: Path, need_co_lmlm: bool, out_root: Path
) -> None:
    log_path = out_root / "_scheduler_warmup.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab", buffering=0) as handle:
        handle.write(b"\n=== HALO uv sync ===\n")
        subprocess.run(
            ("uv", "sync"),
            cwd=repo_root,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )
        if need_co_lmlm:
            if not co_lmlm_dir.is_dir():
                handle.write(b"\n=== Clone Co-LMLM ===\n")
                subprocess.run(
                    (
                        "git",
                        "clone",
                        "https://github.com/lil-lab/Co-LMLM.git",
                        str(co_lmlm_dir),
                    ),
                    cwd=repo_root.parent,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            expected_source = co_lmlm_dir / "src" / "lmlm" / "eval" / "hf_generate.py"
            if not expected_source.is_file():
                raise FileNotFoundError(
                    f"{co_lmlm_dir} is not a valid Co-LMLM checkout "
                    f"(missing {expected_source.relative_to(co_lmlm_dir)})."
                )
            handle.write(b"\n=== Co-LMLM uv sync ===\n")
            subprocess.run(
                ("uv", "sync"),
                cwd=co_lmlm_dir,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=True,
            )
            # Install OpenBLAS once before jobs can race on dpkg.
            if shutil.which("apt-get") and shutil.which("dpkg"):
                installed = subprocess.run(
                    ("dpkg", "-s", "libopenblas0"),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                ).returncode == 0
                if not installed:
                    handle.write(b"\n=== Install OpenBLAS ===\n")
                    prefix: tuple[str, ...] = ()
                    if getattr(os, "geteuid", lambda: 1)() != 0:
                        if not shutil.which("sudo"):
                            raise PermissionError(
                                "libopenblas0 is missing and sudo is unavailable."
                            )
                        prefix = ("sudo",)
                    subprocess.run(
                        (*prefix, "apt-get", "update"),
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
                    subprocess.run(
                        (*prefix, "apt-get", "install", "-y", "libopenblas0"),
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Schedule the 35-job cross-model HALO evaluation over a GPU pool."
    )
    parser.add_argument(
        "--sets",
        default=os.environ.get("SETS", ",".join(DATASETS)),
        help="Comma-separated prompt sets.",
    )
    parser.add_argument(
        "--models",
        default=os.environ.get("SCHEDULER_MODELS", ",".join(DEFAULT_MODELS)),
        help="Comma-separated model backends.",
    )
    parser.add_argument(
        "--phases",
        default=os.environ.get("SCHEDULER_PHASES", ",".join(CO_LMLM_PHASES)),
        help="Comma-separated Co-LMLM phases.",
    )
    parser.add_argument(
        "--gpus",
        default=os.environ.get("GPUS", "0,1,2,3,4,5,6,7"),
        help="Comma-separated physical GPU ids.",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=None,
        help="Concurrent jobs; defaults to the number of GPU ids.",
    )
    parser.add_argument(
        "--workers-per-job",
        type=int,
        default=int(os.environ.get("SUITE_WORKERS", "1")),
        help=(
            "Co-LMLM worker processes within each one-GPU job. Defaults to "
            "SUITE_WORKERS or 1."
        ),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path(os.environ.get("OUT_ROOT", repo_root / "out-cross-model")),
    )
    parser.add_argument(
        "--co-lmlm-dir",
        type=Path,
        default=Path(
            os.environ.get("CO_LMLM_DIR", repo_root.parent / "Co-LMLM")
        ),
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "INDEX_DIR", repo_root / "data" / "co-lmlm-fineweb-wiki-index"
            )
        ),
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore successful scheduler markers (audit-level resume still applies).",
    )
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--detach",
        action="store_true",
        help="Start the scheduler in the background and return immediately.",
    )
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument(
        "audit_args",
        nargs=argparse.REMAINDER,
        help="Arguments after '--' are forwarded to every audit job.",
    )
    args = parser.parse_args(argv)
    args.repo_root = repo_root
    if args.audit_args and args.audit_args[0] == "--":
        args.audit_args = args.audit_args[1:]
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    datasets = _csv(args.sets)
    models = _csv(args.models)
    phases = _csv(args.phases)
    gpus = _csv(args.gpus)
    out_root = args.out_root.expanduser().resolve()
    co_lmlm_dir = args.co_lmlm_dir.expanduser().resolve()
    index_dir = args.index_dir.expanduser().resolve()
    job_options = dict(
        repo_root=args.repo_root,
        out_root=out_root,
        co_lmlm_dir=co_lmlm_dir,
        index_dir=index_dir,
        datasets=datasets,
        models=models,
        phases=phases,
        audit_args=args.audit_args,
        workers_per_job=args.workers_per_job,
    )
    jobs = build_jobs(**job_options)
    max_parallel = args.max_parallel if args.max_parallel is not None else len(gpus)
    _validate_runtime_options(
        gpus=gpus,
        max_parallel=max_parallel,
        workers_per_job=args.workers_per_job,
        poll_seconds=args.poll_seconds,
    )

    if args.dry_run:
        print(
            f"Plan {plan_fingerprint(jobs)}: {len(jobs)} jobs "
            f"({len(gpus)} GPUs)"
        )
        for job in sorted(jobs, key=lambda item: (-item.priority, item.key)):
            print(_format_job(job))
        return 0

    _validate_inputs(
        repo_root=args.repo_root,
        out_root=out_root,
        datasets=datasets,
        models=models,
        phases=phases,
        index_dir=index_dir,
        del_off_mode=os.environ.get("DEL_OFF_MODE", "null-retrieval"),
    )
    if args.detach:
        child_args = list(sys.argv[1:] if argv is None else argv)
        child_args.remove("--detach")
        out_root.mkdir(parents=True, exist_ok=True)
        with open(os.devnull, "wb") as sink:
            process = subprocess.Popen(
                (sys.executable, "-m", "halo.scheduler", *child_args),
                cwd=args.repo_root,
                stdin=subprocess.DEVNULL,
                stdout=sink,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        print(f"Submitted scheduler as PID {process.pid}.")
        print(f"Driver log: tail -F {out_root / '_scheduler.log'}")
        print(f"Job logs:   {out_root / '_scheduler_jobs'}")
        return 0

    fingerprint = plan_fingerprint(jobs)
    try:
        with (
            SchedulerLock(out_root, fingerprint) as scheduler_lock,
            _termination_signals(),
        ):
            if not args.skip_warmup:
                warmup_log = DriverLog(out_root / "_scheduler.log")
                warmup_log.write("Starting one-time environment warmup.")
                warmup_log.close()
                try:
                    _warm_up(
                        repo_root=args.repo_root,
                        co_lmlm_dir=co_lmlm_dir,
                        need_co_lmlm="co-lmlm" in models,
                        out_root=out_root,
                    )
                except (OSError, subprocess.SubprocessError) as exc:
                    failure_log = DriverLog(out_root / "_scheduler.log")
                    failure_log.write(
                        f"[FAIL] Environment warmup failed: {exc}; see "
                        f"{out_root / '_scheduler_warmup.log'}"
                    )
                    failure_log.close()
                    return 1
                # Rebuild the fingerprint after warmup changes external code.
                jobs = build_jobs(**job_options)
                fingerprint = plan_fingerprint(jobs)
                scheduler_lock.update_fingerprint(fingerprint)
            return _run_jobs_locked(
                jobs,
                repo_root=args.repo_root,
                out_root=out_root,
                gpus=gpus,
                max_parallel=max_parallel,
                resume=not args.no_resume,
                poll_seconds=args.poll_seconds,
            )
    except SchedulerAlreadyRunning as exc:
        conflict_log = DriverLog(out_root / "_scheduler.log")
        conflict_log.write(f"[FAIL] {exc}")
        conflict_log.close()
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except SchedulerSignal as exc:
        termination_log = DriverLog(out_root / "_scheduler.log")
        termination_log.write(
            f"Scheduler received signal {exc.signum} during warmup; stopped."
        )
        termination_log.close()
        return 128 + exc.signum


if __name__ == "__main__":
    raise SystemExit(main())
