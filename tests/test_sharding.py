import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from test_adversary import (
    FakeGenerator as AdvFakeGenerator,
    FakeVectorEntry,
    FakeVectorIndex,
)
from test_entanglement import (
    PROMPT_A,
    PROMPT_B,
    _sweep_setup,
    _write_sweep_prompts,
)
from test_reuse import (
    _assert_run_dirs_equal,
    _make_backend,
    _wired_run,
    _write_prompts,
)

from halo.cli.persistence import (
    atomic_write_text,
    parse_shard,
    read_partial_jsonl,
)
from halo.cli.runner import run_adversarial_eval, run_entanglement_sweep
from halo.core.neighbors import NeighborConfig
from halo.interventions.adversary import AdversarialConfig
from halo.interventions.closure import ClosureConfig
from halo.interventions.errors import AuditIntegrationError
from models.co_lmlm.backend import CoLMLMAuditBackend


def test_parse_shard() -> None:
    assert parse_shard("0/2") == (0, 2)
    assert parse_shard("3/4") == (3, 4)
    for bad in ("2/2", "-1/2", "1", "a/b", "1/2/3"):
        with pytest.raises(ValueError):
            parse_shard(bad)


def test_read_partial_jsonl_tail_tolerance(tmp_path) -> None:
    path = tmp_path / "rows.partial.jsonl"
    path.write_text('{"a": 1}\n{"b": 2}\n{"torn', encoding="utf-8")
    assert read_partial_jsonl(path) == [{"a": 1}, {"b": 2}]

    corrupt = tmp_path / "corrupt.partial.jsonl"
    corrupt.write_text('{"a": 1}\n{"torn\n{"b": 2}\n', encoding="utf-8")
    with pytest.raises(AuditIntegrationError, match="Corrupt partial"):
        read_partial_jsonl(corrupt)


def test_atomic_write_text(tmp_path) -> None:
    path = tmp_path / "nested" / "file.json"
    atomic_write_text(path, '{"x": 1}')
    assert path.read_text(encoding="utf-8") == '{"x": 1}'
    atomic_write_text(path, '{"x": 2}')
    assert path.read_text(encoding="utf-8") == '{"x": 2}'
    assert list(path.parent.iterdir()) == [path]


def test_standard_shard_merge_equivalence(tmp_path) -> None:
    """Two fact-striped workers plus a finalize run produce byte-identical
    canonical artifacts to a single unsharded run."""
    prompt_path = _write_prompts(tmp_path / "prompts.jsonl")

    backend_ref, _ = _make_backend()
    _wired_run(
        prompt_path=prompt_path,
        output_dir=tmp_path / "reference",
        backend=backend_ref,
        full_dir=tmp_path / "full_ref",
    )

    sharded_dir = tmp_path / "sharded"
    full_dir = tmp_path / "full_sharded"
    worker_calls = []
    for index in range(2):
        backend_worker, gen_worker = _make_backend()
        _wired_run(
            prompt_path=prompt_path,
            output_dir=sharded_dir,
            backend=backend_worker,
            full_dir=full_dir,
            shard=(index, 2),
        )
        worker_calls.append(gen_worker.generate_calls)
    # Stripe 0 = facts p0, p2 (one audited, one gate-skipped after FULL);
    # stripe 1 = fact p1.
    assert worker_calls == [4, 3]
    assert not (sharded_dir / "prompts_results.jsonl").exists()

    backend_final, gen_final = _make_backend()
    _wired_run(
        prompt_path=prompt_path,
        output_dir=sharded_dir,
        backend=backend_final,
        full_dir=full_dir,
    )
    assert gen_final.generate_calls == 0  # pure merge
    _assert_run_dirs_equal(tmp_path / "reference", sharded_dir, "prompts")
    assert not list(sharded_dir.glob("*.partial.jsonl"))
    assert not list(full_dir.glob("*.partial.jsonl"))
    assert (full_dir / "full_results.jsonl").read_bytes() == (
        tmp_path / "full_ref" / "full_results.jsonl"
    ).read_bytes()


def _normalized_sweep_rows(path: Path) -> dict:
    """Sweep rows keyed by (target, role, subject), with the per-process
    provenance tag and timings dropped (the fingerprint cache does not span
    processes, so `sweep.reused` legitimately differs across shard layouts)."""
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        tag = dict(row.get("sweep") or {})
        key = (tag.get("target_key"), tag.get("role"), row.get("prompt_id"))
        tag.pop("reused", None)
        tag.pop("canary_verified", None)
        tag.pop("canary_origin", None)
        row["sweep"] = tag
        row.pop("generation_metadata", None)
        assert key not in rows
        rows[key] = row
    return rows


def test_sweep_radius_shards_merge_equivalence(tmp_path) -> None:
    """Radius-subset workers with the full grid as identity no longer trip
    the resume-config guard, and the finalize run reproduces the sequential
    sweep's rows and analysis."""
    index_ref, _, backend_ref = _sweep_setup()
    prompt_path = _write_sweep_prompts(tmp_path)
    reference_dir = tmp_path / "reference"
    kwargs = dict(
        radii=(0.9, 0.5),
        closure_config=ClosureConfig(),
        neighbor_config=NeighborConfig(mode="cosine", ball=0.5, cap=20),
    )
    reference = run_entanglement_sweep(
        prompt_path,
        backend_ref,
        index=index_ref,
        output_dir=reference_dir,
        **kwargs,
    )

    sharded_dir = tmp_path / "sharded"
    # Prep run: materialize the FULL pass/closures/neighbors, execute nothing.
    index_prep, generator_prep, backend_prep = _sweep_setup()
    prep = run_entanglement_sweep(
        prompt_path,
        backend_prep,
        index=index_prep,
        output_dir=sharded_dir,
        execute_radii=(),
        **kwargs,
    )
    assert prep["partial"] is True
    assert prep["planned_generations"] == 0
    assert (sharded_dir / "full_results.jsonl").exists()
    assert (sharded_dir / "neighbors.json").exists()
    assert not list(sharded_dir.glob("sweep_rho_*.jsonl"))

    for stripe in range(2):
        index_w, _, backend_w = _sweep_setup()
        summary = run_entanglement_sweep(
            prompt_path,
            backend_w,
            index=index_w,
            output_dir=sharded_dir,
            execute_radii=((0.9, 0.5)[stripe],),
            **kwargs,
        )
        assert summary["partial"] is True
        assert summary["entanglement"] == {}

    index_final, generator_final, backend_final = _sweep_setup()
    merged = run_entanglement_sweep(
        prompt_path,
        backend_final,
        index=index_final,
        output_dir=sharded_dir,
        **kwargs,
    )
    assert generator_final.generate_calls == 0  # pure merge + analysis
    assert merged["partial"] is False
    assert merged["entanglement"] == reference["entanglement"]
    for rho in ("0.9000", "0.5000"):
        assert _normalized_sweep_rows(
            sharded_dir / f"sweep_rho_{rho}.jsonl"
        ) == _normalized_sweep_rows(reference_dir / f"sweep_rho_{rho}.jsonl")


def test_sweep_radius_shard_requires_materialized_full_pass(tmp_path) -> None:
    index, _, backend = _sweep_setup()
    prompt_path = _write_sweep_prompts(tmp_path)
    with pytest.raises(AuditIntegrationError, match="materialize"):
        run_entanglement_sweep(
            prompt_path,
            backend,
            index=index,
            radii=(0.9, 0.5),
            execute_radii=(0.9,),
            closure_config=ClosureConfig(),
            neighbor_config=NeighborConfig(mode="cosine", ball=0.5, cap=20),
            output_dir=tmp_path / "sweep",
        )


def _adv_backend() -> CoLMLMAuditBackend:
    index = FakeVectorIndex(
        [FakeVectorEntry("entry-a", np.asarray([1.0, 0.0]), "Paris", "wiki:France")]
    )
    return CoLMLMAuditBackend(AdvFakeGenerator(index))


def _adv_prompts(tmp_path) -> Path:
    prompt_path = tmp_path / "prompts.jsonl"
    prompt_path.write_text(
        json.dumps({"prompt_id": "pA", "prompt_text": PROMPT_A, "gold_object": "Paris"})
        + "\n"
        + json.dumps(
            {"prompt_id": "pB", "prompt_text": PROMPT_B, "gold_object": "Paris"}
        )
        + "\n",
        encoding="utf-8",
    )
    return prompt_path


def test_adversarial_shard_merge_equivalence(tmp_path) -> None:
    prompt_path = _adv_prompts(tmp_path)

    def _kwargs(backend):
        return dict(
            index=backend.generator.index,
            closure_config=ClosureConfig(predicates=("geometric",), radius=0.85),
            adversarial_config=AdversarialConfig(
                rho=0.85, epsilons=(0.05,), templates=("verbatim",)
            ),
        )

    backend_ref = _adv_backend()
    reference = run_adversarial_eval(
        prompt_path,
        backend_ref,
        output_dir=tmp_path / "reference",
        **_kwargs(backend_ref),
    )

    sharded_dir = tmp_path / "sharded"
    full_dir = tmp_path / "shared_full"
    with pytest.raises(AuditIntegrationError, match="materialize"):
        run_adversarial_eval(
            prompt_path,
            backend_ref,
            output_dir=sharded_dir,
            full_dir=full_dir,
            shard=(0, 2),
            **_kwargs(backend_ref),
        )

    # Materialize the shared FULL pass, as the suite's earlier phases would.
    from halo.cli.full_pass import run_full_pass
    from halo.cli.runner import _load_examples

    run_full_pass(
        _adv_backend(),
        _load_examples(prompt_path, None),
        prompt_path,
        None,
        full_dir,
        12,
    )

    for stripe in range(2):
        backend_w = _adv_backend()
        summary = run_adversarial_eval(
            prompt_path,
            backend_w,
            output_dir=sharded_dir,
            full_dir=full_dir,
            shard=(stripe, 2),
            **_kwargs(backend_w),
        )
        assert summary["partial"] is True
        assert summary["attacked_facts"] == 1
        assert summary["executed_generations"] == 3  # del-off, baseline, attack

    backend_final = _adv_backend()
    merged = run_adversarial_eval(
        prompt_path,
        backend_final,
        output_dir=sharded_dir,
        full_dir=full_dir,
        **_kwargs(backend_final),
    )
    assert merged["partial"] is False
    assert merged["executed_generations"] == 0  # pure merge + analysis
    assert merged["evasion"] == reference["evasion"]
    assert merged["margins"] == reference["margins"]
    assert not list(sharded_dir.glob("*.partial.jsonl"))

    def _rows(path: Path) -> list[dict]:
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            row.pop("generation_metadata", None)
            rows.append(row)
        return rows

    assert _rows(sharded_dir / "adversarial_results.jsonl") == _rows(
        tmp_path / "reference" / "adversarial_results.jsonl"
    )
