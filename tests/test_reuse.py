import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from test_colmlm_backend import FakeGenerator, FakeIndex, FakeSearchResult

from halo.cli.full_pass import FullPassStore
from halo.cli.persistence import (
    AUDIT_SCHEMA_VERSION,
    backend_resume_identity,
    ensure_resume_config,
    prompt_digest,
    read_partial_jsonl,
)
from halo.cli.reporting import save_results
from halo.cli.reuse import GenerationReuseStore, ReuseContext, ReuseSource
from halo.cli.runner import StandardAuditPersistence, _load_examples, run_backend_audit
from halo.core.embeddings import QueryEmbeddingSink
from halo.core.states import DatabaseState
from halo.interventions.errors import AuditIntegrationError
from models.co_lmlm.backend import CoLMLMAuditBackend

STATES = [DatabaseState.FULL, DatabaseState.DEL_ON, DatabaseState.DEL_OFF]


class CountingGenerator(FakeGenerator):
    def __init__(self, index, fail_after: int | None = None):
        super().__init__(index)
        self.generate_calls = 0
        self.fail_after = fail_after

    def _tick(self) -> None:
        self.generate_calls += 1
        if self.fail_after is not None and self.generate_calls > self.fail_after:
            raise RuntimeError("simulated crash")

    def generate(self, prompt):
        self._tick()
        return super().generate(prompt)

    def generate_no_retrieval(self, prompt):
        self._tick()
        return super().generate_no_retrieval(prompt)


def _make_backend(
    del_off_mode: str = "null-retrieval", fail_after: int | None = None
) -> tuple[CoLMLMAuditBackend, CountingGenerator]:
    index = FakeIndex(
        [
            FakeSearchResult(
                id="target-entry",
                score=0.95,
                text_value="Paris",
                metadata={"source_id": "wiki:France"},
            ),
            FakeSearchResult(
                id="neighbor-entry",
                score=0.90,
                text_value="Lyon",
                metadata={"source_id": "wiki:Lyon"},
            ),
        ]
    )
    generator = CountingGenerator(index, fail_after=fail_after)
    return CoLMLMAuditBackend(generator, del_off_mode=del_off_mode), generator


def _write_prompts(path: Path) -> Path:
    rows = [
        # p0/p1 pass the answer-mention gate; p2's gold never matches the
        # retrieved value, so it is skipped after its FULL generation.
        {
            "prompt_id": "p0",
            "fact_id": "f0",
            "prompt_text": "What is the capital of France?",
            "gold_object": "Paris",
        },
        {
            "prompt_id": "p1",
            "fact_id": "f1",
            "prompt_text": "What is the capital of Norway?",
            "gold_object": "Paris",
        },
        {
            "prompt_id": "p2",
            "fact_id": "f2",
            "prompt_text": "What is the capital of Italy?",
            "gold_object": "Berlin",
        },
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return path


def _audit_config_payload(backend, prompt_path: Path) -> dict:
    return {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "mode": "standard-audit",
        "backend": backend_resume_identity(backend),
        "del_off_mode": getattr(backend, "del_off_mode", None),
        "prompt_sha256": prompt_digest(prompt_path),
        "limit": None,
        "max_new_tokens": 12,
        "states": [state.value for state in STATES],
        "bootstrap_oracle_from_full": True,
        "closure": None,
    }


def _context(backend, prompt_path: Path) -> ReuseContext:
    return ReuseContext(
        backend_identity=backend_resume_identity(backend),
        prompt_sha256=prompt_digest(prompt_path),
        limit=None,
        max_new_tokens=12,
        del_off_mode=getattr(backend, "del_off_mode", None),
    )


def _wired_run(
    *,
    prompt_path: Path,
    output_dir: Path,
    backend,
    full_dir: Path | None = None,
    reuse_paths: tuple = (),
    canary_rate: float = 0.0,
    shard: tuple[int, int] | None = None,
):
    """Mirrors run_audit.py's standard-path wiring around run_backend_audit."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = prompt_path.stem
    ensure_resume_config(
        output_dir / f"{stem}_audit_config.json",
        _audit_config_payload(backend, prompt_path),
        legacy_artifacts=(output_dir / f"{stem}_results.jsonl",),
    )
    resume = StandardAuditPersistence(
        output_dir=output_dir, stem=stem, expected_states=STATES, shard=shard
    )
    full_store = (
        FullPassStore(
            backend=backend,
            examples=_load_examples(prompt_path, None),
            prompt_path=prompt_path,
            limit=None,
            output_dir=full_dir,
            max_new_tokens=12,
            shard=shard,
        )
        if full_dir is not None
        else None
    )
    reuse_store = (
        GenerationReuseStore(
            backend=backend,
            context=_context(backend, prompt_path),
            source_paths=tuple(reuse_paths),
            canary_rate=canary_rate,
            max_new_tokens=12,
            output_dir=output_dir,
        )
        if reuse_paths
        else None
    )
    sink = QueryEmbeddingSink()
    results = run_backend_audit(
        prompt_path=prompt_path,
        backend=backend,
        states=STATES,
        max_new_tokens=12,
        bootstrap_oracle_from_full=True,
        embedding_sink=sink,
        skip_log_path=output_dir / f"{stem}_skipped_facts.jsonl",
        full_store=full_store,
        reuse_store=reuse_store,
        resume=resume,
        shard=shard,
    )
    if shard is not None:
        if full_store is not None:
            full_store.close()
        return results, reuse_store, full_store
    save_results(results, output_dir / f"{stem}_results.jsonl")
    if len(sink):
        sink.save(output_dir / f"{stem}_query_embeddings.npz")
    if full_store is not None:
        full_store.finalize()
    resume.cleanup()
    return results, reuse_store, full_store


def _bare_run(*, prompt_path: Path, output_dir: Path, backend):
    """Today's flagless path: no persistence, no reuse."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = prompt_path.stem
    sink = QueryEmbeddingSink()
    results = run_backend_audit(
        prompt_path=prompt_path,
        backend=backend,
        states=STATES,
        max_new_tokens=12,
        bootstrap_oracle_from_full=True,
        embedding_sink=sink,
        skip_log_path=output_dir / f"{stem}_skipped_facts.jsonl",
    )
    save_results(results, output_dir / f"{stem}_results.jsonl")
    if len(sink):
        sink.save(output_dir / f"{stem}_query_embeddings.npz")
    return results


def _assert_npz_equal(left: Path, right: Path) -> None:
    with np.load(left) as a, np.load(right) as b:
        assert sorted(a.files) == sorted(b.files)
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key])


def _assert_run_dirs_equal(left: Path, right: Path, stem: str) -> None:
    assert (left / f"{stem}_results.jsonl").read_bytes() == (
        right / f"{stem}_results.jsonl"
    ).read_bytes()
    assert (left / f"{stem}_skipped_facts.jsonl").read_bytes() == (
        right / f"{stem}_skipped_facts.jsonl"
    ).read_bytes()
    _assert_npz_equal(
        left / f"{stem}_query_embeddings.npz",
        right / f"{stem}_query_embeddings.npz",
    )


def test_golden_equivalence_bare_vs_wired_vs_reused(tmp_path) -> None:
    """The primary gate: legacy path, persistence-wired path, and a fully
    reuse-served second phase all produce byte-identical artifacts."""
    prompt_path = _write_prompts(tmp_path / "prompts.jsonl")

    backend_a, gen_a = _make_backend()
    _bare_run(prompt_path=prompt_path, output_dir=tmp_path / "bare", backend=backend_a)

    backend_b, gen_b = _make_backend()
    _wired_run(
        prompt_path=prompt_path,
        output_dir=tmp_path / "wired",
        backend=backend_b,
        full_dir=tmp_path / "full",
    )
    assert gen_a.generate_calls == gen_b.generate_calls == 7  # 2x3 states + 1 skip

    _assert_run_dirs_equal(tmp_path / "bare", tmp_path / "wired", "prompts")
    # Wired run leaves no partials behind, and the shared FULL pass exists.
    assert not list((tmp_path / "wired").glob("*partial*"))
    assert (tmp_path / "full" / "full_results.jsonl").exists()
    assert (tmp_path / "full" / "full_all_query_embeddings.npz").exists()

    # Phase 2: everything served from phase 1 (canary 1.0 regenerates and
    # verifies every reuse, proving the fingerprint claims on this backend).
    backend_c, gen_c = _make_backend()
    _, reuse_store, _ = _wired_run(
        prompt_path=prompt_path,
        output_dir=tmp_path / "reused",
        backend=backend_c,
        full_dir=tmp_path / "full",
        reuse_paths=(tmp_path / "wired" / "prompts_results.jsonl",),
        canary_rate=1.0,
    )
    _assert_run_dirs_equal(tmp_path / "bare", tmp_path / "reused", "prompts")
    summary = reuse_store.summary()
    # FULL rows come from the shared full dir; DEL-ON/DEL-OFF are served.
    assert summary["served"] == 4
    assert summary["served_by_state"] == {"DEL-ON": 2, "DEL-OFF": 2}
    assert summary["canary_checks"] == 4
    assert gen_c.generate_calls == 4  # canary regenerations only


def test_reuse_serves_all_states_without_generating(tmp_path) -> None:
    prompt_path = _write_prompts(tmp_path / "prompts.jsonl")
    backend_a, _ = _make_backend()
    _wired_run(
        prompt_path=prompt_path,
        output_dir=tmp_path / "phase1",
        backend=backend_a,
        full_dir=tmp_path / "full1",
    )

    backend_b, gen_b = _make_backend()
    _, reuse_store, _ = _wired_run(
        prompt_path=prompt_path,
        output_dir=tmp_path / "phase2",
        backend=backend_b,
        reuse_paths=(tmp_path / "phase1" / "prompts_results.jsonl",),
    )
    # Audited facts' FULL rows live in phase 1's results file and are served;
    # only the skipped fact's FULL (never written there) regenerates.
    assert gen_b.generate_calls == 1
    assert reuse_store.summary()["served_by_state"] == {
        "FULL": 2,
        "DEL-ON": 2,
        "DEL-OFF": 2,
    }
    _assert_run_dirs_equal(tmp_path / "phase1", tmp_path / "phase2", "prompts")


def test_del_off_mode_mismatch_disables_only_del_off(tmp_path) -> None:
    """The suite's del-off phase: same FULL pass and manifests, different
    DEL-OFF control. Only the DEL-OFF arm may generate."""
    prompt_path = _write_prompts(tmp_path / "prompts.jsonl")
    backend_a, _ = _make_backend(del_off_mode="null-retrieval")
    _wired_run(
        prompt_path=prompt_path,
        output_dir=tmp_path / "phase1",
        backend=backend_a,
        full_dir=tmp_path / "full",
    )

    backend_b, gen_b = _make_backend(del_off_mode="forbid-token")
    results, reuse_store, _ = _wired_run(
        prompt_path=prompt_path,
        output_dir=tmp_path / "deloff",
        backend=backend_b,
        full_dir=tmp_path / "full",
        reuse_paths=(tmp_path / "phase1" / "prompts_results.jsonl",),
    )
    # 2 forbid-token DEL-OFF generations; FULL served by the full dir,
    # DEL-ON served by fingerprint.
    assert gen_b.generate_calls == 2
    summary = reuse_store.summary()
    assert summary["served_by_state"] == {"DEL-ON": 2}
    assert summary["generated_by_state"] == {"DEL-OFF": 2}
    del_off_rows = [row for row in results if row["state"] == "DEL-OFF"]
    assert all(
        row["retrieval_trace"]["del_off_mode"] == "forbid-token"
        for row in del_off_rows
    )
    # FULL/DEL-ON rows byte-match phase 1's.
    phase1_rows = [
        json.loads(line)
        for line in (tmp_path / "phase1" / "prompts_results.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    for state in ("FULL", "DEL-ON"):
        assert [row for row in results if row["state"] == state] == [
            row for row in phase1_rows if row["state"] == state
        ]


def test_standard_audit_interrupt_resume_equivalence(tmp_path) -> None:
    prompt_path = _write_prompts(tmp_path / "prompts.jsonl")

    backend_ref, _ = _make_backend()
    _wired_run(
        prompt_path=prompt_path,
        output_dir=tmp_path / "reference",
        backend=backend_ref,
        full_dir=tmp_path / "full_ref",
    )

    # Crash during fact p1's FULL generation (call #4 of 7).
    backend_crash, gen_crash = _make_backend(fail_after=3)
    with pytest.raises(RuntimeError, match="simulated crash"):
        _wired_run(
            prompt_path=prompt_path,
            output_dir=tmp_path / "resumed",
            backend=backend_crash,
            full_dir=tmp_path / "full_resumed",
        )
    partials = list((tmp_path / "resumed").glob("*.partial.jsonl"))
    assert partials, "the crashed run must leave partials behind"

    backend_resume, gen_resume = _make_backend()
    _wired_run(
        prompt_path=prompt_path,
        output_dir=tmp_path / "resumed",
        backend=backend_resume,
        full_dir=tmp_path / "full_resumed",
    )
    # Fact p0 (3 generations) was resumed from disk.
    assert gen_resume.generate_calls == 4
    _assert_run_dirs_equal(tmp_path / "reference", tmp_path / "resumed", "prompts")
    _assert_npz_equal(
        tmp_path / "full_ref" / "full_query_embeddings.npz",
        tmp_path / "full_resumed" / "full_query_embeddings.npz",
    )
    _assert_npz_equal(
        tmp_path / "full_ref" / "full_all_query_embeddings.npz",
        tmp_path / "full_resumed" / "full_all_query_embeddings.npz",
    )
    assert not list((tmp_path / "resumed").glob("*.partial.jsonl"))
    assert not list((tmp_path / "full_resumed").glob("*.partial.jsonl"))


def test_full_pass_interrupt_resume_and_legacy_dir(tmp_path) -> None:
    prompt_path = _write_prompts(tmp_path / "prompts.jsonl")
    examples = _load_examples(prompt_path, None)

    backend_crash, _ = _make_backend(fail_after=2)
    store = FullPassStore(
        backend=backend_crash,
        examples=examples,
        prompt_path=prompt_path,
        limit=None,
        output_dir=tmp_path / "full",
        max_new_tokens=12,
    )
    with pytest.raises(RuntimeError, match="simulated crash"):
        store.finalize()
    store.close()
    assert not (tmp_path / "full" / "full_results.jsonl").exists()

    backend_resume, gen_resume = _make_backend()
    resumed = FullPassStore(
        backend=backend_resume,
        examples=examples,
        prompt_path=prompt_path,
        limit=None,
        output_dir=tmp_path / "full",
        max_new_tokens=12,
    )
    assert resumed.missing_keys() == ["p2"]
    resumed.finalize()
    assert gen_resume.generate_calls == 1
    assert not list((tmp_path / "full").glob("*.partial.jsonl"))

    backend_fresh, _ = _make_backend()
    reference = FullPassStore(
        backend=backend_fresh,
        examples=examples,
        prompt_path=prompt_path,
        limit=None,
        output_dir=tmp_path / "full_fresh",
        max_new_tokens=12,
    )
    reference.finalize()
    _assert_npz_equal(
        tmp_path / "full" / "full_query_embeddings.npz",
        tmp_path / "full_fresh" / "full_query_embeddings.npz",
    )
    rows_resumed = read_partial_jsonl(tmp_path / "full" / "full_results.jsonl")
    rows_fresh = read_partial_jsonl(tmp_path / "full_fresh" / "full_results.jsonl")
    assert rows_resumed == rows_fresh

    # A complete legacy dir (no all-events npz) must refuse `get`.
    (tmp_path / "full" / "full_all_query_embeddings.npz").unlink()
    legacy = FullPassStore(
        backend=backend_fresh,
        examples=examples,
        prompt_path=prompt_path,
        limit=None,
        output_dir=tmp_path / "full",
        max_new_tokens=12,
    )
    with pytest.raises(AuditIntegrationError, match="regenerate"):
        legacy.get("p0")


def test_reuse_source_compat_guards(tmp_path) -> None:
    prompt_path = _write_prompts(tmp_path / "prompts.jsonl")
    backend, _ = _make_backend()
    _wired_run(
        prompt_path=prompt_path,
        output_dir=tmp_path / "phase1",
        backend=backend,
        full_dir=tmp_path / "full",
    )
    results_path = tmp_path / "phase1" / "prompts_results.jsonl"

    # Missing file: warns and serves nothing, but does not raise.
    empty = ReuseSource(
        tmp_path / "phase1" / "missing_results.jsonl",
        _context(backend, prompt_path),
        log=lambda _: None,
    )
    assert empty.row("p0", DatabaseState.FULL) is None

    # Missing config sidecar: refuse.
    bare_results = tmp_path / "bare" / "prompts_results.jsonl"
    bare_backend, _ = _make_backend()
    _bare_run(
        prompt_path=prompt_path, output_dir=tmp_path / "bare", backend=bare_backend
    )
    with pytest.raises(AuditIntegrationError, match="no prompts_audit_config"):
        ReuseSource(bare_results, _context(backend, prompt_path))

    # Prompt-digest mismatch: refuse loudly.
    other_prompts = tmp_path / "other.jsonl"
    other_prompts.write_text(
        '{"prompt_id":"x","fact_id":"x","prompt_text":"Q?","gold_object":"A"}\n',
        encoding="utf-8",
    )
    with pytest.raises(AuditIntegrationError, match="prompt_sha256"):
        ReuseSource(results_path, _context(backend, other_prompts))

    # Sweep-tagged rows: refuse.
    tagged_dir = tmp_path / "tagged"
    tagged_dir.mkdir()
    row = json.loads(results_path.read_text(encoding="utf-8").splitlines()[0])
    row["sweep"] = {"rho": 0.9}
    (tagged_dir / "prompts_results.jsonl").write_text(
        json.dumps(row) + "\n", encoding="utf-8"
    )
    (tagged_dir / "prompts_audit_config.json").write_bytes(
        (tmp_path / "phase1" / "prompts_audit_config.json").read_bytes()
    )
    with pytest.raises(AuditIntegrationError, match="sweep/adversarial"):
        ReuseSource(
            tagged_dir / "prompts_results.jsonl", _context(backend, prompt_path)
        )


def test_reuse_canary_mismatch_raises_with_report(tmp_path) -> None:
    prompt_path = _write_prompts(tmp_path / "prompts.jsonl")
    backend_a, _ = _make_backend()
    _wired_run(
        prompt_path=prompt_path,
        output_dir=tmp_path / "phase1",
        backend=backend_a,
        full_dir=tmp_path / "full",
    )

    # Same resume identity, but the index now yields a different value: the
    # cross-phase claim is false and the canary must catch it.
    backend_b, _ = _make_backend()
    backend_b.generator.index.results[0].text_value = "London"
    with pytest.raises(AuditIntegrationError, match="cross_phase_fingerprint"):
        _wired_run(
            prompt_path=prompt_path,
            output_dir=tmp_path / "phase2",
            backend=backend_b,
            reuse_paths=(tmp_path / "phase1" / "prompts_results.jsonl",),
            canary_rate=1.0,
        )
    reports = list((tmp_path / "phase2").glob("reuse_canary_failure_*.json"))
    assert len(reports) == 1
    payload = json.loads(reports[0].read_text(encoding="utf-8"))
    assert payload["reused"]["model_output"] != payload["generated"]["model_output"]


def test_reuse_manifest_sidecar(tmp_path) -> None:
    prompt_path = _write_prompts(tmp_path / "prompts.jsonl")
    backend_a, _ = _make_backend()
    _wired_run(
        prompt_path=prompt_path,
        output_dir=tmp_path / "phase1",
        backend=backend_a,
        full_dir=tmp_path / "full",
    )
    backend_b, _ = _make_backend()
    _, reuse_store, _ = _wired_run(
        prompt_path=prompt_path,
        output_dir=tmp_path / "phase2",
        backend=backend_b,
        full_dir=tmp_path / "full",
        reuse_paths=(tmp_path / "phase1" / "prompts_results.jsonl",),
    )
    manifest_path = tmp_path / "phase2" / "prompts_reuse_manifest.json"
    reuse_store.write_manifest(manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["served"] == 4
    assert payload["generated"] == 0
    assert len(payload["entries"]) == 4
    assert all(
        entry["source"].endswith("prompts_results.jsonl")
        for entry in payload["entries"]
    )


def test_persistence_requires_fact_identifiers(tmp_path) -> None:
    prompt_path = tmp_path / "prompts.jsonl"
    prompt_path.write_text(
        '{"prompt_text":"What is the capital of France?","gold_object":"Paris"}\n',
        encoding="utf-8",
    )
    backend, _ = _make_backend()
    resume = StandardAuditPersistence(
        output_dir=tmp_path, stem="prompts", expected_states=STATES
    )
    with pytest.raises(AuditIntegrationError, match="prompt_id/fact_id"):
        run_backend_audit(
            prompt_path=prompt_path,
            backend=backend,
            states=STATES,
            bootstrap_oracle_from_full=True,
            resume=resume,
        )
    resume.close()
