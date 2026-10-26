from __future__ import annotations

import json
from pathlib import Path

import pytest

from halo.cli.persistence import prompt_facts_digest
from halo.core.examples import AuditExample, DeletionManifest
from halo.paraphrase import (
    analyze_policy,
    build_paraphrase_sets,
    canonical_key_of,
    main,
    mcnemar_exact,
)


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _canonical(fact_id: str, prompt: str, gold: str, subject: str) -> dict:
    return {
        "prompt_id": fact_id,
        "fact_id": fact_id,
        "prompt_text": prompt,
        "gold_object": gold,
        "answer_aliases": [],
        "subject": subject,
    }


def _manifest(*entry_ids: str) -> dict:
    return DeletionManifest(entry_ids=entry_ids, strategy="closure").as_dict()


def _result_rows(key: str, manifest: dict, outputs: dict[str, str], gold: str) -> list[dict]:
    return [
        {
            "fact_id": key.split("::para")[0],
            "prompt_id": key,
            "state": state,
            "prompt": "p",
            "ground_truth": gold,
            "object_aliases": [],
            "deletion_manifest": manifest,
            "model_output": output,
        }
        for state, output in outputs.items()
    ]


@pytest.fixture
def inputs(tmp_path: Path) -> dict[str, Path]:
    canonical = _write_jsonl(
        tmp_path / "prompts_x.jsonl",
        [
            _canonical("f1", "Paris is the capital of", "France", "Paris"),
            _canonical("f2", "Berlin is located in", "Germany", "Berlin"),
            _canonical("f3", "Rome is the capital of", "Italy", "Rome"),
        ],
    )
    variants = _write_jsonl(
        tmp_path / "prompts_x_variants.jsonl",
        [
            {"fact_id": "f1", "prompt_text": "Paris serves as the capital of", "variant": "paraphrase"},
            {"fact_id": "f1", "prompt_text": "Paris is the capital of", "variant": "paraphrase"},
            {"fact_id": "f1", "prompt_text": "Paris, the capital of France, is in", "variant": "paraphrase"},
            {"fact_id": "f1", "prompt_text": "The city of Paris is capital of", "variant": "paraphrase",
             "variant_source": "pararel", "extra_answer_aliases": ["French Republic"]},
            {"fact_id": "f1", "prompt_text": "Paris is the seat of government of", "variant": "paraphrase"},
            {"fact_id": "f2", "prompt_text": "The country where it lies is", "variant": "paraphrase"},
            {"fact_id": "f2", "prompt_text": "Berlin lies in", "variant": "paraphrase"},
            {"fact_id": "f3", "prompt_text": "Rome is capital city of", "variant": "paraphrase"},
            {"fact_id": "f3-nbr0", "prompt_text": "Milan is in", "variant": "neighborhood"},
            {"fact_id": "unknown", "prompt_text": "Nowhere is in", "variant": "paraphrase"},
        ],
    )
    oracle = _write_jsonl(
        tmp_path / "oracle" / "prompts_x_results.jsonl",
        [
            *_result_rows("f1", _manifest("e1"), {"FULL": "France", "DEL-ON": "x", "DEL-OFF": "x"}, "France"),
            *_result_rows("f2", _manifest("e2"), {"FULL": "Germany", "DEL-ON": "x", "DEL-OFF": "x"}, "Germany"),
            *_result_rows("f3", _manifest("e3"), {"FULL": "Italy", "DEL-ON": "x", "DEL-OFF": "x"}, "Italy"),
        ],
    )
    geometric = _write_jsonl(
        tmp_path / "geometric" / "prompts_x_results.jsonl",
        [
            *_result_rows("f1", _manifest("e1", "e9"), {"FULL": "France", "DEL-ON": "x", "DEL-OFF": "x"}, "France"),
            # f2 skipped by this policy; f3 audited with an empty manifest.
            *_result_rows("f3", _manifest(), {"FULL": "Italy", "DEL-ON": "x", "DEL-OFF": "x"}, "Italy"),
        ],
    )
    return {"canonical": canonical, "variants": variants, "oracle": oracle, "geometric": geometric}


def test_build_freezes_manifests_and_filters_variants(tmp_path: Path, inputs) -> None:
    stats = build_paraphrase_sets(
        canonical_prompts=inputs["canonical"],
        variants=inputs["variants"],
        policy_results={"oracle": inputs["oracle"], "geometric": inputs["geometric"]},
        output_dir=tmp_path / "out",
        stem="prompts_x_paraphrase",
        max_per_fact=2,
    )
    oracle_rows = _read_jsonl(tmp_path / "out" / "oracle" / "prompts_x_paraphrase.jsonl")
    geometric_rows = _read_jsonl(tmp_path / "out" / "geometric" / "prompts_x_paraphrase.jsonl")

    # f2 is missing from geometric and f3 has an empty geometric manifest,
    # so only f1 survives the cross-policy intersection.
    assert stats["cohort_facts"] == 1
    assert [row["fact_id"] for row in oracle_rows] == ["f1", "f1"]
    assert [row["prompt_text"] for row in oracle_rows] == [
        "Paris serves as the capital of",
        "The city of Paris is capital of",
    ]
    rejections = stats["variant_rejections"]
    assert rejections["same_as_canonical"] == 1
    assert rejections["answer_in_prompt"] == 1
    assert rejections["over_cap"] == 1
    assert rejections["fact_not_in_canonical"] == 1
    assert rejections["subject_missing"] == 1
    assert "not_audited_by:geometric" in stats["fact_drops"]
    assert "empty_manifest_in:geometric" in stats["fact_drops"]

    # Unique ids that never collide with the canonical row's reuse key.
    ids = [row["prompt_id"] for row in oracle_rows]
    assert ids == ["f1::para0", "f1::para1"]
    assert all(canonical_key_of(item) == "f1" for item in ids)

    # The frozen manifest is the policy's canonical one and round-trips.
    assert oracle_rows[0]["deletion_manifest"]["entry_ids"] == ["e1"]
    assert geometric_rows[0]["deletion_manifest"]["entry_ids"] == ["e1", "e9"]
    example = AuditExample.from_prompt_row(geometric_rows[0])
    assert example.deletion_manifest.entry_ids == ("e1", "e9")
    assert example.deletion_manifest.manifest_id == geometric_rows[0]["deletion_manifest"]["manifest_id"]

    # Extra answer aliases merge into the canonical row's aliases.
    assert oracle_rows[1]["answer_aliases"] == ["French Republic"]
    assert oracle_rows[1]["variant_source"] == "pararel"

    # Policy files differ only in their manifests, so FULL/DEL-OFF reuse
    # across policies passes the manifest-blind facts digest check.
    assert prompt_facts_digest(
        tmp_path / "out" / "oracle" / "prompts_x_paraphrase.jsonl"
    ) == prompt_facts_digest(tmp_path / "out" / "geometric" / "prompts_x_paraphrase.jsonl")
    assert (tmp_path / "out" / "prompts_x_paraphrase_build_stats.json").is_file()


def test_build_requires_existing_policy_results(tmp_path: Path, inputs) -> None:
    with pytest.raises(FileNotFoundError):
        build_paraphrase_sets(
            canonical_prompts=inputs["canonical"],
            variants=inputs["variants"],
            policy_results={"value": tmp_path / "missing_results.jsonl"},
            output_dir=tmp_path / "out",
            stem="s",
        )


def test_build_rejects_inconsistent_manifests_across_states(tmp_path: Path, inputs) -> None:
    rows = _result_rows("f1", _manifest("e1"), {"FULL": "France"}, "France") + _result_rows(
        "f1", _manifest("e2"), {"DEL-ON": "x"}, "France"
    )
    corrupt = _write_jsonl(tmp_path / "corrupt" / "x_results.jsonl", rows)
    with pytest.raises(ValueError, match="two different deletion manifests"):
        build_paraphrase_sets(
            canonical_prompts=inputs["canonical"],
            variants=inputs["variants"],
            policy_results={"oracle": corrupt},
            output_dir=tmp_path / "out",
            stem="s",
        )


def test_analyze_pairs_paraphrases_with_canonical_rows(tmp_path: Path) -> None:
    gold = "France"
    m = _manifest("e1")
    canonical = _write_jsonl(
        tmp_path / "canon_results.jsonl",
        [
            # f1: deleted on the canonical prompt.
            *_result_rows("f1", m, {"FULL": gold, "DEL-ON": "no", "DEL-OFF": "no"}, gold),
            # f2: survives the canonical prompt.
            *_result_rows("f2", m, {"FULL": gold, "DEL-ON": gold, "DEL-OFF": "no"}, gold),
            # f3: FULL wrong on the canonical prompt -> out of the cohort.
            *_result_rows("f3", m, {"FULL": "no", "DEL-ON": "no", "DEL-OFF": "no"}, gold),
        ],
    )
    paraphrase = _write_jsonl(
        tmp_path / "para_results.jsonl",
        [
            # f1::para0 leaks through the paraphrase; f1::para1 does not.
            *_result_rows("f1::para0", m, {"FULL": gold, "DEL-ON": gold, "DEL-OFF": "no"}, gold),
            *_result_rows("f1::para1", m, {"FULL": gold, "DEL-ON": "no", "DEL-OFF": "no"}, gold),
            # f2::para0: paraphrase FULL wrong -> out of the cohort.
            *_result_rows("f2::para0", m, {"FULL": "no", "DEL-ON": "no", "DEL-OFF": "no"}, gold),
            *_result_rows("f3::para0", m, {"FULL": gold, "DEL-ON": gold, "DEL-OFF": gold}, gold),
        ],
    )
    summary = analyze_policy(canonical, paraphrase)
    assert summary["paraphrase_rows"] == 4
    assert summary["pairs_canonical_full_correct"] == 3
    assert summary["pairs"] == 2
    assert summary["facts"] == 1
    assert summary["canonical_del_on_correct"] == 0.0
    assert summary["paraphrase_del_on_correct"] == 0.5
    assert summary["discordant_canonical_deleted_paraphrase_survives"] == 1
    assert summary["discordant_canonical_survives_paraphrase_deleted"] == 0
    assert summary["fact_canonical_survives"] == 0.0
    assert summary["fact_any_paraphrase_survives"] == 1.0
    assert summary["mcnemar_p"] == 1.0


def test_mcnemar_exact_matches_binomial() -> None:
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(0, 10) == pytest.approx(2 / 1024)
    assert mcnemar_exact(5, 5) == 1.0


def test_cli_build_and_analyze(tmp_path: Path, inputs) -> None:
    out = tmp_path / "cli"
    assert (
        main(
            [
                "build",
                "--canonical-prompts", str(inputs["canonical"]),
                "--variants", str(inputs["variants"]),
                "--policy", f"oracle={inputs['oracle']}",
                "--output-dir", str(out),
                "--stem", "prompts_x_paraphrase",
            ]
        )
        == 0
    )
    rows = _read_jsonl(out / "oracle" / "prompts_x_paraphrase.jsonl")
    assert {row["fact_id"] for row in rows} == {"f1", "f2", "f3"}

    para_results = _write_jsonl(
        tmp_path / "para" / "prompts_x_paraphrase_results.jsonl",
        _result_rows("f1::para0", _manifest("e1"), {"FULL": "France", "DEL-ON": "x", "DEL-OFF": "x"}, "France"),
    )
    csv_path = tmp_path / "summary.csv"
    assert (
        main(
            [
                "analyze",
                "--canonical", f"oracle={inputs['oracle']}",
                "--paraphrase", f"oracle={para_results}",
                "--canonical", f"geometric={inputs['geometric']}",
                "--paraphrase", f"geometric={tmp_path / 'absent_results.jsonl'}",
                "--allow-missing",
                "--output", str(csv_path),
            ]
        )
        == 0
    )
    lines = csv_path.read_text().splitlines()
    assert lines[0].startswith("policy,")
    assert len(lines) == 2 and lines[1].startswith("oracle,")


def test_end_to_end_frozen_manifests_and_cross_policy_reuse(tmp_path: Path) -> None:
    """Build paraphrase files from a real canonical oracle run, audit them
    the way the suite does (bootstrap flag on, a per-policy FULL dir), and
    check that (1) the frozen manifest, not a paraphrase-built one, drives
    DEL-ON and (2) a second policy reuses the gate's FULL/DEL-OFF rows."""
    from test_reuse import _make_backend, _wired_run, _write_prompts

    from halo.core.states import DatabaseState

    canonical_prompts = _write_prompts(tmp_path / "prompts.jsonl")
    backend, _ = _make_backend()
    _wired_run(
        prompt_path=canonical_prompts,
        output_dir=tmp_path / "canon" / "oracle",
        backend=backend,
        full_dir=tmp_path / "canon" / "full",
    )
    oracle_results = tmp_path / "canon" / "oracle" / "prompts_results.jsonl"
    oracle_rows = _read_jsonl(oracle_results)
    assert {row["prompt_id"] for row in oracle_rows} == {"p0", "p1"}  # p2 skipped

    # A second policy with a broader frozen set.
    broad = DeletionManifest(
        entry_ids=("target-entry", "neighbor-entry"), strategy="closure"
    ).as_dict()
    geometric_results = _write_jsonl(
        tmp_path / "canon" / "geometric" / "prompts_results.jsonl",
        [{**row, "deletion_manifest": broad} for row in oracle_rows],
    )
    variants = _write_jsonl(
        tmp_path / "variants.jsonl",
        [
            {"fact_id": "f0", "prompt_text": "Which city is the capital of France?", "variant": "paraphrase"},
            {"fact_id": "f1", "prompt_text": "Name the capital city of Norway.", "variant": "paraphrase"},
            {"fact_id": "f2", "prompt_text": "Which city is Italy's capital?", "variant": "paraphrase"},
        ],
    )
    para_dir = tmp_path / "paraphrase"
    stats = build_paraphrase_sets(
        canonical_prompts=canonical_prompts,
        variants=variants,
        policy_results={"oracle": oracle_results, "geometric": geometric_results},
        output_dir=para_dir,
        stem="prompts_paraphrase",
    )
    assert stats["cohort_facts"] == 2  # f2 was never audited canonically

    # Gate: oracle paraphrase run, bootstrap flag on as in the Co-LMLM suite.
    gate_backend, gate_generator = _make_backend()
    gate_rows, _, _ = _wired_run(
        prompt_path=para_dir / "oracle" / "prompts_paraphrase.jsonl",
        output_dir=para_dir / "oracle",
        backend=gate_backend,
        full_dir=para_dir / "oracle" / "full",
    )
    oracle_manifest_ids = {row["prompt_id"]: row["deletion_manifest"]["manifest_id"] for row in oracle_rows}
    for row in gate_rows:
        assert row["prompt_id"].split("::para")[0] in oracle_manifest_ids
        # Frozen, never re-bootstrapped from the paraphrase's FULL pass.
        assert row["deletion_manifest"]["manifest_id"] == oracle_manifest_ids[
            row["prompt_id"].split("::para")[0]
        ]
    assert len(gate_rows) == 2 * 3
    gate_calls = gate_generator.generate_calls

    # Second policy: FULL and DEL-OFF come from the gate, DEL-ON is live.
    geo_backend, geo_generator = _make_backend()
    geo_rows, reuse_store, _ = _wired_run(
        prompt_path=para_dir / "geometric" / "prompts_paraphrase.jsonl",
        output_dir=para_dir / "geometric",
        backend=geo_backend,
        full_dir=para_dir / "geometric" / "full",
        reuse_paths=(para_dir / "oracle" / "prompts_paraphrase_results.jsonl",),
    )
    assert reuse_store.served[DatabaseState.FULL.value] == 2
    assert reuse_store.served[DatabaseState.DEL_OFF.value] == 2
    assert reuse_store.generated[DatabaseState.DEL_ON.value] == 2
    assert geo_generator.generate_calls < gate_calls
    for row in geo_rows:
        assert row["deletion_manifest"]["entry_ids"] == ["neighbor-entry", "target-entry"]

    summary = analyze_policy(oracle_results, para_dir / "oracle" / "prompts_paraphrase_results.jsonl")
    assert summary["paraphrase_rows"] == 2


def test_filters_handle_possessives_and_answers_inside_the_subject() -> None:
    from halo.paraphrase import _variant_reject_reason

    herzig = _canonical("z1", "What position does Denny Herzig play?", "defender", "Denny Herzig")
    assert _variant_reject_reason("What's Denny Herzig's role in football?", herzig, set()) is None
    # The answer embedded in the subject's name is not a leak...
    apple = _canonical("c1", "Apple A5 is created by", "Apple", "Apple A5")
    assert _variant_reject_reason("Apple A5 was developed by", apple, set()) is None
    # ...but the answer anywhere else is.
    assert (
        _variant_reject_reason("Made by Apple, the Apple A5 is from", apple, set())
        == "answer_in_prompt"
    )


def test_facts_whose_variants_are_all_rejected_leave_the_cohort(tmp_path: Path) -> None:
    canonical = _write_jsonl(
        tmp_path / "p.jsonl",
        [_canonical("f1", "Paris is the capital of", "France", "Paris"),
         _canonical("f2", "Berlin is located in", "Germany", "Berlin")],
    )
    variants = _write_jsonl(
        tmp_path / "v.jsonl",
        [{"fact_id": "f1", "prompt_text": "Paris serves as the capital of", "variant": "paraphrase"},
         {"fact_id": "f2", "prompt_text": "It is located in", "variant": "paraphrase"}],
    )
    results = _write_jsonl(
        tmp_path / "r" / "x_results.jsonl",
        _result_rows("f1", _manifest("e1"), {"FULL": "France"}, "France")
        + _result_rows("f2", _manifest("e2"), {"FULL": "Germany"}, "Germany"),
    )
    stats = build_paraphrase_sets(
        canonical_prompts=canonical,
        variants=variants,
        policy_results={"oracle": results},
        output_dir=tmp_path / "out",
        stem="s",
    )
    assert stats["facts_with_accepted_variants"] == 1
    assert stats["cohort_facts"] == 1
    assert stats["rows_per_policy"] == 1
