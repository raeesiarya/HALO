import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from halo.analysis.followups import (
    AuditFact,
    frequency_association_summary,
    frequency_bin_summary,
    matched_frequency_rows,
    oracle_miss_candidates,
    parse_args,
    probe_distribution_diagnostics,
    run_followups,
)


def _audit_row(
    fact_id: str,
    state: str,
    output: str,
    *,
    truth: str = "Paris",
    subject: str = "France",
    value_count: int = 3,
    selected_value: str | None = None,
) -> dict:
    row = {
        "fact_id": fact_id,
        "prompt_id": fact_id,
        "state": state,
        "prompt": f"What is associated with {subject}?\nThe answer is",
        "ground_truth": truth,
        "object_aliases": [],
        "subject": subject,
        "relation": "capital",
        "model_output": output,
        "deletion_manifest": {
            "metadata": {
                "entry_counts": {"value": value_count, "geometric": 1},
                "envelope_top_k": 500,
            }
        },
        "retrieval_trace": {},
    }
    if selected_value is not None:
        row["retrieval_trace"]["selected_candidate"] = {
            "entry_id": f"entry-{fact_id}",
            "source_id": f"source-{fact_id}",
            "value": selected_value,
            "score": 0.8,
        }
    return row


def test_probe_diagnostics_expose_legacy_duplicate_split() -> None:
    facts = {
        "f1": AuditFact("zsre", "f1", ground_truth="Paris", subject="France"),
        "f2": AuditFact("zsre", "f2", ground_truth="Paris", subject="France"),
        "f3": AuditFact("zsre", "f3", ground_truth="Warsaw", subject="Poland"),
        "f4": AuditFact("zsre", "f4", ground_truth="Rome", subject="Italy"),
    }
    summary, assignments = probe_distribution_diagnostics(
        "zsre", facts, folds=3, seed=0
    )
    assert summary["canonical_groups"] == 3
    assert summary["duplicate_groups"] == 1
    assert summary["facts_in_duplicate_groups"] == 2
    by_id = {row["fact_id"]: row for row in assignments}
    assert by_id["f1"]["canonical_group_fold"] == by_id["f2"][
        "canonical_group_fold"
    ]


def test_frequency_rows_are_explicitly_a_top_k_proxy() -> None:
    facts = {
        "f1": AuditFact(
            "trex",
            "f1",
            ground_truth="Paris",
            full_correct=True,
            del_off_correct=False,
            value_count_top_k=1,
            envelope_top_k=500,
        ),
        "f2": AuditFact(
            "trex",
            "f2",
            ground_truth="Warsaw",
            full_correct=True,
            del_off_correct=True,
            value_count_top_k=120,
            envelope_top_k=500,
        ),
    }
    rows = matched_frequency_rows(
        "trex", facts, {"f1": False, "f2": True}, {"f1": True, "f2": True}
    )
    assert rows[0]["answer_bearing_neighbors_top_k"] == 1
    assert rows[0]["frequency_bin"] == "1"
    assert rows[1]["frequency_bin"] == "100+"
    summary = frequency_bin_summary(rows)
    high = next(
        row
        for row in summary
        if row["cohort"] == "full_correct" and row["frequency_bin"] == "100+"
    )
    assert high["co_lmlm_del_off_accuracy"] == 1.0
    assert high["standard_lm_accuracy"] == 1.0

    association = frequency_association_summary(rows * 3)
    co = next(
        row
        for row in association
        if row["cohort"] == "answer_mention"
        and row["model"] == "co-lmlm-del-off"
    )
    assert co["spearman_rho"] > 0
    assert co["odds_ratio_per_doubling"] > 1


def test_oracle_candidates_are_an_adjudication_upper_bound() -> None:
    facts = {
        "miss": AuditFact(
            "zsre",
            "miss",
            prompt="Where is it?",
            ground_truth="Texas",
            full_correct=True,
            del_on_correct=True,
            del_off_correct=False,
            del_on_output="Texas",
            del_off_output="unknown",
            del_on_selected={"entry_id": "e1", "value": "TX", "score": 0.8},
        ),
        "parametric": AuditFact(
            "zsre",
            "parametric",
            ground_truth="Paris",
            full_correct=True,
            del_on_correct=True,
            del_off_correct=True,
        ),
    }
    rows, summary = oracle_miss_candidates("zsre", facts)
    assert [row["fact_id"] for row in rows] == ["miss"]
    assert rows[0]["review_label"] == ""
    assert summary["retrieval_mediated_candidates"] == 1
    assert summary["candidate_rate_given_full"] == pytest.approx(0.5)
    assert summary["candidate_share_of_survivors"] == pytest.approx(0.5)


def test_followup_cli_end_to_end_without_embedding_sidecars(tmp_path) -> None:
    results = tmp_path / "results"
    co_dir = results / "co-lmlm" / "zsre"
    value_dir = co_dir / "policy_matrix" / "value"
    standard_dir = results / "standard-lm-360m-fw" / "zsre"
    smol_dir = results / "smollm2-360m" / "zsre"
    for path in (co_dir, value_dir, standard_dir, smol_dir):
        path.mkdir(parents=True, exist_ok=True)

    audit_rows = []
    plain_rows = []
    for index in range(6):
        key = f"f{index}"
        truth = "Paris" if index % 2 == 0 else "Warsaw"
        subject = f"subject-{index // 2}"
        audit_rows.extend(
            [
                _audit_row(key, "FULL", truth, truth=truth, subject=subject),
                _audit_row(
                    key,
                    "DEL-ON",
                    truth if index == 0 else "unknown",
                    truth=truth,
                    subject=subject,
                    selected_value="Par." if index == 0 else None,
                ),
                _audit_row(
                    key,
                    "DEL-OFF",
                    "unknown",
                    truth=truth,
                    subject=subject,
                ),
            ]
        )
        plain_rows.extend(
            [
                _audit_row(key, "FULL", truth, truth=truth, subject=subject),
                _audit_row(key, "DEL-ON", truth, truth=truth, subject=subject),
                _audit_row(key, "DEL-OFF", truth, truth=truth, subject=subject),
            ]
        )

    def write_jsonl(path: Path, rows: list[dict]) -> None:
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    write_jsonl(co_dir / "prompts_zsre_results.jsonl", audit_rows)
    write_jsonl(value_dir / "prompts_zsre_results.jsonl", audit_rows)
    write_jsonl(standard_dir / "prompts_zsre_results.jsonl", plain_rows)
    write_jsonl(smol_dir / "prompts_zsre_results.jsonl", plain_rows)

    output = tmp_path / "report"
    args = parse_args(
        [
            "--results-root",
            str(results),
            "--output-dir",
            str(output),
            "--datasets",
            "zsre",
            "--folds",
            "3",
            "--skip-bow-probe",
        ]
    )
    summary = run_followups(args)
    assert summary["facts"] == 6
    assert summary["oracle_candidates"] == 1
    assert summary["probe_controls"] == 0
    assert "query-embedding sidecar not found" in summary["query_probe_skips"][0]
    assert (output / "README.md").is_file()
    assert (output / "frequency_associations.csv").is_file()
    with (output / "oracle_miss_adjudication.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["retrieved_value"] == "Par."
