# LMLM Audit

![Tests](badges/tests.svg)
![Coverage](badges/coverage.svg)

This repository contains the code for [*Auditing Forgetting in Limited Memory
Language Models*](https://arxiv.org/abs/2607.00605).

The original audit was built for relational LMLMs. We are now extending it to
[Co-LMLM](https://arxiv.org/abs/2607.07707), where facts are stored as free-form
text and retrieved with continuous keys.

## What are we testing?

LMLMs are designed to keep factual knowledge in an external memory. In theory,
deleting a fact from that memory should make the model forget it. In practice,
the answer may still be available through the model's parameters or through a
different, related memory entry.

For each fact, we run the model in three settings:

- `FULL`: the memory is unchanged and retrieval is enabled.
- `DEL-ON`: the target entry is hidden, but retrieval remains enabled.
- `DEL-OFF`: the target entry is hidden and retrieval is disabled.

Comparing these runs helps us distinguish parametric memory from answers that
are recovered through retrieval. The audit also saves the retrieved entries so
we can inspect where an answer came from.

## Current status

The repository currently supports:

- the original rel-LMLM audit;
- a shared audit format that does not require subject-relation pairs;
- a Co-LMLM backend built around the public model and index interfaces;
- non-destructive deletion by filtering selected entry or source IDs at search
  time;
- retrieval traces, query-embedding sidecars, and cross-state forgetting
  metrics;
- an oracle smoke-test mode that uses the entry retrieved during `FULL` as the
  deletion target; and
- materialized deletion closures (`--closure geometric,semantic,provenance`)
  built from the `FULL` pass, with per-entry attribution artifacts and a
  run-time semantic backstop; and
- entanglement sweeps (`--radius-grid 0.95:0.70:0.05`) that measure deletion
  efficacy against collateral damage on neighbor facts
  (`--neighbor-mode cosine|same-source`) and report per-fact operating
  curves and the entanglement gap G(f).

The Co-LMLM backend has unit-test coverage, but it still needs to be run against
the full released checkpoint and index. The next research step is to move past
single oracle entries and identify all memory entries that express the same
fact.

## Repository structure

The audit code lives in `src/lmlm_audit/`, split into four subpackages:

- `core/` — backend-agnostic pieces: the abstract backend interface, database
  states, forgetting metrics, answer-equivalence checks, and prompt/example
  handling.
- `rel_lmlm/` — the original relational-LMLM backend: model loader, triple
  database, and deletion logic.
- `colmlm/` — the Co-LMLM backend: wrappers around the public model and index
  interfaces, search-time ID filtering for non-destructive deletion, and answer
  extraction.
- `cli/` — the `lmlm-audit` entry point (`run_audit.py`), the audit runner, and
  result/metrics reporting.

Databases and prompt sets are under `data/`: the released LMLM database with
six prompt types, and three custom domains (countries, politicians, sports)
with base/alias/collision/noise variants.

## Setup

The project uses Python 3.12 and [uv](https://docs.astral.sh/uv/).

For the original rel-LMLM audit, place the upstream LMLM repository at
`../LMLM`, then run:

```bash
uv sync
uv run pytest
```

## Running the rel-LMLM audit

```bash
uv run lmlm-audit \
  --database-path data/custom_databases/countries/base.json \
  --prompt-files data/custom_databases/countries/prompts/base/prompts_direct_questions.jsonl \
  --states FULL DEL-ON DEL-OFF \
  --output-dir outputs/audit \
  --wandb-activation off
```

The runner writes one JSONL result file per prompt set, along with per-state and
cross-state metric CSVs.

## Running the Co-LMLM audit

Co-LMLM and rel-LMLM both use the Python package name `lmlm`, so they should be
kept in separate environments. Run this command from the public Co-LMLM
environment after downloading its model and index:

```bash
cd /path/to/Co-LMLM

PYTHONPATH=/path/to/HALOCoLMLM/src:src \
uv run python -m lmlm_audit.cli.run_audit \
  --backend colmlm \
  --colmlm-source-path . \
  --colmlm-model-path /path/to/CoLMLM-360M-FW \
  --index-path /path/to/co-lmlm-wiki-index \
  --entries-db-path /path/to/co-lmlm-wiki-index/entries.db \
  --prompt-files /path/to/prompts.jsonl \
  --bootstrap-oracle-from-full \
  --states FULL DEL-ON DEL-OFF \
  --output-dir /path/to/results
```

An example prompt file is available at
[data/colmlm/prompts_smoke.example.jsonl](data/colmlm/prompts_smoke.example.jsonl).
For a proper experiment, each prompt should use a reviewed deletion manifest
rather than relying on the oracle bootstrap option.

## Papers

- [Auditing Forgetting in Limited Memory Language Models](https://arxiv.org/abs/2607.00605)
- [Pre-training Limited Memory Language Models with Internal and External
  Knowledge](https://arxiv.org/abs/2505.15962)
- [Co-LMLM](https://arxiv.org/abs/2607.07707)

## Citation

```bibtex
@misc{lmlmauditing,
  title         = {Auditing Forgetting in Limited Memory Language Models},
  author        = {Raeesi, Arya and Roed, Hanna},
  year          = {2026},
  eprint        = {2607.00605},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2607.00605},
  doi           = {10.48550/arXiv.2607.00605}
}
```

## License

This project is licensed under the [MIT License](LICENSE).
