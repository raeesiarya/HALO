# HALO

![Tests](badges/tests.svg)
![Coverage](badges/coverage.svg)

HALO audits forgetting in language models with external memory. It separates
answers produced from model parameters from answers recovered through memory.

## Audit design

Each fact is evaluated in three states:

- `FULL`: memory and retrieval are unchanged.
- `DEL-ON`: matching entries are hidden, with retrieval still enabled.
- `DEL-OFF`: matching entries are hidden and factual retrieval is disabled.

Deletion is implemented as search-time filtering. The underlying index is not
modified. Comparing the three states gives post-deletion survival, unaided
answerability, retrieval-mediated correctness, and retrieval interference.

The audited cohort contains facts where the intact model retrieves an entry
that mentions the answer. Most reported deletion rates further condition on
the intact model answering correctly. This is an answer-mention check, not
full verification that the retrieved span supports the proposition.

HALO also includes deletion-radius sweeps, collateral-damage measurements,
query-embedding probes, deletion-policy comparisons, and adversarial writes.

## Setup

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync
uv run pytest
```

On the Linux GPU machines used for Co-LMLM, FAISS also needs OpenBLAS:

```bash
sudo apt-get install -y libopenblas0
```

The run scripts set the library paths needed by the CUDA FAISS wheels.

## Running Co-LMLM

The default run uses T-REx and the released FineWeb plus Wikipedia index. The
index is about 1.05 TB.

```bash
./scripts/setup_data.sh
./scripts/run_audit_co_lmlm.sh
```

Set `INDEX_DIR`, `PROMPTS`, or `OUTPUT_DIR` to use different paths. Extra
arguments are passed to `halo-audit`, for example:

```bash
./scripts/run_audit_co_lmlm.sh \
  --closure geometric \
  --radius-grid 0.95:0.70:0.05
```

The standard Co-LMLM configuration uses retrieval threshold `0.7`. The
default closure combines geometric and gold-answer value filtering. The value
filter is an oracle used for evaluation, not a deployable deletion rule.
Correctness requires the normalized gold answer or an alias to appear as a
complete phrase in the output. Radius sweeps and adversarial runs use only the
geometric closure.

To run the standard audit, radius sweep, adversarial evaluation, DEL-OFF
controls, and policy matrix:

```bash
./scripts/run_audit_suite_co_lmlm.sh
```

Runs resume from existing outputs. `SUITE_WORKERS` controls the number of
single-GPU workers, and `SUITE_PHASES` can select part of the suite. `core`
means `standard,sweep,adversarial`:

```bash
SUITE_WORKERS=8 ./scripts/run_audit_suite_co_lmlm.sh
SUITE_PHASES=core ./scripts/run_audit_suite_co_lmlm.sh
SUITE_PHASES=sweep,adversarial ./scripts/run_audit_suite_co_lmlm.sh
```

The default DEL-OFF control is `null-retrieval`; `forbid-token` is the
sensitivity check. The controls and policy matrix can also be run separately:

```bash
./scripts/run_del_off_sensitivity_co_lmlm.sh
./scripts/run_policy_matrix_co_lmlm.sh
```

## Cross-model runs

The cross-model scheduler runs Co-LMLM, SmolLM2-360M, and
CoLMLM-Standard-LM-Baseline-360M-FW over all prompt sets. Check the planned
jobs before starting a detached run:

```bash
./scripts/run_cross_model_scheduler.sh --dry-run
./scripts/run_cross_model_scheduler.sh --detach
```

The wrapper below submits the same run and returns immediately:

```bash
SCHEDULER_SHARDS=4 ./scripts/run_suite_parallel_cross_model.sh
tail -F out-cross-model/_scheduler.log
```

The main configuration variables are `SETS`, `MODELS`, `GPUS`, `MAX_PARALLEL`,
`SCHEDULER_SHARDS`, `SUITE_WORKERS`, `INDEX_DIR`, and `OUT_ROOT`. The default
output directory is `out-cross-model/`. Repeating the same command resumes an
interrupted run.

## Outputs

Audit outputs include JSONL results, retrieval traces, query embeddings,
closure manifests, metric CSVs, and probe summaries. Single-dataset runs use
`outputs/trex` by default.

The analysis used for Status Update 2 can be reproduced from a completed
cross-model result tree with:

```bash
uv run python scripts/status_update_2_analysis.py --stage all
```

## Repository structure

- `src/halo/`: audit logic, interventions, metrics, and CLI code.
- `src/models/`: Co-LMLM and closed-book model backends.
- `scripts/`: setup, evaluation, scheduling, and analysis scripts.
- `annotations/`: reviewed labels used by the analysis.

## License

This project is licensed under the [MIT License](LICENSE).
