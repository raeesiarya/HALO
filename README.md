# HALO

![Tests](badges/tests.svg)
![Coverage](badges/coverage.svg)

HALO is a causal audit of forgetting in language models with external memory.
It separates knowledge retained in model parameters from knowledge recovered
through related memory entries or nearby retrieval keys.

## Audit design

Each fact is evaluated in three database states:

- `FULL`: memory unchanged and retrieval enabled.
- `DEL-ON`: target entries hidden and retrieval enabled.
- `DEL-OFF`: target entries hidden and retrieval disabled.

Deletion is implemented by search-time filtering; the underlying store is not
modified. Evaluation records include retrieval traces and query embeddings.

The primary entanglement and adversarial cohorts contain facts for which the
`FULL` state is correct, a selected entry passes the normalized answer-mention
heuristic before the answer is visible, and the exact selected event's query
embedding is available. This is an answer-mention cohort, not
proposition-support verification. Coverage and exclusion counts are reported
separately.

The audit includes:

- cross-state parametric leakage L(f), retrieval recovery R(f), and retrieval
  interference I(f);
- deletion closures over geometric, oracle gold-answer-mention, and provenance
  predicates;
- deletion-efficacy and collateral-damage curves over closure radius;
- a linear representational-leakage probe over frozen query embeddings; and
- adversarial survivor entries placed outside the deletion radius.

## Repository structure

- `src/halo/core/`: backend interface, database states, metrics, and analysis.
- `src/halo/interventions/`: closure construction, filtering, support
  judgments, and adversarial interventions.
- `src/halo/cli/`: command-line orchestration and reporting.
- `src/models/co_lmlm/`: backend for the public Co-LMLM release.
- `scripts/`: data setup and evaluation entry points.

## Installation

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
uv run pytest
```

On a GPU host, the Co-LMLM environment additionally requires OpenBLAS as a
system package; `faiss` fails to import without it:

```bash
sudo apt-get install -y libopenblas0
```

The evaluation scripts also extend `LD_LIBRARY_PATH` to the RAPIDS and CUDA
wheel directories, because `faiss-gpu-cu12-cuvs` does not record an RPATH to
them and otherwise fails to resolve `libcuvs.so`, `librmm.so`, and
`libraft.so`.

## Co-LMLM evaluation

The default setup uses the T-REx prompts and the public Co-LMLM retrieval
index. The index requires approximately 113 GB. `INDEX_DIR`, `PROMPTS`, and
`OUTPUT_DIR` override the default paths.

The primary configuration uses Co-LMLM's published factual-evaluation
retrieval threshold of 0.7. Pass
`--co-lmlm-similarity-threshold none` only for a separately labeled
threshold-disabled sensitivity run. Correctness requires the normalized gold
answer or alias to appear as a whole phrase in the output; output fragments of
the gold answer do not count.

```bash
./scripts/setup_data.sh
./scripts/run_audit_co_lmlm.sh
```

Additional `halo-audit` arguments are passed through by the shell entry point.
For example:

```bash
./scripts/run_audit_co_lmlm.sh \
  --closure geometric \
  --radius-grid 0.95:0.70:0.05 \
  --neighbor-mode cosine
```

The complete evaluation consists of the standard three-state audit, the
entanglement sweep, the adversarial evaluation, the DEL-OFF sensitivity
control, and the deletion-policy matrix. All of them run by default:

```bash
./scripts/run_audit_suite_co_lmlm.sh
```

The phases execute sequentially and share a `FULL` pass where applicable.
Partial evaluations resume from disk. Each phase creates a separate W&B run
named `<output-dir>__<mode>`.

The standard audit uses `geometric,value` closure by default, where `value` is
an oracle filter supplied with the ground-truth answer and aliases at
inference time. Radius and adversarial evaluations use `geometric` alone.
Relevant configuration variables
are `SUITE_PHASES`, `STANDARD_CLOSURE`, `SWEEP_CLOSURE`, `ADVERSARIAL_CLOSURE`,
`RADIUS_GRID`, `NEIGHBOR_MODE`, `NEIGHBOR_MIN_COUNT`, and `DEL_OFF_MODE`. The
legacy predicate name `semantic` is accepted as an alias for `value`.

`SUITE_PHASES` narrows the run when needed: `all` (the default), `core`
(`standard`, `sweep`, `adversarial` only), or an explicit comma-separated
subset. This is mainly for resuming a partial run or iterating on one phase —
the `del-off` and `policy` phases are a full standard audit per variant, so the
default suite is roughly three times the cost of `core`:

```bash
SUITE_PHASES=core ./scripts/run_audit_suite_co_lmlm.sh
SUITE_PHASES=sweep,adversarial ./scripts/run_audit_suite_co_lmlm.sh
```

### DEL-OFF controls

Two retrieval-disabled controls are available: `null-retrieval`, which permits
decoding after a failed fact lookup, and `forbid-token`, which prevents fact
retrieval tokens. The sensitivity script stores the two evaluations separately.

```bash
./scripts/run_del_off_sensitivity_co_lmlm.sh
```

`DEL_OFF_MODES` restricts which controls run. The suite's `del-off` phase uses
it to skip the arm the standard phase already covers, writing the remaining
control to `<output-dir>/del_off_sensitivity/<mode>`.

### Deletion policies

Oracle, geometric, value, provenance, and hybrid closure policies can be
evaluated in separate output directories.

```bash
./scripts/run_policy_matrix_co_lmlm.sh
```

The suite's `policy` phase runs the same matrix under
`<output-dir>/policy_matrix/<policy>`.

## Outputs

The default output directory is `outputs/trex`. Outputs include JSONL results,
retrieval traces, query-embedding sidecars, closure manifests, metrics CSVs,
and probe summaries.

## License

This project is licensed under the [MIT License](LICENSE).
