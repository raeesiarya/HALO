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
index is about 1.05 TB. `setup_colmlm.sh` builds the prompt sets and
downloads the index; `setup_nulls.sh` fetches the NULLs artifacts (below);
`setup_data.sh` runs both.

```bash
./scripts/setup_colmlm.sh
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

The cross-model scheduler runs Co-LMLM, SmolLM2-360M,
CoLMLM-Standard-LM-Baseline-360M-FW, and NULLs (`nulls-wiki-1b`, prep jobs
included) over all prompt sets. Check the planned jobs before starting a
detached run:

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
`SCHEDULER_SHARDS`, `SUITE_WORKERS`, `INDEX_DIR`, and `OUT_ROOT`. `GPUS`
defaults to every GPU `nvidia-smi` reports (or `CUDA_VISIBLE_DEVICES` when
set). The default output directory is `out-cross-model/`. Repeating the same
command resumes an interrupted run.

## NULLs (parametric native unlearning)

`nulls-wiki-1b` audits the released NULLs Wikipedia model (arXiv:2606.13873)
as a third deletion paradigm: deletion excludes source-keyed sink-neuron
masks instead of filtering an index. The comparison design — source-level
manifests executed by both substrates, shared-encoder closures, breadth-k
sweeps, sinks-zero vs placebo-sink DEL-OFF controls, and the verification
gate — is specified in `docs/NULLS_AUDIT_DESIGN.md`.

Everything runs through the one suite command: `nulls-wiki-1b` is in the
cross-model scheduler's default model list, and its preparation chain is
part of the same job graph — the shared title-embedding build, per-set
source-title augmentation, and the striped verification gate (which emits
the gated prompt set the audits consume) run as jobs before the `standard`,
`del-off` (complementary DEL-OFF mode), breadth-k `sweep`, and `policy`
phases. The sweep phases appear when the shared-encoder closure artifact
(`data/nulls-closure-embeddings.npz`, built with
`scripts/build_nulls_title_embeddings.py --mode closure` from the article
texts) exists; the policy rows additionally need the corpus. The
source-level policy matrix mirrors Co-LMLM's: provenance is the standard
run, geometric is the sweep group at `NULLS_POLICY_K` (default 4), and the
`value` and `hybrid` rows are audited under `policy_matrix/`; `factq` and
adversarial writes stay Co-LMLM-only by design.

`./scripts/setup_nulls.sh` installs the `nulls` dependency group (litgpt)
and fetches the authors' released artifacts: the checkpoint, the
training-time `title_to_index.pkl` the sink masks are keyed on (always the
authors' artifact — a guessed mapping risks silently mis-addressed sinks,
so no reconstruction path exists), and the training corpus
(`gauravrghosal/wiki_nulls_corpus`, 6.4M articles, bijective with the
mapping). It then builds the shared-encoder closure artifact from the
corpus texts (GPU-hours; `SKIP_CLOSURE=1` defers it — the standard and
DEL-OFF phases run without it, and re-submitting the suite after the build
adds the sweep). `NULLS_PHASES`, `NULLS_DEL_OFF_MODE`,
`NULLS_GATE_MARGIN`, `NULLS_SWEEP_K_GRID`, and the `NULLS_*` artifact-path
variables configure the phases. The matched-corpus Co-LMLM comparison runs
with `--co-lmlm-corpus-allow` restricting retrieval to Wikipedia sources.

The released corpus is the public `wikimedia/wikipedia` `20231101.en`
snapshot, unmodified (identical per-shard row counts; first shard identical
row for row). That snapshot itself omits a number of prominent articles
(e.g. *Paris*, *Germany*, *Physics*, *Autism*), so facts whose subject
article has no sink are excluded by the source-title augmentation step and
reported per set in `<prompts>_nulls.jsonl.exclusions.json` (measured on
the current prompt sets: T-REx 24%, CounterFact 20%, ZsRE 10%, Google-RE
10% excluded). Cross-substrate tables are paired on the intersection
cohort, and the exclusion rate is reported alongside.

## Outputs

Audit outputs include JSONL results, retrieval traces, query embeddings,
closure manifests, metric CSVs, and probe summaries. Single-dataset runs use
`outputs/trex` by default.

The paper figures can be rendered from the aggregated analysis results
under `results/status_update_2/` with:

```bash
uv run python scripts/paper_figures.py
```

Figures and suggested LaTeX captions are written to `figures/` (gitignored).

## Repository structure

- `src/halo/`: audit logic, interventions, metrics, and CLI code.
- `src/models/`: Co-LMLM and closed-book model backends.
- `scripts/`: setup (`setup_colmlm.sh`, `setup_nulls.sh`), evaluation,
  scheduling, and analysis scripts.
- `annotations/`: reviewed labels used by the analysis.

## License

This project is licensed under the [MIT License](LICENSE).
