# HALO

![Tests](badges/tests.svg)
![Coverage](badges/coverage.svg)

A causal audit of forgetting in language models with an external memory. When
a fact is deleted from the memory, is it actually gone — or does the model
still recover it through its parameters, a related entry, or a nearby key?

## How the audit works

For each fact, HALO runs the model in three states and compares:

- `FULL` — memory unchanged, retrieval enabled.
- `DEL-ON` — the target entry hidden, retrieval still enabled.
- `DEL-OFF` — the target entry hidden and retrieval disabled.

Deletion is non-destructive: entries are filtered out at search time, never
removed from the store. Every run records retrieval traces and query
embeddings, so each answer can be attributed to where it came from.

On top of the three-state comparison:

- **Forgetting metrics** — cross-state leakage L(f) and retrieval-recovery
  R(f) per fact.
- **Deletion closures** (`--closure geometric,semantic,provenance`) — instead
  of hiding one oracle entry, materialize the set of entries that express the
  fact, with per-entry attribution.
- **Entanglement sweeps** (`--radius-grid`) — deletion efficacy against
  collateral damage on neighboring facts, reported as per-fact operating
  curves and the entanglement gap G(f).
- **Representational-leakage probe** — a linear readout fit on frozen query
  embeddings over a fact-disjoint split, run automatically with every audit.
- **Adversarial closures** (`--adversarial`) — synthetic survivor entries
  injected just outside the deletion radius, scored by evasion rate and a
  geometry-only margin predictor.

## Repository layout

- `src/halo/` — the audit itself. `core/` holds the backend interface,
  database states, metrics, and analysis; `interventions/` holds the deletion
  closure, filtering search, support judge, and adversarial machinery; `cli/`
  is the `halo-audit` entry point.
- `src/models/` — one subpackage per audited model. Each registers a
  `BackendSpec` with `halo.registry` (how to build the backend, its search
  index, and its arguments), so a new model slots in without touching the
  audit core. `models/co_lmlm/` is the bundled backend.

## Setup

Python 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run pytest
```

## Running an audit

Fetch the audit prompts and the retrieval index for the bundled backend
(~113 GB into `data/`; `INDEX_DIR` overrides the location):

```bash
./scripts/setup_data.sh
```

Then run the audit — this clones and syncs the backend's public source
checkout on first use and fetches the model from Hugging Face automatically:

```bash
./scripts/run_audit_co_lmlm.sh
```

Extra flags pass through to `halo-audit`, e.g. an entanglement sweep:

```bash
./scripts/run_audit_co_lmlm.sh \
  --closure geometric,semantic \
  --radius-grid 0.95:0.70:0.05 \
  --neighbor-mode cosine
```

All three states run in every audit; device, dtype, and attention
implementation are auto-detected. `--index-path` is the memory being audited.
Results, retrieval traces, embedding sidecars, probe CSVs, and closure
manifests are written to the output directory (`outputs/popqa` by default).

An example prompt file is at
[data/colmlm/prompts_smoke.example.jsonl](data/colmlm/prompts_smoke.example.jsonl).
For a proper experiment, give each prompt a reviewed deletion manifest rather
than relying on `--bootstrap-oracle-from-full`.

## License

This project is licensed under the [MIT License](LICENSE).
