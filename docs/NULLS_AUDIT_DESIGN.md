# Auditing parametric native unlearning (NULLs) against external-memory deletion (Co-LMLM): design specification

Status: design locked; v1 implemented 2026-08-24 (backend
`src/models/nulls_wiki/`, source-level closures
`src/halo/interventions/source_closure.py`, corpus filter
`--co-lmlm-corpus-allow`; the whole pipeline — prep chain included — runs as
one job graph via `scripts/run_suite_parallel_cross_model.sh`, artifacts via
`scripts/setup_nulls.sh`). Authors' artifacts released 2026-09-10 and
verified: `title_to_index.pkl` (6,407,814 raw MediaWiki titles, index =
insertion position, zero ambiguity under the §9 title rule; still never
reconstructed — a guessed mapping risks silently mis-addressed sinks) and
the training corpus `gauravrghosal/wiki_nulls_corpus` — the unmodified
public `wikimedia/wikipedia` `20231101.en` snapshot (verified: identical
per-shard row counts, first shard identical row for row), bijective with
the mapping. That snapshot itself omits a number of prominent articles
(Paris, Germany, Physics, ...), which surfaces as the §9 exclusion rate:
T-REx 24%, CounterFact 20%, ZsRE 10%, Google-RE 10% on the current prompt
sets. The closure-embedding artifact is built from that
corpus in setup. Outstanding: the shared-adversarial phase (§8).
Companion analysis: `reviews/ghosal-2026-nulls.md`.
Target standard: every design choice here should survive an adversarial
NeurIPS-style review. Each section ends with the objections we expect and the
answer the design gives.

## 1. Objective and the single-variable principle

We compare two deletion substrates under one audit protocol:

- **Co-LMLM**: knowledge in an external index; deletion = search-time
  filtering of index content.
- **NULLs**: knowledge in source-keyed parametric sinks; deletion = refusing
  to activate the source's sink mask at inference.

The claim we want the audit to license is about the *deletion operator*, not
about the models: given the same deletion request, which substrate forgets the
target more completely, damages less around it, and resists circumvention
better? For that claim to be identifiable, everything upstream of the operator
must be held fixed:

> **Design invariant.** The deletion request, the closure rule that expands
> it, the evaluation prompts, the correctness protocol, and the statistics are
> identical across substrates. The only manipulated variable is where the
> deletion is executed.

Every choice below is derived from this invariant.

## 2. The unifying abstraction: source-level deletion

The two systems have different native deletion granularities: Co-LMLM deletes
*index entries*; NULLs deletes *sources* (Wikipedia articles). Any comparison
run at different granularities is confounded from the start. We therefore fix
the **source (document) as the common deletion unit**:

- A deletion request for fact *f* resolves — via a closure rule (§4) — to a
  set of sources D(f) = {s₁, …, s_k}.
- **Co-LMLM executes D(f)** by search-time filtering of every index entry with
  `source_id ∈ D(f)`. This is already expressible: `DeletionManifest.source_ids`
  is a first-class field and DEL-ON validation enforces it
  (`src/halo/core/backend.py`).
- **NULLs executes D(f)** by excluding the sink masks of every source in D(f)
  during generation.

Both systems receive the *same manifest*. The manifest, not the substrate,
defines what "deleting the fact" means.

Granularity itself remains a reported dimension, not a hidden one: Co-LMLM
additionally supports entry-level deletion (its native mode), and we keep the
existing entry-level runs as a Co-LMLM-internal condition labelled as such.
The cross-substrate tables use source-level manifests only.

*Expected objection:* "You handicapped Co-LMLM by coarsening it to source
level." *Answer:* source level is the **finest granularity both substrates
support**, and the entry-level Co-LMLM numbers are reported alongside as the
upper bound of what finer granularity buys. Coarsening is disclosed, motivated,
and quantified.

## 3. State mapping

| HALO state | Co-LMLM | NULLs |
|---|---|---|
| FULL | index intact, retrieval on | ground-truth source sink active (Sink-On) |
| DEL-ON | entries of D(f) filtered, retrieval live | sinks of D(f) excluded, routing live → next-closest surviving source's sink |
| DEL-OFF | retrieval disabled | all sinks inactive (backbone only) |

DEL-ON for NULLs uses the paper's next-closest routing (their "Sink-Off")
rather than bare exclusion: in both substrates DEL-ON means *the memory system
keeps operating over what survives*. Bare exclusion without routing would make
NULLs' DEL-ON a disguised DEL-OFF and inflate its apparent deletion efficacy.
The activated substitute sink is recorded in the synthetic trace as the
"selected candidate", so the existing DEL-ON validation (nothing from the
manifest may be selected) applies unchanged.

The L/R/S/I metric definitions are computed from the three states exactly as
today; only their reading changes (R = memory-mediated correctness, where
"memory" is the index or the sink pool respectively).

## 4. Closure rules: one semantic space, two executors

Co-LMLM's current geometric closure operates in its retriever's embedding
space; NULLs' natural neighborhoods live in its own source-embedding space.
Radii in two different spaces are incommensurable — this is the single largest
threat to the sweep comparison, so we remove it by construction:

> **Shared-encoder rule.** All cross-substrate closures are computed in one
> **external, model-independent embedding space**: a fixed off-the-shelf text
> encoder E applied to source *text* (article lead/full text, one vector per
> source). Neither system's internal embeddings define the neighborhoods used
> to compare them.

The closure library (`src/halo/interventions/closure.py`) then gains
source-level variants of the existing predicates, all executor-agnostic:

- **provenance-source**: D(f) = {gold article of f}. (The minimal request —
  NULLs' native operation, and the anchor point of every sweep.)
- **geometric-source**: D(f) = gold article + sources within a neighborhood of
  it in E-space (see §5 for how "neighborhood" is parametrized).
- **value-source**: D(f) = sources whose text mentions the (aliased,
  normalized) answer — computable for both substrates because we hold the
  Wikipedia text; remains an **oracle evaluation rule, not a deployable
  policy**, exactly as documented for the current value filter.
- **oracle-source**: value-source with gold answers, bootstrapped as today.
- **hybrid-source**: geometric-source ∩ value-source.
- **factq**: excluded — it requires Co-LMLM's `<FACT-q>` head. Reported as
  Co-LMLM-only, outside the cross-substrate tables.

*Expected objection:* "Your external encoder favors one system." *Answer:* the
encoder never touches either model; it defines the *request*, which both
systems must then execute. We additionally report a sensitivity check with a
second encoder family — if the substrate ranking flips with the encoder, the
result is fragile and we say so; if not, the choice is immaterial.

## 5. The sweep (v2): calibrated operating curves, not raw radii

Even in a shared space, "sweep cosine radius ρ" is the wrong primary axis: the
similarity distribution around a source is heavy-tailed and radius→set-size is
nonlinear, so equal ρ does not mean equal request size. The sweep is therefore
parametrized by **deletion breadth k** — the number of nearest sources deleted
(k ∈ {1, 2, 4, 8, …, K}, k=1 ≡ provenance-source) — with the radius grid
retained only as a secondary Co-LMLM-internal view for continuity with
existing results.

Primary readout per k, per substrate:

- **Efficacy**: post-deletion survival of the target fact.
- **Collateral**: correctness degradation on *control facts* — facts anchored
  in sources ranked by E-distance from the gold source, reported in distance
  bands (nearest band ≈ the facts most at risk). Control facts are drawn from
  the same audited cohort, so they carry FULL-state baselines by construction.

The cross-substrate comparison is the **efficacy–collateral operating curve**
over k, compared (a) as curves with paired-bootstrap confidence bands and
(b) at **matched collateral**: "at the k where each substrate's collateral
first reaches level c, which substrate has forgotten the target more?" This is
the deletion analog of comparing classifiers at matched false-positive rate,
and it is robust to the two substrates having differently-shaped neighborhoods.

*Expected objection:* "k-of-nearest is arbitrary." *Answer:* k is the
operating-point index, not the claim; the claim lives in the curve, and the
matched-collateral comparison is invariant to monotone reparametrizations of
the sweep axis.

## 6. DEL-OFF sensitivity (v2): two implementations of "memory off"

HALO's existing control pair (null-retrieval vs forbid-token) asks whether
DEL-OFF conclusions depend on *how* retrieval is disabled. The NULLs analog
must answer the matching question — and it also answers the strongest
architecture-specific objection to backbone-only evaluation:

- **sinks-zero**: all sink neurons inactive (the paper's `dropout` mode).
- **placebo-sink**: activate the mask of a fixed, unrelated source (drawn once
  per fact from a seeded shuffle of far-away sources, distance ≥ a floor in
  E-space; seed recorded).

The network never runs with zero active sinks during training, so sinks-zero
is off-distribution: degradation could reflect activation-statistics shift
rather than knowledge absence. Placebo-sink keeps the activation distribution
on-distribution while providing no fact-relevant memory. Agreement between the
two modes (as with null-retrieval vs forbid-token today) is what licenses
reading DEL-OFF as "unaided answerability"; disagreement is itself a finding
about the substrate and bounds the claim.

## 7. Policy matrix (v2)

With §4's source-level predicates, the five-row policy matrix (oracle,
geometric, value, provenance, hybrid) runs identically for both substrates —
same manifests, same oracle-reuse scheduling structure. The factq row remains
Co-LMLM-only. The cross-substrate matrix is reported at source granularity;
Co-LMLM's entry-level matrix remains as the system-internal comparison it
already is.

## 8. Phases with no analog — and the shared replacement

- **Adversarial writes**: no write path into NULLs' weights exists. The phase
  stays Co-LMLM-only, explicitly scoped as an external-memory threat model.
- **Shared adversarial phase (extraction)**: circumvention is where post-hoc
  unlearning fails, so a robustness-free comparison would be attacked as
  incomplete. We add a substrate-neutral **paraphrase/rephrasing extraction
  phase**: after DEL-ON deletion of D(f), probe the fact through paraphrased
  and indirect prompts (building on `interventions/adversary.py` and the
  paraphrase-miss findings from the July 2026 feedback). Identical attack
  prompts for both substrates. Full GCG optimization is out of scope for v2
  (compute; and NULLs' paper already reports it) but the design leaves the
  attack-prompt interface open for it.

The NULLs-specific threat surface that has no Co-LMLM analog — adversarially
steering the *router* to activate a deleted sink — is documented as
out-of-audit-scope in v2 and flagged as future work; we do not claim router
robustness.

## 9. Cohorts

- **Per-substrate cohort** (as today): facts the intact system demonstrably
  carries. Co-LMLM: intact retrieval surfaces an answer-mentioning entry.
  NULLs: the **verification gate** below.
- **NULLs verification gate**: sinks are keyed by article title through a
  PRNG; a wrong title silently activates an unrelated mask and mimics
  deletion. Every fact must therefore pass: gold-answer log-likelihood under
  Sink-On strictly exceeds it under placebo-sink by a margin δ (calibrated on
  a held-out sample so that the false-pass rate is estimable). Facts failing
  the gate are excluded as *unmappable*, and the exclusion rate is reported —
  it is a substantive statistic about title/dump mismatch, not bookkeeping.
- **Intersection cohort** (primary for cross-substrate tables): facts in both
  per-substrate cohorts with a correct FULL answer in both systems. All
  headline comparisons are paired on this cohort; per-substrate cohorts are
  reported alongside so the conditioning is visible.

## 10. Corpus and model controls

- **Matched-corpus configuration (primary)**: Co-LMLM runs with FineWeb
  entries filtered out at search time (source-level filtering — the mechanism
  §2 already requires), so its effective memory is Wikipedia, matching NULLs'
  training corpus. The full-index Co-LMLM runs remain as secondary results.
- **Declared mismatches** (not fixable without training, stated in the
  limitations table): parameter count (~1B vs 360M), Wikipedia dump version
  (reconciled as far as titles allow; residual mismatch shows up in the §9
  exclusion rate), training objective and epochs. Tokenizer is shared
  (SmolLM2), which removes one nuisance variable.
- **Scale reference (`smollm2-1.7b`)**: the off-the-shelf parametric entry at
  NULLs' scale — what `smollm2-360m` is to Co-LMLM, one size up. SmolLM2's
  released sizes are 135M / 360M / 1.7B, so 1.7B is the nearest neighbour of
  a ~1B model in the same family and with the same tokenizer, and it brackets
  NULLs' Sink-On correctness from above where the 360M entries cannot. It is
  a *reference, not a control*: SmolLM2-1.7B is trained on the full SmolLM2
  web corpus and NULLs on Wikipedia only, so the bracket is loose by training
  data as well as by scale, and no causal claim rests on it. Registered as a
  closed-book backend (`src/models/smollm2_1_7b/`) and scheduled alongside
  `nulls-wiki-1b` rather than with the 360M block.
- **No data-matched parametric control exists for NULLs.** Co-LMLM's
  `standard-lm-360m-fw` has no NULLs analogue: the authors released
  `NULLS-Wikipedia-Full` and `NULLS-HarryPotter`, and the one repository that
  looked like a no-sink Wikipedia LM (`gauravrghosal/wikipedia_full_8x`) is
  empty (checked 2026-09-12). The data- and parameter-matched controls are
  therefore NULLs' own DEL-OFF modes (`sinks-zero`, `placebo-sink`, §6);
  building an external one would require training a Wikipedia-only 1B LM
  ourselves.
- **Protocol**: HALO's generation-based whole-phrase correctness is primary
  for both systems (greedy decoding, deterministic). Truth-ratio-style
  likelihood scoring is computed as a secondary readout for both, giving a
  bridge to the NULLs paper's numbers and a robustness check on the
  correctness protocol itself. If generation-based FULL accuracy turns out
  too low to power the audit (checkable cheaply by forwarding ``--limit`` to
  the suite), the protocol question reopens — a decision point, not
  something to discover after the full run.

## 11. Statistical protocol

- Unit of analysis: the fact (prompts within a fact are correlated; all
  resampling clusters by fact).
- Headline comparisons: paired differences on the intersection cohort with
  cluster-bootstrap 95% CIs; operating curves with bootstrap bands; no
  point-estimate-only claims.
- Determinism: greedy decoding everywhere; NULLs runs additionally record the
  activated sink set per generation, making every row exactly reproducible.
- Multiplicity: the phase structure fixes the family of headline comparisons
  in advance (this document is the pre-registration); exploratory cuts are
  labelled as such.
- Seeds: single released checkpoint per system — no training-seed variance is
  claimable, and we say so; all variance statements are across facts.

## 12. Threats to validity (summary table for the paper)

| Threat | Mitigation | Residual |
|---|---|---|
| Different deletion granularity | Source-level unification (§2) | Entry-level advantage reported separately |
| Incommensurable neighborhood spaces | Shared external encoder (§4), breadth-k sweep + matched-collateral readout (§5) | Encoder choice → dual-encoder sensitivity check |
| Backbone-only is off-distribution | Placebo-sink control (§6) | Disagreement bounds DEL-OFF claims |
| Silent sink-mask mismatch | Verification gate + reported exclusion rate (§9) | Gate miscalibration → δ calibrated on held-out sample |
| Corpus mismatch | Wikipedia-only Co-LMLM as primary config (§10) | Dump-version drift, epochs |
| Scale mismatch (1B vs 360M) | Declared; direction of expected bias discussed; `smollm2-1.7b` brackets NULLs' scale from above (§10) | Unfixable without training; the bracket is corpus-confounded |
| No data-matched parametric control for NULLs | Own DEL-OFF modes serve the role (§6); absence of a released no-sink Wikipedia LM stated (§10) | Cross-substrate closed-book comparison stays a reference, not a control |
| Robustness omitted | Shared paraphrase-extraction phase (§8) | GCG and router attacks scoped out, flagged |
| Cohort conditioning | Intersection cohort primary, per-substrate shown (§9) | Conditional claims only — stated as such |

## 13. Implementation mapping

1. `src/models/nulls_wiki/` — backend package: litgpt/SeqTD loading (vendored
   `LLaMAMLPSeqTD`), state mapping (§3), synthetic traces, sink-set capability
   fingerprints, placebo-sink and sinks-zero DEL-OFF modes.
2. `src/halo/interventions/source_closure.py` — source-level predicates over
   a precomputed E-space artifact; executor stays in each backend.
3. Source-embedding artifacts — `scripts/build_nulls_title_embeddings.py`:
   routing mode (upstream-matched title space) and closure mode (shared E
   over article texts; the one artifact not auto-built by the suite).
4. Prompt-set augmentation + gate — `scripts/augment_source_titles.py` and
   `scripts/nulls_verification_gate.py` (striped; emits the gated prompt
   set).
5. `src/models/smollm2_1_7b/` — the scale reference (§10): thin registration
   over the shared closed-book implementation, deletion flags rejected up
   front exactly as for `smollm2-360m`.
6. `src/halo/scheduler.py` — third model category for `nulls-wiki-1b`, prep
   included: shared embeddings job -> per-set augment -> striped gate ->
   standard -> del-off + closures -> sweep(k-grid) + policy (value/hybrid
   rows from a streamed corpus pass; provenance = standard, geometric =
   sweep at NULLS_POLICY_K); striped like Co-LMLM phases (three real states
   ⇒ no cross-state reuse). One command:
   `scripts/run_suite_parallel_cross_model.sh`. The shared-adversarial
   phase (§8) is future work.
7. Co-LMLM side — source-level filtering via manifest `source_ids` (already
   supported) + `--co-lmlm-corpus-allow` for the Wikipedia-only config.
8. Analysis — matched-collateral readout, operating curves, paired bootstrap;
   figure updates in `scripts/paper_figures.py` distinguishing substrates
   (not yet built).
