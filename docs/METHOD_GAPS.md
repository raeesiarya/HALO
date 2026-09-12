# What the Method section is missing

Audit of `\section{Method}` (draft of 2026-09-12) against the repository as of
`e5bb20e` (branch `19-cross-model-suite-scheduler`). Written in English because
every suggestion is drop-in paper text.

Verdict up front: the Method section is an accurate and fairly complete
description of **one** experiment — the entry-level Co-LMLM audit with two
closed-book calibration points. The repository now implements **three deletion
substrates, four non-Co-LMLM models, two granularities of closure, two sweep
parametrizations, two verification layers, and a statistical protocol** that the
section does not mention at all. The gaps below are grouped from "the section
describes a different paper than the repo runs" (§1–§3) down to "one sentence is
slightly wrong" (§7).

Legend: **[BLOCKER]** = a reviewer will call the paper incomplete or the claim
unsupported; **[GAP]** = missing but additive; **[FIX]** = existing sentence is
inaccurate w.r.t. the code; **[APPX]** = belongs in an appendix you already cite
but have not written into Method.

---

## 1. NULLs is entirely absent [BLOCKER]

The section's opening move is *"The audit is defined against a minimal
interface: a system is auditable if it can decode under a memory state fixed per
generation …"*. That abstraction was written precisely so a second substrate
could be dropped into it — and the repo now has one
(`src/models/nulls_wiki/`, spec in `docs/NULLS_AUDIT_DESIGN.md`, design locked
2026-08-24, v1 implemented, authors' artifacts verified 2026-09-10). None of it
appears in Method. Concretely missing:

### 1.1 The subject itself
`nulls-wiki-1b` audits the released NULLs Wikipedia model (arXiv:2606.13873,
~1B params, SmolLM2 tokenizer, ~6.4M article-level sources). Knowledge lives in
source-keyed sparse **sink-neuron masks** in the MLP, not in an external index;
deletion is *refusing to activate a source's mask*. The Setting paragraph names
only Co-LMLM as "our primary subject" and never says a second subject exists.

Needs: a paragraph defining the second backend in the same terms as Co-LMLM
(what an *entry* and a *query* are for NULLs: the entry is a source/article, the
"query" is the router's nearest-source lookup), and an explicit statement that
the minimal interface is what makes the two comparable.

### 1.2 The single-variable principle and source-level unification [BLOCKER]
This is the load-bearing design decision of the whole cross-substrate
comparison and it is nowhere in Method: Co-LMLM deletes *index entries*, NULLs
deletes *sources*, so **the source (document) is fixed as the common deletion
unit**; a deletion request resolves to a set of sources D(f); both substrates
execute the *same manifest* — Co-LMLM by search-time filtering on
`source_id ∈ D(f)`, NULLs by sink-mask exclusion
(`src/halo/interventions/source_closure.py`, `DeletionManifest.source_ids`).
Entry-level Co-LMLM stays as a *Co-LMLM-internal* condition and an upper bound
on what finer granularity buys.

Without this paragraph, the natural reviewer objection ("you handicapped
Co-LMLM by coarsening it") has no answer in the paper, and the reader cannot
tell that the reported Co-LMLM entry-level numbers and the cross-substrate
numbers are different cohorts of run.

### 1.3 State mapping for NULLs [BLOCKER]
Table missing from Method (design §3, implemented in
`src/models/nulls_wiki/backend.py`):

| HALO state | Co-LMLM | NULLs |
|---|---|---|
| FULL | index intact, retrieval on | gold source's sink mask active (Sink-On) |
| DEL-ON | entries of D(f) filtered, retrieval live | sinks of D(f) excluded, routing live → next-closest **surviving** source's mask |
| DEL-OFF | retrieval disabled | sink memory disabled |

The DEL-ON choice needs a justified sentence in Method: NULLs' DEL-ON uses the
paper's next-closest routing, *not* bare exclusion, because bare exclusion would
make NULLs' DEL-ON a disguised DEL-OFF and inflate its apparent efficacy. The
substitute sink is recorded as the "selected candidate" in a synthetic trace, so
the existing DEL-ON validation (nothing from the manifest may be selected)
applies unchanged. This is exactly the kind of thing a reviewer will assume you
got wrong unless you say it.

Also missing: the statement that eqs. (3a–3d) and the partition (4) are computed
identically for NULLs, with only the *reading* of R changing (memory-mediated
correctness, where memory = index or sink pool).

### 1.4 The shared-encoder closure rule [BLOCKER]
Method defines closures only in Co-LMLM's own retriever space (eq. 5, radius ρ
on ⟨q̂_f, k⟩). For the cross-substrate comparison the repo deliberately does
**not** use either model's internal space: all cross-substrate closures are
computed in one external, model-independent text-encoder space E over source
text, one vector per source (`SourceSpace`, built by
`scripts/build_nulls_title_embeddings.py --mode closure`; the builder refuses to
let a routing-space artifact be used as a closure artifact,
`source_closure.py:70-80`). Source-level predicates
(`SOURCE_PREDICATES = ("provenance", "geometric", "value", "hybrid")`):

- `provenance-source` = {gold article} (k=0; NULLs' native operation, the sweep anchor);
- `geometric-source` = gold + k nearest sources in E;
- `value-source` = gold + envelope sources whose **text** mentions the answer (oracle);
- `hybrid-source` = gold + (geometric-k ∩ value) — **note: intersection**;
- `factq` = Co-LMLM-only, explicitly outside the cross-substrate tables.

Two consequences for the text as written:
1. The Method's hybrid is *"the hybrid union geometric ∪ value ∪ provenance"*.
   The source-level hybrid is an **intersection** (`source_closure.py:230-235`).
   Two different objects share one name in the paper. Rename one.
2. The rationale sentence ("the encoder never touches either model; it defines
   the *request*, which both systems must execute") is the answer to an obvious
   objection and should be in Method, together with the promised dual-encoder
   sensitivity check — which is **still unbuilt** (see §5.3).

### 1.5 The sweep is parametrized differently for NULLs [BLOCKER]
Method's Entanglement paragraph sweeps a cosine radius ρ. The cross-substrate
sweep is parametrized by **deletion breadth k** — the number of nearest sources
deleted, k ∈ {1,2,4,8,16} (`NULLS_SWEEP_K_GRID`, `scheduler.py:415`) — with the
radius grid retained only as a Co-LMLM-internal view. The stated reason belongs
in Method: in a shared space the similarity distribution around a source is
heavy-tailed and radius→set-size is nonlinear, so equal ρ is not equal request
size; k is the operating-point index and the claim lives in the curve.

Also missing: the **matched-collateral readout** — "at the k where each
substrate's collateral first reaches level c, which substrate has forgotten the
target more?" — described in the design as the deletion analogue of comparing
classifiers at matched FPR. Method currently offers only G(f) (eq. 9), which is
a Co-LMLM-space quantity and has no cross-substrate meaning.

Also missing: collateral for the cross-substrate sweep is reported over
**control facts in E-distance bands** from the gold source, not over a cosine
ball of query embeddings. That is a different N(f) than eq. (8b) uses.

### 1.6 DEL-OFF sensitivity has a NULLs analogue [GAP]
Method describes the Co-LMLM DEL-OFF pair (null-retrieval vs forbid-token) and
ablates it. The NULLs pair is implemented and unmentioned:
`sinks-zero` (backbone only, upstream's `dropout` mode) vs `placebo-sink` (a
deterministic, seeded, provably far-away source's mask is activated; cosine
ceiling 0.5 by default, `--nulls-placebo-ceiling`). The argument is stronger
than the Co-LMLM one and should be stated: the network never runs with zero
active sinks during training, so `sinks-zero` is off-distribution and its
degradation could be an activation-statistics shift rather than knowledge
absence. Agreement between modes is what licenses reading DEL-OFF as unaided
answerability.

### 1.7 The verification gate and the NULLs cohort [BLOCKER]
Method defines exactly two cohorts (audited `F`, FULL-correct `F_full`), both
via the retrieval-trace answer-mention test. NULLs has no retrieval trace, so
its per-substrate cohort is defined by a **verification gate**
(`src/models/nulls_wiki/gate.py`, `scripts/nulls_verification_gate.py`): sinks
are keyed by article title through a PRNG, so a wrong title silently activates
an unrelated mask and *mimics deletion*. Every fact must show gold-answer
log-likelihood under Sink-On exceeding that under a placebo mask by margin δ;
failures are excluded as **unmappable** and the exclusion rate is reported as a
substantive statistic.

Method also needs the **intersection cohort** as the primary unit for every
cross-substrate table (facts in both per-substrate cohorts, FULL-correct in
both), with per-substrate cohorts shown alongside.

Implementation caveat worth an honest sentence: the design says δ is *calibrated
on a held-out sample so the false-pass rate is estimable*, but δ = 0 everywhere
in the repo (`nulls_verification_gate.py:52`, `NULLS_GATE_MARGIN` at
`scheduler.py:467`) and no calibration procedure exists — `grep -i calibrat`
hits only the design doc's promise. The CLI help states the intent: "0 records
margins without excluding on them; calibrate from the summary's margin
distribution."

Writing δ = 0 into the paper is a perfectly defensible option, but it needs to
be stated precisely, because δ = 0 is **not** a no-op gate. It still excludes
(a) *unmappable* facts whose title is absent from `title_to_index`, and (b)
every fact whose Sink-On log-likelihood falls **below** placebo. So it catches
the global failure mode the gate exists for — a wrong mapping shows up as a
margin distribution centred on zero, and roughly half those facts fail on sign
alone — while admitting weak-positive margins. What it does *not* deliver is an
estimable false-pass rate, which is what calibration was for.

The reporting infrastructure is already there: `gate.finalize` writes
`margin_mean`, `margin_median`, `margin_stdev` and `exclusion_rate` to
`<output>.summary.json`. So the δ = 0 option costs one Method sentence plus the
margin distribution in an appendix, not new code.

Nitpick if δ = 0 goes into the text: `gate.py:83` tests
`observed_margin >= margin`, so the paper's inequality should be ≥, not the
design doc's "strictly exceeds" (immaterial in floating point, but the text
should match the code it describes).

### 1.8 Corpus, scale, and the missing control [BLOCKER]
None of this is in Method:

- **Matched-corpus configuration (primary)**: Co-LMLM runs with FineWeb filtered
  out at search time (`--co-lmlm-corpus-allow`, `models/co_lmlm/__init__.py:60`)
  so its effective memory is Wikipedia, matching NULLs' training corpus;
  full-index runs are secondary. ⚠️ **The flag is not wired into the cross-model
  scheduler** (it appears only in the backend and the README) — so as the suite
  currently runs, the "primary" matched-corpus configuration is not produced.
  Either wire it or drop the word "primary".
- **Dump/corpus provenance**: the released NULLs corpus is the unmodified public
  `wikimedia/wikipedia 20231101.en` snapshot (verified per-shard row counts,
  first shard row-for-row), bijective with `title_to_index.pkl` (6,407,814 raw
  MediaWiki titles).
- **The exclusion rate is large and must be in Method, not buried**: that
  snapshot omits prominent articles (*Paris*, *Germany*, *Physics*, *Autism*),
  so facts whose subject article has no sink are dropped:
  **T-REx 24%, CounterFact 20%, ZsRE 10%, Google-RE 10%**.
- **Declared, unfixable mismatches**: parameter count (~1B vs 360M), Wikipedia
  dump version, training objective and epochs. Tokenizer is shared (SmolLM2),
  which removes one nuisance variable — say so, it is a real strength.
- **No data-matched parametric control exists for NULLs.** Co-LMLM's
  `standard-lm-360m-fw` has no NULLs analogue: the one repository that looked
  like a no-sink Wikipedia LM (`gauravrghosal/wikipedia_full_8x`) is empty
  (checked 2026-09-12). The matched controls are therefore NULLs' own DEL-OFF
  modes. This asymmetry is a limitation the paper must own explicitly, because
  the parametric-channel calibration argument in the current last sentence of
  the "Three states" paragraph does *not* transfer to NULLs.
- **Secondary readout**: the design commits to truth-ratio-style likelihood
  scoring alongside generation-based correctness for both systems, as a bridge
  to the NULLs paper's numbers and a robustness check on the correctness
  protocol. `answer_logprob` exists (`nulls_wiki/backend.py:336`) but is used
  only by the gate; there is no truth-ratio readout in the analysis pipeline.
  Either build it or cut the claim.

### 1.9 PopQA: no measured exclusion rate yet [GAP]
PopQA **is** in the NULLs pipeline. `data/prepare_popqa_audit.py` emits
`subject` from PopQA's `s_wiki_title` (the exact Wikipedia article title —
arguably the cleanest provenance signal of all five sets, with `subj` kept as
an alias) as of 2026-09-11, `models/nulls_wiki/titles.candidate_titles`
resolves on that field, and the scheduler builds the NULLs prep/gate/audit
chain for every selected dataset with no PopQA special case. `setup_nulls.sh`
rebuilds prompts unconditionally precisely so a stale pre-2026-09-11
`data/prompts.jsonl` cannot silently drop the whole set.

What *is* missing is a number: the exclusion table in
`docs/NULLS_AUDIT_DESIGN.md` (T-REx 24%, CounterFact 20%, ZsRE 10%,
Google-RE 10%) was measured before that fix and has no PopQA row. So Method
can say five sets throughout, but the per-set exclusion table it will cite
needs a PopQA entry — and PopQA's rate is the one most likely to differ, since
its subjects are already exact article titles rather than surface strings
requiring MediaWiki normalization.

---

## 2. Two of the four non-Co-LMLM models are missing [BLOCKER]

The Method names **SmolLM2-360M** and **the Standard LM baseline**. The registry
has four non-Co-LMLM backends, all scheduled by
`scripts/run_cross_model_scheduler.sh`:

| Backend | Checkpoint | Role | In Method? |
|---|---|---|---|
| `smollm2-360m` | `HuggingFaceTB/SmolLM2-360M` | off-the-shelf parametric reference at Co-LMLM's scale | ✅ |
| `standard-lm-360m-fw` | `lil-lab/CoLMLM-Standard-LM-Baseline-360M-FW` | **data- and architecture-matched** control for Co-LMLM | ✅ (but under-described, see below) |
| `smollm2-1.7b` | `HuggingFaceTB/SmolLM2-1.7B` | scale reference for NULLs | ❌ |
| `nulls-wiki-1b` | `gauravrghosal/NULLS-Wikipedia-Full` | third deletion substrate | ❌ (§1) |

- **SmolLM2-1.7B [GAP]**: same role for NULLs that SmolLM2-360M plays for
  Co-LMLM, one size up. SmolLM2's released sizes are 135M/360M/1.7B, so 1.7B is
  the nearest neighbour of a ~1B model in the same family with the same
  tokenizer, and it brackets NULLs' Sink-On correctness *from above* where the
  360M entries cannot. It is a **reference, not a control** — trained on the full
  SmolLM2 web corpus vs NULLs' Wikipedia-only, so the bracket is confounded by
  training data as well as scale, and no causal claim rests on it. All of that
  nuance needs to be in the text; a bare extra bar in a figure invites the
  objection.
- **`standard-lm-360m-fw` is under-sold [FIX]**: Method calls it "a standard LM
  with Co-LMLM's architecture trained on its corpus without memory". Sharpen it:
  it is the *released* Standard-LM checkpoint from the Co-LMLM paper — same
  SmolLM2-360M architecture, same FineWeb-Edu corpus, unannotated text, no
  retrieval — i.e. the matched-data control that isolates the externalization
  recipe with training data held constant, while SmolLM2-360M is the
  off-the-shelf reference. Two baselines doing two different jobs; right now the
  sentence reads as if they were interchangeable.
- **The "three states coincide" claim [FIX]**: correct in code (each closed-book
  `__init__.py` rejects `--closure/--radius-grid/--adversarial/--bootstrap-oracle-from-full`
  up front and all three states re-run the same computation), but it must now be
  stated for *three* closed-book backends, and explicitly **not** for
  `nulls-wiki-1b`, where the three states are genuinely different computations.

---

## 3. The statistical protocol is missing [BLOCKER]

Method reports plain cohort means (eq. 6) and nothing else. Meanwhile
`scripts/robustness_analysis.py` computes, and the paper presumably uses:

- **Wilson 95% CIs** for S/R/I/L on the FULL-correct cohort, for baseline
  closed-book correctness on identical facts, and for adversarial attack-gain;
- **paired McNemar exact tests** for (a) Co-LMLM L vs each baseline on shared
  facts and (b) null-retrieval vs forbid-token DEL-OFF controls;
- **backfill sensitivity** (entanglement restricted to strictly within-ball
  neighbours);
- **factq subcohort** policy comparison on identical facts;
- **sweep-channel reconciliation** (combining each sweep radius's DEL-ON with the
  standard run's DEL-OFF to get S/R/I for pure geometric deletion at every ρ);
- **sweep deletion cost** (entries deleted per fact per radius: median/mean/p90/
  max, truncation share);
- **artifact-rate decomposition** into a parametric part and a retrieval-only
  remainder.

And `docs/NULLS_AUDIT_DESIGN.md` §11 commits to more, none of it implemented:
unit of analysis = the fact with all resampling clustered by fact; paired
differences on the intersection cohort with **cluster-bootstrap 95% CIs**;
operating curves with bootstrap bands; "no point-estimate-only claims";
pre-registered family of headline comparisons with exploratory cuts labelled;
and an explicit statement that with a single released checkpoint per system **no
training-seed variance is claimable** — all variance statements are across facts.

`grep -rn bootstrap src scripts` returns only `--bootstrap-oracle-from-full`.
So: the design promises cluster bootstrap, the code delivers Wilson + McNemar,
and Method promises nothing. Pick one and write it down. A short "Statistical
protocol" paragraph is the single highest-value addition to the section.

Also unstated and needed: **determinism** (greedy decoding everywhere,
`--max-new-tokens 12` by default), and the reuse **canary** — 1% of reusable
generations are re-executed and asserted equal (`--reuse-canary-rate 0.01`),
which is a real soundness argument for the cross-phase reuse the suite depends
on.

---

## 4. Co-LMLM machinery that exists, is used in figures, and is unmentioned

### 4.1 The factq policy row [GAP]
Method describes `<FACT-q>` well as a *predicate*, but not that it is run as a
full **policy row** with its own results (`policy_matrix/factq/` for all five
sets), its own figures (`figures/factq_coverage.png`, `factq_vs_value.png`),
and a coverage problem: facts the ensemble never covered are recorded with
`factq_query_count = 0` rather than silently treated as complete
(`closure.py:ClosureResult.factq_query_count/factq_truncated`), which is why the
robustness script compares policies on the factq-covered subcohort. Method
should say that the factq row is compared on that subcohort, otherwise the
policy table is apples-to-oranges.

Also unmentioned: the generator is `lil-lab/CoLMLM-Question-Generator` (LoRA
adapter over `Qwen/Qwen2.5-1.5B-Instruct`), questions are generated from the
FULL-retrieved entry with the answer span marked as fact span 1, oversampled 2×
and deduplicated. The "never from the audit prompt" invariant is correctly
stated in Method and is enforced in code — good, keep it.

### 4.2 Adversarial evaluation is richer than described [GAP]
Method describes a single survivor at cosine ρ−ε with four value templates.
`src/halo/interventions/adversary.py` also implements **four injection
topologies**: `single`, `aliased` (`count` survivors all conveying the answer),
`collided` (survivor + wrong-answer decoys at the same cosine), `saturated`
(decoys placed *closer* than the survivor at ρ+ε, so the survivor must win
top-1 against active competition). Defaults: ε ∈ {0.01, 0.02, 0.05}, count 3,
seed 0. The reported numbers filter `topology == "single"`
(`status_update_2_analysis.py`), so the other three are currently computed and
unused — either report them (saturated is the strongest result if it works) or
say in Method that the reported attack is the single-survivor topology and the
others are appendix.

Implementation detail worth one appendix sentence [APPX]: injections are
**spliced into search results at filter level**, not written into the FAISS
index; `Ev` is measured against a store that is the filtered index plus the
synthetic entry. That is what makes eq. (10) an exact intervention rather than
an index rebuild.

Also unmentioned: the **margin geometry** `s_del`/`s_surv` and
`margin = s_del − s_surv` recorded per closure, plus the top-5 surviving
candidates (`ClosureResult.top_survivors`) — the "margin predictor" the
adversarial runner reports. If it is in the results, define it in Method.

### 4.3 Interference calibration [GAP]
`scripts/interference_calibration.py` tests a specific reading of I(f): that the
harmful splice sits just above the retrieval threshold τ, i.e. interference is a
calibration failure. It compares the cosine of the DEL-ON selected entry between
benign and interference outcomes among facts where interference is possible at
all (FULL-correct ∧ DEL-OFF-correct). Method introduces I as a channel but never
says it is diagnosed; one sentence and a pointer would pay for itself, since
"why does deletion ever *hurt*" is the first question a reader has after eq. (3d).

### 4.4 Value-filter residual adjudication [GAP]
The value policy's retrieval-mediated survivors (facts correct only with
retrieval on, under the oracle answer filter) are dumped and **manually
adjudicated** into surface variants vs associative cues
(`annotations/status_update_2/value_survivor_labels.csv`; the analysis script
carries `REPORTED_SPLIT = {surface_variants: 49, associative_cues: 61}` of 110
and warns on divergence). This is the honest answer to "your `match` predicate
misses paraphrases" — a known open item from the July 2026 feedback — and it is
human-labelled evidence, which reviewers weight highly. It is not in Method.

### 4.5 Bookkeeping the Method promises implicitly [APPX]
Recorded per run and never described: skipped facts (`*_skipped_facts.jsonl`),
closure truncation flags (geometric page-full at `max_closure_size = 10 000`;
factq truncation), `index_nprobe`, the audited event index, per-entry predicate
attribution (`caught_by`) in the closure artifacts, and per-run audit configs.
Method says "the geometric closure is materialized up to a size cap with an
explicit truncation flag" — good — but the cap value (10 000), the envelope
(500), and the nprobe are never given.

---

## 5. Promised-but-unbuilt work that Method must not imply exists

Flagging these because Method is written in the present tense throughout, and
three things it would naturally be read as covering do not exist yet:

1. **Shared adversarial / paraphrase-extraction phase** (design §8): a
   substrate-neutral paraphrase and indirect-prompt extraction phase after
   DEL-ON, identical attack prompts for both substrates. Listed as
   "Outstanding" in the design header. Method's Adversarial paragraph is
   Co-LMLM-only by construction (no write path into NULLs' weights) — say that
   explicitly, and say that robustness comparison is future work, or a reviewer
   will read the adversarial section as the robustness comparison.
2. **Dual-encoder sensitivity check** (design §4/§12): the stated mitigation for
   "your external encoder favours one system". Unbuilt (also tracked in the
   project memory as an open item).
3. **Cross-substrate figures**: `scripts/paper_figures.py` has no substrate
   dimension — `DATASETS`, `POLICIES`, `SMOL`, `STD` are Co-LMLM-shaped
   constants. Same for `src/halo/analysis/followups.py`, whose model list is
   hard-coded to `co-lmlm`, `standard-lm-360m-fw`, `smollm2-360m` (no NULLs, no
   1.7B) — so the frequency/memorization and probe-control analyses currently
   cannot include the new models.
4. **No NULLs results exist yet**: `results/` contains only `co-lmlm/`,
   `smollm2-360m/`, `standard-lm-360m-fw/`. Whatever Method claims about NULLs
   is at present a description of a pipeline, not of numbers.

Also: the router-steering threat (adversarially steering NULLs' router to
activate a deleted sink) is explicitly out of scope in v2. Method should say so
in the same breath as the adversarial-write threat model, so the scoping reads
as deliberate rather than overlooked.

---

## 6. Method↔code discrepancies in the Co-LMLM part

These are places where the current text does not match what the code does.

1. **[FIX] `match` is not shared by the source-level value predicate.**
   Method: "`match` … is shared by the cohort definition below, the value
   predicate (5), and the density count (7)." True at entry level
   (`interventions/judge.py:22` implements the bidirectional rule with the
   3-character floor exactly as eq. 2). But the **source-level** value rule
   (`source_closure.py:142-148`) is *one-directional only* — whole-phrase
   containment of an alias in the source text, no reverse containment, no length
   floor. The cross-substrate value closure therefore uses a different predicate
   than eq. (2). Either align the code or add a sentence.
2. **[FIX] Entanglement targets are additionally conditioned on FULL-correct.**
   Method: "G(f) … defined only for facts with a non-empty neighbor set."
   `core/entanglement.py:71` also skips every target that was incorrect under
   FULL. The reported G distribution is over FULL-correct targets with non-empty
   N(f). Say so — it matters for reading "share with G=0".
3. **[FIX] Collateral's denominator.** Eq. (8b) divides by |N(f)| while only
   FULL-correct neighbours can contribute to the numerator, so X is bounded
   above by the FULL-correct share of N(f) and is *not* on the same scale as E.
   The code does exactly this and records `neighbors_full_correct` per point.
   It is a defensible choice but it needs one clarifying clause, otherwise the
   operating curve and G(f) look like they trade off two commensurable rates.
4. **[GAP] Neighbour construction constants are absent.** Defaults
   (`cli/args.py`): `--neighbor-mode cosine`, `--neighbor-ball 0.5`,
   `--neighbor-cap 20`, `--neighbor-min-count 5`. Method gives the 3 904/7 407
   backfill figure for T-REx but never the ball, the cap, or the minimum — so
   the reader cannot tell what "capped per fact" or "a minimum count" mean.
   Also unmentioned: a second neighbour mode, `same-source`, exists.
5. **[GAP] The probe's hyperparameters and second mode.** Method describes
   ridge + proposition-grouped k-fold + hashed character-trigram answer features.
   Unstated: k = 5, λ = 1.0, feature dim 2048, seed 0, the MIN_PROBE_FACTS = 4
   guard, and that `ProbeConfig` also supports a `classification` mode
   (one-hot targets over the candidate set) which the paper does not use.
6. **[FIX] The probe's third control.** Method: "a prior that always predicts
   the cohort's most frequent gold answer." What the analysis actually reports
   is `majority_gold_share = max(counts)/n` — the in-cohort majority *share*,
   not a cross-validated prior. A properly folded majority baseline **does**
   exist in `analysis/followups.py:_prior_baselines` (`global_majority_accuracy`,
   plus a stronger per-**relation** majority prior that Method does not mention
   at all). Use the folded numbers, and consider reporting the relation prior —
   it is the harder control and the one a reviewer will ask for.
7. **[FIX] Which embedding table the mean-token control uses.** Method says
   SmolLM2-360M, and the paper pipeline does exactly that
   (`status_update_2_analysis.load_embed_table`, with the stated reason: the
   Standard-LM tokenizer config does not load under the pinned transformers
   version). But `analysis/followups.py --embedding-model` documents itself as a
   *Co-LMLM* checkpoint. Two code paths, one control. Make sure the paper
   describes the one that produced the numbers, and consider deleting the other.
8. **[GAP] φ is not part of the library.** The Matthews correlation between
   L_rep and L exists only in `scripts/status_update_2_analysis.py:298,492`, not
   in `core/probe.py` or `core/metrics.py`, and is not recomputed by the
   robustness script's validation pass. Fine for the paper, worth knowing for
   reproducibility claims.
9. **[GAP] Probe embeddings are event0, closures are the selected event.**
   Method already handles this honestly ("taken at the first retrieval event of
   each FULL trace, which for a small number of facts is not the audited
   answer-mention event"). Confirmed in code: `load_probe_samples` filters
   `event0`, while `full_query_vector` resolves `selected_event_index`. Consider
   quantifying "a small number" — the trace records
   `selected_answer_visible_before_event`, so it is computable.
10. **[GAP] The artifact rate's eligibility condition.** Method defines the
    artifact rate but not that it is computed only over facts whose DEL-ON trace
    is *complete* (`trace_available ∧ trace_complete ∧ retained_candidates`,
    `metrics.trace_is_complete`) — a different denominator from the rest of the
    table, reported separately in the CSVs
    (`retrieval_artifact_eligible_count`).
11. **[GAP] `trace_has_gold_equivalent` is stronger than `match`.** The artifact
    rate's "no retained candidate mentions the answer" test also accepts a full
    subject–relation–object equivalence when the candidate carries structured
    fields. Worth a clause since the artifact rate is defined by its negation.
12. **[APPX] Retrieval threshold and decoding.** τ = 0.7 (the Co-LMLM
    factual-eval setting) appears in the README and as the factq default but
    never in Method, which introduces τ symbolically and never instantiates it.
    Same for greedy decoding and the 12-token generation cap.

---

## 7. Smaller textual issues

- **"Roughly 2.2B entries"**: fine, but the index is ~1.05 TB and that number
  (plus nprobe) belongs in the compute appendix you already cite.
- **DEL-OFF operator names**: `null-retrieval` / `forbid-token` are correct and
  match `scheduler.py:119`. When NULLs enters, the sentence needs to say the
  *pair* generalizes: two implementations of "memory off" per substrate, and
  agreement within each pair is what licenses the DEL-OFF reading.
- **"Deletion is non-destructive search-time filtering … which makes all three
  exact interventions on one model and index"**: true for Co-LMLM; for NULLs the
  equivalent statement is that masks are computed on the fly from the source id,
  so exclusion is likewise non-destructive and every run records the activated
  sink set. Worth generalizing rather than duplicating.
- **The `rel_lmlm` citation** (`\citet{raeesi2026relational}` for the three
  states) is fine as provenance, but note the audit is 5×3 now — don't let the
  sentence imply a fourth substrate.
- **Eq. (5) `src(v) = src(f)`**: in code, provenance exclusion happens *by source
  id at search time and is unbounded by the observed envelope*; what appears in
  the closure artifact is attribution only (`closure.py`, the comment at the
  `source_ids` block). The manifest semantics (source-level, unbounded) vs the
  artifact semantics (observed entries only) is exactly the distinction that
  makes §1.2's source-level unification possible, so it is worth one sentence
  even in the Co-LMLM-only framing.

---

## 8. Suggested restructuring of the section

Minimal-churn plan that gets all of the above in without rewriting what works:

1. **Setting** — split into *Systems* (Co-LMLM; NULLs; the three closed-book
   references) and *Interface* (the minimal auditable interface, unchanged).
   Add the single-variable principle + source-level unification here; it is the
   premise everything downstream depends on.
2. **Three states and channel decomposition** — keep eqs. (1)–(6) verbatim; add
   the state-mapping table (§1.3) and generalize the closed-book collapse
   sentence to three models.
3. **Cohorts** — promote to its own short paragraph (it currently hides inside
   "Three states"): audited cohort, FULL-correct cohort, NULLs verification gate,
   intersection cohort, exclusion rates.
4. **Deletion closures** — keep the entry-level definitions; add a "source-level
   closures and the shared encoder" sub-paragraph, and rename one of the two
   hybrids.
5. **Answer density** — unchanged (it is already correctly framed as a property
   of the fact and the index; note it has no NULLs analogue).
6. **Entanglement** — keep the ρ-sweep as Co-LMLM-internal; add breadth-k and
   matched collateral for the cross-substrate curve.
7. **Representational leakage** — unchanged except hyperparameters and the
   corrected prior control.
8. **Adversarial closure** — keep; add the explicit scoping sentence (Co-LMLM
   external-memory threat model only; NULLs router attacks and the shared
   paraphrase-extraction phase are future work).
9. **New: Statistical protocol** — the paragraph from §3.
10. **New or appendix: Threats to validity** — `docs/NULLS_AUDIT_DESIGN.md` §12
    is already a paper-ready table (threat / mitigation / residual). Lift it.

---

## 9. Checklist

| # | Item | Severity | Where it lives in the repo |
|---|---|---|---|
| 1 | NULLs as a second subject | BLOCKER | `src/models/nulls_wiki/` |
| 2 | Source-level unification / single-variable principle | BLOCKER | `source_closure.py`, design §1–2 |
| 3 | NULLs state mapping + next-closest routing | BLOCKER | `nulls_wiki/backend.py`, design §3 |
| 4 | Shared-encoder closure space | BLOCKER | `source_closure.py`, `build_nulls_title_embeddings.py` |
| 5 | Breadth-k sweep + matched collateral | BLOCKER | `scheduler.py:415`, design §5 |
| 6 | Verification gate + intersection cohort + exclusion rates | BLOCKER | `nulls_wiki/gate.py`, design §9 |
| 7 | Corpus/scale controls; no data-matched NULLs control | BLOCKER | design §10 |
| 8 | PopQA exclusion rate unmeasured (set itself is included) | GAP | `titles.py`, design §9 table |
| 9 | Statistical protocol (Wilson/McNemar; bootstrap promised) | BLOCKER | `robustness_analysis.py`, design §11 |
| 10 | SmolLM2-1.7B scale reference | GAP | `src/models/smollm2_1_7b/` |
| 11 | Standard-LM described as the matched-data control | FIX | `standard_lm_360m_fw/__init__.py` |
| 12 | NULLs DEL-OFF pair (sinks-zero / placebo-sink) | GAP | `nulls_wiki/__init__.py`, design §6 |
| 13 | factq as a policy row + covered subcohort | GAP | `policy_matrix/factq/`, `robustness_analysis.py` |
| 14 | Adversarial topologies, ε grid, splice-level injection | GAP | `interventions/adversary.py` |
| 15 | Interference calibration diagnostic | GAP | `interference_calibration.py` |
| 16 | Value-survivor manual adjudication (49/61 of 110) | GAP | `annotations/status_update_2/` |
| 17 | Source-level value predicate ≠ eq. (2) | FIX | `source_closure.py:142` |
| 18 | Entanglement targets also conditioned FULL-correct | FIX | `entanglement.py:71` |
| 19 | Collateral denominator clarification | FIX | `entanglement.py` |
| 20 | Neighbour/probe/closure constants unstated | APPX | `cli/args.py`, `core/probe.py` |
| 21 | Majority-prior control is a share, not a folded prior | FIX | `status_update_2_analysis.py`, `followups.py` |
| 22 | Unbuilt: shared paraphrase phase, dual-encoder check, cross-substrate figures/analysis, NULLs results | BLOCKER (scoping) | design §8/§12, `paper_figures.py`, `followups.py`, `results/` |
