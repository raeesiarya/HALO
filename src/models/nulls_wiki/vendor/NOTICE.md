# Vendored NULLs (SeqTD) code

Source: https://github.com/AR-FORUM/NULLS at commit
`5753ef7b6f24f31b0d4f0f8b7e56f4535a0b505c` (cloned 2026-08-24), paper
arXiv:2606.13873 (Ghosal, Maini, Raghunathan).

The upstream repository carries **no license file** as of that commit. The
code is vendored here for research reproduction of the released checkpoints
(`gauravrghosal/NULLS-Wikipedia-Full`); redistribution terms must be
clarified with the authors before this repository is published with the
vendor directory included.

Files and the edits made (everything else is verbatim):

- `masking.py` — `batch_seqtied_mask_mult` and `find_multiple` copied
  verbatim from `MemSinks/src/src/SeqTDModel.py`. The mask function is the
  bit-for-bit contract between source ids and sink neurons; it must never be
  reformulated (its arithmetic intentionally relies on deterministic int64
  wraparound in `torch.pow`). `union_exclusion_mask` is a HALO addition
  (multi-source exclusion as the union of per-source masks; single-tensor
  path unchanged), unit-tested for N=1 equivalence.
- `seqtd_config.py` — `SeqTDConfig` from `MemSinks/src/src/SeqTDConfig.py`.
  Edits: imports rewritten to this package; the `litgpt`-catalog fallbacks
  (`from_name`/`name_to_config`) removed because checkpoints ship a
  `model_config.yaml`; `mlp_class` resolves `LLaMAMLPSeqTD` from
  `.seqtd_model`.
- `seqtd_model.py` — `CausalSelfAttention`, `LLaMAMLPSeqTD`, `BlockSeqTD`,
  `GPTSeqTD` from `MemSinks/src/src/SeqTDModel.py`. Edits: imports rewritten
  (mask function from `.masking`); `do_softcapping` imported with a
  fallback definition (verbatim from litgpt 0.5.5) because the repo's
  torch<2.5 pin caps litgpt at 0.5.4, which predates it — the released
  checkpoint has both softcapping options null, so the function is never
  called for it, and every other litgpt helper the code uses
  (`apply_rope` with 3-D cos/sin, `build_rope_cache`, `build_mask_cache`,
  `KVCache`, `batched_index_select`) is signature- and
  semantics-compatible between 0.5.4 and the 0.5.5+ the upstream code was
  written against; debug `print`s dropped;
  `LLaMAMLPSeqTD.forward` additionally accepts a *sequence* of
  `exclude_seq_ids` tensors whose masks are unioned before exclusion
  (upstream supports a single id; the single-id path is unchanged and the
  union path is unit-tested for N=1 equivalence in
  `tests/test_nulls_backend.py`); `torch.cuda.empty_cache()` per forward
  removed (throughput).

Upstream's `SeqTDGenerate.py` was *not* vendored: its loop feeds one token
per step without a KV cache and is not a faithful decoder. The audit backend
implements its own full-context greedy loop instead
(`src/models/nulls_wiki/backend.py`).
