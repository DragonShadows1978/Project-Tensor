# APA-MF2.1 — close the MF2 review's findings: S < L contract, −inf-safe softmax sentinels, item 1 against the definition, test gaps

David, 2026-10-07: "Fix it dude, fix it." The MF2 blind review (`apa-mf2rev` 8100409; `docs/APA_MF2_REVIEW.md`,
`artifacts/apa_mf2rev/verdict.json`) returned **REJECT_HOLD_PROMOTION**:
- item 1 REJECT;
- item 2 REJECT;
- item 3 APPROVE-WITH-NITS.

The lead ran its GPU probes on 2026-10-07 (receipts `artifacts/apa_mf2rev/gpu_*_20261007T21*.json` on that branch):
- identity, sliding, GRAPA and census PASS;
- negative, short and rule RED, as the review found;
- leak RED only on rows where the pre-existing quantizer overflows (magnitude 1e30);
- extension not built.

What the findings mean:
- **F1 (BREAK, item 2).** The engine softmaxes start their running max at a FINITE sentinel:
  - `kernels.cu` reduce MODE 1 `-3.4e38f` (min MODE 2 `3.4e38f`), around line 560;
  - `causal_softmax` `float mx = -3.0e38f`, around line 7225;
  - `ops.cpp:841` softmax uses `reduce_max`.

  A visible score below the sentinel (for example −3.3e38) makes the row NaN once every masked entry is −inf. This
  predates MF2; the old −1e4 bias leaked on such rows instead.
- **F2 (BREAK, item 1).** On constructed near-tie inputs, BK4's one-pass FMA statistics differ from the composed
  ones by up to 2.8e-3 relative. That is outside the 2e-5 band, with no dot-product error involved. The rule probe
  shows the COMPOSED path matches the fp64 DEFINITION (`composed_vs_fp64: 0`), while the KERNEL deviates. Causes:
  cancellation in one-pass `E[x²]−E[x]²`, and the single-visible-key quirk. **The lead's MF2 target ("bit-identical
  to BK4") was wrong.** The contract is the mathematical definition.
- **F3 (SUSPICIOUS, items 2 and 3).** Causal S < L leaves rows empty and gives NaN. The review recommends raising
  before any compute, which matches the CUDA kernels (`kernels.cu`: "bottom-right causal requires S >= L").
- **F4 (SUSPICIOUS).** Single-site reversion gaps:
  - fp16 additive-bias reversion is not discriminated;
  - an `apa.cpp` kNegInf-only reversion is not caught;
  - a count-divisor mutant survives the rule fixtures.
- **F9 (SUSPICIOUS).** `test_apa_phase6._ref` still masks SCORES top-left, while its statistics are bottom-right.

## Mechanism & scope

- Measures:
  1. Causal S < L is rejected with a clear error before any compute, on every causal path in the writable files.
  2. With −inf masking, every row that has at least one visible finite score produces finite output, whatever the
     score magnitude, because the max reductions and softmaxes start at −inf. Every finite input stays byte-identical.
  3. `apa_quant_attention`'s refine selection equals the fp64 DEFINITION: population mean and std of the visible
     |bulk| (fp32 bulk dot), threshold `mean + z·std`, and select visible ∧ |bulk| ≥ thr. Agreement is within an fp32
     tie band on all inputs, adversarial near-ties included. Kernel deviations from the definition are measured and
     REPORTED, not fixed here.
  4. The F4 and F9 test gaps are closed.
- Mechanism doc: `tensor_cuda/tensor_cuda/functional.py` (the `_causal_mask` comment block). Also
  `docs/APA_MF2_REVIEW.md` on `apa-mf2rev`, and `kernels.cu` BK4 lines 8040–8155 (the kernel deviation being
  reported).
- Exercises:
  - `tensor_cuda/src/kernels.cu`: the generic reduce kernel's MODE 1/2 init, `causal_softmax`'s max init, and any
    other max/min SENTINEL on an attention softmax path (audit; change only sentinels);
  - `tensor_cuda/tensor_cuda/functional.py` (S < L check);
  - `tensor_cuda/tensor_cuda/quant.py` (S < L check; selection statistics);
  - `apa_cuda/csrc/apa.cpp` (S < L `TORCH_CHECK`);
  - the tests listed below;
  - the reviewer's probes (`.../tensor_fork_apa_wt_mf2rev/tests/apa_mf2rev/gpu_probe.py`; copy it).
- Does not cover:
  - changing any APA kernel's selection statistics (BK4 and 'a' keep their one-pass rule; the deviation is reported
    for David);
  - the quantizer's |k|² overflow above about 1e19;
  - the extension build (needs CUDA 13);
  - `tensor_gpu_v2`'s finite −1e9 fills;
  - promotion.
- Required evidence:
  - an S < L raise receipt for each path;
  - the F1 negative probe: before RED, after PASS;
  - byte-identity before (MF2 HEAD) vs after on finite inputs:
    - real captures in 3 dtypes through SDPA (inference and grad), `causal_softmax` and `apa_quant_attention`;
    - GRM standard logits on one wikitext window;
    - the GRAPA forward;
    - a general `reduce_max` / `reduce_min` sweep;
  - definition-agreement tables: real captures, the review's constructed near-tie cases and synthetic cases. Each
    with composed-vs-fp64 and kernel('a','h')-vs-fp64 counts, inside and outside the band;
  - mutation receipts, including the F4 mutants.
- Operand census:
  - real captures (Qwen3-1.7B 28×16/8×D128 and 4B 36×32/8×D128, S=8192, bf16);
  - synthetic L ∈ {1, 3, 64, 513}, S ∈ {L, L+1, 4L+5}, D ∈ {64, 96, 128};
  - near-tie constructions from F2 (S ∈ {1, 2, 256, 2048});
  - finite magnitudes up to ±3.4e38 for the softmax-sentinel cases.
- Baseline isolation:
  - "before" = `apa-mf2` HEAD 048949d (copy the files to `artifacts/apa_mf2_1/before/`);
  - the fp64 oracle is your own, written from the definition;
  - the byte-identity arms are separate builds or imports of the before and after code, never two runs of the new
    code.
- Forbidden configuration:
  - a test that encodes BK4's arithmetic as the contract;
  - a tie band wider than about 2e-5 relative without a measured justification;
  - changing a reduction's semantics beyond its ±inf start value;
  - the retired 595.84 driver override (it breaks CUDA).

Seat: Claude Opus 5.5, high (an Agent-tool subagent; it has the GPU). The Codex re-review follows.

## Where to work

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_mf2`
(branch `apa-mf2`, HEAD 048949d plus this order). Writable only:
- `tensor_cuda/src/kernels.cu` (reduction and softmax max/min INIT SENTINELS only);
- `tensor_cuda/tensor_cuda/functional.py` and `tensor_cuda/tensor_cuda/quant.py` (S < L checks; the
  `apa_quant_attention` selection block);
- `apa_cuda/csrc/apa.cpp` (S < L check only);
- `tensor_cuda/tests/test_apa_phase6.py` (`_ref` score mask only);
- `tensor_cuda/tests/test_apa_mf1_quant.py` (the S < L test becomes "raises");
- `tensor_cuda/tests/test_apa_mf2_*.py` and `apa_cuda/tests/test_apa_mf2_*.py` (move the BK4-parity assertions to
  definition parity, plus the F4 gap tests), and new `test_apa_mf2_1_*.py`;
- `scripts/apa_mf2_1_*`, `artifacts/apa_mf2_1/`, `docs/APA_MF2_1_LEDGER.md` and `logs/`.

Rebuild the engine into `artifacts/apa_mf2_1/build/`.

**READ-ONLY:** the production repo `/mnt/ForgeRealm/Project-Tensor` (outside `Fork/`), GraftRepository,
GRAPA-Native-LLM, every other worktree, and the HF cache.

NO git. NO subagents. NO DELETES (`delete-me … --session apa-mf2-1`). Never kill a process you did not start. GPU:
`flock -w 900 /tmp/forge-gpu.lock timeout 600 …`, native driver only. NEVER set the 595.84
`LD_LIBRARY_PATH`/PATH override.

## Required

1. Tests first, with a failing receipt for the F1, F3, F4, F9 and definition-contract cases.
2. The S < L raise on all four paths. Then update MF1's S < L test to expect it.
3. The sentinel change: −inf/+inf starts, with an audit of every max/min reduction start in `kernels.cu` touched by
   softmax or attention. Prove byte-identity on finite inputs.
4. The item-1 selection is the definition: stable centered two-pass population statistics over the visible keys,
   with fp32 bulk. Move MF2's BK4-parity tests to definition parity. Keep a NON-gating observation test that records
   kernel-vs-definition deviations.
5. The F4 and F9 gaps: the fp16 additive-reversion discriminator, the `apa.cpp` kNegInf-only mutant check (by
   source/stand-in), the count-divisor mutant killed, and `phase6 _ref` bottom-right.
6. Re-run the reviewer's probe cells (copied, with definition-based variants where the probe asserted BK4
   exactness), plus the GRM `"apa"` chunked KL (as MF2) for MF2 vs MF2.1.
7. Mutation ≥ 0.80, including the F4 mutants and a "sentinel back to −3e38" mutant.

## Gates — registered BEFORE any GPU gate

`artifacts/apa_mf2_1/registration.json` and its `.sha256`, create-only.
- **G1:** the full `tensor_cuda/tests` suite (`-q -p no:cacheprovider --ignore=tensor_cuda/tests/test_selector_accuracy.py`).
  Its FAILED/ERROR set equals the fd32b6d baseline's 52 ids exactly: no new failures, and every test you edited
  passes. `apa_cuda` tests are equal or better.
- **G2:** S < L raises on every path, with a clear message, before any matmul.
- **G3:** the F1 negative probe passes. Finite inputs are byte-identical before vs after on every listed arm. No NaN
  or Inf for any row with a visible finite score.
- **G4:** composed-vs-fp64-definition selection agreement: 0 mismatches outside the registered tie band on every real,
  synthetic and near-tie cell. Kernel deviations are reported in a separate table.
- **G5:** mutation ≥ 0.80, with the F4 mutants killed.
- **Sanity/adjudication:** every anomaly is explained in one line.

Lead predictions:
- P1: the sentinel change is byte-identical on all finite inputs;
- P2: the composed path is at 0 out-of-band mismatches against the definition everywhere;
- P3: the kernels' out-of-band deviations concentrate on rows with ≤ 2 visible keys and on constructed near-ties
  (zero on real data);
- P4: the GRM KL is within noise of MF2.

## Rules

- HOUSE_RULES apply. Registration before gates. RED honesty.
- Prior Art Directive:
  - Chan, Golub & LeVeque (1983), on cancellation in one-pass variance;
  - Welford (1962);
  - FlashAttention-2's −inf initialization (Dao, 2023);
  - PyTorch's `amax`/`softmax` −inf identity;
  - mutation testing (DeMillo et al., 1978).

  Mark anything unverifiable "unverified — lead to check".
- Keep to about 2.5 hours.

## Done

You cannot write the report file. Write `artifacts/apa_mf2_1/verdict.json` and `docs/APA_MF2_1_LEDGER.md`. Then give
the final message, which is the report:
- the G1–G5 verdicts with the registration sha;
- the tables;
- predictions;
- the files changed, with the diff stat and one patch per item (`artifacts/apa_mf2_1/item{1..4}.patch`);
- deviations;
- open items for David, including the kernel-statistics deviation;
- receipts cited as `path:line "quote"`.
