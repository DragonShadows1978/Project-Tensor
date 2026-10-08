# APA-MF2.2 — fix N1 (a new regression), make the definition statistics range-robust (N2), and adjudicate every red GPU probe

David, 2026-10-07: "Fix it dude, fix it."

The MF2.1 blind re-review (`apa-mf21rev` 04b3e02: `docs/APA_MF2_1_REVIEW.md`, `artifacts/apa_mf21rev/verdict.json`)
rejects the chain over two findings:
- **N1, BREAK, a NEW regression from MF2.1 item 2 (93810ef).** The two non-causal fused online-softmax kernels
  (`kernels.cu` near 5070–5088 and 6185–6213) now start `m = -INFINITY`. When the first computed or stored score is
  itself −inf (a finite-operand overflow, for example fp32 Q=1e20·e0 with K0=−1e20·e0, or fp16 partial-INT4 with
  Q=2·e0 and K0=−60000·e0), `corr = expf(m - m_new)` = `expf(-inf - -inf)` = NaN, and that poisons the row. The old
  −3e38 start avoided this. The lead confirmed it on the GPU: the `fused_overflow` cell outputs NaN.
- **N2, BREAK, pre-existing range weakness.** `quant.py:203` squares the centered values in fp32: `var = (dev_*dev_).sum/cnt`.
  This overflows for |bulk| near 1e20 (threshold +inf, selects nothing) and underflows near 1e-30 (variance 0). Both
  fall outside the fp64 definition band. The lead confirmed it on the GPU: the `definition` cell is RED.
- **N5:** the MF2.1 receipts claim "481/481 finite-input keys". The reviewer recounts 529 total, 526 non-design equal
  and 419 strict-finite non-design; 481 matches none of these.

The lead's run of the re-review's GPU probes (`.../tensor_fork_apa_wt_mf21rev/logs/apa_mf21rev_gpu_*_20261007T2*.log`)
produced further REDs that need adjudication:
- **sentinel:** 1 bf16 row, `'NoneType' object has no attribute 'float'`;
- **reductions:** 10 bf16 rows, the same error;
- **training:** 6 `GRAPA_native_train/bfloat16/*` rows, with `x.grad is None` or non-finite (fp32 and fp16 pass);
- **masks:** `geometry/2048x2048`, 36/2048 elements off by about 2.3e-4 at magnitude 255 against an absolute
  tolerance of 2e-4;
- **identity_mf2:** the harness's `048949d` arm exits 1;
- **short:** PASS.

## Mechanism & scope

- Measures:
  1. Both non-causal fused kernels return the same values as before on every input where the old code returned a
     finite value. On rows whose first scores are −inf they return the correct finite result, never NaN.
  2. `apa_quant_attention`'s selection statistics match the fp64 definition, within the registered 2e-5 relative
     band, across the whole finite fp32 range of |bulk| (1e-38 to 3e38), including the review's N2 witnesses.
  3. Every red reviewer probe row is classified as a product defect (pre-existing or new) or a probe/harness defect,
     with evidence from running the same probe against the before engines. Product defects inside the writable
     scope are fixed.
  4. The MF2.1 receipt counts are corrected.
- Mechanism doc: `tensor_cuda/src/kernels.cu` (the online-softmax recurrence comments at the two kernels). Also
  `docs/APA_MF2_1_REVIEW.md` on branch `apa-mf21rev`. The FlashAttention-2 guarded recurrence is the prior art.
- Exercises:
  - `tensor_cuda/src/kernels.cu`: the two non-causal fused recurrences only;
  - `tensor_cuda/tensor_cuda/quant.py`: the selection statistics block only;
  - the re-review probes, copied from `.../tensor_fork_apa_wt_mf21rev/tests/apa_mf21rev/gpu_probe.py` and
    `scripts/apa_mf21rev_gpu.sh`;
  - the before engines: MF2 `artifacts/apa_mf2/build` (048949d) and MF2.1 `artifacts/apa_mf2_1/build`, both in this
    worktree. For production `fd32b6d`, build it from `artifacts/apa_mf2/before/prod_fd32b6d` or from the reviewer's
    baselines, and record its sha.
- Does not cover:
  - the BK4/'a' kernel statistics (deviation reported; David decides);
  - the quantizer's |k|² overflow above about 1e19;
  - the extension build (needs CUDA 13);
  - `tensor_gpu_v2`;
  - the fully-blocked-row NaN policy (unchanged).
- Required evidence:
  - N1 before/after on the reviewer's witnesses, plus a sweep of −inf-first rows: fp32/bf16/fp16, int4-partial,
    and S ∈ {2, 5, 33, 1025};
  - byte-identity of both kernels against MF2 (048949d) and against fd32b6d on finite non-overflow inputs, from
    separate engine processes;
  - an N2 range-sweep table (|bulk| scales 1e-38…3e38, S ∈ {1, 2, 3, 256, 2048}, causal and non-causal);
  - real-capture definition agreement unchanged at 0 out of band;
  - the adjudication table: every red row with its cell, row name, MF2.1 outcome, MF2 outcome, fd32b6d outcome,
    class and fix;
  - the N5 erratum;
  - mutation.
- Operand census:
  - real captures (Qwen3-1.7B and 4B, S=8192, bf16);
  - synthetic L ∈ {1, 3, 64, 513}, S ∈ {L, L+1, 4L+5}, D ∈ {64, 96, 128};
  - the reviewer's witnesses;
  - finite fp32 magnitudes across the full range;
  - fp16 and int4-partial paths.
- Baseline isolation:
  - "before" arms run in separate processes against separately built engines (MF2 `8ca4de6c…`, MF2.1 `b5837bfe…`,
    production as built and pinned);
  - the fp64 oracle is your own, from the definition.
- Forbidden configuration:
  - reverting the −inf start instead of guarding the recurrence (the start must stay −inf; F1 must remain closed);
  - widening the tie band;
  - calling a red row a "probe defect" without showing the same row on a before engine;
  - the retired 595.84 driver override.

Seat: Claude Opus 5.5, high (an Agent-tool subagent; it has the GPU). A Codex re-review follows.

## Where to work

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_mf2`
(branch `apa-mf2`). Writable only:
- `tensor_cuda/src/kernels.cu` (the two non-causal fused online-softmax recurrences);
- `tensor_cuda/tensor_cuda/quant.py` (the selection statistics);
- new `tensor_cuda/tests/test_apa_mf2_2_*.py`;
- `scripts/apa_mf2_2_*`, `artifacts/apa_mf2_2/`, `docs/APA_MF2_2_LEDGER.md` and `logs/`.

Rebuild the engine into `artifacts/apa_mf2_2/build/`. If an adjudicated product defect sits outside these files,
REPORT it and do not fix it.

**READ-ONLY:** the production repo `/mnt/ForgeRealm/Project-Tensor` (outside `Fork/`), GraftRepository,
GRAPA-Native-LLM, every other worktree, and the HF cache.

NO git. NO subagents. NO DELETES (`delete-me … --session apa-mf2-2`). Never kill a process you did not start. GPU:
`flock -w 900 /tmp/forge-gpu.lock timeout 600 …`, native driver only. NEVER set the 595.84 `LD_LIBRARY_PATH`/PATH
override.

## Required

1. Tests first, with failing receipts for N1 and N2.
2. **N1:** guard both recurrences as FlashAttention-2 does. When `m_new == -INFINITY`, use a safe reference (for
   example 0) so `corr` and `p` evaluate to 0 rather than NaN. Keep the −inf start. Prove byte-identity on finite
   non-overflow inputs against MF2 and fd32b6d.
3. **N2:** range-robust statistics. Scale the visible |bulk| by its row maximum before the moments, or use an
   equivalent overflow/underflow-safe form, and compare in the scaled domain or rescale exactly. Show the full-range
   sweep, and show that real-capture agreement stays at 0 out of band.
4. **Adjudication:** run every red reviewer row against the MF2 and fd32b6d engines and classify it.
   - bf16 `.grad is None`: is it a pre-existing engine limitation for bf16 leaves, or new?
   - The 2048 tolerance: is the error relative 1e-6, i.e. a probe tolerance issue?
   - The identity harness exit: find the cause.
   - Fix any new product regression that is inside scope.
5. **N5 erratum:** recount the MF2.1 identity records and write the correct numbers.
6. **Mutation ≥ 0.80**, including: an unguarded recurrence, unscaled statistics, and a reverted −inf start (must
   re-open F1).

## Gates — registered BEFORE any GPU gate

`artifacts/apa_mf2_2/registration.json` and its `.sha256`, create-only.
- **G1:** the full `tensor_cuda/tests` suite (`-q -p no:cacheprovider --ignore=tensor_cuda/tests/test_selector_accuracy.py`):
  the FAILED/ERROR set equals the fd32b6d baseline's 52 ids, with no new failures.
- **G2 (N1):** the witnesses and the sweep are finite and correct, and byte-identical on finite non-overflow inputs
  (against MF2 and fd32b6d).
- **G3 (N2):** 0 out-of-band mismatches against the fp64 definition across the full-range sweep and on the real
  captures.
- **G4:** every red row adjudicated, with before-engine evidence; no unexplained red.
- **Mutation:** ≥ 0.80.
- **Sanity:** claims are stated within the measured domain only; no universal prose beyond what was measured.

## Rules

- HOUSE_RULES apply. Registration before gates. RED honesty.
- Prior Art Directive:
  - FlashAttention-2's guarded online softmax (Dao, 2023);
  - Milakov & Gimelshein, online normalizer (2018);
  - scaled norms / hypot-style rescaling (BLAS `nrm2`, Blue 1978);
  - Chan, Golub & LeVeque (1983).

  Mark anything unverifiable "unverified — lead to check".
- Keep to about 2 hours.

## Done

Write `artifacts/apa_mf2_2/verdict.json` and `docs/APA_MF2_2_LEDGER.md` (a report file is refused for subagents).
Then give the final message:
- the G1–G4 and mutation verdicts, with the registration sha;
- the N1 and N2 tables;
- the adjudication table;
- the erratum;
- the files changed, with the diff stat and one patch per item (`artifacts/apa_mf2_2/item{1,2}.patch`);
- deviations and open items;
- receipts cited as `path:line "quote"`.
