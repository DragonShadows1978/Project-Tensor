# APA-MF2 — close the three items the MF1 review left open: an exact kernel rule for fix B, the finite-bias causal leak, and the apa_cuda extension mask

David, 2026-10-07, on the three open MF1 items: "Fix it dude, fix it."

Base: branch `apa-mf2` = `apa-mf1` at 42e19e6. That branch is production `fd32b6d`, plus fix A (b716e47), plus fix B
(66f0474), plus receipts. The MF1 blind review is `apa-mf1rev` 3bee9fa: `docs/APA_MF1_REVIEW.md`,
`artifacts/apa_mf1rev/verdict.json` and `lead_gpu_results.md`. Read all three first. Its probes are in
`.../WorkTree/tensor_fork_apa_wt_mf1rev/tests/apa_mf1rev/`, read-only; copy them.

## The three items

1. **Fix B is not exactly the fused kernel's rule** (review R3: "faithful_to_kernel_rule: no"):
   - **B1:** `quant.py` computes the statistics on `bulk` after `tc.matmul` has stored it in bf16
     (`matmul.cu:76 "cp, CUDA_R_16BF"`). The BK4 kernel thresholds its own unrounded fp32 dot product. Agreement with
     the kernel's diagnostic mask is 0.99867–0.99947 on real captures.
   - **B2:** the non-causal branch uses `mean`/`std`: centered, population, at `ops.cpp:918`. The kernel uses one-pass
     moments `squares/count − mean²`.
   - **B3:** the selection lacks the kernel's `visible &&` predicate (`kernels.cu:8077`). At a zero threshold, it
     selects invisible entries.
   - **The test reference:** `tensor_cuda/tests/test_apa_phase6.py::_ref` encodes the OLD diluted rule and fails under
     fix B. It must be updated to the kernel rule.
2. **N1, the finite causal bias leaks:**
   - `functional.py:38` builds the causal bias as `np.full((L, S), -1e4)`, added to the scores. A large enough score
     overcomes it and a future key leaks.
   - The reviewer's case: Q=e0 on all rows, K7=80008·e0, other K=0, V7=1, other V=0, fp32, L3 S9 D64, refine 1,
     scale 1/8. Row 0 must output 0 and does not. The lead re-ran it on the GPU: still RED.
   - It predates MF1. The comment chose −1e4 "so it stays representable in fp16/bf16 without becoming −inf".
3. **M1, `apa_cuda/csrc/apa.cpp:186–191`** builds `auto cmask = (ci > ri)`, which is TOP-LEFT. Rectangular causal
   calls through the `apa_attention_cuda` extension are blind to the prefix. The extension is not built on this
   machine.

## Mechanism & scope

- Measures:
  - **Item 1:** `apa_quant_attention`'s causal and non-causal refine selection is bit-identical to the fused BK4
    kernel's own diagnostic mask on identical operands (fp32), and identical within a counted tie band for bf16/fp16.
  - **Item 2:** masked keys get exactly zero probability for every finite score, while every realistic output stays
    byte-identical.
  - **Item 3:** the extension's causal mask is bottom-right, and verified by running it.
- Mechanism doc: `tensor_cuda/tensor_cuda/functional.py` (the `_causal_mask` comment, lines 26–37: the convention
  and its failure mode). Also `docs/APA_MF1_REVIEW.md` on branch `apa-mf1rev`, and `kernels.cu` BK4 lines 8040–8155
  (the rule being matched).
- Exercises:
  - `tensor_cuda/tensor_cuda/quant.py::apa_quant_attention`;
  - `tensor_cuda/tensor_cuda/functional.py::_causal_mask` and its consumers:
    - `functional.py:88` SDPA;
    - `quant.py:155`;
    - GRM `core/gemma4_tc.py:740` (read-only consumer);
    - GRAPA `grapa/attention.py` (read-only consumer);
  - `apa_cuda/csrc/apa.cpp` and the `apa_attention_cuda` extension build;
  - `tensor_cuda/tests/test_apa_phase6.py` (reference function only);
  - real captures at `.../WorkTree/tensor_fork_apa_wt_spq3/artifacts/apa_spq3/captures/` (Qwen3-1.7B/4B, S=8192).
- Does not cover:
  - the CUDA kernels in `tensor_cuda/src` (already predicate-masked; do NOT edit them; if item 2 turns out to need a
    kernel change, STOP and report);
  - `tensor_gpu_v2`'s and `apa_cuda`'s legacy |ranking| selection rule (a different, older rule; list it, do not
    change it);
  - the SP, packed and K1 entries;
  - any promotion (David).
- Required evidence:
  - **Item 1:** per-cell selection agreement with `bp_kernel_4_diagnostic` variants 'a' and 'h' on real captures
    (2 models × 3 layers × L ∈ {64, 512, 2048} × bulk {4, 8}), before (MF1 A+B) and after. Also the
    synthetic B1/B2/B3 cases from the review, before and after.
  - **Item 2:**
    - the leak case and a sweep of adversarial magnitudes, before and after;
    - byte-identity, before vs after, of `F.scaled_dot_product_attention` and `apa_quant_attention` on real captures
      (fp32/bf16/fp16, causal, S ≥ L);
    - byte-identity of GRM Qwen3-1.7B standard-mode logits on one wikitext-2 window (chunked prefill, 32 steps);
    - byte-identity of a GRAPA attention forward on small real-shaped inputs (import read-only);
    - no NaN or Inf in any valid S ≥ L case.
  - **Item 3:** the extension build log; the reviewer's `red-extension` probe and the `apa_cuda` tests, before and
    after.
- Operand census:
  - the real captures: Qwen3-1.7B (28 layers × 16 query / 8 KV heads × D128) and Qwen3-4B (36 × 32/8 × D128),
    S=8192, bf16;
  - synthetic L ∈ {1, 3, 64, 513}, S ∈ {L, L+1, 4L+5}, D ∈ {64, 96, 128};
  - realistic score magnitudes are well under 1e3. Adversarial ones go up to 1e30 in fp32 and to the dtype maximum
    in bf16/fp16.
- Baseline isolation:
  - "before" is the MF1 A+B tree at 42e19e6, copied read-only to `artifacts/apa_mf2/before/`, plus production
    `fd32b6d` for the leak case;
  - the oracle is your own fp64 numpy written from the definition, not MF1's or the reviewer's helpers;
  - kernel diagnostic masks come from the engine's own entry on identical operands.
- Forbidden configuration:
  - matching the kernel by changing the KERNEL;
  - a "byte-identity" check that runs both arms on the new code;
  - tie-band tolerance on fp32 item-1 cells (they must be exact);
  - a mask that turns S ≥ L rows into NaN;
  - the retired 595.84 driver override (it now breaks CUDA).

Seat: Claude Opus 5.5, high (an Agent-tool subagent; it has the GPU). A blind Codex review follows.

## Where to work

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_mf2`
(branch `apa-mf2`). Edits, builds and GPU runs are AUTHORIZED there, and only in:
- `tensor_cuda/tensor_cuda/quant.py` (the `apa_quant_attention` selection block);
- `tensor_cuda/tensor_cuda/functional.py` (`_causal_mask` and its direct use in SDPA only);
- `apa_cuda/csrc/apa.cpp` (causal mask only);
- `tensor_cuda/tests/test_apa_phase6.py` (the `_ref` reference only);
- new `tensor_cuda/tests/test_apa_mf2_*.py` and `apa_cuda/tests/test_apa_mf2_*.py`;
- `scripts/apa_mf2_*`, `artifacts/apa_mf2/`, `docs/APA_MF2_LEDGER.md` and `logs/`;
- the extension build output inside `apa_cuda/` (build/, the in-place `.so`).

Build the engine into `artifacts/apa_mf2/build/` (copy `scripts/apa_mf1_build.sh`; the C++ is unchanged, so record
whether the `.so` sha256 matches MF1's `8a13e5b3…cc437`).

**READ-ONLY, FORBIDDEN TO WRITE:**
- the production repo `/mnt/ForgeRealm/Project-Tensor` (everything outside `Fork/`);
- `/mnt/ForgeRealm/GraftRepository`;
- `/mnt/ForgeRealm/GRAPA-Native-LLM`;
- every other worktree;
- the HF cache.

NO git (the lead commits). NO subagents. NO DELETES (`delete-me <path> --reason "..." --session apa-mf2`). Never kill
a process you did not start. Run gates synchronously.

**GPU:** `flock -w 900 /tmp/forge-gpu.lock timeout 600 …`, with the NATIVE driver (595.91.07). Do NOT set
`LD_LIBRARY_PATH` or PATH to `/mnt/ForgeRealm/engines/nvidia-userspace-595.84`; since the 10-07 reboot it breaks
CUDA.

## Required

1. **Tests first.** Write tests encoding the exact BK4 rule (selection bit-identity against the kernel's diagnostic
   mask on fp32 operands), the leak case, and the extension's rectangular reference. Run them on the current tree and
   save the failing receipt.
2. **Item 1.** Make `apa_quant_attention`'s selection the BK4 rule exactly:
   - statistics and threshold comparison on the unrounded fp32 bulk dot product;
   - population one-pass moments with the kernel's clamp;
   - the visible predicate;
   - `>=`.

   Decide from `kernels.cu` which bulk values the kernel uses in the softmax for unrefined keys, and match that too,
   or report the difference. Update `test_apa_phase6._ref` to the kernel rule and nothing else. If an exact match is
   impossible without a kernel change, stop and report the residual.
3. **Item 2.** Make masked keys contribute exactly zero for every finite score in every dtype, keeping the bottom-right
   convention.
   - Use a predicate-based mask (where/masked_fill) or an infinite bias; check that the engine softmax handles `-inf`
     with no NaN for S ≥ L rows.
   - Prove byte-identity on realistic inputs. With the old −1e4 bias, exp already underflows to exactly 0 in fp32 for
     ordinary scores.
   - Audit every consumer of `_causal_mask` and of any other finite-bias causal mask in the writable files.
   - GRM's `gemma4_tc.py:740` and GRAPA call `_causal_mask` directly. Check their usage stays correct, and report
     whether either relies on the bias being finite.
4. **Item 3.** Fix `apa.cpp` to bottom-right, with the same convention. Build the extension in place
   (`apa_cuda/setup.py`, torch C++ extension). Run the reviewer's `red-extension` probe and `apa_cuda/tests` before and
   after. If it cannot be built, record the exact error and verify by source reasoning only, labeled as such.
5. **Real data and GRM.**
   - The item-1 table on the captures.
   - The GRM Qwen3-1.7B `"apa"` chunked-prefill KL against standard (as MF1 item 6), for MF1 A+B vs MF2.
   - The GRM standard-mode byte-identity (item 2).
6. **Mutation.** At least 6 registered mutants, killed ≥ 0.80:
   - bf16 bulk for the statistics restored;
   - the visible predicate dropped;
   - sample variance;
   - the bias back to −1e4;
   - apa.cpp back to top-left;
   - the kernel-rule count off by one.

## Gates — registered BEFORE any GPU gate

Write `artifacts/apa_mf2/registration.json` and its `.sha256` create-only.
- **G1:** the full `tensor_cuda/tests` suite (`-q -p no:cacheprovider --ignore=tensor_cuda/tests/test_selector_accuracy.py`).
  - The FAILED/ERROR id set must equal production `fd32b6d`'s baseline. The 52 ids are listed in
    `.../tensor_fork_apa_wt_mf1/logs/apa_mf1_g1_baseline_unchanged.log`.
  - `test_apa_phase6::test_apa_causal` must PASS again with the updated reference.
  - The `apa_cuda` tests equal or improve on before.
- **G2 (item 1):**
  - fp32 selection agreement with the BK4 diagnostic mask is exactly 1.0 in every real and synthetic cell;
  - bf16/fp16 differ only within a counted tie band of |bulk|−thr ≤ 2e-5·max(1, |thr|);
  - the review's B1, B2 and B3 cases flip from RED to GREEN.
- **G3 (item 2):**
  - the leak case and the adversarial sweep give exactly zero weight on masked keys;
  - every realistic arm is byte-identical before vs after (real captures in 3 dtypes, GRM standard logits, GRAPA
    forward);
  - no NaN or Inf for S ≥ L.
- **G4 (item 3):** the extension builds and the rectangular probe passes; or the build failure is recorded verbatim
  and the item is labeled source-verified only.
- **Mutation:** ≥ 0.80 killed.
- **Sanity/adjudication:** every anomaly is explained in one line.

Lead predictions:
- P1: fp32 item-1 agreement goes from about 0.9987–0.9995 to exactly 1.0;
- P2: the item-2 change is byte-identical on every realistic input, because exp(−1e4 + s) is already 0.0 in fp32
  for |s| < 1e4−104;
- P3: GRM `"apa"` chunked KL against standard moves by less than 5% relative to MF1 A+B, because the rule change is
  numerically tiny.

## Rules

- HOUSE_RULES apply. Registration before gates. RED honesty.
- Prior Art Directive:
  - `kernels.cu` BK4 (Project-Tensor, 2026);
  - 63960a4;
  - FlashAttention-2 (Dao, 2023) for predicate masking with −inf;
  - PyTorch SDPA `attn_mask` semantics for boolean masks;
  - mutation testing (DeMillo et al., 1978).
- Keep to about 2.5 hours.

## Done

You CANNOT write `docs/APA_MF2_REPORT.md` (the harness refuses report files from subagents). Write
`artifacts/apa_mf2/verdict.json` and `docs/APA_MF2_LEDGER.md`. Then give the final message, which is the report:
- G1–G4 and mutation verdicts, with the registration SHA-256;
- the before/after tables for items 1–3;
- the GRM and GRAPA results;
- predictions held or missed;
- the files changed, with the diff stat, and one patch per item (`artifacts/apa_mf2/item{1,2,3}.patch`), so the lead
  can commit them separately;
- deviations and open items;
- receipts cited as `path:line "quote"`.
