# APA-MF1 Implementation Ledger: rectangular causal masks

Order: `orders/APA_MF1_RECTANGULAR_CAUSAL_MASKS.md` (immutable). Seat: Claude Opus 5.5 (Agent-tool subagent).
Worktree: `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_mf1` (branch apa-mf1,
from fd32b6d). No git was run. No subagents were spawned. Nothing was deleted: one scratch directory was moved with delete-me (entry 14).
Every GPU command ran as `flock -w 900 /tmp/forge-gpu.lock timeout 600 ...` with the NVML exports
(`scripts/apa_mf1_env.sh`). Every gate ran synchronously.

## Entries (chronological, 2026-10-07 EDT)

1. **11:22 Read.** Read the order, HOUSE_RULES and AGENTS.md. Mechanism doc read: `functional.py:20-43` (the
   `_causal_mask` bottom-right convention, k = 1 + S − L). The order measures exactly that convention at the
   listed sites, so this is not a scope mismatch.
2. **11:25 Before copies.** The fd32b6d files (quant.py, _core.py, test_parity.py, functional.py) were copied from the
   production checkout into `artifacts/apa_mf1/before/` (read-only, `SHA256SUMS`). They are byte-equal to this worktree's
   files at fork time. The C++/CUDA sources are byte-equal to production (`cmp`, no diff).
3. **11:27 Build.** `scripts/apa_mf1_build.sh`, a copy of the apa_sp2/sp3 pattern, built in 1:00
   (`logs/apa_mf1_build.log:15 "Build complete in artifacts/apa_mf1/build"`). The module sha256 is
   `8a13e5b33a86f9d8f70da26d9730ad27acdaf7746166fafe36ac99d3b97cc437` (`logs/apa_mf1_build.log:14`). Deltas from
   the pattern: the ROOT guard is this worktree, the .cu objects compile in parallel, and `-Xptxas=-v` (verbose print only) is
   dropped. MF1 edits no C++, so before, A and A+B all use this one .so.
4. **11:28 G1 lead baseline, unchanged tree** (deviation: this ran BEFORE the registration; it is the baseline input,
   not a gate verdict, and the registration pins its sha256).
   `logs/apa_mf1_g1_baseline_unchanged.log:1835 "37 failed, 719 passed, 18 skipped, 2 warnings, 15 errors"`. The
   `--ignore=tensor_cuda/tests/test_selector_accuracy.py` is required: that file is a CLI script that reads
   `sys.argv[1]` at import, which is a collection error.
5. **11:36 Registration.** `artifacts/apa_mf1/registration.json`, written create-only and read-only, sha256
   `8129cfbadc7daa7c3c31885720196b1cbfd8ddf5e7d6052cca43e29fca6cefec`
   (`artifacts/apa_mf1/registration.json.sha256:1`).
6. **11:37 Tests first.** New files `tensor_cuda/tests/test_apa_mf1_quant.py`, `apa_cuda/tests/test_apa_mf1_tgv2.py`,
   `apa_cuda/tests/test_apa_mf1_parity_ref.py` and the fp64 reference `scripts/apa_mf1_ref.py`, written from the definition
   (`visible_mask`: col ≤ S−L+row; no triu(k=1) on a rectangle).
7. **11:39 Before receipt** (unchanged tree, rule A): `logs/apa_mf1_before.log:6609 "108 failed, 70 passed"`. Every
   rectangular shape fails at least one test. The quant reference cases L=1, S=2 pass before: with 2 keys and z=1.036,
   nothing is selectable under either alignment. The eligibility test fails that shape (excluded_visible=2).
   Rule B on the unchanged tree: `logs/apa_mf1_before_ruleB.log:2988 "54 failed, 11 passed, 12 skipped"`.
   - *Print-only fix after this receipt:* in `check()`, `eligible_frac` did not broadcast the (L,S) stats mask over H.
     It was under-reported by a factor of H=2 in the reference-test print lines. No assertion changed. The eligibility
     test's own numbers were correct throughout.
8. **11:42 Fix A.** quant.py:181 now derives the stats mask from the shared `F._causal_mask` bias
   (`masked_fill(causal_add.float(), 0.0)`; masked_fill tests cond ≠ 0). tensor_gpu_v2: one helper
   `_causal_mask_bool` (`_core.py:1606`) is called at all 9 causal-mask constructions: the order's 3 lines plus 6 more
   found by the audit (flash fwd and bwd, APA refine ≥ 1 fwd and bwd, APA tiled fwd and bwd). test_parity.py:110 and :155 use
   `k=1 + (S - L)`. Snapshots are in `artifacts/apa_mf1/variants/{quant_A,tgv2_core_A,test_parity_A}.py`.
9. **11:43 A receipt** (rule A): `logs/apa_mf1_after_A.log:156 "178 passed in 56.88s"`. This includes all 12 quant and
   17 tensor_gpu_v2 square byte-identity tests.
   B-failing-before receipt (rule B on state A): `logs/apa_mf1_A_ruleB_before_fixB.log:2373 "36 failed, 29 passed, 12 skipped"`.
10. **11:44 Fix B, first attempt INVALID.** The AfterImage PreToolUse hook blocked the Edit. The Bash call issued in the same
    block ran anyway, against state A. Its receipts were renamed, not deleted:
    `logs/apa_mf1_after_AB.INVALID_editblocked_stateA.log`, `receipts/after_AB_ruleB.INVALID_editblocked_stateA.xml`,
    `fixB.INVALID_empty.patch`, `variants/quant_AB.INVALID_is_stateA.py`.
11. **11:45 Fix B applied.** quant.py:183-197 computes the causal threshold from visible-count population mean/std, as
    one-pass E[x²]−E[x]² clamped at 0. That is the fused-kernel form (`kernels.cu:8113`). The count is clamped ≥ 1 for
    out-of-contract S < L. `artifacts/apa_mf1/fixB.patch` (+15/−1) holds the B diff. The A+B receipt:
    `logs/apa_mf1_after_AB.log:156 "166 passed, 12 skipped"` (the square identity tests skip under rule B by design).
12. **11:46 G1, A+B tree.** `logs/apa_mf1_g1_after_AB.log:1848 "38 failed, 783 passed, 30 skipped, 2 warnings, 15 errors"`.
    Per-id comparison (`scripts/apa_mf1_g1_compare.py`): exactly one baseline id changed.
    `tensor_cuda.tests.test_apa_phase6::test_apa_causal` went passed → failure
    (`logs/apa_mf1_g1_after_AB.log:1783 "FAILED tensor_cuda/tests/test_apa_phase6.py::test_apa_causal"`). Cause: its `_ref`
    (`test_apa_phase6.py:47-50`) encodes the diluted stats, which fix B removes by design. The file is not in the writable list.
    **G1, A-only tree** (quant.py temporarily swapped to the A snapshot, restored and sha-verified):
    `logs/apa_mf1_g1_after_A.log:1821 "37 failed, 796 passed, 18 skipped, 2 warnings, 15 errors"`. 0 changed ids; 77 new MF1 ids passed.
    **apa_cuda** test_parity.py, before copy and after: `logs/apa_mf1_g1_apa_cuda.log:164 "6 failed"` and `:330 "6 failed"`,
    all `ModuleNotFoundError: No module named 'apa_attention_cuda'` (the extension is not built here). Same outcome both times.
13. **11:52-12:04 Real captures.** `scripts/apa_mf1_real.py` covered 6 (model, layer) runs × 3 L × 2 bits × 3 arms, plus the kernel diagnostic.
    Receipts are in `artifacts/apa_mf1/real/*.json` and `logs/apa_mf1_real_*.log`.
14. **12:05-12:10 GRM.** `scripts/apa_mf1_grm.py` imported GRM read-only (`find -newer` marker: no GRM file was written). Chunked
    receipt `logs/apa_mf1_grm_chunked.log:13`, control receipt `logs/apa_mf1_grm_control.log:13`. Item 6 took 5.5 min of its 45 min box.
15. **12:11-12:13 Mutation.** 13/13 mutants killed. Every restore was sha-verified, and the tree hashes after the run equal those
    before it (`logs/apa_mf1_mutation.log`, `artifacts/apa_mf1/mutation/M*.json`).
16. **12:14 onward: summary, audit, patches.** `scripts/apa_mf1_summary.py` writes `artifacts/apa_mf1/{tables.md,summary.json}`.
    `fixA.patch` and `fixB.patch` were verified: applied in order to the before copies, they reproduce all 3 tree files byte-for-byte.
    The scratch `artifacts/apa_mf1/fixB_tmp` was moved with `delete-me` (id `20261007-121732_apa-mf1_fixB_tmp`).

17. **12:20 Report file refused.** A Write of the report draft was refused by the harness: "Subagents should return
    findings as text, not write report files. Include this content in your final response instead." That guard was not
    bypassed. `docs/APA_MF1_REPORT.md` is therefore NOT written. The full report is the seat's final message.
    `artifacts/apa_mf1/verdict.json` (machine-readable) was written; its ranges were checked against `summary.json`.

## Decisions

- quant.py fix A derives the boolean from `causal_add` (= `F._causal_mask`). It does not add another inline triu, which follows the order's
  "one shared helper" preference. functional.py is not edited.
- tensor_gpu_v2 gets one helper at all 9 sites. Inside the writable file, the order's "every causal branch" clause covers the 6 sites it did
  not list.
- S < L does not raise. It stays finite (fix B clamps count ≥ 1); see the audit.
- Fix B is limited to `tensor_cuda.quant.apa_quant_attention`. tensor_gpu_v2's own APA rule (|ranking|, diluted) and apa_cuda's
  `compute_threshold` are listed as divergences only.

## Prior art (also at each code site)

- Bottom-right causal alignment for rectangular blocks: FlashAttention-2 (Tri Dao, 2023) bottom-right `causal` when
  seqlen_q ≠ seqlen_k. Taken: the convention. Project-Tensor 63960a4 (2026-06-09), `F._causal_mask` k = 1 + S − L. Taken: the
  exact formula and the reuse of that bias as the stats mask.
- Visible-count population statistics: fused kernel `kernels.cu:8113` (BK4 forward; `count = S-L+qi+tid+1`, one-pass,
  clamp at 0) and GRAPA `attention.py:94-105` (`_visible_counts`, `cnt = arange(S-L+1, S+1)`, one-pass + relu). Taken: the
  rule and the one-pass form. Mine: the clamp count ≥ 1 for S < L, and placement in quant.py. GRAPA was not imported.
- The fp64 two-pass reference and the tie-band adjudication: no prior art known to me.
- Mutation testing: DeMillo, Lipton & Sayward (1978); HOUSE_RULES §8. Taken: the method.
- System-Claude's M4a "2^E tag" leak detector: optional, NOT used.
- Teacher-forced next-token KL vs a reference model: common LM-eval practice. Unverified for a first source; lead to check
  (search terms: "KL divergence teacher forcing quantized LLM evaluation").

## Registered successors (found, out of scope)

- `apa_cuda/csrc/apa.cpp:186-191` is top-left (`cmask = (ci > ri)`). The apa_cuda extension needs the same fix. It is not buildable here.
- `tensor_cuda/tests/test_apa_phase6.py:36,47-50`: a square-only top-left reference with diluted stats. It needs a B-consistent
  reference if B is promoted.
- `tensor_cuda/tests/test_attention_phase5.py:19`: a square-only top-left reference.
- tensor_gpu_v2 APA and apa_cuda `compute_threshold` stats are diluted (stats over S); their selection is on |ranking|.
