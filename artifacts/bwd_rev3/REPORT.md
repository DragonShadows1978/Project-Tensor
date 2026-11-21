# BWD-REV3: blind cross-model review of BWD-R3 (Claude Opus 5.5; author Codex)

Filed by the lead from the reviewer's final message (the seat's write was refused). Text verbatim.

The review found no BLOCKER and no MAJOR issues in BWD-R3: 3 MINOR and 6 NIT. My write of `REPORT.md` was refused by the tool, so this message is its full content for the lead to file. `findings.json` (v1 and final) and every check were written.

**What the evidence is.** The candidate's native code needs CUDA, so I did not run it. I transcribed its index arithmetic line by line into NumPy fp64 and compared it with two references at 73 boundary geometries:
- the tests' copy of Voice's fp64 VJP;
- my own dot-product test against a brute-force ONNX forward.

All 73 pass: max error 1.8e-13 against Voice's VJP and 9.9e-14 on the dot-product test. Every result below that says "emulation" comes from that transcription. After sealing v1 I also read the round's GPU receipts and reran the R3 tests CPU-only on the round's built packages; their module sha256s match G1.

**Seals.**
- **v1** (`findings_v1.json`), written before any BWD-R3 receipt was read:
  - sha256 `dcc3fdb0cc20fa76134a3ccc8bae061d64ed37d2d7f9a60f6813a76413a759f9`
  - seal line (same command): `date 2026-11-19T05:28:34Z Thu Nov 19 12:28:34 AM EST 2026`
  - mtime `2026-11-19 00:28:34.9025 -0500`
- **final** (`findings.json`):
  - sha256 `4848a6121a2a532c2480960b91aacdf9ab1fcd90404f48129c3bd4ad9605f27a`
  - seal line: `date 2026-11-19T05:33:30Z`
- **v1 → final:** only F04 changed. v1 read the 123 ms as the steady per-call cost; the receipts show it is a one-time outlier. The performance point moved to a new NIT, F09.

## Findings

**BWDR3R-F01 — MINOR.** No oracle comparison covers the GEMM dW lane with N > 1 and groups > 1.
- **Where:** `tc_signal/src/conv_backward.cu:274-275` (the code itself is correct); `tests/test_bwd_r3_paths.py:90-96`; no case in `test_bwd_r3_numeric.py`.
- **Why it matters:** the per-group offsets use `(n * g.groups + group)`. Every oracle-compared geometry on the GEMM lane has either N = 1 or groups = 1. The only N = 2 GEMM synthetic cases (active-slots, dlpack) have groups = 1 and only compare the candidate with itself.
- **Proof (emulation):** mutant M1, `(n + group)`, changes dW by 84–113% on N = 2, groups = 2 shapes (`nonequiv.log`). It is detected by 0 of the 58 oracle-test geometries (`reach.log`).
- **Fix:** add an oracle-compared case per op with N = 2, groups = 2 and more than 64 coefficients per group (for example 16 channels, K = 3), with length > 1024.

**BWDR3R-F02 — MINOR.** The fixed-tree dW lane is never run past its first time tile.
- **Where:** `conv_backward.cu:202-203, 268-271`; no test reaches it.
- **Why it matters:** every tree-lane geometry in the tests is single-tile. Production reaches this lane: the 1,090-group ×2 transposed conv and the F0/N conv both go past 1,024 frames on utterances of roughly 25 s and up.
- **Proof (emulation):** mutant M2, `time = j` instead of `begin + j`, changes dW by 139–170% and is detected by 0 of 58.
- **Fix:** add oracle-compared depthwise cases for both ops (×2 with output_padding 1), with tiled length > 1024 and N = 2.

**BWDR3R-F03 — NIT.** The merge-order witness (`test_bwd_r3_numeric.py:110-124`) covers only the GEMM lane, N = 1, three full tiles.
- Nothing pins the tree lane's cross-tile order, the cross-batch order, or a partial last tile. Random-data checks cannot see an order change, because it only moves rounding.
- **Fix:** add a K = 1 depthwise variant, an N = 2 variant with the cancelling terms in different batches, and a partial last tile.

**BWDR3R-F04 — MINOR.** The 123 ms conv_transpose backward is a one-time outlier, about 8× the warm cost, and every runtime row is single-shot with no warm-up.
- **Where:** `test_bwd_r3_runtime.py:206-273`; the round's log `bwd_r3_gpu_memory-convT_…4172125.log`.
- **The numbers, all in the same process:**

  | Run of conv_transpose1d_kokoro | Position in the line | Native time |
  |---|---|---|
  | xwb | first cell | 122.58 ms |
  | x only | later | 14.48 ms |
  | w only | later | 0.34 ms |

- **Why it is not the per-call cost:** in all 7 other cases xwb is within 0.1 ms of x + w. So about 108 ms is paid once, by the first cell.
- **Possible cause (unverified, no GPU):** the convT forward makes no cuBLAS call, so the timed backward GEMM is the process's first cuBLAS GEMM. The conv1d line's forward already uses cuBLAS, and its first cell shows no outlier.
- **Fix:** rerun with one warm-up call and 5 timed repeats, or put xwb after x and w, and report the first call separately. If the cause is confirmed, a tiny warm GEMM in `StrictGemm::initialize` would move the cost to handle creation.

**BWDR3R-F05 — NIT.** The lane-selection constants live in the kernel file, and the rule is undocumented.
- **Where:** `conv_backward.cu:19-22, 237`; README.
- block 256, tile 1024, the 32 MiB caps and `size <= 64` are in `conv_backward.cu`, not `tuning.h`.
- README never says when the tree replaces the GEMM. The two lanes give different bits for the same math.
- **Fix:** move the constants to `tuning.h` with their reasons, and document the rule.

**BWDR3R-F06 — NIT.** The F10 refusal treats `out` as an operand in three ops but not in lstm.
- **Where:** `_norms.py:107,120`; `_small.py:135`; `_lstm.py:110`.
- layer_norm, group_norm and rint pass `out`, so a requires-grad `out` with frozen inputs raises "no backward yet", which names the wrong problem.
- **Fix:** pick one rule, with its own message, apply it to every op, and test it.

**BWDR3R-F07 — NIT.** The workspace check compares a member with itself.
- **Where:** `conv_backward.cu:46-48, 81-86, 288-293`.
- `verify()` checks `installed_bytes_` against `workspace_size`. A later `cublasSetStream` would silently drop the workspace, and no comment records that invariant.
- `_conv_backward_config` allocates 32 MiB as a side effect.
- The resident per-device workspace sits outside the pool and is not in README.
- **Fix:** comment the invariant and document the resident allocation.

**BWDR3R-F08 — NIT (unverified).** Bitwise dW may depend on operand alignment, not only on values.
- **Where:** `conv_backward.cu:274-278`; `tensor_cuda/include/tc/core.h:112`.
- NDArray carries an element offset, and the per-batch offsets are odd multiples of 4 bytes when T_out is odd (4,801 at the generator). If `cublasGemmEx` chooses kernels by pointer alignment, equal values at different alignments could sum in a different order.
- Reruns of one call are unaffected, so G5 holds.
- **Fix:** lead to check (search "cublasGemmEx pointer alignment kernel selection reproducibility"); then either scope the bitwise claim or pack the left operand into an aligned tile.

**BWDR3R-F09 — NIT.** The transposed dX gather reads dy at stride s across adjacent threads.
- **Where:** `conv_backward.cu:129-157`.
- **Warm rates:** conv_transpose1d_kokoro about 77 G terms/s and conv_transpose1d_03 about 99 G/s, against about 554 G/s for the conv1d gather (conv1d_23).
- Correctness is unaffected and speed is outside this order.
- **Fix, if round 4's 10 s timing needs it:** compute dense transposed dX as a strict-handle GEMM plus a fixed-order gather, which DESIGN §5 allows.

## Answers

**Q1 — dX for every geometry class.** Correct, by emulation (73/73).
- **Read:** `input_grad` (`:121-157`), the forward kernels (`conv.cu:144-201`, which use the same maps the backward inverts) and `backward_geom` (dy checked against the forward planner's resolved shape).
- **Ran:** `checks/rev3_emul.py`. It covered:
  - conv classes: K1, dense, dilation 3 and 5, stride 5 with K20, stride 6 with K12 and pads 3/3;
  - asymmetric pads (2,0), (0,3), (4,1), (0,4);
  - K > T and T_out = 1 for both ops, plus a stride-7 conv;
  - N = 2 and 3; groups 2 and 3; depthwise with 9, 10 and 16 groups;
  - transposed: ×2 depthwise with output_padding 1, dense 10/20 and 6/12, the stride-5 tail, dilation 2 and 3 with output_padding 1 and 2, and pads that cut into the output_padding bins.
- **Oracle provenance:** the copied Voice oracle matches Voice's `ops.py@24c1eea` verbatim (6/6 functions; `copycheck.log`).

**Q2 — dW and dB.** Correct, by emulation, including GEMM-lane multi-tile cases with a partial last tile, N = 2 and groups = 2, and every geometry forced onto the tree lane.
- **Pack indices:** right for both ops. GEMM operand reading (`lda=count`, `ldb=length`, `ldc=columns`) matches both weight layouts. The group partition is right.
- **Tile boundaries:** right, including 1,024 crossing a stride. Time is the GEMM's reduction axis and maps independently per position.
- **Merge order:** fixed (batch outer, tile ascending, group inner, sequential on one stream). There are no float atomics anywhere.
- **dB:** sums every position, so it includes bias-only and output_padding bins.
- **When the tree lane is chosen:** per-group coefficients ≤ 64, or more than 8,388,608 floats, or pack rows × min(length, 1024) above that, or length > INT_MAX. At the real geometries that is conv1d_01, conv_transpose1d_01 and the BP1 small2 cells (`census.txt`).

**Q3 — strict GEMM handle.** Read `conv_backward.cu:33-119`.
- It is per device, created under `DeviceGuard`, and set up in the required order: stream first, then workspace (setting the stream resets the workspace).
- The 32 MiB workspace lives for the process. A failed init cleans up the handle and the workspace.
- The call has no TF32 or atomics fallback (pedantic compute, pedantic math, atomics not allowed, default algorithm). The receipt shows cuBLAS 12.6.4 and 56 SMs. r3m07 was killed numerically at 6.365× the bound after the configuration report printed.
- The call and the check read the same `compute_` and `algo_` members; the workspace check is self-referential (F07).
- Every cuBLAS status is checked; only `cudaFree` on the init-failure path is not.
- It is safe for the single dispatch lane: the GIL is held across `handle_for`, and the VJP runs under the GIL.

**Q4 — memory.** Ran `rev3_census.py` over all 52 BP1 and BP2 conv geometries.
- Largest pack 9,011,200 B; largest partial 13,393,920 B; both under 33,554,432 B.
- Neither grows with T, and nothing else is sized by T apart from the gradients themselves.
- On an error path the gradient and scratch tensors are Python-owned and released once the stream passes the queued work; the handle and workspace stay resident by design.
- Measured peaks (recomputed): 49,670,784 and 72,762,368 B, which include the 32 MiB workspace.

**Q5 — tracked path and F10.** Read the `__init__.py`, `_norms.py`, `_small.py` and `_lstm.py` diffs and the `attach_vjp` contract.
- **Selection:** any TensorCUDA operand requiring grad, with grad mode on.
- **Refusals:** `out=` raises ValueError and foreign operands raise TypeError, both before any export.
- **Forward:** runs into fresh storage, then `mark_modified()`, then attach.
- **`b=None`:** a frozen `x.detach()` alias holds slot 2, with no allocation.
- **F10:** fires only under grad mode with a TensorCUDA operand requiring grad, and never inside a VJP callback.
- **Untracked forward:** unchanged, confirmed by the round's forward-guard lines (exit 0).
- **Host tests:** pass in my CPU run (refuses-out 6, refuses-foreign 6, planner/no_grad 2, F10 14).

**Q6 — tests and gates.** Static census: 20 functions, 151 IDs (33 host).

| Function | IDs | What it reaches | Compared with |
|---|---|---|---|
| test_gpu_bp1 / test_gpu_bp2 | 16 / 36 | full tracked path, both dW lanes | fp64 oracle |
| test_gpu_biasless | 8 | the `b=None` slot | fp64 oracle |
| test_gpu_cancel_unit_and_structural_zeros | 2 | tree lane | exact |
| test_gpu_tile_merge | 1 | GEMM lane, 3 tiles | bitwise witness |
| test_gpu_canary, test_gpu_configuration, test_host_canary_gap | 1 each | GEMM call, handle getters | numeric / config |
| fd: candidate 4, tiny 2, key 4 (key = answer key only) | 10 | candidate projections | fp64 finite differences |
| host refusal and planner tests | 28 | Python guards, real planner | — |
| test_gpu_active_slots, test_gpu_no_grad_and_dlpack | 6, 2 | slot masking, DLPack | self-consistency only |
| test_gpu_saved_revisions, test_gpu_repeat, test_gpu_memory_and_runtime | 12, 3, 24 | revisions, determinism, measurement | — |

- **Caught:**
  - a GEMM-lane tile-boundary error (M3 detected by 16 geometries);
  - a group-boundary error at N = 1;
  - an output_padding-bin error (r3m06);
  - an N > 1 error at groups = 1 (M4 detected by 6 geometries).
- **Not caught:**
  - N > 1 with groups > 1 on the GEMM lane (F01);
  - any tree-lane error past the first tile (F02);
  - tree-lane or cross-batch merge order (F03).
- **Mutants:** all 8 native mutants executed their edit (r3m09 three times) and were killed; r3m08 was killed on the host.
- **Plain run (mine):** `env -i`, candidate packages only, no fixture variables → 29 passed, 122 skipped (99 "fixture root absent", 23 "CUDA disabled for CPU lane"). This matches G9. No Voice worktree paths are used and every skip has a reason.
- **With the Voice fixture roots set:** 33 passed, 118 skipped; finite-difference worst bound ratio 1.08e-3; canary gap 6.36538904 bounds.
- **Outside R3:** on a no-device host, tc_signal prints "synchronize() at exit failed … cudaErrorNoDevice" at exit. That is existing P1-T behaviour.

**Q7 — anything else, and the 123 ms.** The 123 ms does not point to a code defect: it is the first cell of a fresh process, and the warm parts total 14.82 ms (F04). The receipt's measurement method cannot tell a one-time cost from a per-call cost, so the lead should rerun it with a warm-up and repeats. Nothing found blocks on correctness. F01 and F02 should be closed before promotion, because they leave production-reachable lanes without any gate.

**Your L99 readings, recomputed:**
- **GPU lines:** all 31 exited 0 between 05:11 and 05:14 UTC. Seven earlier g2-paths attempts exited 76 while waiting for a lease and produced no worker output.
- **G3:** 24 + 4 + 12 + 10 + 10 + 4 + 3.
- **Mutants:** each was killed after executing its edit once; r3m09 executed it three times.
- **Memory peaks:** match your readings.

## Deviations
- `rev3_census.py` raised a KeyError on the BP2 non-conv entries; fixed once by filtering on name.
- `rev3_emul.py` was extended once to add GEMM-lane multi-tile cases, and its summary count was corrected. No exception occurred.
- `rev3_testcensus.py` counted the BP parametrizations as 1; fixed once.
- The first plain run set HOME=/tmp, which hid the user-site pytest ("No module named pytest", rc 1); rerun with HOME=$HOME.
- The `REPORT.md` write was refused by the tool; its content is this message.

All commands ran with `CUDA_VISIBLE_DEVICES` empty, under `nice -n 19` and `timeout`, as `python3 -B -P` with single-threaded BLAS. Wall time was about 20 minutes. Nothing was deleted, and there were no git writes. The worktree is clean apart from `?? artifacts/bwd_rev3/`. Scratch was left in `/tmp/bwd_rev3_plain/` and `/tmp/bwd_rev3_scratch/`.

## Prior art
- **Dot-product (adjoint) test** for checking each VJP against its forward operator: Claerbout, *Imaging the Earth's Interior* (1985). Unverified — lead to check "Claerbout dot product test adjoint".
- **ONNX Conv / ConvTranspose operator specs (2017–)** for the brute-force forward.
- **Mutation analysis:** DeMillo, Lipton and Sayward (1978), used for M1–M4.
- **Mine, no prior art known to me:** the line-by-line transcription emulation and the reach-by-geometry census.
- **What the candidate cites:** Dumoulin and Visin (2016) for the adjoint maps; Chellapilla, Puri and Simard (2006) for im2col plus GEMM; Harris (2007) for the reduction tree; the cuBLAS 12.6 docs for pedantic compute, reproducibility and the stream/workspace reset. Its use matches those sources as I know them. Unverified — lead to check "cublasSetStream resets workspace cublasSetWorkspace" and "cublasGemmEx CUBLAS_COMPUTE_32F_PEDANTIC reproducibility".

Files are in `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_bwd/WorkTree/tensor_fork_bwd_wt_rev3/artifacts/bwd_rev3/`:
- findings_v1.json
- findings_v1.seal
- findings.json
- findings_final.seal
- checks/rev3_census.py, rev3_emul.py, rev3_reach.py, rev3_nonequiv.py, rev3_copycheck.py, rev3_testcensus.py
- checks/census.json, census.txt, emul.log, reach.log, nonequiv.log, copycheck.log, testcensus.log, plain.log, fixtures.log
