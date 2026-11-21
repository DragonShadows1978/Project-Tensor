# BWD promotion record: TensorCUDA backward passes, R1 to R4b

**Change.** This adds TensorCUDA's own backward pass for Voice's Kokoro decoder. The owner ruled on 2026-11-18 that
TensorCUDA writes its own backward pass rather than borrowing one, for usability, re-usability and a stronger engine.
The lane was built in the fork `tensor_fork_bwd`, in five accepted rounds plus one fix round:
- **R1, the autograd hook and matmul backward.** Native autograd recording in `tensor_cuda` (`autograd.cpp`,
  `autograd.h`, `ops.cpp`, `bindings.cpp`) and a native self-test in the module (`bwd_selftest.cpp`). R1 also fixed a
  production defect: every bf16 tensor factory returned a tracked cast of a hidden fp32 leaf, so gradients never
  reached the user's tensor. On production `3138cf54`, all five bf16 factories gave `grad=None`.
- **R2, `tc_signal` small ops and AdaIN backward** (`_small.py`, `_norms.py`).
- **R3, conv1d and conv_transpose1d backward** in `tc_signal`.
- **R4a, the Kokoro decoder and generator chain:**
  - the instance-norm affine backward;
  - `checkpoint`, with an input-revision check that refuses a checkpointed function that writes its own inputs, and
    refuses recomputation after an input changed.
- **R4b, the full chain to the waveform and the MFCC loss with Voice's tail, and public conv backward calls:**
  - `tc_signal.conv1d_backward(x, w, grad_out, *, stride, padding, dilation, groups, needs) -> (dx, dw, db)`;
  - `tc_signal.conv_transpose1d_backward(x, w, grad_out, *, stride, padding, output_padding, dilation, groups, needs) -> (dx, dw, db)`.
- **PF1, the cross-model review's fixes:**
  - the checkpoint replay restores grad mode when the replayed function returns a non-Tensor. This was a pre-existing
    production defect;
  - the `checkpoint` docstring states its contract;
  - a GPU test of the scale gradient with `x` frozen.

**Assembly.** The promotion tree is production main `952c6b0e` (APA-PF2 merged), plus the reviewed lane tip `e3b2d9aa`'s
files for every product and test path the lane changed since its fork base `3138cf54`. The lane's `bindings.cpp`
hunks are applied by 3-way over PF2's, cleanly; it is the only file both changed. For `tensor_cuda` and `tc_signal`,
with identical hunk text:
- (promotion tree minus production) equals (lane minus `3138cf54`): 45 files, +5,541/−40;
- (promotion tree minus lane) equals (production minus `3138cf54`): PF2's 10 files.

Outside those two packages, the tree equals production plus the receipts listed below.

**Receipts carried** (`artifacts/bwd_d`, `bwd_r1` to `bwd_r4b`, `bwd_rev1` to `bwd_rev4b` and `bwd_rev4ax`):
- each round's filed reports, ledgers, findings maps, registrations and tolerances;
- Voice's ranking-eligible list;
- the final per-test-ID comparisons;
- PF1's records;
- every review's report and findings.

Harness scripts, staged packages and per-round GPU outputs stay in the fork.

**Assembly correction (MAINCHK4 Amendment 2).**
- The first assembly carried `artifacts/bwd_r1/TOLERANCES.json` and `TOLERANCES_BF16_R5.json` but not their seal files,
  `TOLERANCES.sha256` and `TOLERANCES_BF16_SHA256_R5.txt`. The R1 tests read both seals: `bwd_r1_support/oracle.py`
  `tolerances()` and `test_bwd_r1_bf16.py` `bf16_tolerances()`.
- MAINCHK4's GPU comparison found 64 R1 tests failing on the promotion tree with FileNotFoundError on those two
  paths, while passing on the lane: 45 in `test_bwd_r1_core.py`, 15 in `test_bwd_r1_bf16.py` and 4 in
  `test_bwd_r1_matmul.py`.
- Both seals are now carried byte for byte from the reviewed lane branch `bwd-r4b-full-chain`. No product or test
  file changed.

**Scope.**
- **Covers:**
  - gradients through `tensor_cuda` ops, `tc_signal`'s small ops, norms, AdaIN and convolutions;
  - checkpointed recomputation with its input-revision contract;
  - the public conv backward calls;
  - the Kokoro decoder chain as Voice uses it.
- **Does not cover:**
  - installing a rebuilt engine binary (a separate request);
  - speed beyond the recorded timings;
  - optimizers or training loops;
  - Voice's own gradient driver (Voice builds it in its repo);
  - the three owner-name lines already in production (`tc_signal/tc_signal/__init__.py:3`, `tc_signal/README.md:4`,
    `tensor_cuda/bench/geometries.py:11`), which this promotion leaves untouched.

**Review: cross-model on every authored part.**

| Part | Author | Reviewer | Result |
|---|---|---|---|
| R1 (r1 to r6) | Codex gpt-6.2-sol | Claude Opus 5.5, BWD-REV1 (blind, four passes) | no blocker or major left |
| R2 (and r2) | Codex | Claude, BWD-REV2 (blind) | 3 MINOR, fixed once |
| R3 (and r2) | Codex | Claude, BWD-REV3 (blind) | 3 MINOR test gaps, fixed once |
| R4a r1 to r4 | Codex | Claude, BWD-REV4 (blind), re-check and confirm | 3 MAJOR, then 1 MAJOR, then none |
| R4a r5b to r5f | Claude Opus 5.5 (the Codex seat was stopped by a provider filter) | Codex, BWD-REV4AX | PASS; 3 MINOR and 1 NIT, fixed once in PF1 |
| R4b (r1 to r10) | Claude Opus 5.5 | Codex, BWD-REV4B: a review, two re-checks, four confirms | CLEAN for closure; every finding was in the qualification harness, none in a gradient |
| PF1 | Codex | the lead, with planted checks (standing rule for MINOR and NIT) | verified |

Every finding is fixed or closed with a recorded reason. See `review_receipt.json` and each review's folder.

**Gate evidence** (lead GPU lines):
- **R1:** 11 of 11 (r6).
- **R2:** 10 of 10.
- **R3:** 33 of 33. Voice's 98-case forward census is bitwise on the R3 packages.
- **R4a:**
  - all 412 lane GPU tests pass;
  - the chain matches Voice's fp64 key on all 313 parameters and the style vector;
  - a 10 s utterance fits in 3.6 GB with checkpointing.
- **R4b:**
  - G1 passes and the finite-difference checks are 40 of 40, for both the linear and the MFCC loss;
  - G5's sampled estimate is 3.61 GB against the 4.0 GB bar;
  - backward takes 0.9 s for 10 s of audio;
  - the rebuilt reference reproduces the sealed manifest digest;
  - final receipts `7c30e219`.
- **PF1:**
  - 517 tests on the rebuilt package: 516 pass, and 1 skips because it needs two GPUs;
  - the F03 check passes on both packages and catches its planted fault;
  - receipts `e3b2d9aa`.
- **Voice:** `tail_vjp.py` (Voice main `c6d61df`) now calls the public conv backward. Its gradients are byte-identical
  to the previous file in 13 of 13 cases.

**Before push:** a merged-tree check (MAINCHK4) builds this tree and compares it per test ID with both parents, on CPU
and on GPU.

**Approval:** pending. It goes to the owner as a decision request after MAINCHK4 passes. The engine binary install is
not part of this merge.
