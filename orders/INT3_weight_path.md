# ORDER INT3 — 3-bit weight quantization path for tensor_cuda

YOUR WRITABLE TARGET is this worktree (/mnt/ForgeRealm/Project-Tensor-int6,
branch `int3-weights` — a fork checkout of Project-Tensor) — edits,
builds, and test runs inside it are AUTHORIZED. Also writable: /tmp.
The canonical Project-Tensor checkout and every other repo are READ-ONLY.
The lead merges this branch later; you never run git.

## Context

Target model: Qwen3.8-27B (BF16 weights downloading separately; NOT part
of this order). At 3 bits/weight a 27B packs to ~10.2GB + fp16 g128
scales — the point is fitting the 12GB card. This order is the KERNEL
path only; the model adapter is a later, separate order.

The engine already has INT4 (g128, symmetric variant, three paths:
dequant / GEMV / fused tile), INT8, and — on this branch's tip commit
4d10951 — INT6 (same three paths, INT4-mirrored surface). You are adding
INT3 as the third sibling. The NumPy reference math ALREADY supports
3-bit: `tensor_cuda/tensor_cuda/quantization/affine.py` has a vectorized
`_pack_int3` (little-endian bit stream, code k at bit offset 3k) and
`quantize_symmetric_per_group` / `quantize_affine_per_group` accept
bits=3. Your CUDA layout MUST match that existing reference bit-for-bit —
do not invent a second 3-bit layout.

## Task

1. **Read the INT6 commit first** (`git show 4d10951` is allowed —
   read-only git is fine, you just never write git state): kernels
   (`tensor_cuda/src/kernels.cu`), dispatch (`src/ops.cpp`,
   `src/bindings.cpp`), headers (`include/tc/core.h`, `include/tc/ops.h`),
   Python surface (`tensor_cuda/__init__.py`, `quantization/`), and the
   gate/test conventions (`tests/int6_weight_gates.py`,
   `tests/test_int6_weights.py`, `tests/test_quantization_math.py`).
   Mirror that structure exactly; do not invent a parallel idiom.
2. **INT3 format**: 3-bit codes, group-wise FP16 scaling at group 128,
   symmetric variant as the primary path — dequant is (q − 4)·s, i.e.
   z = −4·s derived in-register from an empty zeros tensor, matching the
   INT4 (z=−8·s) and INT6 (z=−32·s) conventions. Packing: eight codes
   per three little-endian bytes — 24-bit word
   `w = q0 | q1<<3 | q2<<6 | q3<<9 | q4<<12 | q5<<15 | q6<<18 | q7<<21`
   — which is exactly affine.py's bit-offset-3k stream when K is a
   multiple of 8 (group 128 guarantees it). Document the layout in the
   header comment with a worked byte example, INT6-style.
3. **Implement**: CUDA `int3_dequant`; CUDA `int3_linear` GEMV consuming
   packed INT3 directly (no fp16 expansion of the whole weight);
   `int3_linear_fused` tile path ONLY if the INT6 tile kernel's
   structure makes it near-free — otherwise register it as a named
   successor. Python surface mirrors the INT6/INT4 API (same argument
   shapes/naming). Weight-only: activations stay fp16 everywhere.
4. **Gates (write them, run what the sandbox allows)**:
   - CUDA unpack/dequant bit-exactness vs the affine.py NumPy reference
     on synthetic grids covering all 8 code values × scale signs ×
     group boundaries — the reference is the law for the layout;
   - GEMV parity vs dequant-then-matmul reference at real Qwen3.8-27B
     shapes (K ∈ {1024, 2048, 5120, 6144, 17408}), rel err ~1e-3 class;
   - quantize→dequant round-trip error reported (max|Δ|, rel-fro) on
     Gaussian AND outlier-heavy synthetic weights — at 3 bits expect
     the outlier-heavy numbers to be ugly; report them honestly, they
     are characterization, not a pass/fail gate;
   - existing engine test suite still green in this fork. The GroupNorm
     suite failure is KNOWN pre-existing (reproduced on canonical at the
     INT6 order); reproduce-and-report, do not fix, do not count as RED.
5. If the sandbox has NO GPU (known possibility): build everything, run
   CPU-verifiable gates (packing/round-trip logic vs the NumPy
   reference), and mark each GPU gate PENDING-LEAD with its exact
   invocation command. That is a valid, complete delivery — say so
   plainly, never fake a GPU receipt.

## Rails

- NO git writes. NO subagents. GPU test runs (if GPU available):
  foreground, `flock -w 3600 /tmp/forge-gpu.lock`, `timeout 590` each;
  background children inside a flocked shell require setsid.
- RED honesty: a failing gate, a compile error you can't clear, or a
  design dead-end is a result — report it verbatim with the error.
- Do not touch INT4/INT6/INT8 code paths except at shared dispatch
  seams; any shared-seam edit gets a one-line justification.
- SCOPE LAW: kernel gates establish speed, memory shape, reconstruction
  error, and parity vs a dense reference ONLY. No claims about model
  quality at 3 bits — that is a later, separately gated evaluation.

## Done — final message must contain, verbatim

- Files created/modified (exact paths) + packing layout doc excerpt.
- Each gate: PASS with its printed numbers, FAIL with error text, or
  PENDING-LEAD with the exact command to run.
- Whether the fused tile path shipped or is a named successor.
- Any shared-seam edits and why.
- Any deviation from this order.
