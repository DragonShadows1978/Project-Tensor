# ORDER INT6 — 6-bit weight quantization path for tensor_cuda

YOUR WRITABLE TARGET is this worktree (/mnt/ForgeRealm/Project-Tensor-int6,
branch `int6-weights` — a fork checkout of Project-Tensor) — edits,
builds, and test runs inside it are AUTHORIZED. Also writable: /tmp.
The canonical Project-Tensor checkout and every other repo are READ-ONLY.
The lead merges this branch later; you never run git.

## Context

Program: Qwen3-1.7B name-checker plan (GraftRepository
docs/QWEN3_1P7B_NAMECHECKER_PLAN.md, phase P3). The engine has INT4
(g128 + symmetric-8 variant, three paths: dequant, GEMV, tile/two-stage)
and INT8; it has NO INT6 weight path. You are adding one, matching the
house INT4 conventions so a model adapter can swap bit-width without
restructuring.

## Task

1. **Read the existing INT4 implementation first** (kernels + Python
   surface): packing layout, group scaling (g128; symmetric variant
   derives z=−8·s in-register from an empty zeros tensor), dispatch
   points (dequant / GEMV / tile paths), and the test conventions in
   tests/ (e.g. the symmetric-INT4 gate). Mirror the structure; do not
   invent a parallel idiom.
2. **INT6 format**: 6-bit codes, group-wise FP16 scaling at group 128,
   symmetric variant (z = −32·s) as the primary path (matching the
   house preference); packing = 4 values per 3 bytes, layout documented
   in a header comment with a worked byte example.
3. **Implement**: quantize (host-side is fine) + pack; CUDA dequant;
   CUDA GEMV consuming packed INT6 directly (no fp16 expansion of the
   whole weight); tile/two-stage path ONLY if the GEMV path's structure
   makes it near-free — otherwise register it as a named successor.
   Python surface mirrors the INT4 API (same argument shapes/naming).
4. **Gates (write them, run what the sandbox allows)**:
   - dequant bit-exactness on synthetic grids covering all 64 code
     values × scale signs × group boundaries;
   - GEMV parity vs dequant-then-matmul reference at real shapes
     (K∈{2048, 6144}, M∈{2048, 151936-row chunked}) — rel err ~1e-3
     class;
   - quantize→dequant round-trip error reported (max|Δ|, rel-fro) on
     Gaussian and outlier-heavy synthetic weights;
   - existing engine test suite still green in this fork.
5. If the sandbox has NO GPU (known possibility): build everything,
   run CPU-verifiable gates (packing/round-trip logic in NumPy
   reference), and mark each GPU gate PENDING-LEAD with its exact
   invocation command. That is a valid, complete delivery — say so
   plainly, never fake a GPU receipt.

## Rails

- NO git. NO subagents. GPU test runs (if GPU available): foreground,
  `flock -w 3600 /tmp/forge-gpu.lock`, `timeout 590` each; background
  children inside a flocked shell require setsid.
- RED honesty: a failing gate, a compile error you can't clear, or a
  design dead-end is a result — report it verbatim with the error.
- Do not touch INT4/INT8 code paths except at shared dispatch seams;
  any shared-seam edit gets a one-line justification in the receipt.

## Done — final message must contain, verbatim

- Files created/modified (exact paths) + packing layout doc excerpt.
- Each gate: PASS with its printed numbers, FAIL with error text, or
  PENDING-LEAD with the exact command to run.
- Whether the tile/two-stage path shipped or is a named successor.
- Any shared-seam edits and why.
- Any deviation from this order.
