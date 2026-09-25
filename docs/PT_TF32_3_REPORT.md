# PT-TF32-3 report

**Fork rebuilt; CPU handoff complete. GPU certification remains BLOCKED.**
The final author suite passes 36 CPU tests. The historical suite retains two
old binary-pin failures (111 passed). No GPU was used; no live engine or
read-only GRAPA source was edited, rebuilt or imported as an engine.

The candidate accumulates forward threshold moments in FP64, then saves one
FP32 threshold used consistently by both halves. A controlled CPU ablation
reduces threshold-induced selection flips on all four registered shapes;
residual FP32/score rounding remains. **Native downstream dK is not claimed
fixed.** The two edge failures are dV: TF32 probability midpoint jumps explain
the three reported element discrepancies within 2.2e-8. Legacy assertions and
tolerances remain unchanged, with added independent FP64 edge checks.

GEMM now has shared host descriptor/device heuristic diagnostics and actual
dispatch readback for algorithm, numerical flags, compute type, transposes,
workspace and alignment. The deprecated math-mode compatibility query reports
its actual API status; supported HMMA/TF32/FP32 flags remain mandatory. The
existing receipts already reported Tensor Core flags, so silent CUDA-core
fallback was not established. Explicit TF32 outputs now use stream-ordered
allocation to remove a plausible allocation bottleneck. The lead will measure
both global pooling off and model pooling on. **>=5x is not claimed.**

Full gradients use a pipe and RAM by default, preserving exact comparisons;
checkpoint saves are suppressed in timing children. `--keep-grads` opts into
retained dumps. Every lane checks 8 GiB free, low-space/ENOSPC yields BLOCKED,
and sanitizer error counts determine their verdict independently of pytest.
Missing summaries or execution coverage cannot pass. The default gradient/
checkpoint dump total printed by the runner is zero.

The [ledger](PT_TF32_3_LEDGER.md) contains code-site reasoning, source seals,
verbatim failures, exact commands and the 1705-second lane schedule under an
1800-second global deadline, including new healthy/control/step-time runs.
[BLOCKED_REPORT.json](../artifacts/pt_tf32_3/BLOCKED_REPORT.json) enumerates the
unmeasured certification gates. The original BF16 attention source and object,
and FP16/BF16 GEMM branch text, are byte-identical; runtime parity remains
unmeasured. No blind review or training-stability claim is made.

## Done

```text
BUILD_RC=0 SOURCE_PINS_UNCHANGED=True RECEIPT=/mnt/ForgeRealm/wt/pt-tf32/artifacts/pt_tf32_3/build_02/receipt.json
36 passed in 1.19s
60 tests collected in 0.13s
2 failed, 111 passed in 2.22s
E           ValueError: fork binary drift
E       ValueError: fork engine drift
PRESERVED 656 OF 668 UNEXPECTED_CHANGES []
BF16_FP16_GEMM_BRANCH_BYTE_IDENTICAL True
KERNELS_CU_AND_OBJECT_BYTE_IDENTICAL True
```

The two RED CPU results belong to immutable PT-TF32-1/2 engine registrations.
The new PT-TF32-3 registration and final manifest pass. Final binary SHA256:
`a1bf6ce6b5212b83f4652547fd87e3508ae824f9ab461850680c2081320f8fbe`.

## Prior art

Taken: [NVIDIA TF32/cuBLASLt/WMMA (2020–2024)](https://docs.nvidia.com/cuda/archive/12.6.2/cublas/),
[Kini and Hemstad, stream-ordered allocation (2021)](https://developer.nvidia.com/blog/using-cuda-stream-ordered-memory-allocator-part-1/),
Higham's floating-point error analysis (2002), existing
[Ootomo/Yokota residual products (2022)](https://arxiv.org/abs/2203.03341),
[Dao's FlashAttention-2 (2023)](https://arxiv.org/abs/2307.08691),
CC39/CC41/PT-TF32-1/2 loaders and owner tiles (2026), POSIX pipes, NumPy and
SHA256. Ours: placement of wider threshold moments and opt-in output pooling,
midpoint diagnosis, host dispatch receipts, bounded transient output adapters
and exact scalar calibration receipts. These are local integrations, with no
novelty claim for the underlying methods. No prior art known to me for the
user's exact four-arm cosine calibration rule. Deprecated math-mode ABI slot
8 remains an **unverified — lead to check** mapping against NVIDIA's historical
`CUDA 11 cublasLt.h CUBLASLT_ALGO_CAP_MATHMODE_IMPL`; the installed header omits
that enumerator. Runtime API status is retained rather than guessed.
