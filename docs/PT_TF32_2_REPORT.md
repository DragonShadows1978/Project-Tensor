# PT-TF32-2 report

**Implemented and rebuilt in the fork; GPU certification BLOCKED.**
Author CPU suite: 112 passed, one historical binary-pin failure retained RED.
All 432 protected historical files match their initial hashes. No GPU ran.

The controlled CPU ablation identifies selection rounding as the dominant dK
error source in its fixture: correcting only the predicate changes dK rel-L2
from `0.010012751525230686` to `0.00046814694494103505`. Correcting only dO·V or
the outer product leaves the error near `0.0100`. This is CPU-model evidence;
the original native dK failure is **not claimed fixed** until its GPU gate runs.

The candidate uses residual-corrected TF32 products for selection scores,
dO·V and saved forward P·V, plus the exact one-visible-key softmax identity.
The seven native edge failures remain unconfirmed; their test tolerances were
not raised. Aligned GEMM now selects cuBLASLt algorithms with explicit TF32
HMMA/FP32-accumulation capability flags and a 32 MiB workspace; ragged GEMM
uses padded WMMA tiles. Neither ≥5× speed nor native accuracy is claimed.

The new immutable registration derives attention bounds from 10-bit TF32
rounding and FP32 accumulation. It independently checks complete FP64
forward/backward, same-state isolated backward, and selection. These statistical
engineering bounds are not worst-case proofs. Healthy/control use a frozen
four-run healthy-state cosine spread, followed by separate holdouts. Original
onset and timing limits remain, as do all historical RED receipts.

[Ledger and exact bounded lead sequence](PT_TF32_2_LEDGER.md) provide the handoff.
The runner is `scripts/pt_tf32_2_slot.py`; it runs one sequential slot with a
1,800-second global deadline, all 56 GPU tests, 27 GEMM cells, four attention
shapes, memcheck/racecheck/synccheck, calibration, state holdouts and step time.
`artifacts/pt_tf32_2/BLOCKED_REPORT.json` enumerates all unmeasured gates.

## Done

```text
BUILD_RC=0 SOURCE_PINS_UNCHANGED=True RECEIPT=/mnt/ForgeRealm/wt/pt-tf32/artifacts/pt_tf32_2/build_02/receipt.json
1 failed, 112 passed in 2.34s
E       ValueError: fork engine drift
56 tests collected in 0.12s
PROTECTED_FILES 432 CHANGED []
```

The RED CPU test intentionally still verifies PT-TF32-1's binary pin. The
PT-TF32-2 fork registration passes its own pin check. Static SASS contains
TF32 HMMA instructions; this is not runtime dispatch/performance certification.
No blind verification occurred. Native correctness, all GPU gates and training
stability remain **not claimed fixed**.

## Prior art

Taken: [NVIDIA TF32 (2020) and cuBLASLt/WMMA](https://docs.nvidia.com/cuda/archive/12.6.2/cublas/index.html);
[Ootomo and Yokota (2022)](https://arxiv.org/abs/2203.03341) input residual
decomposition; [FlashAttention-2 (Dao, 2023)](https://arxiv.org/abs/2307.08691)
through existing BP-KERNEL-2/3/4; standard singleton softmax/variance identities;
[Higham (2002)](https://nhigham.com/accuracy-and-stability-of-numerical-algorithms/)
floating-point error models; NumPy/pytest/SHA256/CUDA-event receipt methods.
CC39/CC41 and PT-TF32-1 (2026) supply the state/policy harness. POSIX process
groups/Python subprocess supply deadline containment. Ours: local integration,
predicate diagnosis, targeted three-product placement, tail-safe fallback,
capability receipts and independent TF32 gates. The three-product adaptation
is not the cited paper's complete FP32-equivalent implementation. The exact
cosine-spread rule comes from the user; no prior art known to me for that rule.
No novelty claim is made for the underlying numerical or tiling methods.
