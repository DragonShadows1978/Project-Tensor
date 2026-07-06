# Quant Weight Sweep Implementation Ledger

This ledger is the execution record for the quant weight sweep. The immutable
implementation plan stays fixed; this file records what actually happened.

## 2026-07-06 14:51:49 EDT

Action: Baseline orientation.

Repo state:
- Repository: `/mnt/ForgeRealm/Project-Tensor`
- Branch: `codex/quant-sweep-house-rules`
- Initial status: clean branch with no task changes.

Findings:
- Existing Project-Tensor low-bit APIs are present for affine group
  quantization and native CUDA low-bit linear paths.
- Existing test coverage includes INT4, INT2, INT3, symmetric INT4, and shared
  quantization math.
- Project-Tensor does not currently expose a generic model-PPL runner for this
  low-bit weight path. Model-PPL validation is therefore tracked as a later
  bridge phase, not part of the first kernel sweep result.

House-rule documents created:
- `docs/QUANT_SWEEP_IMPLEMENTATION_PLAN.md`
- `docs/QUANT_SWEEP_IMPLEMENTATION_LEDGER.md`
- `docs/QUANT_SWEEP_SYNTHESIS.md`

Next action:
- Commit the house-rule baseline before adding executable sweep code.

## 2026-07-06 14:55 EDT

Action: House-rule baseline committed.

Receipt:
- Commit: `2bb30b1 docs: register quant sweep house rules`

Action: Added Phase 1 sweep harness.

Files changed:
- `scripts/quant_weight_sweep.py`

Implementation notes:
- The script compares dense `tc.matmul(..., trans_b=True)` against INT4,
  INT3, and INT2 fused low-bit linear paths.
- It uses the existing Project-Tensor quantization math, not a separate
  quantizer implementation.
- It reports latency, BF16 memory compression, output deviation, top-1
  agreement, cosine similarity, and weight reconstruction error.
- Raw JSON artifacts are written under `artifacts/quant_sweep/`, which remains
  local because the repo ignores `*.json`.

Next action:
- Run the quick BF16 CUDA sweep and record the output.

## 2026-07-06 14:57 EDT

Action: Ran the first quick BF16 quant sweep.

Command:
- `env PYTHONPATH=/mnt/ForgeRealm/Project-Tensor/tensor_cuda PYTHONDONTWRITEBYTECODE=1 python3 scripts/quant_weight_sweep.py --quick --compute-dtype bfloat16 --reps 20 --warmup 5`

Artifact:
- `artifacts/quant_sweep/quant_weight_sweep_20260706_145705.json`

Result summary:

| Shape | Mode | Latency ms | Speed vs dense | Weight MiB | Compression | Output rel RMSE | Cosine | Top-1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| decode_mid M=1 N=2048 K=4096 | BF16 | 0.0257 | 1.00x | 16.00 | 1.00x | 0.0000 | 1.000000 | 1.000 |
| decode_mid M=1 N=2048 K=4096 | INT4 | 0.0192 | 1.34x | 4.25 | 3.76x | 0.1134 | 0.993651 | 0.000 |
| decode_mid M=1 N=2048 K=4096 | INT3 | 0.0753 | 0.34x | 3.25 | 4.92x | 0.2479 | 0.970577 | 0.000 |
| decode_mid M=1 N=2048 K=4096 | INT2 | 0.0681 | 0.38x | 2.25 | 7.11x | 0.5737 | 0.869037 | 0.000 |
| prefill_mid M=16 N=2048 K=4096 | BF16 | 0.0314 | 1.00x | 16.00 | 1.00x | 0.0000 | 1.000000 | 1.000 |
| prefill_mid M=16 N=2048 K=4096 | INT4 | 0.3031 | 0.10x | 4.25 | 3.76x | 0.1184 | 0.993077 | 0.812 |
| prefill_mid M=16 N=2048 K=4096 | INT3 | 0.4045 | 0.08x | 3.25 | 4.92x | 0.2513 | 0.970125 | 0.750 |
| prefill_mid M=16 N=2048 K=4096 | INT2 | 0.3682 | 0.09x | 2.25 | 7.11x | 0.5985 | 0.860530 | 0.500 |

Interpretation:
- Positive result: INT4 decode on this shape is smaller and faster than dense.
- Negative result: INT3 and INT2 are not faster than dense on this decode shape,
  and both have materially larger output deviation than INT4.
- Negative result: all low-bit fused prefill paths lose badly to BF16 dense
  cuBLAS on this shape. This points at a prefill kernel optimization gap, not a
  memory-capacity result.
- Harness result: the first quick run produced no progress output while CPU
  packing was running. That made the script feel hung even though it completed.

House-rule clarification from the user:
- Failure is a result, and sometimes the most important result. SCRIBE and the
  Translation line are the precedent: days of failed evidence still count as
  research output and must be preserved rather than hidden.

Follow-up action:
- Update the harness to print progress and record CPU quantization time so
  slow packing is visible in future receipts.

## 2026-07-06 15:01 EDT

Action: Reran the quick BF16 quant sweep after adding progress output and
CPU quantization timing.

Command:
- `env PYTHONPATH=/mnt/ForgeRealm/Project-Tensor/tensor_cuda PYTHONDONTWRITEBYTECODE=1 python3 scripts/quant_weight_sweep.py --quick --compute-dtype bfloat16 --reps 20 --warmup 5`

Artifact:
- `artifacts/quant_sweep/quant_weight_sweep_20260706_150100.json`

Result summary:

| Shape | Mode | Pack ms | Latency ms | Speed vs dense | Weight MiB | Compression | Output rel RMSE | Cosine | Top-1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| decode_mid M=1 N=2048 K=4096 | BF16 | 0.0 | 0.0272 | 1.00x | 16.00 | 1.00x | 0.0000 | 1.000000 | 1.000 |
| decode_mid M=1 N=2048 K=4096 | INT4 | 14624.0 | 0.0184 | 1.48x | 4.25 | 3.76x | 0.1134 | 0.993651 | 0.000 |
| decode_mid M=1 N=2048 K=4096 | INT3 | 11620.1 | 0.0699 | 0.39x | 3.25 | 4.92x | 0.2479 | 0.970577 | 0.000 |
| decode_mid M=1 N=2048 K=4096 | INT2 | 8524.7 | 0.0684 | 0.40x | 2.25 | 7.11x | 0.5737 | 0.869037 | 0.000 |
| prefill_mid M=16 N=2048 K=4096 | BF16 | 0.0 | 0.0315 | 1.00x | 16.00 | 1.00x | 0.0000 | 1.000000 | 1.000 |
| prefill_mid M=16 N=2048 K=4096 | INT4 | 14670.7 | 0.3350 | 0.09x | 4.25 | 3.76x | 0.1184 | 0.993077 | 0.812 |
| prefill_mid M=16 N=2048 K=4096 | INT3 | 11494.8 | 0.4340 | 0.07x | 3.25 | 4.92x | 0.2513 | 0.970125 | 0.750 |
| prefill_mid M=16 N=2048 K=4096 | INT2 | 8487.3 | 0.3381 | 0.09x | 2.25 | 7.11x | 0.5985 | 0.860530 | 0.500 |

Additional receipt:
- Total script wall time: 133.13 seconds.
- The script now prints per-shape/per-bit progress while CPU packing runs.
- The JSON payload now records `artifact`, `total_wall_seconds`, and
  per-bit `quantize_ms`.

Interpretation:
- The prior conclusions held after instrumentation.
- CPU packing is a real bottleneck in the current reference quantizer. This is
  not a CUDA kernel problem, but it matters for large sweep ergonomics.
- The fused low-bit path is currently decode-useful for INT4 on this shape.
- The fused low-bit path is not currently a prefill-speed win on this shape.

## 2026-07-06 15:04 EDT

Action: Ran low-bit regression tests.

Command:
- `env PYTHONPATH=/mnt/ForgeRealm/Project-Tensor/tensor_cuda PYTHONDONTWRITEBYTECODE=1 PYTEST_ADDOPTS='-p no:cacheprovider' python3 -m pytest tensor_cuda/tests/test_int4_linear.py tensor_cuda/tests/test_int4_symmetric.py tensor_cuda/tests/test_intn_linear.py tensor_cuda/tests/test_quantization_math.py -q`

Result:
- `16 passed in 0.97s`

Next action:
- Commit the Phase 1 sweep harness and updated ledger/synthesis.

## 2026-07-06 15:12 EDT

Action: Corrected validation scope after user review.

User correction:
- The Phase 1 work was not real model validation. It was kernel smoke testing
  plus a structured synthetic linear sweep.
- It did not run a model memory ceiling test.
- It did not run model perplexity.
- Therefore it did not establish whether INT3 or INT2 fall off a cliff at the
  model level.

Corrected status:
- Kernel API/correctness: tested.
- Structured synthetic linear behavior: tested.
- Model memory behavior: not tested.
- Model PPL behavior: not tested.
- Production viability for INT2/INT3 weights: not established.

Repo finding:
- GraftRepository model adapters use `QuantLinearTC` for real model weights,
  but that wrapper is currently hardwired to INT4.
- Project-Tensor exposes native `intn_linear` and `intn_linear_fused`, but no
  real model loader is currently wired to select INT2/INT3 weights.

Required next step for an actual answer:
- Add a selectable weight-bit path to the real model adapter layer, likely by
  generalizing `QuantLinearTC` or adding an `IntNQuantLinearTC` wrapper.
- Run a real model memory test.
- Run real PPL over the established text-window protocol.
- Record failure as a result if INT3/INT2 collapse.
