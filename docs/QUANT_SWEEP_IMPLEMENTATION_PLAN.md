# Quant Weight Sweep Implementation Plan

Status: immutable after the initial house-rule commit.
Created: 2026-07-06
Branch: `codex/quant-sweep-house-rules`

## Objective

Build and run a reproducible Project-Tensor quantization sweep for low-bit
weight paths, starting with the native CUDA affine group-quantized kernels.
The first runnable target is a kernel-level sweep across BF16 dense, INT4,
INT3, and INT2 linear layers. Model-level perplexity is a later validation
phase and must not be inferred from synthetic linear-layer metrics.

## House Rules

- This implementation plan is the fixed source of intent after its initial
  commit. Do not revise it for execution details.
- The implementation ledger records commands, file changes, results, failures,
  and follow-up decisions as they happen.
- The narrative synthesis is the human-readable version of the ledger. It
  explains what the results mean without replacing the ledger receipts.
- Claims must identify their evidence class: kernel sweep, unit test, model
  perplexity, or external literature.
- Do not claim model quality from a kernel sweep. Kernel sweeps only establish
  speed, memory shape, reconstruction error, and output deviation against a
  dense reference.

## Phase 0: Baseline And Receipts

1. Create the house-rule documents.
2. Record the branch, repo state, existing low-bit APIs, and missing model-PPL
   bridge in the ledger.
3. Commit the plan/ledger/synthesis baseline before adding executable code.

## Phase 1: Runnable Kernel Sweep

1. Add a deterministic sweep script that uses the existing Project-Tensor
   low-bit math and CUDA kernels:
   - `quantize_affine_per_group`
   - `int4_linear_fused`
   - `intn_linear_fused`
   - BF16 dense `matmul` as the runtime reference
2. Measure each bit width on layer-like matrix shapes with fixed seeds.
3. Report:
   - packed weight memory estimate
   - compression against BF16 dense weights
   - latency
   - output RMSE and relative RMSE against BF16 dense
   - output cosine similarity
   - top-1 agreement when outputs are interpreted as logits
   - weight reconstruction error
4. Write raw JSON artifacts under `artifacts/quant_sweep/`. These artifacts are
   intentionally local by default because the repo ignores `*.json`.

## Phase 2: Verification

1. Run the new sweep in quick mode on the local GPU.
2. Run the existing low-bit unit tests:
   - `tensor_cuda/tests/test_int4_linear.py`
   - `tensor_cuda/tests/test_int4_symmetric.py`
   - `tensor_cuda/tests/test_intn_linear.py`
   - `tensor_cuda/tests/test_quantization_math.py`
3. Record the exact commands and outcomes in the ledger.
4. Update the synthesis with the meaning of the results.

## Phase 3: Model-PPL Bridge

The model-PPL bridge is a separate phase. It requires binding this weight
quantization path to a real model loader and running a real text corpus through
the model. Acceptable targets are local models already present on the machine,
but the bridge must be implemented explicitly. Until then, any PPL statement is
out of scope for this plan.

## Acceptance Criteria

- The house-rule documents exist and the plan remains unchanged after its
  initial commit.
- The sweep script can run from a clean checkout with the local TensorCUDA
  build on `PYTHONPATH`.
- The ledger contains command receipts and result summaries.
- The synthesis explains what the results do and do not prove.
- Existing low-bit kernel tests pass after the sweep script is added.
