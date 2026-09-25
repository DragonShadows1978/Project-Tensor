# PT-TF32-1 report

**Built in the authorized fork; author CPU checks pass. GPU precision, speed,
onset behavior and the 6.5-second step target remain BLOCKED / UNPROVEN.**
The full archival BP suite remains RED. Not claimed fixed: the training spike
or training throughput.

The default-off GEMM switch (`set_tf32_gemm`, getter, `TC_TF32_GEMM=1`) uses
cuBLAS FAST_TF32 for FP32 tensors and preserves each graph's backward mode.
Separate h_tf32/g1_tf32 kernels use TF32 WMMA, FP32 storage/accumulation and
original FP32 selected-pair scalar scores. Both checkpoint initial inference
and replay are covered when h_tf32 is explicitly selected. Existing kernel
source and BF16/FP16 cuBLAS branches are byte-identical to the saved fork.

Evidence: successful CUDA 12.6/SM89 build; static TF32 HMMA instructions;
70 passing CPU tests, including native host guards; 10/10 CPU-model mutations
detected. Forty-nine GPU tests collect but were not run. Existing BP suites:
121 passed, seven failed, one pre-existing module skip. Five failures predate
this work; two are frozen setter/CMake pins changed by the authorized feature.
No assertion was weakened. Failed attempts and one compiler-temporary boundary
slip are recorded in the ledger.

TF32 operand rounding remains a material limitation: FP32 buffers and
accumulators do not imply FP32 multiplication accuracy. The strict 2× error
spread relative to the FP32 a kernel may reject this path. The exact-onset
gradient and healthy/control checks are mandatory before a remedy claim.
The optional FP32-forward/BF16-backward split is described but deferred until
the prerequisite timing result exists.

[Ledger, exact lead commands, integration and receipts](PT_TF32_1_LEDGER.md)
provide the handoff. Scripts `pt_tf32_1.py` and `pt_tf32_grapa.py` pin the fork
and preserve historical registrations; all lead outputs stay in new fork
directories. `artifacts/pt_tf32_1/BLOCKED_REPORT.json` enumerates the unrun GPU
lanes. `artifacts/pt_tf32_1/SHA256SUMS` records the delivered files.

## Prior art

Taken: [NVIDIA Ampere TF32 (2020), CUDA 11 WMMA and cuBLAS modes](https://developer.nvidia.com/blog/accelerating-ai-training-with-tf32-tensor-cores/);
[FlashAttention (Dao et al., 2022)](https://arxiv.org/abs/2205.14135) and
[FlashAttention-2 (Dao, 2023)](https://arxiv.org/abs/2307.08691), through
BP-KERNEL-2/3/4's ownership, recomputation, softmax/VJP and APA selection;
CC39/CC41's state harness and block policy. The deferred split draws on
[Micikevicius et al., ICLR 2018](https://arxiv.org/abs/1710.03740) and CC39-B's
specific precision ablation. NumPy, pytest, CUDA events, SHA256 and mutation
testing supply the testing/receipt methods; ancillary dates and search leads
are recorded in the ledger/code.

Ours: opt-in FP32 integration, TF32 fragment adaptation with FP32 coefficient
storage, shared-buffer reuse, host guards, and this order's contained harness.
No novelty claim for TF32, tiling, mixed precision or selective attention.
