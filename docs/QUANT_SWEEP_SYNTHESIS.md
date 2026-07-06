# Quant Weight Sweep Narrative Synthesis

The quant sweep is being split into two validation layers.

The first layer is a runnable Project-Tensor kernel sweep. It compares BF16
dense linear layers with the current native INT4, INT3, and INT2 affine
group-quantized CUDA paths. This can be run immediately because the low-bit
math and fused kernels already exist in Project-Tensor. The result will tell us
how the low-bit paths behave in memory, latency, weight reconstruction, and
output deviation against a dense reference.

The second layer is model-level perplexity. That requires a real model loader
using these weight paths and a real text corpus. Project-Tensor does not yet
have that generic bridge, so this run will not claim PPL. Any PPL result must
come from a separate model-bound validation phase.

As of the initial setup on 2026-07-06, the correct first move is to get the
kernel sweep running with receipts, preserve the implementation plan as the
fixed intent, and use the ledger for command-level evidence.

After the baseline commit, the sweep harness was added as a direct consumer of
the existing TensorCUDA APIs. That keeps this phase focused: it tests the
native low-bit CUDA paths we already have, instead of introducing a second
quantization implementation that would make the result harder to interpret.

The first run produced both useful wins and useful failures. INT4 decode at
M=1, N=2048, K=4096 compressed the weight footprint from 16.00 MiB to 4.25 MiB
and ran faster than the dense BF16 reference. That is a real kernel-level win.
However, INT2 and INT3 did not become faster on the same decode shape, and they
paid much larger output error. On the M=16 prefill shape, every fused low-bit
path lost badly to dense BF16 cuBLAS. That does not make the work useless; it
identifies the current boundary. Decode can benefit from the low-bit path,
while prefill needs a better kernel strategy before it should be sold as a
speed win.

The house-rule interpretation is explicit now: a failure is still a result. The
SCRIBE and Translation work are the precedent for this project style. If a path
does not work, the result is preserved with receipts so the next decision is
better informed.

The second run added progress and CPU pack timings. That exposed a separate
finding: the reference NumPy packer dominates this quick sweep's wall time.
For the 2048x4096 matrix, packing took about 14.6 seconds for INT4, 11.5-11.6
seconds for INT3, and 8.5 seconds for INT2 each time the weight was quantized.
The CUDA kernel timings are much smaller than that. So there are two different
tracks now: runtime fused-kernel behavior, and offline quantization throughput.

Regression coverage stayed green after adding the sweep harness: the existing
INT4, symmetric INT4, INT2/INT3, and quantization math tests passed as a
16-test set.

The scope must be corrected. This was not real model validation. It was kernel
smoke testing plus a structured linear sweep. That is useful, but it is not the
same as loading a model, measuring resident memory, and running PPL. In
particular, the current evidence does not answer the question that matters most:
whether INT3 or INT2 survive model perplexity, or whether they fall off a cliff
the way very low-bit model weights often do.

The next real validation step is to wire the low-bit weight path into an actual
model adapter. GraftRepository already has real Qwen/Gemma/MiniCPM/DeepSeek
adapters that use `QuantLinearTC`, but that wrapper is still INT4-specific.
Project-Tensor has the native INT2/INT3 kernels; the missing bridge is a model
weight wrapper that can choose INT4, INT3, or INT2 and then run the established
memory and PPL protocols on real text. Until that exists and is run, INT3 and
INT2 are only kernel-tested, not model-tested.
