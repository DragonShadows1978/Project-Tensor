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
