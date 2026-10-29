# GPU harness precision/health clarification before measurements
Registration is unchanged. Pool E=max(base-off maxabs error) by input dtype and
tensor name over both small shapes and all variants, then freeze max(8E,1e-5).
No arm data participates. This interprets "each output/gradient" in registration;
c's intentionally-zero baseline dk alone cannot estimate BF16 accumulation error.
Variant c intentionally has zero dV, so L19 prohibits crediting its all-zero
outputs in any bitwise gate. It remains covered numerically by L-G3 plus the
strict nonzero unselected-key-1 witness. Train setter already refuses c.
Every credited tensor in a bitwise gate is checked nonzero and nonconstant.
Nsight CUDA kernel records supply L-G6 kernel-only times; CUDA events additionally
measure backward wrapper time, explicitly including allocations/zeroing.
Prior art: existing Project-Tensor deletion ablation c (2026), NVIDIA Nsight
Systems/CUDA events (years unverified - lead to check). Harness integration ours.
