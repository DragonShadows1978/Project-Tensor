# APA-SP2 implementation seat delivery

**Host build PASS; G1 PASS; G2/G3 BLOCKED_NO_GPU, RED.** No measured e_q or epsilon performance result is supplied by this seat. Evidence for the future curve is **kernel sweep**: **this establishes nothing about model quality**. No epsilon is selected or recommended. David owns the pick.

1. Derivation doc and three bounds

[Derivation](../../docs/APA_SP2_DELTA_DERIVATION.md) justifies every inequality,
including the inclusive prefix, full-set maximum, ordered partition containment,
upward float32 conversion, and the conditional nature of empirical margins.
The rule derives `delta = ln(1/epsilon) + 2 e_q`, constant per pass.

- Individual skip: `skip_j => w_j < epsilon w_star` if e_q bounds all eligible bulk errors.
- Aggregate skipped probability: `sum_skipped w_j <= min(1, N epsilon w_star)`.
- Softmax perturbation: `||w_tilde-w||_1 <= min(2, 2 N epsilon w_star (1-exp(-e_q)))`.

The additional margin ensures the retained bulk exponential is below
`epsilon exp(s_star-e_q)`, so each skipped key's mass error divided by the
original denominator is at most `epsilon w_star (1-exp(-e_q))`. The document
also bounds the denominator change and absolute output norm, and explains why
relative-Frobenius error is not bounded by this argument.

2. e_q table

[Small blocked table](E_Q_TABLE.md), [JSON blocked report](GPU_BLOCKED.json).

| Bulk bits | D | Max / p99.9 / mean | e_q |
|---|---|---|---|
| 2 | 64 | BLOCKED G2 | not frozen |
| 2 | 128 | BLOCKED G2 | not frozen |
| 4 | 64 | BLOCKED G2 | not frozen |
| 4 | 128 | BLOCKED G2 | not frozen |

Widths are the 2/4-bit rotated Lloyd-Max TurboQuant path in
`tensor_cuda/tensor_cuda/quant.py`, pinned by `test_apa_phase6.py:test_apa_bits`.
These are reconstructed floating keys, not the packed INT4/INT8 GEMM paths.
G2 runs the existing GPU quantizer, measures kernel-order bulk against FP64
exact logits, and independently pins diagnostic dot arithmetic on sampled
keys. Two seeded draws per shape, 32 evenly spaced prefill queries (one decode
query), all eligible keys. The maximum over all calibration observations is
pooled by (bits,D) and rounded upward to fp32; p99.9 and mean are descriptive.
The maximum covers observed errors; it cannot certify unseen inputs.
After all 64 G2 jobs, `run freeze` writes immutable `e_q_table.json` and its SHA,
BEFORE any G3 row. An independent sweep seed checks every held-out eligible
logit for margin exceedances, reporting RED without retuning.

3. Kernel and launcher changes

| File / line | Change |
|---|---|
| [apa_sp2.cuh:3](../../tensor_cuda/src/apa_sp2.cuh#L3) | Host delta derivation and upward fp32 conversion. |
| [apa_sp2.cuh:16](../../tensor_cuda/src/apa_sp2.cuh#L16) | Epsilon/e_q entry delegates to original direct-delta launcher for prefill and split-K. |
| [apa_sp2.cuh:27](../../tensor_cuda/src/apa_sp2.cuh#L27) | Untimed `apa_selective_sp_scores_kernel` diagnostic. |
| [apa_sp2.cuh:95](../../tensor_cuda/src/apa_sp2.cuh#L95) | Diagnostic invokes unchanged baseline stats for counts. |
| [kernels.cu:7505](../../tensor_cuda/src/kernels.cu#L7505) | Include inside the existing SP addition region. |
| [ops.h:19](../../tensor_cuda/include/tc/ops.h#L19), [bindings.cpp:685](../../tensor_cuda/src/bindings.cpp#L685) | Additive declarations and raw epsilon/diagnostic bindings. |
| [apa_sp2.py:22](../../tensor_cuda/tensor_cuda/apa_sp2.py#L22) | SHA-checked frozen margin loader; immutable entries. |
| [apa_sp2.py:72](../../tensor_cuda/tensor_cuda/apa_sp2.py#L72) | Public SP epsilon launcher; lookup by (bits,D), no exact-dot prepass or epsilon default. |
| [__init__.py:1286](../../tensor_cuda/tensor_cuda/__init__.py#L1286) | Additive exports; existing APIs retain their behavior. |

`TC_APA_SP` stays default OFF and requires exactly `1`. The original
`tc._C.apa_selective_attention_sp(..., delta, ...)` remains for SP1 reproduction.
The public `tc.apa_selective_attention_sp(..., epsilon, bulk_bits=..., margins=...)`
requires a frozen table and supports its calibrated fp32 / D64,D128 /
scale=1/sqrt(D) regime. Kq must come from `tc.quantize_sp2_keys(k,bits)`.
All 107 pre-existing CUDA bodies, including SP1/SP1.1, are byte-identical and
hash-pinned. Original receipts and registrations remain unchanged. Split-K
still uses the inherited 2048-key partitions and existing merge/sink handling.
[Source patch](source_additions.patch), [final build](build_final.log),
[build manifest](build/manifest.json).

4. G1 results and G2/G3 lead commands

**61 passed in 3.78s** (52 existing SP1/SP1.1 CPU tests plus 9 SP2 tests).
10,000 random known-error draws exercise skips, mass/perturbation, suffix
extension, partition containment and epsilon nesting. Host scalar/guard
parity and both CPU emulators check the epsilon/direct-delta seam; actual
GPU output/mask bit parity remains in G3. All five registered scalar mutants
were killed (1.0, required >=0.8), without mutating production files.
Freeze/pooling, failed-exit resume logic, strict crossings and prediction
counts have synthetic CPU checks. No synthetic fixture is a real G2 table.
[Final G1 receipt](G1_FINAL.json), [pytest transcript](g1_verified.log),
[runner checks](runner_cpu_checks.json).

From `/mnt/ForgeRealm/Project-Tensor-wt-apa-sp2`:

```bash
timeout --signal=TERM --kill-after=5s 550s bash scripts/apa_sp2_build.sh
timeout --signal=TERM --kill-after=5s 190s bash scripts/apa_sp2_cpu.sh
timeout 60s bash scripts/apa_sp2_lead_gpu.sh list
timeout --signal=TERM --kill-after=5s 590s bash scripts/apa_sp2_lead_gpu.sh resume
timeout 60s bash scripts/apa_sp2_lead_gpu.sh summary
```

Each `resume` executes ONE next job, then returns. Failed jobs stop resume.
[Exact commands for every job](lead_commands.txt) enumerate all 64 G2 classes,
the CPU `run freeze` step, and all 448 G3 class/epsilon rows. Running freeze
before G2 currently raises exactly:
`RuntimeError: BLOCKED: G2 missing/failed eq_b2_prefill_s2048_d64_c0_h4_kv4`.
No epsilon sweep may use a missing or stale table. Each GPU invocation takes
`/tmp/forge-gpu.lock`, waits <=5s for the lease, refuses existing compute PIDs,
uses one device, bounds the worker at 510s, and cools for 30s in foreground
Python while holding the lease. Full run timeout 590s plus <=5s termination
grace. Timeout logs and exit receipts persist. No unbounded sweep loop.

[EPSILON_CURVE.md](EPSILON_CURVE.md) and [curve JSON](epsilon_curve.json) contain
every registered class and grid position, baseline reference rows, crossing
fields and the current blocked state (0/448 rows). G3 times same-session
interleaved old/SP calls, reports actual fractions, full-output dense-fp32
relative Frobenius and max abs, and CUDA-event plus wall speed. Baseline
fraction uses the diagnostic stats/dot decisions, gated against actual
baseline output at unchanged atol=rtol=0.001. Quantization and diagnostics
are outside BOTH timings. Dense timing is omitted to keep each gate bounded.

5. Registration and predictions

Registration SHA256:
`3b8df8bfdd6e3d351395f96ca654bfe9070a26bfaed390d73cc620546a7ff027`.
Parent SP1: `12059d49f39abe9450b34989f06ac5ca8a736d80e3127c9d41c1ed97e650c9fe`.
Parent SP1.1: `382194c1f9d94c146bddcf533b7becadee60fe7d212a2029fabf133243c683b4`.
[Registration](registration.json), [separate amendment](amendment_001.json).

P1/P2/P3/P4 are all **BLOCKED: GPU UNMEASURED**. None is scored HIT/MISS from
CPU substitutes. Conditional mathematical monotonicity passed G1; it does
not establish P2's empirical fraction ordering. The summary scores complete
registered groups, uses strict deviation improvement, and reports missing
coverage explicitly instead of manufacturing a crossing.

6. Deviations, residual risks, RED, process safety and model

RED: the maximum is finite calibration evidence, not a universal error bound.
The one-sided weight/mass theorem assumes a valid full-key bound and ideal
mixed logits; float dot, expf and merge errors remain separate numerical
risks. At N=32768 and epsilon=1e-3, N epsilon=32.768; at epsilon=1e-4 it is
3.2768, so the worst-case probability bound caps at 1 and is useless.
No GPU score diagnostic, runtime parity, speed or error curve was executed
on this seat. The wrapper therefore remains unvalidated on GPU. No model
quality claim or epsilon recommendation follows from this work.

Execution refinements are explicit: G2 samples prefill queries while G3
checks every output/logit; each epsilon gets its own bounded class invocation;
only fp32 gets a calibrated table; dense timing is omitted. Additional
checks are in the separate amendment. A commands-file audit initially
mistook a comment ending `checkout.` for a target (`AssertionError`); the
checker was corrected to parse command lines, with the cause and passing
rerun retained in `runner_cpu_checks.json`. No runtime change or tolerance
change was used to hide a failure. Author tests are baseline verification;
House Rules section 8 blind verification is the lead's responsibility.

All writes, builds and runtime imports used this worktree. No git, subagents,
GPU workloads, shell background jobs, services, other-worktree edits or
foreign process termination. All compute ran as timeout-bounded foreground
children; completed build/test children are reaped. The latest dispatch base
is cd87939; embedded order says 62a982e, recorded without git inspection.
[Continued ledger](../../docs/APA_SP1_LEDGER.md),
[execution metadata](execution_metadata.json).

Actual model: **gpt-6-astra**, reasoning effort **xhigh**, verified from the
current dispatch log header `logs/apa_sp2_astra_r1.log:5-12`.
