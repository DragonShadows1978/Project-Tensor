# APA-SP4G amendment 7 — long ceiling

**RED — ceiling unmeasured/pending.** 0/21 new cells have validated terminal receipts. CPU-only dispatched seat; lead runs the GPU. Old receipts and rails are unchanged. G2/G3 rows establish nothing about model quality by themselves.

## Cells, rails and estimates

Each cell has a 1,500 s cooperative worker rail and a 1,560 s outer budget, including a ≤20 s foreground flock wait and 30 s cooldown. Run each command separately, A ascending first, then B, then C. `resume` executes exactly one cell. After the first CUDA OOM in an arm, larger registered cells write inferred NON_FIT_AFTER_OOM receipts without a lease or model load. RAIL means unknown fit; it does not stop later rungs or authorize a retry. Unexpected errors block descendants.

`estimate = load_8192 + prefill_8192 × (S/8192)^2`. This is the requested quadratic extrapolation anchored to measured times, not a measured S² exponent or new timing result. The original A 4096→8192 prefill ratio is about 2.03. C uses the prefill component of the clean 8192 decode receipt (delta=3); scoring, captures and decode are excluded. Estimates include one fresh load (~75–76 s measured); outer planning adds 50 s. Instrumentation adds unmeasured overhead.

| Arm | 8192 anchor | Load s | Prefill s |
|---|---|---:|---:|
| A | [ceiling_A_8192.json](jobs_a1/ceiling_A_8192.json) | 75.189820 | 105.246930 |
| B | [ceiling_B_8192.json](jobs_a1/ceiling_B_8192.json) | 75.917528 | 113.353087 |
| C | [decode_a6_C_8192.json](jobs_a6/decode_a6_C_8192.json) | 75.728816 | 118.808985 |

| Cell | Extrapolated worker s | Prefill s | KV MiB (global + fixed sliding) | State |
|---|---:|---:|---:|---|
| `ceiling_long_A_16384` | 496.18 | 420.99 | 256 + 319.6875 | READY_GPU |
| `ceiling_long_A_24576` | 1022.41 | 947.22 | 384 + 319.6875 | BLOCKED |
| `ceiling_long_A_32768` | 1759.14 | 1683.95 | 512 + 319.6875 | BLOCKED |
| `ceiling_long_A_49152` | 3864.08 | 3788.89 | 768 + 319.6875 | BLOCKED |
| `ceiling_long_A_65536` | 6810.99 | 6735.80 | 1024 + 319.6875 | BLOCKED |
| `ceiling_long_A_98304` | 15230.75 | 15155.56 | 1536 + 319.6875 | BLOCKED |
| `ceiling_long_A_131072` | 27018.40 | 26943.21 | 2048 + 319.6875 | BLOCKED |
| `ceiling_long_B_16384` | 529.33 | 453.41 | 256 + 319.6875 | BLOCKED |
| `ceiling_long_B_24576` | 1096.10 | 1020.18 | 384 + 319.6875 | BLOCKED |
| `ceiling_long_B_32768` | 1889.57 | 1813.65 | 512 + 319.6875 | BLOCKED |
| `ceiling_long_B_49152` | 4156.63 | 4080.71 | 768 + 319.6875 | BLOCKED |
| `ceiling_long_B_65536` | 7330.52 | 7254.60 | 1024 + 319.6875 | BLOCKED |
| `ceiling_long_B_98304` | 16398.76 | 16322.84 | 1536 + 319.6875 | BLOCKED |
| `ceiling_long_B_131072` | 29094.31 | 29018.39 | 2048 + 319.6875 | BLOCKED |
| `ceiling_long_C_16384` | 550.96 | 475.24 | 256 + 319.6875 | BLOCKED |
| `ceiling_long_C_24576` | 1145.01 | 1069.28 | 384 + 319.6875 | BLOCKED |
| `ceiling_long_C_32768` | 1976.67 | 1900.94 | 512 + 319.6875 | BLOCKED |
| `ceiling_long_C_49152` | 4352.85 | 4277.12 | 768 + 319.6875 | BLOCKED |
| `ceiling_long_C_65536` | 7679.50 | 7603.78 | 1024 + 319.6875 | BLOCKED |
| `ceiling_long_C_98304` | 17184.22 | 17108.49 | 1536 + 319.6875 | BLOCKED |
| `ceiling_long_C_131072` | 30490.83 | 30415.10 | 2048 + 319.6875 | BLOCKED |

## Both prediction sets (registered before CPU gates)

**Lead predictions:**

1. Standard OOMs between 12K and 16K (16 heads x S squared bf16 scores on global layers = 8 GB at 16K on top of 6.8 GB weights).
2. Two-pass and single-pass reach 32K with resident under 10 GB; wall is time, not memory, until at least 64K.
3. Single-pass reaches at least one rung further than two-pass because it has no O(S) bulk/rank/recon transients.

**Seat predictions:**

1. All three arms fit 16K; expect 24K to complete too. Standard is not expected to OOM at 16K: June adaptive queries reach a 64-row floor, so its score tensor there is 32 MiB, not 8 GiB.
2. At 32K all arms would plausibly reside below 10 GiB if completed, but registered quadratic timing predicts RAIL there and above. KV alone is 512 MiB global plus 319.6875 MiB sliding at 32K. Memory may remain feasible at 64K; no measured fit claim.
3. I do not predict a guaranteed extra rung for C. Both B and C use the same chunked reconstructed Kq, and B already streams attention without full bulk/rank score matrices. C may save kernel scratch but this does not remove shared Kq reconstruction.

These are predictions/reasoning. The adapter uses `PREFILL_CHUNK=512` and an adaptive 64-row floor. At 16K, the late standard score tensor is 16×64×16384×2 = 32 MiB; a single-shot 8 GiB score tensor is not allocated by this protocol. The B fused path already streams attention; B and C both construct Kq through the same June chunked quantizer. Source: `/mnt/ForgeRealm/GraftRepository/core/gemma4_tc.py:689–712,724–742,843–889`.

## Ceiling measurements — Gemma-4-12B-it QAT q4_0, bf16 KV/compute, global layers only

Prefill-only from the pinned prefix; no scoring/decode/capture arrays; `Model.prefill` invokes the same June model entry used by PPL context feeding. Pooling ON before load; B bulk4/r=0.15; C frozen delta=3.0; `apa_min_context=0`; KV quantization OFF. Logical global cache is 8 layers × K/V × 1 head × 512 × 2 bytes = 16,384 bytes/token. Sliding tuple caches retain 1023 rows = 335,216,640 bytes (319.6875 MiB), fixed for this grid; capacity upper bound 320 MiB. Kq, cache copies, RoPE, weights and allocator retention are separate from logical KV payload.

| Cell | Outcome | Worker wall s | Peak resident MiB | Sampled peak lower bound MiB | Pool reserved high MiB | Completed tokens |
|---|---|---:|---:|---:|---:|---:|
| `ceiling_long_A_16384` | READY_GPU | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNRUN |
| `ceiling_long_A_24576` | BLOCKED | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNRUN |
| `ceiling_long_A_32768` | BLOCKED | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNRUN |
| `ceiling_long_A_49152` | BLOCKED | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNRUN |
| `ceiling_long_A_65536` | BLOCKED | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNRUN |
| `ceiling_long_A_98304` | BLOCKED | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNRUN |
| `ceiling_long_A_131072` | BLOCKED | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNRUN |
| `ceiling_long_B_16384` | BLOCKED | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNRUN |
| `ceiling_long_B_24576` | BLOCKED | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNRUN |
| `ceiling_long_B_32768` | BLOCKED | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNRUN |
| `ceiling_long_B_49152` | BLOCKED | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNRUN |
| `ceiling_long_B_65536` | BLOCKED | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNRUN |
| `ceiling_long_B_98304` | BLOCKED | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNRUN |
| `ceiling_long_B_131072` | BLOCKED | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNRUN |
| `ceiling_long_C_16384` | BLOCKED | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNRUN |
| `ceiling_long_C_24576` | BLOCKED | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNRUN |
| `ceiling_long_C_32768` | BLOCKED | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNRUN |
| `ceiling_long_C_49152` | BLOCKED | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNRUN |
| `ceiling_long_C_65536` | BLOCKED | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNRUN |
| `ceiling_long_C_98304` | BLOCKED | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNRUN |
| `ceiling_long_C_131072` | BLOCKED | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNRUN |

An exact resident peak is reported only from NVML own-PID accounting `maxMemoryUsage`, when already supported/enabled. If unavailable, its value is null and `peak_status=RED_PEAK_UNAVAILABLE`; synchronous boundary samples are labelled a lower bound. No accounting mode/device settings are changed. CUDA pool used/reserved high-water counters cover the pool only. Subtracting the logical KV payload from a process peak still includes weights, scratch, allocator retention and CUDA overhead; it is not pure transient usage. OOM/RAIL receipts retain partial measurements and full-S theoretical KV; partial KV is not relabelled a completed S cache.

## June 8 GB RTX 3070 context

| June path | Prefill evidence |
|---|---|
| bf16 KV/compute, QAT INT4 body | ~10–11K solid; 12K ragged (7805 MiB in one run, OOM in ladder); 16K OOM |
| qv / INT8 V | 12K solid at 7802 MiB; 16K OOM |

Evidence class: external local June port ledger, `/mnt/ForgeRealm/GraftRepository/docs/GEMMA4_PORT_LEDGER.md:49–55`, SHA256 `01c0903c65422ced60e857a0c37b934ec4902b4d5a13eb8511586f349fa77e99`. Different GPU/configuration; not an A7 measurement. Order wording “bf16 weights” is a terminology discrepancy: the June resident body was QAT INT4; full 12B bf16 weights do not fit in 8 GB.

## Fingerprints, CPU gates and RED

Registration021 SHA256 `e13f685f3c89fa9430b643aca320a4e93247a83fcfbea9456892c5ec50fbca8c`. Original registration SHA256 `099a8bd9e1bb94909a81c110521d2d3a9d642d5fac27d49f6820b97cb3fbbd1e`.
Fingerprint022 SHA256 `4c0b8738d42d136138ea3a0894e1f191c08902342356f9ce98d3b9196bd6b797`.
Author CPU suite: **178 passed, 0 failed, 0 skipped**; mutation kills 8/8, threshold0.80. CPU doubles check harness semantics, not GPU numerics. Blind verification is lead owned and UNRUN.

Current device probe: `{"cudaGetDeviceCount": 100, "device_count": 0, "error": "no CUDA-capable device is detected", "scope": "current dispatched sandbox; zero A7 GPU workers/model loads", "status": "NO_CUDA", "utc": "2026-09-08T03:35:30Z"}`.

Exact commands and blocking dependencies: `lead_commands.txt`, `GPU_BLOCKED_A7.json`. Not claimed fixed: unmeasured memory ceiling, time censoring, historical exactness RED, or unavailable exact resident peak on devices without accounting. Archived A6 results below remain historical; amendment7 makes no quality claim.

## Process safety and seat

No git, subagents, background jobs/waits, process signals/kills, service edits, product/kernel edits or model writes. One foreground cell per flock lease. CUDA OOM is distinguished from host MemoryError, worker exit137/124 and arbitrary errors. Deadline checks are cooperative, including before/after load and at block/chunk boundaries. A hung native operation cannot be forcibly bounded under no-kill. The outer elapsed receipt includes cooldown and flags any overrun; this is not a hard no-kill wall-time guarantee.

Seat: **gpt-6-astra / reasoning xhigh**, live header `logs/apa_sp4g_a7_r1.log`. Model under test: **Gemma-4-12B-it QAT q4_0 exact (symmetric-8 g32)**. Worktree/head is the lead-provided `apa-sp4g` at `29882ae`; no git command used to verify it.

## Prior art

June Gemma port/floor and SP3/SP4G (2026) supply adaptive chunks, pooling, attention dispatch and foreground receipts. A7 adds long-ceiling registration, dimensional KV reporting and scoped telemetry; no new attention algorithm. [BLASST, Yuan et al. (2025/2026)](https://arxiv.org/abs/2512.12087) supplies inherited running-max softmax selection; [FlashAttention-2, Dao (2023)](https://arxiv.org/abs/2307.08691) supplies inherited online-softmax implementation context. Primary abstracts checked this seat. ThriftAttention/Sharratt (2026), weight-sensitive precision, and TurboQuant/Zandieh (2025), key quantization, are inherited unchanged; unverified — lead to check arXiv2605.23081 and2504.19874. [NVIDIA NVML accounting](https://docs.nvidia.com/deploy/nvml-api/structnvmlAccountingStats__t.html) and local CUDA12.6/NVML headers (2024) supply process accounting and pool-counter ABIs. No new profiler algorithm.

Make/Feldman (1979) dependencies, SHA256/NIST (2001) provenance, classical dimensional analysis and quadratic extrapolation, DeMillo/Lipton/Sayward (1978) mutation tests are reused. Historical citations unverified this seat — lead to check Make a program for maintaining computer programs; FIPS180; Hints on Test Data Selection. No prior art known to me for a distinct new method introduced here; no novelty claim.
