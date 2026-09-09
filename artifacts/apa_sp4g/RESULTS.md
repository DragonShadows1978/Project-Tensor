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

---

## Archived A6 report (unchanged snapshot)

# APA-SP4G amendment 6

**RED residuals. D/A per-call exactness: PASS by lead ruling; literal stated bounds: RED.** C/E are released from the historical0.005 PPL dependency by amendment6 and scheduled behind the new propagation stop. A6 GPU measurements are completed for the diagnostic.

**Evidence discrepancy:** Receipt c1 max_abs=3.528594970703125e-5; b15 layer5 max_abs=3.910064697265625e-5 / relF=9.043649367824469e-7 exceed the stated bounds. Lead PASS is recorded as a ruling, never as literal threshold compliance. Bounds are not relaxed.

The recorded PASS is a lead interpretation of finite same-input agreement. It does not certify every call, bitwise equality, or compliance with the stated scalar bounds. No threshold or old receipt was edited.

## Per-call D/A exactness

Evidence class: prior card per-call diagnostics, validated receipt and payload provenance; fp32 SP versus standard on identical input values. Max-abs limit3.5e-5; relF limit7e-7.

| Cell | Layer | L | S_all | fp32 max-abs | fp32 relF | Literal bounds |
|---|---:|---:|---:|---:|---:|---|
| [diag_a3_call_l05_c0](jobs_a3/diag_a3_call_l05_c0.json) | 5 | 512 | 512 | 3.0517578125e-05 | 4.91997946485157e-07 | PASS |
| [diag_a3_call_l05_c1](jobs_a3/diag_a3_call_l05_c1.json) | 5 | 511 | 1023 | 3.52859497070312e-05 | 6.59743943485455e-07 | RED |
| [diag_a5_call_l05_b00](jobs_a5/diag_a5_call_l05_b00.json) | 5 | 64 | 1087 | 2.6702880859375e-05 | 6.98806004193954e-07 | PASS |
| [diag_a5_call_l05_b15](jobs_a5/diag_a5_call_l05_b15.json) | 5 | 64 | 2047 | 3.91006469726562e-05 | 9.04364936782447e-07 | RED |
| [diag_a5_call_l47_b15](jobs_a5/diag_a5_call_l47_b15.json) | 47 | 64 | 2047 | 1.02519989013672e-05 | 4.31577308095211e-07 | PASS |

## Model perplexity — registered numerical-path floor ±2.56 PPL

Measured window0: A bf16=52.4804870852055, A32=49.9211789381388; |A32−A|=2.55930814706673, rounded floor±2.56. This is a deterministic numerical-path sensitivity measurement, not a statistical confidence interval. The lead applies this floor to every model-level arm difference here. It is **extrapolated, not independently measured**, for windows1–3, their pooled NLL, and8192. An outside-floor gap alone does not establish a robust gain.

**Gemma4 QAT INT4; bf16 production arms; APA on the8 global layers only;40 sliding layers unchanged. Every PPL comparison uses the ±2.56 floor.** Short rows score1024 targets each; pooled2048 uses4096 total targets;8192 scores512.

| Row | A bf16 | B | C | D | E | P2: C − B against ±2.56 |
|---|---:|---:|---:|---:|---:|---|
| 2048_w0 | 52.48048709 | 49.41041564 | 51.88388607 | 53.47239047 | 53.16165234 | +2.47347043; C − B inside the floor; not resolvable on this model |
| 2048_w1 | 537.97325533 | 571.49025268 | 609.35601969 | 582.57631198 | 525.40126632 | +37.86576701; C − B outside the floor; outside the registered floor |
| 2048_w2 | 362.99618674 | 374.40827539 | 371.11944625 | 355.94325393 | 359.97875530 | -3.28882914; C − B outside the floor; outside the registered floor |
| 2048_w3 | 73.45901658 | 74.71952840 | 73.60051639 | 74.95015168 | 75.18971208 | -1.11901201; C − B inside the floor; not resolvable on this model |
| 2048 | 165.64427445 | 167.64929211 | 171.42515228 | 169.78875584 | 165.81782708 | +3.77586017; C − B outside the floor; outside the registered floor |
| 8192 | 38.86389187 | 38.56135027 | 39.43323066 | 39.38228832 | 37.99512343 | +0.87188039; C − B inside the floor; not resolvable on this model |

Validated PPL receipt paths/hashes and **every available arm pair** with its floor classification are in `PPL_TABLE_A6.json`. No within-floor difference is labelled a win or loss.

| Row | B − A | D − A | Interpretation against ±2.56 |
|---|---:|---:|---|
| 2048_w0 | -3.07007145 | +0.99190338 | B−A: outside the registered floor; D−A: not resolvable on this model |
| 2048_w1 | +33.51699735 | +44.60305665 | B−A: outside the registered floor; D−A: outside the registered floor |
| 2048_w2 | +11.41208865 | -7.05293281 | B−A: outside the registered floor; D−A: outside the registered floor |
| 2048_w3 | +1.26051182 | +1.49113510 | B−A: not resolvable on this model; D−A: not resolvable on this model |
| 2048 | +2.00501766 | +4.14448138 | B−A: not resolvable on this model; D−A: outside the registered floor |
| 8192 | -0.30254160 | +0.51839645 | B−A: not resolvable on this model; D−A: not resolvable on this model |

Historical D32=53.4723904674738, A32=49.9211789381388; D32−A32=+3.55121152933498: outside the registered floor. All144 fp32 pins complete. D32 PPL is bit-identical to D bf16 PPL on window0 (difference0, not resolvable on this model); this does not certify bit-identical logits. Historical gate≤0.005 stays **RED**, annotated **inapplicable under the lead numerical-path ruling**. Error retained verbatim: `A4_D32_A32_EXACTNESS_FAILED; STOP, lead investigates`. Not claimed fixed: historical model-PPL exactness.

## Propagation receipt and predictions

Cell: `diag_a6_propagation_A32_vs_A_2048_w0`; state: PASS, AMPLIFIED; None

**Lead prediction:** Same amplification profile, relF growing to ~0.1 by layer29; if flat stop, the lead was wrong.

**Seat prediction:** Expect amplification too: layer29 relF roughly 0.05 to 0.2 and at least 5 times layer5. Exact A/D magnitudes need not repeat; this is a prediction, not a measurement.

AMPLIFIED if L29 relF >= .05 AND >= 5*L5; FLAT if L29 <= 2*L5; otherwise INCONCLUSIVE. Flat or inconclusive => RED STOP all C/E, no retry or widened fork.

One new A32 load; compare against saved A bf16 residuals from `jobs_a5/diag_a4_propagation_A_2048_w0.json`. Native fp32 Q/K/Kq/V/output pins, bf16 cast before o_proj, complete144-call schedule, and A32 PPL reproduction are mandatory. All2047 query rows after each complete global block and final norm; sum squared errors and reference norms across blocks.

| Layer | D−A bf16 relF | D−A max-abs | A32−A relF | A32−A max-abs |
|---|---:|---:|---:|---:|
| 5 | 0.00482106304 | 0.25 | 0.00482107767 | 0.25 |
| 11 | 0.00743735688 | 0.25 | 0.00753248699 | 0.25 |
| 17 | 0.0103417778 | 2.45117188 | 0.0112727392 | 3.1328125 |
| 23 | 0.0382708158 | 67.5 | 0.0310998025 | 70 |
| 29 | 0.124512706 | 40.6875 | 0.144096183 | 37.375 |
| 35 | 0.128977027 | 32.125 | 0.142266821 | 32.6875 |
| 41 | 0.144904834 | 24.5 | 0.156331575 | 15.453125 |
| 47 | 0.0736480457 | 3.46875 | 0.0700486792 | 2.71875 |
| final_norm | 0.119381166 | 55.5 | 0.117511898 | 56.3125 |

Evidence class: prior card residual diagnostic, `jobs_a5/diag_a4_propagation_2048_w0.json`. D/A L29-to-L5 relF ratio=25.827. The approximately0.3% figure describes attention-output differences; post-block L5 residual relF is0.4821%. Observed amplification is a mechanism lead, not proof of chaos or a complete causal explanation of PPL.

## C/E cells, estimates and commands

Calibration target is B's measured0.151605081488344, rounded0.152, on the actual window0 global-layer pair population; tolerance±0.01. Twelve registered bisection trials; after a match, later trials carry its immutable result without loading the model. C uses one frozen delta. E uses the inherited conditional empirical bound from all32 B/C layer-margin receipts. Neither E’s finite e_q sample nor G2/G3 rows establish model quality by themselves.

Run **each command separately in the foreground**, in `lead_commands.txt` order. `resume` runs exactly one next cell. Historical0.005 does not block this DAG; missing/flat/inconclusive A6 propagation does. Model-cell estimates include75–130s load, based on earlier75–126s receipts; estimates are not new measurements. GPU calls add up to20s lease wait and30s cooldown. Worker cooperative rail285s, call budget599s.

| Cell | Worker estimate (s) | State |
|---|---:|---|
| `diag_a6_propagation_A32_vs_A_2048_w0` | 90–275 | PASS |
| `trial_a6_00` | 90–275 | PASS |
| `trial_a6_01` | 90–275 | PASS |
| `trial_a6_02` | 90–275 | PASS |
| `trial_a6_03` | 90–275 | PASS |
| `trial_a6_04` | 90–275 | PASS |
| `trial_a6_05` | 90–275 | PASS |
| `trial_a6_06` | 90–275 | PASS |
| `trial_a6_07` | 90–275 | PASS |
| `trial_a6_08` | 90–275 | PASS |
| `trial_a6_09` | 90–275 | PASS |
| `trial_a6_10` | 90–275 | PASS |
| `trial_a6_11` | 90–275 | PASS |
| `freeze_a6` | 1–15 | PASS |
| `ppl_a6_C_2048_w0` | 90–275 | PASS |
| `ppl_a6_C_2048_w1` | 90–275 | PASS |
| `ppl_a6_C_2048_w2` | 90–275 | PASS |
| `ppl_a6_C_2048_w3` | 90–275 | PASS |
| `ppl_a6_C_2048` | 1–30 | PASS |
| `ppl_a6_C_8192` | 90–275 | PASS |
| `margin_a6_C_2048_l05` | 10–90 | PASS |
| `margin_a6_C_2048_l11` | 10–90 | PASS |
| `margin_a6_C_2048_l17` | 10–90 | PASS |
| `margin_a6_C_2048_l23` | 10–90 | PASS |
| `margin_a6_C_2048_l29` | 10–90 | PASS |
| `margin_a6_C_2048_l35` | 10–90 | PASS |
| `margin_a6_C_2048_l41` | 10–90 | PASS |
| `margin_a6_C_2048_l47` | 10–90 | PASS |
| `margin_a6_C_8192_l05` | 60–270 | PASS |
| `margin_a6_C_8192_l11` | 60–270 | PASS |
| `margin_a6_C_8192_l17` | 60–270 | PASS |
| `margin_a6_C_8192_l23` | 60–270 | PASS |
| `margin_a6_C_8192_l29` | 60–270 | PASS |
| `margin_a6_C_8192_l35` | 60–270 | PASS |
| `margin_a6_C_8192_l41` | 60–270 | PASS |
| `margin_a6_C_8192_l47` | 60–270 | PASS |
| `eq_a6` | 1–30 | PASS |
| `ppl_a6_E_2048_w0` | 90–275 | PASS |
| `ppl_a6_E_2048_w1` | 90–275 | PASS |
| `ppl_a6_E_2048_w2` | 90–275 | PASS |
| `ppl_a6_E_2048_w3` | 90–275 | PASS |
| `ppl_a6_E_2048` | 1–30 | PASS |
| `ppl_a6_E_8192` | 90–275 | PASS |
| `decode_a6_C_2048` | 90–275 | PASS |
| `decode_a6_C_8192` | 90–275 | PASS |

Exact command per cell and dependency reason: `GPU_BLOCKED_A6.json`; complete command list: `lead_commands.txt`.

## CPU gates, fingerprints and RED residuals

Ruling019 SHA256 `ac2cc1519634cd74e0540790dce7b5aa2a4900a8922d52c6a1f1d8305fa130b3`. Original registration unchanged: `099a8bd9e1bb94909a81c110521d2d3a9d642d5fac27d49f6820b97cb3fbbd1e`.
`amendment_020_a6_fingerprint.json` SHA256 `fe32afcfd5383e9cc395f690e6078ea2b3406f96d323bf5faa94e6138a3daca4`.
`CPU_GATES_A6.json` SHA256 `356a36a0f3b1b01b0a748e373f8c92fdd4e4d4c6078b87bf9aef81c77b35222c`.

Author CPU baseline: 148 passed, 0 failed, 0 skipped; mutations 8/8, threshold0.80. Blind review is lead-owned and UNRUN. CPU doubles validate harness wiring, never actual GPU numerics.

Not claimed fixed: original0.005 PPL gate, literal3.5e-5/7e-7 bound discrepancy, unmeasured A6 propagation and C/E quality while pending; no extension of the floor’s measured scope. Product, kernel, adapter, model, order, historical execution files and receipts preserved byte-for-byte.

Process safety: no git, subagents, background jobs/waits, signals/kills, services or model writes. No A6 GPU worker or model load in this dispatched seat. CUDA visibility is recorded in `a6_device_visibility.json`. Cooperative deadlines cannot forcibly bound a hung native call without violating no-kill; clean decode checks before/after measurement only, preserving the unwrapped timed path. Each lead command is planned below10minutes; no hard no-kill wall-time guarantee is claimed.

Seat: **gpt-6-astra, reasoning xhigh**, live session header `logs/apa_sp4g_a6_r1.log`. Model under test: **Gemma-4-12B-it QAT q4_0 exact (symmetric-8 g32)**; bf16 engine, fp32 global-attention diagnostic only.

## Prior art

SP4G A2/A4/A5 and June Gemma port/floor (2026) provide the precision seam, residual capture, native-mask replay, calibration and clean decode. A6 adds ruling/floor reporting, combines A32 with that capture, and replaces successor dependencies. No new attention algorithm or novelty claim.

[Haber and Ruthotto (2017), Stable Architectures for Deep Neural Networks](https://arxiv.org/abs/1705.03341) provides dynamical-system stability context; [Higham and Mary (2022), Mixed precision algorithms in numerical linear algebra](https://doi.org/10.1017/S0962492922000022) provides mixed-precision error context. Primary abstracts/metadata verified this seat. Neither establishes Gemma QAT rounding chaos or this PPL floor; that connection is an experimental inference to test.

Inherited prior art: BLASST/Yuan (2025/2026) running-max selection, ThriftAttention/Sharratt (2026) weight-sensitive precision, FlashAttention2/Dao (2023) online softmax, TurboQuant/Zandieh (2025) quantization; unverified this seat — lead to check arXiv2512.12087,2605.23081,2307.08691,2504.19874. Classical bisection, Frobenius norms, NIST SHA256 (2001), Make/Feldman (1979) dependency DAGs; DeMillo/Lipton/Sayward (1978) mutation testing, unverified — lead to check Hints on Test Data Selection. No prior art known to me for a distinct new method introduced here.
