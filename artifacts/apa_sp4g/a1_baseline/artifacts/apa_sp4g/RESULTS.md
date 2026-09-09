# APA-SP4G results

**CPU PASS_CPU_ONLY; GPU BLOCKED: no device in this seat; zero GPU cells executed.** No Gemma PPL, margin, ceiling or decode result is claimed without a valid current receipt.

Registration SHA: `099a8bd9e1bb94909a81c110521d2d3a9d642d5fac27d49f6820b97cb3fbbd1e`. Model reference A is the engine's QAT INT4 Gemma-4-12B-it (exact q4_0 symmetric-8 group32 import, BF16 compute). A 12B BF16 weight set is roughly 24GB and does not fit the 12GB card; no torch or full-BF16 model reference was loaded. This tests attention changes inside this engine; it cannot establish parity to a BF16 model.

## Adapter facts and PROTOCOL-G

Source inspection: `gemma4_tc.py:568/658` gates APA by strict `S_all > apa_min_context`; EVERY B/C/D/E cell sets `apa_min_context=0` and `fast_max_seq=0`. Thus the short windows actually exercise fused APA. A stays standard. B is reconstructed-key `apa_selective_attention` at lines 609/712, bulk 4/r=.15, with INT4/GEMM opt-ins off. C/D/E dispatch from that SAME binding with Q[B,16,L,512] and K/Kq/V[B,1,S,512], BF16, scale=1. No V padding or head expansion difference between arms. Hidden activations can differ as attention changes.

Both SP launchers already instantiate 512 (kernels.cu:7496, apa_sp1_1.cuh:179) and map 16 query heads to KV=1. Tests: `test_d512_mqa_prefill_contract_and_dense_pin`, `test_d512_mqa_splitk_contract_and_dense_pin`, `test_compiled_d512_mqa_flag_device_scalar_guards`; native numerical receipt will be `kernel512`. No production source or kernel-body edits.

`KVRing.quantized_keys` at gemma4_tc.py:353-388 quantizes only `[kq_count:count)`, in 512-row chunks at cold start and ONE new row thereafter. CPU behavior pin: `test_live_gemma_kq_cache_only_quantizes_new_rows`. This path retains the June incremental fix. Prefill still quantizes whole prefixes (in 2048-row chunks above 4096); that is distinct from decode. Gemma K/V shared projection does not imply equal stored tensors: K is normalized/roped and V uses scale-free normalization without RoPE.

Offline wikitext-2-raw-v1 test: 4358 Arrow rows, newline join exactly as June floor, no fallback. Default -it tokenizer, no chat template or per-window BOS. **292282 tokens**, canonical little-endian int64 SHA `8bb85a61060d4221fef55f0b134914c55e6110ffbfd10913aed54e99f130d1cb`; `.npy` SHA `9d8e686bb23a738847c75d9e0aaf5dc8482c243944e4bb71122f1cdc31da0e0d`. `tokens.npy` and all tokenizer/corpus inputs are pinned in registration.

One feeding scheme for all arms: fresh caches per window; prefix of S−scored−1 tokens through June adaptive PREFILL_CHUNK=512, then 64-query cached blocks. Logits at positions S−scored−1 through S−2 predict targets S−scored through S−1. Short: four consecutive 2048-token windows at offsets 0/2048/4096/6144, exactly 1024 targets each (4096 total). Long: prefix 0, last 512 targets within input at 8192/16384/32768. FP64 log-softmax; aggregate exp(total NLL / total targets), never average PPL. June floor starts one query later and actually scores 1023; the mandated1024/fp64 correction is pre-registered.

C calibration is SP3-style prefix 0 at S2048, ALL eligible global-layer pairs, separate from the four-window PPL population. First of at most 12 native-C trials within ±0.01 of B freezes ONE delta for every layer, scored window and length. PPL fractions are reported independently; no hidden rematching. E uses epsilon=.01 and upward-rounded log(100)+2eq, where eq is the maximum actual SP-bulk versus FP64 exact real-key error over B/C and S2048/8192. This is a finite calibration maximum and a conditional bound, not a theorem about unseen keys.

## Model perplexity — only the eight global layers change; forty sliding layers see at most 1024 valid keys and never APA

A is the reference. Differences originate in the global-layer attention changes and propagate through subsequent layers.

| Input S | Arm | Status | Targets | PPL | PPL minus A | Global refine fraction |
|---:|---|---|---:|---:|---:|---:|
| 2048 | A | UNRUN | — | — | — | — |
| 2048 | B | UNRUN | — | — | — | — |
| 2048 | C | UNRUN | — | — | — | — |
| 2048 | D | UNRUN | — | — | — | — |
| 2048 | E | UNRUN | — | — | — | — |
| 8192 | A | UNRUN | — | — | — | — |
| 8192 | B | UNRUN | — | — | — | — |
| 8192 | C | UNRUN | — | — | — | — |
| 8192 | D | UNRUN | — | — | — | — |
| 8192 | E | UNRUN | — | — | — | — |
| 16384 | A | UNRUN | — | — | — | — |
| 16384 | B | UNRUN | — | — | — | — |
| 16384 | C | UNRUN | — | — | — | — |
| 16384 | D | UNRUN | — | — | — | — |
| 16384 | E | UNRUN | — | — | — | — |
| 32768 | A | UNRUN | — | — | — | — |
| 32768 | B | UNRUN | — | — | — | — |
| 32768 | C | UNRUN | — | — | — | — |
| 32768 | D | UNRUN | — | — | — | — |
| 32768 | E | UNRUN | — | — | — | — |

Short in-model D/A gate: UNRUN. Tolerance remains 0.005; RED blocks C calibration. Long D/A differences remain independently visible above.

## G2 global margins — kernel sweep on real model activations

All causal pairs, no sampling. Bands preserve absolute causal alignment and must replay the captured output bitwise. Error percentiles are nearest-rank over the entire layer population; no averaging percentiles. Unrefined mass is the mean per-query exact softmax mass on skipped keys; relative weight is max exp(exact_skipped − row_max_exact). These are different quantities.

| S | Arm | Layer | Status | Mean error | p99 | p99.9 | Max error | Mean unrefined mass | Max skipped relative weight | Fraction |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 2048 | B | 5 | UNRUN | — | — | — | — | — | — | — |
| 2048 | B | 11 | UNRUN | — | — | — | — | — | — | — |
| 2048 | B | 17 | UNRUN | — | — | — | — | — | — | — |
| 2048 | B | 23 | UNRUN | — | — | — | — | — | — | — |
| 2048 | B | 29 | UNRUN | — | — | — | — | — | — | — |
| 2048 | B | 35 | UNRUN | — | — | — | — | — | — | — |
| 2048 | B | 41 | UNRUN | — | — | — | — | — | — | — |
| 2048 | B | 47 | UNRUN | — | — | — | — | — | — | — |
| 2048 | C | 5 | UNRUN | — | — | — | — | — | — | — |
| 2048 | C | 11 | UNRUN | — | — | — | — | — | — | — |
| 2048 | C | 17 | UNRUN | — | — | — | — | — | — | — |
| 2048 | C | 23 | UNRUN | — | — | — | — | — | — | — |
| 2048 | C | 29 | UNRUN | — | — | — | — | — | — | — |
| 2048 | C | 35 | UNRUN | — | — | — | — | — | — | — |
| 2048 | C | 41 | UNRUN | — | — | — | — | — | — | — |
| 2048 | C | 47 | UNRUN | — | — | — | — | — | — | — |
| 8192 | B | 5 | UNRUN | — | — | — | — | — | — | — |
| 8192 | B | 11 | UNRUN | — | — | — | — | — | — | — |
| 8192 | B | 17 | UNRUN | — | — | — | — | — | — | — |
| 8192 | B | 23 | UNRUN | — | — | — | — | — | — | — |
| 8192 | B | 29 | UNRUN | — | — | — | — | — | — | — |
| 8192 | B | 35 | UNRUN | — | — | — | — | — | — | — |
| 8192 | B | 41 | UNRUN | — | — | — | — | — | — | — |
| 8192 | B | 47 | UNRUN | — | — | — | — | — | — | — |
| 8192 | C | 5 | UNRUN | — | — | — | — | — | — | — |
| 8192 | C | 11 | UNRUN | — | — | — | — | — | — | — |
| 8192 | C | 17 | UNRUN | — | — | — | — | — | — | — |
| 8192 | C | 23 | UNRUN | — | — | — | — | — | — | — |
| 8192 | C | 29 | UNRUN | — | — | — | — | — | — | — |
| 8192 | C | 35 | UNRUN | — | — | — | — | — | — | — |
| 8192 | C | 41 | UNRUN | — | — | — | — | — | — | — |
| 8192 | C | 47 | UNRUN | — | — | — | — | — | — | — |


E calibration: UNRUN.

## G3 ceilings and clean decode — kernel sweep / in-model timing

EVERY arm retains June adaptive prefill from the same pinned prefix. FIT requires completed synchronized prefill; explicit allocation errors are OOM; a timeout is RAIL with unknown fit. Resident is an own-PID snapshot after load/prefill or at a caught failure, NOT a peak. Pool high water is a separate counter, NOT whole-device residency. A single successful grid point is not an extrapolated ceiling.

| Arm | S | Status/outcome | Fit | Resident after/failure MiB | Pool reserved high MiB |
|---|---:|---|---|---:|---:|
| A | 4096 | UNRUN | — | — | — |
| A | 8192 | UNRUN | — | — | — |
| A | 16384 | UNRUN | — | — | — |
| A | 24576 | UNRUN | — | — | — |
| A | 32768 | UNRUN | — | — | — |
| B | 4096 | UNRUN | — | — | — |
| B | 8192 | UNRUN | — | — | — |
| B | 16384 | UNRUN | — | — | — |
| B | 24576 | UNRUN | — | — | — |
| B | 32768 | UNRUN | — | — | — |
| C | 4096 | UNRUN | — | — | — |
| C | 8192 | UNRUN | — | — | — |
| C | 16384 | UNRUN | — | — | — |
| C | 24576 | UNRUN | — | — | — |
| C | 32768 | UNRUN | — | — | — |
| D | 4096 | UNRUN | — | — | — |
| D | 8192 | UNRUN | — | — | — |
| D | 16384 | UNRUN | — | — | — |
| D | 24576 | UNRUN | — | — | — |
| D | 32768 | UNRUN | — | — | — |
| E | 4096 | UNRUN | — | — | — |
| E | 8192 | UNRUN | — | — | — |
| E | 16384 | UNRUN | — | — | — |
| E | 24576 | UNRUN | — | — | — |
| E | 32768 | UNRUN | — | — | — |

Clean decode: June fused GEMV/RMSNorm/softmax, pool ON BEFORE QAT load; no attention-class wrapper, no interposer. C installs only the required native SP binding dispatch. Greedy 32 synchronized steps; sole per-step tensor host copy is one device argmax int64. First step included; no throwaway warmup because Gemma mutates its KVRing. At 32K require same-arm 8192 estimate setup+16*prefill+4*decode+15 <285s; planned rail is unknown fit, not OOM.

| Arm | Starting S | Status/outcome | Steps | ms/token including argmax | Resident after MiB |
|---|---:|---|---:|---:|---:|
| A | 2048 | UNRUN | — | — | — |
| A | 8192 | UNRUN | — | — | — |
| A | 32768 | UNRUN | — | — | — |
| B | 2048 | UNRUN | — | — | — |
| B | 8192 | UNRUN | — | — | — |
| B | 32768 | UNRUN | — | — | — |
| C | 2048 | UNRUN | — | — | — |
| C | 8192 | UNRUN | — | — | — |
| C | 32768 | UNRUN | — | — | — |

**G2/G3 rows establish nothing about model quality by themselves.**

## Registered predictions

| Lead | Prediction | Seat prediction registered alongside |
|---|---|---|
| P1 | D equals A within 0.005 ppl at D=512/MQA. | D/A short difference <=.005; BF16 summation risk remains. |
| P2 | C <= B + .02 short bulk4; B-C < .118. | C <= B+.02 and abs(B-C)<.118 on short protocol. |
| P3 | every global layer: B unrefined mass >= .8; C <= .3 at matched fraction. | C average unrefined mass < B, but lead universal per-layer thresholds likely fail for at least one layer. |
| P4 | E fraction >= .95 and real-key eq >= .5. | eq>=.5 and E fraction>=.95 likely, even qk-normalized 512 keys. |
| P5 | SP ceiling >= B >= A; A fails by16384. | standard may fit16384 on12GB with June adaptive chunks; lead A-fails prediction likely false; no strict ceiling ordering assumed. |

Assessments: `{"P1": {"2048": "UNRUN", "8192": "UNRUN", "16384": "UNRUN", "32768": "UNRUN"}, "P2": "UNASSESSED", "P3": "UNASSESSED", "P4": "UNASSESSED", "P5": "UNASSESSED; incomplete grid or rail is not an OOM"}`. Predictions are not measurements or pass criteria except the explicitly registered D/A exactness gate.

## CPU gates, lead commands, and wall estimates

`{"status": "PASS_CPU_ONLY", "registration_sha256": "099a8bd9e1bb94909a81c110521d2d3a9d642d5fac27d49f6820b97cb3fbbd1e", "evidence_class": "author unit tests / mutation baseline / host CUDA compile and link / source inspection", "pytest": {"passed": 38, "failed": 0, "skipped": 0, "log": "logs/apa_sp4g_cpu_handoff.log", "log_sha256": "957989af213ecaaa62c8fae9371351c602c064dd055911e89a82230ee3086d84", "warnings": "2 SWIG import DeprecationWarnings retained; final swigvarlink warning also retained"}, "mutation": {"killed": 8, "nonerror": 8, "rate": 1.0, "threshold": 0.8, "current_source_pins": true, "receipt": "artifacts/apa_sp4g/mutations_sealed/results.json", "sha256": "0a41bdcf68c47fcc2356b076f7eb6e81470df18ffa4cfc4414c9d9c65383116b"}, "build": {"status": "PASS_HOST_ONLY", "manifest_sha256": "9dadd8332a94ab53c40d34f2f591e40edab160a6f1498bfb68842a6a518e3e4e", "log": "logs/apa_sp4g_build.log", "log_sha256": "2300811ceb19d3ea093e7eaaab1014f65869e03790e3f3aeb2c7a62047ce36dc"}, "source_preservation": {"all_registered_preexisting_files_unchanged": true, "registered_source_files": 62, "external_inputs": 36, "production_edits": 0}, "adapter_import": {"evidence_class": "host import / device visibility", "engine": "/mnt/ForgeRealm/Project-Tensor-wt-apa-sp4g/artifacts/apa_sp4g/build/_tensor_cuda.cpython-312-x86_64-linux-gnu.so", "adapter": "/mnt/ForgeRealm/GraftRepository/core/gemma4_tc.py", "cudaGetDeviceCount": 100, "device_count": 0, "GPU_jobs": 0}, "registered_cells": 1408, "blind_verification": "lead-owned UNRUN; author baseline only"}` (complete source pins: `CPU_GATES.json`).

Full cell manifest: `cells.json`; exact dependency-ordered commands and PER-JOB estimates: `lead_commands.txt`. Each resume runs at most ONE cell, stops on RED/stale evidence, and never overwrites a receipt. Independent cells can be selected by explicit run commands after a different cell hits a rail. No automatic retries or unbounded batch.

```bash
cd /mnt/ForgeRealm/Project-Tensor-wt-apa-sp4g
bash scripts/apa_sp4g_lead_gpu.sh list
bash scripts/apa_sp4g_lead_gpu.sh run kernel512
bash scripts/apa_sp4g_lead_gpu.sh run ppl_A_2048_w0
bash scripts/apa_sp4g_lead_gpu.sh resume
bash scripts/apa_sp4g_lead_gpu.sh summary
```

QAT 12B load estimate ALONE: 30–120s, unmeasured, including GGUF read/repack/upload and per-layer garbage collection. Each model job estimate 60–285s including load; long runs may exceed the rail and become RED. Kernel 2–45s; each 128-query margin band 5–120s; population aggregation 1–120s. Worker TERM 285s plus5s owned-child grace =290s hard maximum. Outer TERM 585s plus3s grace =588s, under 590. Lease wait 20s; foreground 30s cooldown while retaining lock after GPU jobs. Only owned timeout children may be signalled; no process discovery-to-kill behavior.

There are 1408 bounded cells, including 1280 G2 band cells. This conservative split is costly: those band cooldowns alone total 10.67 hours if every band is dispatched separately. It prioritizes explicit bounded all-pair receipts over an unverified claim that a whole 8192 layer fits 285s. A lead-authorized batching amendment could reduce lease overhead; none is silently applied. Full retained G2 float64 error populations require about 73GB plus captures; cell disk rails never reduce coverage.

## Prior art

- **Gemma June 2026:** port ledger and floor script: model/caches/flags/feeding reused; new experiment harness.
- **SP3 2026:** calibration, provenance, receipts, clean decode lesson; new architecture wiring.
- **BLASST Yuan et al. 2025/2026:** https://arxiv.org/abs/2512.12087; running-max comparison; APA refines keys and retains denominator.
- **ThriftAttention Sharratt 2026:** https://arxiv.org/abs/2605.23081; precision-selection/softmax-weight motivation; no FP4 implementation port.
- **FlashAttention-2 Dao 2023:** https://arxiv.org/abs/2307.08691; existing online softmax/work partition; no new kernel.
- **TurboQuant Zandieh et al. 2025:** https://arxiv.org/abs/2504.19874; existing rotated scalar-codebook reconstructed BF16 kq; no QJL residual.
- **SP2 2026:** conditional delta log(1/epsilon)+2eq reused; empirical finite max not universal proof.
- **standard methods:** NLL Shannon1948, nearest-rank statistics, bisection, content hashing NIST2001, leases, Make Feldman1979 dependency invalidation; no novelty claimed.

arXiv abstract pages verified using web tool this seat; comparator detail also checked in lead prior-art comparison; no external benchmark reproduced. Code comments and ledger state what is reused. SP4G adds experiment wiring, not a new attention algorithm. Bulk4 is reconstructed BF16 Kq computed by existing floating-point dots; this does not demonstrate packed FP4 arithmetic or compressed KV residency.

## Deviations, RED, residual risks, and process safety

- No GPU was available in this seat. P1–P5, native diagnostic bit parity, real-model exactness/quality, observed fractions, memory fit, and throughput remain unmeasured until lead receipts exist. Host compilation is not CUDA numerical validation.
- June off-by-one correction/fp64 and forced fused threshold 0 are registered. The floor default fast_max_seq=4096 would use blend at short S; the order explicitly requests fused B. INT4/GEMM opt-ins and K/V-storage quantization are off for tensor parity. Adaptive prefill can change the lead ceiling prediction; no full dense-prefill result is implied.
- E's eq is a finite maximum over B/C activations, not a universal bound on E-induced or unseen activations/longer sequences. Transfer is a measured hypothesis. C's matched fraction is the prefix calibration population, not a guarantee of ±0.01 on all scored windows or decode partition-local maxima.
- Diagnostic masks and captures add overhead only to diagnostic/PPL cells. Clean decode has no diagnostics. Full clean numerical output parity at real Gemma shapes and long rails is still untested.
- `Not claimed fixed`: no observed GPU failure was reproduced or repaired. No production source changed. Author tests/mutations are baseline evidence; House Rules blind verification is lead-owned and UNRUN. Bulk8 secondary is deferred; no bulk8 claims.
- Receipts are atomic create-only, dependency/fingerprint checked per kind. Array creation hashes plus exact stat identity protect large artifacts on reuse; metadata JSON is rehashed. Unknown source transitions invalidate reuse and require separate amendments. Timeout/OOM evidence is preserved, never promoted to a successful model-quality result.
- No git commands, subagents, shell background jobs, live-service changes, or foreign-process signals. All execution was local preparation, host build and CPU verification. Read-only SP3 modules/receipts, Graft adapter/docs, and models were preserved.
- Model identity/effort honesty: GPT-6 (system-provided family; exact serving variant not exposed); not exposed in this seat; no invented runtime identifier.

Receipt audit failures: `[]`.

Sources of truth: registration.json, cells.json, CPU_GATES.json, GPU_BLOCKED.json, build/manifest.json, jobs/*.json when run, logs/apa_sp4g_*, and docs/APA_SP4G_LEDGER.md.
