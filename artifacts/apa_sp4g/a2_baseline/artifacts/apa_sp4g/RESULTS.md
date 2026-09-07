# APA-SP4G results

**CPU PASS_CPU_ONLY; GPU receipts present, inspect tables.** No Gemma PPL, margin, ceiling or decode result is claimed without a valid current receipt.

Registration SHA: `099a8bd9e1bb94909a81c110521d2d3a9d642d5fac27d49f6820b97cb3fbbd1e`. Model reference A is the engine's QAT INT4 Gemma-4-12B-it (exact q4_0 symmetric-8 group32 import, BF16 compute). A 12B BF16 weight set is roughly 24GB and does not fit the 12GB card; no torch or full-BF16 model reference was loaded. This tests attention changes inside this engine; it cannot establish parity to a BF16 model.

## Amendment 1 status

Original card evidence: kernel512 PASS; ppl_A_2048_w0 RED with `AttributeError: /usr/local/cuda-12.6/lib64/libcudart.so.12: undefined symbol: cudaGetDeviceDefaultMemPool`. Both original receipts remain byte-identical in jobs/. The CUDA pool typo is repaired and all four APIs resolve on model-module import before model loading. New executions write jobs_a1/. Not claimed fixed: model PPL execution on the card has not been rerun in this seat.

Kernel512 remains valid through amendment_003_a1_fingerprint.json: raw common/registry/gpu closure files changed, so byte-identical closure is NOT claimed. The exact reviewed bridge pins before/after hashes; kernel512 function bytes, kernel dispatch, runtime loader, compiled build and numerical inputs remain unchanged. Unknown transitions fail closed; no RED receipt is eligible.

## Adapter facts and PROTOCOL-G

Source inspection: `gemma4_tc.py:568/658` gates APA by strict `S_all > apa_min_context`; EVERY B/C/D/E cell sets `apa_min_context=0` and `fast_max_seq=0`. Thus the short windows actually exercise fused APA. A stays standard. B is reconstructed-key `apa_selective_attention` at lines 609/712, bulk 4/r=.15, with INT4/GEMM opt-ins off. C/D/E dispatch from that SAME binding with Q[B,16,L,512] and K/Kq/V[B,1,S,512], BF16, scale=1. No V padding or head expansion difference between arms. Hidden activations can differ as attention changes.

Both SP launchers already instantiate 512 (kernels.cu:7496, apa_sp1_1.cuh:179) and map 16 query heads to KV=1. Tests: `test_d512_mqa_prefill_contract_and_dense_pin`, `test_d512_mqa_splitk_contract_and_dense_pin`, `test_compiled_d512_mqa_flag_device_scalar_guards`; native numerical receipt `jobs/kernel512.json` PASSED on the lead card. No production source or kernel-body edits.

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

All causal pairs, no sampling. One whole-layer cell per (S, arm, layer); internal 128-query tiles preserve absolute causal alignment and must replay the captured output bitwise. Error percentiles are nearest-rank over the entire layer population; no averaging percentiles. Unrefined mass is the mean per-query exact softmax mass on skipped keys; relative weight is max exp(exact_skipped − row_max_exact). These are different quantities.

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

`{"status": "PASS_CPU_ONLY", "registration_sha256": "099a8bd9e1bb94909a81c110521d2d3a9d642d5fac27d49f6820b97cb3fbbd1e", "evidence_class": "author CPU unit suite / real libcudart symbol lookup / copied-source mutation baseline; existing host build verified unchanged", "pytest": {"passed": 48, "failed": 0, "skipped": 0, "log": "logs/apa_sp4g_a1_cpu_final.log", "log_sha256": "a189a35e35afe046739ce50fa6d37143d165069047966742d14bd6be7ee5793d", "warnings": "2 SWIG DeprecationWarnings retained; final swigvarlink warning retained"}, "symbol_resolution": {"test": "test_worker_ctypes_symbols_resolve_at_import_without_cuda_calls", "negative_test": "test_misspelled_pool_symbol_reds_during_import", "library": "/usr/local/cuda-12.6/lib64/libcudart.so.12", "symbols": ["cudaGetDevice", "cudaDeviceGetDefaultMemPool", "cudaMemPoolGetAttribute", "cudaMemPoolSetAttribute"], "api_calls_during_import": 0}, "mutation": {"killed": 8, "nonerror": 8, "rate": 1.0, "threshold": 0.8, "receipt": "artifacts/apa_sp4g/mutations_a1_final/results.json", "sha256": "45640a49eb50bf77a51a6010a8c91aaba3b5d111091b0aa5b52d1f44d777742d", "earlier_result": "7/8=.875 PASS; stale-fingerprint seed was retargeted to intended clause; original receipt retained"}, "build": {"status": "UNCHANGED_FROM_CARD_PASS", "manifest_sha256": "9dadd8332a94ab53c40d34f2f591e40edab160a6f1498bfb68842a6a518e3e4e", "rebuilt": false}, "source_preservation": {"registered_sources": 62, "external_inputs": 36, "production_edits": 0, "original_registration_sha256": "099a8bd9e1bb94909a81c110521d2d3a9d642d5fac27d49f6820b97cb3fbbd1e"}, "fingerprint_amendment_sha256": "5b7e2d490a46e3f00ba1cc0a2e58e99c5275bfab2e0f0a30391af074a4ece50a", "registered_cells": 128, "blind_verification": "lead-owned UNRUN; author baseline only"}` (complete source pins: `CPU_GATES_A1.json`).

Full cell manifest: `cells.json`; exact dependency-ordered commands and PER-JOB estimates: `lead_commands.txt`. Each resume runs at most ONE cell, stops on RED/stale evidence, and never overwrites a receipt. Independent cells can be selected by explicit run commands after a different cell hits a rail. No automatic retries or unbounded batch.

```bash
cd /mnt/ForgeRealm/Project-Tensor-wt-apa-sp4g
bash scripts/apa_sp4g_lead_gpu.sh list
bash scripts/apa_sp4g_lead_gpu.sh run kernel512
bash scripts/apa_sp4g_lead_gpu.sh run ppl_A_2048_w0
bash scripts/apa_sp4g_lead_gpu.sh resume
bash scripts/apa_sp4g_lead_gpu.sh summary
```

QAT 12B load estimate ALONE: 30–120s, unmeasured, including GGUF read/repack/upload and per-layer garbage collection. Each model job estimate 60–285s including load; long runs may exceed the rail and become RED. Kernel 2–45s; whole-layer margins at2048:10–90s, at8192:60–270s, including replay, CPU exact dots and full-population percentiles, with no model load. Estimates are unmeasured. Explicit fallback bands5–120s and fallback aggregation1–120s. Worker TERM 285s plus5s owned-child grace =290s hard maximum. Outer TERM 585s plus3s grace =588s, under 590. Lease wait 20s; foreground 30s cooldown while retaining lock after GPU jobs. Only owned timeout children may be signalled; no process discovery-to-kill behavior.

There are 128 default cells, including32 whole-layer G2 cells (16 per length). S8192 cooldown falls from1024 to16 jobs:512 minutes to8 minutes; both lengths together960s=16 minutes. ONE KV head is shared by16 query heads: S8192 has536,936,448 causal query-head/key pairs per layer, and S2048 has33,570,816. All are included. Internal replay tiles remain128 queries; full retained float64 errors require about73GB plus captures. A whole-layer job hitting the unchanged rail stays RED. Only then may the lead explicitly run that layer’s margin_band and margin_summary cells listed in lead_commands_fallback.txt; default resume never dispatches fallback cells. A valid completed fallback summary can satisfy the layer dependency while retaining the original RED. Per-cell12GiB disk headroom covers retained errors plus population scratch.

## Prior art

Amendment1 introduces no new algorithm or prior art. Reuses SP3 a4(2026) exact reviewed fingerprint transitions, existing SP4G/June(2026) replay and population statistics, and NVIDIA CUDA Runtime12.6(2024) ABI verified against installed cuda_runtime_api.h/driver_types.h. New work is symbol spelling, import-time resolution and dispatch coalescing only.

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

- No GPU was available in this seat. P1–P5, real-model diagnostic replay/exactness/quality, observed fractions, memory fit, and throughput remain unmeasured until valid lead receipts exist; the original D512/MQA kernel pin passed. Host compilation is not CUDA numerical validation.
- June off-by-one correction/fp64 and forced fused threshold 0 are registered. The floor default fast_max_seq=4096 would use blend at short S; the order explicitly requests fused B. INT4/GEMM opt-ins and K/V-storage quantization are off for tensor parity. Adaptive prefill can change the lead ceiling prediction; no full dense-prefill result is implied.
- E's eq is a finite maximum over B/C activations, not a universal bound on E-induced or unseen activations/longer sequences. Transfer is a measured hypothesis. C's matched fraction is the prefix calibration population, not a guarantee of ±0.01 on all scored windows or decode partition-local maxima.
- Diagnostic masks and captures add overhead only to diagnostic/PPL cells. Clean decode has no diagnostics. Full clean numerical output parity at real Gemma shapes and long rails is still untested.
- `Not claimed fixed`: the full model cell has not yet been rerun on the card. The reported symbol failure was reproduced and repaired at CPU import/symbol resolution scope. No production source changed. Author tests/mutations are baseline evidence; House Rules blind verification is lead-owned and UNRUN. Bulk8 secondary is deferred; no bulk8 claims.
- Receipts are atomic create-only, dependency/fingerprint checked per kind. Array creation hashes plus exact stat identity protect large artifacts on reuse; metadata JSON is rehashed. Unknown source transitions invalidate reuse and require separate amendments. Timeout/OOM evidence is preserved, never promoted to a successful model-quality result.
- No git commands, subagents, shell background jobs, live-service changes, or foreign-process signals. All execution was local preparation, host build and CPU verification. Read-only SP3 modules/receipts, Graft adapter/docs, and models were preserved.
- Amendment1 seat: gpt-6-astra, reasoning xhigh, confirmed by logs/apa_sp4g_a1_r1.log. Original registration identity fields remain immutable.

Receipt audit failures: `[]`.

Sources of truth: registration.json, cells.json, CPU_GATES_A1.json, GPU_BLOCKED_A1.json, amendment_002_a1_execution.json, amendment_003_a1_fingerprint.json, build/manifest.json, jobs/*.json and jobs_a1/*.json, logs/apa_sp4g_*, and docs/APA_SP4G_LEDGER.md.
