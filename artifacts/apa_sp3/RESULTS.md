# APA-SP3 — RED / model-quality result not established

Evidence class: **model perplexity** for scored rows; **kernel sweep** for activation margins and decode timing.
**G2/G3 rows establish nothing about model quality by themselves.**

Registration SHA256: `d9b6511702a894f72174795141c72b2097b1bbe6110e810a1d4d8bfc3cd3498c` (immutable).
Model: openbmb/MiniCPM3-4B; INT4 affine groups of 128; BF16 compute; 62 layers, 40 heads, composite D=96 and zero-padded V=96.
Execution seat: gpt-6-astra, reasoning effort xhigh, confirmed by logs/apa_sp3_r1.log. No GPU model execution by this seat.

## G0 protocol and parity

Published targets: A=20.065 and B=19.817, absolute tolerance 0.01, same process. These are historical targets, not reproduced values.
Source: `/mnt/ForgeRealm/GraftRepository/docs/MiniCPM3-MLA_Results.md`.
The named `/mnt/ForgeRealm/AI-AtlasForge/workspace/APA-Quant-Rust_LLM_testing/mission_b74b7906/test_graft_e1_mla.py` is a graft-router experiment, with no PPL/corpus prefix calculation.
Dispatch finding: documented `/tmp/minicpm3_engine_bench.py`, `minicpm3_reference.py`, `minicpm3_ceiling.py` were absent. Exact original guide source, concatenation, prefix, special-token policy and scorer were not recovered. Token SHA is **null** in the immutable base registration; any later recovery must be pinned in a separate amendment.
The later `GraftRepository/tests/minicpm3_apa_recovery.py` uses six WikiText-or-local-document windows in cached chunks and scores 511 targets; it does not establish the original protocol. No fallback corpus is substituted.
Every model worker requires a separately pinned recovery amendment and rechecks A/B in that process before an SP arm. G0 miss stops PPL arms; D miss stops SP model arms. No tolerances change.

## Perplexity and prefill table — model perplexity

| Bits | S | Arm | Status | PPL last-512 | ms wall | Peak resident MiB* | Refined fraction | Layer min..max | δ |
|---|---:|---|---|---:|---:|---:|---:|---|---:|
| 4 | 1024 | A standard | BLOCKED / unrun | — | — | — | — | —..— | — |
| 4 | 1024 | B apa_two_pass | BLOCKED / unrun | — | — | — | — | —..— | — |
| 4 | 1024 | C apa_sp_matched | BLOCKED / unrun | — | — | — | — | —..— | — |
| 4 | 1024 | D apa_sp_refine_all | BLOCKED / unrun | — | — | — | — | —..— | — |
| 4 | 1024 | E apa_sp_provable | BLOCKED / unrun | — | — | — | — | —..— | — |
| 4 | 8192 | A standard | BLOCKED / unrun | — | — | — | — | —..— | — |
| 4 | 8192 | B apa_two_pass | BLOCKED / unrun | — | — | — | — | —..— | — |
| 4 | 8192 | C apa_sp_matched | BLOCKED / unrun | — | — | — | — | —..— | — |
| 4 | 8192 | D apa_sp_refine_all | BLOCKED / unrun | — | — | — | — | —..— | — |
| 4 | 8192 | E apa_sp_provable | BLOCKED / unrun | — | — | — | — | —..— | — |
| 8 | 1024 | A standard | BLOCKED / unrun | — | — | — | — | —..— | — |
| 8 | 1024 | B apa_two_pass | BLOCKED / unrun | — | — | — | — | —..— | — |
| 8 | 1024 | C apa_sp_matched | BLOCKED / unrun | — | — | — | — | —..— | — |
| 8 | 1024 | D apa_sp_refine_all | BLOCKED / unrun | — | — | — | — | —..— | — |
| 8 | 8192 | A standard | BLOCKED / unrun | — | — | — | — | —..— | — |
| 8 | 8192 | B apa_two_pass | BLOCKED / unrun | — | — | — | — | —..— | — |
| 8 | 8192 | C apa_sp_matched | BLOCKED / unrun | — | — | — | — | —..— | — |
| 8 | 8192 | D apa_sp_refine_all | BLOCKED / unrun | — | — | — | — | —..— | — |

*Peak is explicitly an estimate: exact intercepted cudaMalloc allocation high-water plus the pre-call device/context offset. Internal driver transient allocations can be missed. Raw pooling is OFF. No background sampling thread. Diagnostic/capture timings never fill this table.
8192 rows request full prefill plus last-512 logits in one forward. A deadline or OOM is RED, including if logits do not fit. No chunking or precision fallback. Bulk8 is optional secondary; no E8 is registered.

## Empirical margin table — kernel sweep on model activations

| Bits | Arm | S | Layer | Error mean | p99 | p99.9 | max | Unrefined mass mean | p99 | max | Max skipped w/w* | Fraction |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 / 8 | B / C | 1024 / 8192 | 0–61 | BLOCKED | — | — | — | — | — | — | — | — |

0/248 primary B/C layer receipts available. Every row requires all eligible heads, queries and keys; future masked keys are excluded. Bulk scores come from a native SP-order FP32 CUDA score probe; exact scores are float64 Q.K dots at the actual float32 scale (amendment_002_native_bulk.json). B blend error replays its exact native bulk chunks and requires the captured byte SHA; B/C also expose sp_error in JSON for the E margin (amendment_003_B_native_scores.json). Error percentiles are exact nearest ranks after float32 storage; mean/max use float64 errors. Actual selection masks come from native SP diagnostics or literal instrumented B copies, checked bit-identical against B output on the same inputs.

The heuristic tail cannot yet be judged: no real-activation error, unrefined-mass or skipped-relative-weight rows have run. A skipped key remains in the softmax at bulk precision; these measurements will quantify its exact-softmax importance, not mass removed from attention. Synthetic sweep outliers do not supply this model’s margin.

The model comparison is incomplete: reproduced original-token G0 parity and a scored C comparison at matched actual fraction are both required; see their individual statuses above. G2/G3 rows establish nothing about model quality by themselves. The ε=1e−3 margin is conditional on the measured finite error envelope; E’s separate capture checks its transfer, and neither check supplies a universal quantization bound or a CUDA rounding proof.

## Decode table — kernel sweep / in-model timing

| Bits | Starting S | Arm | Status | tokens/s | Steps | Prefill seconds |
|---|---:|---|---|---:|---:|---:|
| 4 | 2048 | A | BLOCKED / unrun | — | — | — |
| 4 | 2048 | B | BLOCKED / unrun | — | — | — |
| 4 | 2048 | C | BLOCKED / unrun | — | — | — |
| 4 | 8192 | A | BLOCKED / unrun | — | — | — |
| 4 | 8192 | B | BLOCKED / unrun | — | — | — |
| 4 | 8192 | C | BLOCKED / unrun | — | — | — |
| 4 | 32768 | A | BLOCKED / unrun | — | — | — |
| 4 | 32768 | B | BLOCKED / unrun | — | — | — |
| 4 | 32768 | C | BLOCKED / unrun | — | — | — |
| 8 | 2048 | A | BLOCKED / unrun | — | — | — |
| 8 | 2048 | B | BLOCKED / unrun | — | — | — |
| 8 | 2048 | C | BLOCKED / unrun | — | — | — |
| 8 | 8192 | A | BLOCKED / unrun | — | — | — |
| 8 | 8192 | B | BLOCKED / unrun | — | — | — |
| 8 | 8192 | C | BLOCKED / unrun | — | — | — |
| 8 | 32768 | A | BLOCKED / unrun | — | — | — |
| 8 | 32768 | B | BLOCKED / unrun | — | — | — |
| 8 | 32768 | C | BLOCKED / unrun | — | — | — |

Identical teacher-forced continuations, 32 measured steps, CUDA-synchronized wall time per token; expanded MLA for A/B/C, absorbed decode OFF. Cache prefill is excluded from tokens/s but included in the 480s worker ceiling. At starting S=32768, measured attention lengths are 32769–32800: explicitly beyond the trained window, with no quality claim.
**G2/G3 rows establish nothing about model quality by themselves.**

## Registry, predictions and gates

C uses one global δ per bitwidth, matched over all layers at S=1024 to B’s actual diagnostic fraction ±0.01. Fixed grid on B activations initializes at the smallest tied δ; at most eight actual C fraction-only trials, bounded bisection on [0,32]. First match freezes δ for long prefill/decode; no PPL-based tuning. Report per-layer min/max/std.
D uses finite float32 max δ, requires every eligible pair selected and |D−A|≤0.005 PPL. E uses upward_float32(ln(1000)+2·eq), eq the upward float32 global SP-arithmetic error maximum across B/C, both lengths, all layers. Missing layer blocks E. E’s own captures separately check the finite-envelope transfer.

| Owner | ID | Registered prediction | Status |
|---|---|---|---|
| Lead | P1 | D equals A within 0.005 ppl | UNASSESSED; requires applicable model receipts |
| Lead | P2 | C within 0.05 ppl of B at matched fraction | UNASSESSED; requires applicable model receipts |
| Lead | P3 | E fraction >=0.95, real eq >=0.8, E ppl within 0.01 of A | UNASSESSED; requires applicable model receipts |
| Lead | P4 | bulk4 p99 error <0.5*max; C skipped weight ratio >0.1 on some layer while ppl unmoved (P2 tolerance) | UNASSESSED; requires applicable model receipts |
| Lead | P5 | C/B decode tokens/s at S=32768 >=2 | UNASSESSED; requires applicable model receipts |
| Seat | S1 | G0 likely RED on current engine even after tokens recovered; documented matmul/softmax drift exceeds 0.01 | UNASSESSED; requires applicable model receipts |
| Seat | S2 | D likely differs from A by >0.005 ppl: BF16 cuBLAS/softmax rounding differs from fused FP32 accumulation | UNASSESSED; requires applicable model receipts |
| Seat | S3 | E fraction >=0.95 at bulk4; conditional finite bound will not establish general low-precision usefulness | UNASSESSED; requires applicable model receipts |
| Seat | S4 | C within 0.05 of B if realised fraction matches; skipped weight ratio >0.1 likely | UNASSESSED; requires applicable model receipts |
| Seat | S5 | C/B full-model decode at S=32768 <2 because latent expansion, re-quantization and INT4 projections remain in both arms | UNASSESSED; requires applicable model receipts |

CPU gates: {"D96_pins": ["test_d96_prefill_contract_and_dense_pin", "test_d96_splitk_contract_and_dense_pin"], "G0": "NOT REPRODUCED; source/token provenance missing; preflight refuses before lease", "GPU": "BLOCKED; cudaGetDeviceCount=100, no device; 0 GPU jobs", "adapter_cpu_import": "PASS; 6 modules, no model instantiation", "blind_verification": "lead-owned, unrun", "dry_run_cells": 688, "evidence_class": "unit test / host compile / code inspection", "host_build": {"core_log": "artifacts/apa_sp3/build.log", "diagnostic_log": "artifacts/apa_sp3/diagnostic_rebuild.log", "manifest_sha256": "85b1ada58e1732432f55ce39ddc918b02f102df31543371f1cdf192446abe726", "status": "PASS"}, "mutation": {"killed": 6, "manifest": "artifacts/apa_sp3/mutation_manifest_final.json", "nonerror": 6, "rate": 1.0, "receipt": "artifacts/apa_sp3/mutation_results_final.json", "scope": "author baseline; not blind verification", "threshold": 0.8}, "observer_host_gate": {"GPU_allocation_accounting": "still untested; no device", "after_preload": "cudaMalloc failed: no CUDA-capable device is detected", "expected": "cudaMalloc failed: no CUDA-capable device is detected", "status": "PASS_HOST_FAILURE_PRESERVATION"}, "pins": {"preexisting_kernel_bodies": 107, "preexisting_source_files": 24, "registration_sha256": "d9b6511702a894f72174795141c72b2097b1bbe6110e810a1d4d8bfc3cd3498c", "unchanged": true}, "pytest": {"failed": 0, "log": "artifacts/apa_sp3/cpu_after_observer_fix.log", "log_sha256": "e3b5ed70339f6e7fd37062881ce10bb786760d5273efeca28262f0c881d57f0d", "passed": 20, "skipped": 0}, "status": "PASS_CPU_ONLY", "supersedes": "CPU_GATES.json: observer RTLD_LOCAL resolution fix and new build seal"}
D=96 tests: `test_d96_prefill_contract_and_dense_pin`, `test_d96_splitk_contract_and_dense_pin`, and compiled flag/device/scalar guards for both geometries. Native GPU numerical pins are delivered in job `kernel96`, unrun here.
Author tests and mutations are baseline evidence only. Independent blind verification under House Rules §8 is lead-owned and unrun; no subagents were launched.

## Lead commands and bounds

```bash
cd /mnt/ForgeRealm/Project-Tensor-wt-apa-sp3
timeout 30s bash scripts/apa_sp3_lead_gpu.sh list
timeout --kill-after=2s 590s bash scripts/apa_sp3_lead_gpu.sh run kernel96
# Recover original protocol/token pins in a separate amendment before G0:
timeout --kill-after=2s 590s bash scripts/apa_sp3_lead_gpu.sh run g0
# Each resume invocation runs at most ONE cell; stops on RED/stale receipts:
timeout --kill-after=2s 590s bash scripts/apa_sp3_lead_gpu.sh resume 4
timeout 30s bash scripts/apa_sp3_lead_gpu.sh summary
```
Complete commands: `artifacts/apa_sp3/lead_commands.txt`; every cell and per-job estimate: `dry_run.json`. Optional secondary uses `resume 8` after primary prerequisites. No multi-hour automatic batch is launched.
Each leased operation: flock --wait 20, worker timeout 480s plus 5s own-child termination grace, foreground 30s cooldown; outer guard TERM at 585s plus 3s grace. Device/PID inspection fails closed. Never signal a foreign PID. Operator keeps right of way; advisory-lock cooperation is required.
Planning estimates, not timings: G0 35–150s; 1024 PPL 45–180s; 1024 capture/calibration 50–250s; 8192 prefill/capture 120–480s or RED timeout/OOM; per-layer G2 3–60s at 1024, 30–480s at 8192; decode including cache prefill 60–480s or RED. Actual upper-bound compliance is enforced, not predicted.
Memory reasoning: a single BF16 40-head S² score tensor needs 5.0 GiB at 8192 and 80 GiB at 32768, before intermediates and ~2.9 GB model residency. Standard full prefill at 32768 cannot fit on 12 GB; its decode setup will record OOM. 8192 standard fit and B/SP full-prefill deadlines remain unverified. No alternate cache-building scheme is silently substituted.

## Prior art

| Work | What is reused / what SP3 adds |
|---|---|
| [BLASST](https://arxiv.org/abs/2512.12087) | Yuan et al. 2025/2026, arXiv:2512.12087: running-max log threshold; APA promotes individual keys instead of omitting blocks |
| [ThriftAttention](https://arxiv.org/abs/2605.23081) | Sharratt 2026, arXiv:2605.23081: selective precision and weight-sensitive error motivation; no FP4 tensor-core implementation port |
| [FlashAttention-2](https://arxiv.org/abs/2307.08691) | Dao 2023, arXiv:2307.08691 and Milakov/Gimelshein 2018 online normalizer: stable online softmax/work partitioning; existing SP implementation reused, no FA2 port |
| [TurboQuant](https://arxiv.org/abs/2504.19874) | Zandieh et al. 2025, arXiv:2504.19874: rotation and scalar codebook via existing quant.py; reconstructed BF16 keys, no QJL residual or packed FP4 arithmetic |
| APA | David Perry 2026, docs/APA_PAPER_DRAFT.md: all-key denominator, |bulk| z selection and precision promotion |
| SP2 | existing conditional log(1/epsilon)+2eq proof; here model finite max measurement, no new universal theorem |
| SP3 | new experiment wiring, global model calibration, captured-activation tail audit and fail-closed receipts; empirical percentiles and matched-control experiments are standard, no novelty claimed |

The APA draft is David Perry (2026), `docs/APA_PAPER_DRAFT.md`; treated as local prior art, not independently verified theorem. Arm A uses standard scaled dot-product attention (Vaswani et al. 2017, https://arxiv.org/abs/1706.03762), not an external FA2 package. Standard NLL (Shannon 1948), nearest-rank order statistics, directed rounding, memory maps, ELF allocation interposition, content hashes and leases are not new algorithms. Mutation-testing historical attribution is an unverified lead: DeMillo/Lipton/Sayward 1978, search “Hints on Test Data Selection”. Code sites and ledger contain the same provenance distinctions.
Bulk4/8 here means a TurboQuant codebook reconstructed to BF16 Kq. Existing SP bulk dots execute floating-point instructions; this experiment cannot establish packed FP4 speed or compressed KV residency. No source/kernels outside the authorized new harness/test paths were edited.

## Deviations, RED and residuals

- Original token SHA and exact guide protocol unavailable: registration honestly records null; no PPL reproduction claimed. Separate amendment is required before model gates. This is a delivered block, not a waived gate.
- Missing local SP2/SPD1 artifacts were read from sibling worktrees, read-only. SP1/SP1.1 registration hashes are corroborated by the local ledger and SP2 parent registration; their original JSONs are absent here.
- Native GPU behavior, diagnostic bit parity, BF16 D≈A, long-context fit/time and actual model quality remain untested. Kernel-body hash equality proves source preservation, not GPU correctness.
- Resident peak is an explicitly qualified estimate. Cold/warm effects and single-call prefill variability remain; decode measures 32 steps and includes expansion/quantization overhead common to arms.
- Full long-context captures use tens of GB of disk and native diagnostic masks add quadratic transient memory. Disk/OOM/time failures remain RED; no sampling reduction.
- No git, subagents, shell background jobs, live-service changes or foreign-process termination. Only explicitly bounded own child processes may be terminated by timeout. Host build/tests complete; lead GPU and blind verification pending.

Files: `scripts/apa_sp3_*`, `tensor_cuda/tests/test_apa_sp3.py`, `docs/APA_SP3_LEDGER.md`, and `artifacts/apa_sp3/` (registration, source pins, build, CPU/mutation receipts, exact commands, blocked JSON and this renderer output).
