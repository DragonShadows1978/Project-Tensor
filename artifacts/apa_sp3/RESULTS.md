# APA-SP3 — RED / model-quality result not established

Evidence class: **model perplexity** for scored rows; **kernel sweep** for activation margins and decode timing.
**G2/G3 rows establish nothing about model quality by themselves.**

Registration SHA256: `d9b6511702a894f72174795141c72b2097b1bbe6110e810a1d4d8bfc3cd3498c` (immutable).
Model: openbmb/MiniCPM3-4B; INT4 affine groups of 128; BF16 compute; 62 layers, 40 heads, composite D=96 and zero-padded V=96.
Execution seat: gpt-6-astra, reasoning effort xhigh, confirmed by logs/apa_sp3_r2.log. No GPU model execution by this seat.

## G0 PROTOCOL-2 determinism

Historical context only: A=20.065 / B=19.817 came from an unrecoverable single guide window on an 8 GB RTX 3070. They are not targets. No further recovery attempted.
Authorized amendment: `orders/APA_SP3_AMENDMENT_2.md`; immutable implementation JSON `artifacts/apa_sp3/protocol_amendment.json`, SHA256 `db95e3ecb958ef6c31a105ee77be19c455ae8b210ed8a8082194389294843224`. Base registration is unchanged.
Offline wikitext-2-raw-v1 test: 4358 rows joined by newline, 1289979 characters, 333337 tokens. Stream file SHA256 `5684e72cbab28236391ebcac9113aeca5b318ce6eeed69c410f1094950269a28`; canonical little-endian int64 bytes SHA256 `d3c64882a9cbd4264a21f60ac239680d3a7d0631ab06d0d8a5d4fac8286d268a`.
Tokenizer: snapshot AutoTokenizer defaults; one BOS <s> id 1 at the beginning of the entire stream; no EOS, chat template or per-window BOS. Direct read-only cached test Arrow; no fallback corpus or cache writes.
Reference: `/mnt/ForgeRealm/GraftRepository/tests/minicpm3_bulkbits_floor.py::get_text` and `window_nll`. Reuse newline join and six consecutive disjoint windows. The reference loop scores 511 targets; amendment 2 explicitly requires all 512, which this scorer implements.
Feeding: one full prefill per window, fresh KV cache each time, six windows of 1024 at offsets 0,1024,2048,3072,4096,5120. Logits 511:1023 predict tokens 512:1024, fp64 log-softmax; PPL = exp(total NLL / 3072). Same feeding for every arm. Long rows use the prefix at token 0 and the last 512 in-input targets.
G0: four fresh processes A1,B1,A2,B2; repeats of each arm must agree within 0.001 PPL. Process identities and target hashes are checked. Repeat 1 supplies each baseline after both repeats pass. B-A within +/-0.3 PPL is a prediction, never a stop gate. The preserved in-process A/B safeguard compares six-window PPL to these fresh baselines at 0.001 before later model arms.
G0 miss stops model arms; D must still refine all eligible keys and satisfy |D-A|<=0.005. C calibration and G2 retain the registered single-prefix scope; their diagnostic fractions are labeled separately from six-window PPL.
G0 status: STALE; repeat differences: null; B-A: None; prediction met: None.

## Perplexity and prefill table — model perplexity

| Bits | S | Arm | Status | PPL last-512 | ms wall | Peak resident MiB* | Refined fraction | Layer min..max | δ |
|---|---:|---|---|---:|---:|---:|---:|---|---:|
| 4 | 1024 | A standard | STALE | — | — | — | — | —..— | — |
| 4 | 1024 | B apa_two_pass | STALE | — | — | — | — | —..— | — |
| 4 | 1024 | C apa_sp_matched | STALE | — | — | — | — | —..— | — |
| 4 | 1024 | D apa_sp_refine_all | STALE | — | — | — | — | —..— | — |
| 4 | 1024 | E apa_sp_provable | BLOCKED / unrun | — | — | — | — | —..— | — |
| 4 | 8192 | A standard | RED | — | — | — | — | —..— | — |
| 4 | 8192 | B apa_two_pass | STALE | — | — | — | — | —..— | — |
| 4 | 8192 | C apa_sp_matched | STALE | — | — | — | — | —..— | — |
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

The model comparison is incomplete: PROTOCOL-2 fresh-process G0 determinism and a scored C comparison at matched actual fraction are both required; see their individual statuses above. G2/G3 rows establish nothing about model quality by themselves. The ε=1e−3 margin is conditional on the measured finite error envelope; E’s separate capture checks its transfer, and neither check supplies a universal quantization bound or a CUDA rounding proof.

## Decode table — kernel sweep / in-model timing

| Bits | Starting S | Arm | Status | tokens/s | Steps | Prefill seconds |
|---|---:|---|---|---:|---:|---:|
| 4 | 2048 | A | STALE | — | — | — |
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
| Lead | P1 | D equals A within 0.005 ppl | UNASSESSED; P2 uses fresh B; requires applicable model receipts |
| Lead | P2 | C within 0.05 ppl of B at matched fraction | UNASSESSED; P2 uses fresh B; requires applicable model receipts |
| Lead | P3 | E fraction >=0.95, real eq >=0.8, E ppl within 0.01 of A | UNASSESSED; P2 uses fresh B; requires applicable model receipts |
| Lead | P4 | bulk4 p99 error <0.5*max; C skipped weight ratio >0.1 on some layer while ppl unmoved (P2 tolerance) | UNASSESSED; P2 uses fresh B; requires applicable model receipts |
| Lead | P5 | C/B decode tokens/s at S=32768 >=2 | UNASSESSED; P2 uses fresh B; requires applicable model receipts |
| Seat | S1 | G0 likely RED on current engine even after tokens recovered; documented matmul/softmax drift exceeds 0.01 | HISTORICAL PREMISE RETIRED by lead amendment 2; retained verbatim |
| Seat | S2 | D likely differs from A by >0.005 ppl: BF16 cuBLAS/softmax rounding differs from fused FP32 accumulation | UNASSESSED; P2 uses fresh B; requires applicable model receipts |
| Seat | S3 | E fraction >=0.95 at bulk4; conditional finite bound will not establish general low-precision usefulness | UNASSESSED; P2 uses fresh B; requires applicable model receipts |
| Seat | S4 | C within 0.05 of B if realised fraction matches; skipped weight ratio >0.1 likely | UNASSESSED; P2 uses fresh B; requires applicable model receipts |
| Seat | S5 | C/B full-model decode at S=32768 <2 because latent expansion, re-quantization and INT4 projections remain in both arms | UNASSESSED; P2 uses fresh B; requires applicable model receipts |
| Lead amendment 2 | B-A | Fresh B-A within +/-0.3 at bulk4, prediction only | UNASSESSED |

CPU gates: {"D96_pins": ["test_d96_prefill_contract_and_dense_pin", "test_d96_splitk_contract_and_dense_pin"], "GPU": "BLOCKED: cudaGetDeviceCount=100, device_count=0, no CUDA-capable device is detected; 0 GPU jobs", "adapter_cpu_import": "PASS; six modules pinned; no model instantiation; host_protocol2.json", "amendment_guards": {"names": ["test_protocol2_forged_amendment_red", "test_protocol2_stale_amendment_red", "test_protocol2_wrong_stream_sha_red"], "receipt": "artifacts/apa_sp3/guard_rejections_protocol2.json", "status": "PASS_ALL_REJECTED"}, "blind_verification": "lead-owned, unrun", "dry_run_cells": 692, "evidence_class": "unit test / host compile / code inspection", "fingerprint": {"artifacts/apa_sp3/adapter_import_cpu.json": "89f6369bbf700eff2209abc07b337775a59259c7abd465fa19be5072e9e6a490", "artifacts/apa_sp3/amendment_001_execution.json": "c680b376073a9790c7fb995c73d8ce9bbbc46512381133b0fae6e8861f9897b5", "artifacts/apa_sp3/amendment_002_native_bulk.json": "622b55b4b66a580fe38dd6f25e2471da5abb8dc7b76c590176a1b5618ffd1ba4", "artifacts/apa_sp3/amendment_003_B_native_scores.json": "b2689d07f8f70481d1f0740de256975a7be93cd902a4eb07831f94fd43bf00d5", "artifacts/apa_sp3/build/manifest.json": "cc66365b0733abd6c17e21bd2ed32420003a038d3440b33c82f49e5ecbc063e1", "artifacts/apa_sp3/registration.json": "d9b6511702a894f72174795141c72b2097b1bbe6110e810a1d4d8bfc3cd3498c", "artifacts/apa_sp3/weight_identity.json": "16d121d485ad7a21fdeb8cf2b881b8ecc1bb1532185e6fd425a8cbcdc8356ead", "scripts/apa_sp3_build.sh": "c9b28b49872d515deff42ba6ecf5c840f5ffab6c516533025f435208e0054c66", "scripts/apa_sp3_common.py": "2365412261effca3a0dbe4d80ee1b9b730510a3d50b52c826375a0c65cb261d5", "scripts/apa_sp3_control.py": "485cd771a747ec07b5f258be7b22c3c25608a2242d4b212542c5350e468ff734", "scripts/apa_sp3_diag_bindings.cpp": "e3612f50d4cc9456e79ab063889bc62071b2f3fa365e0890902bebe232d24f17", "scripts/apa_sp3_gpu.py": "9a680340294b9d0dcac3ca9fc731cb91304ae7f9253f597e43649707bdd33a09", "scripts/apa_sp3_lead_gpu.sh": "e5f970e7a29bee103f82c82bc0acf4318fd6087c76979ead4266ef60209fed86", "scripts/apa_sp3_make_diag.py": "2b01cb3c58434d88cc08f51797eb89fa101e527a1b093b3867cb26f99fd010ef", "scripts/apa_sp3_metrics.py": "26e9b9e13f15baae7069e981dea064939c8b232f8b43e3517625f44792c178a3", "scripts/apa_sp3_model.py": "e95f2ba8ee4407a46792c1c332cd7a31d62981775429ec6becf275f471507cf2", "scripts/apa_sp3_mutations.py": "affc907439e27b2c71d334093d9be819fb78d4eb4add0e0a2982f8a0b1cfb2c0", "scripts/apa_sp3_peak.cpp": "fb439af92e0e968ff2d09b7a53f58ddb69a32e442953ca455c89506615acb7bc", "scripts/apa_sp3_protocol2.py": "f585a942aae501f2efeac8469531ee315a3c09bee98853433aa1de7335c93899", "scripts/apa_sp3_report.py": "5c5e475cf62c9c4348b340488cc7c066ddc09062ce0676c9c773a8a1fb1b9b50"}, "g0_preflight": "PASS for g0_A_1; no lease/device/model needed", "host_build": {"command": "timeout --kill-after=5s 575s env PYTHONDONTWRITEBYTECODE=1 bash scripts/apa_sp3_build.sh", "log": "artifacts/apa_sp3/build_protocol2.log", "log_sha256": "3ef7b7a60d88ed51464b8f312cf59a875eeaec3cbf47c5c473ee6736d61d4f39", "manifest_sha256": "cc66365b0733abd6c17e21bd2ed32420003a038d3440b33c82f49e5ecbc063e1", "status": "PASS"}, "mutation": {"killed": 6, "manifest": "artifacts/apa_sp3/mutation_manifest_protocol2.json", "nonerror": 6, "rate": 1.0, "receipt": "artifacts/apa_sp3/mutation_results_protocol2.json", "source_pins_current": true, "threshold": 0.8}, "observer_host_gate": "PASS native no-device failure preserved; GPU allocation accounting remains untested", "pins": {"amendment_sha256": "db95e3ecb958ef6c31a105ee77be19c455ae8b210ed8a8082194389294843224", "preexisting_kernel_bodies": 107, "preexisting_source_files": 24, "registration_sha256": "d9b6511702a894f72174795141c72b2097b1bbe6110e810a1d4d8bfc3cd3498c", "unchanged": true}, "protocol": {"feeding": "one_full_prefill_per_window_no_cache_between_windows", "status": "REGISTERED_PROTOCOL_2", "token_count": 333337, "token_sha256": "d3c64882a9cbd4264a21f60ac239680d3a7d0631ab06d0d8a5d4fac8286d268a", "tokens_file_sha256": "5684e72cbab28236391ebcac9113aeca5b318ce6eeed69c410f1094950269a28"}, "pytest": {"failed": 0, "log": "artifacts/apa_sp3/cpu_protocol2_delivery.log", "log_sha256": "bc597605a824565d6cc6b7a4d38324edf67198280b8360727e56107cfb2c1075", "passed": 57, "skipped": 0}, "status": "PASS_CPU_ONLY", "supersedes": "CPU_GATES_FINAL.json (r1 retained unchanged)"}
D=96 tests: `test_d96_prefill_contract_and_dense_pin`, `test_d96_splitk_contract_and_dense_pin`, and compiled flag/device/scalar guards for both geometries. Native GPU numerical pins are delivered in job `kernel96`, unrun here.
Author tests and mutations are baseline evidence only. Independent blind verification under House Rules §8 is lead-owned and unrun; no subagents were launched.

## Lead commands and bounds

```bash
cd /mnt/ForgeRealm/Project-Tensor-wt-apa-sp3
timeout 30s bash scripts/apa_sp3_lead_gpu.sh list
timeout --kill-after=2s 590s bash scripts/apa_sp3_lead_gpu.sh run kernel96
# Four separately leased fresh processes, then aggregate determinism:
timeout --kill-after=2s 590s bash scripts/apa_sp3_lead_gpu.sh run g0_A_1
timeout --kill-after=2s 590s bash scripts/apa_sp3_lead_gpu.sh run g0_B_1
timeout --kill-after=2s 590s bash scripts/apa_sp3_lead_gpu.sh run g0_A_2
timeout --kill-after=2s 590s bash scripts/apa_sp3_lead_gpu.sh run g0_B_2
timeout --kill-after=2s 590s bash scripts/apa_sp3_lead_gpu.sh run g0
# Each resume invocation runs at most ONE cell; stops on RED/stale receipts:
timeout --kill-after=2s 590s bash scripts/apa_sp3_lead_gpu.sh resume 4
timeout 30s bash scripts/apa_sp3_lead_gpu.sh summary
```
Complete commands: `artifacts/apa_sp3/lead_commands.txt`; every cell and per-job estimate: `dry_run.json`. Optional secondary uses `resume 8` after primary prerequisites. No multi-hour automatic batch is launched.
Each leased operation: flock --wait 20, worker timeout 480s plus 5s own-child termination grace, foreground 30s cooldown; outer guard TERM at 585s plus 3s grace. Device/PID inspection fails closed. Never signal a foreign PID. Operator keeps right of way; advisory-lock cooperation is required.
Planning estimates for workers, not timings: each fresh baseline 60–480s; G0 aggregation 1–10s; model PPL/capture/match/decode 140–480s including twelve control prefills, or RED timeout/OOM; per-layer G2 3–60s at 1024, 30–480s at 8192. The 480s worker cap is unchanged; six-window controls increase deadline risk. Add 30s cooldown plus up to 20s lease / 40s setup-receipt overhead; outer bound 590s. Actual upper-bound compliance is enforced, not predicted.
Memory reasoning: a single BF16 40-head S² score tensor needs 5.0 GiB at 8192 and 80 GiB at 32768, before intermediates and ~2.9 GB model residency. Standard full prefill at 32768 cannot fit on 12 GB; its decode setup will record OOM. 8192 standard fit and B/SP full-prefill deadlines remain unverified. No alternate cache-building scheme is silently substituted.

## Prior art

| Work | What is reused / what SP3 adds |
|---|---|
| GraftRepository MiniCPM3 floor protocol (2026) | get_text newline join and six-window teacher-forced NLL reference; lead amendment fixes 512 targets, chooses full prefills and fp64. New immutable token manifest and fresh-process gate wiring; no new scoring algorithm. |
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

- Original registration token SHA remains null and original S1 is retained as historical; lead amendment 2 supplies the new protocol and token pins. This seat did not run fresh A/B or GPU gates; their receipt statuses are shown above. Current blocked receipt: `artifacts/apa_sp3/GPU_BLOCKED_PROTOCOL2.json`. Historical values are not targets.
- Floor reference deviations explicitly authorized by amendment: full-prefill feeding, 512 instead of 511 scored targets, fp64 instead of fp32 NLL. Default BF16 matches the adapter; no compute dtype deviation. Read-only Arrow load uses exactly the specified cached test split.
- Missing local SP2/SPD1 artifacts were read from sibling worktrees, read-only. SP1/SP1.1 registration hashes are corroborated by the local ledger and SP2 parent registration; their original JSONs are absent here.
- Native GPU behavior, diagnostic bit parity, BF16 D≈A, long-context fit/time and actual model quality remain untested. Kernel-body hash equality proves source preservation, not GPU correctness.
- Resident peak is an explicitly qualified estimate. Cold/warm effects and single-call prefill variability remain; decode measures 32 steps and includes expansion/quantization overhead common to arms.
- Full long-context captures use tens of GB of disk and native diagnostic masks add quadratic transient memory. Disk/OOM/time failures remain RED; no sampling reduction.
- No git, subagents, shell background jobs, live-service changes or foreign-process termination. Only explicitly bounded own child processes may be terminated by timeout. Host build/tests complete; lead GPU and blind verification pending.

Files: `scripts/apa_sp3_*`, `tensor_cuda/tests/test_apa_sp3.py`, `docs/APA_SP3_LEDGER.md`, and `artifacts/apa_sp3/` (registration, source pins, build, CPU/mutation receipts, exact commands, blocked JSON and this renderer output).

## A4 full-context reference — model perplexity

T: pinned HF MiniCPM3 snapshot, bf16 weights, full SDPA attention. Flash is preferred only when eligible on actual Q/K=96, V=64 tensors; otherwise efficient is forced. Math fallback is disabled. Actual backend is UNRUN until a T receipt exists.
1024 uses six independent windows / 3072 total targets. Long rows use prefix 0 / 512 targets. Only required LM-head rows are projected in new T and 32K engine cells. Attention is never chunked in these PPL/reference cells.

| S | Arm | Status | PPL | Engine minus T | SDPA backend |
|---:|---|---|---:|---:|---|
| 1024 | T | UNRUN | — | — | — |
| 1024 | A | STALE | — | — | — |
| 1024 | B | STALE | — | — | — |
| 1024 | C | STALE | — | — | — |
| 1024 | D | STALE | — | — | — |
| 8192 | T | UNRUN | — | — | — |
| 8192 | A | RED | — | — | — |
| 8192 | B | STALE | — | — | — |
| 8192 | C | STALE | — | — | — |
| 8192 | D | UNRUN | — | — | — |
| 32768 | T | UNRUN | — | — | — |
| 32768 | A | UNRUN | — | — | — |
| 32768 | B | UNRUN | — | — | — |
| 32768 | C | UNRUN | — | — | — |
| 32768 | D | UNRUN | — | — | — |

INT4 engine versus bf16 T gaps are observations, not RED parity failures. D@32768 requires T with identical targets; the 0.005 D/A gate remains confined to existing engine controls. D@32768 adds a layer-0 first-128-query refine-all check; no full 62-layer 32K diagnostic claim.

## A4 ceiling grid — kernel sweep / memory shape

| Arm | S | Status | Fit | Outcome | Peak resident MiB estimate |
|---|---:|---|---|---|---:|
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

Measured grid summary: `{"A": {"grid_complete": false, "max_successful_grid_S": null}, "B": {"grid_complete": false, "max_successful_grid_S": null}, "C": {"grid_complete": false, "max_successful_grid_S": null}}`.
A timeout is unknown fit, never an OOM or successful prefill. Every grid point is independent. Largest successful S is only a grid result, not an extrapolated capacity or model-quality finding.

## A4 capture split and immutable receipt handling

8192: [0,16), [16,32), [32,48), [48,62), at most 188s planning estimate each. 32768: 62 single-layer ranges, 188s each under the registered quadratic extrapolation. These are unmeasured estimates; timeout rails remain authoritative.
Ranges restore predecessor hidden activations; each layer sees the full token prefix. Row-block diagnostic replay is checked bitwise against native output. The unchanged margin ids depend on the original capture id, now an aggregation cell. New/changed receipts are in jobs_a4; legacy RED/PASS receipts are preserved.
Aggregation rehashes each layer manifest and checks exact stat identity of every array since its completed range SHA256. It pins all 62 manifests as one set. Margin workers rehash their input arrays. 32768 captures are B/C bulk4; no new 32768 margin/E calibration cells were authorized.

Fingerprint compatibility is governed by amendment_006_fingerprint.json: per-kind import closures and exact reviewed source transitions. Unknown closure changes reject reuse. Source eligibility and current runtime/build prerequisites are reported separately in a4_receipt_audit.json.

