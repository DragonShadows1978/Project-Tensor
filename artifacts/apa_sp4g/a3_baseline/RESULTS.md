# APA-SP4G A2 results

**RED: D/A exactness and old G2 replay remain unresolved on the card. A2 CPU PASS_CPU_ONLY.** This seat has no CUDA device; new diagnostics are prepared, not claimed executed or fixed. Historical A1 measurements below retain their original fingerprints.

## Model perplexity — QAT INT4 Gemma-4-12B-it; only eight global layers change

| S / scored targets | A standard | B two-pass4 r=.15 | D refine-all | D−A |
|---|---:|---:|---:|---:|
| 2048 / 4096 | 165.644274 | 167.649292 | 169.788756 | +4.144481 (2.502%) |
| 8192 / 512 | 38.863892 | 38.561350 | 39.382288 | +0.518396 (1.334%) |

Evidence class: model perplexity, `jobs_a1/ppl_{A,B,D}_{2048,8192}.json`. Short is four2048-token windows with **1024 targets each,4096 total**, as the immutable protocol and receipts specify (the lead card header said2048×4 targets). Window PPLs:

| Arm | w0 | w1 | w2 | w3 |
|---|---:|---:|---:|---:|
| A | 52.4805 | 537.9733 | 362.9962 | 73.4590 |
| B | 49.4104 | 571.4903 | 374.4083 | 74.7195 |
| D | 53.4724 | 582.5763 | 355.9433 | 74.9502 |

Raw untemplated wikitext PPL near165 is the known -it regime, not evidence by itself of a port failure. The June refine sweep recorded standard121.74 and attributed high raw-text PPL to the model’s template binding (`/mnt/ForgeRealm/GraftRepository/docs/GEMMA4_PORT_LEDGER.md:287-295`, historical source evidence; its HF control was pending). The52→538 window spread is context supplied by the lead as template-boundness; this amendment does not run a new templating experiment.

PROTOCOL-G unchanged: 292282 canonical int64 tokens, SHA `8bb85a61060d4221fef55f0b134914c55e6110ffbfd10913aed54e99f130d1cb`; raw test newline join, -it default tokenizer, no chat template; cached64-query blocks, fp64 NLL, aggregate exp(total NLL/targets). `registration.json` SHA `099a8bd9e1bb94909a81c110521d2d3a9d642d5fac27d49f6820b97cb3fbbd1e`. B fused at threshold0, fast_max_seq0, scale1, bulk4; clean decode incremental kq_count retained. No torch/BF16-weight reference.

## Hypotheses and registered card cells

| Hypothesis | Diagnostic cell | Current finding / decision |
|---|---|---|
| a: K/V source | `diag_a2_source_2048` (UNRUN) | Source inspection argues against storage cost: QUANT_V/QUANT_KV4 off, standard and APA prefill share exact k/v tuples; reconstructed kq is separate. Runtime same-state Q/K/V identity, value comparison, and standard-over-APA-input check pending. |
| b: scale/softcap | `diag_a2_scale_2048` (UNRUN) | Both global prefill routes scale1.0; no attention-logit softcap in either branch. Final-logit processing is outside this fork. Runtime scale and same-tensor outputs pending. |
| c: shared K=V projection | `diag_a2_value_2048` (UNRUN) | Shared kraw feeds K norm and scale-free V RMSNorm; V is not roped. Same V variable at both forks. Per-layer same-state comparison pending. |
| d: p-RoPE/qk-norm order | `diag_a2_rope_2048` (UNRUN) | Q/K RMSNorm then p-RoPE precede both branches (gemma4_tc.py:523-544); repeat same x/cos/sin/offset/cache, compare post-fork Q/K bitwise. Runtime pending. |
| e: accumulation/intermediates | `diag_a2_fp32_2048_w0` (UNRUN) | Concrete numerical-path difference: A materializes bf16 QK logits and probabilities; D keeps fused FP32 dot/softmax/value accumulation. A32 precursor plus D32 one-window treatment and per-call comparisons quantify it; no full-model FP32 claim. Cause of PPL gap remains unnamed until card treatment. |

Registration before code/gates: `amendment_006_a2_hypotheses.json` and `amendment_007_a2_execution.json`. The four source/scale/value/RoPE jobs each run standard window0 and repeat its first global call per layer from the same x, cos, sin, offset and immutable tuple cache. They compare Q/K/V and post-output projection, both bf16 and FP32 attention outputs. A completed diagnostic PASS means a measurement exists, not that a parity hypothesis passed. Full receipt probes are authoritative.

**A′ decision:** standard over the exact K/V at the APA entry is implemented only as a registered diagnostic comparator (`standard` in a2_model.py); storage quantization is explicitly off. No independent AP reference arm is adopted or storage cost claimed without runtime evidence. The same-state test returns original A output to preserve its propagation. Kq is never substituted for exact K. If storage evidence contradicts source inspection, lead may activate A′ with separate PPL receipts and report A′−A.

**Tolerance proposal to lead:** preserve abs(D−A)≤0.005 PPL and both existing REDs. There is no cause-confirmed Gemma-specific replacement rule yet. Conditional on confirmed storage difference, use abs(D−A′)≤0.005 and report storage cost separately. Do not widen0.005. Four PPL points exceed MiniCPM3’s0.0005 refine-all evidence by orders of magnitude; this seat does not dismiss the gap as rounding. BF16 intermediate rounding is a specific candidate requiring the FP32 treatment. A32/D32 retain QAT/bf16 everywhere except global attention; they cannot establish full-model FP32 parity.

## Replay finding and change

All eight historical8192 B margins are RED `NATIVE_REPLAY_NOT_BITWISE`. Source inspection found no atomic reduction in the active D512 B kernel and no cuBLAS blend in the forced-fused route. The old harness reconstructs128-query bands and uses final-prefix Kq for early calls. It did not preserve original-call Kq or masks. Quantizer FP32 matmul dimensions change with prefix length, so row independence in real arithmetic does not prove bitwise Kq identity. This is a concrete replay-context defect in the evidence chain; its responsibility for the observed error versus native nondeterminism is **not yet determined**.

`diag_a2_replay_B_8192_l05` repeats the first captured512-query call with final Kq and with regenerated original-prefix Kq, then compares the old128-band shape. It reports repeat variation, Kq/context treatment effect, max-abs and relative error (denominator max(abs(reference),1e-30)), without accepting a tolerance. Old capture regeneration is diagnostic only; it cannot certify the original mask.

The correction records Q/K/Kq/V/output and packed native mask at every call during the **actual PPL run**, using lossless bf16 bit payloads. G2 replays the entire original call and requires bitwise output **and mask** before tiling FP64 statistics. No selection is recomputed on statistical bands. The PPL output remains the original native output and the diagnostic copy must match it bitwise. New cells `ppl_capture_{B,C}_{2048,8192}_w0` and `margin_a2_{B,C}_{2048,8192}_l{05,11,17,23,29,35,41,47}` share dependency hashes and exact selected/pair counts. C remains blocked by original exactness/calibration; no bypass.

Population amendment: short G2 uses actual window0 PPL queries0..2046, long0..8190 (2047/8191 rows), every eligible pair across16 heads. Last input token is a scored target, not an executed query. This is explicitly distinct from old independent prefix captures of2048/8192 rows. All-population nearest-rank percentiles are retained; no sampling or averaging band percentiles. Extra capture/I/O cost may hit285s and must remain RED. No end-to-end margin fix claimed before card replay.

## Rail non-fits and clean decode

`RAIL_NONFITS_A2.json` records every `ppl_*_16384` and16K/24K/32K ceiling as **RAIL non-fit under285s**. This is a time-budget non-fit; memory capacity is unknown, not OOM. Five observed returncode124 terminations are distinguished from administratively deferred unrun rows. No retries scheduled. All additional≥16K work waits for a lead long lease.

| Arm | 4096 ceiling | 8192 ceiling | decode2048 ms/token | decode8192 ms/token |
|---|---|---|---:|---:|
| A | FIT | FIT | 31.3229 | 73.3171 |
| B | FIT | FIT | 38.7183 | 87.7506 |
| D | FIT | FIT | — | — |

Evidence: corresponding `jobs_a1/ceiling_*` and `decode_*` receipts. G2/G3 rows establish nothing about model quality by themselves.

## Fingerprints, CPU gates and lead commands

30 historical PASS receipts remain valid, including kernel512, all A/B/D PPL, measured decode/ceiling and capture_B_8192. All historical RED bytes remain unchanged. All A1 scripts, tests, runner and build are byte-identical; A2 renders the refreshed report through its isolated runner. Old captures remain historical evidence but lack actual-PPL per-call masks required by new G2. No old RED is reused as PASS. New jobs write create-only `jobs_a2/` with a separate complete closure. Unknown transitions reject. `amendment_008_a2_fingerprint.json`, `CPU_GATES_A2.json`, `GPU_BLOCKED_A2.json`, and `receipt_audit_A2.json` provide exact inventories.

CPU gate: `{"status": "PASS_CPU_ONLY", "registration_sha256": "099a8bd9e1bb94909a81c110521d2d3a9d642d5fac27d49f6820b97cb3fbbd1e", "evidence_class": "author CPU suite / lossless payload, context/mask tests / copied-source mutation baseline; no GPU claim", "pytest": {"passed": 67, "failed": 0, "skipped": 0, "original_tests": 48, "a2_tests": 19, "log": "logs/apa_sp4g_a2_cpu_final.log", "sha256": "83f8750ff0ac5586089b5ba7645551b77986546567093a2ad5cf84d3c9f80990", "warnings": "2 SWIG import warnings and final swigvarlink warning retained"}, "mutation": {"status": "PASS", "killed": 8, "nonerror": 8, "rate": 1.0, "threshold": 0.8, "receipt": "artifacts/apa_sp4g/mutations_a2/results.json", "sha256": "50e6338865bd5ed7495559dbbac9d438cd80995bde9fdc3ec13bda0e37a7ce56"}, "fingerprint_amendment_sha256": "f484d1619e9010756575800e8fcda37c5df4c930dbc6e84bbf85002c60798b6f", "historical_pass_receipts_valid": 30, "historical_red_receipts_preserved": 15, "build": "UNCHANGED_FROM_CARD_PASS; no rebuild", "blind_verification": "lead-owned UNRUN"}`. Author baseline only; blind verification remains lead-owned UNRUN.

Use refreshed `lead_commands.txt` and `cells.json`. Each command is one foreground lease; no batch. A2 runner rejects historical/long cell IDs. Original runner and report generator are retained to preserve measured closures; use the A2 summary and do not use the old long schedule.

```bash
cd /mnt/ForgeRealm/Project-Tensor-wt-apa-sp4g
bash scripts/apa_sp4g_a2_lead_gpu.sh list
bash scripts/apa_sp4g_a2_lead_gpu.sh run diag_a2_source_2048
bash scripts/apa_sp4g_a2_lead_gpu.sh resume
bash scripts/apa_sp4g_a2_lead_gpu.sh summary
```

Per-cell wall estimates are registered and unmeasured for A2: parity/FP32/PPL-capture60–285s; replay5–90s; margin2048 10–90s,8192 60–270s. A1 measured QAT loading mostly75–76s (first A84.7s); original30–120s planning range retained. Extra paired attention and disk writes may hit the rail. Worker285s TERM/290s hard, outer588s, lease wait20s, foreground cooldown30s. No A2 GPU worker or model load was started in this seat.

## Prior art

A2 reuses the June Gemma port/floor (2026) MQA standard branch and quantizer; SP3 (2026) same-tensor references, native-mask replay and exact dependency provenance; IEEE754 (2019) float bit representation and NumPy packbits (system year unverified — lead to check: NumPy packbits bitorder release). Controlled same-state and precision ablations are standard experimental methods; no prior art known to me for any distinct novel method introduced here, and no novelty claimed. New work is diagnostic/capture wiring. Code sites and ledger carry the same annotations.

- Gemma June 2026: port ledger and floor script: model/caches/flags/feeding reused; new experiment harness.
- SP3 2026: calibration, provenance, receipts, clean decode lesson; new architecture wiring.
- BLASST Yuan et al. 2025/2026: https://arxiv.org/abs/2512.12087; running-max comparison; APA refines keys and retains denominator.
- ThriftAttention Sharratt 2026: https://arxiv.org/abs/2605.23081; precision-selection/softmax-weight motivation; no FP4 implementation port.
- FlashAttention-2 Dao 2023: https://arxiv.org/abs/2307.08691; existing online softmax/work partition; no new kernel.
- TurboQuant Zandieh et al. 2025: https://arxiv.org/abs/2504.19874; existing rotated scalar-codebook reconstructed BF16 kq; no QJL residual.
- SP2 2026: conditional delta log(1/epsilon)+2eq reused; empirical finite max not universal proof.
- standard methods: NLL Shannon1948, nearest-rank statistics, bisection, content hashing NIST2001, leases, Make Feldman1979 dependency invalidation; no novelty claimed.

Prior literature annotations are retained from the immutable registration; no new external experiment reproduced. BLASST running-max comparison, ThriftAttention weight-sensitive precision, FA2 online softmax and TurboQuant reconstructed-key quantization are inherited, not newly implemented.

## RED, process safety and identity

Not claimed fixed: D/A PPL exactness; actual GPU replay; whether old error is context or nondeterminism; FP32 treatment; C/E calibration and quality; new margin rail clearance; device memory ceilings above8192. No tolerance was widened. No card result is fabricated from CPU tests. No GPU device (cudaGetDeviceCount100, count0, `no CUDA-capable device is detected`).

No git, subagents, background jobs/waits, process kills/signals, live-service changes, model writes, product/kernel edits or SP3 edits. All executed commands foreground and under10min. Driver retains bounded owned-child timeout semantics for lead use.

Seat: gpt-6-astra, reasoning xhigh, recorded in `logs/apa_sp4g_a2_r1.log`. Model under test: Gemma-4-12B-it QAT q4_0 symmetric-8 group32, bf16 engine compute.

A2 measured diagnostic results: `{}`. A2 failures: `[]`.
