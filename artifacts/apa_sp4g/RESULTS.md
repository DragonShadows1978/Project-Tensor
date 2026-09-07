# APA-SP4G amendment 3

**RED: D/A model exactness remains unresolved. A3 diagnostics are prepared; no harness fix, kernel change, or A′ comparator is claimed validated. C/E remain blocked.**

The fork/merge audit is in `docs/APA_SP4G_LEDGER.md` and its hash-pinned copy `A3_FORK_MERGE_AUDIT.md`. Source inspection finds identical scale1, no sinks, bottom-right causal mask, H16/KV1 mapping, exact K/V, and shared output projection/block scalar. Prediction: no single argument correction; inspect the cast/o_proj boundary and downstream propagation if nominal FP32 outputs agree. This is reasoning, not a measured cause.

## Existing card evidence

| Measurement | Value / finding | Evidence |
|---|---|---|
| `diag_a2_source_2048` | PPL 52.480487085; PASS diagnostic; max per-call FP32 difference 4.00543212890625e-05 | `jobs_a2/diag_a2_source_2048.json` |
| `diag_a2_scale_2048` | PPL 52.480487085; PASS diagnostic; max per-call FP32 difference 4.00543212890625e-05 | `jobs_a2/diag_a2_scale_2048.json` |
| `diag_a2_value_2048` | PPL 52.480487085; PASS diagnostic; max per-call FP32 difference 4.00543212890625e-05 | `jobs_a2/diag_a2_value_2048.json` |
| `diag_a2_rope_2048` | PPL 52.480487085; PASS diagnostic; max per-call FP32 difference 4.00543212890625e-05 | `jobs_a2/diag_a2_rope_2048.json` |
| `diag_a2_fp32_A_2048_w0` | PPL 49.921178938; PASS diagnostic; max per-call FP32 difference 4.00543212890625e-05 | `jobs_a2/diag_a2_fp32_A_2048_w0.json` |
| `diag_a2_fp32_2048_w0` | PPL 53.472390467; PASS diagnostic; max per-call FP32 difference 4.38690185546875e-05 | `jobs_a2/diag_a2_fp32_2048_w0.json` |

16/16 B margin receipts PASS. The corrected actual-PPL original-call capture establishes bitwise replay of output and native selection; lead confirms the old context bug. Old RED receipts are preserved. C capture/margins reported `FileNotFoundError` by the lead because calibration never ran behind the RED exactness gate; no C result is inferred.

A32=49.9211789381 improves on A bf16=52.48049 by about2.56 PPL; D32=53.4723904675 is essentially unchanged from D bf16. D32−A32=3.5512115293. All144 per-call FP32 A/SP comparisons in each precision receipt have max_abs≤4.38690185546875e-5. This tension requires direct reference and merge measurements; it does not identify an A-only semantic operation or justify widening0.005.

Historical model PPL: four2048-token windows,1024 scored targets each: A165.644, B167.649, D169.789;8192 last512: A38.864, B38.561, D39.382 (rounded lead card values; exact `jobs_a1/ppl_*` receipts authoritative). Historical full report preserved at `a3_baseline/RESULTS.md`; its A2 pending-state narrative is superseded here. High raw-wikitext PPL and52→538 window spread are the lead’s known -it template-bound regime, not a new template experiment. Model input and feeding remain PROTOCOL-G. All≥16384 RAIL non-fits remain deferred under A2; no retries or OOM reinterpretation.

## Registered diagnostics and interpretations

| Cell | Input / work | Outcome meaning | Status |
|---|---|---|---|
| `diag_a3_call_l05_c0` | first actual prefix call L512/S512/offset0 | Tests nominal full-prefix mask, scale, sinks and head/V mapping; saves native QK/probabilities and fork/merge outputs. | UNRUN / GPU blocked in seat |
| `diag_a3_call_l05_c1` | second actual prefix call L511/S1023/offset512 | Adds cached-prefix causal offset; discrepancy appearing only here directs attention to cache/offset handling. | UNRUN / GPU blocked in seat |
| `diag_a3_sweep_l05` | both saved calls;24 FP32 combinations each | Scale1 or512^-0.5 × sink None/zero × causal on/off × native/absolute-rowwise/zero-rowwise offsets; classify single-change matches across BOTH calls. | UNRUN / GPU blocked in seat |

The real PPL prefix is1023 tokens, so c1 is511 queries. Neither forcing flag enters the adapter chunk-size formula. Prior calls propagate original A. Each diagnostic stops by a caught local exception after the selected attention finishes; it measures no PPL. Saved arrays include q/k/kq/v, native scores/probabilities/attention, A/SP bf16 and FP32 outputs, dense NumPy FP32, staged-bf16 NumPy, and outputs after FP32→bf16 and o_proj. Exact source tuple-cache/QKV/replay checks reject changed inputs or missing operations.

Per-call max-abs and relative Frobenius `||candidate-reference||F/max(||reference||F,1e-30)` use float64 reductions. Numerical FP32 agreement requires BOTH max-abs≤1e-4 and relative-Frobenius≤1e-5. BF16 metrics are descriptive; no BF16 acceptance tolerance or PPL widening. NumPy dot/softmax/value arithmetic is FP32. Staged-bf16 emulation is descriptive and does not claim bitwise cuBLAS summation. Native A/isolation/replacement projection and SP diagnostic/native output must replay bitwise; D’s mask must select every eligible key.

None maps to nullptr in the SP binding. Zero sink means an extra zero-logit/zero-value denominator term. The SP ABI has no offset argument: explicit absolute and lost-offset probes invoke existing L1 split-K on each allowed key prefix. This changes geometry and is labelled as such; it is not a product fix. Causal-off offset labels reuse their literally identical native calls. BF16 nominal controls live in the call cells; the suspect sweep is FP32 to separate large semantic differences from materialization effects.

## Fix or A′ decision path

- **nominal_agrees:** If both nominal fp32 branches match dense and each other, no argument culprit established. Quantify cast/projected differences; stop RED pending lead propagation investigation. Do not automatically activate A prime or C.
- **single_argument:** Only candidate differing in ONE effective suspect, correcting a nominal mismatch on BOTH calls, and consistent with dense expected semantics qualifies as a candidate. Preserve diagnostics, name exact harness defect, append immutable treatment registration/source fingerprint; re-run D window0 with abs(D-A)<=.005, then original full short/8192 exactness before original C calibration/freeze/PPL/margins. No present auto-unblock.
- **semantic:** If standard differs from dense by a named reproducible operation absent in SP, leave kernel unchanged. Append A prime registration naming exact removed op, code boundary, A prime-A cost, same inputs/flags/feeding; gate D vs A prime at .005. Lead chooses reference and calibration DAG. Unknown op cannot be registered as executed.
- **neither:** If neither or multiple inconsistent candidates match, retain RED; no knob selection or tolerance widening.

Reserved successor IDs: `ppl_a3_D_fixed_2048_w0`, `ppl_a3_AP_2048_w0`, `exactness_a3_D_AP_2048_w0`. They are conditional registration paths, not executable cells: the runner rejects them until a separate immutable amendment names the measured culprit, exact harness treatment or removed operation, fingerprints, reference and dependencies. There is no known op to remove honestly before these diagnostics. A named harness fix must first clear original D-vs-A window0 at0.005; original full-short and8192 exactness then protect existing C calibration/freeze/PPL/margins. A′ requires lead adoption of the like-for-like comparator and A′−A cost reporting. Existing exactness REDs are never rewritten.

## Measured A3 comparisons

No A3 card receipts yet. No fabricated output-error numbers or culprit.

## Fingerprints, CPU and lead handoff

Registration010 SHA `d2da08a10e08413887bced5da9829cbfbca0e1ffb5545f1cdae3f63db5e46852`. Source/payload/receipt identity is fail-closed in new `jobs_a3/`. All23 preexisting execution files and72 historical receipt files retain exact bytes; no old receipt is upgraded. Original registration, compiled build, product adapter and kernels stay pinned. Source closure and validation identities: `amendment_011_a3_fingerprint.json`, `CPU_GATES_A3.json`, `GPU_BLOCKED_A3.json`, `DELIVERY_CHECKS_A3.json`.

CPU gate: `{"status": "PASS_CPU_ONLY", "passed": 88, "failed": 0, "skipped": 0, "wall_s": 0.9, "baseline_log": "logs/apa_sp4g_a3_cpu_final.log", "baseline_log_sha256": "2f5bc73e423e7454fad8d6f51862602470b3dbcec2e41cf1cfdf54d04a8ec8c6", "first_baseline": "logs/apa_sp4g_a3_cpu_01.log", "first_passed": 87, "followup_reason": "amendment_012: direct standard-NumPy check; finite tolerance agreement is not transitive. Bounds unchanged.", "mutations": {"path": "artifacts/apa_sp4g/mutations_a3_final/results.json", "sha256": "1abc501388702d79570df224878644a2ce725d2019b5f758f359a2e9496d6f86", "killed": 10, "nonerror": 10, "rate": 1.0, "threshold": 0.8}, "python_ast_files": 9, "shell_syntax": "PASS", "warnings": "2 existing SWIG import warnings plus final swigvarlink warning retained unsuppressed", "fingerprint_amendment_sha256": "fdd3cf12deea588373c6b3e804255bb2c53dd0a9985c31022f9e4e115e8cbcb1", "evidence_class": "author CPU baseline and copied-source mutations; not blind review or GPU measurement", "blind_review": "lead-owned UNRUN", "GPU_workers": 0, "model_loads": 0}`. Author baseline and copied-source mutants only; blind review remains lead-owned UNRUN.

Use `lead_commands.txt` (three dependency-ordered foreground calls) and `cells.json`. Estimates: each capture90–250s including model load; sweep30–270s with no model load. Historical QAT load≈75–85s, planning30–120s; A3 estimates unmeasured. Worker285s/hard290s, outer588s, lease≤20s, cooldown30s; ≥2GiB disk headroom. Every cell has its own receipt and log. Stop on RED; no background chain.

```bash
cd /mnt/ForgeRealm/Project-Tensor-wt-apa-sp4g
bash scripts/apa_sp4g_a3_lead_gpu.sh run diag_a3_call_l05_c0
bash scripts/apa_sp4g_a3_lead_gpu.sh run diag_a3_call_l05_c1
bash scripts/apa_sp4g_a3_lead_gpu.sh run diag_a3_sweep_l05
bash scripts/apa_sp4g_a3_lead_gpu.sh summary
```

## Prior art

June Gemma port/floor (2026) supplies the standard MQA branch, exact shared KV, chunking and PPL protocol. SP3 (2026) supplies same-tensor comparison/native replay and registered receipt dependencies; Make/Feldman (1979) and SHA256/NIST (2001) are provenance precedents. Dense attention is the established [Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762) operation, with ordinary stable softmax and Frobenius error norms. IEEE754 (2019) informs ties-to-even BF16 bit rounding. New work is capture, comparison and bounded sweep wiring; no novel attention/selector/optimization claim. No prior art known to me for a distinct novel method introduced here.

Inherited SP uses [BLASST, Yuan et al. (2025/2026)](https://arxiv.org/abs/2512.12087) running-max comparison, [ThriftAttention, Sharratt (2026)](https://arxiv.org/abs/2605.23081) weight-sensitive precision motivation, [FlashAttention-2, Dao (2023)](https://arxiv.org/abs/2307.08691) online softmax/work partition, and [TurboQuant, Zandieh et al. (2025)](https://arxiv.org/abs/2504.19874) reconstructed-key quantization. No kernels or selection rules changed. Author mutation gates follow DeMillo/Lipton/Sayward (1978), unverified — lead to check: Hints on Test Data Selection. A3 externally checked arXiv titles/authors/years for Vaswani, FA2, BLASST and TurboQuant; other literature annotations are inherited from original registration, not a new reproduction.

## RED, safety and identity

Not claimed fixed: D/A exactness, a named argument or A-only-op cause, C/E calibration/quality, new GPU numerical comparisons, long-context memory capacity. G2/G3 rows establish nothing about model quality by themselves. GPU unavailable to this seat; exact visibility receipt is `a3_device_visibility.json`. Zero GPU workers and zero model loads started here. No git, subagents, background jobs/waits, process kills/signals, services, model writes or product/kernel edits. All executed calls foreground and under10min. Runner preserves existing bounded owned-worker timeouts for the lead.

Seat: **gpt-6-astra / reasoning xhigh**, `logs/apa_sp4g_a3_r1.log`. Model under test: **Gemma-4-12B-it QAT q4_0 symmetric-8 group32**, bf16 engine compute; original -it token stream unchanged.
