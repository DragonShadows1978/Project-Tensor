# APA-SP4G amendment 5 — scored-call diagnosis

**RED. Not claimed fixed:** D32/A32 exactness, the source of the PPL gap, and C/E quality. Diagnostic PASS means capture completed; it is not a model-quality gate.

Historical card: D32=53.4723904674738, A32=49.9211789381388, absolute difference=3.55121152933498 > 0.005. All144 native dtype pins are complete. Receipt: `jobs_a4/ppl_a4_D32_2048_w0.json`.

## 1. Registered per-call comparison

Prediction (immutable017): **each of these calls disagrees beyond1e-3 relative Frobenius in fp32**, with the defect in SP cache/offset/causal/dispatch handling, not standard. This remains a hypothesis until measured.

| Cell | Actual PPL (L,S_all,offset) | Order-stated S_all | Dispatch from pinned source | State |
|---|---|---|---|---|
| `diag_a5_call_l05_b00` | (64,1087,1023) | 1088 | prefill `apa_selective_sp_kernel<T,512,DIAG>`,1024 CTAs x32 threads | UNRUN |
| `diag_a5_call_l05_b15` | (64,2047,1983) | 2048 | prefill `apa_selective_sp_kernel<T,512,DIAG>`,1024 CTAs x32 threads | UNRUN |
| `diag_a5_call_l47_b15` | (64,2047,1983) | 2048 | prefill `apa_selective_sp_kernel<T,512,DIAG>`,1024 CTAs x32 threads | UNRUN |

Geometry correction is source/receipt evidence: the1023-token prefix is512+511; the16 scored input blocks cover positions1023..2046 and predict targets1024..2047. Thus the actual key spans are1087 and2047. No added token or changed schedule. Registered shapes are not newly captured tensors where state is UNRUN.

The launcher condition is `L==1` for split-K; allL64 calls select prefill. This is a determination from the pinned launcher and observed arguments, not a GPU profiler trace. Both causal paths use last-key `S_all-L+i`. The per-call manifest records every query bound, `position_offset`, scale1, no sink, q/k/v bitwise parity, Kq repeat parity, four dtype/diagnostic launches, actual masks, and cache metadata. `kq_count=null` is explicit: these are tuple caches, not KVRing; the adapter reconstructs whole-span Kq.

| Cell | SP/A fp32 relF | SP/dense fp32 relF | A/dense fp32 relF | Outcome |
|---|---:|---:|---:|---|
| `diag_a5_call_l05_b00` | UNRUN | UNRUN | UNRUN | UNRUN |
| `diag_a5_call_l05_b15` | UNRUN | UNRUN | UNRUN | UNRUN |
| `diag_a5_call_l47_b15` | UNRUN | UNRUN | UNRUN | UNRUN |

Each capture also saves max-absolute and relative Frobenius comparisons in bf16, dense fp32 and staged-bf16 references, native standard intermediates, fp32-to-bf16 casts, and o_proj outputs. History is standard-A propagation for identical-input comparison, including at layer47; no D-history parity claim.

## 2. Change and conditional D32 rerun

No attention argument or kernel fix is claimed. A5 adds actual scored-block capture and a completed-result dependency. Correct arguments plus SP/dense disagreement and standard/dense agreement identify the SP path as the defect boundary for the lead; they do not identify the internal faulty operation. If all agree, the prediction fails at these calls and propagation remains the lead. Mixed results remain mixed.

`ppl_a4_D32_2048_w0` is registered for a fresh, create-only `jobs_a5/` receipt after all three diagnostics complete and at least one SP/A fp32 relF exceeds0.001. It reruns unchanged dtype-pinned D32. A confirmed harness argument fix requires a separately fingerprinted amendment. The old RED receipt is retained; C/E stay blocked behind the original0.005 gate and are not automatically released by an A5 rerun.

## 3. Ungated propagation

`diag_a4_propagation_A_2048_w0`, `diag_a4_propagation_D_2048_w0`, and aggregate `diag_a4_propagation_2048_w0` now run through the A5 runner and `jobs_a5/`. The two captures require A32 and D32 having completed; numerical D32 RED is accepted with full provenance, dtype, schedule and PPL checks. Missing, stale or partial worker results are rejected. The aggregate requires the two completed captures. None depends on an A5 per-call outcome or D32 passing.

- `diag_a4_propagation_A_2048_w0`: UNRUN.
- `diag_a4_propagation_D_2048_w0`: UNRUN.
- `diag_a4_propagation_2048_w0`: UNRUN.

## 4. Fingerprint, CPU gates, lead commands and safety

Registration017 SHA256: `68835c48c2fb800ed475bd25affc12920a8c1b54b2292dedefb3f7de70b4ccf0`. Order unchanged. Historical sources and receipts are pinned by `a5_before.json`; previous reports/commands are preserved in `a5_baseline/`.

- `amendment_018_a5_fingerprint.json` SHA256 `fc168bc3c51e0a3db6ec7d34f30992b002994ef1236669fd304023e9debf263b`.
- `CPU_GATES_A5.json` SHA256 `74c94eb3d8dd2a3e5b4ea63c9d4c3eb454a4bfb1fec13b945703a180ad5ba43f`.
- `DELIVERY_CHECKS_A5.json` SHA256 `2f357741ecc9069231fbc71a005f82ed13d44b916b967da6b043c5354734da91`.

Author CPU suite: 129 passed, 0 failed, 0 skipped; mutations 8/8, threshold0.80. No GPU or blind-review claim. Required shape test: `test_a5_capture_shapes_include_actual_L64_cached_blocks`; the actual scoring-driver test reaches all three registered blocks with CPU model doubles.

Run one command at a time from `lead_commands.txt`; each diagnostic is independent, so a failed call must not skip propagation. Use `bash scripts/apa_sp4g_a5_lead_gpu.sh run CELL`. The old A4 runner intentionally retains its immutable gate; use A5 for these amended cells.

Planning estimates:90–275s/model cell (load75–130s),1–30s/aggregate; unmeasured for A5. Cooperative worker deadline285s including load, foreground GPU lease wait20s, cooldown30s, disk floor12GiB. No signal timeout or process kills: Python-boundary checks cannot forcibly bound a hung native call. No GPU worker was started in this dispatched seat. All actual tool calls were foreground and under10minutes. No git, subagents, background jobs/waits, process signals, model writes, service operations, or kernel/product edits.

Local visibility: `a5_device_visibility.json`, CUDA100, count0, `no CUDA-capable device is detected`. `GPU_BLOCKED_A5.json` is the dispatched-seat GPU-blocked handoff; lead GPU measurements must produce their own receipts. Seat: **gpt-6-astra / reasoning xhigh**, `logs/apa_sp4g_a5_r1.log`. Model under test: Gemma-4-12B-it, QAT q4_0 exact symmetric-8 g32, bf16 engine, fp32 global-attention ablation.

## Prior art

SP4G A3/A4 (2026) supplies same-input replay, standard-branch instrumentation, dense comparison, dtype pins, fork/merge capture and residual sum-of-squares aggregation. A5 contributes scored-call selection, cache/dispatch receipts and the completed-result dependency only. [Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762) supplies the attention reference; [BLASST, Yuan et al. (2025/2026)](https://arxiv.org/abs/2512.12087), [FlashAttention-2, Dao (2023)](https://arxiv.org/abs/2307.08691), and [TurboQuant, Zandieh et al. (2025)](https://arxiv.org/abs/2504.19874) are inherited running-max, online-softmax and quantization prior art; kernels and quantizer remain unchanged. ThriftAttention/Sharratt (2026), inherited weight-sensitive precision: unverified — lead to check arXiv2605.23081. Ordinary Frobenius norms, NIST SHA256 (2001), Make/Feldman (1979) dependency DAGs, scoped instrumentation and cooperative deadlines are standard methods. DeMillo/Lipton/Sayward (1978) mutation testing: unverified — lead to check “Hints on Test Data Selection”. No prior art known to me for a distinct novel method introduced here; no novelty claim.
