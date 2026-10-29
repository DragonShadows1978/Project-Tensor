# APA-SPM1 report: memory ceiling, flash vs APA-SP (Qwen3-1.7B / Qwen3-4B, RTX 4070 SUPER 12 GB)

**Status.**
- Lead-run, Opus 5.5.
- Registration: `artifacts/apa_spm1/registration.json`, sha256 `52da8c871f9e68f51e6006dda45e7ef7e7ff3a07e5a2296a34de73bd3d5e0ce3`.
- Verdict: `artifacts/apa_spm1/verdict.json`, produced by `scripts/apa_spm1_report.py` from the run receipts.
- 51 ceiling runs, plus 4 G1 runs and 2 probes.
- Not yet cross-model reviewed (§8; see open items).

**Evidence class:** a measured device-memory ceiling (fit or OOM) with a real model and real tokens. It is NOT a
quality claim at long context, and NOT a speed claim.

## Answer

Flash holds the longest context on both models. APA-SP, as it is stored today, runs out of memory first.

| Model | F, flash (512 B per key per KV head) | Q, fork packed APA-SP (580 B) | P, production APA-SP storage (768 B) |
|---|---:|---:|---:|
| Qwen3-1.7B | **70,656** | 64,512 (**0.913×** flash) | 48,128 (**0.681×**) |
| Qwen3-4B | **24,576** | 22,528 (**0.917×**) | 16,384 (**0.667×**) |
| Analytic byte ratio | 1 | 0.883 | 0.667 |

Ceilings are at 1,024-token resolution. Each is the largest S0 that completes a 2,048-chunk prefill plus 16 decode steps.

Why: all three arms keep full-precision K and V on the GPU.
- P adds a bf16 copy of the 4-bit keys (`kq`), which is +50%.
- Q adds the 4-bit TurboQuant codes plus a norm per key, which is +13%.

Each ceiling therefore tracks bytes per token. The APA bulk tier is a compute and bandwidth saving (§2.3), not a storage
saving, because the exact copy it refines from stays resident.

## Gates

- **G1 wiring: PASS.** At S0=4096 on 1.7B, over 16 steps:
  - F vs the unmodified HF model: mean KL 8.1e-4, top-1 16/16.
  - P vs F: 2.4e-3, top-1 16/16.
  - Q vs F: 1.5e-3, top-1 16/16.

  Receipt: `artifacts/apa_spm1/g1.json`.
- **G2 accounting: PASS.** In all 41 runs that completed prefill, measured resident bytes equal the analytic bytes
  exactly.
- **G3 adjudication: PASS.**
  - Every OOM is a real exhaustion: free memory was 18–202 MiB at the last boundary.
  - Each failed while allocating a cache-growth tensor:
    - torch `cat` for F, for example "Tried to allocate 142.00 MiB" at decode step 1, layer 14, at S0=71,680;
    - the engine `from_host` upload of the new rows for P and Q.
  - Rungs above a failing rung fail at the same chunk and layer (4B F 26,624, 28,672 and 32,768 all fail at chunk
    24,576:26,624, layer 17), which is deterministic.
  - No non-monotone outcome. No foreign GPU process in any run.
  - At the ceiling, resident cache was 92–96% of the post-load free memory for every arm:
    - 1.7B: F 7,728, P 7,896, Q 7,993 of 8,364 MiB;
    - 4B: F 3,456, P 3,456, Q 3,589 of 3,984 MiB.

## Rungs

| Model | Arm | Rungs (S0 tokens: outcome) | Ceiling S0 |
|---|---|---|---:|
| 1.7B | F | 8,192 FIT, 16,384 FIT, 24,576 FIT, 32,768 FIT, 40,960 FIT, 49,152 FIT, 57,344 FIT, 65,536 FIT, 69,632 FIT, 70,656 FIT, 71,680 OOM, 73,728 OOM | **70,656** |
| 1.7B | P | 8,192 FIT, 16,384 FIT, 24,576 FIT, 32,768 FIT, 40,960 FIT, 45,056 FIT, 47,104 FIT, 48,128 FIT, 49,152 OOM | **48,128** |
| 1.7B | Q | 8,192 FIT, 16,384 FIT, 24,576 FIT, 32,768 FIT, 40,960 FIT, 49,152 FIT, 57,344 FIT, 61,440 FIT, 63,488 FIT, 64,512 FIT, 65,536 OOM | **64,512** |
| 4B | F | 8,192 FIT, 16,384 FIT, 24,576 FIT, 25,600 OOM, 26,624 OOM, 28,672 OOM, 32,768 OOM | **24,576** |
| 4B | P | 8,192 FIT, 16,384 FIT, 17,408 OOM, 18,432 OOM, 20,480 OOM, 24,576 OOM | **16,384** |
| 4B | Q | 8,192 FIT, 16,384 FIT, 20,480 FIT, 22,528 FIT, 23,552 OOM, 24,576 OOM | **22,528** |

## Predictions

| ID | Prediction | Result |
|---|---|---|
| P1 | ceiling F > Q > P on both models | **held** |
| P2 | on 1.7B, Q/F = 0.88 ± 0.06 and P/F = 0.67 ± 0.06 | **held**: 0.913 and 0.681 |
| P3 | no APA-SP arm exceeds flash | **held** |

Q measured slightly better than its byte ratio, 0.913 against 0.883. The binding transient is the same one-layer K
`cat` in every arm, and its absolute size is the same. Because the packed workspace is preallocated, Q's
last-moment growth is K and V only.

## How this squares with the earlier "APA-SP runs where standard OOMs" records

Those records are intact and correct, but they compare APA-SP with each port's standard path, which materializes the
S×S scores:
- MiniCPM3: APA-SP fits at 24,576, while standard OOMs at 16,384.
- GPT-OSS: APA-SP runs to 8–12K, while standard OOMs at 2,048.

The paper says so itself (§5: "A fused dense kernel (FlashAttention) avoids materialization too"), and so does SP5
finding 4. In MiniCPM3 the resident cache is the MLA latent: K, `kq` and V are expanded per forward as transients,
so the win there came from not materializing the scores, not from low-bit storage. Re-measuring those ports against
flash is a separate order.

## What WOULD beat flash on memory (PROJECTION, arithmetic only, not built)

This uses the same resident budget flash reached at its ceiling (1.7B: 7,728 MiB; 4B: 3,456 MiB). It assumes the
exact copy moves to host RAM and only refined rows are fetched per step.

| Design (GPU bytes per key per KV head) | 1.7B tokens | 4B tokens | vs flash |
|---|---:|---:|---:|
| Flash, measured (512) | 70,656 | 24,576 | 1.00× |
| Exact K on host; 4-bit codes + V on GPU (324) | ~111,700 | ~38,800 | ~1.58× |
| …plus V also 4-bit on GPU, exact V on host (136) | ~266,000 | ~92,500 | ~3.76× |

Unmeasured, and these decide whether it is worth building:
- the PCIe fetch cost per decode step;
- how stable the refined set is across steps;
- quality with low-bit V on the bulk keys.

Prior art to check before an order: InfiniGen (OSDI 2024), ShadowKV (2024) and Quest (ICML 2024), all of which select
and then fetch. Unlike APA, they drop the unselected keys from the softmax. KIVI and KVQuant (2024) quantize the cache
and discard full precision.

## Deviations

- Prefill for P and Q uses the fork's K1 entry, not SP1.1. SP1.1 needs 0.59 s per layer chunk at S=32K, which would
  exceed the 600 s lease near the ceilings. The probe measured a transient of ≤ 6 MiB for both.
- The harness bridges P and Q through host memory for the new rows only. The engine has no zero-copy torch interop.
  The OOMs for P and Q occur in that upload, which is a cache-growth allocation of the same size class as F's `cat`.
- Layer 0 runs APA in P and Q (SPQ7's quality configuration keeps it exact). This makes no memory difference.

## Open items

1. §8 cross-model review of this measurement. Codex is at no-new-seats until 10-09. Opus 5.5 or Grok (as an extra
   family) are possible.
2. Re-run the record ports (MiniCPM3, GPT-OSS) with a flash arm.
3. If the author wants the memory win, an order for the host-tier exact-K design, starting with a measurement of how stable
   the refined set is across decode steps.

## Files

- `orders/APA_SPM1_MEMORY_CEILING.md`, `docs/APA_SPM1_LEDGER.md`, `docs/APA_SPM1_REPORT.md`.
- `scripts/apa_spm1_{run,ladder,register,g1,report,probe,probe_k1,probe_ref}.py`, `scripts/apa_spm1_{gpu,all}.sh`.
- `artifacts/apa_spm1/{registration.json,registration.sha256,g1.json,verdict.json,rung_table.md,ladder_*.json,runs/}`.
- `logs/apa_spm1_*`.
- Engine binary (read-only, SPQ8 build): sha256 `a9e1b0ab2ca860c08bc52a276b27f7eab9a324b7cf35b561665f5b7fb4ece4bb`.

## Prior art

- FlashAttention-2 (Dao, 2023), via PyTorch SDPA with `causal_lower_right`.
- Chunked prefill: Sarathi (Agrawal et al., 2023) and vLLM (Kwon et al., 2023).
- KV accounting: vLLM PagedAttention (Kwon et al., 2023).
- TurboQuant (arXiv 2504.19874); BLASST (arXiv 2512.12087); APA (Perry, 2026).
- Contrast cited, not taken: KIVI (Liu et al., 2024), KVQuant (Hooper et al., 2024), InfiniGen (Lee et al., OSDI
  2024), ShadowKV (Sun et al., 2024), Quest (Tang et al., ICML 2024). These citations are from memory; lead to check
  through the proxy.
