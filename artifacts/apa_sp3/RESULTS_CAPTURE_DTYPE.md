# APA-SP3 amendment 3 — CPU PASS; GPU capture and continuation RED

Evidence class: unit test / code inspection for this repair. Existing card observations below are model perplexity or kernel sweep as labeled; all ten existing job fingerprints are now STALE. Receipt files and their original statuses/numbers are byte-identical. No native GPU capture rerun by this seat. **Not claimed fixed on GPU.**

## Fix and CPU evidence

`scripts/apa_sp3_model.py:223-226`: convert each diagnostic mask with `mask.float()` before the existing `tc.cat`. The native diagnostic emits uint8 0/1. Float32 represents both exactly; `capture` retains its uint8 cast, int64 selected sum, eligible-pair count, little-endian `packbits`, causal guard and native-output parity check. No native attention arithmetic or dispatch changed.
`tensor_cuda/tests/test_apa_sp3.py::test_capture_blend_nonfloat_cat_preserves_packed_masks` uses a float-only stub cat. Both uint8/int32 cases reproduce the exact original failure: `RuntimeError: op supports float32/float16/bfloat16 only` (`cpu_capture_dtype_repro.log`). The fixed source passes exact byte/count oracles across three query chunks, a partial final chunk and rectangular causal geometry; original q/k/kq/v argument and output identity checked.
**59/59 CPU tests PASS, 0 failed, 0 skipped**, including all 57 unchanged tests; **6/6 registered nonerror mutants killed**, threshold 0.8 unchanged. Receipts: `CPU_GATES_CAPTURE_DTYPE.json`, `cpu_capture_dtype_final.log`, `capture_dtype_mutation_r3/mutation_results_final.json`. All protocol forgery/staleness/stream guards rerun. Author baseline only; independent blind verification is lead-owned and unrun.
`host_capture_dtype.json`: 24 registered production files / 107 kernel bodies preserved; original test file and model file SHA reconstructed by reversing only the additions. Adapter/import/protocol/build/weight-stat pins PASS. Existing binaries were not rebuilt: no build input changed, all module SHA pins and compiled CPU guards reverified. All 692 dry-run cells enumerate in dependency order. Device observation: cudaGetDeviceCount=100, device_count=0, `no CUDA-capable device is detected`. Observer probes preserve the same native failed-allocation error with and without preload; no successful GPU allocation.

## Fingerprints and historical lead observations

The exact complete fingerprint map is compared in `apa_sp3_common.py::require_pass`. Changed keys: `scripts/apa_sp3_model.py` and added `artifacts/apa_sp3/amendment_004_capture_dtype.json` (the existing amendment glob includes it). No exception exists for observational edits or kernel receipts. `receipt_fingerprint_capture_dtype.json` records all ten unchanged receipt SHA values and actual stale-dependency rejections for every former PASS.

| Job | Original status | Historical card observation | Current applicability |
|---|---|---|---|
| kernel96 | PASS | D=96 all-dtype prefill/split-K pins PASS (kernel sweep; no model quality) | STALE; retained |
| g0_A_1 | PASS | PPL=8.655827855 (model perplexity) | STALE; retained |
| g0_B_1 | PASS | PPL=8.778802081 (model perplexity) | STALE; retained |
| g0_A_2 | PASS | PPL=8.655827855 (model perplexity) | STALE; retained |
| g0_B_2 | PASS | PPL=8.778802081 (model perplexity) | STALE; retained |
| g0 | PASS | A=8.655827855; B=8.778802081; repeats identical; B-A=+0.122974226 (model perplexity) | STALE; retained |
| ppl_b4_D_1024 | PASS | PPL=8.655351879 (model perplexity) | STALE; retained |
| ppl_b4_A_8192 | RED | `RuntimeError: cudaMalloc failed: out of memory` | STALE; retained |
| ppl_b4_B_8192 | PASS | PPL=10.403578496 (model perplexity) | STALE; retained |
| capture_b4_B_1024 | RED | `RuntimeError: op supports float32/float16/bfloat16 only` | STALE; retained |

Historical D-A=-0.000475975: lead P1 HIT on the prior source fingerprint. Historical B-A lies within the registered +/-0.3 prediction. These observations remain recorded; they cannot satisfy current dependencies without lead disposition.
`ppl_b4_A_8192` remains RED, registered non-fit: **do not retry**. Its D/8192 dependent also remains blocked. `capture_b4_B_1024` remains the failed original receipt; model code is CPU-fixed, card validation pending. The existing immutable receipt and existing capture directory also preclude a direct retry. No files were removed or receipts rewritten to enable continuation.
`CPU_GATES_PROTOCOL2.json` also has a stale fingerprint, and `mutation_manifest_protocol2.json` has a stale model-source pin. Both remain historical artifacts; the new CPU and mutation receipts cover the repair.

## B call-site audit and C fraction

**Literal SAME call site / SAME q/k/kq/v confirmation: RED.** Source inspection of the read-only MiniCPM3 adapter (`core/minicpm3_tc.py:269-298`) shows:

| Arm at S=1024 | Adapter call and hook | Value input |
|---|---|---|
| B | line 287 `_cublas_blend_attention` -> `Model.blend`, `B_native_blend` | original V64 |
| C/D/E | line 295 `tc.apa_selective_attention` -> `Model.selective` -> SP | V64 concatenated with 32 zeros; output sliced to 64 |

`Model.set` keeps B fast_max_seq=4096 and C/D/E=0. Both branches use the same q_full/k_full/kq construction before dispatch and the same original V64 values before padding. Across separate arm executions later-layer activations can differ. Dispatch is unchanged by this mechanical repair. Claiming a literal common function call or identical V tensor would contradict the pinned source; not claimed fixed.
C matches the **actual native blend score-promotion fraction** at the first S=1024 prefix: sum of selected mask bits across 62 layers / sum of eligible causal pairs across those layers, all 40 heads and 1024 queries. Denominator = 62*40*1024*1025/2 = 1,301,504,000. The native mask records valid keys satisfying `abs(bulk) >= mean(abs(bulk)) + z*sqrt(max(mean(abs(bulk)^2)-mean(abs(bulk))^2,0))`, using the literal native reduction/threshold and causal bounds. `z=norm_ppf(0.9)`; native/diagnostic output equality is checked before capture. This is neither the nominal 0.10 nor a six-window PPL fraction. Per-layer spread remains separately reported.
The blend computes exact rank dots for every key; selected means the exact rank score is used in the blended logits. The fraction does not count avoided rank computations. `apa_sp3_gpu.py` calibration sums selected/pairs from all 62 B margin receipts, and each actual C trial must match within 0.01; delta grid/ties/bisection/freeze rules unchanged. The numerical target is still unknown because capture failed.

## Model table — current provenance-valid perplexity

| Bits | S | Arm | Current status | PPL |
|---|---:|---|---|---:|
| 4 | 1024 | A | STALE | — |
| 4 | 1024 | B | STALE | — |
| 4 | 1024 | C | BLOCKED / unrun | — |
| 4 | 1024 | D | STALE | — |
| 4 | 1024 | E | BLOCKED / unrun | — |
| 4 | 8192 | A | STALE | — |
| 4 | 8192 | B | STALE | — |
| 4 | 8192 | C | BLOCKED / unrun | — |
| 4 | 8192 | D | BLOCKED / unrun | — |
| 4 | 8192 | E | BLOCKED / unrun | — |
| 8 | 1024 | A | BLOCKED / unrun | — |
| 8 | 1024 | B | BLOCKED / unrun | — |
| 8 | 1024 | C | BLOCKED / unrun | — |
| 8 | 1024 | D | BLOCKED / unrun | — |
| 8 | 8192 | A | BLOCKED / unrun | — |
| 8 | 8192 | B | BLOCKED / unrun | — |
| 8 | 8192 | C | BLOCKED / unrun | — |
| 8 | 8192 | D | BLOCKED / unrun | — |

Historical recorded PPL is shown only in the lead-observation table above; STALE values do not fill current result rows.

## G2 margins — kernel sweep on model activations

| Scope | Completed | Required | Result |
|---|---:|---:|---|
| Primary B/C, both lengths, all layers | 0 | 248 | RED / blocked; no values inferred |

No real-activation error, unrefined mass or skipped-relative-weight rows completed. The repaired mask serialization still needs native GPU capture validation before these all-pair measurements can run. No empirical tail claim is supported yet.

C/E model-quality comparisons remain incomplete; kernel and diagnostic plumbing results do not supply PPL evidence. G2/G3 rows establish nothing about model quality by themselves. The E finite-envelope transfer check and its conditional scope remain unchanged.

## Decode — kernel sweep / in-model timing

| Bits | Starting S | Arm | Status | tokens/s |
|---|---:|---|---|---:|
| 4 | 2048 | A | BLOCKED / unrun | — |
| 4 | 2048 | B | BLOCKED / unrun | — |
| 4 | 2048 | C | BLOCKED / unrun | — |
| 4 | 8192 | A | BLOCKED / unrun | — |
| 4 | 8192 | B | BLOCKED / unrun | — |
| 4 | 8192 | C | BLOCKED / unrun | — |
| 4 | 32768 | A | BLOCKED / unrun | — |
| 4 | 32768 | B | BLOCKED / unrun | — |
| 4 | 32768 | C | BLOCKED / unrun | — |
| 8 | 2048 | A | BLOCKED / unrun | — |
| 8 | 2048 | B | BLOCKED / unrun | — |
| 8 | 2048 | C | BLOCKED / unrun | — |
| 8 | 8192 | A | BLOCKED / unrun | — |
| 8 | 8192 | B | BLOCKED / unrun | — |
| 8 | 8192 | C | BLOCKED / unrun | — |
| 8 | 32768 | A | BLOCKED / unrun | — |
| 8 | 32768 | B | BLOCKED / unrun | — |
| 8 | 32768 | C | BLOCKED / unrun | — |

**G2/G3 rows establish nothing about model quality by themselves.**

## Registration, commands and process safety

Registration SHA256 `d9b6511702a894f72174795141c72b2097b1bbe6110e810a1d4d8bfc3cd3498c` and PROTOCOL-2 SHA256 `db95e3ecb958ef6c31a105ee77be19c455ae8b210ed8a8082194389294843224` unchanged. New pre-gate amendment: `amendment_004_capture_dtype.json`, SHA256 `c221a5c91dd27459a6a7f37194b2c7a6d9779325c8933eae7919a171247f947d`; bound to immutable `orders/APA_SP3_AMENDMENT_3.md`. Model openbmb/MiniCPM3-4B, INT4 g128, BF16, token stream, full-prefill feeding, six disjoint 1024 windows, all 512 targets per window and both prediction sets unchanged.
Current blocked report: `GPU_BLOCKED_CAPTURE_DTYPE.json`. All 692 cells and worker estimates remain in `dry_run.json`. `lead_commands.txt` gives the dependency-ordered review index and exact conditional commands; no existing PASS reruns scheduled and no A/8192 retry command. Every GPU command is commented while current receipt prerequisites are RED/stale. Lead owns stale-receipt disposition and capture rerun authorization; this seat performed no GPU job or receipt migration.
Candidate capture after lead resolution: `timeout --kill-after=2s 590s bash scripts/apa_sp3_lead_gpu.sh run capture_b4_B_1024`. This command currently refuses the existing immutable receipt. Worker estimate 140-480s including twelve control prefills; unchanged 480s worker bound, 20s lease wait, 30s foreground cooldown, outer 590s. Kernel estimate 2-20s; fresh baselines 60-480s; G0 aggregate 1-10s; layer margins 3-60s at 1024 or 30-480s at 8192. Estimates are planning values, not measurements.
The unchanged generic summary generator overwrites mutable RESULTS.md/lead_commands.txt with its generic template. Retained r3 handoff copies are `RESULTS_CAPTURE_DTYPE.md` and `lead_commands_capture_dtype.txt`; review those after any later generic summary. Reporter/runner/fingerprint code was not edited.
No git, subagents, background shell jobs/waits, services, foreign-process intervention or process signals. Own foreground commands bounded below ten minutes and completed. No GPU jobs, model load or successful GPU allocation. Read-only adapters/cache and all existing kernel bodies preserved. Execution model **gpt-6-astra**, reasoning effort **xhigh**, launch evidence `logs/apa_sp3_r3.log`.

## Prior art

Unchanged selection/measurement attributions: A ordinary SDPA (Vaswani et al. 2017); B local APA draft (David Perry 2026) and existing TurboQuant reconstruction (Zandieh et al. 2025); C/D existing BLASST-style running-max criterion (Yuan et al. 2025/2026), online normalization (Milakov/Gimelshein 2018) and FlashAttention-2 partitioning lineage (Dao 2023); E local SP2 conditional margin; G2 ThriftAttention weight-sensitive motivation (Sharratt 2026) and standard empirical statistics. Local GraftRepository MiniCPM3 floor protocol (2026) remains the scoring reference. These are retained prior-ledger attributions; no new literature verification or selection/proof/optimization claim in r3.
New code is routine lossless 0/1 dtype adaptation: **no prior art known to me for this specific capture repair**. Regression stubs and independent packed-byte/count oracles are standard testing practice, with no new verification technique claimed. The same annotation appears at the code site and in the ledger. Existing mutation attribution DeMillo/Lipton/Sayward 1978 remains an unverified lead to check, search terms: Hints on Test Data Selection.

Residual RED: native GPU fix validation, literal same-call-site/all-tensor identity, stale current dependencies, retained A/8192 OOM, C/E model quality, all G2/G3 measurements and independent blind verification. No residual is converted to PASS by CPU evidence.
