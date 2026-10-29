# APA-AB1-T1 report: six attention arms on Qwen3-1.7B and Qwen3-4B in one harness

**Verdict.** The harness ran all six arms (plus the fp32-floor instrument) on both models with one flag, one cache
per arm, identical chunking and one peak instrument. Gates as registered: **G0 PASS, G1 RED, G2 RED, G3 PASS (6/7),
G4-G7 met as measurement gates.** Lead predictions: **LA2-LA8 HIT; LA1 and LA9 MISS.** Seat predictions: S1, S4-S10
HIT; S2, S3, S11 MISS.

Registration SHA-256: `08e178e18d5aa3a9a2011c9a2804ea9e7ebaa8bb4e85dc5bffaa59dc3f24dd40` (`artifacts/apa_ab1/registration.json`,
written 2026-10-19T17:11:26Z after G0-G3, before G4). Two crash amendments were used:
`amendment_001.json` (`d9936d6b…f35f`, session memory release) and `amendment_002.json` (`8ba4d6ca…cd9d`, quality
instruments' memory). Pre-gate rules: `gate_rules.json` (`5b82e55d…6d72`).

Scope of every number: RTX 4070 SUPER 12 GB (11.6 GiB usable), native driver 595.99.02, CUDA 12.6 engine build, torch
2.11, transformers 5.12, batch 1, bf16 weights, eager HF loop (no CUDA graphs), wikitext-2 test. Bulk precision is on
every row of every table below. x-FLASH = FLASH's time / the arm's time (below 1 = slower than FLASH).
LA1 is a MISS on its first sentence only: the FLASH arm is bitwise equal to native HF flash (G1.1 PASS on both models).

Headline results (receipts in the tables below):
- **Quality, decode mode (the product path).** SP sits at the fp32 floor on both models: KL(FLASH‖SP) 1.04e-3 against
  the floor KL(F32‖FLASH) 1.03e-3 (1.7B) and 7.97e-4 against 7.84e-4 (4B). SP-tk8 is at the floor too (1.09e-3,
  7.92e-4; top-1 within 0.24 pp of SP's). 2P-TQ16 costs +3.6% PPL on 1.7B and +0.9% on 4B (KL ~110x and ~13x the floor).
  2P-INT4 breaks 1.7B (+571% PPL) and costs +9.9% on 4B.
- **Memory.** Ceilings (largest ladder rung that fits prefill + 32 decode steps): 1.7B: every arm 40K except STD (16K);
  4B: FLASH, SP and 2P-INT4 16K, SP-tk8 and 2P-TQ16 12K, STD 4K. No arm fits 32K on 4B (weights 7.5 GiB plus any
  32K cache exceed the card). Stored bytes per token (layout, G2-verified to the byte): FLASH/STD 114,688 / 147,456;
  SP and 2P-INT4 129,376 / 166,496; SP-tk8 157,888 / 203,456; 2P-TQ16 169,984 / 219,136 (1.7B / 4B).
- **Decode speed.** The eager loop is host-bound: FLASH 24.7-25.1 ms/token on 1.7B and 31.2-32.1 on 4B at every
  context. At 8K, SP is 1.087x (1.7B) and 1.077x (4B) FLASH's ms/token. At 32K on 1.7B SP is 1.99x FLASH: 31.7 ms of
  SP's 39.7 ms summed attention time per step is the contiguous K/V prefix copy the engine ABI forces on a growing
  cache; the SP kernel itself (7.56 ms summed over 27 layers) is faster than flash's (9.23 ms over 28), and SP-tk8's
  kernel is faster still (5.76 ms).
- **Prefill.** SP (K1 Triton + per-chunk F-TQ4 reconstruction) takes 1.78x FLASH's ms at 8K and 2.12x at 32K on 1.7B;
  the two-pass arms take 37-40x at 32K. STD's materialized scores (fp32 inside torch's math path) set its low ceiling.
- **Findings beyond the grid.** (1) Engine ABI: K/V must be contiguous `[B,KVH,S,D]` with S = visible length, so every
  decode step copies the prefix; a K/V row stride (as the packed codes already have) would remove it. (2) G1.5 caught a
  harness defect (int8-V bytes carried as bf16 were NaN-canonicalized by the engine's bf16 slice); fixed before
  registration. (3) torch's caching allocator rounds 1-10 MiB tensors into 20 MiB segments, which breaks the NVML slope
  for per-layer torch caches at 4K (G2 RED for FLASH/STD).

## G0-G7

- **G0 PASS.** Build `artifacts/apa_ab1/build/main` (module `daa0b10c…933f`, EB4 main defines), SASS identical to EB4's
  main build after normalizing the source-path-derived anonymous-namespace hash (`g0/build.json`). Smoke: FLASH, STD,
  F32, 2P-TQ16, 2P-INT4, SP and SP-tk8 each ran prefill 1,983 + 64 decode steps on 2K w0 of Qwen3-1.7B, finite
  (`g0/Qwen3-1.7B_smoke.json`; the first G0 receipt, run before the int8-V fix, is in `g0/superseded_harness_defect/`).
- **G1 RED** (registered rules; receipts `g1/<model>/`):
  - G1.1 PASS both models: the FLASH arm's hidden states and logits are bitwise equal to native HF sdpa under the flash
    pin.
  - G1.2: 1.7B RED, 4B PASS. 1.7B decode mode, SP at delta = FLT_MAX: KL 0.997x the floor, but top-1 0.78 pp below
    F32's (2 of 256 tokens) against a 0.5 pp slack. The other five 1.7B sub-checks and all six 4B sub-checks pass
    (KL/floor 0.958-1.093).
  - G1.3a RED both models: chunked and unchunked FLASH logits are not bitwise (mean KL 1.10e-3 / 9.0e-4). Attention
    alone is bitwise chunked vs unchunked on every captured layer; the registered attribution test (layer-1 o_proj on
    random rows) happened to be shape-invariant, so the rule's attribution clause failed. Diagnosis (`g1_3a_diagnostic.json`):
    the first differing module is layer-0 k_proj/v_proj on 1.7B and q/k/v_proj on 4B, whose GEMMs differ between
    2048-row and 8192-row inputs (cuBLAS shape-dependent kernels).
  - G1.3b PASS both models, every arm: bottom-right alignment with a prefix matches the fp64 reference (relative error
    1.2e-3 to 3.0e-3) and is at least 237x farther from the off-by-one reference.
  - G1.4 PASS both models: SP-tk8 with force_bf16 equals SP's decode logits bitwise at 256/256 steps.
  - G1.5 PASS on 1.7B for all six arms and on 4B for FLASH, 2P-TQ16, 2P-INT4, SP, SP-tk8: every layer's attention
    output at decode step 255 equals a direct call on that layer's captured K/V bitwise, with every cache content equal.
    4B STD: OOM (math SDPA cannot hold the 8K decode-mode run), not checked.
- **G2 RED** (`g2/result.json`): measured cache allocation == layout arithmetic x capacity to the byte for every arm
  on both models (PASS); instrument check PASS; NVML slope (post-prefill, 4K->16K) PASS for the engine-allocated arms
  (0.968-0.998), RED for FLASH (0.690 / 0.917) and STD (0.679 / not measurable: 4B STD OOMs at 8K and above). Cause,
  measured (`g2/allocator_diagnostic_S*.json`): torch serves the 8 MiB per-layer tensors of a 4K cache from 20 MiB
  segments (560 MiB resident for 448 MiB nominal) and keeps co-located activation blocks; 32 MiB tensors at 16K are
  exact.
- **G3 PASS** (`g3/result.json`): 6 of 7 mutants killed: alignment off by one (G1.3b, all arms), layer 0 not exact
  (G1.5 layer 0), bf16 K as kq (G1.5), SP delta = inf (G1.5), sampler on the device total (G2 instrument), appended int8
  V skipped (G1.5). Mutant 1 (HF cache on) NOT ESTABLISHED: its registered check (FLASH slope) already fails without the
  mutant. Supplementary, descriptive: on SP (baseline slope 0.968) the same mutant moves the slope to 1.707.
- **G4 met** (measurement gate): pooled PPL, KL(FLASH‖arm), top-1 and paired dNLL with paired-bootstrap 95% CIs over
  windows at 2K, 8K, 32K (prefill mode) and decode mode, both models; bulk precision on every row. Holes: 4B 32K (every
  arm REF_OOM: the FLASH reference itself does not fit), STD where it OOMs (1.7B 32K; 4B 8K, 32K and decode mode).
- **G5 met**: decode ms/token, host ms, summed attention-hook ms, per-phase ms and prefill ms at 2K, 8K and 32K per arm
  with x-FLASH; variant-B cross-check at 8K on both models.
- **G6 met**: peak VRAM per context point (speed runs; ceiling rungs), stored bytes per token (layout + NVML slope),
  ceilings per arm.
- **G7 met**: one summary table per model below, plus the remaining holes.

## Predictions (registered; full text; judged by the registered rules)

| ID | Result | Prediction (full registered text) | Evidence |
|---|---|---|---|
| LA1 | MISS | G1 holds for every arm. The FLASH arm is bitwise equal to native HF flash. | "G1 per model {\"Qwen3-1.7B\": {\"g1_1\": true, \"g1_2\": false, \"g1_3a\": false, \"g1_3b\": true, \"g1_4\": true}, \"Qwen3-4B\": {\"g1_1\": true, \"g1_2\": true, \"g1_3a\": false, \"g1_3b\": true, \"g1_4\": true}}; G1.5 {\"Qwen3-1.7B\": {\"FLASH\": true, \"STD\": true, \"2P-TQ16\": true, \"2P-INT4\": true, \"SP\": true, \"SP-tk8\": true}, \"Qwen3-4B\": {\"FLASH\": true, \"STD\": false, \"2P-TQ16\": true, \"2P-INT4\": true, \"SP\": true, \"SP-tk8\": true}}; G1.1 bitwise both models: True" |
| LA2 | HIT | SP decode-mode KL is within 2× the FLASH-vs-fp32 floor on both models; SPQ8 measured b at 1.00e-3 (1.7B) and 0.76e-3 (4B). | {"Qwen3-1.7B": {"SP_kl": 0.001042833323538379, "floor_kl": 0.0010289259916999066, "ratio": 1.01351635778536}, "Qwen3-4B": {"SP_kl": 0.0007972869013342879, "floor_kl": 0.0007840940153587911, "ratio": 1.0168256429931555}} |
| LA3 | HIT | 2P-INT4 decode-mode PPL is ≥ 5% worse than FLASH on Qwen3-1.7B, as the INT4 bulk fails there. | {"dppl_pct": 570.6527466464428} |
| LA4 | HIT | SP stores more bytes per token than FLASH (580 against 512 B per key per KV head), so its ceiling is ≤ FLASH's on both models. | {"Qwen3-1.7B": {"bytes_sp": 129376, "bytes_flash": 114688, "ceiling_sp": 40960, "ceiling_flash": 40960}, "Qwen3-4B": {"bytes_sp": 166496, "bytes_flash": 147456, "ceiling_sp": 16384, "ceiling_flash": 16384}} |
| LA5 | HIT | STD has the lowest ceiling of the six, set by the materialized score matrix in chunked prefill. | {"Qwen3-1.7B": {"ceilings": {"FLASH": 40960, "STD": 16384, "2P-TQ16": 40960, "2P-INT4": 40960, "SP": 40960, "SP-tk8": 40960}, "strict_minimum": true, "tied_minimum": false}, "Qwen3-4B": {"ceilings": {"FLASH": 16384, "STD": 4096, "2P-TQ16": 12288, "2P-INT4": 16384, "SP": 16384, "SP-tk8": 12288}, "strict_minimum": true, "tied_minimum": false}} |
| LA6 | HIT | at 8K decode, SP end-to-end ms/token is within 1.10× FLASH. Attention is a minority of a whole-model decode step at batch 1. | {"Qwen3-1.7B": {"SP_ms": 27.237248420715332, "FLASH_ms": 25.059871673583984, "ratio": 1.0868869871120113}, "Qwen3-4B": {"SP_ms": 34.5894889831543, "FLASH_ms": 32.12187194824219, "ratio": 1.076820461736731}} |
| LA7 | HIT | SP-tk8 decode-mode KL against FLASH is ≤ 1.25× SP's on both models, with top-1 within 0.3 pp of SP's. tk8 met SP's attention error at the mean on its gated layers; the nearest-layer f on the rest adds some risk. | {"Qwen3-1.7B": {"kl_ratio": 1.0412357028236405, "top1_tk8": 0.984375, "top1_sp": 0.98203125, "top1_diff_pp": 0.23437499999999778}, "Qwen3-4B": {"kl_ratio": 0.9935093828724231, "top1_tk8": 0.98828125, "top1_sp": 0.98984375, "top1_diff_pp": -0.15625000000000222}} |
| LA8 | HIT | SP-tk8's ceiling is ≤ SP's on both models (712 against 580 B per key per KV head). | {"Qwen3-1.7B": {"ceiling_tk8": 40960, "ceiling_sp": 40960}, "Qwen3-4B": {"ceiling_tk8": 12288, "ceiling_sp": 16384}} |
| LA9 | MISS | at 32K decode, SP-tk8 has the lowest summed attention GPU time per step of all arms on both models. Its end-to-end ms/token gain over FLASH is smaller than its attention-time gain, because the eager loop is host-bound. | {"Qwen3-1.7B": {"attn_ms": {"FLASH": 10.992032021284103, "2P-TQ16": 153.2080804258585, "2P-INT4": 182.61924847960472, "SP": 39.646559834480286, "SP-tk8": 41.91318419575691}, "lowest": "FLASH", "step_gain": 0.46912581070253956, "attn_gain": 0.2622571449104285}, "Qwen3-4B": {"status": "NOT TESTED (32K run did not complete for SP-tk8 or FLASH)", "completed": []}} |
| S1 | HIT | SP decode-mode pooled KL(FLASH||SP) over the 10 decode windows is <= 1.5x the decode-mode floor KL(F32||FLASH) over the same windows, on both models. | {"ratio": {"Qwen3-1.7B": 1.01351635778536, "Qwen3-4B": 1.0168256429931555}} |
| S2 | MISS | 2P-TQ16 decode-mode pooled PPL is >= 2% worse than FLASH (paired, same windows) on both models. | {"dppl_pct": {"Qwen3-1.7B": 3.5895425597215036, "Qwen3-4B": 0.9393739915914523}} |
| S3 | MISS | 2P-INT4 decode-mode pooled PPL is >= 20% worse than FLASH on both models. | {"dppl_pct": {"Qwen3-1.7B": 570.6527466464428, "Qwen3-4B": 9.894664082536098}} |
| S4 | HIT | On Qwen3-4B no arm completes a 32K window: every 4B 32K quality, speed and ceiling row is OOM, and every 4B ceiling is <= 24K. | {"quality32k": {"FLASH": {"0": "REF_OOM"}, "STD": {"0": "REF_OOM"}, "2P-TQ16": {"0": "REF_OOM"}, "2P-INT4": {"0": "REF_OOM"}, "SP": {"0": "REF_OOM"}}, "speed32k": {"FLASH": "OOM", "STD": "OOM", "2P-TQ16": "OOM", "2P-INT4": "OOM", "SP": "OOM", "SP-tk8": "OOM"}, "ceilings": {"FLASH": 16384, "STD": 4096, "2P-TQ16": 12288, "2P-INT4": 16384, "SP": 16384, "SP-tk8": 12288}} |
| S5 | HIT | STD has the lowest ceiling on Qwen3-4B (4K), and on Qwen3-1.7B its ceiling is <= 16K. | {"std_4b": 4096, "std_17b": 16384} |
| S6 | HIT | At 8K decode on Qwen3-1.7B, SP median ms/token <= 1.10x FLASH, and SP-tk8 median ms/token >= 1.15x FLASH (harness quantize-on-append and int8-V materialization overhead). | {"sp_over_flash": 1.0868869871120113, "tk8_over_flash": 1.1832054276752682} |
| S7 | HIT | At 32K decode on Qwen3-1.7B, the engine-ABI materialization phase (contiguous prefix copies + TQ4 pack + q staging) is >= 50% of SP's summed attention-hook time. | {"phases": {"cache_write": 0.29051200463436544, "materialize_pack_stage": 31.720911979675293, "kernel": 7.5629599541425705}, "attn_ms": 39.646559834480286} |
| S8 | HIT | SP full-prefill ms is >= 1.5x FLASH at 8K and at 32K on Qwen3-1.7B (K1 plus per-chunk F-TQ4 reconstruction). | {"ratio_8k": 1.7764416453950915, "ratio_32k": 2.121149586266517} |
| S9 | HIT | Both two-pass arms take >= 10x FLASH full-prefill ms at 32K on Qwen3-1.7B. | {"ratio": {"2P-TQ16": 39.87601375874996, "2P-INT4": 37.32362431726469}} |
| S10 | HIT | SP-tk8 decode-mode pooled KL(FLASH||SP-tk8) is <= 1.25x SP's on both models (agrees with LA7's KL clause). | {"kl_ratio": {"Qwen3-1.7B": 1.0412357028236405, "Qwen3-4B": 0.9935093828724231}} |
| S11 | MISS | Peak VRAM in the 8K speed run: SP <= 1.06x FLASH's peak on both models. | {"peak_ratio": {"Qwen3-1.7B": 1.2156101768003449, "Qwen3-4B": 1.0300338409475465}} |

## Summary tables (G7)
**Qwen3-1.7B, summary (arm x six metrics; every cell scoped: RTX 4070 SUPER 12 GB, B = 1, bf16 weights).**

| Arm | Bulk precision | Quality: decode-mode KL(FLASH‖arm) / dPPL % / top-1 | Peak VRAM MiB @2K / 8K / 32K (speed runs) | Stored B/token: layout / NVML slope | Ceiling (largest FIT rung) | Decode ms/token @2K / 8K / 32K (x-FLASH) | Prefill ms @2K / 8K / 32K (x-FLASH) |
|---|---|---|---|---|---|---|---|
| FLASH | none (exact bf16 flash; K,V stored bf16 16 b/elem) | 1.03e-03 / +0.09% / 0.9879 (vs F32 floor) | 4432 / 4638 / 7374 | 114688 / 84139 | 40960 | 24.8 (1.00) / 25.1 (1.00) / 24.7 (1.00) | 122 (1.00) / 547 (1.00) / 3563 (1.00) |
| STD | none (exact bf16 materialized math SDPA; K,V stored bf16 16 b/elem) | 1.02e-03 / -0.09% / 0.9879 | 4688 / 9822 / OOM | 114688 / 84139 | 16384 | 28.6 (0.87) / 47.5 (0.53) / OOM | 339 (0.36) / 2656 (0.21) / OOM |
| 2P-TQ16 | TQ4 reconstructed to bf16 kq (stored 16 b/elem, F-TQ4), bulk computed fp32 FMA on bf16; refine percentile 0.15 (zthr 1.03643) | 1.13e-01 / +3.59% / 0.8703 | 4954 / 5958 / 10006 | 169984 / 165888 | 40960 | 59.3 (0.42) / 60.5 (0.41) / 164.3 (0.15) | 684 (0.18) / 9078 (0.06) / 142097 (0.03) |
| 2P-INT4 | INT4 sym-7 + fp32 scale per key (4.25 b/elem stored, 68 B/key/KV head), dequantized fp32 in registers, fp32 FMA; refine percentile 0.15 | 2.56e+00 / +570.65% / 0.3973 | 4840 / 5636 / 8696 | 129376 / 125611 | 40960 | 31.0 (0.80) / 63.2 (0.40) / 194.1 (0.13) | 607 (0.20) / 8442 (0.06) / 133002 (0.03) |
| SP | TQ4, 4.25 b/elem stored (codes + fp32 norm); decode computed bf16 hi/lo with fp32 accumulate (EB3); prefill on the bf16 F-TQ4 reconstruction, bf16 tensor-core fp32 accumulate (K1); delta 8.1 row carry | 1.04e-03 / -0.14% / 0.9820 | 4840 / 5638 / 8696 | 129376 / 125611 | 40960 | 26.7 (0.93) / 27.2 (0.92) / 49.1 (0.50) | 201 (0.61) / 972 (0.56) / 7559 (0.47) |
| SP-tk8 | TQ4 as SP; V per 64-key block bf16 or int8 (per-token absmax, fp32 scale) by tk8 at f_L (EB4 T-fresh); prefill = SP | 1.09e-03 / -0.23% / 0.9844 | 4952 / 5920 / 9704 | 157888 / 154624 | 40960 | 32.1 (0.77) / 29.7 (0.85) / 52.6 (0.47) | 202 (0.60) / 954 (0.57) / 7578 (0.47) |

**Qwen3-4B, summary (arm x six metrics; every cell scoped: RTX 4070 SUPER 12 GB, B = 1, bf16 weights).**

| Arm | Bulk precision | Quality: decode-mode KL(FLASH‖arm) / dPPL % / top-1 | Peak VRAM MiB @2K / 8K / 32K (speed runs) | Stored B/token: layout / NVML slope | Ceiling (largest FIT rung) | Decode ms/token @2K / 8K / 32K (x-FLASH) | Prefill ms @2K / 8K / 32K (x-FLASH) |
|---|---|---|---|---|---|---|---|
| FLASH | none (exact bf16 flash; K,V stored bf16 16 b/elem) | 7.84e-04 / +0.04% / 0.9871 (vs F32 floor) | 8520 / 9456 / OOM | 147456 / 147456 | 16384 | 31.2 (1.00) / 32.1 (1.00) / OOM | 303 (1.00) / 1368 (1.00) / OOM |
| STD | none (exact bf16 materialized math SDPA; K,V stored bf16 16 b/elem) | OOM | 9718 / OOM / OOM | 147456 / 147456 | 4096 | 39.2 (0.79) / OOM / OOM | 853 (0.36) / OOM / OOM |
| 2P-TQ16 | TQ4 reconstructed to bf16 kq (stored 16 b/elem, F-TQ4), bulk computed fp32 FMA on bf16; refine percentile 0.15 (zthr 1.03643) | 9.85e-03 / +0.94% / 0.9602 | 8836 / 10186 / OOM | 219136 / 219136 | 12288 | 73.0 (0.43) / 77.7 (0.41) / OOM | 1681 (0.18) / 23360 (0.06) / OOM |
| 2P-INT4 | INT4 sym-7 + fp32 scale per key (4.25 b/elem stored, 68 B/key/KV head), dequantized fp32 in registers, fp32 FMA; refine percentile 0.15 | 1.66e-01 / +9.89% / 0.8562 | 8690 / 9738 / OOM | 166496 / 166912 | 16384 | 45.3 (0.69) / 88.1 (0.36) / OOM | 1544 (0.20) / 22048 (0.06) / OOM |
| SP | TQ4, 4.25 b/elem stored (codes + fp32 norm); decode computed bf16 hi/lo with fp32 accumulate (EB3); prefill on the bf16 F-TQ4 reconstruction, bf16 tensor-core fp32 accumulate (K1); delta 8.1 row carry | 7.97e-04 / +0.12% / 0.9898 | 8690 / 9740 / OOM | 166496 / 166912 | 16384 | 34.1 (0.91) / 34.6 (0.93) / OOM | 415 (0.73) / 2099 (0.65) / OOM |
| SP-tk8 | TQ4 as SP; V per 64-key block bf16 or int8 (per-token absmax, fp32 scale) by tk8 at f_L (EB4 T-fresh); prefill = SP | 7.92e-04 / -0.00% / 0.9883 | 8834 / 10104 / OOM | 203456 / 204459 | 12288 | 36.8 (0.85) / 39.2 (0.82) / OOM | 416 (0.73) / 2112 (0.65) / OOM |


## Quality, refine fraction, free run, speed, memory and ceiling tables (G4-G6)

#### Qwen3-1.7B


**Qwen3-1.7B, summary (arm x six metrics; every cell scoped: RTX 4070 SUPER 12 GB, B = 1, bf16 weights).**

| Arm | Bulk precision | Quality: decode-mode KL(FLASH‖arm) / dPPL % / top-1 | Peak VRAM MiB @2K / 8K / 32K (speed runs) | Stored B/token: layout / NVML slope | Ceiling (largest FIT rung) | Decode ms/token @2K / 8K / 32K (x-FLASH) | Prefill ms @2K / 8K / 32K (x-FLASH) |
|---|---|---|---|---|---|---|---|
| FLASH | none (exact bf16 flash; K,V stored bf16 16 b/elem) | 1.03e-03 / +0.09% / 0.9879 (vs F32 floor) | 4432 / 4638 / 7374 | 114688 / 84139 | 40960 | 24.8 (1.00) / 25.1 (1.00) / 24.7 (1.00) | 122 (1.00) / 547 (1.00) / 3563 (1.00) |
| STD | none (exact bf16 materialized math SDPA; K,V stored bf16 16 b/elem) | 1.02e-03 / -0.09% / 0.9879 | 4688 / 9822 / OOM | 114688 / 84139 | 16384 | 28.6 (0.87) / 47.5 (0.53) / OOM | 339 (0.36) / 2656 (0.21) / OOM |
| 2P-TQ16 | TQ4 reconstructed to bf16 kq (stored 16 b/elem, F-TQ4), bulk computed fp32 FMA on bf16; refine percentile 0.15 (zthr 1.03643) | 1.13e-01 / +3.59% / 0.8703 | 4954 / 5958 / 10006 | 169984 / 165888 | 40960 | 59.3 (0.42) / 60.5 (0.41) / 164.3 (0.15) | 684 (0.18) / 9078 (0.06) / 142097 (0.03) |
| 2P-INT4 | INT4 sym-7 + fp32 scale per key (4.25 b/elem stored, 68 B/key/KV head), dequantized fp32 in registers, fp32 FMA; refine percentile 0.15 | 2.56e+00 / +570.65% / 0.3973 | 4840 / 5636 / 8696 | 129376 / 125611 | 40960 | 31.0 (0.80) / 63.2 (0.40) / 194.1 (0.13) | 607 (0.20) / 8442 (0.06) / 133002 (0.03) |
| SP | TQ4, 4.25 b/elem stored (codes + fp32 norm); decode computed bf16 hi/lo with fp32 accumulate (EB3); prefill on the bf16 F-TQ4 reconstruction, bf16 tensor-core fp32 accumulate (K1); delta 8.1 row carry | 1.04e-03 / -0.14% / 0.9820 | 4840 / 5638 / 8696 | 129376 / 125611 | 40960 | 26.7 (0.93) / 27.2 (0.92) / 49.1 (0.50) | 201 (0.61) / 972 (0.56) / 7559 (0.47) |
| SP-tk8 | TQ4 as SP; V per 64-key block bf16 or int8 (per-token absmax, fp32 scale) by tk8 at f_L (EB4 T-fresh); prefill = SP | 1.09e-03 / -0.23% / 0.9844 | 4952 / 5920 / 9704 | 157888 / 154624 | 40960 | 32.1 (0.77) / 29.7 (0.85) / 52.6 (0.47) | 202 (0.60) / 954 (0.57) / 7578 (0.47) |

**Qwen3-1.7B, prefill mode, 2K (10 windows).** Reference: FLASH; the FLASH row is against the fp32-attention floor (F32). 95% CIs: paired percentile bootstrap over windows (10,000 draws).

| Arm | Bulk precision | Windows / tokens | PPL [CI] | dPPL % vs ref [CI] | KL(ref‖arm) [CI] | Top-1 vs ref [CI] | dNLL [CI] |
|---|---|---|---|---|---|---|---|
| FLASH (vs F32 floor) | none (exact bf16 flash; K,V stored bf16 16 b/elem) | 10 / 20470 | 15.5331 [13.0689, 18.5726] | -0.013 [-0.145, +0.114] | 1.077e-03 [1.02e-03, 1.12e-03] | 0.9865 [0.9842, 0.9884] | -1.27e-04 [-1.5e-03, +1.1e-03] |
| STD | none (exact bf16 materialized math SDPA; K,V stored bf16 16 b/elem) | 10 / 20470 | 15.5351 [13.0800, 18.5714] | +0.013 [-0.114, +0.145] | 1.097e-03 [1.03e-03, 1.17e-03] | 0.9865 [0.9842, 0.9884] | +1.27e-04 [-1.1e-03, +1.5e-03] |
| 2P-TQ16 | TQ4 reconstructed to bf16 kq (stored 16 b/elem, F-TQ4), bulk computed fp32 FMA on bf16; refine percentile 0.15 (zthr 1.03643) | 10 / 20470 | 16.9760 [14.2312, 20.2967] | +9.289 [+7.414, +11.418] | 1.042e-01 [9.78e-02, 1.12e-01] | 0.8762 [0.8650, 0.8866] | +8.88e-02 [+7.2e-02, +1.1e-01] |
| 2P-INT4 | INT4 sym-7 + fp32 scale per key (4.25 b/elem stored, 68 B/key/KV head), dequantized fp32 in registers, fp32 FMA; refine percentile 0.15 | 10 / 20470 | 79.9817 [68.7273, 93.4090] | +414.911 [+379.660, +455.790] | 2.152e+00 [2.07e+00, 2.23e+00] | 0.4401 [0.4224, 0.4597] | +1.64e+00 [+1.6e+00, +1.7e+00] |
| SP | TQ4, 4.25 b/elem stored (codes + fp32 norm); decode computed bf16 hi/lo with fp32 accumulate (EB3); prefill on the bf16 F-TQ4 reconstruction, bf16 tensor-core fp32 accumulate (K1); delta 8.1 row carry | 10 / 20470 | 15.5238 [13.0713, 18.5547] | -0.060 [-0.213, +0.089] | 1.207e-03 [1.12e-03, 1.30e-03] | 0.9860 [0.9845, 0.9875] | -5.98e-04 [-2.1e-03, +8.9e-04] |
| SP-tk8 | TQ4 as SP; V per 64-key block bf16 or int8 (per-token absmax, fp32 scale) by tk8 at f_L (EB4 T-fresh); prefill = SP | = SP row by construction (tk8 is a decode rule; prefill is SP's path); not rerun |  |  |  |  |  |

**Qwen3-1.7B, prefill mode, 8K (10 windows).** Reference: FLASH; the FLASH row is against the fp32-attention floor (F32). 95% CIs: paired percentile bootstrap over windows (10,000 draws).

| Arm | Bulk precision | Windows / tokens | PPL [CI] | dPPL % vs ref [CI] | KL(ref‖arm) [CI] | Top-1 vs ref [CI] | dNLL [CI] |
|---|---|---|---|---|---|---|---|
| FLASH (vs F32 floor) | none (exact bf16 flash; K,V stored bf16 16 b/elem) | 10 / 81910 | 14.4505 [13.0258, 15.8862] | -0.014 [-0.079, +0.043] | 1.032e-03 [1.00e-03, 1.06e-03] | 0.9862 [0.9851, 0.9872] | -1.44e-04 [-7.9e-04, +4.3e-04] |
| STD | none (exact bf16 materialized math SDPA; K,V stored bf16 16 b/elem) | 10 / 81910 | 14.4526 [13.0282, 15.8821] | +0.014 [-0.043, +0.079] | 1.038e-03 [1.01e-03, 1.07e-03] | 0.9862 [0.9851, 0.9872] | +1.44e-04 [-4.3e-04, +7.9e-04] |
| 2P-TQ16 | TQ4 reconstructed to bf16 kq (stored 16 b/elem, F-TQ4), bulk computed fp32 FMA on bf16; refine percentile 0.15 (zthr 1.03643) | 10 / 81910 | 15.6993 [14.0779, 17.2934] | +8.641 [+7.405, +10.154] | 1.111e-01 [1.05e-01, 1.19e-01] | 0.8731 [0.8675, 0.8790] | +8.29e-02 [+7.1e-02, +9.7e-02] |
| 2P-INT4 | INT4 sym-7 + fp32 scale per key (4.25 b/elem stored, 68 B/key/KV head), dequantized fp32 in registers, fp32 FMA; refine percentile 0.15 | 10 / 81910 | 81.2599 [68.5232, 95.1412] | +462.331 [+422.614, +503.650] | 2.285e+00 [2.18e+00, 2.37e+00] | 0.4183 [0.4001, 0.4412] | +1.73e+00 [+1.7e+00, +1.8e+00] |
| SP | TQ4, 4.25 b/elem stored (codes + fp32 norm); decode computed bf16 hi/lo with fp32 accumulate (EB3); prefill on the bf16 F-TQ4 reconstruction, bf16 tensor-core fp32 accumulate (K1); delta 8.1 row carry | 10 / 81910 | 14.4415 [13.0250, 15.8700] | -0.063 [-0.126, +0.009] | 1.154e-03 [1.12e-03, 1.20e-03] | 0.9860 [0.9850, 0.9871] | -6.29e-04 [-1.3e-03, +8.9e-05] |
| SP-tk8 | TQ4 as SP; V per 64-key block bf16 or int8 (per-token absmax, fp32 scale) by tk8 at f_L (EB4 T-fresh); prefill = SP | = SP row by construction (tk8 is a decode rule; prefill is SP's path); not rerun |  |  |  |  |  |

**Qwen3-1.7B, prefill mode, 32K (9 windows).** Reference: FLASH; the FLASH row is against the fp32-attention floor (F32). 95% CIs: paired percentile bootstrap over windows (10,000 draws).

| Arm | Bulk precision | Windows / tokens | PPL [CI] | dPPL % vs ref [CI] | KL(ref‖arm) [CI] | Top-1 vs ref [CI] | dNLL [CI] |
|---|---|---|---|---|---|---|---|
| FLASH (vs F32 floor) | none (exact bf16 flash; K,V stored bf16 16 b/elem) | 9 / 294903 | 13.8764 [13.1717, 14.6008] | -0.000 [-0.027, +0.023] | 1.087e-03 [1.06e-03, 1.11e-03] | 0.9860 [0.9854, 0.9866] | -4.29e-06 [-2.7e-04, +2.3e-04] |
| STD | none (exact bf16 materialized math SDPA; K,V stored bf16 16 b/elem) | 9 receipts: OOM |  |  |  |  |  |
| 2P-TQ16 | TQ4 reconstructed to bf16 kq (stored 16 b/elem, F-TQ4), bulk computed fp32 FMA on bf16; refine percentile 0.15 (zthr 1.03643) | 9 / 294903 | 14.7877 [14.0313, 15.5536] | +6.567 [+6.095, +7.201] | 1.123e-01 [1.08e-01, 1.17e-01] | 0.8703 [0.8667, 0.8737] | +6.36e-02 [+5.9e-02, +7.0e-02] |
| 2P-INT4 | INT4 sym-7 + fp32 scale per key (4.25 b/elem stored, 68 B/key/KV head), dequantized fp32 in registers, fp32 FMA; refine percentile 0.15 | 9 / 294903 | 98.6086 [90.3560, 106.6557] | +610.621 [+579.707, +641.286] | 2.481e+00 [2.42e+00, 2.53e+00] | 0.3984 [0.3887, 0.4076] | +1.96e+00 [+1.9e+00, +2.0e+00] |
| SP | TQ4, 4.25 b/elem stored (codes + fp32 norm); decode computed bf16 hi/lo with fp32 accumulate (EB3); prefill on the bf16 F-TQ4 reconstruction, bf16 tensor-core fp32 accumulate (K1); delta 8.1 row carry | 9 / 294903 | 13.8828 [13.1786, 14.6075] | +0.046 [+0.012, +0.074] | 1.194e-03 [1.16e-03, 1.22e-03] | 0.9855 [0.9849, 0.9860] | +4.59e-04 [+1.2e-04, +7.4e-04] |
| SP-tk8 | TQ4 as SP; V per 64-key block bf16 or int8 (per-token absmax, fp32 scale) by tk8 at f_L (EB4 T-fresh); prefill = SP | = SP row by construction (tk8 is a decode rule; prefill is SP's path); not rerun |  |  |  |  |  |

**Qwen3-1.7B, decode mode (8K windows: prefill 7,936 + 256 teacher-forced steps).** Reference: FLASH; the FLASH row is against the fp32-attention floor (F32). 95% CIs: paired percentile bootstrap over windows (10,000 draws).

| Arm | Bulk precision | Windows / tokens | PPL [CI] | dPPL % vs ref [CI] | KL(ref‖arm) [CI] | Top-1 vs ref [CI] | dNLL [CI] |
|---|---|---|---|---|---|---|---|
| FLASH (vs F32 floor) | none (exact bf16 flash; K,V stored bf16 16 b/elem) | 10 / 2560 | 13.2811 [8.9996, 18.8876] | +0.089 [-0.202, +0.311] | 1.029e-03 [8.95e-04, 1.17e-03] | 0.9879 [0.9840, 0.9918] | +8.86e-04 [-2.0e-03, +3.1e-03] |
| STD | none (exact bf16 materialized math SDPA; K,V stored bf16 16 b/elem) | 10 / 2560 | 13.2693 [8.9835, 18.9032] | -0.089 [-0.310, +0.202] | 1.022e-03 [8.92e-04, 1.15e-03] | 0.9879 [0.9840, 0.9918] | -8.86e-04 [-3.1e-03, +2.0e-03] |
| 2P-TQ16 | TQ4 reconstructed to bf16 kq (stored 16 b/elem, F-TQ4), bulk computed fp32 FMA on bf16; refine percentile 0.15 (zthr 1.03643) | 10 / 2560 | 13.7578 [9.3490, 19.5156] | +3.590 [+1.536, +5.812] | 1.128e-01 [9.35e-02, 1.35e-01] | 0.8703 [0.8582, 0.8852] | +3.53e-02 [+1.5e-02, +5.6e-02] |
| 2P-INT4 | INT4 sym-7 + fp32 scale per key (4.25 b/elem stored, 68 B/key/KV head), dequantized fp32 in registers, fp32 FMA; refine percentile 0.15 | 10 / 2560 | 89.0700 [53.4670, 142.0397] | +570.653 [+408.031, +763.297] | 2.560e+00 [2.27e+00, 2.82e+00] | 0.3973 [0.3543, 0.4441] | +1.90e+00 [+1.6e+00, +2.2e+00] |
| SP | TQ4, 4.25 b/elem stored (codes + fp32 norm); decode computed bf16 hi/lo with fp32 accumulate (EB3); prefill on the bf16 F-TQ4 reconstruction, bf16 tensor-core fp32 accumulate (K1); delta 8.1 row carry | 10 / 2560 | 13.2628 [8.9911, 18.8722] | -0.137 [-0.464, +0.171] | 1.043e-03 [9.17e-04, 1.19e-03] | 0.9820 [0.9781, 0.9855] | -1.38e-03 [-4.6e-03, +1.7e-03] |
| SP-tk8 | TQ4 as SP; V per 64-key block bf16 or int8 (per-token absmax, fp32 scale) by tk8 at f_L (EB4 T-fresh); prefill = SP | 10 / 2560 | 13.2510 [8.9699, 18.8582] | -0.227 [-0.447, +0.049] | 1.086e-03 [9.51e-04, 1.24e-03] | 0.9844 [0.9805, 0.9887] | -2.27e-03 [-4.5e-03, +4.9e-04] |

**Qwen3-1.7B, realized refine fraction per layer (decode mode receipts; prefill rows sampled on the last 64 rows of each chunk, decode every 16th step).**

| Arm | Mode | Layers 1..N-1 mean | Per layer (layer: fraction) |
|---|---|---|---|
| 2P-TQ16 | decode/prefill | 0.1568 | 1: 0.157, 2: 0.162, 3: 0.157, 4: 0.155, 5: 0.158, 6: 0.155, 7: 0.163, 8: 0.158, 9: 0.156, 10: 0.155, 11: 0.152, 12: 0.157, 13: 0.156, 14: 0.158, 15: 0.159, 16: 0.156, 17: 0.157, 18: 0.154, 19: 0.155, 20: 0.157, 21: 0.157, 22: 0.155, 23: 0.160, 24: 0.158, 25: 0.156, 26: 0.153, 27: 0.158 |
| 2P-TQ16 | decode/decode | 0.1573 | 1: 0.159, 2: 0.162, 3: 0.155, 4: 0.149, 5: 0.156, 6: 0.155, 7: 0.161, 8: 0.157, 9: 0.156, 10: 0.153, 11: 0.153, 12: 0.158, 13: 0.157, 14: 0.159, 15: 0.161, 16: 0.157, 17: 0.158, 18: 0.156, 19: 0.156, 20: 0.158, 21: 0.158, 22: 0.157, 23: 0.162, 24: 0.160, 25: 0.157, 26: 0.155, 27: 0.159 |
| 2P-INT4 | decode/prefill | 0.1551 | 1: 0.148, 2: 0.130, 3: 0.155, 4: 0.156, 5: 0.164, 6: 0.157, 7: 0.160, 8: 0.159, 9: 0.156, 10: 0.160, 11: 0.151, 12: 0.158, 13: 0.151, 14: 0.157, 15: 0.161, 16: 0.150, 17: 0.157, 18: 0.155, 19: 0.153, 20: 0.157, 21: 0.158, 22: 0.155, 23: 0.160, 24: 0.159, 25: 0.153, 26: 0.152, 27: 0.156 |
| 2P-INT4 | decode/decode | 0.1556 | 1: 0.148, 2: 0.137, 3: 0.165, 4: 0.149, 5: 0.173, 6: 0.163, 7: 0.153, 8: 0.162, 9: 0.154, 10: 0.161, 11: 0.148, 12: 0.158, 13: 0.147, 14: 0.156, 15: 0.159, 16: 0.148, 17: 0.155, 18: 0.155, 19: 0.153, 20: 0.157, 21: 0.159, 22: 0.154, 23: 0.163, 24: 0.160, 25: 0.155, 26: 0.153, 27: 0.157 |
| SP | decode/prefill | 0.1433 | 1: 0.698, 2: 0.610, 3: 0.101, 4: 0.181, 5: 0.056, 6: 0.100, 7: 0.151, 8: 0.126, 9: 0.135, 10: 0.120, 11: 0.124, 12: 0.166, 13: 0.170, 14: 0.120, 15: 0.111, 16: 0.098, 17: 0.136, 18: 0.125, 19: 0.098, 20: 0.051, 21: 0.059, 22: 0.082, 23: 0.035, 24: 0.044, 25: 0.055, 26: 0.064, 27: 0.049 |
| SP | decode/decode | 0.1165 | 1: 0.666, 2: 0.568, 3: 0.077, 4: 0.163, 5: 0.039, 6: 0.077, 7: 0.116, 8: 0.096, 9: 0.100, 10: 0.082, 11: 0.093, 12: 0.124, 13: 0.137, 14: 0.089, 15: 0.078, 16: 0.072, 17: 0.103, 18: 0.092, 19: 0.068, 20: 0.038, 21: 0.045, 22: 0.056, 23: 0.025, 24: 0.032, 25: 0.039, 26: 0.044, 27: 0.029 |
| SP-tk8 | decode/prefill | 0.1433 | 1: 0.698, 2: 0.610, 3: 0.101, 4: 0.181, 5: 0.056, 6: 0.100, 7: 0.151, 8: 0.126, 9: 0.135, 10: 0.120, 11: 0.124, 12: 0.166, 13: 0.170, 14: 0.120, 15: 0.111, 16: 0.098, 17: 0.136, 18: 0.125, 19: 0.098, 20: 0.051, 21: 0.059, 22: 0.082, 23: 0.035, 24: 0.044, 25: 0.055, 26: 0.064, 27: 0.049 |
| SP-tk8 | decode/decode | 0.1166 | 1: 0.666, 2: 0.567, 3: 0.077, 4: 0.163, 5: 0.039, 6: 0.077, 7: 0.116, 8: 0.096, 9: 0.100, 10: 0.082, 11: 0.093, 12: 0.124, 13: 0.137, 14: 0.089, 15: 0.079, 16: 0.072, 17: 0.103, 18: 0.092, 19: 0.068, 20: 0.038, 21: 0.045, 22: 0.056, 23: 0.026, 24: 0.032, 25: 0.039, 26: 0.044, 27: 0.029 |
| SP-tk8 | decode/tk8_bf16_blocks | 0.0972 | 1: 0.024, 2: 0.024, 3: 0.024, 4: 0.024, 5: 0.024, 6: 0.024, 7: 0.024, 8: 0.253, 9: 0.253, 10: 0.253, 11: 0.253, 12: 0.253, 13: 0.253, 14: 0.103, 15: 0.103, 16: 0.103, 17: 0.103, 18: 0.103, 19: 0.103, 20: 0.055, 21: 0.055, 22: 0.055, 23: 0.055, 24: 0.055, 25: 0.016, 26: 0.016, 27: 0.016 |
| 2P-TQ16 | prefill8192/prefill | 0.1566 | 1: 0.157, 2: 0.162, 3: 0.157, 4: 0.151, 5: 0.158, 6: 0.155, 7: 0.162, 8: 0.158, 9: 0.156, 10: 0.155, 11: 0.152, 12: 0.157, 13: 0.156, 14: 0.158, 15: 0.159, 16: 0.156, 17: 0.157, 18: 0.154, 19: 0.155, 20: 0.157, 21: 0.157, 22: 0.155, 23: 0.160, 24: 0.158, 25: 0.156, 26: 0.153, 27: 0.158 |
| 2P-INT4 | prefill8192/prefill | 0.1548 | 1: 0.148, 2: 0.131, 3: 0.156, 4: 0.152, 5: 0.165, 6: 0.159, 7: 0.159, 8: 0.159, 9: 0.156, 10: 0.161, 11: 0.150, 12: 0.158, 13: 0.150, 14: 0.156, 15: 0.161, 16: 0.149, 17: 0.156, 18: 0.154, 19: 0.153, 20: 0.157, 21: 0.158, 22: 0.155, 23: 0.159, 24: 0.158, 25: 0.153, 26: 0.151, 27: 0.156 |
| SP | prefill8192/prefill | 0.1410 | 1: 0.691, 2: 0.603, 3: 0.102, 4: 0.181, 5: 0.057, 6: 0.101, 7: 0.149, 8: 0.124, 9: 0.133, 10: 0.117, 11: 0.119, 12: 0.160, 13: 0.168, 14: 0.118, 15: 0.110, 16: 0.096, 17: 0.133, 18: 0.121, 19: 0.093, 20: 0.050, 21: 0.058, 22: 0.080, 23: 0.034, 24: 0.044, 25: 0.054, 26: 0.063, 27: 0.048 |
| 2P-TQ16 | prefill32768/prefill | 0.1592 | 1: 0.160, 2: 0.165, 3: 0.171, 4: 0.162, 5: 0.156, 6: 0.164, 7: 0.155, 8: 0.164, 9: 0.159, 10: 0.160, 11: 0.155, 12: 0.158, 13: 0.158, 14: 0.162, 15: 0.155, 16: 0.158, 17: 0.159, 18: 0.161, 19: 0.159, 20: 0.160, 21: 0.159, 22: 0.157, 23: 0.159, 24: 0.159, 25: 0.157, 26: 0.156, 27: 0.153 |
| 2P-INT4 | prefill32768/prefill | 0.1584 | 1: 0.162, 2: 0.171, 3: 0.165, 4: 0.163, 5: 0.156, 6: 0.165, 7: 0.155, 8: 0.163, 9: 0.159, 10: 0.159, 11: 0.151, 12: 0.158, 13: 0.152, 14: 0.160, 15: 0.153, 16: 0.153, 17: 0.158, 18: 0.159, 19: 0.157, 20: 0.159, 21: 0.160, 22: 0.156, 23: 0.159, 24: 0.158, 25: 0.153, 26: 0.156, 27: 0.159 |
| SP | prefill32768/prefill | 0.0771 | 1: 0.549, 2: 0.475, 3: 0.043, 4: 0.121, 5: 0.020, 6: 0.042, 7: 0.060, 8: 0.047, 9: 0.051, 10: 0.042, 11: 0.054, 12: 0.066, 13: 0.086, 14: 0.049, 15: 0.041, 16: 0.042, 17: 0.054, 18: 0.047, 19: 0.034, 20: 0.019, 21: 0.023, 22: 0.030, 23: 0.013, 24: 0.015, 25: 0.021, 26: 0.022, 27: 0.015 |

**Qwen3-1.7B, free run (8 prompts x 128 greedy tokens; divergence vs FLASH, 128 = none).**

| Arm | Prompts OK | Median first divergence | Mean token match | Divergences |
|---|---|---|---|---|
| FLASH | 8/8 | - | - | - |
| STD | 8/8 | 128.0 | 0.801 | [128, 115, 128, 128, 45, 16, 128, 128] |
| 2P-TQ16 | 8/8 | 3.5 | 0.130 | [0, 44, 8, 8, 3, 3, 1, 4] |
| 2P-INT4 | 8/8 | 1.0 | 0.027 | [0, 1, 2, 3, 1, 0, 1, 1] |
| SP | 8/8 | 70.0 | 0.719 | [128, 48, 8, 55, 45, 128, 128, 85] |
| SP-tk8 | 8/8 | 47.5 | 0.535 | [84, 115, 8, 55, 40, 9, 106, 34] |

**Qwen3-1.7B, speed (RTX 4070 SUPER 12 GB; eager HF loop, B = 1; medians; x-FLASH = FLASH / arm).**

| S | Arm | Decode ms/token [IQR] | x-FLASH | Host ms | Attention ms/step [IQR] | x-FLASH attn | Phases ms (write / ABI materialize+pack+stage / kernel) | Prefill ms | x-FLASH prefill | Peak MiB (pre-session MiB; increment) | Outcome |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2048 | FLASH | 24.81 [24.71, 24.94] | 1.000 | 24.83 | 3.74 [3.73, 3.77] | 1.000 | 1.37 / 0.00 / 1.74 | 122 | 1.000 | 4432 (4420; 12) | OK |
| 2048 | STD | 28.58 [28.33, 30.26] | 0.868 | 28.62 | 8.92 [8.69, 9.72] | 0.419 | 1.42 / 0.00 / 7.00 | 339 | 0.360 | 4688 (4420; 268) | OK |
| 2048 | 2P-TQ16 | 59.29 [57.58, 62.17] | 0.419 | 59.31 | 46.68 [46.39, 47.76] | 0.080 | 32.59 / 3.31 / 10.37 | 684 | 0.178 | 4954 (4420; 534) | OK |
| 2048 | 2P-INT4 | 30.95 [30.50, 31.89] | 0.802 | 30.99 | 20.39 [19.94, 21.05] | 0.183 | 0.99 / 2.37 / 16.85 | 607 | 0.201 | 4840 (4422; 418) | OK |
| 2048 | SP | 26.66 [26.45, 27.18] | 0.931 | 26.68 | 5.80 [5.69, 6.52] | 0.644 | 1.18 / 2.90 / 0.86 | 201 | 0.607 | 4840 (4422; 418) | OK |
| 2048 | SP-tk8 | 32.10 [30.43, 33.56] | 0.773 | 32.13 | 9.79 [9.35, 10.20] | 0.382 | 4.18 / 3.30 / 1.27 | 202 | 0.604 | 4952 (4422; 530) | OK |
| 8192 | FLASH | 25.06 [24.91, 25.50] | 1.000 | 25.14 | 4.96 [4.93, 5.22] | 1.000 | 1.37 / 0.00 / 3.20 | 547 | 1.000 | 4638 (4122; 516) | OK |
| 8192 | STD | 47.49 [47.24, 47.71] | 0.528 | 47.54 | 36.88 [36.85, 36.95] | 0.134 | 0.20 / 0.00 / 36.60 | 2656 | 0.206 | 9822 (4122; 5700) | OK |
| 8192 | 2P-TQ16 | 60.46 [58.40, 63.73] | 0.414 | 60.50 | 55.34 [50.84, 59.80] | 0.090 | 19.27 / 11.72 / 24.18 | 9078 | 0.060 | 5958 (4420; 1538) | OK |
| 8192 | 2P-INT4 | 63.25 [63.09, 63.45] | 0.396 | 63.29 | 52.00 [51.97, 52.03] | 0.095 | 0.23 / 7.57 / 44.08 | 8442 | 0.065 | 5636 (4420; 1216) | OK |
| 8192 | SP | 27.24 [26.73, 28.60] | 0.920 | 27.27 | 11.16 [11.07, 11.30] | 0.444 | 1.23 / 8.16 / 1.31 | 972 | 0.563 | 5638 (4420; 1218) | OK |
| 8192 | SP-tk8 | 29.65 [29.33, 30.69] | 0.845 | 29.67 | 15.03 [14.89, 15.49] | 0.330 | 3.76 / 9.35 / 1.54 | 954 | 0.574 | 5920 (4420; 1500) | OK |
| 32768 | FLASH | 24.67 [24.44, 24.94] | 1.000 | 24.69 | 10.99 [10.97, 11.02] | 1.000 | 1.36 / 0.00 / 9.23 | 3563 | 1.000 | 7374 (4122; 3252) | OK |
| 32768 | STD |  |  |  |  |  |  |  |  | - | OOM |
| 32768 | 2P-TQ16 | 164.26 [164.16, 164.35] | 0.150 | 164.35 | 153.21 [153.14, 153.30] | 0.072 | 4.61 / 43.89 / 104.58 | 142097 | 0.025 | 10006 (4420; 5586) | OK |
| 32768 | 2P-INT4 | 194.10 [193.72, 194.43] | 0.127 | 194.21 | 182.62 [182.55, 182.67] | 0.060 | 0.25 / 29.26 / 152.98 | 133002 | 0.027 | 8696 (4420; 4276) | OK |
| 32768 | SP | 49.08 [48.80, 49.91] | 0.503 | 49.18 | 39.65 [39.50, 39.80] | 0.277 | 0.29 / 31.72 / 7.56 | 7559 | 0.471 | 8696 (4420; 4276) | OK |
| 32768 | SP-tk8 | 52.58 [52.40, 52.99] | 0.469 | 52.67 | 41.91 [41.81, 42.19] | 0.262 | 0.59 / 35.44 / 5.76 | 7578 | 0.470 | 9704 (4420; 5284) | OK |

Variant-B entry cross-check (SP decode at 8K through `apa_selective_attention_sp_d1_packed`, `gqa_path=tc`): {"8192": {"outcome": "OK", "decode_ms": 26.744832038879395, "attn_ms": 12.245775945484638, "phases": {"cache_write": 1.1862559895962477, "materialize_pack_stage": 8.9666398614645, "kernel": 1.6769280023872852}, "prefill_ms": 987.9495697021484}}

**Qwen3-1.7B, context ceiling (ladder 4K..40K, prefill + 32 decode steps, fresh process per rung) and stored bytes.**

| Arm | Bulk precision | Layout B/token (B/key/KV head, layers>=1) | NVML slope B/token (after_alloc, 4K->hi) | slope/layout | Ceiling (largest FIT) | First OOM (completed tokens) | Peak MiB per rung |
|---|---|---|---|---|---|---|---|
| FLASH | none (exact bf16 flash; K,V stored bf16 16 b/elem) | 114688 (512) | 84139 | 0.734 | 40960 | - | 4096:4432, 8192:4588, 12288:5080, 16384:5546, 24576:6460, 32768:7366, 40960:8168 |
| STD | none (exact bf16 materialized math SDPA; K,V stored bf16 16 b/elem) | 114688 (512) | 84139 | 0.734 | 16384 | 24576 (14336) | 4096:6032, 8192:9780, 12288:11304, 16384:11128, 24576:11466, 32768:NON_FIT_AFTER_OOM, 40960:NON_FIT_AFTER_OOM |
| 2P-TQ16 | TQ4 reconstructed to bf16 kq (stored 16 b/elem, F-TQ4), bulk computed fp32 FMA on bf16; refine percentile 0.15 (zthr 1.03643) | 169984 (768) | 165888 | 0.976 | 40960 | - | 4096:5310, 8192:5958, 12288:6638, 16384:7318, 24576:8646, 32768:10006, 40960:11184 |
| 2P-INT4 | INT4 sym-7 + fp32 scale per key (4.25 b/elem stored, 68 B/key/KV head), dequantized fp32 in registers, fp32 FMA; refine percentile 0.15 | 129376 (580) | 125611 | 0.971 | 40960 | - | 4096:5114, 8192:5636, 12288:6126, 16384:6650, 24576:7664, 32768:8694, 40960:9544 |
| SP | TQ4, 4.25 b/elem stored (codes + fp32 norm); decode computed bf16 hi/lo with fp32 accumulate (EB3); prefill on the bf16 F-TQ4 reconstruction, bf16 tensor-core fp32 accumulate (K1); delta 8.1 row carry | 129376 (580) | 125611 | 0.971 | 40960 | - | 4096:5116, 8192:5638, 12288:6126, 16384:6652, 24576:7664, 32768:8694, 40960:9546 |
| SP-tk8 | TQ4 as SP; V per 64-key block bf16 or int8 (per-token absmax, fp32 scale) by tk8 at f_L (EB4 T-fresh); prefill = SP | 157888 (712) | 154624 | 0.979 | 40960 | - | 4096:5284, 8192:5918, 12288:6520, 16384:7160, 24576:8398, 32768:9702, 40960:10720 |

#### Qwen3-4B


**Qwen3-4B, summary (arm x six metrics; every cell scoped: RTX 4070 SUPER 12 GB, B = 1, bf16 weights).**

| Arm | Bulk precision | Quality: decode-mode KL(FLASH‖arm) / dPPL % / top-1 | Peak VRAM MiB @2K / 8K / 32K (speed runs) | Stored B/token: layout / NVML slope | Ceiling (largest FIT rung) | Decode ms/token @2K / 8K / 32K (x-FLASH) | Prefill ms @2K / 8K / 32K (x-FLASH) |
|---|---|---|---|---|---|---|---|
| FLASH | none (exact bf16 flash; K,V stored bf16 16 b/elem) | 7.84e-04 / +0.04% / 0.9871 (vs F32 floor) | 8520 / 9456 / OOM | 147456 / 147456 | 16384 | 31.2 (1.00) / 32.1 (1.00) / OOM | 303 (1.00) / 1368 (1.00) / OOM |
| STD | none (exact bf16 materialized math SDPA; K,V stored bf16 16 b/elem) | OOM | 9718 / OOM / OOM | 147456 / 147456 | 4096 | 39.2 (0.79) / OOM / OOM | 853 (0.36) / OOM / OOM |
| 2P-TQ16 | TQ4 reconstructed to bf16 kq (stored 16 b/elem, F-TQ4), bulk computed fp32 FMA on bf16; refine percentile 0.15 (zthr 1.03643) | 9.85e-03 / +0.94% / 0.9602 | 8836 / 10186 / OOM | 219136 / 219136 | 12288 | 73.0 (0.43) / 77.7 (0.41) / OOM | 1681 (0.18) / 23360 (0.06) / OOM |
| 2P-INT4 | INT4 sym-7 + fp32 scale per key (4.25 b/elem stored, 68 B/key/KV head), dequantized fp32 in registers, fp32 FMA; refine percentile 0.15 | 1.66e-01 / +9.89% / 0.8562 | 8690 / 9738 / OOM | 166496 / 166912 | 16384 | 45.3 (0.69) / 88.1 (0.36) / OOM | 1544 (0.20) / 22048 (0.06) / OOM |
| SP | TQ4, 4.25 b/elem stored (codes + fp32 norm); decode computed bf16 hi/lo with fp32 accumulate (EB3); prefill on the bf16 F-TQ4 reconstruction, bf16 tensor-core fp32 accumulate (K1); delta 8.1 row carry | 7.97e-04 / +0.12% / 0.9898 | 8690 / 9740 / OOM | 166496 / 166912 | 16384 | 34.1 (0.91) / 34.6 (0.93) / OOM | 415 (0.73) / 2099 (0.65) / OOM |
| SP-tk8 | TQ4 as SP; V per 64-key block bf16 or int8 (per-token absmax, fp32 scale) by tk8 at f_L (EB4 T-fresh); prefill = SP | 7.92e-04 / -0.00% / 0.9883 | 8834 / 10104 / OOM | 203456 / 204459 | 12288 | 36.8 (0.85) / 39.2 (0.82) / OOM | 416 (0.73) / 2112 (0.65) / OOM |

**Qwen3-4B, prefill mode, 2K (10 windows).** Reference: FLASH; the FLASH row is against the fp32-attention floor (F32). 95% CIs: paired percentile bootstrap over windows (10,000 draws).

| Arm | Bulk precision | Windows / tokens | PPL [CI] | dPPL % vs ref [CI] | KL(ref‖arm) [CI] | Top-1 vs ref [CI] | dNLL [CI] |
|---|---|---|---|---|---|---|---|
| FLASH (vs F32 floor) | none (exact bf16 flash; K,V stored bf16 16 b/elem) | 10 / 20470 | 12.6992 [10.3891, 15.4687] | -0.041 [-0.141, +0.064] | 9.353e-04 [8.83e-04, 9.82e-04] | 0.9865 [0.9846, 0.9883] | -4.05e-04 [-1.4e-03, +6.4e-04] |
| STD | none (exact bf16 materialized math SDPA; K,V stored bf16 16 b/elem) | 10 / 20470 | 12.7043 [10.3926, 15.4752] | +0.041 [-0.064, +0.141] | 9.333e-04 [8.80e-04, 9.86e-04] | 0.9865 [0.9846, 0.9883] | +4.05e-04 [-6.4e-04, +1.4e-03] |
| 2P-TQ16 | TQ4 reconstructed to bf16 kq (stored 16 b/elem, F-TQ4), bulk computed fp32 FMA on bf16; refine percentile 0.15 (zthr 1.03643) | 10 / 20470 | 12.7567 [10.4402, 15.5516] | +0.453 [+0.212, +0.700] | 5.946e-03 [5.42e-03, 6.50e-03] | 0.9698 [0.9659, 0.9740] | +4.52e-03 [+2.1e-03, +7.0e-03] |
| 2P-INT4 | INT4 sym-7 + fp32 scale per key (4.25 b/elem stored, 68 B/key/KV head), dequantized fp32 in registers, fp32 FMA; refine percentile 0.15 | 10 / 20470 | 12.7427 [10.4895, 15.4517] | +0.343 [-1.158, +1.521] | 7.308e-02 [6.63e-02, 8.00e-02] | 0.8933 [0.8818, 0.9036] | +3.42e-03 [-1.2e-02, +1.5e-02] |
| SP | TQ4, 4.25 b/elem stored (codes + fp32 norm); decode computed bf16 hi/lo with fp32 accumulate (EB3); prefill on the bf16 F-TQ4 reconstruction, bf16 tensor-core fp32 accumulate (K1); delta 8.1 row carry | 10 / 20470 | 12.6980 [10.3782, 15.4792] | -0.009 [-0.133, +0.109] | 8.977e-04 [8.52e-04, 9.37e-04] | 0.9871 [0.9852, 0.9887] | -9.08e-05 [-1.3e-03, +1.1e-03] |
| SP-tk8 | TQ4 as SP; V per 64-key block bf16 or int8 (per-token absmax, fp32 scale) by tk8 at f_L (EB4 T-fresh); prefill = SP | = SP row by construction (tk8 is a decode rule; prefill is SP's path); not rerun |  |  |  |  |  |

**Qwen3-4B, prefill mode, 8K (10 windows).** Reference: FLASH; the FLASH row is against the fp32-attention floor (F32). 95% CIs: paired percentile bootstrap over windows (10,000 draws).

| Arm | Bulk precision | Windows / tokens | PPL [CI] | dPPL % vs ref [CI] | KL(ref‖arm) [CI] | Top-1 vs ref [CI] | dNLL [CI] |
|---|---|---|---|---|---|---|---|
| FLASH (vs F32 floor) | none (exact bf16 flash; K,V stored bf16 16 b/elem) | 10 / 81910 | 11.1720 [10.0422, 12.4313] | -0.025 [-0.071, +0.024] | 8.863e-04 [8.51e-04, 9.23e-04] | 0.9874 [0.9865, 0.9881] | -2.49e-04 [-7.1e-04, +2.4e-04] |
| STD | none (exact bf16 materialized math SDPA; K,V stored bf16 16 b/elem) | 10 receipts: OOM |  |  |  |  |  |
| 2P-TQ16 | TQ4 reconstructed to bf16 kq (stored 16 b/elem, F-TQ4), bulk computed fp32 FMA on bf16; refine percentile 0.15 (zthr 1.03643) | 10 / 81910 | 11.2035 [10.0711, 12.4662] | +0.282 [+0.171, +0.386] | 8.657e-03 [7.86e-03, 9.39e-03] | 0.9644 [0.9622, 0.9665] | +2.82e-03 [+1.7e-03, +3.8e-03] |
| 2P-INT4 | INT4 sym-7 + fp32 scale per key (4.25 b/elem stored, 68 B/key/KV head), dequantized fp32 in registers, fp32 FMA; refine percentile 0.15 | 10 / 81910 | 11.5714 [10.3727, 12.8700] | +3.576 [+2.502, +4.830] | 1.016e-01 [9.03e-02, 1.12e-01] | 0.8778 [0.8709, 0.8846] | +3.51e-02 [+2.5e-02, +4.7e-02] |
| SP | TQ4, 4.25 b/elem stored (codes + fp32 norm); decode computed bf16 hi/lo with fp32 accumulate (EB3); prefill on the bf16 F-TQ4 reconstruction, bf16 tensor-core fp32 accumulate (K1); delta 8.1 row carry | 10 / 81910 | 11.1688 [10.0376, 12.4285] | -0.028 [-0.063, +0.009] | 8.857e-04 [8.51e-04, 9.17e-04] | 0.9874 [0.9866, 0.9881] | -2.80e-04 [-6.3e-04, +8.5e-05] |
| SP-tk8 | TQ4 as SP; V per 64-key block bf16 or int8 (per-token absmax, fp32 scale) by tk8 at f_L (EB4 T-fresh); prefill = SP | = SP row by construction (tk8 is a decode rule; prefill is SP's path); not rerun |  |  |  |  |  |

**Qwen3-4B, prefill mode, 32K (9 windows).** Reference: FLASH; the FLASH row is against the fp32-attention floor (F32). 95% CIs: paired percentile bootstrap over windows (10,000 draws).

| Arm | Bulk precision | Windows / tokens | PPL [CI] | dPPL % vs ref [CI] | KL(ref‖arm) [CI] | Top-1 vs ref [CI] | dNLL [CI] |
|---|---|---|---|---|---|---|---|
| FLASH | none (exact bf16 flash; K,V stored bf16 16 b/elem) | 1 receipts: REF_OOM |  |  |  |  |  |
| STD | none (exact bf16 materialized math SDPA; K,V stored bf16 16 b/elem) | 1 receipts: REF_OOM |  |  |  |  |  |
| 2P-TQ16 | TQ4 reconstructed to bf16 kq (stored 16 b/elem, F-TQ4), bulk computed fp32 FMA on bf16; refine percentile 0.15 (zthr 1.03643) | 1 receipts: REF_OOM |  |  |  |  |  |
| 2P-INT4 | INT4 sym-7 + fp32 scale per key (4.25 b/elem stored, 68 B/key/KV head), dequantized fp32 in registers, fp32 FMA; refine percentile 0.15 | 1 receipts: REF_OOM |  |  |  |  |  |
| SP | TQ4, 4.25 b/elem stored (codes + fp32 norm); decode computed bf16 hi/lo with fp32 accumulate (EB3); prefill on the bf16 F-TQ4 reconstruction, bf16 tensor-core fp32 accumulate (K1); delta 8.1 row carry | 1 receipts: REF_OOM |  |  |  |  |  |
| SP-tk8 | TQ4 as SP; V per 64-key block bf16 or int8 (per-token absmax, fp32 scale) by tk8 at f_L (EB4 T-fresh); prefill = SP | = SP row by construction (tk8 is a decode rule; prefill is SP's path); not rerun |  |  |  |  |  |

**Qwen3-4B, decode mode (8K windows: prefill 7,936 + 256 teacher-forced steps).** Reference: FLASH; the FLASH row is against the fp32-attention floor (F32). 95% CIs: paired percentile bootstrap over windows (10,000 draws).

| Arm | Bulk precision | Windows / tokens | PPL [CI] | dPPL % vs ref [CI] | KL(ref‖arm) [CI] | Top-1 vs ref [CI] | dNLL [CI] |
|---|---|---|---|---|---|---|---|
| FLASH (vs F32 floor) | none (exact bf16 flash; K,V stored bf16 16 b/elem) | 10 / 2560 | 9.6504 [6.8651, 12.9855] | +0.037 [-0.083, +0.145] | 7.841e-04 [6.76e-04, 9.12e-04] | 0.9871 [0.9832, 0.9906] | +3.71e-04 [-8.3e-04, +1.5e-03] |
| STD | none (exact bf16 materialized math SDPA; K,V stored bf16 16 b/elem) | 10 receipts: OOM |  |  |  |  |  |
| 2P-TQ16 | TQ4 reconstructed to bf16 kq (stored 16 b/elem, F-TQ4), bulk computed fp32 FMA on bf16; refine percentile 0.15 (zthr 1.03643) | 10 / 2560 | 9.7411 [6.9410, 13.0755] | +0.939 [+0.400, +1.537] | 9.850e-03 [8.12e-03, 1.17e-02] | 0.9602 [0.9523, 0.9672] | +9.35e-03 [+4.0e-03, +1.5e-02] |
| 2P-INT4 | INT4 sym-7 + fp32 scale per key (4.25 b/elem stored, 68 B/key/KV head), dequantized fp32 in registers, fp32 FMA; refine percentile 0.15 | 10 / 2560 | 10.6053 [7.4550, 14.2624] | +9.895 [+5.499, +13.982] | 1.664e-01 [1.25e-01, 2.09e-01] | 0.8562 [0.8285, 0.8844] | +9.44e-02 [+5.4e-02, +1.3e-01] |
| SP | TQ4, 4.25 b/elem stored (codes + fp32 norm); decode computed bf16 hi/lo with fp32 accumulate (EB3); prefill on the bf16 F-TQ4 reconstruction, bf16 tensor-core fp32 accumulate (K1); delta 8.1 row carry | 10 / 2560 | 9.6621 [6.8672, 13.0024] | +0.121 [-0.058, +0.313] | 7.973e-04 [7.17e-04, 8.78e-04] | 0.9898 [0.9867, 0.9930] | +1.21e-03 [-5.8e-04, +3.1e-03] |
| SP-tk8 | TQ4 as SP; V per 64-key block bf16 or int8 (per-token absmax, fp32 scale) by tk8 at f_L (EB4 T-fresh); prefill = SP | 10 / 2560 | 9.6502 [6.8623, 13.0000] | -0.003 [-0.178, +0.186] | 7.921e-04 [7.08e-04, 8.71e-04] | 0.9883 [0.9832, 0.9926] | -2.68e-05 [-1.8e-03, +1.9e-03] |

**Qwen3-4B, realized refine fraction per layer (decode mode receipts; prefill rows sampled on the last 64 rows of each chunk, decode every 16th step).**

| Arm | Mode | Layers 1..N-1 mean | Per layer (layer: fraction) |
|---|---|---|---|
| 2P-TQ16 | decode/prefill | 0.1555 | 1: 0.156, 2: 0.148, 3: 0.146, 4: 0.158, 5: 0.154, 6: 0.158, 7: 0.152, 8: 0.162, 9: 0.157, 10: 0.164, 11: 0.158, 12: 0.157, 13: 0.154, 14: 0.158, 15: 0.152, 16: 0.159, 17: 0.155, 18: 0.155, 19: 0.153, 20: 0.156, 21: 0.157, 22: 0.154, 23: 0.155, 24: 0.153, 25: 0.157, 26: 0.154, 27: 0.157, 28: 0.156, 29: 0.154, 30: 0.157, 31: 0.153, 32: 0.155, 33: 0.155, 34: 0.158, 35: 0.155 |
| 2P-TQ16 | decode/decode | 0.1567 | 1: 0.157, 2: 0.149, 3: 0.147, 4: 0.155, 5: 0.155, 6: 0.158, 7: 0.154, 8: 0.162, 9: 0.156, 10: 0.174, 11: 0.163, 12: 0.157, 13: 0.155, 14: 0.158, 15: 0.154, 16: 0.161, 17: 0.156, 18: 0.156, 19: 0.154, 20: 0.157, 21: 0.158, 22: 0.156, 23: 0.157, 24: 0.154, 25: 0.158, 26: 0.155, 27: 0.159, 28: 0.157, 29: 0.155, 30: 0.158, 31: 0.155, 32: 0.157, 33: 0.155, 34: 0.158, 35: 0.156 |
| 2P-INT4 | decode/prefill | 0.1558 | 1: 0.159, 2: 0.149, 3: 0.147, 4: 0.160, 5: 0.160, 6: 0.159, 7: 0.152, 8: 0.154, 9: 0.157, 10: 0.164, 11: 0.155, 12: 0.156, 13: 0.155, 14: 0.159, 15: 0.153, 16: 0.159, 17: 0.155, 18: 0.155, 19: 0.153, 20: 0.156, 21: 0.157, 22: 0.155, 23: 0.155, 24: 0.154, 25: 0.157, 26: 0.154, 27: 0.157, 28: 0.156, 29: 0.154, 30: 0.158, 31: 0.154, 32: 0.155, 33: 0.155, 34: 0.157, 35: 0.159 |
| 2P-INT4 | decode/decode | 0.1565 | 1: 0.158, 2: 0.149, 3: 0.148, 4: 0.155, 5: 0.159, 6: 0.159, 7: 0.154, 8: 0.148, 9: 0.157, 10: 0.171, 11: 0.155, 12: 0.156, 13: 0.155, 14: 0.158, 15: 0.154, 16: 0.161, 17: 0.157, 18: 0.157, 19: 0.154, 20: 0.157, 21: 0.157, 22: 0.156, 23: 0.157, 24: 0.155, 25: 0.158, 26: 0.155, 27: 0.158, 28: 0.157, 29: 0.155, 30: 0.159, 31: 0.155, 32: 0.157, 33: 0.156, 34: 0.158, 35: 0.161 |
| SP | decode/prefill | 0.1590 | 1: 0.800, 2: 0.578, 3: 0.345, 4: 0.292, 5: 0.294, 6: 0.434, 7: 0.055, 8: 0.091, 9: 0.094, 10: 0.064, 11: 0.070, 12: 0.121, 13: 0.095, 14: 0.117, 15: 0.157, 16: 0.163, 17: 0.224, 18: 0.179, 19: 0.167, 20: 0.138, 21: 0.120, 22: 0.179, 23: 0.095, 24: 0.118, 25: 0.042, 26: 0.065, 27: 0.064, 28: 0.058, 29: 0.043, 30: 0.029, 31: 0.035, 32: 0.032, 33: 0.057, 34: 0.084, 35: 0.065 |
| SP | decode/decode | 0.1345 | 1: 0.771, 2: 0.553, 3: 0.312, 4: 0.253, 5: 0.270, 6: 0.381, 7: 0.039, 8: 0.064, 9: 0.071, 10: 0.042, 11: 0.046, 12: 0.095, 13: 0.070, 14: 0.087, 15: 0.130, 16: 0.121, 17: 0.187, 18: 0.135, 19: 0.137, 20: 0.120, 21: 0.097, 22: 0.142, 23: 0.075, 24: 0.093, 25: 0.032, 26: 0.044, 27: 0.047, 28: 0.044, 29: 0.034, 30: 0.020, 31: 0.025, 32: 0.024, 33: 0.045, 34: 0.062, 35: 0.041 |
| SP-tk8 | decode/prefill | 0.1590 | 1: 0.800, 2: 0.578, 3: 0.345, 4: 0.292, 5: 0.294, 6: 0.434, 7: 0.055, 8: 0.091, 9: 0.094, 10: 0.064, 11: 0.070, 12: 0.121, 13: 0.095, 14: 0.117, 15: 0.157, 16: 0.163, 17: 0.224, 18: 0.179, 19: 0.167, 20: 0.138, 21: 0.120, 22: 0.179, 23: 0.095, 24: 0.118, 25: 0.042, 26: 0.065, 27: 0.064, 28: 0.058, 29: 0.043, 30: 0.029, 31: 0.035, 32: 0.032, 33: 0.057, 34: 0.084, 35: 0.065 |
| SP-tk8 | decode/decode | 0.1345 | 1: 0.771, 2: 0.553, 3: 0.312, 4: 0.253, 5: 0.270, 6: 0.381, 7: 0.039, 8: 0.064, 9: 0.071, 10: 0.042, 11: 0.046, 12: 0.095, 13: 0.070, 14: 0.087, 15: 0.130, 16: 0.121, 17: 0.187, 18: 0.135, 19: 0.137, 20: 0.119, 21: 0.097, 22: 0.142, 23: 0.075, 24: 0.093, 25: 0.032, 26: 0.044, 27: 0.047, 28: 0.044, 29: 0.034, 30: 0.020, 31: 0.025, 32: 0.024, 33: 0.045, 34: 0.062, 35: 0.041 |
| SP-tk8 | decode/tk8_bf16_blocks | 0.1299 | 1: 0.055, 2: 0.055, 3: 0.055, 4: 0.055, 5: 0.055, 6: 0.055, 7: 0.055, 8: 0.055, 9: 0.055, 10: 0.055, 11: 0.253, 12: 0.253, 13: 0.253, 14: 0.253, 15: 0.253, 16: 0.253, 17: 0.253, 18: 0.253, 19: 0.253, 20: 0.253, 21: 0.253, 22: 0.253, 23: 0.253, 24: 0.253, 25: 0.055, 26: 0.055, 27: 0.055, 28: 0.055, 29: 0.055, 30: 0.055, 31: 0.055, 32: 0.016, 33: 0.016, 34: 0.016, 35: 0.016 |
| 2P-TQ16 | prefill8192/prefill | 0.1556 | 1: 0.156, 2: 0.148, 3: 0.146, 4: 0.158, 5: 0.154, 6: 0.158, 7: 0.152, 8: 0.162, 9: 0.156, 10: 0.168, 11: 0.160, 12: 0.157, 13: 0.154, 14: 0.158, 15: 0.152, 16: 0.159, 17: 0.155, 18: 0.155, 19: 0.153, 20: 0.156, 21: 0.157, 22: 0.154, 23: 0.155, 24: 0.153, 25: 0.157, 26: 0.154, 27: 0.157, 28: 0.156, 29: 0.154, 30: 0.158, 31: 0.154, 32: 0.155, 33: 0.155, 34: 0.157, 35: 0.155 |
| 2P-INT4 | prefill8192/prefill | 0.1558 | 1: 0.158, 2: 0.149, 3: 0.147, 4: 0.159, 5: 0.158, 6: 0.160, 7: 0.152, 8: 0.152, 9: 0.157, 10: 0.167, 11: 0.156, 12: 0.156, 13: 0.154, 14: 0.159, 15: 0.153, 16: 0.159, 17: 0.155, 18: 0.155, 19: 0.153, 20: 0.156, 21: 0.156, 22: 0.155, 23: 0.155, 24: 0.153, 25: 0.157, 26: 0.154, 27: 0.157, 28: 0.156, 29: 0.154, 30: 0.158, 31: 0.154, 32: 0.155, 33: 0.155, 34: 0.157, 35: 0.159 |
| SP | prefill8192/prefill | 0.1573 | 1: 0.797, 2: 0.576, 3: 0.351, 4: 0.295, 5: 0.294, 6: 0.427, 7: 0.055, 8: 0.090, 9: 0.094, 10: 0.063, 11: 0.069, 12: 0.120, 13: 0.094, 14: 0.115, 15: 0.155, 16: 0.161, 17: 0.221, 18: 0.172, 19: 0.164, 20: 0.136, 21: 0.118, 22: 0.175, 23: 0.092, 24: 0.114, 25: 0.042, 26: 0.063, 27: 0.063, 28: 0.056, 29: 0.042, 30: 0.028, 31: 0.034, 32: 0.031, 33: 0.055, 34: 0.081, 35: 0.064 |

**Qwen3-4B, free run (8 prompts x 128 greedy tokens; divergence vs FLASH, 128 = none).**

| Arm | Prompts OK | Median first divergence | Mean token match | Divergences |
|---|---|---|---|---|
| FLASH | 8/8 | - | - | - |
| STD | 0/8 | - | - | - |
| 2P-TQ16 | 8/8 | 40.5 | 0.425 | [17, 43, 1, 38, 59, 5, 74, 91] |
| 2P-INT4 | 8/8 | 5.0 | 0.081 | [3, 14, 1, 8, 7, 1, 7, 3] |
| SP | 8/8 | 110.0 | 0.720 | [17, 128, 82, 128, 24, 128, 128, 92] |
| SP-tk8 | 8/8 | 128.0 | 0.740 | [128, 128, 128, 128, 7, 5, 128, 92] |

**Qwen3-4B, speed (RTX 4070 SUPER 12 GB; eager HF loop, B = 1; medians; x-FLASH = FLASH / arm).**

| S | Arm | Decode ms/token [IQR] | x-FLASH | Host ms | Attention ms/step [IQR] | x-FLASH attn | Phases ms (write / ABI materialize+pack+stage / kernel) | Prefill ms | x-FLASH prefill | Peak MiB (pre-session MiB; increment) | Outcome |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2048 | FLASH | 31.15 [31.06, 31.30] | 1.000 | 31.17 | 4.70 [4.69, 4.75] | 1.000 | 1.71 / 0.00 / 2.22 | 303 | 1.000 | 8520 (7922; 598) | OK |
| 2048 | STD | 39.25 [39.03, 39.45] | 0.794 | 39.27 | 19.02 [18.82, 19.15] | 0.247 | 0.39 / 0.00 / 18.50 | 853 | 0.355 | 9718 (7922; 1796) | OK |
| 2048 | 2P-TQ16 | 72.99 [72.66, 74.37] | 0.427 | 73.01 | 55.98 [55.22, 58.65] | 0.084 | 36.21 / 4.80 / 14.83 | 1681 | 0.180 | 8836 (7922; 914) | OK |
| 2048 | 2P-INT4 | 45.34 [45.17, 45.52] | 0.687 | 45.38 | 24.66 [24.51, 24.89] | 0.191 | 0.28 / 2.70 / 21.52 | 1544 | 0.196 | 8690 (7924; 766) | OK |
| 2048 | SP | 34.11 [33.86, 34.49] | 0.913 | 34.13 | 7.23 [7.22, 7.31] | 0.650 | 1.52 / 3.71 / 1.03 | 415 | 0.729 | 8690 (7924; 766) | OK |
| 2048 | SP-tk8 | 36.83 [36.78, 37.22] | 0.846 | 36.85 | 10.95 [10.88, 11.06] | 0.430 | 4.61 / 4.20 / 1.27 | 416 | 0.727 | 8834 (7924; 910) | OK |
| 8192 | FLASH | 32.12 [31.93, 32.56] | 1.000 | 32.15 | 6.38 [6.34, 6.47] | 1.000 | 1.76 / 0.00 / 4.11 | 1368 | 1.000 | 9456 (7922; 1534) | OK |
| 8192 | STD |  |  |  |  |  |  |  |  | - | OOM |
| 8192 | 2P-TQ16 | 77.75 [76.31, 80.38] | 0.413 | 77.78 | 57.92 [57.11, 58.45] | 0.110 | 8.21 / 16.13 / 33.43 | 23360 | 0.059 | 10186 (7922; 2264) | OK |
| 8192 | 2P-INT4 | 88.07 [87.97, 88.16] | 0.365 | 88.15 | 66.25 [66.20, 66.33] | 0.096 | 0.28 / 9.80 / 56.02 | 22048 | 0.062 | 9738 (7922; 1816) | OK |
| 8192 | SP | 34.59 [34.41, 35.07] | 0.929 | 34.61 | 15.53 [15.46, 15.61] | 0.411 | 1.59 / 11.44 / 1.92 | 2099 | 0.652 | 9740 (7922; 1818) | OK |
| 8192 | SP-tk8 | 39.23 [38.74, 40.29] | 0.819 | 39.26 | 19.92 [19.77, 20.12] | 0.320 | 4.46 / 12.96 / 2.25 | 2112 | 0.648 | 10104 (7922; 2182) | OK |
| 32768 | FLASH |  |  |  |  |  |  |  |  | - | OOM |
| 32768 | STD |  |  |  |  |  |  |  |  | - | OOM |
| 32768 | 2P-TQ16 |  |  |  |  |  |  |  |  | - | OOM |
| 32768 | 2P-INT4 |  |  |  |  |  |  |  |  | - | OOM |
| 32768 | SP |  |  |  |  |  |  |  |  | - | OOM |
| 32768 | SP-tk8 |  |  |  |  |  |  |  |  | - | OOM |

Variant-B entry cross-check (SP decode at 8K through `apa_selective_attention_sp_d1_packed`, `gqa_path=tc`): {"8192": {"outcome": "OK", "decode_ms": 34.54159927368164, "attn_ms": 15.51537600159645, "phases": {"cache_write": 1.5059679970145226, "materialize_pack_stage": 11.380208000540733, "kernel": 2.2104480043053627}, "prefill_ms": 2101.1661376953125}}

**Qwen3-4B, context ceiling (ladder 4K..40K, prefill + 32 decode steps, fresh process per rung) and stored bytes.**

| Arm | Bulk precision | Layout B/token (B/key/KV head, layers>=1) | NVML slope B/token (after_alloc, 4K->hi) | slope/layout | Ceiling (largest FIT) | First OOM (completed tokens) | Peak MiB per rung |
|---|---|---|---|---|---|---|---|
| FLASH | none (exact bf16 flash; K,V stored bf16 16 b/elem) | 147456 (512) | 147456 | 1.000 | 16384 | 24576 (0) | 4096:8890, 8192:9466, 12288:10042, 16384:10618, 24576:11488, 32768:NON_FIT_AFTER_OOM, 40960:NON_FIT_AFTER_OOM |
| STD | none (exact bf16 materialized math SDPA; K,V stored bf16 16 b/elem) | 147456 (512) | 147456 (hi 8192, flagged) | 1.000 | 4096 | 8192 (2048) | 4096:11302, 8192:10664, 12288:NON_FIT_AFTER_OOM, 16384:NON_FIT_AFTER_OOM, 24576:NON_FIT_AFTER_OOM, 32768:NON_FIT_AFTER_OOM, 40960:NON_FIT_AFTER_OOM |
| 2P-TQ16 | TQ4 reconstructed to bf16 kq (stored 16 b/elem, F-TQ4), bulk computed fp32 FMA on bf16; refine percentile 0.15 (zthr 1.03643) | 219136 (768) | 219136 (hi 12288, flagged) | 1.000 | 12288 | 16384 (0) | 4096:9308, 8192:10196, 12288:11084, 16384:7954, 24576:NON_FIT_AFTER_OOM, 32768:NON_FIT_AFTER_OOM, 40960:NON_FIT_AFTER_OOM |
| 2P-INT4 | INT4 sym-7 + fp32 scale per key (4.25 b/elem stored, 68 B/key/KV head), dequantized fp32 in registers, fp32 FMA; refine percentile 0.15 | 166496 (580) | 166912 | 1.002 | 16384 | 24576 (0) | 4096:9066, 8192:9748, 12288:10400, 16384:11086, 24576:7986, 32768:NON_FIT_AFTER_OOM, 40960:NON_FIT_AFTER_OOM |
| SP | TQ4, 4.25 b/elem stored (codes + fp32 norm); decode computed bf16 hi/lo with fp32 accumulate (EB3); prefill on the bf16 F-TQ4 reconstruction, bf16 tensor-core fp32 accumulate (K1); delta 8.1 row carry | 166496 (580) | 166912 | 1.002 | 16384 | 24576 (0) | 4096:9066, 8192:9750, 12288:10400, 16384:11086, 24576:7988, 32768:NON_FIT_AFTER_OOM, 40960:NON_FIT_AFTER_OOM |
| SP-tk8 | TQ4 as SP; V per 64-key block bf16 or int8 (per-token absmax, fp32 scale) by tk8 at f_L (EB4 T-fresh); prefill = SP | 203456 (712) | 204459 | 1.005 | 12288 | 16384 (0) | 4096:9284, 8192:10114, 12288:10908, 16384:11382, 24576:NON_FIT_AFTER_OOM, 32768:NON_FIT_AFTER_OOM, 40960:NON_FIT_AFTER_OOM |

## Findings and residuals

1. **Engine ABI forces an O(S) copy per decode step (finding for the lead).** Every engine entry (`apa_eb3`, `apa_eb4`,
   `apa_selective_attention`, `apa_selective_attention_int4`, the variant-B entry) reads K/V (and kq, int8 V, V scale) as
   contiguous `[B,KVH,S,D]` with S the visible length; only the packed codes take a capacity stride. A growing cache
   therefore materializes the prefix (`Tensor.slice`) for every call. On 1.7B at 32K this phase is 31.7 ms (SP) and
   35.4 ms (SP-tk8) of the summed attention time per step, against kernel time of 7.56 / 5.76 ms (FLASH: 9.23 ms, no
   copy, strided torch views). Minimal fix, stated not implemented: a K/V row-stride argument on the entries.
2. **Kernel-level ranking holds in the model loop; end-to-end does not.** Summed kernel time at 32K on 1.7B: SP-tk8
   5.76 < SP 7.56 < FLASH 9.23 ms per step, the EB3/EB4 ordering. End to end the order flips (FLASH 24.7, SP 49.1, SP-tk8
   52.6 ms/token) because of finding 1 and because the eager loop is host-bound (~25 ms/token floor). LA9 misses for
   this reason.
3. **Two-pass quality.** The two-pass rule refines 15.5-15.9% of keys per layer (realized fraction, every layer; the
   z-threshold is literal mean+z*std of |bulk|), yet loses model quality: 2P-TQ16 KL is ~110x the floor on 1.7B and
   ~13x on 4B. Its keys are the same TQ4 codes SP uses; SP's running-max rule (realized 2-80% per layer, early layers
   high) stays at the floor. 2P-INT4 (per-key absmax/7 INT4) is far worse on 1.7B (+571% PPL, top-1 0.40).
4. **STD = torch's math SDPA.** torch 2.11's default (`allow_fp16_bf16_reduction_math_sdp=False`) computes the math path
   in fp32 on bf16 inputs, so STD's quality equals the F32 floor instrument and its scores are materialized in fp32.
   This sets its ceilings (16K on 1.7B, 4K on 4B) and its prefill cost (0.21-0.36x FLASH).
5. **Harness defects found by gates (fixed, receipts kept).** (a) G1.5: the int8-V cache carried as bf16 pairs was
   NaN-canonicalized by the engine's bf16 slice (`dimcopy_kernel` converts through float); fixed with an fp32 carrier
   before registration. (b) Amendment 001: arm memory was released only after the caller dropped its reference; OOM
   and REF_OOM receipts it caused were superseded and re-run. (c) Amendment 002: the quality instruments' transients
   pushed 2P-TQ16 at 32K over the card; metric blocks were sub-blocked (bitwise identical) and the refine statistic
   computed per KV head.
6. **Peak baselines differ per run.** In speed runs the pre-session NVML baseline is 4,420 MiB except FLASH and STD at
   8K (4,122 MiB): their warmup sessions reused and released torch's model-load cache, the engine arms' did not. The
   speed table gives the peak, the pre-session baseline and the increment. Ceiling rungs (single session, fresh
   process) share one baseline per model (4,404 / 7,858 MiB). Quality-job peaks are not used as peak results.
7. **Borderline fits.** SP-tk8 on 4B at 16K: the G2 run (capacity exactly 16,384) completed at 11,660 MiB, but the
   ceiling rung (capacity 16,416 = 16K + 32 decode steps) OOMs: cudaMalloc's 2 MiB rounding adds ~280 MiB when the
   per-layer tensors are just over a power of two. The registered ceiling is the rung's: 12K.
8. **RSS rail breach (seat error).** One diagnostic run (`g1_3a_diagnostic`, 1.7B, first version) held every layer's
   submodule outputs on the host and peaked at 26.8 GB RSS against the 20 GB rail. No other job exceeded 12.1 GB.
9. **Order text vs layout.** The order lists 2P-INT4 at "4.5 bits per element stored"; the DF1 workspace stores 64 B of
   codes plus a 4 B fp32 scale per key per KV head, 4.25 bits per element (plus bf16 K and V: 580 B in total).

10. **SP prefill path and selection parity.** SP's prefill uses the SPO1 K1 Triton kernel (it covers causal GQA D = 128
   bf16 with a KV prefix) on the bf16 F-TQ4 reconstruction of the cache's own codes. On 8K window 0, its selection on
   the last 64 prefill rows disagrees with the decode kernel's (EB3, same rows and prefixes) on 5.0e-4 of decisions on
   1.7B and 5.3e-4 on 4B (per layer 1.9e-4 to 8.4e-4; `artifacts/apa_ab1/selparity/<model>.json`): near-threshold
   decisions between K1's bf16 tensor-core bulk and EB3's hi/lo bulk. Descriptive, run after registration.

## Remaining holes

- Qwen3-4B at 32K: no arm fits on this card (prefill quality REF_OOM on window 0 for every arm; the other eight 32K
  windows were not attempted because the reference allocation fails deterministically; speed OOM; ceiling < 32K for
  every arm). A 32K quality or speed number for 4B needs a larger card or an off-card tier (forbidden here).
- STD rows that OOM: 1.7B 32K (quality and speed); 4B 8K, 32K, decode mode, free run and 8K speed; 4B G1.5 STD.
- G1 RED items stand as registered: 1.7B G1.2 decode SP top-1 (2 tokens of 256) and G1.3a's attribution clause on both
  models (diagnosed, not re-registered). G2's slope is RED for torch-allocated caches (diagnosed). G3 mutant 1 is not
  established.
- Speed numbers are eager-HF numbers with host-side Python per layer; no CUDA-graph capture (EB3/EB4 lab limits forbid
  it for those entries). CPU contention from other tenants (load average 4-8 at the speed runs; recorded per receipt)
  is a noise source; medians over 64 synced steps with IQR are reported.
- End-to-end SP/SP-tk8 decode includes the ABI copy (finding 1). The phase split is a GPU-timeline event split inside a
  host-bound loop, so "kernel" phases can include launch gaps.
- Decode-mode quality uses 10 windows x 256 steps = 2,560 scored tokens per arm per model: CIs are wide (e.g. SP dPPL
  [-0.46, +0.17]% on 1.7B). The free run uses 8 prompts.
- The two-pass refine fractions are a torch replication of the kernel rule on sampled rows (not kernel counters); SP's
  come from K1/EB3/EB4 diagnostics. After amendment 002 the two-pass statistic is evaluated per KV head.
- The variant-B cross-check is timed only (EB3's G1 established bitwise identity with variant B).
- Model-level quality here is next-token metrics on one corpus; no task evaluation, no long-context retrieval.
- Author-run checks only; no blind review has been run.

## Prior art

- **FlashAttention / FlashAttention-2** (Dao et al. 2022; Dao 2023): the FLASH arm (torch's flash backend) and its
  bottom-right causal convention for chunked prefill with a prefix; K1's tiling. Taken unchanged.
- **Flash-Decoding** (Dao, Haziza, Massa, Sizov 2023): split-K decode with a merge, inside EB3/EB4. Taken unchanged.
- **TurboQuant** (Zandieh et al. 2025, arXiv 2504.19874): TQ4 codes, rotation and codebook; the F-TQ4 reconstruction
  formula from SPQ6. Taken unchanged.
- **KIVI** (Liu et al. 2024, arXiv 2402.02750): per-token value quantization lineage of EB2b's int8 V. Taken unchanged.
- **ThriftAttention** (Sharratt 2026, arXiv 2605.23081): block precision promotion (tk8). Taken unchanged via EB2b/EB4.
- **BLASST** (Yuan et al. 2025, arXiv 2512.12087): running-max criterion in SP's selection. Taken unchanged.
- **APA** (Perry 2026, docs/APA_PAPER_DRAFT.md 2.1-2.3): bulk/refine split with the full denominator; the two-pass rule
  (mean + z std of |bulk|). Taken unchanged.
- **House work** (Project-Tensor 2026): SPQ7/SPQ8 window protocol and DecodeBridge pattern; SP4G A7 ladder rule and
  NVML own-PID sampler; EB2b/EB3/EB4 kernels; DF1 INT4 workspace; SPO1 K1; P1T-M1 DLPack (not in this branch).
- **Interop**: pybind11 cross-module casting (Jakob et al. 2016); Numba CUDA Array Interface (2018); DLPack (2017).
  Ours: three read-only/metadata accessors.
- **Chunked prefill** in serving systems (e.g. SARATHI, Agrawal et al. 2023; vLLM). Unverified as a citation, lead to
  check: "SARATHI chunked prefill 2023". Taken: the loop shape only.
- **Statistics**: paired percentile bootstrap (Efron 1979); perplexity on disjoint windows (standard HF recipe);
  mutation testing (DeMillo, Lipton, Sayward 1978).
- Shell helpers `scripts/apa_ab1_{run_g2,waitlock,batch,run_ceiling}.sh`: no prior art known to me (job plumbing around
  util-linux flock and GNU timeout).
- No prior art known to me for the specific harness assembly (single-copy arm caches behind one HF hook, phase-split
  attention timing, the per-run baseline accounting). No novelty is claimed; nothing here is a new attention algorithm.

## Scope

- **Evidence classes:** end-to-end model evaluation (teacher-forced and free-run next-token metrics) on wikitext-2
  test; whole-process memory measurements (NVML own-PID); end-to-end eager-loop timings (CUDA events, host wall);
  author-run bitwise/fp64 reference checks (G1); compile-free mutation tests (G3).
- **Covers:** Qwen3-1.7B and Qwen3-4B, the six registered arms plus the F32 instrument, contexts 2K/8K/32K (quality and
  speed) and 4K-40K (ceilings), batch 1, bf16 weights, layer 0 exact for every APA arm, SP/SP-tk8 at delta 8.1 row carry,
  two-pass at refine percentile 0.15, SP-tk8 at EB2b's f_L with the nearest-layer fill (per-layer tables in the
  registration).
- **Does not cover:** other models (Tiers 2-3); other GPUs; batch > 1; CUDA-graph capture; task-level or long-context
  retrieval quality; host-RAM KV tiers (forbidden); production promotion; blind review.
- **Process:** no git, no subagents, no deletes (scratch moved to `artifacts/apa_ab1/scratch_retired/`; one
  `scripts/__pycache__` created by a CPU check was removed under the Python-cache exemption), GPU only under
  `flock -w 900 /tmp/forge-gpu.lock timeout 600`, native driver, no engine source change. Several foreground calls were
  moved to the background by the tool while queued on the lock; each was awaited to completion before the next GPU job.

## Every file written

All paths are relative to the worktree. `artifacts/apa_ab1/written_files.txt` holds the full sorted list (1835 entries including this report and the list itself); the exact brace-compressed listing below expands to the same set (checked by `scripts/apa_ab1_finalize.py`). Outside the worktree: nothing written by this seat (the tool harness kept its own background-task output files under <local scratch>).

```text
artifacts/apa_ab1/{amendment_001,amendment_002,gate_rules,registration}.{json,sha256}
artifacts/apa_ab1/{amendment_002_metrics_equivalence,gates_summary,windows}.json
artifacts/apa_ab1/tokens_test.npy
artifacts/apa_ab1/written_files.txt
artifacts/apa_ab1/build/{path_copy_build,path_copy_gpu}.diff
artifacts/apa_ab1/build/interop/_apa_ab1_interop.cpython-312-x86_64-linux-gnu.so
artifacts/apa_ab1/build/interop/module.sha256
artifacts/apa_ab1/build/main/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
artifacts/apa_ab1/build/main/{bindings,device_link,kernels}.o
artifacts/apa_ab1/build/main/module.sha256
artifacts/apa_ab1/build/triton_cache/{2GAM2PLXVOBGB5IFF5IIOQX6P5WPZTAI3BG5C3AT7G6U46SK4STQ,3J2IKH6FL7UMXBDIVBOPN4XYPIB5346KF63AE27VRU3QF6SXWQHQ,3PKERZAQN5VUGTZC6E6IMECPOO7QJDR5DR5VNABREYKB3C2AI4WQ,3QIPNB25JQDR6GCDCVX7AMGCMMDMNGBOBC6BHXCXH4TXNCNVJF4Q,5BHXNZPEK2SHCKT3CS5GV46WG42HKIP3GE2EAP6NXTQIETGSZANQ,5LEHQE3B2IW5XFAGVLZGGVGYE37UVJGXJ6CS2MXFYCXTWJBBBCRQ,6LTBRKRU3PQGVMB5HBGQKD2PS5XBFX7LNJP4AEIXVFUM7OU3W7VA,6QMH32OPD5N4TFYNTSAZNX7PJF5R5LIBAPMI57ISCAFW567JXYVA,7RBC2ORFWQLEYXTNH6ZRZ7LCVHIXDFQ3QQJZEMUI34GK7R2SV4GQ,B7TRPTCQIJ2LSUUTNZDZA5HS6EHRS3JA7FEPOHLLNVTQXWFVSYTA,BP3HF4TEATCLKFZNHCXZZFY2CRG66IXKNKYLB6JEEO6OWRDU5OTQ,BPHXKNXZGLJSMEFAIFMKXXLBCI4TI2VZLCHZU7ZMGILYI7BZI4WQ,BQB4WO4LB6GR5BFF6GLBMZEIGZ2CXZXZUAF47YFJECP2A3JEMZTQ,C3MY5MD3P6KBXAJMY46P5DMCBRCW5L57VKGHTCM2AN4PG43AVXXQ,CDTRCPUMC7Q66PGX6BH6FKGQL4SGDT2SYG5QEWE4265KJOA3ICBA,CDXFURSYPEGYM763SWTWW5B2WXDONJCFC7K7YHBMFXWKBHV3RODA,CZLXRNFNFLICZVVKM62B5JPR5O7TEXIITFUC7WLWNQX4ECND6X6A,DACONV7ZIJYHV3SMVH2KA2MBHCW2WKTVVA5IMMIYTHP6MNIBC3OA,DZK6T7L2CPFMN3RHE3W7HM5RIXCFBNDHURNEO65LS4YGPJB4D43Q,EZG3DSX2CRSO2SJCQLJ6UCSK22CO4LKKQC4F6BHTV66V2KMRAWJA,G5EQNF3JGICCH4Y33E4S5O3L73FLRKBIRLYGQLIIARDVUVEPKJ6A,GSEPSKG4IJ627AZJ4WSMYSQB376K7ADW3MHQ33VSAPQ7B3DV7HAA,H6CLRUVCFQGI6NM75X6EO33EQHMMZRDIRSYQAPMTFSSGDPL6HKQA,HSGIWEZKYFMXB7E25GXVNE3QCCEX7NTOYRIFB3VTDMRC2YN6UXVQ,HT2TKQ65DOIYSGMWXGRTYNF6COFTSXGJEG2R4NVE6BUPUG7ZYX3A,I7T3BV3AL4UVH6XQWBU5Z2IK5OWZB6V3WNJZC42WYUPJZC3GSHIA,J4ZAWTHOKLE7S4NDZDFOCSSSAXOOVL3ON5FFOH2LOVZPG22ZSSXA,JYBYY573FTGVKT775RN5YZX4S3O46IB4D7RIHLNB76FC5XB2DSNQ,KKG4V4IWNZNQ63F2XQ2UKP7JLOM5KKDBWF5M4NILTZXQIJQBISNA,L4HMTMSHQCUP7D74JAF5ZPCO3BWL36NEOSXOXRLRZWQFWWEVDN7A,LBXBCHAXS7XTN6TC2MIA7MY6FZNOMORSDEH3USD5JMLFU62OLGLQ,LEGFVKKCBMVTCDGXNDKFTTIOUX6CRISP5VPEKEFVZTGSG2LNOCWA,LTXQB473C66VIQPBB2SBEY5VI637QMYIEJ7TNF7HROWXFQHHWRTA,MBYGH2J6ISDWTKLGWPJU7GFH3V4PNKBN3OFAEP4JYE6WDEY44VTA,MRMPPM47XVNHWWEU7BHJ7DUEVMWOSQ5IQDED6VJF64R7A3FATY4Q,NDBR4VXUWDPD6VMTAPX6ZBI3DOKUI32UCYEAONHKRU5Q3PHH5HKA,NO5S75RDVNGXWHYDQRYO7MDGNOJEMXLKWNNON52SCQCAJK6HSFJA,ORWJFNDAAGMW5QR4WJZGQ7GCWQFJYKIW775FS2ECUW5HSQXKOTKA,P2OICOHYPBSTUD3OGM4XMSDZOJISUH4CHRJX2W3TTMUP4QWSRO5A,PI25MEJPI2WJ634ZK7NIFOQYDCNAEQFPEGAJ6FPVXKG5IWPEYZ4Q,QL3Y7KGHUJFQQYZTVU2F7PEOLJ5DG4D5CXZ6W5DUPGIJPBKTLLPA,RGMT4J5CFD5WTVCNXUJEYCJENIF4ACNEMCZKW3DLV5Q2IPQ7MD6A,RH5JJKI5OP6H6D7XTQS5YMPZ22WKJLYEPZTS57VEYEM2MZOZ6BIA,RJVGE3MV7F5UDZG4DS7YJREVMGG7UYSAUXYBD56XG4KU4FY6RJSQ,S5KX32RHEPWWGKTJKWR2ORU2Y7U77K2BZAY3Y3HPIGDKQW427UWQ,SN6D4IBEPDKINCIPEPAIT4DN3CSBOLYMCLDKVBBLQXCTLZBAIM4A,STIJ6EFYYIC72ZDLCTUEEUDX5D3K3C4BRTU6MPZLA6C2KWIDTBMA,TPPE2HZZ4FASNCYABLVJFPLYQGXNRZ7PWNUR5MQFI2O7K5JO35BA,U2RM2IU2NPDBDF2XBTWJLPIPZWQZIYGBFNRMIHPNXI7N3IOFYC2Q,UGPVMRAHWZVISQR4IL4LCFPFGJFWONIZZMLLIC3FZCYBDNNBBJ2Q,VQ7GMMPO474DGK3ENOGRIC43AA7TETIFOZNMBWXLNJ5QETBSLZHQ,VQPBOJVJQ3HYFDHWBUWCGSUDZJCTI6XZLD5P7VMPC6JYGPCRDVWA,VXDUYSBBI3A2VIGWT3ZKVKVPJ34IIKQCZ7FQJGAI35P6ND3F2NRQ,WL47NGLGKUWCUEW23WDJ4BJJJFR7VTMFLC255NA7WX7ISA47DORQ,WM2RYQNDJU4VVBXTE3L7EZXS7BSQU2JV5OKDBJKAHAWPMBTPLH5A,WMKHQU4HEAPDEBX5WXEJ7SNPPLKCYN6C5FQLYPRNSEAK6AYIBBBA,XSG6EGX7QI3BF6MAW4LOUKI6DQ7WA2ZNMZIOHIDMG3XUJ2BROR5A,YG5GHGJVKSUJBGFIWM7XCCQPN64GZV7HG4V647IMTH3QAWMOQA7A,YKNYPRIFH44LVDR47DJU3WYMI3BEYIPLHY4P62FWC7ZTPKVDOV7Q,YYQMFFI335SFCMTZFECQFBHWGJ4YKPU6PKBE6VEVF3SHVPQ2S5AA,ZGFL6MX4KZB5STRQJROIH56TBENF5KWEZKCULOCLGALUHH54SLSA,ZMUKP75KS4JCJEEYUHRF4FBFNNQCBPQ4PSJVAJQAOLB2N7J2235Q,ZSHBVCOHLS7A6O7K5WCYOCV7AARH5KLKQXN6VNOXNMRHIVNSVJGA,ZVAYKCTSAPKM5M7JQKCFM3PNI6A7QFLAPZTLDHFTYDMQ6BCRIBIQ}/__grp___k1.json
artifacts/apa_ab1/build/triton_cache/{2GAM2PLXVOBGB5IFF5IIOQX6P5WPZTAI3BG5C3AT7G6U46SK4STQ,3J2IKH6FL7UMXBDIVBOPN4XYPIB5346KF63AE27VRU3QF6SXWQHQ,3PKERZAQN5VUGTZC6E6IMECPOO7QJDR5DR5VNABREYKB3C2AI4WQ,3QIPNB25JQDR6GCDCVX7AMGCMMDMNGBOBC6BHXCXH4TXNCNVJF4Q,5BHXNZPEK2SHCKT3CS5GV46WG42HKIP3GE2EAP6NXTQIETGSZANQ,5LEHQE3B2IW5XFAGVLZGGVGYE37UVJGXJ6CS2MXFYCXTWJBBBCRQ,6LTBRKRU3PQGVMB5HBGQKD2PS5XBFX7LNJP4AEIXVFUM7OU3W7VA,6QMH32OPD5N4TFYNTSAZNX7PJF5R5LIBAPMI57ISCAFW567JXYVA,7RBC2ORFWQLEYXTNH6ZRZ7LCVHIXDFQ3QQJZEMUI34GK7R2SV4GQ,B7TRPTCQIJ2LSUUTNZDZA5HS6EHRS3JA7FEPOHLLNVTQXWFVSYTA,BP3HF4TEATCLKFZNHCXZZFY2CRG66IXKNKYLB6JEEO6OWRDU5OTQ,BPHXKNXZGLJSMEFAIFMKXXLBCI4TI2VZLCHZU7ZMGILYI7BZI4WQ,BQB4WO4LB6GR5BFF6GLBMZEIGZ2CXZXZUAF47YFJECP2A3JEMZTQ,C3MY5MD3P6KBXAJMY46P5DMCBRCW5L57VKGHTCM2AN4PG43AVXXQ,CDTRCPUMC7Q66PGX6BH6FKGQL4SGDT2SYG5QEWE4265KJOA3ICBA,CDXFURSYPEGYM763SWTWW5B2WXDONJCFC7K7YHBMFXWKBHV3RODA,CZLXRNFNFLICZVVKM62B5JPR5O7TEXIITFUC7WLWNQX4ECND6X6A,DACONV7ZIJYHV3SMVH2KA2MBHCW2WKTVVA5IMMIYTHP6MNIBC3OA,DZK6T7L2CPFMN3RHE3W7HM5RIXCFBNDHURNEO65LS4YGPJB4D43Q,EZG3DSX2CRSO2SJCQLJ6UCSK22CO4LKKQC4F6BHTV66V2KMRAWJA,G5EQNF3JGICCH4Y33E4S5O3L73FLRKBIRLYGQLIIARDVUVEPKJ6A,GSEPSKG4IJ627AZJ4WSMYSQB376K7ADW3MHQ33VSAPQ7B3DV7HAA,H6CLRUVCFQGI6NM75X6EO33EQHMMZRDIRSYQAPMTFSSGDPL6HKQA,HSGIWEZKYFMXB7E25GXVNE3QCCEX7NTOYRIFB3VTDMRC2YN6UXVQ,HT2TKQ65DOIYSGMWXGRTYNF6COFTSXGJEG2R4NVE6BUPUG7ZYX3A,I7T3BV3AL4UVH6XQWBU5Z2IK5OWZB6V3WNJZC42WYUPJZC3GSHIA,J4ZAWTHOKLE7S4NDZDFOCSSSAXOOVL3ON5FFOH2LOVZPG22ZSSXA,JYBYY573FTGVKT775RN5YZX4S3O46IB4D7RIHLNB76FC5XB2DSNQ,KKG4V4IWNZNQ63F2XQ2UKP7JLOM5KKDBWF5M4NILTZXQIJQBISNA,L4HMTMSHQCUP7D74JAF5ZPCO3BWL36NEOSXOXRLRZWQFWWEVDN7A,LBXBCHAXS7XTN6TC2MIA7MY6FZNOMORSDEH3USD5JMLFU62OLGLQ,LEGFVKKCBMVTCDGXNDKFTTIOUX6CRISP5VPEKEFVZTGSG2LNOCWA,LTXQB473C66VIQPBB2SBEY5VI637QMYIEJ7TNF7HROWXFQHHWRTA,MBYGH2J6ISDWTKLGWPJU7GFH3V4PNKBN3OFAEP4JYE6WDEY44VTA,MRMPPM47XVNHWWEU7BHJ7DUEVMWOSQ5IQDED6VJF64R7A3FATY4Q,NDBR4VXUWDPD6VMTAPX6ZBI3DOKUI32UCYEAONHKRU5Q3PHH5HKA,NO5S75RDVNGXWHYDQRYO7MDGNOJEMXLKWNNON52SCQCAJK6HSFJA,ORWJFNDAAGMW5QR4WJZGQ7GCWQFJYKIW775FS2ECUW5HSQXKOTKA,P2OICOHYPBSTUD3OGM4XMSDZOJISUH4CHRJX2W3TTMUP4QWSRO5A,PI25MEJPI2WJ634ZK7NIFOQYDCNAEQFPEGAJ6FPVXKG5IWPEYZ4Q,QL3Y7KGHUJFQQYZTVU2F7PEOLJ5DG4D5CXZ6W5DUPGIJPBKTLLPA,RGMT4J5CFD5WTVCNXUJEYCJENIF4ACNEMCZKW3DLV5Q2IPQ7MD6A,RH5JJKI5OP6H6D7XTQS5YMPZ22WKJLYEPZTS57VEYEM2MZOZ6BIA,RJVGE3MV7F5UDZG4DS7YJREVMGG7UYSAUXYBD56XG4KU4FY6RJSQ,S5KX32RHEPWWGKTJKWR2ORU2Y7U77K2BZAY3Y3HPIGDKQW427UWQ,SN6D4IBEPDKINCIPEPAIT4DN3CSBOLYMCLDKVBBLQXCTLZBAIM4A,STIJ6EFYYIC72ZDLCTUEEUDX5D3K3C4BRTU6MPZLA6C2KWIDTBMA,TPPE2HZZ4FASNCYABLVJFPLYQGXNRZ7PWNUR5MQFI2O7K5JO35BA,U2RM2IU2NPDBDF2XBTWJLPIPZWQZIYGBFNRMIHPNXI7N3IOFYC2Q,UGPVMRAHWZVISQR4IL4LCFPFGJFWONIZZMLLIC3FZCYBDNNBBJ2Q,VQ7GMMPO474DGK3ENOGRIC43AA7TETIFOZNMBWXLNJ5QETBSLZHQ,VQPBOJVJQ3HYFDHWBUWCGSUDZJCTI6XZLD5P7VMPC6JYGPCRDVWA,VXDUYSBBI3A2VIGWT3ZKVKVPJ34IIKQCZ7FQJGAI35P6ND3F2NRQ,WL47NGLGKUWCUEW23WDJ4BJJJFR7VTMFLC255NA7WX7ISA47DORQ,WM2RYQNDJU4VVBXTE3L7EZXS7BSQU2JV5OKDBJKAHAWPMBTPLH5A,WMKHQU4HEAPDEBX5WXEJ7SNPPLKCYN6C5FQLYPRNSEAK6AYIBBBA,XSG6EGX7QI3BF6MAW4LOUKI6DQ7WA2ZNMZIOHIDMG3XUJ2BROR5A,YG5GHGJVKSUJBGFIWM7XCCQPN64GZV7HG4V647IMTH3QAWMOQA7A,YKNYPRIFH44LVDR47DJU3WYMI3BEYIPLHY4P62FWC7ZTPKVDOV7Q,YYQMFFI335SFCMTZFECQFBHWGJ4YKPU6PKBE6VEVF3SHVPQ2S5AA,ZGFL6MX4KZB5STRQJROIH56TBENF5KWEZKCULOCLGALUHH54SLSA,ZMUKP75KS4JCJEEYUHRF4FBFNNQCBPQ4PSJVAJQAOLB2N7J2235Q,ZSHBVCOHLS7A6O7K5WCYOCV7AARH5KLKQXN6VNOXNMRHIVNSVJGA,ZVAYKCTSAPKM5M7JQKCFM3PNI6A7QFLAPZTLDHFTYDMQ6BCRIBIQ}/_k1.{cubin,json,llir,ptx,source,ttgir,ttir}
artifacts/apa_ab1/build/triton_cache/AXMFTKA7EPE6S5L5FLHVIMBVTBUB5PSYDHEUZP2AF3HIDJWWOOBQ/__triton_launcher.cpython-312-x86_64-linux-gnu.so
artifacts/apa_ab1/build/triton_cache/XU5DT2AO5BD5AEHEYGLPP5LRDFHHCUEJT4LGDVLB4STXUGVGHFPA/cuda_utils.cpython-312-x86_64-linux-gnu.so
artifacts/apa_ab1/ceiling/{Qwen3-1.7B,Qwen3-4B}/{2P-INT4_S12288,2P-INT4_S16384,2P-INT4_S24576,2P-INT4_S32768,2P-INT4_S4096,2P-INT4_S40960,2P-INT4_S8192,2P-TQ16_S12288,2P-TQ16_S16384,2P-TQ16_S24576,2P-TQ16_S32768,2P-TQ16_S4096,2P-TQ16_S40960,2P-TQ16_S8192,FLASH_S12288,FLASH_S16384,FLASH_S24576,FLASH_S32768,FLASH_S4096,FLASH_S40960,FLASH_S8192,SP-tk8_S12288,SP-tk8_S16384,SP-tk8_S24576,SP-tk8_S32768,SP-tk8_S4096,SP-tk8_S40960,SP-tk8_S8192,SP_S12288,SP_S16384,SP_S24576,SP_S32768,SP_S4096,SP_S40960,SP_S8192,STD_S12288,STD_S16384,STD_S24576,STD_S32768,STD_S4096,STD_S40960,STD_S8192}.json
artifacts/apa_ab1/dev/{residue_FLASH,residue_SP}.json
artifacts/apa_ab1/dev/smoke1/quality/Qwen3-1.7B/decode/{2P-INT4,2P-TQ16,FLASH,SP,SP-tk8,STD}/w0.{json,npz}
artifacts/apa_ab1/dev/smoke1/quality/Qwen3-1.7B/prefill2048/{2P-INT4,2P-TQ16,FLASH,SP,STD}/w0.{json,npz}
artifacts/apa_ab1/dev/smoke1/speed/Qwen3-1.7B/{2P-INT4_S8192,FLASH_S8192,SP-tk8_S8192,SP_S8192}.json
artifacts/apa_ab1/free/Qwen3-1.7B/{2P-INT4,2P-TQ16,FLASH,SP,SP-tk8,STD}/{p0,p12,p16,p20,p24,p28,p4,p8}.json
artifacts/apa_ab1/free/Qwen3-4B/{2P-INT4,2P-TQ16,FLASH,SP,SP-tk8,STD}/{p0,p12,p16,p20,p24,p28,p4,p8}.json
artifacts/apa_ab1/g0/{Qwen3-1.7B_smoke,build}.json
artifacts/apa_ab1/g0/superseded_harness_defect/Qwen3-1.7B_smoke.json
artifacts/apa_ab1/g1/{Qwen3-1.7B,Qwen3-4B}/{g1_1,g1_2,g1_3a,g1_3a_diagnostic,g1_3b,g1_4,g1_5_2P-INT4,g1_5_2P-TQ16,g1_5_FLASH,g1_5_SP,g1_5_SP-tk8,g1_5_STD}.json
artifacts/apa_ab1/g1/superseded_harness_defect/Qwen3-1.7B_g1_5_SP-tk8.json
artifacts/apa_ab1/g1/superseded_harness_defect/README.txt
artifacts/apa_ab1/g2/{allocator_diagnostic_S16384,allocator_diagnostic_S4096,result}.json
artifacts/apa_ab1/g2/Qwen3-1.7B/{2P-INT4_S16384,2P-INT4_S4096,2P-TQ16_S16384,2P-TQ16_S4096,FLASH_S16384,FLASH_S4096,SP-tk8_S16384,SP-tk8_S4096,SP_S16384,SP_S4096,STD_S16384,STD_S4096}.json
artifacts/apa_ab1/g2/Qwen3-4B/{2P-INT4_S16384,2P-INT4_S4096,2P-TQ16_S12288,2P-TQ16_S4096,FLASH_S16384,FLASH_S4096,SP-tk8_S16384,SP-tk8_S4096,SP_S16384,SP_S4096,STD_S4096}.json
artifacts/apa_ab1/g2/oom_receipts/{1792429083873086441_g2_Qwen3-4B,1792429110203379463_g2_Qwen3-4B,1792429126219748863_g2_Qwen3-4B,1792429315485616028_g2_Qwen3-4B}.json
artifacts/apa_ab1/g3/result.json
artifacts/apa_ab1/g3/mutant1/g2/Qwen3-1.7B/{FLASH_S16384,FLASH_S4096}.json
artifacts/apa_ab1/g3/mutant1_supplementary_SP/g2/Qwen3-1.7B/{SP_S16384,SP_S4096}.json
artifacts/apa_ab1/g3/mutant2/Qwen3-1.7B/g1_3b.json
artifacts/apa_ab1/g3/mutant3/Qwen3-1.7B/g1_5_SP.json
artifacts/apa_ab1/g3/mutant4/Qwen3-1.7B/g1_5_2P-TQ16.json
artifacts/apa_ab1/g3/mutant5/Qwen3-1.7B/g1_5_SP.json
artifacts/apa_ab1/g3/mutant6/g2/Qwen3-1.7B/FLASH_S4096.json
artifacts/apa_ab1/g3/mutant7/Qwen3-1.7B/g1_5_SP-tk8.json
artifacts/apa_ab1/quality/Qwen3-1.7B/decode/{2P-INT4,2P-TQ16,FLASH,SP,SP-tk8,STD}/{w0,w12,w16,w20,w24,w28,w32,w35,w4,w8}.{json,npz}
artifacts/apa_ab1/quality/Qwen3-1.7B/prefill2048/{2P-INT4,2P-TQ16,FLASH,SP,STD}/{w0,w12,w16,w20,w24,w28,w32,w35,w4,w8}.{json,npz}
artifacts/apa_ab1/quality/Qwen3-1.7B/prefill32768/{2P-INT4,2P-TQ16,FLASH,SP}/{w0,w1,w2,w3,w4,w5,w6,w7,w8}.{json,npz}
artifacts/apa_ab1/quality/Qwen3-1.7B/prefill32768/STD/{w0,w1,w2,w3,w4,w5,w6,w7,w8}.json
artifacts/apa_ab1/quality/Qwen3-1.7B/prefill8192/{2P-INT4,2P-TQ16,FLASH,SP,STD}/{w0,w12,w16,w20,w24,w28,w32,w35,w4,w8}.{json,npz}
artifacts/apa_ab1/quality/Qwen3-4B/decode/{2P-INT4,2P-TQ16,FLASH,SP,SP-tk8}/{w0,w12,w16,w20,w24,w28,w32,w35,w4,w8}.{json,npz}
artifacts/apa_ab1/quality/Qwen3-4B/decode/STD/{w0,w12,w16,w20,w24,w28,w32,w35,w4,w8}.json
artifacts/apa_ab1/quality/Qwen3-4B/prefill2048/{2P-INT4,2P-TQ16,FLASH,SP,STD}/{w0,w12,w16,w20,w24,w28,w32,w35,w4,w8}.{json,npz}
artifacts/apa_ab1/quality/Qwen3-4B/prefill32768/{2P-INT4,2P-TQ16,FLASH,SP,STD}/w0.json
artifacts/apa_ab1/quality/Qwen3-4B/prefill8192/{2P-INT4,2P-TQ16,FLASH,SP}/{w0,w12,w16,w20,w24,w28,w32,w35,w4,w8}.{json,npz}
artifacts/apa_ab1/quality/Qwen3-4B/prefill8192/STD/{w0,w12,w16,w20,w24,w28,w32,w35,w4,w8}.json
artifacts/apa_ab1/quality/superseded_amendment_001/Qwen3-1.7B_prefill32768/{2P-INT4_w2,2P-INT4_w3,2P-INT4_w4,2P-INT4_w5,2P-INT4_w6,2P-INT4_w7,2P-INT4_w8,2P-TQ16_w0,2P-TQ16_w1,2P-TQ16_w2,2P-TQ16_w3,2P-TQ16_w4,2P-TQ16_w5,2P-TQ16_w6,2P-TQ16_w7,2P-TQ16_w8}.json
artifacts/apa_ab1/quality/superseded_amendment_002/Qwen3-1.7B_prefill32768/{2P-TQ16_w0,2P-TQ16_w1,2P-TQ16_w2,2P-TQ16_w3}.json
artifacts/apa_ab1/scratch_retired/dev_smoke1_partial/errors/1792425617184303636_quality-decode_Qwen3-1.7B.json
artifacts/apa_ab1/scratch_retired/dev_smoke1_speed_v1/speed/Qwen3-1.7B/{FLASH_S8192,SP_S8192}.json
artifacts/apa_ab1/scratch_retired/g1_4b_oom_errors/{1792427256411498940_g11_Qwen3-4B,1792427311049742583_g13b_Qwen3-4B,1792427995200773304_g13a_Qwen3-4B,1792428024423181110_g13a_Qwen3-4B,1792428107325597539_g15_Qwen3-4B}.json
artifacts/apa_ab1/scratch_retired/sass_identity_dumps/{ab1_kernels,ab1_norm,eb4_kernels,eb4_norm}.sass
artifacts/apa_ab1/selparity/{Qwen3-1.7B,Qwen3-4B}.json
artifacts/apa_ab1/speed/{Qwen3-1.7B,Qwen3-4B}/{2P-INT4_S2048,2P-INT4_S32768,2P-INT4_S8192,2P-TQ16_S2048,2P-TQ16_S32768,2P-TQ16_S8192,FLASH_S2048,FLASH_S32768,FLASH_S8192,SP-tk8_S2048,SP-tk8_S32768,SP-tk8_S8192,SP_S2048,SP_S32768,SP_S8192,SP_S8192_variantB,STD_S2048,STD_S32768,STD_S8192}.json
artifacts/apa_ab1/summary/{ceilings,free,quality,speed}.json
artifacts/apa_ab1/summary/predictions.{json,md}
artifacts/apa_ab1/summary/{report_head,report_tail,tables}.md
docs/{APA_AB1_T1_LEDGER,APA_AB1_T1_REPORT}.md
logs/{apa_ab1_batch_status,apa_ab1_build_interop,apa_ab1_build_main,apa_ab1_ceil_Qwen3-1.7B_2P-INT4_S12288,apa_ab1_ceil_Qwen3-1.7B_2P-INT4_S16384,apa_ab1_ceil_Qwen3-1.7B_2P-INT4_S24576,apa_ab1_ceil_Qwen3-1.7B_2P-INT4_S32768,apa_ab1_ceil_Qwen3-1.7B_2P-INT4_S4096,apa_ab1_ceil_Qwen3-1.7B_2P-INT4_S40960,apa_ab1_ceil_Qwen3-1.7B_2P-INT4_S8192,apa_ab1_ceil_Qwen3-1.7B_2P-TQ16_S12288,apa_ab1_ceil_Qwen3-1.7B_2P-TQ16_S16384,apa_ab1_ceil_Qwen3-1.7B_2P-TQ16_S24576,apa_ab1_ceil_Qwen3-1.7B_2P-TQ16_S32768,apa_ab1_ceil_Qwen3-1.7B_2P-TQ16_S4096,apa_ab1_ceil_Qwen3-1.7B_2P-TQ16_S40960,apa_ab1_ceil_Qwen3-1.7B_2P-TQ16_S8192,apa_ab1_ceil_Qwen3-1.7B_FLASH_S12288,apa_ab1_ceil_Qwen3-1.7B_FLASH_S16384,apa_ab1_ceil_Qwen3-1.7B_FLASH_S24576,apa_ab1_ceil_Qwen3-1.7B_FLASH_S32768,apa_ab1_ceil_Qwen3-1.7B_FLASH_S4096,apa_ab1_ceil_Qwen3-1.7B_FLASH_S40960,apa_ab1_ceil_Qwen3-1.7B_FLASH_S8192,apa_ab1_ceil_Qwen3-1.7B_SP-tk8_S12288,apa_ab1_ceil_Qwen3-1.7B_SP-tk8_S16384,apa_ab1_ceil_Qwen3-1.7B_SP-tk8_S24576,apa_ab1_ceil_Qwen3-1.7B_SP-tk8_S32768,apa_ab1_ceil_Qwen3-1.7B_SP-tk8_S4096,apa_ab1_ceil_Qwen3-1.7B_SP-tk8_S40960,apa_ab1_ceil_Qwen3-1.7B_SP-tk8_S8192,apa_ab1_ceil_Qwen3-1.7B_SP_S12288,apa_ab1_ceil_Qwen3-1.7B_SP_S16384,apa_ab1_ceil_Qwen3-1.7B_SP_S24576,apa_ab1_ceil_Qwen3-1.7B_SP_S32768,apa_ab1_ceil_Qwen3-1.7B_SP_S4096,apa_ab1_ceil_Qwen3-1.7B_SP_S40960,apa_ab1_ceil_Qwen3-1.7B_SP_S8192,apa_ab1_ceil_Qwen3-1.7B_STD_S12288,apa_ab1_ceil_Qwen3-1.7B_STD_S16384,apa_ab1_ceil_Qwen3-1.7B_STD_S24576,apa_ab1_ceil_Qwen3-1.7B_STD_S4096,apa_ab1_ceil_Qwen3-1.7B_STD_S8192,apa_ab1_ceil_Qwen3-4B_2P-INT4_S12288,apa_ab1_ceil_Qwen3-4B_2P-INT4_S16384,apa_ab1_ceil_Qwen3-4B_2P-INT4_S24576,apa_ab1_ceil_Qwen3-4B_2P-INT4_S4096,apa_ab1_ceil_Qwen3-4B_2P-INT4_S8192,apa_ab1_ceil_Qwen3-4B_2P-TQ16_S12288,apa_ab1_ceil_Qwen3-4B_2P-TQ16_S16384,apa_ab1_ceil_Qwen3-4B_2P-TQ16_S4096,apa_ab1_ceil_Qwen3-4B_2P-TQ16_S8192,apa_ab1_ceil_Qwen3-4B_FLASH_S12288,apa_ab1_ceil_Qwen3-4B_FLASH_S16384,apa_ab1_ceil_Qwen3-4B_FLASH_S24576,apa_ab1_ceil_Qwen3-4B_FLASH_S4096,apa_ab1_ceil_Qwen3-4B_FLASH_S8192,apa_ab1_ceil_Qwen3-4B_SP-tk8_S12288,apa_ab1_ceil_Qwen3-4B_SP-tk8_S16384,apa_ab1_ceil_Qwen3-4B_SP-tk8_S4096,apa_ab1_ceil_Qwen3-4B_SP-tk8_S8192,apa_ab1_ceil_Qwen3-4B_SP_S12288,apa_ab1_ceil_Qwen3-4B_SP_S16384,apa_ab1_ceil_Qwen3-4B_SP_S24576,apa_ab1_ceil_Qwen3-4B_SP_S4096,apa_ab1_ceil_Qwen3-4B_SP_S8192,apa_ab1_ceil_Qwen3-4B_STD_S4096,apa_ab1_ceil_Qwen3-4B_STD_S8192,apa_ab1_cpu_tests,apa_ab1_d_17b_a,apa_ab1_d_17b_b,apa_ab1_d_17b_c,apa_ab1_d_17b_d,apa_ab1_d_4b_a,apa_ab1_d_4b_b,apa_ab1_d_4b_c,apa_ab1_d_4b_d,apa_ab1_d_4b_e,apa_ab1_dev_smoke1,apa_ab1_dev_smoke1_decode,apa_ab1_dev_smoke1_decode2,apa_ab1_dev_speed2_2P-INT4,apa_ab1_dev_speed2_FLASH,apa_ab1_dev_speed2_SP,apa_ab1_dev_speed2_SP-tk8,apa_ab1_dev_speed_FLASH,apa_ab1_dev_speed_SP,apa_ab1_f_17b_a,apa_ab1_f_17b_b,apa_ab1_f_4b_a,apa_ab1_f_4b_b,apa_ab1_g0,apa_ab1_g0_r2,apa_ab1_g1_g11_Qwen3-1.7B,apa_ab1_g1_g11_Qwen3-4B,apa_ab1_g1_g12_Qwen3-1.7B,apa_ab1_g1_g12_Qwen3-4B,apa_ab1_g1_g13a_Qwen3-1.7B,apa_ab1_g1_g13a_Qwen3-4B,apa_ab1_g1_g13b_Qwen3-1.7B,apa_ab1_g1_g13b_Qwen3-4B,apa_ab1_g1_g13diag_Qwen3-1.7B,apa_ab1_g1_g13diag_Qwen3-4B,apa_ab1_g1_g14_Qwen3-1.7B,apa_ab1_g1_g14_Qwen3-4B,apa_ab1_g1_g15_Qwen3-1.7B_a,apa_ab1_g1_g15_Qwen3-1.7B_b,apa_ab1_g1_g15_Qwen3-4B_a,apa_ab1_g1_g15_Qwen3-4B_b,apa_ab1_g2_Qwen3-1.7B_2P-INT4_S16384,apa_ab1_g2_Qwen3-1.7B_2P-INT4_S4096,apa_ab1_g2_Qwen3-1.7B_2P-TQ16_S16384,apa_ab1_g2_Qwen3-1.7B_2P-TQ16_S4096,apa_ab1_g2_Qwen3-1.7B_FLASH_S16384,apa_ab1_g2_Qwen3-1.7B_FLASH_S4096,apa_ab1_g2_Qwen3-1.7B_SP-tk8_S16384,apa_ab1_g2_Qwen3-1.7B_SP-tk8_S4096,apa_ab1_g2_Qwen3-1.7B_SP_S16384,apa_ab1_g2_Qwen3-1.7B_SP_S4096,apa_ab1_g2_Qwen3-1.7B_STD_S16384,apa_ab1_g2_Qwen3-1.7B_STD_S4096,apa_ab1_g2_Qwen3-4B_2P-INT4_S16384,apa_ab1_g2_Qwen3-4B_2P-INT4_S4096,apa_ab1_g2_Qwen3-4B_2P-TQ16_S12288,apa_ab1_g2_Qwen3-4B_2P-TQ16_S16384,apa_ab1_g2_Qwen3-4B_2P-TQ16_S4096,apa_ab1_g2_Qwen3-4B_FLASH_S16384,apa_ab1_g2_Qwen3-4B_FLASH_S4096,apa_ab1_g2_Qwen3-4B_SP-tk8_S16384,apa_ab1_g2_Qwen3-4B_SP-tk8_S4096,apa_ab1_g2_Qwen3-4B_SP_S16384,apa_ab1_g2_Qwen3-4B_SP_S4096,apa_ab1_g2_Qwen3-4B_STD_S12288,apa_ab1_g2_Qwen3-4B_STD_S16384,apa_ab1_g2_Qwen3-4B_STD_S4096,apa_ab1_g2_Qwen3-4B_STD_S8192,apa_ab1_g3_m1_S16384,apa_ab1_g3_m1_S4096,apa_ab1_g3_m1supp_S16384,apa_ab1_g3_m1supp_S4096,apa_ab1_g3_m2,apa_ab1_g3_m3,apa_ab1_g3_m4,apa_ab1_g3_m5,apa_ab1_g3_m6,apa_ab1_g3_m7,apa_ab1_q_17b_2k,apa_ab1_q_17b_32k_a,apa_ab1_q_17b_32k_b,apa_ab1_q_17b_32k_c,apa_ab1_q_17b_32k_d,apa_ab1_q_17b_32k_e,apa_ab1_q_17b_32k_f,apa_ab1_q_17b_32k_g,apa_ab1_q_17b_32k_h,apa_ab1_q_17b_32k_i,apa_ab1_q_17b_32k_j,apa_ab1_q_17b_32k_k,apa_ab1_q_17b_8k_a,apa_ab1_q_17b_8k_b,apa_ab1_q_4b_2k,apa_ab1_q_4b_32k,apa_ab1_q_4b_8k_a,apa_ab1_q_4b_8k_b,apa_ab1_registration,apa_ab1_selparity_Qwen3-1.7B,apa_ab1_selparity_Qwen3-4B,apa_ab1_speed_17b_2P-INT4_S2048,apa_ab1_speed_17b_2P-INT4_S32768,apa_ab1_speed_17b_2P-INT4_S8192,apa_ab1_speed_17b_2P-TQ16_S2048,apa_ab1_speed_17b_2P-TQ16_S32768,apa_ab1_speed_17b_2P-TQ16_S8192,apa_ab1_speed_17b_FLASH_S2048,apa_ab1_speed_17b_FLASH_S32768,apa_ab1_speed_17b_FLASH_S8192,apa_ab1_speed_17b_SP-tk8_S2048,apa_ab1_speed_17b_SP-tk8_S32768,apa_ab1_speed_17b_SP-tk8_S8192,apa_ab1_speed_17b_SP_S2048,apa_ab1_speed_17b_SP_S32768,apa_ab1_speed_17b_SP_S8192,apa_ab1_speed_17b_STD_S2048,apa_ab1_speed_17b_STD_S32768,apa_ab1_speed_17b_STD_S8192,apa_ab1_speed_4b_2P-INT4_S2048,apa_ab1_speed_4b_2P-INT4_S32768,apa_ab1_speed_4b_2P-INT4_S8192,apa_ab1_speed_4b_2P-TQ16_S2048,apa_ab1_speed_4b_2P-TQ16_S32768,apa_ab1_speed_4b_2P-TQ16_S8192,apa_ab1_speed_4b_FLASH_S2048,apa_ab1_speed_4b_FLASH_S32768,apa_ab1_speed_4b_FLASH_S8192,apa_ab1_speed_4b_SP-tk8_S2048,apa_ab1_speed_4b_SP-tk8_S32768,apa_ab1_speed_4b_SP-tk8_S8192,apa_ab1_speed_4b_SP_S2048,apa_ab1_speed_4b_SP_S32768,apa_ab1_speed_4b_SP_S8192,apa_ab1_speed_4b_STD_S2048,apa_ab1_speed_4b_STD_S32768,apa_ab1_speed_4b_STD_S8192,apa_ab1_speed_Qwen3-1.7B_SP_S8192_variantB,apa_ab1_speed_Qwen3-4B_SP_S8192_variantB}.log
scripts/{apa_ab1_arms,apa_ab1_common,apa_ab1_corpus,apa_ab1_dev_alloc,apa_ab1_dev_metrics_check,apa_ab1_dev_probe,apa_ab1_dev_residue,apa_ab1_finalize,apa_ab1_g1,apa_ab1_gates,apa_ab1_jobs,apa_ab1_nonfit,apa_ab1_register,apa_ab1_report,apa_ab1_runner,apa_ab1_selparity,apa_ab1_tuning}.py
scripts/{apa_ab1_batch,apa_ab1_build,apa_ab1_gpu,apa_ab1_interop_build,apa_ab1_run_ceiling,apa_ab1_run_g2,apa_ab1_waitlock}.sh
scripts/apa_ab1_interop.{cpp,py}
tests/test_apa_ab1_harness.py
```
