# APA-SPM2 ledger (seat: Claude Opus 5.5, Agent-tool subagent)

Order: `orders/APA_SPM2_ORIGINAL_SETUP_FLASH.md` (immutable). Worktree `tensor_fork_apa_wt_spm2`, branch `apa-spm2`.
No git, no subagents, no deletes. GPU: `scripts/apa_spm2_gpu.sh` = `flock -w 900 /tmp/forge-gpu.lock timeout 600`,
native driver 595.91.07 (the 595.84 override is refused by the wrapper, never set).

## Timeline (2026-10-07, EDT)

| When | What |
|---|---|
| 21:08 | Read order, SPM1 order/harness/report, SP3 RESULTS (A4 grid), SP5 §11, paper §4.1/§5, SP3 harness, GRM MiniCPM3/Gemma-4 ports, HOUSE_RULES. Mechanism check: the order measures materialized vs fused attention memory in the original MiniCPM3 setup (paper §5, §4.1(a)) — matches; no scope mismatch. |
| 21:09 | GPU idle: 249 MiB used by graphics clients (Xorg, cinnamon, wezterm); no compute process. Disk 96 GB free. |
| 21:18 | Built the cudaMalloc observer `artifacts/apa_spm2/build/libapa_spm2_peak.so` from `scripts/apa_spm2_peak.cpp` (design from SP3's `apa_sp3_peak.cpp`). |
| 21:19 | Pre-registration kernel probe `logs/apa_spm2_probe_r1.log` (not a ceiling run). |
| 21:22–21:26 | Harness smokes at S0=2048 (tag `smoke`, unscored): all three non-standard arms FIT; load 25–55 s. |
| 21:28 | Registration `artifacts/apa_spm2/registration.json`, sha256 `e38cf9b6688f4dc43b72be8b876c31d0e08970c48c5bd6ebb5a3a295bf3d6d04` (+ `tokens_gemma4.npy`, 549,572 tokens). |
| 21:29–21:33 | G1 runs (tag `g1`), then `scripts/apa_spm2_g1.py` -> `artifacts/apa_spm2/g1.json`: **PASS**. |
| 21:34 | First standard-ladder launch crashed before any rung (`TypeError` Path + str in the log path; `logs/apa_spm2_ladder_standard_r0_crash.log`). Fixed the driver; relaunched 21:35. No rung lost. |
| 21:35–21:48 | (A) standard ladder: 4K..19,456 FIT, 20,480 OOM -> ceiling **19,456** (`artifacts/apa_spm2/ladder_standard.json`). |
| 21:48–22:26 | (A) flash_class: 4K..28,672 FIT, 32,768 **RAIL** (killed by the lease in the last prefill chunk, layer 37). |
| 22:26–23:08 | (A) apa_sp: 4K..28,672 FIT, 32,768 **RAIL** (prefill done at 532 s; lease hit in decode step 4). |
| 23:09–23:47 | (A) apa_two_pass: 4K..24,576 FIT, 28,672 **RAIL** (last prefill chunk, layer 57). |
| 23:47 | (B) kernel grid r0: torch_flash_forced cells recorded `_fused_sdp_choice` as a TypeError (keyword-only args). Records moved to `artifacts/apa_spm2/kernel/grid_r0_superseded/`, logs renamed `*_r0_superseded.log`; call fixed; re-run. |
| 23:47–23:59 | (B) kernel grid r1. Six cells (torch_efficient_forced_expanded x5, torch_math 8192) RAN but crashed in the second, profiled identification call (generator context manager re-entered); no records written; logs renamed `*_r1_identify_crash.log`; fixed (fresh context per call) and re-run 23:59–00:00 (`logs/apa_spm2_kernel_grid_r2.log`). 60/60 cells recorded. |
| 00:00–00:50 | (B) model level, Gemma-4-12B QAT, standard: 4K..28,672 FIT, 32,768 **RAIL**. |
| 00:50–00:56 | (B) model level, apa: 4,096 FIT, 8,192 FIT. |
| 00:53:28–00:53:56 | **Environment change, not by this seat**: an `apt-get install --assume-yes … nvidia-driver-595-open …` upgraded the NVIDIA userspace 595.91.07 -> 595.99.02 while kernel module 595.91.07 stays loaded. `nvidia-smi`: "Failed to initialize NVML: Driver/library version mismatch"; dmesg "NVRM: API mismatch". Receipt `artifacts/apa_spm2/gemma_blocked_driver_mismatch.json`. The apa 8,192 rung had mapped the 595.91 libraries before the swap (nvidia-smi 595.91.07 recorded at its start). |
| 00:56–01:12 | The apa ladder sat in its foreign-compute wait loop (nvidia-smi failing). The seat stopped its own driver process (pid 565310) at ~01:12. Gemma apa from 12,288 and Gemma flash_class: **BLOCKED_ENVIRONMENT, not run**. No reboot, module reload or userspace override (native driver only; outside seat authority). |
| 01:20 | `scripts/apa_spm2_tables.py` -> `artifacts/apa_spm2/tables.md`; `scripts/apa_spm2_verdict.py` -> `artifacts/apa_spm2/verdict.json`. |

## Decisions and deviations (recorded as they were made)

1. **Engine binary**: SPM1's pinned SPQ8 build (sha256 `a9e1b0ab…4bb`), reused read-only, as the order allows. No build.
2. **flash_class δ**: the SP entry rejects non-finite δ (`kernels.cu`: "delta finite nonnegative"), so δ = FLT_MAX
   (3.4028235e38), which refines every key (APA-SP3 arm D used the same value). This is the order's "δ=+inf" in the
   only form the entry accepts.
3. **flash_class injection point**: the port's `standard` branch with the engine SDPA replaced by the SP entry
   (kq aliased to K, V unpadded at 64). The APA branch would have computed and stored a TurboQuant kq and a padded V
   first; the standard branch carries neither, which is what a fused exact kernel needs.
4. **Scores dtype**: the order's background says engine standard builds fp32 S×S scores. In this engine at bf16 compute
   `tc.matmul` returns the input dtype (`tensor_cuda/src/matmul.cu:44` region, `NDArray out(out_shape, a.dtype, ...)`),
   so standard's scores, masked scores and softmax are bf16 L×S tensors. Recorded; it changes the analytic wall, not the arm.
5. **Chunking**: 2048 (SPM1's protocol), applied to every arm. SP3's A4 grid and the paper's 3K→32K used a single full
   prefill; the difference is explained in the report.
6. **Ladders run detached** (`setsid nohup`) with the seat waiting in the foreground on the driver PID: one rung can take
   the full 600 s lease, which does not fit a single 600 s tool call. Every rung is still one fresh process under the lease.
7. **Optional HF torch-flash MiniCPM3 arm**: not run (bf16 HF weights ~8 GB on a 12 GB card; not comparable with INT4).
8. **Harness fixes after registration** (no protocol change; every record carries its script sha256): ladder log-path
   bug (before any rung); kernel `_fused_sdp_choice` keyword call; kernel identification context re-entry; kernel peak
   read moved before the finiteness check (that check allocates). The registration's arms, ladders, gates and
   predictions are unchanged.
9. **Model level stopped by the environment** (item above); not a seat decision about the arms.

## OOM adjudication (one line each)

- (A) standard 20,480: `prefill 18432:20480`, layer 59, `RuntimeError: cudaMalloc failed: out of memory` in the engine
  SDPA's `tc.matmul(weights, value)` while two bf16 40×2048×20,480 score-class tensors (3.1 GiB each) are live; engine
  high-water 11,144 MiB, boundary min free 7,074 MiB (the scores are intra-call transients).
- (B) torch_default / torch_math 16,384: MATH backend, "Tried to allocate 16.00 GiB" = 16×16,384²×4 B fp32 scores.
- (B) torch_default / torch_math 32,768: "Tried to allocate 64.00 GiB" = 16×32,768²×4 B.
- (B) torch_default / torch_math 49,152 and 65,536: fails earlier, on 9.00 GiB (= 49,152²×4 B) and 4.00 GiB
  (= 65,536²×1 B) S×S-class allocations, after the math path's fp32 copies already hold 6.3 / 9.4 GiB.
- (B) engine_standard_sdpa 16,384: `scores + _causal_mask(...)` after the 8 GiB bf16 16×S×S scores (high-water 10,240 MiB);
  32K+: the first `tc.matmul(query, key)` score allocation.
- (B) engine_port_global_standard 16,384: `tc.causal_softmax(sc)` output after the 8 GiB scores (high-water 8,192 MiB);
  32K+: the first score `tc.matmul`.

## Results summary (receipts: `artifacts/apa_spm2/tables.md`, `artifacts/apa_spm2/verdict.json`)

- (A) ceilings at 1,024 resolution: standard **19,456** (OOM 20,480). flash_class, apa_sp: last FIT 28,672, RAIL 32,768.
  apa_two_pass: last FIT 24,576, RAIL 28,672. Fused-arm memory ceilings: NOT measured (lease).
- (A) engine cudaMalloc high-water slope over FIT rungs: flash_class 97.5 MiB per 1K tokens; apa_sp and apa_two_pass
  185.5 MiB per 1K (identical from 12,288 up: the port's whole-span kq re-quantization, not the kernel); standard
  ~405 MiB per 1K (score tensors).
- (B) kernel: torch flash REFUSED (D>256), cuDNN REFUSED (head_dim>128), efficient REFUSED with GQA (1 vs 16 heads),
  default+GQA -> MATH, OOM at 16,384; default/efficient with K/V expanded -> EFFICIENT to 65,536 (3,072 MiB); engine
  standard paths OOM at 16,384; engine apa_selective / apa_int4 / SP-exact run to 65,536 at 1,089 / 1,040 / 1,024 MiB.
- (B) model: Gemma standard FIT to 28,672 (high-water 6,850 -> 7,393 MiB, ~25 KB per token), RAIL 32,768; apa FIT 4,096
  and 8,192; then BLOCKED_ENVIRONMENT.
- Gates: G1 PASS; G2 PASS for every ladder that ran (Gemma apa/flash_class blocked); G3 PASS (60/60 cells); Sanity PASS.

## Probe (pre-registration, `logs/apa_spm2_probe_r1.log`)

Per layer call at MiniCPM3 prefill-chunk shape (H=40, D=96, L=2048, random bf16):

| Entry | S=8,192 | S=32,768 |
|---|---:|---:|
| SP δ=4.28125 (apa_sp) | 0.221 s | 0.928 s |
| SP δ=FLT_MAX, kq=K, VD=64 (flash_class) | 0.251 s | 1.094 s |
| two-pass `apa_selective_attention` | 0.279 s | 1.277 s |

Gemma-4 global geometry (16q/1kv, D=512, L=S=4096, causal): `apa_selective_attention` 1.60 s; `apa_selective_attention_int4`
0.185 s; SP exact 0.320 s. All three accept D=512 (TC_APA_MAXD = 512).

Consequence registered as S2: the three fused (A) arms are expected to hit the 600 s lease (RAIL) before any memory wall.

## G1 (S0=4096, chunk 2048, 16 decode steps; reference = standard)

| Arm vs standard | Mean KL | Max KL | Top-1 | Threshold | Verdict |
|---|---:|---:|---:|---|---|
| flash_class | 4.36e-4 | 1.59e-3 | 15/16 | KL ≤ 1e-3, top-1 ≥ 15/16 | PASS |
| apa_sp | 5.31e-4 | 1.25e-3 | 16/16 | KL ≤ 5e-2, top-1 ≥ 12/16 | PASS |
| apa_two_pass | 9.32e-3 | 4.57e-2 | 15/16 | KL ≤ 5e-2, top-1 ≥ 12/16 | PASS |

Free after load 9,011.6 MiB in every run. Latent bytes measured = analytic (35,712 B/token) in every G1 run.

## Prior art (this ledger's entries)

- FlashAttention (Dao et al., 2022) and FlashAttention-2 (Dao, 2023), head_dim ≤ 256: the class flash_class stands in for.
- Online softmax (Milakov & Gimelshein, 2018).
- PyTorch SDPA backend dispatch (flash / mem-efficient (xFormers cutlass FMHA, Lefaudeux et al. 2022; Rabe & Staats
  2021) / cuDNN / math), `torch._fused_sdp_choice`.
- MLA latent cache: DeepSeek-V2 (2024); MiniCPM3 (MiniCPM Team, 2024).
- Chunked prefill: Sarathi (Agrawal et al., 2023); vLLM (Kwon et al., 2023).
- APA (Perry, 2026); APA-SP's running max follows BLASST (arXiv 2512.12087); TurboQuant (arXiv 2504.19874).
- LD_PRELOAD allocation interposition: standard Unix practice, design from APA-SP3 `apa_sp3_peak.cpp`.
- Ceiling protocol: APA-SPM1, APA-SP5 §11 (2026).
- Citations from memory, unverified in this sandbox — lead to check ("FlashAttention-2 head dimension 256",
  "PyTorch scaled_dot_product_attention backends", "xFormers memory-efficient attention", "Self-attention does not need
  O(n^2) memory Rabe Staats").

## Lead addendum, 2026-10-08: Gemma model level re-run on driver 595.99.02

Run by the lead (Claude Opus 5.5) after the author rebooted at about 09:35 and NVML and the kernel module both reported
595.99.02. The seat's records, `verdict.json` and `tables.md` are unchanged. Receipts: `artifacts/apa_spm2/gemma_r2_table.md`,
`artifacts/apa_spm2/verdict_gemma_r2.json` (from `scripts/apa_spm2_gemma_r2_tables.py`), rung records
`artifacts/apa_spm2/gemma/ceiling_r2/`, ladders `ladder_gemma_{apa,flash_class}_ceiling_r2.json`, logs
`logs/apa_spm2_G_ceiling_r2_*.log` and `logs/apa_spm2_gemma_r2_chain.log`.

- **Deviation:** `scripts/apa_spm2_gemma_ladder.py` hardcodes tag `ceiling`, and those records are write-once. The lead
  copy `scripts/apa_spm2_gemma_ladder_tagged.py` adds only `argv[2]` = tag. Run script, registration, port, tokens,
  observer and lease are unchanged (record `script_sha256` d12aace4…, same as the seat's).
- **Driver control: PASS.** standard S0=8,192 and apa S0=4,096 / 8,192 on both drivers: observer peak byte-identical
  (6,921.19 / 6,895.84 / 7,003.81 MiB), min free within 6 MiB.
- **Ladders** (chain 11:21:58–12:50:26 EDT: control 11:22–11:25, apa 11:25–12:09, flash_class 12:09–12:50; the first
  wording said 12:21, corrected after review F13): apa FIT 4,096..24,576, RAIL 28,672 (lease hit at prefill pos 25,728, layer 35);
  flash_class FIT 4,096..20,480, RAIL 24,576 (pos 23,552, layer 17). No OOM in any Gemma arm.
- **Memory** (observer peak slope over FIT rungs): standard 22.9, flash_class 22.7, apa 32.2 MiB per 1K tokens.
  flash_class < apa at every common S0 (S3 holds on Gemma too); flash_class ≈ standard. Min free at the last FIT:
  standard 3,776 MiB (28,672), apa 3,649 MiB (24,576), flash_class 3,951 MiB (20,480). Linear projections of the memory
  wall (unmeasured): standard ~213K, apa ~152K, flash_class ~217K.
- **Why everything RAILs, not OOMs** (corrected after APA-SPM2-REV F03; the first wording claimed a fixed ~32 MB
  score bound past ~16K and that "only stored state differs", and both were wrong): the port's `__call__` (GRM
  `core/gemma4_tc.py` ~849–890) shrinks the prefill chunk with context, step = ⌊1 MiB / S_ctx⌋ tokens rounded down to
  a multiple of 64, floor 64 (full rule, review N02: step = max(64, floor_to_64(min(512, max(64, ⌊1,048,576 / (pos + 512)⌋))))). The step reaches 64 at about 8K context. Up to S = 16,384 each global bf16 score tensor
  (16 × step × S × 2 B) stays ≤ 32 MiB. Past that the floor binds and it grows with S: 64 MiB at 32,768, 128 MiB with
  the mask-add copy. That is small next to the 3.6+ GiB free, so the lease binds first, and the ladders rank by
  prefill speed: standard < apa (1.02–1.25x) < flash_class (1.07–1.33x). Score transients do differ between arms
  (standard materializes them, flash_class does not), but the observer peaks do not show it at these S (standard
  7,193 vs flash_class 7,200 MiB at 20,480).
- **apa's extra 9.2 MiB/1K:** the port keeps a full-D quantized-key ring `kqb` per global layer for decode
  (`gemma4_tc.py:353–370`), and its prefill path re-quantizes the whole key span on every chunk (`:685–704`). At
  8 global layers × D 512, a bf16 ring is 8 KiB/token. Review probe C06 confirmed the ring is bf16; the split of
  the rest of the 9.2 MiB/1K (returned full-span kq copies, prefill re-quantization) is still not isolated.
- **Wiring:** the SP entry ran in every flash_class rung (`sp_calls` 224..2,275) and in no standard or apa rung.
- **Gates on the addendum:** G2 PASS (every Gemma ladder ends on an explained RAIL, `signal 15 (lease timeout)`, still
  in prefill). S8 consistent (standard and apa RAIL).

## Lead corrections after APA-SPM2-REV (Codex gpt-6.1-sol high, 2026-10-08)

Review: `apa-spm2rev` ad84008, `docs/APA_SPM2_REVIEW.md`, `artifacts/apa_spm2rev/verdict.json`: REJECT as citable on
F01 and F03; every measured number reproduced (415 table fields, 0 mismatches). Lead-run GPU probes:
`artifacts/apa_spm2rev/gpu_lead/LEAD_RUN_20261008.md` (5/6 PASS; C05 RED as registered, boundary reproduced).

| Finding | Claim | Correction |
|---|---|---|
| F01 (BREAK) | Report: the APA kernels use ~⅓ of torch EFFICIENT's 3,072 MiB "only because EFFICIENT needs K/V expanded" | Wrong. At D=512 EFFICIENT allocates the bf16 output (1,024 MiB at 64K) plus an fp32 output accumulator (2,048 MiB). K/V are stride-0 views and allocate nothing. Probe C03: the delta is 3,221,225,472 B whether K/V are views or materialized. The exact SP entry allocates only its output at prefill (1,024 MiB) plus linear decode scratch (C02); the INT4 entry adds 16 MiB packed keys and 0.25 MiB scales (1,040.25 MiB), and the selective port call 1,089.0 MiB including caller quantization (narrowed after review N01). Expansion is what lets EFFICIENT accept GQA at all; it is not where the memory goes. |
| F03 (BREAK) | Lead addendum: fixed ~32 MB score bound past ~16K; only stored state differs | Wrong; corrected in place above. |
| F04 | Report scored S8 "Partial" | S8 held: the prediction reads "≤ 16384 or RAIL", and standard and apa RAILed. |
| F05 | Observer and free-memory columns | The observer is exact for interposed cudaMalloc payloads only (C01: async bytes, the driver API, other runtime handles and torch's allocator are invisible). Engine rows (observer) and torch rows (torch allocator) are payload deltas, not comparable total-device peaks. "Min free at layer boundary" is not an intra-call minimum and misses transients. |
| F06 | "flash_class matches standard to bf16 precision" | G1 passes its registered wiring gate (mean KL ≤ 1e-3, top-1 ≥ 15/16). That is not bitwise or ULP equality. The kernel's exactness is established by source (F15) and probe C02 (relative RMSE ~1.65e-3 vs fp64). |
| F08 | MiniCPM3 APA's extra slope comes from the port's whole-span re-quantization | Supported, not exclusive. C06: the quantizer grows 92,320 B/token = 88.0 MiB per 1,000 tokens, which matches the 185.5 − 97.5 = 88 MiB/1K gap between APA and flash_class. No allocation-site ablation was done. |
| F02 | Gemma flash_class "no kq storage" | True of the SP call (kq aliased to K, no ring), but the port's caller still slices and concatenates K into an unused kq copy at S > 4096 (C06: twice K's bytes). Gemma flash_class model-level memory therefore slightly overstates a minimal fused caller. |
| F11 | Slopes quoted in one unit | MiniCPM3 slopes (97.5 / 185.5 / ~405) are approximate per-1,000-token endpoint/tail slopes; the Gemma slopes are least-squares per 1,024 tokens over FIT rungs. |
| F13 | Driver control and r2 runs | The r2 records show NVML 595.99.02, but the SPM2 harness did not enforce NVML == kernel module, and three small controls do not show timing invariance across drivers. The r2 rail ranking stands on the r2 runs alone. |
| F14 | Paper 8 GB 3K→32K comparison | The cited receipt documents the historical card and PPL, not the context run; provenance of the 3K→32K run is incomplete. |
| F16 | Detached ladder drivers | A disclosed procedural deviation (seat deviation 5); every rung was a fresh leased process. No numerical effect demonstrated: the replay shows record/table consistency, not a counterfactual (review N03). |
| C05 | Standard OOM site | Reproduced at the same rung and prefill chunk (peak 11,165.5 vs 11,144.3 MiB), but at layer 61 in the mask add, not layer 59 in the final matmul. The cause (a second 3.2 GiB score-class tensor while one is live) stands; the exact failing allocation varies run to run. |

## Citable claims after review round 2 (APA-SPM2-REV-R2: APPROVE-WITH-NITS, no blocking findings)

Receipt: `apa-spm2rev` `docs/APA_SPM2_REVIEW_R2.md`, `artifacts/apa_spm2rev/r2/verdict.json`. Only these claims are
citable, each with its qualification. Do not cite the report draft without its correction context.

| Id | Claim | Qualification |
|---|---|---|
| M1 | MiniCPM3-4B: standard measured ceiling 19,456 (OOM 20,480); exact (flash-class) and APA-SP FIT 28,672, RAIL 32,768; two-pass FIT 24,576, RAIL 28,672; at least 1.47x completed context over materialized standard. | INT4 GRM port, 2,048-token chunks + 16 decode, 12 GB card, 600 s lease; 1,024-token bisection; fused memory ceilings unknown; the C05 OOM-site replication is RED (boundary reproduced). |
| M2 | Gemma-4 D=512 globals (bf16, causal, 16q/1kv, full prefill): torch flash refuses; default SDPA with GQA picks MATH and first OOMs at 16K; EFFICIENT with expanded heads runs to 64K; the engine fused kernels run to 64K; both engine standard paths first OOM at 16K. | The recorded torch 2.11 stack and operands only; it does not explain the external report's stack. |
| M3 | At 64K: EFFICIENT 3,072 MiB (1,024 output + 2,048 fp32 accumulator); exact SP 1,024, INT4 1,040.25, selective port 1,089.0 MiB. | Each counter's own payload delta, not total-device peaks; output-only holds for exact-SP prefill only. |
| M4 | Gemma-4-12B model level: last FIT / RAIL standard 28,672 / 32,768, APA 24,576 / 28,672, flash-class 20,480 / 24,576; no OOM observed. The adaptive chunk floor makes one score tensor 64 MiB at 32K, not a fixed 32 MiB cap. | RAIL is unknown fit; similar observer peaks do not erase score-transient differences; no speed claim. |
| M5 | MiniCPM3 APA's excess memory is dominated by the port's whole-span key re-quantization (92,320 B/token, matching the 88.04 MiB per 1,000 tokens gap). Gemma's kqb ring is bf16 and caller copies exist. | No exclusive whole-model decomposition; the Gemma 9.2 MiB/1,024 residual is not isolated. |
| M6 | The exact SP entry is fused all-key attention (diagnostics off); six fp64-oracle cases within 0.01 max-abs / relative RMSE with exact output and linear-scratch memory; G1 meets its registered wiring gate. | Tolerance-tested, not bitwise or ULP equality, not a model-quality certification. |
| M7 | P1 undecided; P2 and S1–S8 held in their registered scope; P3 consistent in the tested arms; slopes and projections reproduce. | P3 is not a universal theorem; MiniCPM3 slopes are per-1,000 endpoint/tail, Gemma per-1,024 OLS; projections are arithmetic only; the historical 8 GB 3K→32K result is not verified by SPM2 (F14). |
