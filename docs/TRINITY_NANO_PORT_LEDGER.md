# Trinity Nano Port — Implementation Ledger

Receipts for the Trinity Nano port. Plan: docs/TRINITY_NANO_PORT_PLAN.md
(immutable). Synthesis: docs/TRINITY_AFMOE_SYNTHESIS.md.

## 2026-07-08 (opening)

Work order opened (David: "Lets try that Arcee Trinity - Biggest version
that'll fit on the card"). Sizing receipt: Mini (26B-A3B, config fetched)
needs 13.8-14.6GB at INT4-with-scales vs 12,282 MiB card — not resident;
expert-streaming rejected (7-15ms/token PCIe tax + weeks of work). Nano
(6.4B-A1B, config fetched via Trinity-Nano-Preview; bare Trinity-Nano
401s) selected. Predictions T1-T4 frozen in plan before any code.

Next action: P0 architecture scout (dispatched: Grok, read-only).

## 2026-07-08 (P0 complete — NoPE CONFIRMED; no blockers; license clear)

Action: architecture scout (Grok, read-only, code citations) — full
findings docs/TRINITY_NANO_P0_FINDINGS.md (fold-in accepted).

- **T1 CONDITION HOLDS: full_attention layers are NoPE** — RoPE applied
  only when layer_types[i]=="sliding_attention" (modeling_afmoe.py ~326,
  374-376; no else branch). 14 full layers at indices 3,7,...,55.
- Port map (parity-critical): gated attention = sigmoid(gate_proj(h))
  ⊙ attn_out before o_proj, elementwise heads×head_dim; MoE router =
  sigmoid → topk(scores+expert_bias) → gather → route_norm → ×2.826 →
  shared expert + scatter_add (expert_bias in top-k SELECTION only);
  QK-Norm standard RMS (ones-init, NOT zero-centered); softmax fp32;
  scale 1/√128; muP embed ×√1024; untied lm_head; dual pre/post
  residual norms; dense layers 0-1 SwiGLU inter=3072; sliding RoPE
  θ=10000 no scaling, window 2048.
- License: OpenMDW-1.1 (permissive, retain notices). Download: 3 shards,
  12.24 GB. Early-stop rail NOT triggered.

Next action: weights download (started), then P1 reference capture.

## 2026-07-08 (P1/P2 implementation pass - P1 captured; P2 parity RED)

Hard-rail receipts:

- No commit made. `docs/TRINITY_NANO_PORT_PLAN.md` was not edited.
- Download rail passed: `/home/vader/.claude/jobs/5b7f4f6b/tmp/trinity_download.log:3`
  contains `DOWNLOAD-DONE`.
- GPU work stayed bounded: each TensorCUDA GPU receipt below ran under 10 min.

Files added:

- `core/__init__.py:1` exports the Trinity Nano TensorCUDA class.
- `core/trinity_nano_tc.py:120` explicit fp32 softmax attention path;
  `:144` dense LinearTC; `:165` affine INT4 QuantLinearTC; `:208`
  RMSNormTC; `:223` host embedding; `:240` safetensor shard source; `:287`
  config loader; `:365` RoPE cache; `:415` Trinity MoE/router; `:497`
  attention with NoPE/sliding RoPE and sigmoid attention gate; `:581`
  decoder block with dual pre/post norms; `:689` full TrinityNano_TC with
  `kv_caches`/`caches`, `last_token_only`, bf16 and INT4 load modes.
- `scripts/trinity_nano_reference_capture.py:66` tokenizer receipts; `:258`
  HF greedy/reference capture plus all-layer pre-RoPE K/V capture; `:344`
  local-only CPU bf16 runner with remote-code compatibility shims.
- `scripts/trinity_nano_tc_parity.py:113` layer-streaming parity runner;
  `:202` logit compare; `:215` pre-RoPE K/V compare; `:282`
  incremental-vs-refeed check; `:343` receipt driver.
- `scripts/trinity_nano_int4_residency.py:42` all-resident INT4 load receipt.

P1 reference capture:

- Artifact root: `artifacts/trinity_nano/reference_capture/`.
- HF remote class: `AfmoeForCausalLM`, `torch_dtype=bfloat16`,
  `device_map=cpu`, `attn_implementation=eager`, 8 greedy steps, wall
  `361.7431166480019s`.
- Compatibility shims needed for current local Transformers v5.12:
  injected `bos_token_id=0`, `eos_token_id=3`, `pad_token_id=12`; installed
  `ROPE_INIT_FUNCTIONS["default"]`; no-op
  `PreTrainedModel._initialize_missing_keys`; and remote mask wrappers for
  `inputs_embeds`/`cache_position` API drift.
- Probe 0 plain short: input length 5; generated IDs
  `[8849, 43, 296, 5481, 2525, 320, 6364, 43]`; text
  `" Paris, the largest city in France,"`; step-0 top5
  `[8849, 4999, 296, 320, 290]` with logits
  `[21.5, 17.875, 17.875, 17.5, 17.375]`.
- Probe 1 chat short: input length 16; generated eight `3` / `<|im_end|>`
  tokens; step-0 top5 `[3, 72, 581, 691, 2150]`.
- Probe 2 long sliding-boundary: input length 2305 (>2048); generated IDs
  `[47, 252, 47, 13304, 252, 47, 25387, 252]`; text `"0 0 alpha 0 theta "`;
  step-0 top5 `[47, 930, 48, 1668, 478]`.
- First-probe pre-RoPE K/V captured for all 56 layers in
  `probe_00_prerope_kv_bf16_bits.npz`, stored as raw bf16 u16 bits.

Tokenizer receipts:

- Tokenizer class: `TokenizersBackend`.
- `bos_token_id=0` / `<|begin_of_text|>`, `eos_token_id=3` / `<|im_end|>`,
  `pad_token_id=12` / `<|pad|>`.
- `add_bos_token=False`, `add_eos_token=False`.
- Plain `add_special_tokens=False` for "The capital of France is":
  `[581, 4533, 323, 6364, 351]`.
- Plain `add_special_tokens=True` prepends BOS:
  `[0, 581, 4533, 323, 6364, 351]`.
- Chat template present, sha256
  `c295e73aea820982584a1f874fa71c61b1f3e6856adc6ef1d7efe339b936f2ad`.

P2 TensorCUDA build details:

- Implemented the P0 port map directly: full-attention layers skip RoPE;
  sliding layers use theta 10000 and window 2048; Q/K per-head RMSNorm;
  attention score scale `1/sqrt(128)`; fp32 softmax; sigmoid attention gate
  before `o_proj`; muP embedding scale `sqrt(1024)=32`; dual attention/MLP
  pre/post norms; untied lm_head; dense layers 0-1 SwiGLU; MoE sigmoid router
  with `expert_bias` only in top-k selection, then route_norm and route_scale
  `2.826`.
- Minimal-kernel decision: no TensorCUDA kernel changes. The NoPE path is an
  attention-branch choice in the model class; afmoe-specific gate/router logic
  is implemented in Python on existing TensorCUDA primitives.
- MoE dispatch caveat: current TensorCUDA lacks the exact HF
  `scatter_add` route accumulation primitive, so the diagnostic path evaluates
  selected experts in stable ascending expert order and accumulates weighted
  outputs token-by-token. This is likely the first place to inspect for the
  remaining bf16 parity delta.

T4 parity gate (bf16 mode):

- Receipt: `artifacts/trinity_nano/tc_parity/bf16_probe0_step1_fp32_softmax.json`.
- Result: RED. Step-0 argmax matched (`8849`), but top-5 exact failed
  (`0/1`), max_abs_diff `7.0`, mean_abs_diff `1.0362961292266846`.
- Reference top5: `[8849, 4999, 296, 320, 290]`; TensorCUDA top5:
  `[8849, 487, 290, 4999, 296]`.
- Pre-RoPE K/V compare ran against all 56 layers: max K abs diff
  `3.3359375`, max V abs diff `0.25390625`.
- Incremental decode == refeed was NOT run at 16 steps: the current
  layer-stream parity harness costs about 45s for one full step, so 16 cached
  + refeed steps would exceed the <=10 min GPU rail. The script supports it;
  the receipt records `requested_steps=0` for the bounded run.
- Lead-ledger correction: do not mark T4 green from this pass. The current
  bf16 implementation is architecturally mapped but not yet numerically
  parity-clean.

INT4 deviation receipt:

- Receipt: `artifacts/trinity_nano/tc_parity/int4_probe0_step1.json`.
- Result: expected PTQ deviation is large in current form. Step-0 argmax
  changed from reference `8849` to `373`; top5 exact `0/1`; max_abs_diff
  `15.0`, mean_abs_diff `2.554452419281006`.
- TensorCUDA INT4 generated text for the first token decode was `".\n\n"`.
- INT4 pre-RoPE K/V compare: max K abs diff `18.84375`, max V abs diff
  `1.234375`.

T3 partial residency receipt:

- Receipt: `artifacts/trinity_nano/residency/int4_all_resident_load.json`.
- All 56 layers loaded all-resident in INT4 with lm_head loaded.
- `nvidia-smi` sampled resident memory: before `194 MiB / 12282 MiB`, after
  load `3398 MiB / 12282 MiB`, after free `386 MiB / 12282 MiB`.
- Loader-estimated quantized linear storage: `3138453504` bytes
  (`2993.1 MiB`); host embedding storage: `819986432` bytes.
- Long-context prefill was not run in this pass; full 131k remains P3 work.

Verification:

- Syntax: `python3 -m py_compile core/trinity_nano_tc.py
  scripts/trinity_nano_reference_capture.py scripts/trinity_nano_tc_parity.py
  scripts/trinity_nano_int4_residency.py` passed.
- Existing TensorCUDA suite command:
  `PYTHONPATH=/mnt/ForgeRealm/Project-Tensor/tensor_cuda python3 -m pytest
  tensor_cuda/tests -q --ignore=tensor_cuda/tests/test_selector_accuracy.py`
  reported `190 passed, 1 failed in 34.37s` under GPU-visible execution.
- The failure is outside this port: `tensor_cuda/tests/test_ext_phase7.py::
  test_norms_and_conv1d` fails in `tensor_cuda/tensor_cuda/nn.py:515`
  (`GroupNorm` reshape element-count mismatch). The ignored selector script
  is not a normal pytest test; it parses `sys.argv[1]` as an integer and fails
  collection when invoked as part of the suite.

Residuals / next concrete moves:

- First debug target: isolate MoE route accumulation against the HF reference
  at layer 2 -> 3, because early layers are close and the largest K delta first
  jumps at the first full-attention/NoPE layer after MoE traffic starts.
- Second target: add a direct HF-vs-TC per-layer hidden-state capture, not only
  pre-RoPE K/V, so the first numerically divergent submodule is pinned before
  any quantization work continues.
- Do not use the ad hoc repro route-hook attempt as evidence: under the current
  remote-code/Transformers shim stack, a later one-step recapture preserved the
  argmax but did not reproduce the original full top5 ordering. The saved P1
  capture remains the T4 target, but HF CPU capture stability should be
  rechecked after the layer-level hook is made less invasive.

## 2026-07-08/09 (T4 fp32 A/B disposition — SEMANTIC PARITY PROVEN)

Order: one-run disambiguation. Question: is the remaining bf16 delta
(cross-device accumulation) or a residual semantic miss?

Hard rails held:

- No git commit. `docs/TRINITY_NANO_PORT_PLAN.md` not edited.
- Product/kernel code not touched. Harness-only edits in
  `scripts/trinity_nano_reference_capture.py` (`--torch-dtype`) and
  `scripts/trinity_nano_tc_parity.py` (`--compute-dtype` plumbs
  `BlockTC.COMPUTE_DTYPE` + `LinearTC.DTYPE`; no product edits required —
  class already exposed the knobs).
- GPU wall for the load-bearing TC run: **331.57s** (<=10 min).

### Method

1. HF reference, CPU, `torch_dtype=float32`, probe 0, 8 greedy steps,
   inv_freq reinject, eager attn. Artifact root:
   `artifacts/trinity_nano/reference_capture_fp32/`.
2. TrinityNano_TC layer-stream, `compute_dtype=float32` (weights cast to
   fp32 via LinearTC.DTYPE), weight_mode plain, probe 0, 8 steps,
   decode_check_steps=0. Receipt:
   `artifacts/trinity_nano/tc_parity/fp32_probe0_steps8_vs_hf.json`.
3. Disposition summary:
   `artifacts/trinity_nano/tc_parity/fp32_t4_disposition.json`.

### Verdict table (fp32 TC vs HF-CPU-fp32)

| step | top5_exact | max_abs_diff | mean_abs_diff | argmax |
|-----:|:----------:|-------------:|--------------:|-------:|
| 0 | True | 2.8729e-05 | 4.2536e-06 | 8849 |
| 1 | True | 2.5749e-05 | 3.3309e-06 | 45 |
| 2 | True | 6.5804e-05 | 6.6244e-06 | 671 |
| 3 | True | 2.2888e-05 | 3.2149e-06 | 351 |
| 4 | True | 2.8610e-05 | 4.0378e-06 | 4999 |
| 5 | True | 3.8147e-05 | 5.0702e-06 | 320 |
| 6 | True | 3.0518e-05 | 4.8705e-06 | 296 |
| 7 | True | 3.4809e-05 | 5.0008e-06 | 6766 |

- **top-5 exact: 8/8**
- **overall max_abs_diff: 6.5804e-05** (collapses toward ~1e-5)
- generated text: `" Paris. It is located in the north"`
- generated ids: `[8849, 45, 671, 351, 4999, 320, 296, 6766]`
  (token-for-token equal to HF fp32 reference)

### Pre-RoPE K/V max abs (step-0 prefill, 56 layers)

- max_k_abs_diff: **1.3506e-04** (worst layer 2)
- max_v_abs_diff: **1.7583e-05** (worst layer 3)
- k median ~8.3e-06; v median ~5.9e-07
- reference: `probe_00_prerope_kv_fp32.npz`

### Wall / GPU

- HF fp32 capture wall: 3.31s (CPU; model already page-cache warm)
- TC fp32 8-step wall: **331.57s** (~5.5 min)
- gpu_before: `193 / 12282 MiB`; gpu_after: `1179 / 12282 MiB`

### Registered gate disposition (T4)

**SEMANTIC PARITY PROVEN.** Under matching fp32, TrinityNano_TC
reproduces HF top-5 exactly for all 8 steps and max_abs collapses from
the bf16 residue (~1.375 vs fixed-bf16 ref) to ~6.6e-05. Per-layer
pre-RoPE K/V also collapses (bf16 was max_k 0.45 / max_v 0.037; fp32 is
1.4e-4 / 1.8e-5).

**Honest read:** the remaining bf16 cross-device delta is **accumulation
noise** (GPU bf16 vs CPU-HF bf16), not a residual semantic miss in the
port map (NoPE/sliding RoPE, dual norms, Afmoe RMSNorm cast-before-
weight, gated attention, MoE sigmoid+bias selection, route_norm×scale).
T4 is therefore dispositioned GREEN for semantic correctness; bf16
top-5 near-ties remain an expected numeric residue, not a logic bug.

### Scope residuals (not claimed by this receipt)

- Incremental-decode==refeed not re-run in fp32 (decode_check_steps=0
  under GPU rail; layer-stream cost ~41s/step).
- Only probe 0 (5-token plain short). Chat and long-sliding probes not
  re-gated in fp32 this order.
- INT4 path still expected-deviating (prior receipt); not re-tested.
- Plan predictions T1–T3 remain out of this order.

Harness flags added (scripts only):

- `trinity_nano_reference_capture.py --torch-dtype {bfloat16,float32}`
  + fp32 K/V npz writer.
- `trinity_nano_tc_parity.py --compute-dtype {bfloat16,float32}`
  + dtype-matched K/V compare.

## 2026-07-08/09 (P3 envelope — CACHE-PATH RED; T3/T2 not run)

Order: P3 of Trinity Nano port (T3 residency + T2 APA + cache-path
logic). Hard rails: no git commit; plan untouched; GPU ≤10 min/run;
harness `scripts/trinity_nano_*`; product only for APA dialect hook.

### Files added/changed

- `scripts/trinity_nano_p3_envelope.py` (new): cache-path / T3 staged
  residency / T2 APA zero-flip harness.
- `core/trinity_nano_tc.py` (**PRODUCT — APA dialect hook, flagged**):
  - `TrinityAttentionTC`: `attention_mode` / `refine_percentile` /
    `bulk_bits` / `apa_min_context`; APA path via
    `tc.apa_selective_attention` on full/NoPE layers only when
    `S > apa_min_context` (sliding always STANDARD).
  - `TrinityNano_TC.set_attention_mode(..., full_only=True)` selective
    engagement of the 14 full-attention layers.
  - `configure_moe_empty_cache` harness speed knob.
- Plan `docs/TRINITY_NANO_PORT_PLAN.md` not edited.

### Task 1 — CACHE-PATH LOGIC: RED (stop rail)

Receipt: `artifacts/trinity_nano/p3_envelope/cache_path_20260708_222031.json`

| step | cached_tok | refeed_tok | max_abs_logit_diff |
|-----:|-----------:|-----------:|-------------------:|
| 0 | 653 | 653 | 0.0 |
| 1 | 114 | 55500 | 7.40625 |

- `token_for_token_equal`: **false**
- `first_mismatch_step`: **1**
- `completed_steps`: 2 / 16 (broke on first mismatch)
- prompt_len=16, weight_mode=int4 resident, wall 2.54s
- GPU: before 3397 MiB, after 3417 MiB / 12282

**Diagnosis receipt:**
`artifacts/trinity_nano/p3_envelope/cache_path_diagnosis_20260708.json`

Root cause class (not a mask/offset wiring bug):

1. step-0 exact (prefill path deterministic).
2. Layer-0 `k_proj` outputs for identical prefix rows differ when the
   INT4 linear is invoked at L=16 vs L=17 (`k_proj` prefix maxabs
   0.015625; after k_norm 0.25). Embed + LN prefix exact 0.0.
3. QuantLinearTC INT4 row instability confirmed fused and unfused
   (prefix maxabs ~0.0039 on a random probe).
4. Divergence compounds across all 56 layers → final logits maxabs
   7.40625, argmax 114 vs 55500.
5. Wiring that still looks correct: `position_offset` decode, bottom-
   right causal for L=1+S, RoPE only on sliding, NoPE full, window
   keep, returned caches used.

### Tasks 2–3 — NOT RUN (order stop after cache mismatch)

- T3 staged residency (8k/32k/96k/131072): **not started**.
- T2 APA zero-flip at 8k: **not started** (APA dialect hook is in
  tree but ungated).

### Timing micro-receipts (pre-gate, for rail projection only)

With `empty_cache_interval=0` on INT4 resident:

| L | prefill wall | ms/tok |
|--:|-------------:|-------:|
| 16 | 0.89s | ~56 |
| 128 | 5.09s | ~40 |
| 256 | 10.05s | ~39 |
| 512 | 20.25s | ~40 |
| chunked 1024 | 40.09s | ~39 |

Projection (not a T3 receipt): 8k ≈ 5.5 min (under rail); 32k ≈ 22 min
(over 10-min rail). Weight resident load ~3.4GB / 3398 MiB prior.

### Verdicts vs plan predictions

| pred | status |
|------|--------|
| T4 | GREEN (prior fp32 disposition; unchanged) |
| cache-path (P2 gate residue) | **RED** under INT4-resident refeed |
| T3 full-context residency | **not tested** this order |
| T2 APA selective engagement | **not tested** this order (hook present) |

### Residuals / successors

- Not claimed fixed: cache-path. Numeric INT4 row instability across
  seq lens is the load-bearing cause of the observed flip.
- Successor options (operator decision): (a) kernel row-stable INT4
  GEMM; (b) gate variant that compares cache-decode against a second
  decode that reuses the same prefilled KV (same numeric path); (c)
  length-stable compute path for the logic check.
- T3/T2 remain queued after cache-path disposition.
- fp32-compute cache recheck blocked mid-probe by
  `rope_apply: table/x dtype mismatch` when COMPUTE_DTYPE=float32
  against bf16 rope tables — separate harness fix if pursued.

## 2026-07-08/09 (cache-path dtype seam + fp32-compute disposition)

Order: fix rope table/x dtype seam; re-run cache-path under higher-
precision compute; if MATCH, INT4 severity (per-step logit gaps).
Hard rails: no git commit; plan untouched; GPU ≤10 min/run; product
only for cast-compatibility.

### Files changed

- `core/trinity_nano_tc.py` (**PRODUCT — cast-compat, flagged**):
  `TrinityAttentionTC`: cast `cos`/`sin` to `q.dtype` before
  `rope_apply` / `apply_rotary` when table dtype ≠ activation dtype.
- `scripts/trinity_nano_p3_envelope.py`:
  - `--compute-dtype {bfloat16,float32}` → post-load
    `fp32_compute_int4_dequant` (INT4 weights stay packed; activations/
    KV/rope/router LinearTC cast to fp32).
  - rope rebuild under new COMPUTE_DTYPE; LinearTC router_gate cast.
  - `--task cache_severity`: full 16-step INT4 run, no early break,
    per-step top1−top2 margins + cross-arm logit gaps.
  - mismatch dump: position_offset, input lens, KV lengths both arms.

Mode chosen: **fp32 COMPUTE + INT4-dequant resident** (not bf16 layer-
stream). Weights ~3.4 GB INT4; fp32 full weights would be ~25.6 GB
(over VRAM). Peak used ~3441 / 12282 MiB at 16-tok.

### Gate A — fp32 compute cache-path: GREEN (logic PROVEN)

Receipt: `artifacts/trinity_nano/p3_envelope/cache_path_fp32_20260708_223055.json`

| metric | value |
|--------|------:|
| mode | fp32_compute_int4_dequant |
| token_for_token_equal | **true** (16/16) |
| max_abs_logit_diff (worst step) | **2.69e-4** |
| first_mismatch_step | null |
| wall_s | 16.15 |
| min top1−top2 margin (cache) | 0.0233 |
| median margin | 0.369 |

Step-0 maxabs 0.0; residual cross-arm logit noise ≤ 2.7e-4 never
flips argmax.

**Disposition:** cache wiring PROVEN under higher-precision compute.
Prior INT4 mismatch is **length-instability numerics**, not a cache
logic bug.

### Gate B — INT4 severity (same seed/prompt, bf16 compute): RED tokens

Receipt: `artifacts/trinity_nano/p3_envelope/cache_severity_20260708_223141.json`

| metric | value |
|--------|------:|
| token_for_token_equal | false |
| completed_steps | 16 (no early break) |
| n_flips | **12 / 16** |
| first_mismatch_step | **1** (cached 114 vs refeed 55500) |
| max_abs_logit_diff | **7.78** |
| wall_s | 19.68 |

Per-step severity (selected):

| step | c_tok | r_tok | match | maxabs | c_margin | r_margin | cross_gap_c |
|-----:|------:|------:|:-----:|-------:|---------:|---------:|------------:|
| 0 | 653 | 653 | Y | 0.00 | 1.19 | 1.19 | 0.00 |
| 1 | 114 | 55500 | N | 7.41 | 1.25 | 0.13 | **4.00** |
| 4 | 25537 | 80040 | N | 5.73 | 0.06 | 0.88 | 2.00 |
| 7 | 4064 | 4064 | Y | 6.19 | 1.56 | 0.06 | 0.00 |
| 9 | 4064 | 39 | N | 4.94 | 0.06 | 0.31 | 0.63 |

KV lengths both arms at mismatch: all layers S=17 (cache-length
aligned; not an off-by-one wiring dump).

**Severity read (evidence: receipt):** not knife-edge near-ties on a
shared surface. Arms diverge by multi-logit maxabs (≈4–8) after step-0;
individual-arm top1−top2 margins can be small (0.0625) *and* cross-arm
preference gaps large (step-1 cross_gap=4.0). Class: **severe INT4
seq-len GEMM instability**, not soft-tie noise.

### Verdict table

| claim | status | evidence class |
|-------|--------|----------------|
| rope dtype seam | fixed (product cast + harness rebuild) | unit-level: run completed past prior crash |
| cache-path logic | **GREEN / PROVEN** | e2e gate, fp32 compute, 16-tok greedy |
| INT4 cache vs refeed | RED tokens (expected under numerics) | e2e severity receipt |
| T3 / T2 | still not run | out of this order |

### Residuals

- INT4 decode token identity still unstable across refeed lens; not
  claimed fixed.
- Kernel row-stable INT4 (or same-path refeed baseline) remains the
  product successor for usable INT4 incremental decode.
- T3 staged residency + T2 APA still queued.
- Plan `docs/TRINITY_NANO_PORT_PLAN.md` not edited.

## 2026-07-08/09 (T3 residency curve + T2 APA — INT4 + fp32 compute)

Order: T3 staged residency + T2 APA receipts under the cache-path
disposition mode (**INT4 weights + fp32 COMPUTE**; bf16-compute
quarantined). Hard rails: no git commit; plan untouched; GPU ≤10 min
per stage (project wall from prior stage before ascending); harness
`scripts/trinity_nano_*` only.

### Files changed (harness only)

- `scripts/trinity_nano_p3_envelope.py`:
  - T3 v2: `compute_dtype`/`mode_label` on receipt; projection math
    ledgered before each stage; peak VRAM + single-point theoretical
    KV-delta extrapolation to 131k; split verdict string
    `VRAM-CONFIRMED-BY-EXTRAPOLATION at S=<last>, wall-blocked`.
  - T2 v2: per-layer backend map after prefill; engagement assert
    (exactly 14 full `apa_selective_full_nope`, 0 sliding leaks);
    flip_details with per-step logit gaps (fp32 meaningful); no knob
    tuning on flip.
- Plan `docs/TRINITY_NANO_PORT_PLAN.md` **not edited**.

### Mode (binding)

| field | value |
|-------|-------|
| weight_mode | int4 resident (~3.14 GB quantized linear bytes) |
| compute | float32 (`fp32_compute_int4_dequant`) |
| KV storage | follows COMPUTE_DTYPE → **4-byte** under this mode |
| plan T3 assumed | fp16/bf16 (2-byte) KV — both footprints ledgered |
| GPU | RTX 4070 SUPER 12282 MiB |

---

### T3 — staged residency curve

Receipt: `artifacts/trinity_nano/p3_envelope/t3_residency_20260708_225457.json`
Summary: `artifacts/trinity_nano/p3_envelope/summary_20260708_225457.json`

**Calibration** (L=512, used for first-stage projection only):

| metric | value |
|--------|------:|
| wall_s | 16.55 |
| ms/tok | 32.321 |
| peak_used_mib | 3495 |

**Projection math (rail = 600 s):**

1. Before S=8192 (from calibration):
   `projected = (32.321/1000)*8192 = 264.77 s` → under rail → **RUN**
2. After S=8192 measured ms/tok=35.826; before S=32768:
   `projected = (35.826/1000)*32768 = 1173.93 s` → **STOP** (over rail)
3. Remaining stages inherit stop:
   - S=98304: `(35.826/1000)*98304 = 3521.80 s` skipped_after_rail_stop
   - S=131072: `(35.826/1000)*131072 = 4695.74 s` skipped_after_rail_stop

**Curve table** (evidence: e2e staged prefill, INT4+fp32):

| S | status | projected_wall_s | measured_wall_s | ms/tok | tok/s | peak_used_mib (nvidia-smi) |
|--:|--------|-----------------:|----------------:|-------:|------:|---------------------------:|
| 8192 | **ran** | 264.77 | **293.48** | 35.826 | 27.91 | **4044** |
| 32768 | skipped_projected_over_rail | 1173.93 | — | — | — | — |
| 98304 | skipped_after_rail_stop | 3521.80 | — | — | — | — |
| 131072 | skipped_after_rail_stop | 4695.74 | — | — | — | — |

**VRAM extrapolation to 131k** (single measured point + theoretical KV Δ):

- Method: `peak(S) ≈ peak(8192) + (kv_theory_fp32(S) − kv_theory_fp32(8192))`
- peak(8192)=4044; kv_fp32(8192)=392 MiB; residual_non_kv=3652 MiB
- kv_fp32(131072)=3752 MiB → **peak_131k_extrap = 7404 MiB ≤ 12282** (fits)
- fp16-equivalent: residual + kv_fp16(131072)=3652+1876 → **5528 MiB ≤ 12282** (fits)
- wall_131k_extrap ≈ 4696 s (~78 min) — Python expert-loop wall; not claimed measured

| claim | value |
|-------|-------|
| t3_full_131k_confirmed (measured prefill) | **false** |
| ceiling_S_reached | **8192** |
| **t3_verdict** | **VRAM-CONFIRMED-BY-EXTRAPOLATION at S=8192, wall-blocked** |
| vs plan T3 | **SPLIT** — memory premise holds by extrapolation; full-context prefill wall not closed under 10-min rail |

Successor for wall: grouped-GEMM / faster prefill path (out of this order).

---

### T2 — APA selective engagement @ S=8192

Receipt: `artifacts/trinity_nano/p3_envelope/t2_apa_20260708_230041.json`
Summary: `artifacts/trinity_nano/p3_envelope/summary_20260708_230041.json`

Mode: same INT4+fp32; `full_only=True`; `apa_min_context=2048` (default);
greedy 16 tokens; STANDARD then APA (same seed/prompt).

**Engagement (per-layer after APA prefill):**

| class | n | backend | indices |
|-------|--:|---------|---------|
| full / NoPE | **14/14** | `apa_selective_full_nope` | 3,7,11,15,19,23,27,31,35,39,43,47,51,55 |
| sliding | **42/42** | `standard_sliding_band` | (held standard) |
| APA leak on sliding | **0** | — | [] |
| full non-APA | **0** | — | [] |

`apa_fired_exactly_14_full_not_sliding`: **true**

STANDARD histogram: `standard_full_nope`×14 + `standard_sliding_band`×42.

**Zero-flip / walls:**

| metric | STANDARD | APA | delta |
|--------|---------:|----:|------:|
| prefill_wall_s | 296.19 | 296.55 | +0.36 |
| total_wall_s (prefill+16 decode) | 299.82 | 300.06 | **+0.23** |
| prefill_peak_mib | 3985 | 3857 | — |
| mean_nll (ppl proxy) | 3.2257 | 3.2187 | −0.0070 |
| tokens equal 16/16 | — | — | **false** |
| n_flips | — | — | **13 / 16** |
| first_flip_step | — | — | **3** (steps 0–2 match) |

**Token sequence:**

- std: `[1473, 46, 1197, 434, 3809, 2615, 45133, 33378, 8785, 10378, 443, 352, 173501, 53479, 1280, 1184]`
- apa: `[1473, 46, 1197, 1187, 1595, 2150, 55223, 320, 83177, 1280, 1432, 6414, 326, 6573, 50411, 40740]`

**Flip logit gaps (fp32; selected — full table in receipt `flip_details`):**

| step | std_tok | apa_tok | std_gap (pref−other) | apa_gap | cross maxabs | std top1−top2 |
|-----:|--------:|--------:|---------------------:|--------:|-------------:|--------------:|
| 3 | 434 | 1187 | 0.270 | 0.586 | 2.29 | **0.018** (near-tie on std) |
| 4 | 3809 | 1595 | 1.46 | 5.62 | 6.64 | 0.219 |
| 5 | 2615 | 2150 | 1.53 | 2.26 | 9.80 | 0.190 |
| 7 | 33378 | 320 | 11.98 | 5.91 | 13.52 | 0.785 |
| 8 | 8785 | 83177 | 9.12 | 14.51 | 19.50 | 0.487 |
| 14 | 1280 | 50411 | 13.71 | 8.69 | 18.54 | 0.615 |

Class (evidence: e2e fp32 arms): **not pure soft-tie noise**. Step-3 is near-tie on STANDARD (margin 0.018) with modest cross maxabs 2.29; later steps show multi-logit cross-arm divergence (maxabs 6–20) after the trajectories fork. **FINDING about APA-on-NoPE (kv=2, head_dim=128, full layers)** — not treated as automatic gate failure; knobs not tuned.

| claim | status | evidence class |
|-------|--------|----------------|
| APA engages on exactly 14 full layers | **GREEN** | per-layer backend receipt |
| sliding held STANDARD | **GREEN** | 0 APA leaks |
| zero-flip STANDARD vs APA @ 8k/16 | **RED tokens** (13 flips) | e2e greedy; flip_details |
| vs plan T2 | **SPLIT** — engagement premise holds; token-identity gate does not |

status field on receipt: `token_flip_finding`

---

### Verdict table (this order)

| pred | status | note |
|------|--------|------|
| T3 full-context residency | **SPLIT** | VRAM-CONFIRMED-BY-EXTRAPOLATION @ S=8192; wall-blocked above 8k |
| T2 APA eligibility | **SPLIT** | engagement clean; zero-flip fails as APA-on-NoPE finding |
| cache-path (prior) | GREEN under fp32 | unchanged |
| plan file | untouched | immutable |

### Residuals / successors

- T3 32k/96k/131k **not measured** — wall rail; VRAM fit is extrapolation
  (single-point + theoretical KV), not multi-stage linear fit.
- T3 measured under **fp32 KV (4-byte)**; plan text said fp16 KV. Both
  extrap peaks fit 12282; fp16-equivalent peak ~5528 MiB.
- Grouped-GEMM / faster prefill owns the T3 wall residual.
- T2 flip is a finding, not claimed fixed; no APA percentile/bits retune.
- INT4 bf16-compute remains quarantined (not used).
- P4 NoPE-graft (GraftRepository) still separate order.

## 2026-07-09 (T1 NoPE-graft — RELATIVE result banked, absolute BLOCKED on gen floor)

Action: T1 width sweep + template/compute-mode disambiguation
(Grok, GraftRepository; scripts/trinity_nope_graft_width_sweep.py +
trinity_t1_floor_reverb.py; artifacts/trinity_nope_graft/).

RELATIVE FINDING (banked, robust): NO GPT-OSS-style width-triggered
degradation transition on Trinity's NoPE full layers. Generation is
BIT-IDENTICAL across live_shift 0 / 117 / 789 (bf16 stream) — where
GPT-OSS collapsed clean→salad between shift 115 and 387. The arena's
mixed NoPE/RoPE contract wired cleanly (route hit, mount seated,
driver adapter absorbed inject_kv/live_shift/rope-skip-on-NoPE with
NO product edits). Consistent with T1's direction: NoPE sites show no
positional hole cliff.

ABSOLUTE T1 NOT CONFIRMED — blocked below the finding, honestly:
- The port has NO clean-English free-gen floor in either failing mode.
  INT4+fp32 free-gen = bos_loop; bf16-stream chat = "1234" degraded.
  Treatment proof it is a PORT residual not the arena: same NATURAL
  prompt generates clean English under bf16 ("Paris. It is located
  in the north-central...") — so the engine/adapter are sound; the
  chat/free-gen path is the floor. HF REFERENCE ALSO loops on chat
  free-gen (EOS loop) — Trinity's chat usability is thin upstream,
  not just in our port.
- Template ruled OUT as the cause (real captured template, still
  looped) — the GPT-OSS "template ghost" did not recur here.
- Value-recovery unclassable: 12GB forces layer-stream (no resident
  graft seats), so the readout half of T1 can't be exercised in a
  bounded run.

VERDICT: T1 RELATIVE-CONFIRMED (no width cliff), T1 ABSOLUTE-BLOCKED.
Successors required before absolute close: (a) row-stable INT4 GEMM
(quarantine fix — unlocks resident clean free-gen); (b) a working
Trinity chat/generation recipe (thin even in HF ref — investigate
sampler/stop config, or use natural-continuation probes instead of
chat-format probes for the graft readout). Both are Project-Tensor
engine/recipe work, not GRM-arena work.
