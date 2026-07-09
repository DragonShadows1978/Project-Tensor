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
