# APA-SPD1 — The speed chain on identical shapes: dense engine → SDPA/flash → APA two-pass → APA single-pass

David, 2026-09-06: "scour the various metrics and see if we can figure
out speed from APA to APA-SP to standard/flash attention." The lead
scoured; the receipts do not chain, because no two of them share a
shape:

- APAMQ-E1 (`artifacts/apamq_e1/RESULTS.md`, D=128, GPT-OSS geometry):
  engine `standard` (dense) vs `fused_apa`/`int4_apa` — prefill L=512
  over S=4K–64K: standard 1.0–54 ms vs fused_apa 18–314 ms (APA 6–17×
  SLOWER, 2 GiB → 2 MiB); decode L=1: standard 0.14–1.6 ms vs fused_apa
  0.57–8.0 ms (3–6× slower). Memory win, speed loss.
- APA-SP1/SP1.1 (`artifacts/apa_sp1/gpu/`, B=1, H=4/8, D=64/128,
  L=S prefill and L=1 decode): APA-SP vs APA two-pass — prefill
  1.09–1.90×, split-K decode 0.88–5.3×. No dense row.
- `docs/APA_PAPER_DRAFT.md` §4.5: an OLD prototype bench (B=2, H=4,
  D=64, r=0.15) vs SDPA — APA slower below ~1024 tokens, 2.1× faster
  at 2048. Different kernel generation; not comparable to either
  receipt above.

This order measures all four on the SAME registered shapes, same
process, same seeded inputs, so the chain is one table. Kernel-sweep
evidence only.

YOUR WRITABLE TARGET is the worktree you were dispatched into
(`/mnt/ForgeRealm/Project-Tensor-wt-apa-spd1`, branch `apa-spd1`
forked from `main`). Your sandbox has NO GPU: build the bench,
CPU-verify its plumbing, deliver a leased bounded runner + blocked-
report; the lead runs the card. Production code READ-ONLY (this is a
benchmark, not a change); `scripts/apa_spd1_*`, `tensor_cuda/tests/`,
`artifacts/apa_spd1/`, `logs/`, `docs/APA_SPD1_LEDGER.md` writable.

## Contenders (all on the same (q, k, v) tensors, same dtype family)

1. **engine dense** — tensor_cuda's own standard attention path (the
   `standard` path E1 measured; name the entry point). Exact reference
   for accuracy.
2. **torch SDPA** — `torch.nn.functional.scaled_dot_product_attention`
   (torch 2.11+cu130 is installed on the host) with each backend the
   host supports, forced one at a time via `sdpa_kernel`: math,
   efficient, flash (report which are available on sm_89; if flash is
   unavailable say so). This is the "standard/flash attention" row.
   Also `flash_attn` if importable (state version) — optional.
3. **APA two-pass** — `apa_selective_kernel` family at its registered
   refine percentile (the E1/SP1 settings), TurboQuant bulk bits as
   registered.
4. **APA single-pass** — `apa_selective_sp_kernel` (prefill) and the
   split-K SP kernel (decode) via `TC_APA_SP=1`, at the δ that
   matches the two-pass refine fraction (SP1's registered per-shape δ)
   AND, as a second row, at ε=1e-3 through the SP2 launcher if the
   frozen e_q table exists in the main checkout by run time
   (`artifacts/apa_sp2/…FROZEN…`); if not, say so and skip that row.

## Shapes (registered)

Exactly SP1's grid so every row lines up with existing receipts:
prefill L=S∈{512, 2048, 8192}, D∈{64,128}, causal∈{0,1}, GQA
(H=4,KV=4) and (H=8,KV=2); decode L=1, S∈{2048, 8192, 32768}, same D
and GQA. B=1. Plus ONE E1-comparable cell per family (L=512 prefill at
S=8192 and S=32768, D=128, KV=4) so the E1 numbers can be cross-read.

## Measurement

Per cell per contender: median and IQR of ≥ 7 timed calls after
warm-up, CUDA-event timed, single stream, `torch.cuda.synchronize`
where torch is involved; peak allocated memory delta; output relative
Frobenius and max-abs vs the engine dense reference (fp32). Same
inputs for every contender (seeded, generated once per cell, converted
to each contender's dtype — state the dtype per contender; if a
contender only runs fp16/bf16 while another runs fp32, say so in the
table header). One leased job per cell (≤ 10 min), `list|run|resume|
summary`, cooldown between jobs, never kills anything.

## Registered predictions (lead)

- P1: engine dense and torch SDPA-math agree on output to fp32
  tolerance on every cell; SDPA-flash/efficient are faster than engine
  dense by ≥ 2× on prefill S≥2048.
- P2: APA two-pass is SLOWER than engine dense on every prefill cell
  at S ≤ 8192 and on decode S ≤ 8192; the crossover, if any, is at
  decode S=32768.
- P3: APA-SP closes ≥ half the gap between APA two-pass and engine
  dense on prefill S=8192, and is within 2× of engine dense on decode
  S=32768; it does not beat SDPA-flash anywhere.
- P4: memory: APA variants ≤ 1/50 of dense's peak transient on
  prefill S=8192.
Register your own beside these before running.

## Gates

G1 (CPU, you): bench plumbing tests (shape grid, dtype handling,
contender registry, table assembly, accuracy comparator), a
`--dry-run` that enumerates every cell without a device. G2 (GPU,
lead): the table. Deliver `artifacts/apa_spd1/SPEED_CHAIN.md` writer
that renders the final table from receipts with the sentence "kernel
sweep; this establishes nothing about model quality" and a reading of
where each contender wins and loses, plus one paragraph reconciling
with E1 and the paper §4.5 numbers where shapes allow.

## Rules (binding)

Prior Art Directive applies (HOUSE_RULES): cite what each contender
is (FlashAttention 2, PyTorch SDPA backends, TurboQuant, APA) in the
bench header. NO git. NO subagents. NO background waits (foreground,
explicit `timeout`, every call < 10 min). Never kill a process you did
not start. Registration immutable; amendments separate. RED honesty.

## Done (verbatim)

1. Contender registry (entry points, dtypes, availability on this
   host as far as CPU checks can tell); prior-art citations.
2. Registration sha + predictions (lead's and yours).
3. G1 results; G2 blocked-report with exact lead commands.
4. Files; deviations; risks; RED; process safety; model id and effort.
