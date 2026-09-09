# APA-SP3 — The model test: single-pass APA inside MiniCPM3-4B, perplexity against standard and two-pass, plus the empirical tail margin

David, 2026-09-06: "if the SP algorithm doesn't actually allow for low
precision then it's useless." SP1/SP1.1/SP2/SPD1 were kernel sweeps;
they establish speed, memory shape and deviation vs dense, and NOTHING
about model quality. This order is the model-quality test (evidence
class: model perplexity) and the empirical replacement for SP2's
worst-case margin, on the model that already has the best-instrumented
APA receipts on this engine: MiniCPM3-4B (MLA, composite 96-dim keys,
ppl@1024 last-512: standard 20.065, APA two-pass bulk-4 refine-0.10
19.817, ground truth 17.357; APA context ceiling = the full 32,768
window). Vehicle chosen by David over Trinity Nano.

YOUR WRITABLE TARGET is the worktree you were dispatched into
(`/mnt/ForgeRealm/Project-Tensor-wt-apa-sp3`, branch `apa-sp3` forked
from `main`). Writable: `scripts/apa_sp3_*`, `tensor_cuda/tests/`,
`artifacts/apa_sp3/`, `logs/`, `docs/APA_SP3_LEDGER.md`, and — ONLY if
the model path needs it — a minimal, hash-pinned, flag-guarded addition
to the SP launcher (`TC_APA_SP` stays default OFF; every pre-existing
kernel body byte-identical). READ-ONLY: everything else in this repo;
`/mnt/ForgeRealm/GraftRepository` (the MiniCPM3 adapter
`core/minicpm3_tc.py` and `docs/MiniCPM3-MLA_Results.md`);
`/mnt/ForgeRealm/AI-AtlasForge/workspace/APA-Quant-Rust_LLM_testing/mission_b74b7906/`
(the harness that produced the 20.065 / 19.817 receipts —
`test_graft_e1_mla.py` and its neighbours); the HF cache
(`~/.cache/huggingface/hub/models--openbmb--MiniCPM3-4B`). Import or
COPY what you need from the read-only trees into `scripts/apa_sp3_*`;
never edit them. Your sandbox has NO GPU: build, CPU-verify plumbing,
deliver leased bounded runner + blocked-report; the lead runs the card
(RTX 4070 SUPER, 12 GB).

## Context (read first)

1. `docs/APA_SP1_LEDGER.md`, `docs/APA_SP1_1_REPORT.md`,
   `docs/APA_SP2_DELTA_DERIVATION.md`, `artifacts/apa_sp2/EPSILON_CURVE.md`,
   `artifacts/apa_spd1/SPEED_CHAIN.md` — the kernel-sweep story so far:
   the SP rule `refine_j iff bulk_j ≥ max(bulk[0..j]) − δ` is exact and
   monotone; at SP1's grid-matched δ it refines the same ~15% as the
   two-pass z-score rule, faster and closer to dense; the PROVABLE δ
   (`ln(1/ε)+2·e_q`) refines ~100% at 2/4-bit bulk because the
   worst-case e_q (max |bulk−exact| on synthetic keys) is 1–2.3 logit
   units while the mean is 0.08–0.27.
2. The MiniCPM3 adapter: `attention_mode == "apa_selective"` builds
   composite 96-dim keys (nope 64 + rope 32), pads V to 96, quantizes
   `kq` at `bulk_bits` (doc receipts at 4; adapter default 8), and calls
   `tc.apa_selective_attention(q_full, k_full, kq, vpad, scale, z, causal)`
   with `z = norm_ppf(1 − refine_percentile)`. The SP entry
   `tc._C.apa_selective_attention_sp(q, k, kq, v, sinks, …, delta, causal)`
   has the same tensor shape contract; the cap ladder (≤64/≤128/≤256)
   covers D=96 — VERIFY and pin with a test, prefill AND split-K.
3. House rules: `/mnt/Shared/HOUSE_RULES.md` (Prior Art Directive §,
   seat laws §8/§9); `docs/KERNEL_OPT_IMPLEMENTATION_PLAN.md` evidence
   classes.

## Mission

**G0 — parity first.** Reproduce the published standard ppl@1024
last-512 (20.065) and two-pass bulk-4 refine-0.10 (19.817) IN-PROCESS
with your harness before any SP arm runs, same tokens, same INT4 weight
path, same protocol as `docs/MiniCPM3-MLA_Results.md` (state the
dataset/prefix source you found in the read-only harness; if the
protocol cannot be reproduced to ±0.01 say so and stop the ppl arms —
that is a RED, not a tolerance to widen).

**G1 — the arms** (all ppl@1024 last-512 on IDENTICAL tokens; bulk 4
primary, bulk 8 secondary if wall permits):
- A `standard` (reference).
- B `apa_two_pass` refine 0.10 (the 19.817 row).
- C `apa_sp_matched`: SP with ONE global δ chosen so the realised
  refine fraction over all layers matches B's realised fraction ±0.01
  (measure B's actual fraction; it is not exactly 0.10). Report the
  per-layer fraction spread.
- D `apa_sp_refine_all`: δ large enough to refine every key. Must
  equal A to fp noise; this is the in-model correctness pin of the SP
  kernel and the SP2 "bonus" measured on a model.
- E `apa_sp_provable`: SP2's δ at ε=1e-3 with e_q taken from THIS
  model's real keys (G2 below), bulk 4. Report its realised fraction.
- Also per arm at S=8192 (the T2-style long prefill, prefill-only,
  identical tokens): ms wall, peak resident, and the last-512 ppl if
  the wall permits (state it if it does not).

**G2 — the empirical margin** (this replaces SP2's worst-case bound
with measurements on real activations): during arm B/C prefill at
S=1024 and S=8192, per layer: (i) |bulk_j − s_j| over all keys: mean,
p99, p99.9, max — the real e_q; (ii) under arm C's δ, the softmax mass
carried by UNREFINED keys per query (mean, p99, max) and the maximum
relative weight `w_j / w*` of any skipped key (the ε the heuristic
actually achieves); (iii) the same two numbers for B's z-score
selection. Deliver as JSON + one table + a two-paragraph reading:
does the heuristic tail skip keys that matter by weight, and does ppl
care.

**G3 — decode speed in-model** (SPD1 said SP wins long decode):
tokens/s at S∈{2048, 8192, 32768} for A, B, C (split-K SP), same
prompt, ≥ 32 decode steps, CUDA-event or wall with sync, stated.

## Registered predictions (lead) — register yours beside them BEFORE the card runs

- P1: D equals A within 0.005 ppl.
- P2: C is within ±0.05 ppl of B (the heuristic tail costs no model
  quality at matched fraction).
- P3: E refines ≥ 0.95 of keys at bulk 4 (real-key max e_q is still
  ≥ 0.8 logit units) and its ppl equals A within 0.01.
- P4: real-key p99 |bulk−exact| at bulk 4 is < 0.5 of the real-key max
  (the worst case is outliers); the max relative weight of a skipped
  key under C exceeds 0.1 on at least one layer while ppl is unmoved.
- P5: SP decode at S=32768 is ≥ 2× two-pass tokens/s in-model.

## Gates & delivery

Registration first: `artifacts/apa_sp3/registration.json` (IMMUTABLE;
amendments in separate files) citing SP1/SP1.1/SP2 registration shas,
the protocol source, the token sha, the δ selection rule for C, the
e_q statistic for E. CPU gates (you): shape/contract tests for D=96
on both SP kernels, harness plumbing, `--dry-run` enumerating every
cell, hash pins on every pre-existing kernel body. GPU (lead): deliver
`scripts/apa_sp3_lead_gpu.sh` with `list|run|resume|summary`, flock on
`/tmp/forge-gpu.lock` (`--wait`), every job ≤ 590 s (split arms into
jobs; MiniCPM3-4B INT4 is ~2.9 GB resident, leave the rest for the
S=8192/32768 rows and say what does not fit), cooldown between jobs,
never kills anything, exact commands file. `artifacts/apa_sp3/RESULTS.md`
writer renders the ppl table, the G2 margin table, the decode table,
each with its evidence class stated ("model perplexity" / "kernel
sweep") and the sentence that G2/G3 rows establish nothing about model
quality by themselves.

## Rules (binding)

NO git. NO subagents. NO background waits (foreground, explicit
`timeout`, every call < 10 min). Never kill a process you did not
start; other projects share this machine and the GPU; the operator has
absolute right of way. Registration immutable. Existing kernel bodies
byte-identical. RED honesty: a parity miss at G0, a D≠A, or an arm that
cannot fit is a result; report it, do not tune around it. Ledger as it
happens in `docs/APA_SP3_LEDGER.md`.

## Done (verbatim in your final message)

1. G0 parity numbers (reproduced / not) with the protocol source path.
2. Contender/arm registry, δ rule for C, e_q rule for E; D=96 pin
   test names.
3. Registration sha + predictions (lead's and yours).
4. CPU gate results; GPU blocked-report with exact lead commands and
   the per-job wall estimate.
5. Prior art: what each arm/measurement takes from published work
   (BLASST, ThriftAttention, FlashAttention-2, TurboQuant, the APA
   paper draft) versus what is yours, annotated at code site + ledger,
   or "none known".
6. Files; deviations; residual risks; RED; process safety; model id
   and reasoning effort.
