# APA-SP3 amendment 6 (lead, 2026-09-07) — reconcile in-model decode speed with the June receipt

The SP3 chain is complete (branch `apa-sp3` fa751e4; you are in a fresh
worktree `/mnt/ForgeRealm/Project-Tensor-wt-apa-sp3-a6`, branch
`apa-sp3-a6`, forked from it). Every other question is answered. One
column is not usable: **decode throughput**.

`decode_pool_b4_A_2048` (alloc pool ON, 32 synced steps) reads 4.98
tokens/s for STANDARD; B 3.28; C 2.58; both APA arms 0.90 at S=8192.
The June receipt for THIS model on THIS engine
(`GraftRepository/docs/MiniCPM3-MLA_Results.md`, "DECODE speed pass",
engine 8501a5c) is 21.6 ms/token = 46 tokens/s on the 8 GB 3070 with
`tc.set_alloc_pooling(True)`. A 4070 SUPER reading ten times slower
than a 3070 is a harness artifact, not a measurement. Candidates, in
the order I would check them: (1) the per-layer diagnostic wrapper you
install around attention (`owner.original_attn` hook) and whatever it
does per step; (2) `lg.float().numpy()` + `np.isfinite` on the full
73,448-vocab logits every step (a host round trip per token); (3)
`last_token_only` / cache handling forcing recompute; (4) pooling
engaged after weight load vs before (your a5 deviation note); (5) the
LD_PRELOAD cudaMalloc interposer still active in `decode_pool`.

## Mission

1. Reproduce the June decode path exactly as the results doc describes
   it (same adapter, `set_alloc_pooling(True)` before load, no wrappers,
   no interposer, no per-step host copies beyond the argmax) on the
   pinned stream at S=2048 and report ms/token for STANDARD. If it is
   within 2× of 21.6 ms/token, the harness is the problem; if it is
   not, the engine on this card is, and that is a finding.
2. Bisect the gap: toggle each candidate above one at a time (register
   the ladder first) and report the ms/token at each rung. One leased
   job per rung, ≤ 290 s worker rail.
3. Register `decode_clean_b{bits}_{arm}_{S}` cells (A/B/C at 2048, 8192,
   and 32768 if the rail allows; note 32768 prefill alone exceeds the
   rail on this engine, so state it as a non-fit if so) using the
   cleanest configuration that still records tokens/s and the pool's
   peak; P5 (SP ≥ 2× two-pass at 32K) is evaluated on these if 32K
   fits, else reported unassessable with the 8192 ratio.
4. Fingerprint amendment as before; existing receipts untouched
   (expected: no invalidation). `lead_commands.txt` refreshed. CPU
   gates for the new kind and for the wrapper-off path (assert no hook
   is installed in `decode_clean`). Prior Art Directive applies.

Same rules: no git, no subagents, foreground only, < 10 min per call,
never kill anything, registration immutable, byte-identical kernels.
Your sandbox has no GPU: build, CPU-verify, blocked-report; the lead
runs the card.

## Done (verbatim)

1. The reproduction result (ms/token, configuration) and the bisect
   ladder with numbers or the blocked-report for each rung.
2. New cells, rails, estimates; the no-hook pin test name.
3. Fingerprint amendment; receipts affected (expected none).
4. Prior art; deviations; RED; process safety; model id and effort.
