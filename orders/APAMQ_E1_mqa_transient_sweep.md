# APAMQ-E1 — MQA-Geometry Kernel Transient Sweep

YOUR WRITABLE TARGET is /mnt/ForgeRealm/Project-Tensor — specifically
`scripts/` and `artifacts/apamq_e1/` (create it). Edits there, builds of
YOUR OWN scripts, and GPU runs are AUTHORIZED. Run first, report after;
do not ask permission for anything non-destructive. A registered order
IS the permission.

HARD READ-ONLY: `tensor_cuda/src/`, `tensor_cuda/tests/`, the built
engine .so, and everything under `docs/` and `orders/`. Do NOT rebuild
the engine — another seat imports the same .so at runtime. If a needed
kernel entry point is missing or broken, STOP and report RED; do not
patch.

No git — the lead commits. No subagents. RED honesty: a failed or
partial sweep is a result; report it as such, never fake or
extrapolate a cell. No monitor-idling: if blocked, say so and stop.

GPU: serialize every GPU-touching run under
`flock -w 7200 /tmp/forge-gpu.lock`. Another seat (Gemma model runs,
~9GB resident) shares the card; hold the lock only around actual GPU
work. The operator has absolute right of way.

## Context (pre-nailed premise — do not re-litigate)

Plan: `docs/APA_MQA_ROOTCAUSE_PLAN.md` (read it first). Hypothesis
under test, H-A: the fused `apa_selective_attention` path's
peak-transient advantage over the standard attention path scales with
q-heads × S and is INDEPENDENT of KV-head count. Score tensors are
(q_heads, L, S) regardless of kv_heads, so MQA (kv=1) should see the
same fused-path memory advantage as GQA. Prior record says APA on the
MQA model got only ceiling parity; this sweep establishes whether the
kernel-level memory economics themselves discriminate by kv count.

Evidence class: KERNEL SWEEP. You may claim speed, memory shape, and
transient sizes. You may NOT claim anything about model quality;
synthetic random tensors are correct here, not a limitation.

Anchors: `tensor_cuda/src/kernels.cu:1053` TC_APA_MAXD = 512 (D=512 is
legal in the fused path now). Existing usage patterns for both paths:
`tensor_cuda/tests/test_apa_selective.py` and the other test_apa_*.py
files show how apa_selective_attention and the standard/blend
attention are invoked, including bulk-quantized keys (4-bit bulk,
refine ~0.10 practice point).

## The sweep

Write `scripts/apamq_e1_sweep.py`. Matrix:

- Geometry (kv_heads, q_heads): (1,16) MQA, (4,16), (8,16), (16,16) MHA
  control. B=1.
- Head dim D: 128 and 512.
- Context S: 4096, 8192, 16384, 32768, 65536.
- Shapes: PREFILL-CHUNK — L=512 query rows vs S keys, rectangular
  causal (bottom-right aligned, the with-cache regime); DECODE — L=1.
- Paths: (a) STANDARD — the engine's standard attention compute as the
  ports use it (de-expanded / grouped where that is the current
  convention; note exactly which ops you called); (b) FUSED APA —
  apa_selective_attention at the practice operating point (4-bit bulk,
  refine 0.10). fp16/bf16 inputs matching port convention.

Per cell measure: (1) engine pool high-water delta across the call
(reset/read pool stats around it), (2) nvml device memory sampled
before/during/after, (3) wall time (warm, median of ≥3 after 1
warm-up). One PROCESS per (path, geometry, D) row — iterate S inside,
ascending, so an OOM ends the row cleanly; an OOM is DATA (record the
S where the row walls), not a failure. Cap any single cell at 120s.

Output: `artifacts/apamq_e1/results.json` (every cell: config, pool
peak bytes, nvml peak MiB, wall ms, or OOM marker) and
`artifacts/apamq_e1/RESULTS.md` — one table per D: rows = (path, kv),
cols = S, cells = pool-peak MiB and ms. Plus a short factual notes
section: which engine ops each path actually invoked, and anything
anomalous. NO verdicts on H-A/T1 — raw numbers only; adjudication is
the lead's.

## Done

Your final message MUST contain verbatim: the exact commands run, the
full RESULTS.md tables pasted inline, the results.json path, any cells
that OOMed or errored with the error text, and an explicit statement
of which engine entry points each path used. If anything was skipped,
name it and say why. Honest partial > polished incomplete.
