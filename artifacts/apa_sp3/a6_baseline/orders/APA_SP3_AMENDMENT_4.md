# APA-SP3 amendment 4 (lead, 2026-09-06) — 8192 capture split, torch reference for the long rows, ceiling probe, 32K perplexity

## Where the card stands (PROTOCOL-2, bits 4; receipts under `artifacts/apa_sp3/jobs/`)

- A 8.655828, B 8.778802 (both bit-identical across fresh processes),
  D 8.655352, **C (δ=4.28125, fraction 0.104 vs B 0.103) 8.661116**,
  all ppl@1024; at S=8192 last-512: B 10.4036, **C 10.0424**; A OOM
  (registered non-fit). Lead P1 HIT; P2 MISSED in the good direction
  (C is 0.118 BELOW B, not within ±0.05).
- G2 at S=1024, all 62 layers, arm B: real-key |bulk−exact| per-layer
  mean median 0.25 (max-layer 2.03), p99 median 0.91, max median 2.86
  and 105.7 on layer 0; the z-score selection leaves 92% of softmax
  mass on UNREFINED keys on average and skips a relative-weight-1.0 key
  on every layer. P4 HIT.
- **`capture_b4_B_8192` RED `WORKER_TIMEOUT`** at layer 60/62 of the
  480 s rail (partial directory archived by the lead under
  `captures_stale_r2/`). 192 cells sit behind it (E's e_q, E ppl,
  8192 margins, E captures). The lead is currently running the 160
  cells that do not depend on it (decode, bits-8) in THIS worktree.

## You are dispatched into a SEPARATE worktree

`/mnt/ForgeRealm/Project-Tensor-wt-apa-sp3-a4`, branch `apa-sp3-a4`
forked from `apa-sp3` HEAD. Do not touch `Project-Tensor-wt-apa-sp3`
(a leased chain is running there; its receipts must not be raced).
Same writable set and rules as before. The lead merges your branch
into `apa-sp3` after the running chain ends.

## Mission

1. **Split the S=8192 (and 32768) captures** so no job exceeds the
   480 s worker rail: per-layer-range capture jobs (e.g. two or four
   ranges) that write into the same capture layout the margin cells
   read, with a final `capture_*` aggregation cell that verifies every
   layer is present and sha-pins the set. Margin cells depend on the
   aggregation cell exactly as they depended on the capture before.
   Registered planning estimate per range must be < 300 s using the
   observed 8192 rate (60 layers in ~480 s).
2. **Torch reference arm `T`** (evidence class: model perplexity,
   reference): the same HF snapshot under PyTorch (`transformers`,
   `trust_remote_code`, bf16 weights, `attn_implementation="sdpa"`,
   fused kernel forced via `torch.nn.attention.sdpa_kernel` — flash if
   sm_89 supports it for this head dim, else efficient; state which),
   scoring the SAME 512 targets on the SAME pinned token stream as
   every other arm at S ∈ {1024 (six windows), 8192, 32768}. This is
   the ground-truth column for the long rows the engine's dense path
   cannot reach (S×S at 40 heads is 10.7 GB at 8192). Each T cell is
   its own leased job (bf16 4B ≈ 8 GB resident; if 32768 does not fit,
   say so in the receipt, do not chunk the attention). Note the June
   receipts had a GT of 17.357 vs engine 20.065 on a different
   protocol; expect a gap at 1024 too (INT4 weights vs bf16) and
   REPORT it, it is not a RED.
3. **Ceiling probe per arm on 12 GB** (evidence class: kernel sweep /
   memory shape): max S each of A, B, C prefills without OOM on the
   grid {4096, 8192, 16384, 24576, 32768}, prefix from the pinned
   stream, one leased job per (arm, S), bounded < 300 s each, peak
   resident recorded. The June 8 GB numbers (3,072 / 32,768) are
   context, not targets.
4. **ppl at 32768** for B and C (last-512 within input, prefix 0),
   only if the ceiling probe says they fit; D at 32768 pinned against
   T (not A).
5. **Receipt validity across your edit:** your fingerprint rule marked
   every receipt stale after r3's one-line model edit. That rule is
   right for the files a cell actually executes and wrong as a blanket.
   Implement a fingerprint AMENDMENT (separate JSON, bound to this
   order's sha and the registration sha) that records the r3→a4 file
   deltas and rules which existing PASS receipts remain valid: a
   receipt stays valid iff none of the files its cell kind imports
   changed semantically for that kind. State the per-kind import
   closure you used. The lead will NOT re-run 150 cells for a registry
   edit; if your change genuinely invalidates a kind (e.g. you touched
   the ppl window scorer), say so and list them.

Registration immutable; new cells are registered in an amendment
JSON, not by editing `registration.json`. Prior Art Directive
applies (cite what T's SDPA path is). CPU gates for every new piece,
`--dry-run` enumerating the new cells, refreshed `lead_commands.txt`
(dependency order, wall estimates). No git, no subagents, foreground
only, < 10 min per call, never kill anything. Your sandbox has no GPU.

## Done (verbatim)

1. New/changed cells (ids, kinds, deps, estimates); capture split
   scheme; T arm entry point and SDPA backend; ceiling grid.
2. Fingerprint amendment: which existing receipts stay valid and why
   (per-kind closure), which do not.
3. CPU gate results; blocked-report; exact lead commands.
4. Prior art; deviations; RED; process safety; model id and effort.
