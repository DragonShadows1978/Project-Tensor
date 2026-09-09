# APA-SP4G amendment 7 (lead, 2026-09-08) — the memory ceiling, with a long lease authorized by David

David, 2026-09-08: "let's check OOM on Gemma 4." SP4G's ceiling grid
found all arms fit 4096 and 8192 on 12 GB and every 16K-and-up cell
died on the 285 s worker rail before reaching memory. The ceiling is
unmeasured. **David authorizes long leases for this amendment:** up to
1,500 s worker / 1,560 s outer per cell, one cell per lease, still
`flock --wait` on `/tmp/forge-gpu.lock`, still foreground, still no
kills. This is the only order on this machine with that rail; do not
generalize it.

Same worktree (`/mnt/ForgeRealm/Project-Tensor-wt-apa-sp4g`, branch
`apa-sp4g` at 29882ae, history rewritten to drop capture arrays —
`git` is still not yours), same rules otherwise.

## Mission

1. Register kind `ceiling_long` (`ceiling_long_{A,B,C}_{S}`) with the
   1,500 s worker rail: prefill-only from the pinned stream at S ∈
   {16384, 24576, 32768, 49152, 65536, 98304, 131072}, per arm A
   (standard), B (two-pass r=0.15), C (single-pass, frozen δ=3.0),
   ascending, stopping an arm at its first OOM (later S for that arm
   become registered non-fits without running). Each receipt: fit /
   OOM / rail, peak resident, wall, and the KV-cache size at that S
   (global MQA cache bytes per token × S; sliding fixed) so the report
   can separate cache from transient. Use the June adapter's chunked
   prefill exactly as the ppl cells do; pooling ON.
2. Register the lead predictions: standard OOMs between 12K and 16K
   (16 heads × S² bf16 scores on the global layers = 8 GB at 16K on
   top of 6.8 GB weights); two-pass and single-pass reach 32K with
   resident under 10 GB and the wall is time, not memory, until at
   least 64K; single-pass reaches at least one rung further than
   two-pass because it has no O(S) bulk/rank/recon transients. Register
   yours beside them.
3. If the wall at some S is the 1,500 s rail rather than OOM, record
   RAIL and the extrapolated wall from the measured S² scaling; do not
   retry.
4. Fingerprint amendment (no existing receipts affected); CPU gates
   for the new kind and rail; refreshed `lead_commands.txt` (ascending
   per arm, A first so the standard wall is known early) and
   blocked-report; RESULTS.md gains the ceiling table with the June
   8 GB 3070 numbers (bf16 weights: prefill ~10–11K solid, 16K OOM;
   qv 12K) as context. Prior Art Directive applies (nothing new).

## Done (verbatim)

1. Cell ids, rails, per-cell wall estimates from the 8192 receipts'
   scaling.
2. Both prediction sets.
3. Fingerprint amendment; CPU gates; lead commands; RED; process
   safety; model id and effort.
