# APA-SP5 amendment 1 (lead, 2026-09-08) — make the perplexity instrument decidable, and finish items 4–6 as cells

Your r1 report is accepted as delivered; the lead verified the ppl,
exactness and sink receipts on disk and that `tensor_cuda/` is
untouched. Deviations 1–3 are accepted. Same worktree, same rules.

## Ruling on the RED

Correct call not to widen a gate. But "more windows need a longer
lease" is not right: a window is a cell. PROTOCOL-O windows are ~40 s
each; sixteen windows per arm are sixteen cells, each far under the
rail. The instrument is not lease-bound, it is N-bound.

## Mission

1. **Extend to N=16 as per-window cells** (`ppl_{arm}_W1024_w{00..15}`)
   for A, B, C, D, E at bulk 4, plus **per-window A32** (full layers in
   fp32) so the floor is a distribution, not one number. Aggregate cells
   compute pooled ppl over 16 windows (8,192 targets) and report, per
   window, each arm's deviation from A in multiples of that window's
   own floor. Register the prediction now: with 16 windows the pooled
   C − B lands inside the pooled floor, and window 1 stays an outlier
   in both directions. If instead C is consistently above A across
   windows by more than the per-window floor, that is the finding and
   it points at the δ=3.16 selection on this model (item 2 below).
2. **Item 4 as registered replay cells, window 1 first:** capture the
   kernel's own selection mask during the actual ppl forward (SP4G a2
   construction, bitwise replay), per full layer, arms B and C, on
   window 1 and on one calm window (2): |bulk−exact| mean/p99/max,
   unrefined softmax mass, max relative weight of a skipped key,
   fraction, and the sink mass per query. The question to answer: on
   window 1, what does the running-max tail skip that the z-score tail
   refines, and vice versa — is the 2.9× spread a mass story or a
   sink story?
3. **Item 5 decode:** you established the port re-quantizes every
   step (`gpt_oss20b_tc.py:741`). Register `decode_{A,B,C}_{2048}` in
   the clean configuration anyway (32 synced steps) and report the
   gap as an adapter cost with that line cited; no fix, no patch.
4. **Item 6 ceiling cells:** `ceiling_{B,C}_{4096,8192,16384}` under
   the normal rail (A is a registered non-fit at 2048 already; keep
   its receipt). Peak VRAM per rung from `cudaMemGetInfo`.
5. Fingerprint amendment (nothing existing invalidated expected);
   refreshed `lead_commands.txt` in dependency order with walls; you
   may run window cells yourself on the card under the lease rules,
   but hand the bulk of the 16×6 window set to the lead's detached
   loop rather than sitting on the lease for an hour. No git, no
   subagents, no background tasks, never kill anything. Prior Art
   Directive applies (nothing new expected).

## Done (verbatim)

1. New cell ids and walls; the aggregate's method; your prediction
   for item 1 beside mine.
2. What you ran yourself, with results.
3. Fingerprint amendment; lead commands; RED; process safety; model
   id and effort.
