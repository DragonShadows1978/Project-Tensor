# APA-SP3 amendment 2 (lead, 2026-09-06) — the original protocol is unrecoverable; register PROTOCOL-2 and measure the baselines fresh

Your r1 G0 finding is CONFIRMED by the lead: the 20.065 / 19.817 receipts
came from `/tmp/minicpm3_engine_bench.py`, `/tmp/minicpm3_reference.py`,
`/tmp/minicpm3_ceiling.py` on one 512-token window of one guide-corpus
document; none of those exist on any disk, the guide document is not
identified in the results doc, and the tokens cannot be reconstructed.
Nothing in this repo or GraftRepository history holds them. Stop trying
to recover them. Those two numbers are now HISTORICAL CONTEXT (8 GB
RTX 3070, see amendment 1), not targets.

This is r2 of the same order in the same worktree
(`/mnt/ForgeRealm/Project-Tensor-wt-apa-sp3`, branch `apa-sp3`; your r1
work is committed by the lead as-is). Same writable target, same
read-only boundaries, same rules. `registration.json` stays IMMUTABLE:
this amendment is the "separately reviewed implementation amendment"
your `protocol()` guard asks for. Implement it as a NEW amendment
status your guard accepts, bound by sha to this file AND to the
registration; do not loosen any other check.

## PROTOCOL-2 (registered by this amendment)

- **Corpus:** `wikitext-2-raw-v1`, split `test`, from the offline HF
  datasets cache (`HF_DATASETS_OFFLINE=1`; snapshot
  `~/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3`,
  ~1.29 M characters). Rows joined with `"\n"` exactly as
  `GraftRepository/tests/minicpm3_bulkbits_floor.py::get_text` does
  (cite it: that script is the most recent MiniCPM3 perplexity protocol
  on this engine and its window scorer is the reference for yours).
- **Tokenizer:** the MiniCPM3-4B snapshot's own tokenizer
  (`AutoTokenizer.from_pretrained(<snapshot>)`), no special tokens
  beyond its defaults; state exactly what it adds.
- **Canonical stream:** one 1-D little-endian int64 array of the whole
  test split, saved to `artifacts/apa_sp3/protocol2_tokens.npy`, sha256
  recorded in the amendment JSON. It must exceed 32,800 tokens (it will;
  say the count).
- **Scoring:** `last_512_targets_within_input` at window 1024, six
  consecutive non-overlapping windows starting at token 0 (the floor
  script's `WINDOW, SCORED, N_WINDOWS = 1024, 512, 6`), all 512 targets
  per window scored in fp64 log-softmax from the model's logits, ppl =
  exp(total NLL / total targets). Whether you feed the 1024 tokens in
  one prefill or prefix-512 + 64-token steps as the floor script does,
  state it, pin it, and use the SAME feeding for every arm. Long rows
  (S=8192 / 32768) take their prefix from the same stream starting at
  token 0 and score the last 512 within the input.
- **Model path:** unchanged from your registration (INT4 g128 weights,
  bf16 compute dtype as registered — note the results doc's runs were
  the engine's INT4 path of June; if your compute dtype differs from
  what `core/minicpm3_tc.py` selects by default, say so and use the
  adapter's default).

## G0 under PROTOCOL-2

Replace the 20.065/19.817 targets with a determinism gate: arm A run in
two FRESH processes agrees to ±0.001 ppl, and arm B likewise. Then A and
B are the baselines every other arm is compared to. Registered
expectation (prediction, NOT a gate): B − A within ±0.3 ppl at bulk 4,
consistent with the floor doc's "4-bit free" finding; a B − A outside
that is a finding to report, not a reason to stop.

## Everything else stands

Arms A–E, C's δ rule, E's real-key e_q rule, G2 margins, G3 decode,
both prediction sets (P2's "±0.05 of B" now refers to the fresh B),
the cell registry, the runner, RED honesty, no git, no subagents,
foreground only, < 10 min per call, never kill anything you did not
start. Your sandbox has no GPU: build, CPU-verify (re-run your gates
including the amendment guard tests: a forged amendment, a stale
amendment, a stream with the wrong sha, must each RED), and refresh
`lead_commands.txt` + the blocked-report. Ledger the amendment in
`docs/APA_SP3_LEDGER.md`.

## Done (verbatim)

1. Amendment JSON path + sha; token count + sha; the exact feeding
   scheme; the guard test names.
2. What changed in scripts (files/lines) and what did not.
3. CPU gate results; refreshed GPU blocked-report + exact lead commands
   in dependency order with wall estimates.
4. Prior art (unchanged or added); deviations; RED; process safety;
   model id and reasoning effort.
