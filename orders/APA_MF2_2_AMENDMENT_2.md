# APA-MF2.2 Amendment 2 — causal composed rows whose visible keys are all blocked return 0, not NaN

The lead's decision on Amendment 1's G2 RED (72 cells; `artifacts/apa_mf2_2a/verdict.json`,
`census_table.md`):
- **The cells:** the causal composed paths are `functional.scaled_dot_product_attention` (causal infer and grad) and
  `quant.apa_quant_attention` (p 1 and 0.15). On a row whose every VISIBLE key scores −inf, they return NaN; this
  started at MF2 (66287b1).
- **fd32b6d's value there was finite but wrong:** `V[S-1]`, a future key leaking through the finite −1e4 mask (F4).
- **The decision: option (b).** Such rows return **0**. This is the FlashAttention-2 convention, and it matches the
  two fused kernels after Amendment 1.
- **Why:** NaN poisons padded batches. With left padding, the first query rows see only padding keys; their NaN
  enters the next layer's K/V, and 0 × NaN = NaN in the other rows' PV.
- **The rule from now on:** no path returns NaN for a row whose scores are all −inf. Rows whose scores contain NaN may
  stay NaN.

Base: `apa-mf2` HEAD f3cde75.

## Mechanism & scope

- Measures: on every causal composed path, the output of a row whose visible keys all score −inf (or whose
  `attn_mask` blocks every visible key) is exactly 0, in forward. In backward, gradients are finite and 0 for that
  row. All other outputs stay byte-identical to MF2.2 + Amendment 1.
- Mechanism doc: `tensor_cuda/tensor_cuda/functional.py` (the `_causal_mask` comment block and the
  `scaled_dot_product_attention` masking comments). Also `orders/APA_MF2_2_AMENDMENT_1.md` and
  `artifacts/apa_mf2_2a/census_table.md`.
- Exercises:
  - `functional.scaled_dot_product_attention`: the causal composed path, the grad path, and the fused
    `causal_softmax` inference path if it is reachable for such rows;
  - `quant.apa_quant_attention`: the selection and softmax for blocked rows;
  - the Amendment-1 census harness (`scripts/apa_mf2_2a_census.py`), re-run with an "after2" arm.
- Does not cover:
  - the fused non-causal kernels (done in Amendment 1);
  - the BK4/'a' kernels;
  - rows containing NaN scores;
  - the extension build.
- Required evidence:
  - the census re-run: fd32b6d / MF2 / MF2.1 / MF2.2 / after1 / after2, with zero "fd32b6d finite → after2 NaN"
    cells and zero NaN on any all-(−inf) row on any path;
  - byte-identity of after2 against after1 on every other census and identity cell;
  - finite backward gradients on blocked rows;
  - the full suite;
  - mutation.
- Operand census: the same as Amendment 1, plus a left-padded batch-style case: Lq=S, the first k keys masked for
  all queries via `attn_mask` −inf, k ∈ {1, 3}, causal.
- Baseline isolation: separate engine processes per arm, as in Amendment 1.
- Forbidden configuration:
  - restoring a finite additive mask (it re-opens F4);
  - masking blocked rows by changing any row that is not blocked;
  - the retired 595.84 override.

Seat: the same Claude Opus 5.5 high seat.

## Where to work

The same worktree and rules. Writable only:
- `tensor_cuda/tensor_cuda/functional.py` (the causal SDPA masking and softmax lines);
- `tensor_cuda/tensor_cuda/quant.py` (the `apa_quant_attention` softmax and selection lines);
- new `tensor_cuda/tests/test_apa_mf2_2b_*.py`;
- `scripts/apa_mf2_2b_*`, `artifacts/apa_mf2_2b/`, `docs/APA_MF2_2_LEDGER.md` (append) and `logs/`.

**The engine build directories of other items are READ-ONLY.** Build only into `artifacts/apa_mf2_2b/build/` if you
need a build. Always pass the build-dir override; Amendment 1 overwrote the MF2.2 binary this way.

NO git. NO subagents. NO DELETES. Native driver only. Foreground runs with explicit timeouts.

## Gates — registered BEFORE any GPU gate

`artifacts/apa_mf2_2b/registration.json` and its `.sha256`, create-only.
- **G1:** the full suite's FAILED/ERROR set equals the 52 baseline ids.
- **G2:** the census has zero cells where fd32b6d is finite and after2 is non-finite, and zero all-(−inf) rows that
  are NaN on any path.
- **G3:** after2 is byte-identical to after1 on every non-blocked cell, and the identity set is unchanged.
- **G4:** the left-padded case returns exactly 0 on the padded rows and finite values everywhere.
- **Mutation:** ≥ 0.80, including "blocked-row guard removed" for each path.

## Done

Append to the ledger, write `artifacts/apa_mf2_2b/verdict.json` and `artifacts/apa_mf2_2b/item4.patch`, and give a
final message with the verdicts, the census table, the files and diff stat, and receipts cited as `path:line "quote"`.
