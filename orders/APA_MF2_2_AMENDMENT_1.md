# APA-MF2.2 Amendment 1 — no new NaN where production was finite: restore fd32b6d's fully-blocked-row result

The lead's decision on the MF2.2 open item "fully blocked rows":
- In the two non-causal fused kernels, a row whose every score is −inf now returns NaN on MF2.1/MF2.2. It returned 0
  on fd32b6d and MF2.
- In padded batch inference, such a row's NaN can spread: the next layer's K/V for that position are NaN, and a
  masked weight of 0 × NaN = NaN still poisons the other rows' PV.
- The chain may not introduce a NaN anywhere production returned a finite value.

The base is `apa-mf2` HEAD 86f0a18 (MF2.2 items 3299218 and e47ef4b, plus receipts). The plan of record remains
`orders/APA_MF2_2_REVIEW_FIXES.md`; this amendment adds items 3 and 4.

## Mechanism & scope

- Measures:
  - **Item 3:** for each path, the output on fully-blocked rows (every score −inf, or every key masked) equals
    fd32b6d's output. That means 0 wherever fd32b6d gave 0, and unchanged wherever fd32b6d already gave NaN.
  - **Item 4:** a census over fd32b6d vs the chain HEAD of every path on special rows, finding any other place where
    production was finite and the chain is not.
    - The special rows: all −inf; all masked; −inf first; a single visible key; S == 1.
    - The paths: `fused_sdpa_noncausal`, `apa_int4_sdpa_noncausal` (all and partial), eager softmax, `causal_softmax`,
      SDPA infer and grad, and `apa_quant_attention` at p 1 and 0.15.
- Mechanism doc: `tensor_cuda/src/kernels.cu` (the online-softmax recurrence comments in the two fused kernels).
  Also `orders/APA_MF2_2_REVIEW_FIXES.md`.
- Exercises: the two fused kernels' warp merge and finalize (the guard on `gmax` and on the final normalizer); the
  eager and composed paths as observed. Change code only where the census shows a new non-finite value.
- Does not cover:
  - outputs where fd32b6d already returned NaN (record them; do not change them);
  - the BK4/'a' kernels;
  - the quantizer overflow;
  - the extension build.
- Required evidence:
  - the census table: path × special row × engine (fd32b6d, MF2, MF2.1, MF2.2, after), finite or NaN, plus the value;
  - byte-identity of after vs MF2.2 on every ordinary finite input already in the MF2.2 identity set;
  - mutation.
- Operand census: fp32/fp16 (bf16 where the path supports it); D ∈ {64, 96, 128}; Lq ∈ {1, 3}; S ∈ {1, 2, 33, 1025};
  int4 all and partial.
- Baseline isolation: separate engine processes for fd32b6d (`artifacts/apa_mf2_2/build_fd32b6d`), MF2, MF2.1, MF2.2
  and after.
- Forbidden configuration:
  - reverting the −inf start or the N1 guard (F1 and N1 must stay closed);
  - changing any output that fd32b6d already returned as NaN;
  - the retired 595.84 override.

Seat: the same Claude Opus 5.5 high seat as MF2.2.

## Where to work

The same writable target and rules as MF2.2.
- `kernels.cu`: the two non-causal fused kernels' recurrences, merge and finalize only.
- New tests: `tensor_cuda/tests/test_apa_mf2_2a_*.py`.
- `scripts/apa_mf2_2a_*`, `artifacts/apa_mf2_2a/`, the ledger section `docs/APA_MF2_2_LEDGER.md` (append), and
  `logs/`.
- Rebuild into `artifacts/apa_mf2_2a/build/`.

NO git. NO subagents. NO DELETES. The native driver only.

## Gates — registered BEFORE any GPU gate

`artifacts/apa_mf2_2a/registration.json` and its `.sha256`, create-only.
- **G1:** the full suite's FAILED/ERROR set equals the 52 baseline ids.
- **G2:** the census shows no path and special row where fd32b6d is finite and after is not.
- **G3:** after is byte-identical to MF2.2 on the ordinary finite identity set, and the N1 witnesses stay correct.
- **Mutation:** ≥ 0.80, including "fully-blocked guard removed".

## Done

Append to the ledger, write `artifacts/apa_mf2_2a/verdict.json` and `artifacts/apa_mf2_2a/item3.patch`, and give a
final message with the verdicts, the census table, the files and diff stat, and receipts cited as `path:line "quote"`.
