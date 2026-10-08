# APA-MF2.2 Amendment 3 — close V7 (long-key statistics precision) and adjudicate the last red probes

The full-chain review (`apa-mf22rev` 887ad21: `docs/APA_MF2_2_REVIEW.md`, `artifacts/apa_mf22rev/verdict.json`) rejects
promotion on one blocker only.
- **V7 (BREAK).** At S = 2^20 visible keys with tightly clustered |bulk| (`Kq_j = sort(1 − (j mod 256)·2^-20)·e0`, fp32,
  D64, scale 1, refine 0.15), the fp32 statistics in `quant.py:219` (`mean = u.sum(...)/cnt`) drift. The threshold
  comes out at 1.0000026 against the definition's 0.99995145, giving 208,896 mismatches, 122,880 of them outside the
  band. The lead confirmed it on the GPU: `long_keys` is RED for both the exact operand and the public-quantizer
  operand.
- **Everything else is approved or approved with nits:**
  - N1 CLOSED; N2 CLOSED-WITH-NITS;
  - Amendment 1 APPROVE; Amendment 2 APPROVE-WITH-NITS;
  - MF1 fixes A and B APPROVE-WITH-NITS.

The lead's run of the review's GPU probes (`logs/apa_mf22rev_gpu_*_20261008T*.log` on `apa-mf22rev`):
- **PASS:** n1, n2, backward, paths, left_padding, geometry (exhaustive L 1..2048), identity (702 arrays, 0
  non-blocked changes) and author_tests.
- **RED:**
  - long_keys (V7);
  - guards ×3: `*/NaN_softmax_policy`, one per dtype;
  - real ×1: Qwen3-1.7B L16 bf16 L3, 2/384 output elements outside rtol 0.06 / atol 0.02, for example 0.734 vs
    0.803;
  - consumers ×2: read-only GRAPA/GRM callers of primitive softmax not covered by the guard (V1, non-blocking).

Nits:
- **V3:** the `functional.py:116` comment says PyTorch SDPA returns NaN for a fully masked row. Installed torch 2.11
  SDPA returns 0, and only the primitive returns NaN.
- **V4:** identity-count wording.
- **V5:** `test_apa_phase6._ref` still uses a finite −1e9 mask and one-pass statistics, so it disagrees with the
  independent oracle on a witness.

Base: `apa-mf2` HEAD (a2e3ca1, plus this order).

## Mechanism & scope

- Measures:
  1. **V7:** the selection statistics match the fp64 definition, within the registered relative band of 2e-5, for
     S up to 2^20 visible keys, including tightly clustered rows. This must hold with no regression on the earlier N2
     range sweep, the real captures, the near-tie set or the identity set.
  2. Each red row in the guards and real cells is classified as a product defect or a probe defect, with evidence.
     Product defects in scope are fixed.
  3. The V3, V4 and V5 nits are fixed.
- Mechanism doc: `tensor_cuda/tensor_cuda/quant.py` (the selection-statistics docstring and comments). Also
  `docs/APA_MF2_2_REVIEW.md` on `apa-mf22rev`, and Chan, Golub & LeVeque (1983) for shifted or compensated moments.
- Exercises:
  - `quant.py`: the statistics block;
  - the guard helpers in `functional.py`: for the NaN policy, if it is a product defect;
  - the V3 comment;
  - the `test_apa_phase6._ref` reference;
  - the review probes, copied from `.../tensor_fork_apa_wt_mf22rev/tests/apa_mf22rev/gpu_probe.py`.
- Does not cover:
  - GRAPA and GRM consumers (read-only; V1 is reported to David);
  - the BK4/'a' kernels;
  - the quantizer overflow;
  - the extension build.
- Required evidence:
  - V7 before and after, on the review's witnesses plus a sweep: S ∈ {2^10, 2^14, 2^17, 2^20}; cluster widths
    {2^-20, 2^-16, 2^-10, 1}; causal and non-causal; fp32 and bf16;
  - the full N2 range sweep and the real-capture table, unchanged at 0 out of band;
  - the adjudication table for the guards and real rows;
  - the full suite;
  - mutation.
- Operand census: the review's long-key witnesses; earlier sweeps; real captures (Qwen3-1.7B and 4B, S=8192). The
  memory cost of fp64 or extra passes at S=2^20 must be reported.
- Baseline isolation: separate processes per engine and package arm; your own fp64 oracle.
- Forbidden configuration:
  - widening the band;
  - special-casing the witness;
  - any statistic that changes the definition (only its arithmetic precision may change);
  - the retired 595.84 override;
  - building into another item's build directory.

Seat: the same Claude Opus 5.5 high seat.

## Where to work

The same worktree and rules. Writable only:
- `tensor_cuda/tensor_cuda/quant.py` (statistics block);
- `tensor_cuda/tensor_cuda/functional.py` (the V3 comment; the guard helpers only if the adjudication shows a product
  defect);
- `tensor_cuda/tests/test_apa_phase6.py` (`_ref` only);
- new `tensor_cuda/tests/test_apa_mf2_2c_*.py`;
- `scripts/apa_mf2_2c_*`, `artifacts/apa_mf2_2c/`, `docs/APA_MF2_2_LEDGER.md` (append) and `logs/`.

Build only into `artifacts/apa_mf2_2c/build/`, if needed.

NO git. NO subagents. NO DELETES. Native driver only. Foreground runs with explicit timeouts.

## Gates — registered BEFORE any GPU gate

`artifacts/apa_mf2_2c/registration.json` and its `.sha256`, create-only.
- **G1:** the full suite's FAILED/ERROR set equals the 52 baseline ids.
- **G2 (V7):** 0 out-of-band mismatches on every V7 witness and sweep cell.
- **G3 (no regression):** the N2 sweep, the near-tie set and the real captures stay at 0 out of band. Outputs are
  byte-identical to a2e3ca1 on the identity set, unless selection statistics changed only through precision, in which
  case the differences are listed and bounded.
- **G4:** every red guards and real row is adjudicated; product defects in scope are fixed.
- **Mutation:** ≥ 0.80, including "fp32 native-order statistics restored".

## Done

Append to the ledger and write `artifacts/apa_mf2_2c/verdict.json` and `artifacts/apa_mf2_2c/item5.patch`. The final
message gives the verdicts, the V7 and adjudication tables, the files and diff stat, and receipts cited as
`path:line "quote"`.
