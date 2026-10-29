# APA certificates in Lean

This folder proves five claims about the selection rules of Adaptive Precision Attention (APA). They are proved in
Lean 4 (`leanprover/lean4:v4.34.1`) with Mathlib at commit `d13f23b723b8a846827a245b89c10fc7d3f11612`. Each claim comes
with a second theorem showing that its hypotheses can actually be met, so the claim is not true vacuously.

| Lean name | The claim in plain words |
|---|---|
| `sp_skip_final` | Single-pass APA (SP) refines key `j` when its cheap score is within δ of the running maximum. A key that rule skips would also be skipped against every later running maximum and against the final maximum. A skip is never wrong in hindsight, so one pass is exact for skips. |
| `sp_mask_superset` | Every key the final-maximum rule refines is also refined by the SP rule. SP can refine more keys, never fewer, and the extra set can be strictly larger. |
| `splitk_contains` | Under split-K decoding, a partition's local running maximum is never above the global one. So split-K refines at least the keys global SP refines. Its output is still not equal to the global one: with cheap scores [2, 0], exact scores [2, 10], values [0, 1], δ = 0 and two one-key partitions, the outputs differ by more than 0.8. This holds both with an attention sink of logit 0, as in the original test, and without one. |
| `z_not_single_pass` | The two-pass kernel's z-score rule (refine when \|score\| ≥ mean + z·std over the whole row) cannot be decided in one irrevocable pass. Rows [1,0,0,0] and [1,0,0,10] share their first key, yet for every z in [0, √3] that key is refined in the first row and not in the second. |
| `sp2_certificate` | Suppose every cheap score is within `e_q` of the exact one, and SP with δ = ln(1/ε) + 2·`e_q` skips key `j`. Then the key's true softmax weight is below ε times the largest true weight. |

## What is checked, and what is not

- **`ApaCerts/Spec.lean` holds the statements.** They were written before any proof, and they are the trust boundary:
  a proof only shows that its statement holds. Whether each statement says what the APA results claim was checked
  separately, by a review from a different model family from the one that wrote the statements. That review found two
  defects. The statements were amended and the amendment was reviewed again (see "Review record").
- **`ApaCerts/Proofs.lean` holds the proofs.**
- **`ApaCerts/Check.lean` ties the two together.** The build fails unless each proof has exactly its statement's type.
  It also prints the axioms each theorem uses. The expected set is Lean's standard three, `propext`, `Classical.choice`
  and `Quot.sound`. No `sorry` and no added axioms appear.
- **Not covered:**
  - The statements are over the real numbers, so floating-point rounding is outside them.
  - They concern the selection rules, not the CUDA kernels' code. The z-score definition mirrors the production
    kernel's formula; that correspondence is by reading, not by proof.

## Build

```sh
# install elan (https://github.com/leanprover/elan), then:
cd lean/apa_certs
lake exe cache get        # prebuilt Mathlib for the pinned commit
lake build ApaCerts.Check
```

A successful build ends with `Build completed successfully`, after ten `depends on axioms: [propext, Classical.choice,
Quot.sound]` lines. Without the cache step, Lake builds Mathlib from source, which takes hours.

## Review record

- **Statements written:** TX3-X5 statement order, with proofs by a separate seat.
- **Statement-fidelity review TX3-X5-REV1:**
  - 4 of 5 statements faithful.
  - The split-K counterexample omitted the original test's attention sink.
  - One non-vacuity witness did not exercise its claim's hypothesis.
- **Amendment and re-proof:** TX3-X5B.
- **Re-review TX3-X5-REV2:** APPROVE; all five statements faithful, all five witnesses adequate, no open findings.

The reports are in `docs/TX3_X5_REV1_REPORT.md` and `docs/TX3_X5_REV2_REPORT.md`.

## Prior art

- `#print axioms` and the split between statements and proofs follow common Lean practice for checked challenge
  statements. A separate statement file, written first and pinned by a check file, is the comparator-challenge
  pattern.
- The online softmax and split-K merge are standard (Milakov and Gimelshein 2018; Dao et al., FlashAttention, 2022).
- The running-maximum comparator behind the single-pass rule comes from BLASST (Yuan, Shinn et al., "BLASST: Dynamic
  BLocked Attention Sparsity via Softmax Thresholding", arXiv:2512.12087, 2025). BLASST uses it to skip blocks.
- This project's contributions:
  - applying that comparator per key to precision, instead of to skipping;
  - the two-sided quantization margin;
  - the z-rule impossibility;
  - split-K containment and its counterexample;
  - the δ certificate;
  - the statements and proofs here.
