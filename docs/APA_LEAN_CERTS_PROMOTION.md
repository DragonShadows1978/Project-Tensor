# Promotion record: APA certificates in Lean (`lean/apa_certs`)

Promoted 2026-10-28 into `main` under the infrastructure rule (fork, test, reclaim, promote). The author approved it the
same day; the approval is quoted in the local promotion ticket. The review receipt is
`artifacts/apa_lean_certs_promotion/review_receipt.json`.

## Scope

**Covers:**
- `lean/apa_certs/`: a Lean 4 project (`leanprover/lean4:v4.34.1`, Mathlib `d13f23b`) with five APA claims, their
  proofs, a check file, and a README;
- the README paragraph that points to it;
- the X5 orders, reports and receipts (`orders/TX3_X5*`, `docs/TX3_X5*`, `artifacts/x5*/`);
- this record.

**Does not cover:**
- **No engine code.** `tensor_cuda/` is unchanged, and no runtime path changes.
- **Floating point.** The statements are over the real numbers.
- **The kernels.** How the CUDA kernels correspond to the rules is checked by reading, not by proof.

## Evidence

| Check | Result |
|---|---|
| Build | `lake build ApaCerts.Check` exits 0 (8,926 jobs). The lead rebuilt it on 2026-10-28 |
| Axioms | Each of the ten theorems depends only on `propext`, `Classical.choice` and `Quot.sound`. No `sorry` |
| Statement-first | `Spec.lean` was written by the lead before any proof. `Check.lean` fails the build unless each proof has exactly its statement's type |
| Cross-model review 1 | TX3-X5-REV1 (gpt-6.1-sol): S1, S2, S4 and S5 faithful. Two findings: the split-K counterexample lacked the original test's sink (MAJOR); one non-vacuity witness missed its antecedent (MINOR) |
| Amendment | TX3-X5B: both statements amended as the review proposed, and the proofs updated |
| Cross-model review 2 | TX3-X5-REV2 (gpt-6.1-sol): S1–S5 faithful, all five witnesses adequate, both findings closed, APPROVE |
| Reviewed file hashes | `Spec.lean` `36aeba97…`, `Proofs.lean` `6ee2f428…`, `Check.lean` `2de6d3b0…` (full hashes in the receipt) |

## Held back

These receipts are kept in the lab fork and were not promoted:
- `artifacts/x5_rev1/lake_before.json`, a 24.7 MB Lake metadata dump;
- `artifacts/x5_rev2/source_excerpts.txt`, a reviewer's excerpt copy of internal notes.
