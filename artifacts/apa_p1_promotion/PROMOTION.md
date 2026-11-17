# APA-P1 promotion record

**Change.** It adds APA's selective-precision (SP) attention to `tensor_cuda`, with packed key storage as the shipped
default. The diff from production main `2478a4d` is 30 files in `tensor_cuda/` (+7,736/−2), plus receipts under
`artifacts/` and the new section 8 of `docs/APA.md`.
- **The port:** `tensor_cuda/tensor_cuda/apa_sp2.py`, `apa_packed.py` and the native SP kernels under
  `tensor_cuda/src/apa_spo/`, with their bindings and tests.
- **The packed default** (APA-P1-D): 4-bit codes, per-key norms and fixed tables, with no K-shaped dense tensor.
  - `storage="dense"` or `TC_APA_SP_STORAGE=dense` keeps the dense representation.
  - An ineligible input falls back to dense and records why.
- **Fix rounds:** F1 to F3 fix the review findings: NaN handling, misaligned bases, norm overflow, and the SASS check.

**Scope.**
- **Covers:**
  - the SP port;
  - the packed default and its margins;
  - the non-finite contract in `docs/APA.md` §8;
  - F1 to F3.
- **Does not cover:**
  - prefill on packed keys (refused with a `ValueError`; the packed prefill kernel, APA-P1-PF2, is in testing);
  - decode or prefill speed;
  - the memory-saving tiered design (APA-MX6);
  - installing a rebuilt engine binary;
  - DS1.

**Review: cross-model on every authored part.**
- The P1 port was written by Codex. Claude reviewed it blind (REV1, `08b36d09`).
- The packed default (P1-D) and the fix rounds F1, F2 and F3 were written by Claude Opus 5.5. Codex reviewed them:
  - REV2 (`71c77cb7`) covered P1-D and F1;
  - REV3 (`73e8dc7c`) covered F2;
  - REV4 (`ed0e2dfa`) covered F3.
- The rounds found 12, 8, 3 and 1 findings.
  - Every one is fixed, except REV4's P1R4-F01. That finding is closed by narrowing the contract (`docs/APA.md` §8,
    clause 4; lead ruling, ledgered).
  - REV4 found no new defect in F3's diff.
- See `review_receipt.json`.

**Gate evidence:**
- F3 ACCEPT, G1 to G7 pass. SASS: 781/781 kernels are equal, with 3 intended changes. The lead's own check is
  `artifacts/apa_p1f3/lead_f3_sass_check.json`.
- Mutants killed in each round.
- Finite identity chains bitwise from round to round (B to F to F2 to F3).
- **QG0-R on Build F3 (APA-P1-QR, `3eca27e1`):** the ratio of KL to the floor is 0.884 and 0.878 on 1.7B, and 0.956
  and 0.924 on 4B, against a bar of ≤ 1.50. Top-1 agreement is within the slack. QR2's carry-over is bitwise.
- **Build F3 module:** `2e557a23eaf228512a2f1a2f0629401d91d02d05705087661be5c48849c0f3e9`.

**Approval.** The owner approved this merge via Ember on 2026-11-17, replying A to the decision request
`20261117-015200_Project-Tensor_p1-promote.md`: merge P1 now, with packed prefill following as its own merge. The
verbatim text is in the local promotion ticket and the lead's register. The approval came with a condition: keep the
APA memory goal in view. That goal is full precision on the top-scoring keys and a quantized tail in storage, and it is
being built as APA-MX6.
