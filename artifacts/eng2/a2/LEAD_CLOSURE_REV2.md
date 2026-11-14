# ENG2-REV2 closure by the lead (2026-11-14): E2R2-F01 and E2R2-F02 closed

This closure follows the owner's 11-03 rule and the ENG1 precedent: one fix round, closed mechanically, with no
third review round.

- **E2R2-F01** (factory shape conversion before the dtype guard): closed.
  - The lead re-ran A2's copy of REV2's `dtype_boundaries.py`, with assertions inverted to the fixed condition: rc 0.
  - The lead also re-ran the paths-only copy, `dtype_boundaries_asis.py`, which is REV2's unmodified defect logic:
    rc 1, at REV2's own assertion that the shape sentinel runs before the dtype guard. The defect no longer
    reproduces.
  - Both runs: CPU, CUDA hidden, at 11-14 ~12:1x. Their output rewrote the copies' own `a2/rev2_out/` files.
- **E2R2-F02** (census omits `RegisteredMargins.lookup`): closed. A2's census has 25 lookup rows per arm, with baseline
  rows equal to candidate rows (A2-G1, `census_review.py` rc 0). The lead's rerun reports `lookup_source_equal: true`.
- **Recorded, not a finding:** `lookup` returns the fp32 margin for str-equal non-strings (`UserString("float32")`),
  in both arms. Its production caller passes `q.dtype`, which is a string.

**Next:** the lead's G5 on the GPU under a fresh `g5_*` root (seal `a9175ff9…df28f`), after GRAPA's current GPU hold.
