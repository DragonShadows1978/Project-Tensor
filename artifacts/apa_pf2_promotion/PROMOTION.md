# APA-PF2 promotion record

**Change.** This adds packed prefill to `tensor_cuda`'s APA selective-precision route, and fixes a non-finite case in
packed decode. The assembly (`4e7311536`) is production main `3138cf54` plus the reviewed tip `apa-pf2f3` for every
path PF2 changed since P1-F3 (`2945c3e9`). PF2's `bindings.cpp` hunk is applied over ENG3's. The resulting
`tensor_cuda` tree equals the reviewed tip plus ENG3's own diff, line for line.

What it contains:
- **The kernel:** `packed_prefill.cuh`, the packed prefill kernel. `apa_packed.py` routes prefill (L > 1) on packed
  keys to it, instead of refusing with a `ValueError`.
- **The guards:** in `packed.cuh` (both loads packed decode reaches), `spq4.cuh` and `spq6.cuh` (the raw `tc` path), a
  raw non-finite stored norm becomes a quiet NaN at the load. That keeps `docs/APA.md` §8 clause 3. Before this,
  packed decode dropped a key whose stored norm was a raw +Inf. Only a direct write to the store can produce one: the
  append path stores NaN.
- **The docs:** `docs/APA.md` §8:
  - packed prefill is now served, with its equality to decode stated;
  - clause 1 is qualified for non-finite sinks;
  - the not-covered list is updated.
- **The tests and receipts:** `test_apa_pf2_nonfinite.py`, and the receipts of PF2, PF2-F1, PF2-F2 and PF2-F3.

**Scope.**
- **Covers:**
  - packed prefill on packed keys;
  - the non-finite contract on the packed prefill and decode paths;
  - the `tc` path's norm loads.
- **Does not cover:**
  - prefill or decode speed (tracked as APA-SP-PREFILL-SPEED);
  - grouped reuse;
  - the memory-saving tiered design (APA-MX7);
  - installing a rebuilt engine binary;
  - model-level quality. Packed prefill's decisions equal packed decode's with `carry="row"`, and its outputs match
    within tolerance.

**Review: cross-model on every authored part.**
- **The PF2 kernel** was written by Claude. Codex reviewed it blind (PF2-REV1, `42b818fd`), finding 2 MAJOR and 3
  MINOR. The lead reproduced both MAJORs on GPU (16/16 each).
- **The fix rounds** PF2-F1, F2 and F3 were written by Codex. Claude reviewed them:
  - PF2-REV2 (`419bc3699`) found 2 MAJOR, 4 MINOR and 2 NIT, all in tests and docs; the engine guards were correct;
  - its re-check (`fb2777b33`) closed all eight, and raised one doc MINOR (fixed, `f480b661c`) and one NIT (closed:
    the test's helper in `scripts/` ships with it).
- Every finding is fixed or closed with a recorded reason. See `review_receipt.json`.

**Gate evidence** (lead GPU lines):
- **PF2:** every registered gate passes (`abb1770c`).
- **PF2-F1:** 56 of 58 lines pass. The other two failed on the production decode defect, and PF2-F2 and F3 supersede
  them.
- **PF2-F2:** 62 of 62 (`8c35792e`). This includes finite decode (752 cases) and finite prefill (4,848 cases), each
  bitwise equal to the previous build.
- **PF2-F3:** 69 of 69 (`497df5ab9`):
  - the shipped test file passes 77 of 77 on GPU, and passes in a fresh tree on CPU;
  - all three mutants are killed;
  - P1's regression passes (643 gated).
- **Mutants killed across the rounds:** 2 in F1 (sink and index), 2 in F2, and 3 in F3.
- **Build F3 module:** `1036ec930a1e15d6c21213d977be22fd7c3846814d783f8fea40949ebb8f2347`.

**Before push:** a merged-tree check (MAINCHK3) builds this tree and compares it per test ID with both parents. Both
CPU and GPU must pass.

**Approval:** the owner's verbatim approval is recorded here when given.
