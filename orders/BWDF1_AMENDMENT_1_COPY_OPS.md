# BWD-F1 Amendment 1: multi-operand copy ops trust the first operand's shape (Part C)

Follows: BWDF1_ZERO_EXTENT_AND_AUDIT_FIXES.md (this branch)
ID reuse: BWD-F1 (this amendment adds Part C to the same round)

This amendment adds one part to the round. Everything in the order stands; where this file adds to a list, the order's
list is extended, never narrowed.

## Why

Voice's read-only review (KV-RV1, `/mnt/ForgeRealm/Project-Voice/artifacts/rv1/RV1.md`, finding F1), after the lead's
zero-extent FYI, found that `cat_nd` checks nothing about its later operands. The lead's reading at `fbd335ae7`
(`tensor_cuda/src/kernels.cu:1513-1535`, unchanged from production `f7282b1d`) confirms it:
- the output shape, dtype and device come from `arrs[0]`; only the cat dimension is summed;
- each operand is then copied with `dimcopy_kernel` using its own shape, with no check that its rank, its non-cat
  extents, its dtype or its device match `arrs[0]`, and no check that `dim` is in range;
- neither the binding (`bindings.cpp`, `m.def("cat", ...)`), `ops::cat` (`ops.cpp:1230-1233`) nor the Python `cat`
  (`__init__.py:1398`) checks them.
A later operand with a larger non-cat extent writes past the output's rows, and past the end of the buffer for the last
one. A first operand with a zero extent gives an empty output that later operands write into. That is reading, not
measurement; C3 measures it. `stack`, `splice_rows` and `evict_rows` are the same kind of operation and are unread.

## Ruling (binding)

- **C1, census.** Add to `CENSUS.json` every engine op that copies several operands into one output, or one operand into
  a region of another (`cat_nd`, `stack`, `splice_rows`, `evict_rows`, and any others found), with file:line, every
  precondition its index arithmetic relies on (rank, each extent, dtype, device, `dim` range, offsets and counts), and
  which of them it checks today.
- **C2, the checks.** Each such op refuses, on the host before any allocation or CUDA call, every operand whose rank,
  non-copy extents, dtype or device differ from what its index arithmetic assumes, and any `dim`, offset or count out
  of range, with `RuntimeError` naming the op, the operand index and the shapes. Accepted inputs give results bitwise
  equal to Build B (G4). Zero-size operands follow NumPy's `concatenate` and `stack` rules where the op has a NumPy
  analogue; otherwise the census states the rule used.
- **C3, gates.**
  - G1 (CPU) gains a refusal row per C2 term per op, with ENG3's CUDA call counter showing no CUDA call.
  - G2 gains a staged memcheck line: Voice's two cases for `cat_nd` (first operand `[1,512,0]` with later `[1,1,1]`,
    cat on dim 1; later operands one frame longer than the first) and one mismatch case per other C1 op. On Build B
    record what happens; on Build E each refuses before any launch and memcheck reports 0 errors.
  - G5 gains `m05_cat_no_extent_check` (`cat_nd`'s extent check removed; G1 row) and `m06_cat_no_dtype_check` (the
    dtype check removed; G1 row).
- **Writable set, extended:** in `tensor_cuda/src/kernels.cu`, `ops.cpp` and `bindings.cpp`, the C1 ops' host-side checks
  and helpers for them.
- **Ceiling:** the round's ceiling becomes 7 hours.

## Mechanism & scope

- **Measures:** whether every multi-operand copy op refuses operands its index arithmetic cannot serve before any
  device work.
- **Mechanism doc:** `/mnt/ForgeRealm/Project-Voice/artifacts/rv1/RV1.md` (finding F1) and this branch's
  `tensor_cuda/src/kernels.cu:1513-1535`.
- **Exercises:** C1's ops through the public API, CPU refusal rows, the staged memcheck line, two more mutants.
- **Does not cover:** new concatenation semantics (no broadcasting across non-cat dimensions); performance.

## Registration checklist (order-lint)

- **L60:** C2's refusals go into `BOUNDARY.json` like Z2's: each term, its rows, and Build B's behaviour for the same
  operands.
- **L81:** C3's Build B expectations come from the lead's and Voice's readings (source records: function, line range and
  the shapes named above; nothing run); G2 records the measurement.
- **L90:** Voice's `grad.py:301` call is a consumer site, report-only.
- **L99:** the lead read `kernels.cu:1513-1535` at `fbd335ae7` and `f7282b1d`, `ops.cpp:1230-1233` and
  `__init__.py:1398-1399`, before sealing this amendment.
