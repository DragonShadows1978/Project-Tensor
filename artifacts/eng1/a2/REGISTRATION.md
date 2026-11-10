# ENG1-A2 registration — immutable, before gates

Order: `orders/ENG1_A2_REV2_FIXES.md`, unchanged. Dispatch:
`e73fafdacdf5d542bb6e9d6b614b90563f11bd38`. Scope diff base: `7cf46bd5`.
The complete 296-file `tensor_cuda/` SHA256 map was computed before the first
filesystem write and saved as `tensor_cuda_before.json`. Earlier receipts and
seals are pinned in `protected_before.json`; tooling, staged arms and tracked
files have separate initial maps. Existing untracked files stay untouched.

## Fixed acceptance criteria

1. Add exactly REV2's six missing selectors (seven exact IDs), preserving the
   original sixteen; list their source paths in JSON and an A2 table.
2. Execute unchanged `stage.py` census: candidate 2133 IDs, comparison **2110**,
   exclusions 23 IDs under 22 selectors; both arms keep their existing pins.
3. Rerun REV2's two trace/classification scripts, byte-identical A2 copies, on
   the same three groups spanning all 49 regression files / 2133 IDs. Require
   **zero observed readers absent from the new list**. Retain trace pytest
   failures and blocked operations as instrument outcomes, not regression gates.
   List selectors not observed by the CPU trace and explain each.
4. In E3 row measurement only, add the order's exact NumPy nonfinite-pattern
   comparison and maximum jointly finite distance (null for no finite pair).
   Preserve existing all-elements distance semantics and numerical targets.
   CPU checks exercise the actual four nonfinite emulation operands, mixed
   finite/nonfinite distances, sign/position/NaN distinctions and row emission.
   Engine E3 execution remains unperformed because it requires GPU.
5. Publish a new seal, preserving every old seal. Execute one candidate_eng1
   line with the existing CPU stand-in and supervisor. Count success only from
   its completed-child parent receipt: child 0, parent 0, no timeout, 47 IDs.
6. Independently run candidate CPU subset: 47 passed, no failures/skips, exact
   ID set equal to `ids_a1_audit.txt` after normalizing staged prefixes.
7. Require scope diff only `tensor_cuda/tests/test_eng1.py`; entire tensor_cuda
   map differs only at that file; src/ byte-identical; protected receipts,
   old seals, engine/package bytes and unrelated tracked files unchanged.

All runs: CUDA_VISIBLE_DEVICES empty; thread counts 1; Python bytecode writes
disabled; one foreground command under timeout, nice 19 for CPU gates. Scratch
only `/tmp/eng1_a2_*`. No git writes, subagents, deletes, builds or GPU calls.
If an acceptance criterion remains unmet, report REJECT and preserve receipts.
Budget: 90 minutes from dispatch work; no post-result threshold changes.

## Output routing and staged selection copies

The old instruments hard-code prior output locations and A1 seal selection.
A2 adapters may change runtime path constants only: the trace scratch root,
the seal/order selection and the audit output root guard (restricted to the
new A2 audit directory). Instrument code, read attribution, classification,
origin checks, pin checks, child/parent evaluation and thresholds stay intact.
Use byte-identical REV2 copies in A2 so their __file__-relative outputs land
there. Seal these adapters as registration inputs.

`stage.py arms` refreshes only staged copies of this same `test_eng1.py`, as
required by its seal's equality assertion. These are staged selection files;
no other staged test, package or module changes. Census files retain their
instrument-defined `*_a1_*` names in tool/. This order explicitly authorizes
those staged files; A1/review directories and old seals remain immutable.

## Prior art

REV2 (2026) q2_source_readers.py / q2_classify_reads.py supply the unchanged
trace and classifier. PEP 578 (Steve Dower, 2019) supplies audit hooks;
pytest supplies runtest attribution/deselection. NumPy supplies isfinite,
where, array_equal(equal_nan=True), boolean masking and max absolute distance;
the exact equality expression and requirement are REV2 R2-F02. ENG1/GN1 A8
(2026) supply staged seals and completed-child parent receipts; SHA256 follows
NIST FIPS 180-4 (2015). Taken: existing instruments and the ordered mechanical
fixes; new here: list completion, added measurement fields and A2 output routing.
No new engine algorithm. External literature unverified — lead to check:
`PEP 578 Dower 2019`, `NumPy array_equal equal_nan isfinite`, `FIPS 180-4 2015`.
