# ENG2-A2: fix ENG2-REV2's two MINOR residuals (CPU round; restage)

**Dispatched 2026-11-14** on branch `eng2`, at the commit carrying this order (after REV2's receipts, `9406b0e4`).
`orders/ENG2_DTYPE_REFUSAL_AND_HALF_NORMS.md` and `orders/ENG2_A1_REV1_FIXES.md` stand unchanged except for the fixes
below.

## The findings (`artifacts/eng2/rev2/findings.json`, sha256 `45301021…772d`; `REPORT.md`)

1. **E2R2-F01 (MINOR).** In `make` (`tensor_cuda/tensor_cuda/__init__.py:70-72`), a list or tuple shape is normalized
   (`tuple(shape[0])`) before the dtype is checked. A shape object with side effects runs, and can draw from NumPy's
   RNG, before the registered TypeError.
   - **Fix:** in `make`, check the dtype (`_np_dtype(dtype, name=True)`, then the name test) before any shape handling.
     Keep the registered TypeError and the supported-dtype behaviour.
   - **Test:** REV2's shape-iterator sentinel case, in `test_eng2.py` (additions only).
2. **E2R2-F02 (MINOR).** The census (`artifacts/eng2/dtype_census.py`) omits `RegisteredMargins.lookup`
   (`tensor_cuda/tensor_cuda/apa_sp2.py:53`).
   - **Fix:** add its baseline and candidate rows. Its role is metadata validation, with a float32-only contract.
   - Its product behaviour stays unchanged: it already refuses with ValueError.

## Gates

- **A2-G1.** REV2's `dtype_boundaries.py` and `census_review.py`, run from copies under `artifacts/eng2/a2/` (output
  paths rebound only; ledger the diff):
  - the shape sentinel no longer runs, and the RNG state is unchanged for zeros, ones, randn and rand;
  - the census contains `RegisteredMargins.lookup`'s rows.
- **A2-G2.** The candidate passes every CPU ID of `test_eng2.py`. The baseline fails exactly the registered set plus
  the new sentinel IDs. MF1 is still killed on CPU, at its registered line. If the edit shifts a line, re-register
  MF1's hunk and ledger it.
- **A2-G3.** Scope maps show only `make`, `test_eng2.py` (additions), the census script and `artifacts/eng2/a2/`
  changed, plus the re-staged arms and seal. The device-mode dry run and the audit route pass.
- **Closure.** No further review round, per the owner's 11-03 rule and the ENG1 precedent. The lead closes both
  findings by re-running REV2's probes against the result.

## Mechanism & scope

- **Measures:** whether the two REV2 residuals are fixed without changing anything else.
- **Mechanism doc:** `artifacts/eng2/rev2/REPORT.md`.
- **Exercises:**
  - `make`;
  - the census;
  - REV2's probes;
  - the CPU suite;
  - staging.
- **Does not cover:**
  - N1;
  - the C++ defects;
  - the GPU;
  - promotion.
- **Operand census:**
  - the four factories;
  - the census entries;
  - MF1/MF2.
- **Baseline isolation:** `bc02d3a1` is the baseline arm. REV2's receipts are read-only.
- **Required evidence:**
  - the diff;
  - A2-G1 to A2-G3 receipts;
  - the new seal.
- **Forbidden configuration:**
  - changing tolerances, N1 code, expected exits or bounds;
  - editing REV1/REV2/A1 receipts;
  - GPU;
  - git writes;
  - deleting anything.

## Where to work

**Seat:** Claude Opus 5.5, high effort (Agent tool).

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_eng2`. Edits and CPU
runs are AUTHORIZED only in:
- `tensor_cuda/tensor_cuda/__init__.py` (`make` only);
- `tensor_cuda/tests/test_eng2.py` (additions only);
- `artifacts/eng2/dtype_census.py` (additions only);
- the staging files that re-sealing rewrites under `artifacts/eng2/tool/` (as A1 did), plus `gpu_runs/dry_3` and
  `gpu_runs/audit_3` (new);
- everything under `artifacts/eng2/a2/` (new);
- scratch under `/tmp/eng2_a2_*`.

**Rules:**
- No git. No subagents. No background waits. Every command stays under 10 minutes, with `timeout`.
- **No GPU:** dry runs only, with `FORGE_GPU_DRY_RUN=1`.
- **Memory:** any job over 1 GB starts only with at least 12 GB available (`free -g`).
- **No deletes.** Never run `rm`, `rmdir`, `unlink`, `git clean`, `find -delete` or an inline delete call.
- Never kill a process you did not start. Never walk `/`, `/mnt` or `/mnt/Shared` recursively.
- **Your final message is the report.**

## Registration checklist (order-lint)

- **L01:** the instruments are python3 with NumPy and pytest, REV2's scripts, and the tool. Run `--version` first.
- **L06:** n/a. There is no count threshold; every gate is exact set equality on registered IDs.
- **L16:** the cheapest passing fix moves the check but breaks supported-dtype calls. Excluded by A2-G2's full
  candidate pass and the supported-dtype rows.
- **L41:** MF1 and MF2 are shown executing their hunks on CPU, by trace.
- **L48:** staged lines are run as written by the dry run. The only addition is `FORGE_GPU_DRY_RUN=1`.
- **L56:** (a) A slip in running your own tools goes under Deviations: re-run, and continue. (b) At most 3 repairs per
  gate, each ledgered.
- **L60:** the census covers `RegisteredMargins.lookup`'s baseline behaviour for each requested string.
- **L65:** no deletes, as stated.
- **L82:** `grep -n 'def make\|def _factory' tensor_cuda/tensor_cuda/__init__.py` locates the product site. Writable:
  `make`.

## Done

Your final message gives, verbatim:
- the diff;
- A2-G1 to A2-G3 with their receipts;
- the new seal's sha256;
- every deviation;
- a "Prior art" heading.
