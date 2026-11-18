# APA-PF2-F2 r2 (Amendment 1): guard every stored-norm load the packed decode dispatch reaches, not only line 203 (Codex; build and CPU; GPU lines staged)

Follows: APA_PF2_F2_DECODE_NORM_GUARD.md
ID reuse: r2 is the second round of the same APA-PF2-F2 fix; Amendment 1 is its first amendment

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_pf2f2`
(branch `apa-pf2f2`, at the commit carrying this order). The PF2-F2 order's writable set, read-only set, gates, rules
and checklist hold, except as below.

## Why

r1 stopped correctly at its mechanism check, with zero edits (`artifacts/apa_pf2f2/REPORT.md`). The packed decode entry
(`apa_packed.py:547-549`, `gqa_path="r2"`) reaches two kernels:
- `spq1::bulk_shared`, for G = 4 and 8 (`launcher.cuh:266-278`). It loads the stored norm into a scalar at
  `packed.cuh:149` and multiplies at line 178;
- `spq1::bulk`, for the other G values. It loads at line 203.

Eight of the 16 failing `G2_norm` cases are G = 4. The order authorized line 203 only. The lead's census grepped
`norms[` and missed the scalar load: a lead fault.

## Amendment 1 (lead; registered before any r2 step runs)

- **A1.1, edit sites.** Trace the packed decode dispatch from the public entry, for every G, D, dtype and TQ setting
  that `apa_packed.py`'s decode can request. List every stored-norm load it reaches in `DISPATCH_TRACE.json`, with
  file:line and the kernel instantiation.
  - Each such load is authorized for one edit: a non-finite value becomes a quiet NaN at the load, as PF2-F1 did in
    prefill.
  - The census command is `grep -n 'scales\[' tensor_cuda/src/apa_spo/*.cuh`. It gives hits in `packed.cuh` (22, 149,
    203), `spq4.cuh`, `eb3.cuh`, `eb4.cuh`, `spq6.cuh` and `eb2b.cuh`.
  - A hit the trace does not reach stays read-only. List it with the reason (another route, or a store write).
- **A1.2, mutants.** One revert mutant per edited load: `pf2m01_bulk_norm_finite_only` (line 203) and
  `pf2m02_shared_norm_finite_only` (line 149), plus any further edited load. Each killing test runs the G2 norm cases
  that reach that kernel, and must fail.
- **A1.3, SASS:** against Build F1, only the edited kernels' instantiations change.
- **A1.4:** nothing else changes. G1's finite identity covers both kernels' instantiations. The repair allowance
  restarts at 3. The wall is 6 hours from this run's launch.

## Mechanism & scope

- **Measures:** whether packed decode keeps docs/APA.md §8 clause 3 for a raw non-finite stored norm on every kernel
  its dispatch reaches, with finite outputs bitwise unchanged.
- **Mechanism doc:** `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_pf2f2/artifacts/apa_pf2f2/REPORT.md`
  (r1's dispatch evidence), with the PF2-F2 order and docs/APA.md §8.
- **Exercises:**
  - the dispatch trace;
  - each reached norm load;
  - Build F2 and one revert mutant per edited load;
  - the decode and prefill non-finite cases;
  - the staged lines.
- **Does not cover:**
  - the append and packing writes;
  - other APA routes not reached by packed decode;
  - prefill speed;
  - promotion.
- **Operand census:** A1.1's command and hits; the dispatch trace; the 16 G2 norm cases.
- **Baseline isolation:** Build F1 is the finite reference; clause 3 is the non-finite reference.
- **Required evidence:**
  - `DISPATCH_TRACE.json`;
  - the diff;
  - the build and mutant receipts;
  - the CPU results;
  - `STAGED_LINES.md`;
  - `REPORT.md`.
- **Forbidden configuration:**
  - edits beyond the traced loads and the test file;
  - GPU use;
  - git writes;
  - deletes.

## Registration checklist (order-lint)

- Every answer is the PF2-F2 order's, unchanged, except as below.
- **L16:** the cheapest pass would guard only the kernels REV1's cases reach. Excluded: A1.1 traces every
  configuration the decode entry can request, not only the reproduction cases.
- **L35:** A1.1 to A1.4 are the lead's amendment. No gate is narrowed.
- **L41:** each mutant's killing test reaches its kernel, shown by the dispatch trace and a counter or the G
  configuration.
- **L59:** the amendment is committed before launch. r2's start record is the `SHIM-RUN` line of
  `logs/apa_pf2f2_r2.log`.
- **L82:** the census is A1.1's command and hit list. The edit set is the dispatch trace's loads, and every other hit
  stays read-only with its reason.
- **L99:** the lead's readings: r1's report (`launcher.cuh:266-278`; `packed.cuh:149`, `:178` and `:203`), and the
  16 `G2_CASE` lines, 8 of them G = 4.

## Done

As the PF2-F2 order's Done, plus `DISPATCH_TRACE.json` in full.
