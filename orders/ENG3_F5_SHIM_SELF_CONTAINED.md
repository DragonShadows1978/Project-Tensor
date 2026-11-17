# ENG3-F5: make the shipped no-CUDA-call test self-contained (it needs an untracked compiled shim) (Codex; tests only)

Follows: ENG3_F4_REV4_TEST_FIXES.md
ID reuse: ENG3-F5 is the fifth fix round of the same ENG3 change, found by the merged-main check

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_eng3` (branch
`eng3`, at the commit carrying this order; the promoted tip was `7519eb4b`). Edits and CPU runs are AUTHORIZED, limited
to:
- `tensor_cuda/tests/test_eng3_host.py`, only `test_g1_cpu_no_cuda_calls` and its shim path (the `SHIM` constant);
- new `tensor_cuda/tests/support/cudart_count.c`, a byte copy of `artifacts/eng3/tools/cudart_count.c`;
- new `scripts/eng3f5_*.py`, `artifacts/eng3f5/` (new), `logs/eng3f5_*`;
- scratch under `/tmp/eng3f5_*`. Point `TMPDIR` there.

**Read-only:**
- every product file (no rebuild; Build H is `ea723385…`);
- `artifacts/eng3/` (the tracked `cudart_count.c` stays where it is, unchanged);
- every other worktree;
- production.

Never walk `/`, `/mnt` or `/mnt/Shared` recursively.

## Why

MAINCHK1 found the defect: the merged-main check, in `tensor_fork_apa_wt_mainchk1/artifacts/mainchk1/`.
- `test_g1_cpu_no_cuda_calls` asserts that `artifacts/eng3/tools/cudart_count.so` exists, then LD_PRELOADs it. That
  `.so` is a build output and is not tracked.
- On a fresh checkout the test fails before it tests anything. The engine's own suite must not depend on an untracked
  binary.

## The fix (ruled)

- The test compiles the shim from `tensor_cuda/tests/support/cudart_count.c` into its pytest tmp dir, then preloads
  that.
  - Use the compile command recorded in ENG3's receipts under `artifacts/eng3/` (grep for `cudart_count`). If none is
    recorded, use `cc -shared -fPIC -O2 -o <tmp>/cudart_count.so cudart_count.c -ldl`. Record which you used.
  - If no C compiler is available, `pytest.skip` with that reason.
  - A compile failure fails the test, with the compiler's output.
- The support source is a byte copy of the tracked tool source, with the same sha256. Record both.

Nothing else changes.

## Gates

- **G1, CPU** (`CUDA_VISIBLE_DEVICES` empty, Build H): `test_g1_cpu_no_cuda_calls` passes, compiling its shim in its
  tmp dir.
  - Run once in a fresh copy of the test tree with no `artifacts/eng3/tools/cudart_count.so` anywhere on the search
    path.
  - Run once in this worktree.
- **G2, regression:** the CPU suites `test_eng3_host.py` and `test_eng3_f3.py`, per test ID, against F4's CPU receipts.
  Identical, except this one test's mechanism, listed.
- **G3, power:** with the shim built from a copy of the source whose counter never increments (a mutant built in
  scratch only), the test's positive control must fail. That shows the compiled shim is the one in use.
- **G4, scope:** writes only in the writable set; no product change; no git writes; nothing deleted; no GPU.

**Registered acceptance:** ACCEPT when G1 to G4 pass. This is a test-only change, verified by the lead's merged-main
check (MAINCHK1 r2), with no further review.

## Mechanism & scope

- **Measures:** whether the shipped no-CUDA-call test runs from a fresh checkout with no untracked binary.
- **Mechanism doc:** `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_mainchk1/artifacts/mainchk1/REPORT.md`
  (the finding), with `artifacts/eng3/REPORT.md` (the shim's purpose).
- **Exercises:** the test's compile step; its preload; the fresh-tree run; the power mutant.
- **Does not cover:** any other test; product code; GPU.
- **Operand census:** the `SHIM` constant (line 46), its use (lines 501-502), and the support source.
- **Baseline isolation:** Build H is the module, and F4's CPU receipts are the reference.
- **Required evidence:** the diff; G1 to G4 receipts; `REPORT.md`.
- **Forbidden configuration:** any product file change; any other test edit; moving or editing `artifacts/eng3/tools/`;
  GPU use; git writes; deletes.

## Seat

Codex gpt-6.1-sol, high effort.

**Rules:** read `/mnt/Shared/HOUSE_RULES.md` and `/mnt/ForgeRealm/Project-Tensor/AGENTS.md` first. This order is
immutable.
- No git writes, no subagents, no background waits; every command under a `timeout`.
- **No deletes:** never `rm`, `rmdir`, `unlink`, `git clean`, `find -delete` or an inline delete call. Leave and list.
- CPU only.
- **Ceiling:** 2 hours. Your final message is the report; also write `artifacts/eng3f5/REPORT.md`.
- **Prior art:** follow `AGENTS.md`.

## Registration checklist (order-lint)

- **L01:** pytest, python3 and the C compiler, each run with `--version` first.
- **L03:** n/a. No share is gated.
- **L08:** no harness is copied; only the shim source is (byte copy, sha256 recorded). The one literal that changes
  is the `SHIM` path.
- **L09:** G2 compares per test ID. Logs and timings are never compared.
- **L10:** n/a.
- **L11:** n/a. No binary-unchanged claim; the module is not rebuilt.
- **L12:** inherited: F4's per-ID identity, with the one listed exception.
- **L16:** the cheapest pass would skip the test when the `.so` is missing. Excluded: G1 requires a pass in a fresh tree,
  and a skip is allowed only when no compiler exists.
- **L19:** n/a.
- **L32:** n/a.
- **L35:** no gate may be narrowed.
- **L38:** only `test_g1_cpu_no_cuda_calls` and the `SHIM` constant are edited.
- **L40:** pytest runs with `-p no:cacheprovider`, `--basetemp` and `TMPDIR` under `/tmp/eng3f5_*`.
- **L41:** G3's mutant must be the shim the test loads, shown by the counter staying at zero on the positive control.
- **L47:** the expected outcomes come from this order and F4's receipts.
- **L56:** (a) tool slips go under Deviations; (b) at most 3 repairs of your own scripts.
- **L58:** CPU commands run with `CUDA_VISIBLE_DEVICES` empty.
- **L65:** no deletes.
- **L69:** one reference: CPU test outcomes come from F4's receipts (G2).
- **L75:** `artifacts/eng3/tools/cudart_count.c` stays in place and readable. Only a copy is added.
- **L82:** `grep -n 'SHIM' tensor_cuda/tests/test_eng3_host.py` gives lines 46, 501 and 502, the edit sites.
- **L86:** the positive control in `test_g1_cpu_no_cuda_calls` (a device attempt) must fire on the compiled shim, a
  nonzero count, in G1. G3's mutant must silence it.
- **L99:** the lead's readings:
  - MAINCHK1's report names the failing ID and the missing `.so`;
  - `git ls-files artifacts/eng3/tools` lists `cudart_count.c` and no `.so`.
- **L103:** the module is Build H, `tensor_cuda/build-eng3f3/source_H/…`, sha256 `ea7233854f0d…4e43`, built from
  `c42d0880`'s product sources (unchanged since). It is checked before use.
- **L104:** n/a.
- **L109:** helpers run as `python3 -B -P`; none shares a standard-library module name.
- **L111:** n/a.

## Done

Your final message gives, verbatim:
- G1 to G4, each PASS or FAIL;
- the diff summary;
- both source sha256s;
- the deviations;
- every file written.
