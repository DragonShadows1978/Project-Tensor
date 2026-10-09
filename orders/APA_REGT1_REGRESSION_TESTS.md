# APA-REGT1: port three verified detectors to production as regression tests

David, 2026-10-09, verbatim: "On things like regression tests going into production, that's fine unless it really
significantly hurts performance."

## Background

Three detectors from this week's fork work caught real or seeded bugs. None of them is on production main a28e28e.

1. **The ZWS statistics tag (APA-LD2 G3).** It flags the fd32b6d statistics mask (PYSTAT) on 9/9 rectangular causal
   cells and the CUDA statistics-range mutant CU7 on 7/7 fused cells. Main passed 17/18, and the one exception, B04
   (the square control, L = S), was REFUSED-operand. The cause was LD2's ghost-block rule putting t_g = 0 at L = S, a
   harness defect. An exploratory rerun with a corrected ghost block passed on main and on PYSTAT.
   - Report: `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_ld2/docs/APA_LD2_REPORT.md`.
2. **The frontier-first leak probe and the PDWL stagger cuts (APA-LD1/LD2).** These are the frontier-first cut
   placement plus the stagger cuts that catch single-row and window leaks. Both kill 30/30 real small-leak cells with
   0 false alarms.
   - Reports: `…/tensor_fork_apa_wt_ld2/docs/APA_LD2_REPORT.md` and `…/tensor_fork_apa_wt_ld1/docs/APA_LD1_REPORT.md`.
3. **The OM135 exact-integer backward-orientation checks.** These are the IMM derivative self-reduction and the
   integer textbook-formula check. They killed 96/96 live mutant triples with 0/540 false alarms on main.
   - Report: `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_om135/docs/APA_OM135_REPORT.md`.

The sources are on fork branches:
- `apa-ld2` at cb2fac1: `scripts/apa_ld1_pdwl.py`, `scripts/apa_ld2_pdwl.py`, `tensor_cuda/tests/test_apa_ld1_pdwl.py`,
  `tensor_cuda/tests/test_apa_ld2_pdwl.py`.
- `apa-om135` at 2bbee82: `scripts/apa_om135_core.py`, `tensor_cuda/tests/test_apa_om135_imm.py`.

Copy them with `git -C /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa show <commit>:<path>`. Known port issues:
- `test_apa_om135_imm.py:23` hardcodes a fork build path (`MAIN_BUILD = …/tensor_fork_apa_wt_mf2/…`).
- The B04 ghost-block defect above.

Production tests already put `scripts/` on `sys.path`, as in `tensor_cuda/tests/test_apa_mf2_2a_blocked.py`. Follow
that convention.

## Mechanism & scope

- **Measures:**
  - (a) The ported tests pass on the production engine (main a28e28e).
  - (b) Each ported engine-level test still FAILS on at least one known-bad engine or package from the receipts. This
    proves the port did not neuter it.
  - (c) The wall-time cost the new tests add to the default suite.
- **Mechanism doc:**
  - `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_ld2/docs/APA_LD2_REPORT.md`
    (ZWS; frontier-first; B04)
  - `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_ld1/docs/APA_LD1_REPORT.md`
    (PDWL)
  - `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_om135/docs/APA_OM135_REPORT.md`
    (IMM and integer checks)
- **Exercises:**
  - the new `scripts/apa_ld1_pdwl.py`, `scripts/apa_ld2_pdwl.py` and `scripts/apa_om135_core.py`;
  - the new `tensor_cuda/tests/test_apa_ld1_pdwl.py`, `test_apa_ld2_pdwl.py` and `test_apa_om135_imm.py`;
  - `tensor_cuda.quant.apa_quant_attention`, `functional.scaled_dot_product_attention`, `tc.matmul`, batched and
    broadcast matmul, `nn.Linear` backward, and the SDPA backward, all on the pinned main engine.
- **Does not cover:**
  - any change to engine source (`tensor_cuda/src`, `tensor_cuda/include`, `tensor_cuda/tensor_cuda/*.py`);
  - the runtime speed of the engine (these are tests only);
  - new detectors, or retuning a detector's thresholds beyond what the port needs (the B04 ghost-block fix is the
    one allowed exception);
  - gradcheck or other cost benchmarks inside the default suite;
  - GRM and GRAPA.
- **Required evidence:**
  - the diff stat against a28e28e;
  - each new test's pass on main, with per-test wall time from `pytest --durations=0`;
  - each engine-level test's FAIL on a known-bad build or package, with the exact build path and its sha256;
  - the full suite on the pinned main engine;
  - the added wall time in total, CPU-only and GPU;
  - every place the ported code differs from its fork source, each with its reason.
- **Operand census:** the shapes and cells already registered in the source tests. Keep them, but drop or shrink any
  cell whose cost is out of proportion (see the budget below), and state each drop. The B04 square cell either uses
  the corrected ghost block, and is shown to PASS on main and ALARM on PYSTAT where PYSTAT applies, or is dropped
  with the reason stated.
- **Baseline isolation:**
  - **Main:** the pinned main engine
    `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_mf2/artifacts/apa_mf_main/build`
    (`_tensor_cuda` sha256 `d8fda9a8879b2c0de1b659653e8899aa6fbb89da36af03d1339e2a1c9b011e31`). Its
    `../build_manifest.json` lists source sha256s. Check that they equal this worktree's a28e28e sources, and record
    the check.
  - **Known-bad:**
    - LD2's `artifacts/apa_ld2/mutants/PYSTAT_fd32b6d/pkg` and `mutants/CU7_stats_range/build` (and CU5/CU6 if
      useful), under `…/tensor_fork_apa_wt_ld2/`;
    - OM135's mutant builds under `…/tensor_fork_apa_wt_om135/artifacts/apa_om135/mutants/` where they still exist,
      or its Python-level wrapper mutants.
  - Run one engine per process.
- **Forbidden configuration:**
  - editing engine source or any existing test;
  - widening an exact comparison to a tolerance;
  - a test that passes on a known-bad build it was registered to catch;
  - hardcoded `Fork/` or `WorkTree/` paths in the ported tests or scripts. Mutant paths belong only in your evidence
    commands, never in the shipped files;
  - a GPU test that errors, rather than skips, when no CUDA device is present;
  - the retired 595.84 driver override.

**Budget.** Together, the new tests should add no more than 60 s to the default suite on this machine. If they cost
more, keep the strongest cells inside 60 s, and either move the rest behind an opt-in environment variable
(`TC_SLOW_TESTS=1`, documented in the test module) or drop them. State which you did.

Seat: Claude Opus 5.5, high (an Agent-tool subagent; it has the GPU).

## Where to work

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_regt1`
(branch `apa-regt1` at a28e28e = production main). Edits, CPU runs and GPU runs are AUTHORIZED there, in these paths
only:
- new `scripts/apa_ld1_pdwl.py`, `scripts/apa_ld2_pdwl.py`, `scripts/apa_om135_core.py`;
- new `tensor_cuda/tests/test_apa_ld1_pdwl.py`, `test_apa_ld2_pdwl.py`, `test_apa_om135_imm.py`;
- `docs/APA_REGT1_REPORT.md` (new);
- `artifacts/apa_regt1/` (new; logs and evidence only);
- `logs/`.

Everything else is READ-ONLY. That covers the rest of this worktree, every other worktree (read and import only; the
mutant builds are used in place), the production repo `/mnt/ForgeRealm/Project-Tensor` outside `Fork/`, GRM, GRAPA,
`/mnt/ForgeRealm/reference`, the models and the HF cache.

NO git (the lead commits). NO subagents. NO DELETES: use `delete-me <path> --reason "..." --session apa-regt1`. Never
kill a process you did not start. Run everything synchronously; no background waits or monitor-idling.

**GPU:** every GPU command runs as `flock -w 900 /tmp/forge-gpu.lock timeout 600 …`, on the native driver only
(595.99.02; never set the 595.84 override). If a lock wait times out, retry. Load the engine with
`APA_LD1_ENGINE=<build> source <ld2 worktree>/scripts/apa_ld1_env.sh` and
`PYTHONPATH=$APA_LD1_ENGINE:<this worktree>/tensor_cuda`, or the equivalent; state what you used.

## Required

1. **Read** the three reports' relevant sections and the six source files.
2. **Port.** Copy the files, remove the fork paths, fix or drop B04, make GPU tests skip without CUDA, and keep a
   prior-art note at each detector's code site (it is already in the sources; keep it).
3. **Main.** Run the new tests on the pinned main engine and record per-test durations.
4. **Known-bad.** Each engine-level test fails on its known-bad build or package; give one command and its output per
   detector.
5. **Full suite** on the pinned main engine: `pytest tensor_cuda/tests -q -p no:cacheprovider --ignore=tensor_cuda/tests/test_selector_accuracy.py`.
   The FAILED/ERROR set must equal the 52 baseline ids
   (`…/WorkTree/tensor_fork_apa_wt_mf2/logs/apa_mf1_g1_baseline_unchanged.log`; compare with
   `…/WorkTree/tensor_fork_apa_wt_mf2/scripts/apa_mf2_2_g1_compare.py`), plus the new tests passing. Report the suite
   wall time with and without the new files.

## Gates

- **G1 (port):**
  - `git diff a28e28e --stat` shows only the writable paths;
  - no `Fork/` or `WorkTree/` string appears in the six shipped files;
  - the GPU tests skip cleanly with `CUDA_VISIBLE_DEVICES=""`.
- **G2 (main):** every new test passes on the pinned main engine, and the full suite's FAILED/ERROR set equals the
  52 baseline ids.
- **G3 (teeth):** every engine-level detector fails on at least one known-bad build or package from the receipts.
- **G4 (cost):** the added default-suite wall time is ≤ 60 s, and anything slower sits behind `TC_SLOW_TESTS=1`.

## Rules

HOUSE_RULES (`/mnt/Shared/HOUSE_RULES.md`) apply. RED honesty. Prior Art Directive (AGENTS.md): keep the sources'
annotations. Keep to about 2 hours.

## Done

Write `docs/APA_REGT1_REPORT.md`. If the harness refuses report files from subagents, put the report in your final
message. The final message gives:
- G1–G4;
- the per-test duration table on main;
- the known-bad table (detector, build path, sha256, verdict);
- the full-suite comparison and wall times;
- every port difference from the fork sources, with its reason;
- the files and the diff stat;
- deviations;
- receipts cited as `path:line "quote"`.
