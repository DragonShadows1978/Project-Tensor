# APA-PF2-F3: fix PF2-REV2's findings (shipped tests, the −Inf prefill case, the tc-path norm loads, the prefill mutant kill) (Codex; build and CPU; GPU lines staged)

Follows: none in this directory (it follows APA-PF2-F2, `8c35792e`, and APA-PF2-REV2, `419bc3699`)

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_pf2f3`
(branch `apa-pf2f3`, from `apa-pf2f2` at `8c35792e`, plus the lead's doc edit `1cb328d22`). Edits, builds and CPU runs
are AUTHORIZED there, limited to:
- `tensor_cuda/tests/test_apa_pf2_nonfinite.py`;
- `tensor_cuda/src/apa_spo/spq4.cuh` and `spq6.cuh`, only the stored-norm loads named in F04;
- new `scripts/apa_pf2f3_*.py`, `artifacts/apa_pf2f3/` (new), build directories `artifacts/apa_pf2f3/build_*`,
  `logs/apa_pf2f3_*`;
- scratch under `/tmp/pf2f3_*`. Point `TMPDIR` there.

**Read-only:**
- every other product file, including `packed.cuh`, `packed_prefill.cuh` and `docs/APA.md`;
- `artifacts/apa_pf2f2/` and `artifacts/apa_pf2f1/` here (the earlier harnesses, which you may import or copy);
- `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_pf2rev2/artifacts/pf2_rev2/`
  (the review: `findings.json`, `REPORT.md`, `checks/`);
- every other worktree;
- production.

Never walk `/`, `/mnt` or `/mnt/Shared` recursively.

## Why and rulings

PF2-REV2, a Claude review of the Codex-written PF2-F1 and PF2-F2 fixes, found the engine guards correct and complete on
the public routes. It found the shipped test file broken, and gave 8 findings: 0 BLOCKER, 2 MAJOR, 4 MINOR, 2 NIT. Real
findings are fixed. The lead ruled:
- **PF2R2-F01 (MAJOR), fix.** The CPU index test:
  - compiles into pytest's `tmp_path`;
  - drops the `TMPDIR`-prefix and `CUDA_VISIBLE_DEVICES`-equality asserts;
  - runs whatever the GPU visibility, and skips only if no host compiler exists.
- **PF2R2-F02 (MAJOR), fix.** The shipped tests import `tensor_cuda` as found on the import path, as the engine's
  suite does. They never select a seat arm because a tracked file exists. Arm selection stays in the seat gate scripts.
- **PF2R2-F03 (MINOR), fix.** Add −Inf to `run_norm`'s prefill cases. Then each of the four norm loads has a shipped
  case that kills a guard which misses −Inf.
- **PF2R2-F04 (MINOR), fix the code.** Clause 3 says "for any reason", so the raw `packed_run` binding's `tc` path
  keeps it too.
  - Guard the stored-norm loads at `spq4.cuh:112` and `spq6.cuh:143` the same way: a non-finite value becomes a quiet
    NaN.
  - Lines 110 and 141 are the mutation-hook variants of those loads. Guard them the same way, so a hooks build keeps
    the contract.
  - Finite tc outputs stay bitwise equal.
- **PF2R2-F05 and F07 (MINOR, NIT): done by the lead** in `docs/APA.md` §8 (`1cb328d22`). Nothing to do.
- **PF2R2-F06 (NIT), closed:** an infinite denominator cannot occur (REV2's bound). No change.
- **PF2R2-F08 (MINOR), fix.** Give the prefill norm guard a device mutant kill on top of this round's build. Two
  mutants, each killed by the G2 norm cases including −Inf:
  - `pf3m01_prefill_norm_revert`: both prefill norm guards reverted;
  - `pf3m02_prefill_norm_pinf_only`: guards that handle +Inf only.

## Gates

- **G1, finite identity:** Build F3 against Build F2 (`aefe2513…`), outputs bitwise equal on every finite case:
  - PF2-F2's G1 decode and prefill sets;
  - a finite `tc`-path set covering every `tc` instantiation that `packed_run` can request.
- **G2, non-finite:**
  - REV1's sink and norm sets, the 44 partition cases, and the new −Inf prefill cases, on Build F3. Prefill and decode
    give the same non-finite row sets.
  - A raw ±Inf and NaN norm set on the `tc` path makes every reading row non-finite.
- **G3, mutants:** `pf3m01`, `pf3m02`, and `pf3m03_tc_norm_revert` (the F04 guards reverted). Each is built
  incrementally, and its killing test executes its edit and fails.
- **G4, the shipped test file:**
  - on CPU in a fresh checkout of this branch (a `git worktree` copy is not allowed: use `git archive` of the commit
    into scratch), with the default environment and `CUDA_VISIBLE_DEVICES` unset, it passes, including the index test;
  - on GPU, a staged line runs it with Build F3 on the import path, and every test passes.
- **G5, regression:** SASS against Build F2, where only the `tc` instantiations change; P1's packed decode
  regression (PF2-F2's G4).
- **G6, scope:** writes only in the writable set; no git writes beyond `git archive`; nothing deleted; no GPU.

Every staged GPU line runs `forge-gpu-run --min <n> --max 540` with `--min` ≤ 510. Each has a launcher that sets its
full environment and checks the module sha before the lease, and a dry run with exit 78 and `"would_start": true`.

**Registered acceptance:** ACCEPT when G4's CPU part, G6 and G3's CPU part pass, and the lead's GPU lines pass G1 to G5.
REV2's reviewer then checks the fixes, narrowly.

## Mechanism & scope

- **Measures:** whether the shipped non-finite test file runs and passes in a fresh tree; whether every norm load the
  raw binding can reach keeps clause 3; and whether the prefill guard has a device kill.
- **Mechanism doc:** `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_pf2rev2/artifacts/pf2_rev2/REPORT.md`,
  with `docs/APA.md` §8 at `1cb328d22`.
- **Exercises:**
  - the test file;
  - the four `tc`-path loads;
  - Build F3 and three mutant builds;
  - the fresh-tree CPU run;
  - the staged lines.
- **Does not cover:**
  - the decode and prefill guards, already ACCEPTed in PF2-F1 and PF2-F2 (unchanged here);
  - speed;
  - promotion.
- **Operand census:**
  - `grep -n 'scales\[' tensor_cuda/src/apa_spo/spq4.cuh tensor_cuda/src/apa_spo/spq6.cuh` (lines 110, 112, 141, 143,
    and any others it shows);
  - every test in `test_apa_pf2_nonfinite.py`;
  - the `tc` instantiations `packed_run` can request.
- **Baseline isolation:** Build F2 is the finite reference. Clause 3 is the non-finite reference.
- **Required evidence:**
  - the diff;
  - build and mutant receipts with module sha256s;
  - the fresh-tree CPU run's log;
  - `STAGED_LINES.md`;
  - `REPORT.md`.
- **Forbidden configuration:**
  - any edit outside the writable set;
  - changing `packed.cuh` or `packed_prefill.cuh`;
  - GPU use;
  - git writes beyond `git archive`;
  - deletes.

## Seat and rules

Codex gpt-6.2-sol, xhigh effort.

**Rules:** read `/mnt/Shared/HOUSE_RULES.md` and `/mnt/ForgeRealm/Project-Tensor/AGENTS.md` first. This order is
immutable.
- No git writes, no subagents, no background waits; every command under a `timeout` (builds under 60 minutes).
- **No deletes:** never `rm`, `rmdir`, `unlink`, `git clean`, `find -delete` or an inline delete call. Leave and list.
- Never call `forge-gpu-run` without `FORGE_GPU_DRY_RUN=1`.
- **Builds:** at most `-j4` under `nice -n 19`.
- **CPU jobs:** one thread, its own `RLIMIT_AS` of at most 4,000,000,000 bytes; never a systemd scope.
- Stop below 40 GB free on `/mnt/ForgeRealm`.
- **Wall:** 6 hours. Your final message is the report; also write `artifacts/apa_pf2f3/REPORT.md`.
- **Prior art:** follow `AGENTS.md` (IEEE 754 NaN classes; PF2-F1's guard).

## Registration checklist (order-lint)

- **L01:** cmake, nvcc, pytest and python3, each run with `--version` first. Also run `forge-gpu-run` once in dry-run
  mode (exit 78, `would_start` true).
- **L02:** the fresh-tree CPU run of the test file is the smoke.
- **L03:** n/a. No share is gated.
- **L09:** G1 compares outputs as same-width unsigned views after dtype and shape checks.
- **L10:** G1 is bitwise.
- **L12:** inherited: PF2-F2's G1, G2 and G4 sets.
- **L16:** the cheapest pass would make the tests skip. Excluded: G4 counts passed tests, which must equal collected
  tests minus the GPU tests on a CPU host, and all tests on the GPU line.
- **L19:** compared arrays are checked non-empty and nonconstant first, except structurally constant non-finite rows,
  which are counted.
- **L35:** your registration may fix cases and line splits. It may not narrow a gate.
- **L38:** the test file is the edit target. Its old fixture behaviour (arm selection from a tracked file) is removed
  on purpose. List every test whose behaviour changes.
- **L40:** the earlier harnesses write into `artifacts/apa_pf2f2/` or `apa_pf2f1/` by default. Copy what you reuse into
  `scripts/apa_pf2f3_*.py`, rebind the outputs to `artifacts/apa_pf2f3/`, and list them in `COPY_DEPS.json`.
- **L41:** each mutant's killing test executes its edit.
- **L45:** the lead records each staged line's exits and log sha256.
- **L48:** staged lines run as written, from the worktree root.
- **L55:** finite identity holds by construction: the guard selects the original value whenever it is finite.
- **L56:** (a) tool slips go under Deviations; (b) at most 3 repairs of your own scripts.
- **L58:** CPU commands run with `CUDA_VISIBLE_DEVICES` empty, except G4's fresh-tree run, which leaves it unset.
- **L65:** no deletes.
- **L79:** every line has `--min` ≤ `--max 540` − margin 30. Each dry run shows exit 78, no lease and
  `"would_start": true`.
- **L82:** the census above lists the tc sites. Only those loads and the test file are edited.
- **L99:** the lead's readings, from REV2 (`findings.json` final, sha `20d78a4d…`):
  - F01: 1 failed, 76 skipped under another TMPDIR;
  - F02: the fixture's precheck raises FileNotFoundError for the untracked `.so`;
  - F03: the −Inf witness;
  - F04: `spq4.cuh:112` and `spq6.cuh:143`;
  - F08: no `MUTANT_norm.json`.
- **L103:** each staged line rechecks the module sha before the lease.
- **L109:** helpers run as `python3 -B -P`; none shares a standard-library module name.
- **L114:** CPU jobs are bounded by their own `RLIMIT_AS`; none needs systemd in the seat.

## Done

Your final message gives, verbatim:
- the diff;
- the module sha256s;
- the fresh-tree CPU run;
- the mutant table;
- `STAGED_LINES.md` in full;
- the deviations;
- every file written.
