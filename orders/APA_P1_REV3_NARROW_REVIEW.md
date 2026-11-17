# APA-P1-REV3: narrow cross-model review of the REV2 fix round (P1-F2), then the eight policy-test updates

YOUR WRITABLE TARGET is:
- `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_p1rev3/artifacts/apa_p1_rev3/`
  (new; findings, report, receipts and staged reproductions), and
- **in step T only**, `tensor_cuda/tests/test_apa_p1d_policy.py` in this worktree (branch `apa-p1rev3`, at the commit
  carrying this order).

Edits there, CPU runs, and scratch under `/tmp/p1_rev3_*` are AUTHORIZED. Point `TMPDIR` at the scratch directory for
every command.

**Read-only:**
- everything else in this worktree;
- every other worktree, including `tensor_fork_apa_wt_p1f2` (it holds Build F2) and `tensor_fork_apa_wt_p1d` (Build F);
- the production repository.

Read-only git is allowed. **Searching:** never walk `/`, `/mnt` or `/mnt/Shared` recursively. Search inside this
worktree, or inside a named folder of another worktree, by path.

## What you review

P1 ships packed selective-precision keys (APA-SP) as production's default. A blind Codex red team (P1-REV2, receipts
`71c77cb7`) found 2 MAJOR, 4 MINOR and 2 NIT defects in the packed default and its first fix round. P1-F2 (code and
receipts `bc9e6f5e`, report `0ca294a1`) fixed them under the lead's rulings R1 to R8 (`orders/APA_P1F2_REV2_FIXES.md`).

**The diff:** `git diff 71c77cb7 bc9e6f5e -- tensor_cuda scripts/apa_p1f1_sass.py`: 16 files, +904/−76. The product
part (`tensor_cuda/src`, `tensor_cuda/include`, `tensor_cuda/tensor_cuda`) is 14 files, +270/−48. The rest is the new
test file `tensor_cuda/tests/test_apa_p1f2_cpu.py` and F1's SASS script.

**This review is narrow:** the diff, and how it closes REV2's findings. It is not a second full red team of P1. Answer:
1. **Fix completeness.** For each of REV2's P1R2-F01, F02, F04, F05, F06, F07, F08 and F09, is the fix complete
   against the finding's own argument and the lead's ruling (R1 to R8)? Name any case it still misses.
2. **New defects.** Does the diff introduce a defect: a refusal that rejects a valid input, a check after device
   work, a changed finite result, an integer bound off by one, an exception type that changes a caller's contract, or
   a broken binding?
3. **The non-finite contract.** "The packed route never hides a row that dense SP makes non-finite; NaN row sets are
   equal" (the ruling in the F2 order). Does R2's `prepare` change hold it on every packed entry that reads TQ4 keys?
   Check the seat's TQ4 census (`artifacts/apa_p1f2/TQ4_CENSUS.json`) against the code.
4. **The margin pin (R1).** Can any public path still reach `lookup` or attention with a table that is not the pinned
   one?
5. **The index census (R3).** Check `artifacts/apa_p1f2/INDEX_CENSUS.json` against the code. Is any native int that
   receives a capacity, extent or product unguarded?
6. **The residual D12.** A finite key row whose squared norm overflows fp32 stores an Inf norm. Can that hide a dense
   non-finite row, or change a finite result? Rate it.
7. **G6 (SASS).** The gate failed as registered on path-derived names. The lead's recount maps the path hash and the
   path-length count: 781 kernels per build, and only the 3 registered `spq1::prepare` kernels differ
   (`artifacts/apa_p1f2/lead_f2_sass_check.json`). Recount it yourself from the two dumps (read them streaming), with
   your own mapping. Report your count beside the lead's.
8. **Anything else** in the diff that a production engine should not ship.

**Blind first:**
1. Read the diff, REV2's `findings.json` and the F2 order.
2. Write `findings.json` (first version).
3. Only then read F2's report (`artifacts/apa_p1f2/REPORT.md`) and the lead's ledger entry quoted there, to check
   their claims.

## Builds and dumps (read-only)

| Name | Path | sha256 |
|---|---|---|
| Build F (before) | `tensor_fork_apa_wt_p1d/artifacts/apa_p1f1/arms/F/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so` | `5c17147715ffc422b0aa7ba8fae826c554a4caff63b1231c22ce580867efc243` |
| Build F2 (after) | `tensor_fork_apa_wt_p1f2/artifacts/apa_p1f2/arms/F2/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so` | `6f03b2cdd5d7f9bee4456ed34759c6c710cccfab9a50f9ac0fc6a678c076f579` |
| SASS of F | `tensor_fork_apa_wt_p1f2/artifacts/apa_p1f2/sass/F.sass` (237 MB) | in `HELD_BACK.txt` there |
| SASS of F2 | `tensor_fork_apa_wt_p1f2/artifacts/apa_p1f2/sass/F2.sass` (237 MB) | in `HELD_BACK.txt` there |

(Both worktrees are under `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/`.) Check each sha before
use, and refuse on a mismatch.

## Step T: the eight policy tests (after `findings.json` v1 is written)

R1 makes `PackedRegisteredMargins.load` refuse any table other than the pinned one. `test_apa_p1d_policy.py`'s helper
`packed_table()` (lines 132-143) loads a synthetic table with its own digest. So these 8 tests now fail on F2's tree,
as F2 registered:
- `test_stale_generation_is_refused`;
- `test_packed_table_on_dense_kq_is_refused`;
- `test_packed_route_refusals_before_launch`;
- `test_packed_route_forwards_delta_causal_sinks_and_r2[64]` and `[128]`;
- `test_packed_loader_checks_its_own_registration_quantizer_and_entries`;
- `test_routes_are_reported_for_quantize_and_attention`;
- `test_prefill_on_packed_keys_is_refused_naming_dense`.

Change only `test_apa_p1d_policy.py`, so that each of the 8 passes on F2's tree and keeps its test's intent:
- Prefer loading the pinned table (`artifacts/apa_p1d/packed_e_q_table.json`, through the public `load`). Expected
  values derived from the synthetic entries are then derived from the pinned entries.
- A test whose intent is the loader's refusal of a bad synthetic table now asserts that refusal: the table is
  refused, with R1's error, before any device work.
- If a test cannot keep its intent on the pinned table, mark it skipped. The reason names the replacement test in
  `test_apa_p1f2_cpu.py` that covers the same property. Report each such case.

Then run `test_apa_p1d_policy.py`, `test_apa_p1f1_cpu.py` and `test_apa_p1f2_cpu.py` on CPU against Build F2, per test
ID. **Pass condition (gate GT):**
- all IDs pass, except skips you name;
- the collection is otherwise unchanged;
- no other test file changes.

Record before and after per ID in `artifacts/apa_p1_rev3/T_RESULTS.json`. Write the diff to
`artifacts/apa_p1_rev3/T.diff`. This step does not change your review findings; if it shows a product defect, add it
as a new finding.

## Method

- **CPU only.** Your sandbox has no GPU. CPU checks are reading, arithmetic, constructed host-side cases, symbol and
  SASS inspection (`nm`, `cuobjdump`, `c++filt`), and the CPU test files.
  - Run with `CUDA_VISIBLE_DEVICES` empty, `nice -n 19`, single-threaded, and `PYTHONDONTWRITEBYTECODE=1`. Every
    command gets a `timeout` of at most 1,200 s.
  - To run the CPU tests against a build, put its `arms/<name>` folder first on `PYTHONPATH`. Pass
    `-p no:cacheprovider` and `--basetemp` under your scratch, so nothing is written into another tree.
- **GPU reproductions are staged, not run.** For a suspected fault that needs the GPU, write at most 6 scripts under
  `artifacts/apa_p1_rev3/repro/`, named `r<nn>_<short>.py`. Each script:
  - checks the module sha before importing;
  - loads the module by path, as `scripts/apa_p1f1_gpu.py` does;
  - prints one JSON line with its verdict;
  - states the confirming and refuting outcomes in its docstring.

  **REV2's lesson:** a staged script that passed `py_compile` failed on a call signature. So each script must also run
  on CPU up to its first device call (imports, sha checks, table loads with their real signatures, argument parsing),
  with a receipt per script. The lead runs each once, as
  `forge-gpu-run --min 30 --max 600 --label p1rev3-<nn> -- timeout 560s python3 <script>`.
- **A finding** has an ID (`P1R3-Fnn`), a severity (BLOCKER, MAJOR, MINOR or NIT), the file and line, the argument or
  staged reproduction, and a suggested fix. A question checked and found sound is reported as checked, with what you
  ran or read.
- NITs are fixed once and closed without another review round.
- Write `artifacts/apa_p1_rev3/findings.json` early and keep it current. Also write `REPORT.md` there.
- **Ceiling:** 6 hours of wall time. If you cannot finish, report what was checked and what was not.

## Mechanism & scope

- **Measures:** whether P1-F2 closes REV2's eight findings without new defects, and whether the policy tests can be
  brought back to passing with their intent kept.
- **Mechanism doc:** `orders/APA_P1F2_REV2_FIXES.md` (the rulings R1 to R8 and the non-finite contract), with REV2's
  `artifacts/apa_p1_rev2/findings.json` and F2's `artifacts/apa_p1f2/REPORT.md` (in this worktree).
- **Exercises:** reading the 16-file diff and the code it calls; the three CPU test files; F2's censuses; a SASS
  recount; at most 6 staged GPU reproductions; the test-helper update.
- **Does not cover:**
  - P1-D and F1 beyond the diff's reach (REV2 covered them);
  - model quality (QG0-R follows on the shipping module);
  - speed;
  - promotion.
- **Operand census:** every hunk of the 16-file diff; F2's `TQ4_CENSUS.json`, `INDEX_CENSUS.json` and
  `HELPER_CENSUS.json` rows; the 8 named tests.
- **Baseline isolation:** `71c77cb7` (the code at `710cc0d4`) is the baseline. Every entry behaves as there, except
  where R1 to R8 change it.
- **Required evidence:**
  - `findings.json` (v1 sealed before reading F2's report, then final);
  - `REPORT.md`;
  - each check's command and output;
  - the SASS recount;
  - staged reproductions with their CPU-prefix receipts;
  - `T.diff` and `T_RESULTS.json`.
- **Forbidden configuration:**
  - edits outside the writable target;
  - changing any test other than `test_apa_p1d_policy.py`, or changing it before `findings.json` v1 exists;
  - builds;
  - any GPU use;
  - git writes;
  - deleting anything yourself.

## Seat

Codex gpt-6.1-sol, xhigh effort. A Claude seat wrote P1-F2, so this is the cross-model review.

**Rules:**
- Read `/mnt/Shared/HOUSE_RULES.md` and `/mnt/ForgeRealm/Project-Tensor/AGENTS.md` first. This order is immutable.
- No git writes (read-only git is allowed). No subagents. No background waits. Run everything synchronously, each
  command under an explicit timeout.
- **No deletes.** Never run `rm`, `rmdir`, `unlink`, `git clean`, `find -delete` or an inline delete call. Claude's
  delete guard does not cover your seat, so this rule is yours to keep. List anything that should go for the lead.
- Never kill a process you did not start.
- **Prior art:** follow the directive in `AGENTS.md`. Every technique you cite or propose carries its source, or
  "no prior art known to me".

## Registration checklist (order-lint)

- **L01:** the instruments are `git diff` and `git show` (read-only), python3 with NumPy, pytest, `nm`, `cuobjdump` and
  `c++filt`. Run each with `--version` (or `--help`) first. Record the outputs in
  `artifacts/apa_p1_rev3/INSTRUMENTS.log`.
- **L03:** n/a. No share or coverage bar is gated.
- **L09:** GT compares test outcomes per ID. Any bitwise claim you check from receipts uses `np.array_equal` on
  same-width unsigned views, after dtype and shape checks. Logs and timing are excluded.
- **L10:** n/a. No identity is gated; the review reports findings.
- **L11:** n/a. No program-unchanged claim is gated. Question 7 recounts SASS as a finding.
- **L12:** GT inherits F2's per-ID CPU comparison (`artifacts/apa_p1f2/CPU_G3_COMPARISON.json`). It changes exactly the
  8 named IDs' outcome back to pass (or a named skip). Every other pair applies unchanged.
- **L14:** n/a. There is no gain gate.
- **L16:** the cheapest passing work, and what excludes it:
  - "checked, no findings" without running anything: excluded because each answer names what was run or the
    `file:line` read, and question 7 needs your own count;
  - making the 8 tests pass by deleting their assertions: excluded because each kept test keeps its intent (the
    report says how), and each skip names its replacement.
- **L18:** the shipped default stays packed (the owner's 11-02 requirement). The review judges whether the fixes are
  safe, not whether the default should change.
- **L19:** any array you compare is checked nonzero and non-constant first, at full width.
- **L25:** you run no bitwise comparison on a device. GT runs both sides (before: F2's recorded per-ID outcomes; after:
  this run) on CPU, with the same pytest options.
- **L32:** the bounded buffer in scope is `PackedSpKeys`' capacity (R3, R8). Question 5 checks its bounds. No fill
  gate is registered here; F2's G4 filled it.
- **L35:** the order fixes the 8 IDs and GT. You may not widen the test edit beyond `test_apa_p1d_policy.py`, or pass
  GT by skipping without a named replacement. If GT cannot hold, report RED with the reason.
- **L41:** n/a. No mutant is registered. GT's tests are the existing ones.
- **L44:** every count you report (kernels, census rows, test IDs) is recomputed by you from the raw source, and
  printed beside the receipt's count.
- **L45:** you run no GPU line. The lead runs each staged reproduction and records `forge-gpu-run`'s exit, the inner
  `timeout` exit and the log's sha256. Exit 124 is a timeout, never a pass.
- **L48:** staged reproductions run from this worktree's root, unchanged.
- **L55:** for questions 3 and 6, show the arithmetic on one constructed instance each (an Inf key row; a finite row
  whose squared norm overflows fp32) before relying on any receipt.
- **L56:** (a) a slip in running your own tools goes under Deviations: re-run and continue. (b) An exception in your own
  scripts is repaired at most 3 times, each recorded. The STOP list is exhaustive: an L35 clause that cannot hold, and
  the 6-hour ceiling.
- **L58:** every command runs with `CUDA_VISIBLE_DEVICES` empty. You use no GPU.
- **L59:** "v1 sealed before reading F2's report" is shown by mtimes: `findings.blind.v1.json` is written with its
  sha256 and a `date` line by the same command, and the first read of `artifacts/apa_p1f2/REPORT.md` is logged after it
  in your ledger. The start record is the shim's `SHIM-RUN` line in `logs/apa_p1_rev3_r1.log`. The report quotes all
  three.
- **L60:** n/a for a new census. Questions 4 and 5 check F2's `BOUNDARY.json` and censuses against the code.
- **L65:** no deletes, as stated.
- **L79:** staged reproductions run on CPU up to their first device call (Method). Nothing touches the GPU lease.
- **L82:** the one writable product-adjacent file is `tensor_cuda/tests/test_apa_p1d_policy.py` (step T).
  `grep -n 'packed_table(' tensor_cuda/tests/test_apa_p1d_policy.py` lists its uses (lines 132, 362, 399, 405, 436, 517,
  527, 529, 531, 546, 565 at sealing).
- **L99:** the review's premises are the lead's own readings, run before sealing:
  - `git diff --stat 71c77cb7 bc9e6f5e -- tensor_cuda scripts/apa_p1f1_sass.py` gave 16 files, +904/−76;
  - the product part gave 14 files, +270/−48;
  - `git diff --stat 710cc0d4 71c77cb7 -- tensor_cuda` is empty (REV2 added receipts only);
  - the lead recomputed both module shas above, and F2's `CPU_G3_COMPARISON.json` lists the 8 IDs as the only
    differences.

  Re-run the first three first. If any differs, report it before reviewing.
- **L103:** the modules are named by path and sha256 in the table above. Check the sha before each use, and refuse on a
  mismatch.
- **L111:** n/a. No storage design is emulated.

## Done

Your final message gives, verbatim:
- every finding;
- each question's answer, with what you ran or read;
- your SASS count beside the lead's;
- step T: the diff summary, GT's per-ID result, and every skip with its named replacement;
- each staged reproduction's path, sha256, CPU-prefix receipt and stated confirm/refute outcomes;
- `findings.json`'s sha256 (v1 and final);
- what you would delete, if anything (the lead does it);
- a "Prior art" heading.
