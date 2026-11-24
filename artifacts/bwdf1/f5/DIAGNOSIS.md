# BWD-F5 A1: why every BWD-F3 P2 child exited 255

Evidence classes: READ = a file or log quoted below; INFERRED = reasoning from READ facts (the sanitizer was not
run, and may not be). Paths are relative to the worktree roots; `W1` is the sibling worktree
`tensor_fork_apa_wt_bwdf1` (read only).

## READ: the launch path

1. The line (`scripts/bwdf3_gpu_lines.py:83-85`, the `mc` list) wrapped the whole step:
   `forge-gpu-run ... -- timeout 500s compute-sanitizer --tool memcheck --target-processes all --log-file
   <one line log> python3 -B -P scripts/bwdf3_step_p2.py <mode> <arm>`. Only the nine memcheck P2 lines
   (`e1`, `e2`, `e3` on three builds) carry `--target-processes all`; the other memcheck lines (g2, copy, k2) do not.
2. `compute-sanitizer` is the house wrapper (`$HOME/.local/bin/compute-sanitizer`): it runs
   `forge-gpu-lease-check --quiet --require-held` and then `exec`s the real 2024.3 binary with the same arguments.
3. The step (`scripts/bwdf3_step_p2.py`) is the sanitizer's target. Its line 2 runs
   `forge-gpu-lease-check --require-held` (a Python subprocess) and then, for each row, runs a child
   (`bwdf3_step_p2.py:107-108`): `subprocess.run([python3, -B, -P, <step>, --child, mode, arm, rid, out], env=env,
   capture_output=True, text=True, timeout=...)`. The child re-executes the same file, so its line 2 is again
   `forge-gpu-lease-check --require-held`; only then does it import the engine (`G.load_arm`).
4. The step records a row `CHILD-DIED` when the child printed no `ROW ` line
   (`bwdf3_step_p2.py:112-113`), with the child's exit code and `stderr[-800:]`.

## READ: what the lead's run showed (`W1/logs/bwdf3_lead_runner.out:60-77`, `W1/logs/bwdf3_lead_p2_e1_F3_a1.log`)

- All nine memcheck P2 lines exited 255, each in 1 to 2 s (for example `p2_e1_B` 00:39:37 to 00:39:38). The three
  `count` lines (same step, same children, no sanitizer, `LD_PRELOAD` counter) exited 0 and took about 5 s each for 6
  rows (about 0.8 s a row, which is what an engine import plus CUDA work costs). The sanitized lines spent about 0.1 s
  a row.
- Every row of every one of the nine lines is `CHILD-DIED`, `child exit 255: ` with EMPTY stderr and no `ROW` line,
  the controls included (`W1/artifacts/bwdf1/f3/gpu/p2_e1_F3/results.json`).
- Each of the nine `--log-file` logs has exactly two lines: `========= COMPUTE-SANITIZER` and
  `========= Error: Target application terminated before first instrumented API call`. No `ERROR SUMMARY` line.
- The 22 memcheck lines that did NOT pass `--target-processes all` (g2_direct, copy, g2_f01/f03, k2_f01/f04/f07/vals)
  exited 0 and every log ends with an `ERROR SUMMARY` line: `0 errors` on the Build F3 lines (for example
  `W1/artifacts/bwdf1/f3/gpu/k2_f01_F3.memcheck.log`), `15 errors` on `k2_f01_B` / `k2_f01_E`, where the defect is the
  point. (Runner log: 34 lines exit 0, 9 exit 255, the nine P2 memcheck lines.) Some of those steps also start the lease check as a Python
  subprocess (line 2), and `bwdf3_step_g3.py:41` runs a `-c` child, so "a Python subprocess under the sanitizer" alone
  is not the fault; `--target-processes all` plus the row children is the difference.

## INFERRED: what happened

- The exit code 255 and the log text are the sanitizer's own: "terminated before first instrumented API call" is its
  report for a target process that exited without making a CUDA call, and 255 is the status it returns for it. With
  `--log-file` the sanitizer writes its messages to the file, never to the target's stderr, so the child's stderr is
  empty. That fits every READ fact, and rules out Python's own failure path: a failed lease check would have printed
  `not inside a held forge-gpu-run lease` (`SystemExit(str)`, exit 1) on stderr, and a refused row would have printed
  a `ROW` line. The child never reached `print("ROW ...")`.
- The children lived about 0.1 s (about one row's worth of time in which Python starts and the line-2 lease check runs, but
  well short of the roughly 0.8 s an engine import takes). So each child ended at or before its own lease check, ended
  by the sanitizer, not by our code.
- With `--target-processes all` every process in the tree is a sanitizer target: the step itself, each lease-check
  subprocess, and each row child. The step makes no CUDA call of its own, so the line's own exit (255, with the
  step's `DONE` printed) is the same error applied to the step, and the single shared log file (no `%p` in the name)
  holds one report, not one per process. Most likely cause: in the all-processes mode the sanitizer's injection into a
  process started by `subprocess.run` with piped stdout / stderr ended each child before Python's startup finished.
  Which exact step of the injection failed cannot be told from the receipts: the log has no more detail, and the
  sanitizer was not run here.
- Not the lease: `forge-gpu-lease-check` walks `/proc` parents to the `forge-gpu-run` pid and reads
  `FORGE_GPU_LEASE_MODE`; the step's own line-2 check passed (the step wrote `results.json`), and the children get the
  same environment (`env=dict(os.environ)`). Not the modules or the cases: the `count` lines load the same modules and
  run the same case table and passed. Not the arm: B, F2 and F3 all failed identically.

## Fix (A2), and what it removes

Run the step inside the lease and OUTSIDE the sanitizer, and give every row child its own single-target sanitizer:
`compute-sanitizer --tool memcheck --log-file <row log> python3 -B -P scripts/bwdf5_step_p2.py --child ...` (the
default `--target-processes application-only`, the mode the 22 working lines used). That removes the all-processes
mode, the sanitizer around a process that makes no CUDA call, the shared log, and (child output goes to files, not
pipes) the pipe between the sanitized process and its parent. Each child still checks the lease first (line 2), and
the parent records how every child ended (exit code, signal, timeout) as a row. The verdict reads each row's log:
only `ERROR SUMMARY: 0 errors` passes; an empty, missing, `terminated before first instrumented API call` or
summary-less log is a harness failure, never zero errors. The canary line (one control row on Build F4) shows the lead
the outcome before the nine P2 lines are run.

What is NOT known: whether a child survives its own sanitizer on this driver and these modules. The 22 working
lines show a Python step with CUDA calls survives it, and the child is such a step, but the canary is the proof. If
the canary's child still ends with exit 255 and a no-API-call log, the fault is the sanitizer around a `-B -P`
Python child with a lease-check subprocess; the next step is then to run the whole P2 step as its own sanitized
target per row (one `forge-gpu-run` line per row), not another wrapper variant.
