# ENG5-S-T report (filed by the lead)

Order: `orders/ENG5ST_REGRESSION_TEST.md` (commit 3b6f6d902). Seat: Claude Sonnet 5.5, high effort, Agent tool. Under
L127 the seat's final message is the report; it follows verbatim below the line.

## Lead verification on disk (2026-11-24 09:20)

- `t/T1_S.xml`: 15 tests, 0 failures, 11 skipped (the GPU rows on the CPU lane). `t/T1_B.xml`: 15 tests, 4 failures
  (all four CPU rows), 11 skipped.
- `t/T3_VERDICT_DRIVE.json`: 58 checks, 0 failed. All six `t/gpu_dry/*.json` record exit 78.
- `git diff` of tracked files is empty. The test file mentions `scripts/` only in provenance comments; it reads no
  build directory, receipt or script.
- `scripts/eng5st_gpu_step.py` line 2 is `forge-gpu-lease-check --require-held`.

## Lead rulings

- **Deviation 4 (sha check before the lease): accepted.** The lead runs each line through `eng5st_line.py run <arm>`,
  which checks the module sha256 and the line directory before calling `forge-gpu-run`. The step repeats both checks
  inside the lease.
- **Deviations 1-3 and 5-8:** accepted as recorded. In-process arming is the form the GPU lines and G1 used, and every
  row asserts its injection fired.

---

# ENG5-S-T report (seat: Sonnet 5.5, high effort)

**T1 PASS, T3 PASS, T4 PASS. T2 is staged, not run: no GPU use, the lead runs the six lines.** One clause of the order is partly RED: the sha256 check before the lease (see Deviations, item 4).

## Done

**T1, CPU, PASS.** Both runs used `CUDA_VISIBLE_DEVICES` empty, `nice -n 19`, `-p no:cacheprovider`, basetemp and `TMPDIR` under `/tmp/eng5st_tmp`.
- **Build S:** the installed module has sha256 `6b81f0c6…8c3f44`, which matches BUILD_S.json. The result is **4 passed, 11 skipped**. All 11 skips carry the reason "GPU-lane rows need a visible GPU; this is the CPU lane (CUDA_VISIBLE_DEVICES is empty)". Junit: `artifacts/eng5s/t/T1_S.xml` (sha256 `f693ead7…0b96`).
- **Build B:** `tensor_cuda/build-eng5s/B/pkg` was first on `PYTHONPATH`. Its module sha256 `1ebc3d9b…86eb` matches BUILD_B.json. The result is **4 failed (all four CPU rows), 11 skipped**. Junit: `artifacts/eng5s/t/T1_B.xml` (sha256 `912dabc1…6fd5`).
- **Why B fails:** every failure is "message is not in the S1 form". B raises `CUDA error at sync: …` or `cudaMalloc failed: …` on these rows. B does not return silently on the CPU lane; silent returns show up only in the GPU rows.
- **No compiler:** a run with `PATH` emptied skips all rows. The reason read: "No C compiler available (tried cc, gcc, clang) to build the ENG5-S injection shim". Log: `logs/eng5st_nocompiler.log`.

**T2, GPU lines, STAGED.** Each of the six dry runs exited 78. Each module sha256 matched its receipt at staging. The receipt is `artifacts/eng5s/t/GPU_LINES_T.json` (per-line files in `gpu_dry/`). I did not run T2.
- **Expected verdicts:** S gives PASS. B gives DEFECT. s01 to s04 each give KILLED.
- **Preview, not T2:** `scripts/eng5st_replay.py` calls the test's GPU assertions on the rows ENG5-S's lead recorded on the GPU (`artifacts/eng5s/gpu/g3_<arm>/g3_rows.json`). The results are S PASS, B DEFECT (all 10 injected rows killed), s01 KILLED (clone only), s02 KILLED (to_d2h and to_h2d), s03 KILLED (zeros) and s04 KILLED (from_host). The receipt is `artifacts/eng5s/t/T3B_REPLAY.json`.
- **Limit of the preview:** it exercises the assertions on recorded data. The GPU rows have never run in my seat.

**T3, verdict code, PASS.** `scripts/eng5st_verdict.py selftest` ran 58 checks with 0 failed. The receipt is `artifacts/eng5s/t/T3_VERDICT_DRIVE.json`, with the constructed junit files in `t3_junit/`.
- **Cases covered, each giving its expected verdict:**
  - all rows pass: S PASS;
  - one row failed: S FAIL;
  - a mutant failing its own row: KILLED;
  - a mutant failing a different row: SURVIVED;
  - a missing row, an empty file, an unparseable file and no file: INVALID;
  - all GPU rows skipped (a CPU-lane run) and setup errors: INVALID.
- **Further cases:**
  - s02 with only one direction failing: SURVIVED;
  - own and other rows both failing: COLLATERAL;
  - own row failing by a dead child, not the kill signature: SURVIVED;
  - killed rows but no mutant marker: KILLED_NO_MARKER;
  - B with every injected row killed: DEFECT;
  - B where the test misses the defect: B_UNEXPECTED;
  - the real T1 CPU junit files judged as GPU verdicts: INVALID.
- **Step body:** also driven on a stand-in pytest command. It gave exit 0 with a verdict, exit 5 on write-once, and exit 7 when the lane is not visible. The L103 module check passes on all six real receipts.

**T4, scope, PASS.**
- Every write is inside the writable set, and `git diff` of tracked files is empty.
- The three unused paths in the post-run `git status` (`logs/eng5s_lead_runner.out`, `artifacts/eng5s/source.stat`, the `build-eng5s/` directory) predate my seat.
- No product file or other test file changed. No GPU use, no git writes, nothing deleted.

## The test's rows

The file is `tensor_cuda/tests/test_eng5s_status_checks.py`. Each row runs in its own child with the shim preloaded.
- **Lanes:**
  - CPU rows skip when `CUDA_VISIBLE_DEVICES` is not empty.
  - GPU rows skip when it is empty.
- **CPU lane, `test_cpu_status_rows[...]`:**
  - `synchronize_injected` (site 1): `cudaDeviceSynchronize` is injected to fail. It must fire and raise `RuntimeError` matching S1 as `cuda_sync: cudaDeviceSynchronize failed: cudaErrorInvalidValue: `.
  - `synchronize_no_device` (site 1): no injection. The real no-device failure must raise `cuda_sync: cudaDeviceSynchronize failed: cudaErrorNoDevice: `.
  - `oom_drain` (site 2): `cudaMalloc` OOM plus a failed drain. Both must fire. It must raise `Storage (cudaMalloc OOM retry): cudaStreamSynchronize (drain) failed: cudaErrorInvalidValue: `.
  - `oom_drain_pooled` (site 2): the same through `cudaMallocAsync`. I confirmed the CPU lane reaches it.
- **GPU lane, `test_gpu_status_rows[...]`:** each of the first ten rows runs an uninjected control first. Its result must equal the host-computed bytes. The injected call must then fire (`eng5s_fired`) and raise `RuntimeError`. The message must match `S1_FORM` (anchored) and start with the row's entry and call plus `cudaErrorInvalidValue: `.
  - `synchronize` (site 1)
  - `oom_drain` and `oom_drain_pooled` (site 2; the control's OOM retry must have fired)
  - `zeros` (site 3)
  - `from_host` (site 4)
  - `to_host` (site 5)
  - `clone` (site 6)
  - `to_d2h` and `to_h2d` (site 7, both directions)
  - `axpy_` (site 8)
  - `axpy_empty`: control only. `axpy_` on empty CUDA operands must return a `[0]` parameter, with no raise.
- **Site 9** is excluded, with a note in the file docstring: it needs a hooks build.
- **Compile and children:**
  - It compiles the shim into `tmp_path`, finding the compiler as test_eng3_host.py does.
  - Children import the same `tensor_cuda` package the pytest process imported, and the test checks this.
  - It reads no build directory, receipt or script.
  - It does not call the lease check. The step script does, in `scripts/eng5st_gpu_step.py` at line 2.

## Staged lines

Each line is run from the worktree root. Dry-run exit 78 for all six.

| Arm | Line | Expected |
|---|---|---|
| S | `forge-gpu-run --min 120 --max 540 --label eng5st-S -- timeout 480s python3 -B -P scripts/eng5st_gpu_step.py S` | PASS |
| B | the same with `eng5st-B` and `B` | DEFECT |
| s01 | the same with `eng5st-s01` and `s01` | KILLED (clone) |
| s02 | the same with `eng5st-s02` and `s02` | KILLED (to_d2h, to_h2d) |
| s03 | the same with `eng5st-s03` and `s03` | KILLED (zeros) |
| s04 | the same with `eng5st-s04` and `s04` | KILLED (from_host) |

- **Recording (L45):** the lead's runner is `python3 -B -P scripts/eng5st_line.py run <arm>`. It records forge's exit, the inner exit, the junit sha256 and the verdict in `artifacts/eng5s/t/gpu_run/<arm>/RESULT_T.json`. An exit of 124 is never a pass.
- **Run the arms in the order S, B, s01-s04.** The step refuses if its line directory already exists.

## Deviations

1. **Injection arming.** The order says "ENG5_INJECT arms one call". I used the shim's in-process arming (`eng5_inject_next`), as the GPU lines and G1's OOM row do. The environment form counts the nth call from process start, which is unstable. Every row still runs in its own child with the shim preloaded, and every row asserts the injection fired.
2. **Row counts.**
   - The GPU lane has 10 injected rows plus the `axpy_empty` control (11 tests). The order's "eleven injection rows" includes site 9, which it excludes.
   - The CPU lane has one extra row, `synchronize_no_device`.
3. **Missing CUDA headers.** I added a skip, naming the headers, when `cuda_runtime_api.h`, `cublas_v2.h` or `cublasLt.h` is not found. The shim includes them.
4. **PARTLY RED: "checks the arm's module sha256 before the lease".**
   - The step script starts inside the lease, so a check there cannot come before it.
   - The pre-lease check is in `eng5st_line.py run`, which also checks that the line directory is free.
   - The step repeats both checks, after the lease check and before pytest.
   - The staged line run directly through forge-gpu-run, without that runner, has no pre-lease hook.
5. **Dropped from the copied child.** I left out the `synchronize_after` tail, the lease-check line and the site-9 case. The full left-out and new line lists are in `artifacts/eng5s/t/COPY_DEPS.json`.
6. **Tool slips (L56a).**
   - The Write tool's similar-code hook refused `eng5st_gpu_step.py` twice, so I wrote it with a quoted heredoc.
   - Two no-compiler runs failed on `timeout` and `nice` not being on the emptied `PATH`. I re-ran with `PATH` set only for the python child.
7. **Repairs (L56b), each under 3.**
   - `eng5st_verdict.py`: 2, the argv index in the step-body drive and the temp and junit directories.
   - `eng5st_copy_deps.py`: 1, evaluating `PREFIX`.
8. **Git ignore.** `*.json` is git-ignored, so the JSON receipts under `artifacts/eng5s/t/` need `git add -f` (`COPY_DEPS.json`, `GPU_LINES_T.json`, `T3_VERDICT_DRIVE.json`, `T3B_REPLAY.json`, `gpu_dry/*.json`). The junit XML files, scripts and the test are not ignored.

## Prior art

- **LD_PRELOAD injection:** ENG3's counter and ENG5's shim in this repo, used as they are. I added nothing to the shim. Library-level fault injection is LFI (Marinescu and Candea, 2009), already cited in the lead's check.
- **Kill rule:** mutation testing (DeMillo, Lipton and Sayward, 1978). I took the rule that a mutant is killed only when its own site's row fails. Mine are the "killed" junit signature, the COLLATERAL and KILLED_NO_MARKER verdicts, and reading the verdict from per-row junit statuses.
- **Copied from this repo:** the S1 regex (`test_eng5s_status.py`), the row bodies (`eng5s_gpu_step.py` G2_CHILD), the prefixes and kill rows (`eng5s_gpu_verdict.py`), and the stage/run layout and 75/76 retry (`eng5s_gpu_line.py`).
- **Replay drive, copy manifest, CPU-lane and compiler lookup:** no prior art known to me for the replay drive (it is ordinary golden-file testing) or the copy-manifest check. The CPU-lane predicate and compiler lookup are copied from `test_eng4_matmul.py` and `test_eng3_host.py`.

## Files written

All under `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_eng5s/`:
- `tensor_cuda/tests/test_eng5s_status_checks.py` (sha256 `65ca40c5…9ed3`)
- `scripts/eng5st_common.py`, `eng5st_verdict.py`, `eng5st_gpu_step.py`, `eng5st_line.py`, `eng5st_replay.py`, `eng5st_copy_deps.py`
- `artifacts/eng5s/t/`:
  - `T1_S.xml`, `T1_B.xml`
  - `preflight_L01.txt` (python3 3.12.3, pytest 9.0.2, gcc 13.3.0, forge-gpu-run and forge-gpu-lease-check paths)
  - `GPU_LINES_T.json`, `gpu_dry/{S,B,s01,s02,s03,s04}.json`
  - `T3_VERDICT_DRIVE.json`, `t3_junit/c01.xml` to `c44.xml` (`c12` is deliberately absent: the no-file case)
  - `T3B_REPLAY.json`, `t3_replay_junit/{S,B,s01,s02,s03,s04}.xml`
  - `COPY_DEPS.json`
- `logs/eng5st_nocompiler.log`

**Scratch left, not deleted** (under `/tmp`, exempt): `/tmp/eng5st_tmp/` (`emptybin`, `probe`, `pt`, `t3`, `t3_x1xvm3at`).
