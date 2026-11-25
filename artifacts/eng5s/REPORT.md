# ENG5-S report (filed by the lead)

Order: `orders/ENG5S_STATUS_CHECKS.md` (commit 5660f4755). Seat: Claude Opus 5.5, high effort, Agent tool. Under L127 the
seat's final message is the report; it follows verbatim below the line.

## Lead verification on disk (2026-11-24 02:35)

- G1: `g1/G1.xml` and `g1/G1_final.xml` each show tests=612, failures=0, errors=0, skipped=0.
- G3 CPU: `G3_CPU_COMPARE.json` has ids_B 4278, ids_S 4278, pass true.
- Faithfulness: `F2_SUBSET_CHECK.json` has helper_equal true, 9 statements, each present in ENG5-F2's tree, pass true.
- Verdict self-test: `gpu_dry/VERDICT_SELFTEST.json` has 97 checks, 0 failed.
- Dry runs: all 30 receipts in `gpu_dry/` record exit 78.
- Source: `git diff fbd335ae7` over `tensor_cuda/src` and `tensor_cuda/include` is 3 files, +49/-9. The 9 removed
  lines are exactly the nine bare calls of the order's table, and every added line is a status check, the helper or
  a comment. `source.diff` sha256 matches the report.
- Installed module sha256 is 6b81f0c6ce16e679fcd285fea37041ebfbf9e6f4798c3c85568985cc9e8c3f44, Build S's.
- Mutants: MUTANTS.json was written 06:21:31 UTC. The first mutant configure ran at 06:26:54 UTC, so registration came
  before any build.

## Lead rulings on the seat's questions

- **Deviation 2 (S1's slot clause): confirmed.** S1 says the nine checks must not read or clear the last-error slot
  "for these calls". The new checks never touch it. S2 keeps everything else unchanged, so Build B's existing slot
  checks (after from_host and to_host, in cuda_sync, and the clear before the drain) stay.
- **Deviation 9 (G1's "ENG5 status rows"): accepted.** The targeted rows plus ENG5's whole catalogue, under one
  B-versus-S rule, cover more than the order asked.
- **Deviations 1, 3-6 and 8:** accepted as recorded. Deviation 7 (empty axpy_ on a GPU) is settled by the g3_S
  axpy_empty control in the lead's GPU run.

## Prior art check (lead)

- LFI is Marinescu and Candea, "LFI: A Practical and General Library-Level Fault Injector", DSN 2009.
- McKeeman is "Differential Testing for Software", Digital Technical Journal 10(1), 1998.
- Both are confirmed from known references; neither was fetched in this session.

---

## Done

**Verdict:** my part passes as registered. G1, G3's CPU half and G5 pass. G2, G3's GPU half and the four mutant lines are staged, and every dry run exits 78. The round is accepted only once you have run the GPU lines.

- **G1: PASS.** `artifacts/eng5s/g1/G1.xml`: 612 passed. A rerun on the final tree also gave 612 passed (`g1/G1_final.xml`).
  - **Source map:** Build B and Build S differ in exactly `tensor_cuda/src/kernels.cu`, `tensor_cuda/src/apa_spo/packed_prefill.cuh` and `tensor_cuda/include/tc/core.h`. The worktree's engine files and installed module are Build S's. The hooks arms (B_H, S_H) have B's and S's sources; only a copied CMakeLists.txt carries the hooks line.
  - **36 status rows** cover every public route to the nine sites, with host operands, empty CUDA operands, and nonempty CUDA operands while CUDA is hidden:
    - 24 make no CUDA call on Build B, and Build S is identical on each;
    - 9 make CUDA calls, and Build S is identical on each;
    - 3 make CUDA calls, and Build S raises in the S1 form. These are `tc.synchronize` and `C.synchronize` (Build S: `cuda_sync: cudaDeviceSynchronize failed: cudaErrorNoDevice: …`) and axpy_ on empty CUDA operands, which Build B returned from silently.
  - **574 rows of ENG5's full catalogue** (modes host, empty, emptylast, one):
    - 402 make no CUDA call on Build B, and Build S is identical on each;
    - 170 make CUDA calls, and Build S is identical on each;
    - 2 raise in the S1 form: `C.axpy_` in the empty and emptylast modes.
  - **OOM drain on the CPU lane:** with the allocation injected to fail with an out-of-memory error and the drain injected to fail, Build S raises `Storage (cudaMalloc OOM retry): cudaStreamSynchronize (drain) failed: cudaErrorInvalidValue: …`, while Build B carries on to the retry.
- **G2: STAGED.** Lines g3_B and g3_S, ported from ENG5-F2's f2g3. Each dry run exits 78.
- **G3: CPU PASS; GPU STAGED.** On the CPU, 4,278 IDs on each side and 0 differences (`artifacts/eng5s/G3_CPU_COMPARE.json`). The GPU half is lines suite_B_01–12 and suite_S_01–12, each dry run 78.
- **G4: STAGED.** Mutants s01–s04 are registered in MUTANTS.json (06:21:31 UTC, before any mutant was built or run). Their lines are g3_s01–s04, each dry run 78. The verdict code was driven on constructed rows: 97 checks, 0 failed.
- **G5: PASS.** Every write is in the writable set (worktree `git status --ignored` checked). No GPU use, no git writes, nothing deleted.

**Nine-site table** (Build B line at fbd335ae7 → the fix in Build S):

| # | Site | Call | Build B | Fix |
|---|---|---|---|---|
| 1 | `cuda_sync` | cudaDeviceSynchronize | kernels.cu:108 | kernels.cu:121; the existing slot check stays at :122 |
| 2 | Storage OOM retry, the drain | cudaStreamSynchronize(0) | :185 | :201-203; a failed drain raises instead of retrying |
| 3 | `NDArray::zeros` | cudaMemset | :773 | :794 |
| 4 | `NDArray::from_host` | cudaMemcpy | :891 (function :885-897) | :914; the existing slot check stays at :915 |
| 5 | `NDArray::to_host` | cudaMemcpy | :903 (function :898-908) | :928; the existing slot check stays at :929 |
| 6 | `NDArray::clone` | cudaMemcpy | :913 | :941-942 |
| 7 | `NDArray::to` | cudaMemcpy | :923 | :953 |
| 8 | `axpy_` | cudaMemcpy | :5399 | :5431-5432 |
| 9 | packed prefill, the dump's NaN fill | cudaMemset | packed_prefill.cuh:244 | packed_prefill.cuh:246-247 |

The shared helper `check_cuda_status` is at kernels.cu:112-117 (comment at :107-111). Its declaration is at core.h:697 (comment at :694-696). Before editing, I re-read all nine sites against the order's table (S3); they match.

**Diff summary** against fbd335ae7 (`artifacts/eng5s/source.diff`, sha256 594f1db058d23378c6fa135c1271d6c21d9482f353b7c9bfc0738e4f6e65c203):
- Three files, +49/−9: kernels.cu +41/−8, packed_prefill.cuh +4/−1, core.h +4/−0.
- The 9 removed lines are exactly the nine bare calls.
- Faithfulness (`artifacts/eng5s/F2_SUBSET_CHECK.json`, PASS): each of the 9 `check_cuda_status` statements appears verbatim in ENG5-F2's tree, which has no others, and the helper is identical to F2's.
- Counts per role (L90): product 3 files (9 sites, the helper and its declaration); test 2 files (`test_eng5s_status.py` and `eng5s_cuda_shim.c`); harness 11 scripts.

**Every module sha256:**

| Arm | sha256 |
|---|---|
| B | 1ebc3d9b4610b6812ccd96cb3462931b807ef1a99de7b41c2c9f5e9ec12e86eb |
| S (also the module installed in the worktree) | 6b81f0c6ce16e679fcd285fea37041ebfbf9e6f4798c3c85568985cc9e8c3f44 |
| B_H | 6c64129cf3d4c6532686cfc07f23e03f2318a15a3468cee67a83980dff19c48b |
| S_H | 0038f7e1338eeea668d98cdbdd90f6b27045e2d3696d9d67f893543484ee6181 |
| s01_clone_status_dropped | 3ef192348976f4f5a32cce770a86ce30df8e8d5291b71a1efdf64ae0128632b9 |
| s02_to_status_dropped | 592400d56081aa2e3cfd66ecd005b3986c4fb825b03e51e94924dd7255e4e2c0 |
| s03_zeros_status_dropped | 344e7a1f775683eaa53bf4ad329d69f39d056dbbb54ef2b78514a441ef297cf9 |
| s04_from_host_last_error_only | 9e83c5b6914c59711ab3cccd74c829082888ef07305109c52833adc6aa7f4071 |

Every build exited 0 (`artifacts/eng5s/build/build_receipts.json`).

**G3's CPU per-ID result** (series cpu_B_r2 and cpu_S_r2):
- Each side: 4,278 IDs, with 1,545 passed, 2,174 failed, 544 skipped and 15 errors.
- One collection error on both sides: `test_selector_accuracy.py`.
- 0 differences, and no test wrote into the worktree.
- After each run Build S's files and module were restored and sha-checked against BUILD_S.json (L128 ok).
- Peak RSS was 2,657,676 kB.
- `test_terrain_render.py` and this round's own test file were excluded (L40); the tracked wo*.ppm files were not touched.

**Mutant table** (`artifacts/eng5s/MUTANTS.json`; `MUTANT_RESULTS.json` is filled by `scripts/eng5s_arms.py results` after your runs):

| Mutant | Hunk | Killing G2 row | Status |
|---|---|---|---|
| s01_clone_status_dropped | clone's copy status discarded | clone | STAGED (g3_s01) |
| s02_to_status_dropped | to's copy status discarded | to_d2h and to_h2d | STAGED (g3_s02) |
| s03_zeros_status_dropped | zeros' fill status discarded | zeros | STAGED (g3_s03) |
| s04_from_host_last_error_only | from_host goes back to the slot check only | from_host | STAGED (g3_s04) |

- **Kill rule:** a mutant is KILLED when its row's injection fired and the call returned.
- **Not kills:** an unfired injection, a dead child, a missing row, or only one direction of `to` returning.
- **Marker:** each hunk writes an executed-marker line, which the verdict records.

**Each staged GPU line with its dry-run exit** (`artifacts/eng5s/GPU_LINES.json`; you run `python3 -B -P scripts/eng5s_gpu_line.py run <line>` from the worktree root):

| Lines | --min / --max | Inner timeout | Expected peak host | Dry-run exit |
|---|---|---|---|---|
| g3_B, g3_S, g3_s01, g3_s02, g3_s03, g3_s04 | 120 / 540 | 480 s | 3 GiB | 78 each |
| suite_B_01 … suite_B_12, suite_S_01 … suite_S_12 | 480 / 540 | 500 s | 6 GiB | 78 each |

- **The lease check** is at `scripts/eng5s_gpu_step.py:2`. It is repeated as the first statement of each G2 child (G2_CHILD, at step-file line 68).
- **What each g3 line runs:**
  - Ten injected rows: the nine sites, with `to` in both directions and the drain reached through both cudaMalloc and cudaMallocAsync.
  - Each row first runs the same call with no injection, as a success control (S2).
  - One control-only row: axpy_ on empty operands.
  - Site 9 runs on the hooks build of its arm (B_H for B, S_H for S). It is not applicable on the mutant arms, which have no hooks build.
- **Expected verdicts:** g3_S PASS; g3_B RECORDED; s01–s04 KILLED; the suite outcomes per ID identical between B and S.
- **Before the lease:** each line refuses a module whose sha256 differs from its receipt (L103) and refuses a line directory that already exists. Write-once paths are listed per line (L119).
- **Order:** run the g3 lines first. Run the suite lines last, because each chunk swaps its arm's three engine files and module into the worktree.
- **Afterwards, on CPU:**
  - `scripts/eng5s_compare.py` on `h3/gpu_B` and `h3/gpu_S` outcomes;
  - `scripts/eng5s_arms.py results`.
- **Not covered:** multiple GPUs (there is one).
- **Checks before staging (L48):**
  - The verdict self-test (`gpu_dry/VERDICT_SELFTEST.json`) covers each site raising, returning, an unfired injection, a dead child, a missing row, a failed control, and each mutant's kill and survival.
  - The CPU drive (`gpu_dry/G2_CPU_DRIVE.json`) ran every case's child with its CUDA tensors placed on the host, plus the real site-2 rows on the CPU lane. Its site-2 checks all hold. Site 9's setup needs a GPU, so its child fails on the CPU lane; that is expected and recorded.

**Deviations:**
1. **Two extra builds, B_H and S_H.** Site 9 runs only with `dump=True`, and a build without the test hooks refuses dump before reaching the fill. So site 9's G2 row needs hooks builds of B and S. Their engine sources are unchanged; the hooks line is appended to a copy of CMakeLists.txt in /tmp.
2. **How I read S1's "do not read or clear the last-error slot"** — please confirm. The new checks never touch the slot. Build B's own slot checks stay where they were: after from_host (:915), after to_host (:929) and in cuda_sync (:122), plus the existing `cudaGetLastError()` clear before the drain (:197). Removing them would change results on success paths (S2), and keeping them is ENG5-F1/F2's form, which REV2 closed.
3. **Slip, rerun (L56(a)).** The first CPU suite series ran before `/tmp/eng5s_tmp/pt` existed, so every test that uses tmp_path errored on both sides (360 extra errors). I fixed `eng5s_suite.py` (repair 1 of 3) and reran as cpu_B_r2 and cpu_S_r2. The first series is kept as `G3_CPU_COMPARE_r1_basetemp_slip.json`.
4. **G1 test repair (1 of 3).** The catalogue row emptylast `C.int4_linear_fused` returns unwritten memory (ENG5's INT4-K0-UNWRITTEN): Build B against itself gives different bytes (`g1/UNWRITTEN_REPEAT_B_{1,2}.json`). For that one row only the result bytes are not compared; everything else is.
5. **CPU drive repair (1 of 3).** I first replaced `tc.synchronize()` with `pass` inside expressions, which is a syntax error; it now uses `None`.
6. **Public-remote cleanup.** Home paths in test tracebacks inside the suite receipts and logs were rewritten as `$HOME`. The per-ID compare was rerun afterwards and still shows 0 differences.
7. **Behaviour changes to note:**
   - `axpy_` always issues a 0-byte cudaMemcpy on empty operands, as Build B did, and its status is now checked. On the CPU lane that raises cudaErrorNoDevice. On a GPU a 0-byte copy should succeed, and the axpy_empty control in g3_S decides it. ENG5-F2's code is identical here.
   - On a failed device synchronize, the message changes from `CUDA error at sync: …` to `cuda_sync: cudaDeviceSynchronize failed: <name>: …`.
8. **G2 rows differ from F2's f2g3.**
   - Added: the OOM drain through both allocators, site 9, the success controls and the empty axpy_ control.
   - Dropped: F2's cublasCreate row, which is not one of the nine sites.
   - The shim copy now also injects cudaMalloc and cudaMallocAsync, and can arm several injections at once.
9. **How I read "the ENG5 status rows" in G1:** targeted rows for every route to each site, plus ENG5's whole catalogue, both under the same B-versus-S rule.

**Every file written** (under `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_eng5s/`):
- **Engine:** `tensor_cuda/src/kernels.cu`, `tensor_cuda/src/apa_spo/packed_prefill.cuh`, `tensor_cuda/include/tc/core.h`, and the installed module `tensor_cuda/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so` (Build S).
- **Tests:** `tensor_cuda/tests/test_eng5s_status.py` (sha256 aece0c00…), `tensor_cuda/tests/support/eng5s_cuda_shim.c` (sha256 9df7a36d…).
- **Scripts:** `scripts/eng5s_{arms,build,catalogue,compare,gpu_cpudrive,gpu_line,gpu_step,gpu_verdict,probe,subset_check,suite}.py`.
- **Receipts in `artifacts/eng5s/`:**
  - Top level: LEDGER.md, preflight_L01.txt, source.diff, source.stat, MUTANTS.json, MUTANT_RESULTS.json, GPU_LINES.json, G3_CPU_COMPARE.json, G3_CPU_COMPARE_r1_basetemp_slip.json, F2_SUBSET_CHECK.json.
  - `build/`: BUILD_*.json, build_receipts.json and its logs.
  - `g1/`: G1.xml, G1_r1.xml, G1_final.xml, G1_STATUS_ROWS.json, G1_CATALOGUE_ROWS.json, G1_OOM_DRAIN_CPU.json, G1_SUMMARY.json, UNWRITTEN_REPEAT_B_1.json, UNWRITTEN_REPEAT_B_2.json, G1_r1_pytest.log.
  - `h3/`: cpu_B, cpu_S, cpu_B_r2, cpu_S_r2.
  - `gpu_dry/`: 30 dry-run receipts, VERDICT_SELFTEST.json, G2_CPU_DRIVE.json.
  - `suite_src/`: B and S copies of the three engine files.
- **Builds:** `tensor_cuda/build-eng5s/{B,S,B_H,S_H,s01_clone_status_dropped,s02_to_status_dropped,s03_zeros_status_dropped,s04_from_host_last_error_only}`, each with module/, pkg/ and pkg.json (about 500 MB in total).
- **Logs:** `logs/eng5s_suite_cpu_{B,S,B_r2,S_r2}.log`, `logs/eng5s_g1_{r1,r2,final}.log`.
- **Written then restored:** while each suite run was in progress, the suite runner swapped the three engine files and the module into the worktree, then restored Build S's (sha-checked, L128).
- **Leave-and-list (should go, nothing deleted):**
  - scratch `/tmp/eng5s_src_{B,B_H,S_H,s01…,s02…,s03…,s04…}`, `/tmp/eng5s_shim`, `/tmp/eng5s_gpu_shim`, `/tmp/eng5s_tmp`;
  - the superseded series `artifacts/eng5s/h3/cpu_B` and `cpu_S`, kept as receipts of the slip.

## Prior art
- **Returned-status checking:** the CUDA Runtime API's error model (each call returns its status; `cudaGetLastError` returns and resets the thread's last error) and NVIDIA samples' checkCudaErrors. I took "test the call's own status". Mine: the message form and never touching the slot in the helper (after ENG4-F3). Annotated at kernels.cu:107-111.
- **Fault injection by library interposition:** LFI, Marinescu and Candea, 2009 (unverified; lead to check "LFI library fault injection interposition"), and LD_PRELOAD with dlsym(RTLD_NEXT) as in ENG3's counter. Mine: arming several injections at once and injecting the allocator. Annotated in the shim's header.
- **Mutation testing:** DeMillo, Lipton and Sayward, 1978.
- **Differential testing of two builds:** McKeeman, 1998 (unverified; lead to check).
- **No prior art known to me:** the G1 row list, the B-versus-S rule, routing site 9 through the hooks arms, and the subset check.
