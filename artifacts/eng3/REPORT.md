<!-- Filed by the lead verbatim from the r2 seat's final message (the harness refused the seat's own write). -->

# ENG3 r2 report: G1, G1b, G2, G3 and G4 pass; G5 fails (2 scope events, fixed by a move)

The write of `artifacts/eng3/REPORT.md` was refused in r1 ("Subagents should return findings as text"), so this message is the report, to be filed verbatim. The ledger is `artifacts/eng3/LEDGER.md` (r2 sections appended).

**Verdict:** E2, E3 and Amendment 2 are implemented. Device results are unchanged except the waived bool `.numpy()` row. Under the registered rule (ACCEPT only when G1–G5 all pass), the round is **not ACCEPT as written**, because G5 recorded two out-of-set writes. They were found by my own scope check and fixed by a move.

## Gates

- **G1: PASS.** `g3/G1_cpu_E`: 76/76 passed with `CUDA_VISIBLE_DEVICES` empty. It covers:
  - `tc.tensor(cpu)`, `.numpy()`, `zeros` (public + `_core_zeros`), `full` (`_core_full`), `clone` (`_core_clone`), `.to("cpu")` (`_core_to`) and `.astype` on all 18 A1.4 pairs;
  - the 18 refused pairs;
  - `backward()` (the public route to `full` and `clone`);
  - the E4 `fill_` refusal.

  The no-CUDA proof uses an LD_PRELOAD counter of all 32 libcudart functions the module imports (`artifacts/eng3/tools/cudart_count.c`). It shows `host_calls` 0 across every E2 member and dtype; the positive control shows `control_calls` 4.
- **G1b: PASS.** Bool `.numpy()` equals the input exactly. Cases: n = 1, 5, 8, 33 and 1000; all-True; all-False; a seeded random pattern.
  - CPU, CUDA hidden: 8/8 (inside `G1_cpu_E`).
  - GPU: 8/8 cuda rows (inside `G2_gpu_E`, line G3_E).
- **G2: PASS.** 18/18 dispatched pairs show 0 host/device mismatches, and the NumPy E3 reference also has 0 mismatches. The 18 refused pairs give the same refusal text on host and device. `g3/G2_gpu_E`: 121/121.
- **G3: PASS.**
  - Per test ID, B vs E: CPU 3368/3368 identical; GPU 3368/3368 identical. Each build ran with its own sources in the worktree.
  - Device-path check:
    - Bh vs E (every E2 member): 91/93 keys identical. The other 2 are the A2.2 waived bool `.numpy()` rows. All raw bool storage bytes are identical.
    - B vs E (public paths): 24/24.
    - B vs Bh: 24/24 (Bh is a faithful stand-in for B).
- **G4: PASS.** 5/5 mutants killed on their named assertions (table below).
- **G5: FAIL (2 events).** `scripts/eng3_build_arms.sh` and `scripts/eng3_line_steps.sh` were written in `scripts/`, but the writable set there is "new scripts/eng3_*.py".
  - Fix: moved (not deleted) to `artifacts/eng3/tools/` (sha256 `7ea42ee8…`, `26975818…`).
  - Everything else is in the set: no git writes, nothing deleted, GPU only through forge-gpu-run, every suite run reported "new worktree writes []".

## Census and E4 (A1.2)

- **Core members that called CUDA on a host tensor (E2 set):** zeros, full, from_host, to_host, clone, to (host→host via clone), astype.
- **E4 refusals:** one. `fill_` already refuses ("fill_: CUDA arrays only", unchanged), measured in `BOUNDARY.json`:
  - Build B, CUDA hidden: refuses an empty host tensor; a 3-element host tensor cannot even be built ("CUDA error at from_host").
  - Build E: refuses both; the values are unchanged.
- **The 14 compute entries with a device check one call level down:** export_rows, export_rope_rows, splice_rows, evict_rows, dda_raycast, raster_winner_scatter_min, raster_winner_resolve, bake_back_project, bake_cosine_blend, inpaint_island_passes, w8a16_matmul, apa_selective_attention_sp, apa_selective_bwd_variant, bp_kernel_4_diagnostic.
- The other 158 compute entries are unchecked. All 172 are out of scope and belong to the HOST-COMPUTE-HMM successor item.

## Diff summary (`artifacts/eng3/source.diff`; `before/` vs `E_src/`)

- **`kernels.cu`** (+204/−8, all in the core section):
  - New ENG3 host block beside the DISPATCH_FLOAT macro: `host_f32_to_f16`, `host_f16_to_f32`, `host_f32_to_bf16`, `host_bf16_to_f32`, `host_ld`/`host_st`, `host_rintf`, `host_to_u8`, `host_astype_into`.
  - zeros: `std::memset`.
  - full: a host loop using `st<T>`'s rule.
  - from_host / to_host: `cuda_check_last` moved into the device branch.
  - clone: `std::memcpy`.
  - astype: host branch.
  - Device branches are unchanged statement for statement.
- **`bindings.cpp`** (+39):
  - A1.1 private entries `_core_full`, `_core_zeros`, `_core_clone`, `_core_to`. They are not exported by `__init__.py`; no public name was added.
  - A2.1: one Bool branch in `tensor_to_numpy`.
- `core.h` and `__init__.py`: unchanged.

## Module sha256s (`artifacts/eng3/build/MODULES.txt`)

| Build | sha256 |
|---|---|
| B | 9d8be988977d7a197805d7daa9c347008e34e487f30c140687cb5fbbe10d0af6 |
| E | 9f070b6f4427f2b704f09ec96ef2097791a61e0c02b65e0f384169eac1d4986d (installed in the worktree package now) |
| Bh (B engine + A1.1 hooks only) | 17f305eb0f18f646d9cde9544ff8509c662e997fd5fcb333c091ef8dd989571c |
| m01 | 1cbc36ae16400b7f047381b191636fb4e2b4bc13a19932351daa93ae01d43566 |
| m02 | 5521073d262197962e8785de0b4ae47d8da874e5e1ebaa1ce07798aee0e841a4 |
| m03 | 9c3530d7e80bf92bdc0432bd03f574e79f6606f8f807b2c08b749a5d7100d75b |
| m04 | 72a72c1a8e2fd6d26568adec29a65580922e047c0fdb7b06f7c1b0935a4d08ce |
| m05 | 4e7d9fc9259d963981b83060e821887582b3ee41bd90702df1ffc7962d01588a |

## A1.4 pair list (`A14_PAIRS.json`; NDArray::astype at 2478a4d, lines in `before/kernels.cu`)

- **cast_kernel** (748-751), 6 pairs: f32→f16, f32→bf16, f16→f32, f16→bf16, bf16→f32, bf16→f16.
- **cast_to_u8_kernel** (732-738), 3 pairs: f32/f16/bf16→u8.
- **cast_from_u8_kernel** (740-746), 3 pairs: u8→f32/f16/bf16.
- **`dt == dtype` → clone** (728), 6 pairs: f32, f16, bf16, u8, int64 and bool, each to itself.
- **Refused by DISPATCH_FLOAT** (567-573), 18 pairs: int64 or bool paired with any other dtype.

## G2 per pair (host vs device bits, on same-width unsigned views)

| Pair | Elements | Host/device mismatches |
|---|---|---|
| f32→f16 | 10064 | 0 |
| f32→bf16 | 10064 | 0 |
| f16→f32 | 10033 | 0 |
| f16→bf16 | 10033 | 0 |
| bf16→f32 | 10037 | 0 |
| bf16→f16 | 10037 | 0 |
| f32→u8 | 10064 | 0 |
| f16→u8 | 10033 | 0 |
| bf16→u8 | 10037 | 0 |
| u8→f32 | 10256 | 0 |
| u8→f16 | 10256 | 0 |
| u8→bf16 | 10256 | 0 |
| f32→f32 | 10064 | 0 |
| f16→f16 | 10033 | 0 |
| bf16→bf16 | 10037 | 0 |
| u8→u8 | 10256 | 0 |
| int64→int64 | 10006 | 0 |
| bool→bool | 10002 | 0 |

The bf16 sources include the NaN payloads 0x7F81 and 0xFFC1, made through a torch DLPack import plus `_core_to`.

**L55 (one constructed value per rule; host = device = reference in every row):**

| Rule | Input | Result |
|---|---|---|
| f32→bf16 tie | 0x3F808000 | 0x3F80 |
| f32→bf16 tie | 0x3F818000 | 0x3F82 |
| f32→f16 subnormal tie | 1.5·2⁻²⁴ | 0x0002 |
| f32→f16 subnormal tie | 2.5·2⁻²⁴ | 0x0002 |
| f32 NaN with payload → f16 | 0x7FC01234 | 0x7FFF |
| f16 NaN with payload → f32 | 0x7C01 | 0x7FFFFFFF |
| u8 .5 tie | 0.5 | 0 |
| u8 .5 tie | 254.5 | 254 |
| u8 clamp | 255.5 | 255 |

The NaN rules were measured by the exploration probe on Build B (`probe/device_probe.npz`): every half NaN becomes 0x7FFFFFFF, and every float NaN becomes 0x7FFF in both half and bf16.

## G3 per-ID result and intended changes

| Run | IDs | Outcomes | B vs E differences |
|---|---|---|---|
| CPU (`G3_cpu_compare.json`) | 3368 | 935 passed / 2130 failed / 288 skipped / 15 error on both builds | 0 |
| GPU (`G3_gpu_compare.json`) | 3368 | 3266 passed / 40 failed / 47 skipped / 15 error on both builds | 0 |

- Intended-change list (`G3_INTENDED.json`, registered 08:05:45, before the after-runs at 08:29 and later): classes S_source_pin, H_host_path and BOOL_numpy (A2.5).
- Observed intended changes: **none**. The 27 test_eng2 bool rows are passed→passed. The 1508 IDs in source-pin files are unchanged.
- Excluded from the suite and listed: `test_terrain_render.py` (L40) and `test_eng3_host.py` (run separately).

## Mutant table (`MUTANTS.json` registered 07:49:16; `MUTANT_RESULTS.json`)

| Mutant | Gate | Named test | Outcome | Marker in named junit | Other failing IDs |
|---|---|---|---|---|---|
| m01_from_host_checks_cuda | G1 | test_g1_cpu_tensor_numpy[float32] | failed ("CUDA error at from_host") | yes | 70/76 CPU rows, including no_cuda_calls |
| m02_host_astype_truncates | G2 (GPU) | test_g2_host_equals_device[float32-bfloat16] | failed | yes | 7 on GPU, including f16→bf16 and L55 |
| m03_host_clone_cudamemcpy | G1 | test_g1_cpu_no_cuda_calls | failed: host_counts {'cudaMemcpy': 13} | yes | 14 (clone, to and backward rows) |
| m04_u8_no_clamp | G2 (GPU) | test_g2_host_equals_device[float32-uint8] | failed | yes | 7 on GPU (all three →u8 pairs, L55) |
| m05_bool_numpy_float | G1b | test_g1b_bool_numpy[cpu-n5_pattern] | failed (dtype float32; values 2.37e-38…, the garbage the lead measured) | yes | 12 CPU rows; 16/16 G1b rows on GPU |

Result: 5/5 killed.

## GPU lines

Every line: the module sha256 was checked before each attempt (L103), dry run returned 78, and the inner command was under `timeout`. Results are in `artifacts/eng3/gpu/<line>/RESULT.json`.

| Line | forge exit | Inner exit | Timeout | Module | Log sha256 (prefix) |
|---|---|---|---|---|---|
| probe_B | 0 | 0 | no | 9d8be988 | a0d2c551 |
| G3_B1 | 0 | 0 | no | 9d8be988 | ef731c91 |
| G3_B2 | 0 | 0 | no | 9d8be988 | 1adff853 |
| devpath_Bh | 76, then 0 after 300 s | 0 | no | 17f305eb | 370b1235 |
| G3_E | 0 | 0 | no | 9f070b6f | 5a09a798 |
| mut_m02 | 0 | 0 | no | 5521073d | e3d4b757 |
| mut_m04 | 0 | 0 | no | 72a72c1a | d0c43058 |
| mut_m05 | 0 | 0 | no | 4e7d9fc9 | c72930e4 |

The first probe attempt was stopped by me while it waited for the lease (receipts in `gpu/probe_B_attempt1_stopped`). Total GPU wall time including lease waits was about 57 minutes; actual GPU use was about 25 minutes.

## Deviations

- **First CPU before-run read E's sources.** `g3/cpu_B_srcE` ran Build B's module with E's sources in the worktree. It is kept; G3 uses reruns with each build's own sources (`scripts/eng3_srcswap.py`).
- **L56(b) tool repairs (2 of 3):**
  - The cudart counter segfaulted because `RTLD_NEXT` cannot see libcudart; fixed with `dlopen(RTLD_NOLOAD)`.
  - The suite runner stopped at collection errors; fixed with `--continue-on-collection-errors` on both builds. The first try is kept as `cpu_B_try1_interrupted`.
- **L56(a) slip:** v1 of `eng3_g3_compare.py` waived only one of the two keys that read bool through `.numpy()`. Its output is kept as `CMP_Bh_vs_E_hooks_v1_keylabel.json`; v2 waives exactly those two keys.
- **Build E was rebuilt twice:** once after A2.1, and once after a comment-only fix of the NaN-constant comment; the second rebuild gave the same sha256. The first mutant trees were superseded before any build (`mutants_superseded_v1`, `/tmp/eng3_old`).
- **Bh build added.** Build B has no entry for zeros/full/clone/to, so Bh serves as their device reference. It matches B 24/24 on the public paths.
- **G5 event:** the two `.sh` helpers, as described under Gates.
- **Home paths:** pytest tracebacks in `gpu/G3_E/run.log`, `gpu/G3_B2/run.log` and `logs/eng3_suite_cpu_*.log` contain `$HOME` paths (`~/.local` site-packages). They are left unedited so the logged sha256s still match; screen them before any push.

## Prior art

- IEEE 754-2008: binary16 and binary32, and conversion with roundTiesToEven.
- The bfloat16 format (Google Brain) and the RNE formulation that adds 0x7FFF plus the low bit (as in c10::BFloat16; unverified, lead to check: "c10 BFloat16 round_to_nearest_even").
- CUDA Math API `__float2half_rn` / `__float2bfloat16_rn`, and the PTX ISA `cvt` instruction.
- LD_PRELOAD with `dlsym` interposition (ltrace-style call counting).
- Private test hooks (CPython `_testcapi`, `torch._C._*`; unverified).
- NumPy `bool_` is one byte per element.
- In-repo: TX3-F1B (fork worktree `tensor_fork_apa_wt_tx3f1`, `tc/numpy_export.h`, not merged) fixed the same bool `.numpy()` defect earlier. Cited in the code comment; this round's branch is not a copy of it.
- Mutation testing: DeMillo, Lipton and Sayward (1978).

Mine: the integer formulation of each device rule, the value census, the counter-based no-CUDA proof, the host/device same-bits harness, and the Bh device reference.

## L109

Helpers run as `python3 -B -P`. None of these names is a standard-library module name (checked against `sys.stdlib_module_names`): eng3_run, eng3_shamap, eng3_census, eng3_seat_check, eng3_probe_device, eng3_gpu_line, eng3_suite, eng3_make_arms, eng3_srcswap, eng3_devpath, eng3_g3_compare, eng3_mutant_results, eng3_boundary, test_eng3_host.

## Files written

Paths are under `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_eng3/`.

- **Product:**
  - `tensor_cuda/src/kernels.cu`
  - `tensor_cuda/src/bindings.cpp`
  - the installed module `tensor_cuda/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so` (Build E)
- **Tests:** `tensor_cuda/tests/test_eng3_host.py`
- **Scripts:** `scripts/eng3_run.py`, `eng3_shamap.py`, `eng3_census.py`, `eng3_seat_check.py`, `eng3_probe_device.py`, `eng3_gpu_line.py`, `eng3_suite.py`, `eng3_make_arms.py`, `eng3_srcswap.py`, `eng3_devpath.py`, `eng3_g3_compare.py`, `eng3_mutant_results.py`, `eng3_boundary.py`
- **`artifacts/eng3/` files:** `LEDGER.md`, `preflight_L01_r2.txt`, `A14_PAIRS.json`, `MUTANTS.json`, `MUTANT_RESULTS.json`, `G3_INTENDED.json`, `BOUNDARY.json`, `boundary_rows.jsonl`, `source.diff`
- **`artifacts/eng3/` directories:**
  - `before/` and `E_src/` (3 files each)
  - `probe/device_probe.npz`
  - `tools/`: `cudart_count.c`, `cudart_count.so`, `eng3_build_arms.sh`, `eng3_line_steps.sh`
  - `build/` (22 files, including `BUILD_E.json` and `MODULES.txt`)
  - `g3/` (65 files)
  - `gpu/` (59 files)
  - `devpath/` (14 files)
  - `mutants/` (7 files)
  - `mutants_superseded_v1/` (7 files)
- **Builds:** `tensor_cuda/build-eng3/{B,E,Bh,m01…m05}`, 650 files, 224 MB.
- **Logs:** `logs/eng3_*` (build, suite and line logs).
- **Scratch, left in place (leave-and-list):** `/tmp/eng3_tmp`, `/tmp/eng3_old`, `/tmp/eng3_src_{Bh,m01…m05}`.
