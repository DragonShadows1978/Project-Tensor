# APA-P1-D report (filed by the lead, verbatim from the seat's final message; the harness refused the seat's own write)

**APA-P1-D: ACCEPT on every gate. G1 to G10 all PASS and G5b HELD.** Packed storage is now the SP default, with the dense route behind an explicit opt-out. This round changes no device code, and Build B's SASS equals Build A's on all 778 kernels. The packed margin table is measured, frozen and pinned in code.

One required file is missing: the harness refused `artifacts/apa_p1d/REPORT.md` from this seat ("Subagents should return findings as text, not write report files"). The full report is this message; please file it as REPORT.md.

## Gates
| Gate | Verdict | Receipt |
|---|---|---|
| G1 policy | **PASS** | S4: 40/40 CPU tests. L1 PASS (4,608/4,608 bitwise). L2 PASS (30/30). |
| G2 object | **PASS** | All R-C fields exist (CPU test). L4: owned buffers are exactly 68 B per key per KV head, and no K-shaped tensor is reachable. |
| G3 wrapper | **PASS** | R-D matching rules pass on CPU and device. The delta helper equals native `_C.apa_sp2_delta` on all 116 boundary fixtures; both refuse the same domain. |
| G4 callers | **PASS** | Census re-run at this commit. 80 current-source rows: 78 unchanged and correct, 2 explicit `storage="dense"`. Plus 93 new rows from this round's test and harness. 0 unassigned. Same (file, entry) multiset as S3's 80. |
| G5 margins | **PASS** | Registered and pinned (18:42 EST) before L5 (18:50 EST). Both entries measured and frozen, sha pinned in `apa_packed.py`. |
| **G5b** | **HELD** | 4:64 is +2.4e-7 relative to the dense entry; 4:128 is −4.1e-7. |
| G6 prefill | **PASS** | L3 was run before R-E was written. Branch implemented and tested (CPU test; L2 `RE_prefill_on_packed_refused`). |
| G7 SASS | **PASS** | B vs A: 778 kernels, 0 differences. A vs A1: 778 kernels, 0 differences after mapping nvcc's path hashes. |
| G8 mutants | **PASS** | 9/9 killed, each by its named assertion. |
| G9 regression | **PASS** | CPU: 689 IDs, 0 differences. L6 GPU: 44 files, 2,078 IDs, 0 differences. No intended changes listed. |
| G10 scope | **PASS** | Writes only in the writable set plus `/tmp/apap1d_*`. GPU only through forge-gpu-run. No git writes, nothing deleted. Exception: REPORT.md was not written (see top). |

## Diff summary
Tracked files: 5 changed, +60/−7.
- `tensor_cuda/tensor_cuda/apa_sp2.py`:
  - `quantize_sp2_keys(k, bulk_bits, *, storage=None)` implements R-A and R-B.
  - The wrapper sends a `PackedSpKeys` to the packed route.
  - A packed table with a dense Kq raises.
  - The old body moved verbatim into `_dense_sp`, so the dense branch is byte-identical (L55).
- `tensor_cuda/tensor_cuda/__init__.py`: the `apa_sp2` import/export lines now also export `apa_packed`, `PackedSpKeys` and `PackedRegisteredMargins`.
- `tensor_cuda/CMakeLists.txt`: `option(TC_APA_PACKED ... ON)`. Test hooks stay off.
- `scripts/apa_pidv.py:269` and `scripts/apa_sp2_gpu.py:53`: `storage="dense"` added at the quantizer calls only.

New `tensor_cuda/tensor_cuda/apa_packed.py`:
- the storage policy;
- the R-B eligibility check with its fallback reason;
- `derive_delta`, the one shared delta helper;
- `PackedSpKeys`, built on `ApaSpq1Workspace`, which holds only a weak reference to K. Its exact-K identity is the source object, an append generation and the storage revision. It has explicit `append`, `invalidate` and `check_source`;
- `PackedRegisteredMargins` and its loader;
- `packed_attention`, which uses r2 with partition carry;
- route receipts through `last_route()` (R-H).

There are no C++ changes: `bindings.cpp`, `packed_launcher.cuh` and `tensor.cuh` are untouched.

## Modules and SASS
- Build A (e2fb85a1, `-DTC_APA_PACKED=ON`, attempt 2): `ebe9ec8d82271e643dee427345a635dc01321aabba3708b3acb80ee30239ea91`
- Build B (this tree, default configuration): `8728e67e26027dbafe8a61592decb38e68171fa7b5634239ce0af7897c4e8c4e`
- SASS B vs A: **PASS, 778 kernels, 0 differences** (`SASS_COMPARISON.json`).
- SASS A vs A1 (closes S3's open item): 0 differing kernels in 778 (`SASS_A_VS_A1_r2.json`).
  - The raw first pass showed 520 name-only differences, because the two trees sit at different paths and nvcc derives symbol hashes from the path.
  - Mapping only `__nv_static_N__<hex>`, `_GLOBAL__N__<hex>` and `_cu_<hex>_<n>` (names and call targets) leaves no difference. No instruction text was normalized.
  - The source changes between f7e6f49b and e2fb85a1 (eb3, eb4, conv, kernels, bindings, ops) change no kernel body.

## CPU and per-ID results
- `test_apa_p1d_policy.py`: 40 passed, on the stub `_C` and tensors.
- P1's G3 set, before (arm A in the e2fb85a1 tree) against after (arm B_v2 in this worktree):
  - 689 IDs, 0 differences, identical collections.
  - Both arms: FAILED 125, PASSED 410, ERROR 15, SKIPPED 139. These are historical failures, retained as they were.
  - The outcomes are also identical to A1's port column.
- No ID needed L12's replacement predicate.

## Mutants (9/9 killed)
| Mutant | Test | Expected failure, found |
|---|---|---|
| epsilon_as_delta | `forwards_delta_causal_sinks_and_r2` | "delta must be derived from epsilon and e_q" |
| dense_table_on_packed | `dense_table_on_packed_keys_is_refused` | DID NOT RAISE |
| two_bits_to_four | `default_two_bit_takes_dense_…` | `assert isinstance(out, FakeTensor) and out.tag == "dense_kq"` |
| stale_packed_keys | `stale_generation_is_refused` | DID NOT RAISE |
| causal_offset_dropped | `forwards_…` | "is_causal must be forwarded" |
| sinks_dropped | `forwards_…` | "sinks must be forwarded" |
| dense_kq_retained | `owns_no_k_shaped_tensor_…` | "packed object retains a K-shaped tensor" |
| env_ignored | `environment_dense_selects_dense` | "env dense must select dense" |
| explicit_packed_falls_back | `explicit_packed_that_cannot_be_honoured_raises` | DID NOT RAISE |

## Caller dispositions
- Census command, run for each of `.`, `/mnt/ForgeRealm/GRAPA-Native-LLM`, `/mnt/ForgeRealm/GraftRepository` and `/mnt/ForgeRealm/Project-Voice`. It is S3's command byte for byte:
  `CUDA_VISIBLE_DEVICES= timeout 40s rg -n --sort path --hidden --no-ignore -g '*.{py,pyi,ipynb,cpp,cu,cuh,h,c,sh,js,ts,rs}' -g '!**/.git/**' -g '!**/__pycache__/**' -g '!**/.venv/**' -g '!**/node_modules/**' -g '!**/build*/**' -g '!**/Fork/**' -g '!**/WorkTree/**' '\b(apa_selective_attention_sp(_epsilon|_d1_packed|_d1)?|quantize_sp2_keys|apa_spq1_(dump|rotate)|apa_spo2_dump|apa_sp2_(delta|scores|baseline_thresholds)|apa_sp1_1_baseline_diagnostics)\b' <root>`
- Matches: Project-Tensor 986, GRAPA 11 (archived patch), GraftRepository 0, Project-Voice 0. The full output is in `CALLER_CENSUS_COMMANDS_OUTPUT.txt`.
- Python call rows: 420 in total.
  - Current source: 80 S3 rows plus 93 new rows.
  - Historical/frozen artifact rows: 247, listed and never edited.
- The two quantizer controls now pass `storage="dense"`.
- A packed object passed to a raw dense `_C` entry (sp_epsilon, sp, sp2_scores) raises TypeError (L2).

## Packed table beside the dense table
Packed table sha `1897fedd2a8d05945e34433c75a7afd3b33feb64e0a6b2ca410d611d18868e98`; registration sha `025b1a5116ceae3e0f272d5a8823970c31d0d65e709b785cd81e6a2cbbda1538`; quantizer `turboquant_lloyd_max_rotated_fp32_packed_tq4_r2`; 32 classes.

| Entry | Table | count | max | p99.9 | mean | e_q |
|---|---|---|---|---|---|---|
| 4:64 | packed | 13,762,608 | 1.0106459616 | 0.3549634 | 0.0751212 | **1.010646104812622** |
| 4:64 | dense (5b6f8bfb…) | 13,762,608 | 1.0106457381 | 0.3549630 | 0.0751212 | 1.010645866394043 |
| 4:128 | packed | 13,762,608 | 0.8716109843 | 0.3425747 | 0.0764184 | **0.8716110587120056** |
| 4:128 | dense | 13,762,608 | 0.8716113419 | 0.3425650 | 0.0764184 | 0.8716114163398743 |

The instrument pin held: the fp64 recomputation of the packed bulk scores deviated by at most 1.12e-6.

## GPU lines (each preceded by a dry run that returned 78; no exit 76 occurred)
| Line | Attempts | Result |
|---|---|---|
| L3 | 1, exit 0 | Not bitwise equal in all 8 cases. Mismatched elements per case range from 15,298/32,640 to 601,840/1,048,576. Norms are bitwise equal on 33–49% of rows. Max difference ≤1.4e-6, except one code flip of 0.075 at D128 S2048 B2. |
| L5 | r1 exit 1 (my script had not set TC_APA_SP; nothing was measured), r2 exit 0 | 32/32 targets, then frozen. |
| L1 | 1, exit 0 | 512 cases, 4,608/4,608 bitwise equal, both sides informative. The packed default's output differs bitwise from dense in 512/512 cases, so it is really packed and not a silent fallback. |
| L2 | 1, exit 0 | 30/30. Every refusal is raised with 0 spied device calls. |
| L4 | 1, exit 0 | 6/6 cases at 68 B per key per KV head. The measured saving matches the prediction to within one 2 MiB allocation granule (≤1.05 MB short; criterion ±3 granules = 6.3 MB): 6.29 vs 7.27 MB at S2,048, 29.36 vs 29.10 MB at S8,192, 115.34 vs 116.39 MB at S32,768. No K-shaped tensor. Decode output is nonzero. |
| L6a | r1 exit 124 (outer timeout under load ~50, and my relative receipt path lost arm A's receipts); r2–r4 exit 0 after resuming | 22 files, PASS. |
| L6b | 1, exit 0 | 21 files, PASS. |
| L6c | 1, exit 0 | dlpack, 257 IDs, PASS. |

GPU time used is about 115 minutes; wall time is about 4 h 15 min.

## R-E branch
**Refuse prefill on packed keys.** A call with L > 1 raises `ValueError` whose message names `storage='dense'`, records the route, and creates no second representation.

## Deviations
- **D1.** Build A attempt 1: configure timed out at 300 s under load. `build_A/` is left in place. Attempt 2 (`build_A_r2/`) passed.
- **D2.** Builds and long lines ran detached and were waited on in the foreground with `timeout 580 tail --pid`, because the tool's foreground call limit is 10 minutes.
- **D3.** L3 and L2 ran with `--max 580 / timeout 540s` instead of the registered 600/560, to fit inside one tool call.
- **D4.** Script repairs:
  - L5 repair 1: set TC_APA_SP.
  - L6 repair 1: absolute receipt path.
  - L6 repair 2: resumable, with `--max-seconds`.
  - SASS A-vs-A1: two mapping passes.
- **D5.** In `MARGIN_REGISTRATION.json` I first typed an uncomputed sha for `apa_sp2_gpu.py`. I replaced it with the real sha before the registration was hashed and pinned, and before L5.
- **D6.** "No -D flags" for Build B was read as no feature flag. The A1 script's toolchain and path flags are identical on both builds.
- **D7.** `scripts/apa_sp2_gpu.py list` cannot run here because its registration file is git-ignored. I read production's copy read-only, after checking its sha against the tracked pin.
- **D8.** `TC_APA_SP_STORAGE=packed` is treated as the default policy, so 2-bit falls back with a reason. Only the explicit argument raises. This needs your ruling.
- **D9.** L6 arm A ran e2fb85a1's own tests and scripts from `/tmp/apap1d_treeA` (git archive).

Residuals:
- There is no multi-GPU device-mismatch test, because the machine has one GPU.
- On the device, the visible-rows guard is reached through the source-identity check.
- Model quality is not covered.

## Prior art
- **TurboQuant** (Zandieh et al., 2025, arXiv 2504.19874): the rotated Lloyd-Max 4-bit codes, unchanged.
- **BLASST** (Yuan et al., 2025, arXiv 2512.12087): the running-max criterion, inherited unchanged.
- **SP2's two-error margin argument** (Project-Tensor, 2026): delta = upward-rounded fp32(−log ε + 2·e_q). The helper re-states the native rule.
- **PyTorch-style owned storage handles and version counters** (Paszke et al., 2019): the object owns its buffers and uses the storage revision to detect in-place edits. Binding that with an append generation into the exact-K identity is mine.
- **ApaSpq1Workspace** (P1, 2026): the base the packed object is built on.
- **Mutation testing** (DeMillo, Lipton and Sayward, 1978).
- **Test doubles** (Meszaros, 2007).
- **ripgrep and Python ast**, used through S3's census.
- **cuobjdump** and P1's A1 symbol mapping.
- **The refuse-prefill branch and the policy precedence** (argument, then environment, then shipped default): no prior art known to me.
- Unverified, lead to check. Search terms: "TurboQuant 2504.19874", "BLASST 2512.12087", "PyTorch tensor version counter".

## Files written (all under `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_p1d/`)
- **Modified:**
  - `tensor_cuda/CMakeLists.txt`
  - `tensor_cuda/tensor_cuda/__init__.py`
  - `tensor_cuda/tensor_cuda/apa_sp2.py`
  - `scripts/apa_pidv.py`
  - `scripts/apa_sp2_gpu.py`
- **New code:**
  - `tensor_cuda/tensor_cuda/apa_packed.py`
  - `tensor_cuda/tests/test_apa_p1d_policy.py`
  - `scripts/apa_p1d_build.py`
  - `scripts/apa_p1d_stage.py`
  - `scripts/apa_p1d_gpu.py`
  - `scripts/apa_p1d_sass.py`
  - `scripts/apa_p1d_sass_a1.py`
  - `scripts/apa_p1d_mutants.py`
  - `scripts/apa_p1d_cpu_g3.py`
  - `scripts/apa_p1d_census.py`
- **`artifacts/apa_p1d/` (1.1 GB):**
  - `INSTRUMENTS.log`
  - `build_A/`: failed attempt, left in place
  - `build_A_r2/`, `build_B/`
  - `arms/{A,B,B_v2}/`: copied packages, no symlinks
  - `sass/`, `SASS_COMPARISON.json`, `SASS_A_VS_A1.json`, `SASS_A_VS_A1_r2.json`
  - `MARGIN_REGISTRATION.json`, `MARGIN_REGISTRATION.sha256`
  - `EXPECTED.json`, `REGISTRATIONS.sha256`
  - `mutants/` (`MUTANTS.json`, nine mutant directories, `_baseline/`, `MUTANT_RESULTS.json`)
  - `margins/`: 32 receipts and npy files
  - `packed_e_q_table.json`, `packed_e_q_table.sha256`
  - `gpu/{L1,L2,L3,L4,L5,L6a,L6b,L6c}/` with their logs, start, exit and dry-run files
  - `cpu_g3/`, `CPU_G3_COMPARISON.json`
  - `CALLER_DISPOSITIONS.json`, `CALLER_CENSUS_COMMANDS_OUTPUT.txt`
  - `BOUNDARY.json`
- **Scratch, left and listed:**
  - `/tmp/apap1d_{tmp,buildA,buildB,pytest,mut,g3,l6}`
  - `/tmp/apap1d_treeA` (about 1 GB)
  - Python caches `scripts/__pycache__/` and `tensor_cuda/tests/__pycache__/` (exempt)
