# BWD-PROMO-F1 A1 execution report

G1–G5 passed for the seat's build, CPU, staging and scope checks. F03's GPU catching check and the rebuilt package’s GPU suite await the lead. [Exact lead commands](STAGED_LINES_PF1.md)

| Gate | Verdict | Evidence |
|---|---|---|
| G1 build | PASS | CUDA 12.6 Release sm_89 with the R5D flags. The native source census has one delta, replay in bindings.cpp. The Python census has one delta, the checkpoint docstring. [Build and source hashes](BUILD_PF1.json) · [Build log](../../logs/bwd_pf1_build.log) |
| G2 fixes | PASS for seat | The F02 unit test caught `AssertionError: checkpoint replay cast leaked grad mode` on the baseline and passed on the rebuild. F01 and F04 passed source inspection. F03 collected and skipped with `CUDA disabled for CPU lane` on all three packages. Its GPU expectations are baseline pass, planted copy fail with `AssertionError: scale gradient is zero for frozen x`, and rebuild pass. [F02 smoke](../../logs/bwd_pf1_f02_smoke.log) · [CPU checks](G2_CPU_PF1.json) · [Lead commands](STAGED_LINES_PF1.md) |
| G3 CPU per ID | PASS | 4,265 records per arm. F02 changed from failed to passed. All other outcomes, phase results and per-test skip reasons matched. [Comparison](G3_PER_ID_PF1.json) |
| G4 staging | PASS | All 20 dry admissions exited 78 with `"would_start": true`. The 17 candidate file lines cover every `test_bwd_*.py` in both suites. The runtime census uses the R5F TF32 rule and the R5E pytest user-site fix. All 20 lines check package SHAs before admission. [Commands](STAGED_LINES_PF1.md) · [Dry receipts](GPU_DRY_PF1.json) · [Staged pins](STAGED_PINS_PF1.json) |
| G5 scope | PASS | Product diff restricted to replay, the wrapper docstring, and two append-only test files. Both orders and the baseline package rehashed. Zero GPU runs, git commands or workspace deletions. [Scope receipt](SCOPE_PF1.json) · [Every workspace file written](WRITES_PF1.txt) |

Core SHA-256 is `2820bdb0cd1ccfcede2b758289e86a3d0a4297d3ee2edf416ef186293f1998b9`. The candidate contains the rebuilt core and the edited wrapper. The planted package changes only `tc_signal/_norms.py` and returns a zero scale gradient when x is frozen. [Product diff](DIFF_PF1.patch) · [One-hunk plant diff](PF1M03_DIFF_PF1.patch) · [All package file SHAs](MODULES_PF1.json)

| CPU package | Passed | Failed | Skipped | Setup errors | Collection errors | Pytest exit |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 1,383 | 2,172 | 694 | 15 | 1 | 1 |
| Rebuilt | 1,384 | 2,171 | 694 | 15 | 1 | 1 |

The collection error on both arms was `ValueError: invalid literal for int() with base 10: 'worker'` in `test_selector_accuracy.py`, which parses the harness argv at import. The prior R9 candidate receipt has the same shared failure and error counts. [Baseline transcript](../../logs/bwd_pf1_suite_baseline.log) · [Rebuilt transcript](../../logs/bwd_pf1_suite_candidate.log) · [Prior R9 receipt](G6_PER_ID_R4B_R9.json)

## Deviations

Early read-only instrumenting missed two paths, truncated one combined read, and omitted the shell timeout on the initial order, rules and memory reads. Targeted, bounded reads recovered the workspace evidence. One authored runner repair before launching it restored shell-quoting losses and routed pytest scratch under `/tmp/bwd_pf1_*`. The build uses a verified source copy because the existing CMake target writes the module into its configured source directory. The F02 test uses the dict return from REV4AX so it catches the native cast error directly. [Ledger](LEDGER_R4B.md) · [Registration](REGISTRATION_PF1.md)

## Prior art

- Chen et al. (2016) and PyTorch autograd (Paszke et al., 2019) supply activation recomputation and saved-input revision checking. PF1 clarifies which TensorCUDA calls receive those Storage checks and narrows its PyTorch comparison.
- Stroustrup (1994) documents C++ scope-based cleanup. PF1 places the replay cast in the existing restoration region and takes the invalid dict probe from REV4AX (2026).
- Ulyanov et al. (2016) and Ioffe and Szegedy (2015) supply the normalization affine derivative. F03 reuses the R4a r5d fp64 fixture and freezes x. DeMillo, Lipton and Sayward (1978) supply mutation analysis. The lead specified the zero-gradient mutation.
- Project-Tensor MAINCHK3, R4b G6, R4a R5E/R5F and System-Claude GW3 (2026) supply per-ID reporting, isolated runs, the pytest environment correction, the TF32 census and dry admission. PF1 adds skip reasons, per-file staging, and the three F03 package arms. NIST FIPS 180-4 (2015) supplies SHA-256 for the pins.

External references are unverified — lead to check “Chen sublinear memory cost 2016”, “PyTorch reentrant checkpoint saved tensor version counters”, “Stroustrup exception cleanup 1994”, “Ulyanov instance normalization 2016”, “Ioffe Szegedy affine gradients 2015”, “DeMillo Lipton Sayward mutation analysis 1978”, and “NIST FIPS 180-4 SHA-256 2015”.
