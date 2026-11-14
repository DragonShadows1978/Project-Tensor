# ENG1-A1 — ACCEPT at G1–G4

All four registered gates pass. Ready for the lead’s narrow cross-model review of the five fixes before G5. Evidence classes: source identity, CPU unit execution, CPU IEEE arithmetic emulation, and CPU tooling audit. GPU execution, an actual M15 numerical kill, and promotion remain outside this seat’s results.

Dispatch: `3b6a5a867f75a399a563ddf9d693133ec938552b`; G1 diff base: `5031c765`. Immutable order: `orders/ENG1_A1_REV1_FIXES.md`; immutable local registration: `REGISTRATION.md`.

## Gates

| Gate | Result | Receipts |
|---|---|---|
| G1 scope | PASS | `gates.json`, `tensor_cuda_before.json`, `tensor_cuda_after.json`, `test_eng1.diff` |
| G2 CPU | PASS | `candidate_attempt2_parent.json`, `candidate_attempt2_pytest.json`, `candidate_attempt2.xml`, `M16_ow_zero_parent.json`, `M16_ow_zero_pytest.json`, `M16_ow_zero.xml`, `review_reproductions.json` |
| G3 isolation | PASS | `isolation_baseline.json`, `isolation_candidate.json`, their `_origin.json` receipts, `origin_refusal.json`, `SOURCE_READING_TESTS.md`, `IMPORT_SITES.md` |
| G4 staging | PASS | `../tool/STAGED_GPU_COMMANDS.sh`, `../tool/staging_seal_a1.json`, `audit_final.log`, `../tool/audit_a1_final/candidate_eng1/attempt-0001/parent_result.json` |

G1 changes only `tensor_cuda/tests/test_eng1.py` outside the order’s artifact/log exclusions. The complete initial/final tensor_cuda maps differ only at that file; `tensor_cuda/src/` is byte-identical. Candidate module in both its original build and staged arm remains `3e6d5805fb8b89ebfeb2b32c173fd097ef1cfe4a01b3349dc177d4b0c8c0ad53`. Review receipts and the old seal are unchanged. Old seal SHA256: `d1d55f4134f941db3118e139f5bf21702af4917d01964de08edc4dd4f3e206cc`. The full test-file diff is `test_eng1.diff`.

G2 collects **156 ENG1 IDs**, including both E2 witnesses and all four OW-zero refusals; the candidate CPU subset is **47 passed**,109 deselected, child exit0. Every new refusal ID passes on the candidate. `../tool/ids_a1_eng1_candidate.txt` and `../tool/ids_a1_audit.txt` list exact collected/CPU IDs.

## R1-F01: correction to the round 1 E3 readout

Round 1’s n·u numbers are **accuracy targets (not error bounds)**. This report supersedes the bound wording in the read-only round 1 report. The numerical targets are unchanged. Longest-path counts omit conditioning, cancellation, subnormal intermediates and accumulation; no E3 accuracy result gates acceptance.

`test_eng1.py` now supplies a NumPy primitive tape for the same forward/VJP graphs as nn.py/ops.cpp, rounding at every primitive and gradient accumulation. Stored operands are read independently of engine outputs. Non-GroupNorm classes store intermediate results in their half dtype; GroupNorm upcasts input/affine work to float32, then casts output and input VJP back. Means are sum plus scalar scale; generic sums use fp32 serial coordinate order, and trailing sums use the source’s fp32 256-lane tree. Pow is correctly rounded fp32 in this IEEE reference; CUDA powf implementation error remains separately measurable. Batch/Instance/GroupNorm var() has its own mean node; LayerNorm reuses its centered value. E2’s widened pow VJP is emulated in fp32 with its final dtype cast.

Each of the **40 E3 IDs** records engine error vs float64, emulation error vs float64, and engine distance from emulation, plus nonfinite counts and the rounding used. Distances are null when either vector is nonfinite; the counts retain that result. `e3_emulation.json` contains all40 CPU emulation rows. Its engine columns explicitly say not executed/G5; no engine measurement is fabricated.

| Review case (forward, near_eps) | Emulation max absolute error vs float64 | Frozen accuracy target | Absolute difference from review |
|---|---:|---:|---:|
| RMSNorm float16 | 0.015658268723244273 | 0.0084315781451264459 | 2.78e-17 |
| LayerNorm bfloat16 | 0.20484508346333996 | 0.11953872186366855 | 2.78e-17 |

Both reproduce within the pre-registered absolute tolerance5e-12. `arithmetic_receipts.py` writes these receipts and checks ties/subnormals/NaNs; `emulation_instrument_checks.json` additionally records hand VJP and reduction-order checks.

Not claimed fixed: the IEEE emulation has nonfinite near_eps float16 backward values in LayerNorm(64), RMSNorm(64), BatchNorm1D(64), InstanceNorm2D(96). Correct final half pow-gradient casting can itself overflow before later norm-graph multiplication. The census measures this composed range limitation; no norm class changes.

## R1-F02: subprocess binding and source-reader class

Before pytest, every staged child prepends its selected arm’s pinned package parent to inherited PYTHONPATH, then starts a fresh Python origin process with that exact environment and cwd. The process records tensor_cuda.__file__, extension path and extension SHA256. Any mismatch refuses before test imports. The runner saves `origin_check.json`, embeds it in invocation/child records, and the parent/scorer independently validate paths, hash, exit, cwd and first PYTHONPATH entry. Audit mode skips CUDA API queries.

The review’s original preload/plain-python-subprocess scenario was rerun in fresh CPU parents for both arms (`probe_isolation.py`, credited to `../rev1/d4_child_probe.py`). Both unbound subprocesses reproduced exit1, `ModuleNotFoundError: No module named tensor_cuda`. With binding, both exit0 and use their own pin:

| Arm | Origin / module pin | Result |
|---|---|---|
| baseline | `../tool/arms/baseline/tensor_cuda/tensor_cuda/__init__.py`; `73ededa0e73357c8578e58e5eadd87df9f2f3b2d64d631eba32499f2fdb9523e` | origin preflight0; review scenario0 |
| candidate | `../tool/arms/candidate/tensor_cuda/tensor_cuda/__init__.py`; `3e6d5805fb8b89ebfeb2b32c173fd097ef1cfe4a01b3349dc177d4b0c8c0ad53` | origin preflight0; review scenario0 |

`origin_wrong_pin.json` / `origin_refusal.json` exercise rejection with an intentionally wrong expected pin: `subprocess pin mismatch`, before pytest. `IMPORT_SITES.md` lists every live tool/*.py PYTHONPATH/sys.path/preload hit with disposition.

The following16 exact source-path reader IDs form their own class. They remain in candidate_regression; baseline_regression deselects them. Parent/scorer remove this class from paired regression comparison. Candidate collection2133; numerical comparison2117. `SOURCE_READING_TESTS.md` and `../tool/source_reading_tests.json` give each ID’s explicit/expanded paths, including helper-mediated source hashing. Their candidate execution is staged for G5; this seat collected them without running the full regression suite.

- `tensor_cuda/tests/test_apa_sp1_1.py::test_registration_and_every_existing_kernel_body_pinned`
- `tensor_cuda/tests/test_apa_sp3.py::test_every_preexisting_source_and_kernel_body_pinned`
- `tensor_cuda/tests/test_apa_sp3.py::test_d96_prefill_contract_and_dense_pin`
- `tensor_cuda/tests/test_apa_sp3.py::test_d96_splitk_contract_and_dense_pin`
- `tensor_cuda/tests/test_apa_sp3_a4.py::test_original_model_and_scorer_bytes_unchanged`
- `tensor_cuda/tests/test_apa_sp3_a6.py::test_interposer_shell_new_kind_policy`
- `tensor_cuda/tests/test_apa_sp3_a6.py::test_a6_bridge_all_old_kinds_future_edits_and_receipt_bytes`
- `tensor_cuda/tests/test_apa_sp4g_a2.py::test_a2_registration_precedes_edits_and_a1_receipts_closures_preserved`
- `tensor_cuda/tests/test_apa_sp4g_a3.py::test_a3_registration_preservation_and_real_prefix_geometry`
- `tensor_cuda/tests/test_apa_sp4g_a4.py::test_a4_registration_preservation_and_transitive_gate`
- `tensor_cuda/tests/test_apa_sp4g_a5.py::test_a5_propagation_ungated_and_registration_preserved`
- `tensor_cuda/tests/test_apa_sp4g_a7.py::test_a7_registration_grid_rails_anchors_and_preservation`
- `tensor_cuda/tests/test_apa_sp4g_a2.py::test_a2_source_pins_pre_branch_norm_rope_v_and_scale`
- `tensor_cuda/tests/test_apa_sp4g_a2.py::test_a2_runner_preserves_bounds_and_uses_isolated_driver`
- `tensor_cuda/tests/test_apa_sp4g_a3.py::test_a3_runner_bounds_and_no_historical_source_changes`
- `tensor_cuda/tests/test_apa_sp4g_a7.py::test_a7_foreground_shell_safety_and_rail_isolation`

## R1-F03: non-power-of-two upstream and M15

Both positive scalar witnesses use the original E2 tolerance derivation: `(u_dtype + 2^-20)*abs(float64_reference) + half_min_subnormal`. The fp32 chain includes the registered <=4ulp powf assumption and two products, followed by one final dtype rounding. No tolerance was chosen from device results.

| Dtype | Stored x | Stored g | p | Late-cast error | Early-cast error | Derived absolute tolerance |
|---|---:|---:|---:|---:|---:|
| float16 | 0.1492919921875 | 0.30078125 | 3 | 4.4278567656874657e-07 | 1.4816003385931253e-05 | 9.8690635996212528e-06 |
| bfloat16 | 0.0002994537353515625 | 0.30078125 | 3 | 1.0960832241835305e-10 | 3.560529648893862e-10 | 3.1615316569940429e-10 |

`e2_witnesses.json` records stored values, analytic derivative, rounding ladders and +/-4ulp power sensitivity (the displayed late/early results stay unchanged). The float16 witness was selected by a finite CPU IEEE scan before any GPU run; the bfloat16 witness is the review’s exact construction.

M15 mutates only bfloat16 to cast `p*x^(p-1)` early, before upstream multiplication. It is registered in `MUTANTS.json`, built from a fresh scratch source copy with CUDA12.6 and-j2, and its CPU import origin is pinned. Build receipts/diff: `../mutants/M15/{configure.json,build.json,receipt.json,mutation.diff}`. Engine pin: `87fe59c9f40211e7b0f53ae3bc91e99455856f1f4a5e83964f0d22eee441a041`. **Actual engine kill not run: it requires GPU.** The CPU arithmetic predicts the new bfloat16 ID kills it; the staged M15 line and registered killer bind the future G5 run.

## R1-F04: independent OW=0 refusal and M16

The new geometry `(0,1,6,2)`, kernel(3,3), stride(1,1), padding(0,0) gives OH=floor((6-3)/1)+1=4 and OW=floor((2-3)/1)+1=0. All four public callers use `_e5_call` -> binding -> make_dims before allocation/launch. N=0 keeps this CPU-reachable without a device.

M16 weakens only `ow<=0` to `ow<0`. It builds from a separate fresh source copy; build receipts/diff: `../mutants/M16/{configure.json,build.json,receipt.json,mutation.diff}`. Engine pin: `c597a30c3195ce5507ad8ea7e9e2a15ff7350457c6d8ef460952cd564dc34c27`. The actual pinned mutant executed and **failed all four registered IDs** with `Failed: DID NOT RAISE <class RuntimeError>`, child exit1. The candidate passed each ID. `M16_ow_zero_pytest.json`, JUnit and parent exit receipt name every execution:

- `tensor_cuda/tests/test_eng1.py::test_e5_refuse_cpu[im2col-ow_zero]`
- `tensor_cuda/tests/test_eng1.py::test_e5_refuse_cpu[col2im-ow_zero]`
- `tensor_cuda/tests/test_eng1.py::test_e5_refuse_cpu[avg_pool2d-ow_zero]`
- `tensor_cuda/tests/test_eng1.py::test_e5_refuse_cpu[max_pool2d-ow_zero]`

## R1-F05: reshape caller correction

Correct count: **37 engine calls on32 lines, plus9 NumPy calls**. `CALLER_CENSUS.md` preserves the review’s complete46-call itemization with credit; `caller_census.json` independently rechecks the live source. “32 lines” follows the review’s physical lines containing `reshape(` convention. An AST expression’s start can precede its method token (functional.py189–190); counting expression-start lines gives33 and is not this readout. Round 1’s “30 engine callers” statement is superseded here; its read-only report remains intact.

## Staging and seals

`../tool/STAGED_GPU_COMMANDS.sh` contains all24 commands. Every child runs the origin preflight; expected parent exit is0 for every line, meaning the completed child matches registration. The parent records the real return code and timeout flag after child exit. Bounds and expected exits are sealed in gpu_lines.json.

| Label | Expected child exit | Expected parent exit | Inner / outer seconds |
|---|---|---:|---|
| baseline_eng1 | 1 | 0 | 900 / 990 |
| candidate_eng1 | 0 | 0 | 900 / 990 |
| baseline_regression | 0,1 | 0 | 1050 / 1140 |
| candidate_regression | 0,1 | 0 | 1050 / 1140 |
| baseline_terrain | 0,1 | 0 | 1050 / 1140 |
| candidate_terrain | 0,1 | 0 | 1050 / 1140 |
| C_COPY | 0 | 0 | 420 / 510 |
| C_UBSAN | 0 | 0 | 900 / 990 |
| M01 | 1 | 0 | 420 / 510 |
| M02 | 1 | 0 | 420 / 510 |
| M03 | 1 | 0 | 420 / 510 |
| M04 | 1 | 0 | 420 / 510 |
| M05 | 1 | 0 | 420 / 510 |
| M06 | 1 | 0 | 420 / 510 |
| M07 | 1 | 0 | 420 / 510 |
| M08 | 1 | 0 | 420 / 510 |
| M09 | 1 | 0 | 420 / 510 |
| M10 | 1 | 0 | 420 / 510 |
| M11 | 1 | 0 | 420 / 510 |
| M12 | 1 | 0 | 420 / 510 |
| M13 | 1 | 0 | 420 / 510 |
| M14 | 1 | 0 | 900 / 990 |
| M15 | 1 | 0 | 420 / 510 |
| M16 | 1 | 0 | 420 / 510 |

New seal: `../tool/staging_seal_a1.json`, SHA256 `5d326628631d31fbe41d253efc02e1b2cbc45cfb773d78d55d40573aac3408a6`. Script SHA256: `0d0a4635ee830cb9b8ceabfdc9e279b116a0a386d58577f2d01ac7d1bf9d9fd9`. Twenty arms are sealed, none pending. Original seal retained byte-identical; the first A1 seal is retained as `staging_seal_a1_preliminary.json`. A1 census/ID/baseline-prediction files use their own names; original round1 receipts retain original bytes.

Final audit: `../tool/audit_a1_final/`, candidate_eng1 child0, parent0, timeoutfalse,47 passing CPU IDs. START, origin, invocation, JUnit, child and parent receipts share the final seal. `scorer_parent.json` records the scorer’s expected exit1: candidate_eng1 passes in CPU mode; the scorer explicitly refuses audit data as G5 acceptance. Other GPU lines have not run.

## Execution record and files

`LEDGER.md` records tool changes, commands/results, preserved failures and decisions. `FILES_WRITTEN.md` / `FILES_WRITTEN.json` itemize every deliverable/touched registration path and retained scratch file, with hashes; `TOOLING_A1.diff` gives the live tool/config changes. Every existing arm’s test_eng1.py copy was refreshed; package/module bytes were preserved. The two new arms copy the candidate package with their own mutant modules.

Preserved instrument failures: the first standalone CPU wrapper used an invalid pytest hook argument and failed before collecting; fixed in a fresh attempt (candidate_attempt2). An initial report pass found missing A1 copies of unchanged census lists; they were copied and resealed before the final audit. A source-census assertion initially counted AST expression-start lines rather than the registered physical-line convention; corrected without changing the required count. Receipts and ledger retain these distinctions. Initial read-only rules/memory discovery commands used TMPDIR=/tmp before scratch setup; all implementation, build, collection, test and audit runs used /tmp/eng1_a1_eng1, hidden CUDA and one-thread environments (builds-j2). No git writes, engine source edits, other-test edits, GPU use or deletions.

Nothing new needs binning in the worktree. Scratch is retained at `/tmp/eng1_a1_eng1`, `/tmp/eng1_a1_M15`, `/tmp/eng1_a1_M16` (all are /tmp-exempt). No process outside this seat was stopped.

## Prior art

- IEEE754-2019 roundTiesToEven; NumPy float16 and explicit bfloat16 high-word tie-to-even rounding: taken for per-primitive/storage emulation. New here: the five registered norm graphs, ordered reduction/VJP tape and three-way measurement rows.
- Wengert(1964), reverse-mode differentiation tape; the engine’s existing ops.cpp/autograd.cpp supplies primitive VJP and accumulation order. New: CPU arithmetic transcription for this census.
- REV1 numerics.py/coverage_gaps.py/d4_child_probe.py/CALLER_CENSUS.md(2026): credited reproduction operands, subprocess failure scenario, and itemized reshape census.
- Micikevicius et al.(2018), mixed precision; Higham(2002), unit roundoff/gamma_n. Existing E2 derivations stand; the E3 longest-path heuristic is explicitly only an accuracy target.
- DeMillo, Lipton and Sayward(1978), mutation testing: one-decision M15/M16 edits and registered killing witnesses.
- Python/PSF environment inheritance/PYTHONPATH, pytest report/deselection hooks; SHA256, NIST FIPS180-4(2015); GN1 A8/ENG1 completed-child/seal method(2026). Taken for origin and evidence binding, not an authentication claim.

External literature remains **unverified — lead to check**: IEEE754-2019 roundTiesToEven, NumPy float16, bfloat16 rounding ties-to-even, Wengert1964 differentiation, CUDA12.6 powf maximum error4ulp, DeMillo1978 mutation testing, Python PYTHONPATH subprocess inheritance. Existing round1 prior-art annotations stand.
