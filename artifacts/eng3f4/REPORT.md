# ENG3-F4 report

G1 PASS. G2 PASS. G4 PASS. G5 PASS. G3 STAGED, NOT RUN.
Acceptance and ENG3 promotion remain pending the lead's three G3 lines.
No further review requested by this order; no promotion performed here.

## Test results and receipts

Suite-run evidence: G1.json and host/G1_all/RESULT.json record both full
files on pinned Build H: **267 passed, 46 GPU-only skips**, pytest exit0.
The runner skips exactly 45 host GPU rows; the existing device-index test
skips for unavailable CUDA. None of the six no-device rows skips. Separate
unset-parent run: **6 passed**, pytest exit0, host/G1_unset/RESULT.json.
Before editing the same unset-parent selection failed all six on
`AssertionError: assert None == ''` (BEFORE_F01.log/xml).

Regression evidence: G2.json compares 268 host IDs and 3,368 inherited
engine IDs to F3 r2's pinned-H receipts. **Zero outcome differences, missing
IDs, or collection-file differences.** All 31 touched IDs and their before/
after outcomes are individually listed, including the GPU-only skip.
The engine suite remains RED as inherited: pytest exit1, 815 passed,
1,946 failed,349 error,258 skipped. This is per-ID parity, not a green
engine suite. Raw engine outcomes/failures: g3/g3-cpu_H/outcomes.json;
host reference: artifacts/eng3f3/host/G1_H_r2/RESULT_r2.json; engine reference:
artifacts/eng3f3/g3_r2/g3-cpu_H/outcomes_r2.json. No tests wrote outside
the registered output set; engine observed-writes list is empty.

Negative unit-test evidence: G4.json and host/G4_power/RESULT.json record
**24 failed**, pytest exit1, with CVD empty and ENG3_EXPECT_CUDA=1. All24
print `ENG3F4_BRANCH expected=1 branch=overflow query_success=False
query_rc=100` before failing. Each failure names the registered expectation
and cudaGetDevice disagreement. This proves registration binds on this
no-device runtime; no incorrect GPU-path pass is accepted. Raw transcript:
G4_power.log. This expected failure is G4 PASS, never a green pytest run.

Scope evidence: G5.json, SCOPE_BEFORE/AFTER.json, GIT_BEFORE/AFTER.json.
No unexpected write or missing file; no git administrative metadata change.
H live product-source hashes and both pinned modules match build receipts.
Only the two allowed test files changed, plus new allowed scripts/artifacts
and the shim-owned logs/eng3f4_r1.log. No product edit, rebuild, git command,
subagent, GPU lease, deletion or DELETE_ME move. Audit covers workspace
metadata, product/module hashes, AST boundaries and git admin metadata;
it is not a global filesystem audit.

## Diff summary

- ENG3R4-F01: each of six no-device rows uses the existing subprocess helper
  with CUDA_VISIBLE_DEVICES=''. Original exception/message assertions moved
  intact into child mode no_device_query, with per-row success marker and
  checked child exit. All six remain collected on GPU parents.
- ENG3R4-F02: for all three hooks and both cuda spellings, require exact
  `t.device == f'cuda:{current.value}'`, retaining the shape check.
- ENG3R4-F03: only the 24 bare-cuda rows now choose the expected branch from
  ENG3_EXPECT_CUDA=0/1, verify the real query agrees, and print the branch.
  Absent the variable, the documented empty-CVD CPU convention defaults0;
  all other parent environments default1. No probe-selected expectation.
  CPU rows and original OverflowError shape/dtype checks remain unchanged.
- ENG3R4-F04: unused overflow child mode removed. MemoryError child retained
  byte-for-byte at the AST level. Tests.diff records every test edit;
  DIFF_SUMMARY.json gives insertion/removal counts. No product algorithm.

New helpers stage the full H-only 14-ID F3 file plus the 24 bare-cuda rows
(38 passes required); inherited G/H g3-g1 and g3-g2 comparisons preserve
r2's split forms and add the 49-row overflow/device-index parity check.
Each comparison requires all24 registered CUDA rows to take OverflowError
with complete shape/dtype checks on both G and H. CPU-only exceptions are
not allowed in this lane. Exact label/GPU-path behavior is **not claimed
fixed by CPU execution alone**; lead G3 verification remains outstanding.

## Staged GPU lines and expected exits

Run unchanged from worktree root:

```bash
timeout 590s env FORGE_GPU_DRY_RUN=0 bash artifacts/eng3f4/staged/launch-gpu-tests.sh
timeout 590s env FORGE_GPU_DRY_RUN=0 bash artifacts/eng3f4/staged/launch-g3-g1.sh
timeout 590s env FORGE_GPU_DRY_RUN=0 bash artifacts/eng3f4/staged/launch-g3-g2.sh
```

Each expected: **forge0, inner0, launcher0**. The first requires38 passing
IDs, including six no-device child rows and exact labels; comparison forms
require G/H per-ID parity plus all49 host check IDs passing. Inherited split
pytest exits/counts remain separately recorded and need not be clean.
Each actual forge command is `timeout 550s forge-gpu-run --min 30 --max 540
--label eng3f4-<line> -- timeout 510s nice -n 19 /usr/bin/python3 -B -P
<root>/scripts/eng3f4_stage.py run <line>`.

Launchers establish the complete environment via env -i, set
ENG3_EXPECT_CUDA=1, explicitly set libraries/device/thread/temp/hash state,
and inherit no pytest/ENG3 controls. Tests/helpers/launchers/frozen split
and module records are SHA256-pinned in INPUT_PINS.json. G/H module SHA256s
are checked before lease and again in workers. Each arm executes the real
CPU host prefix, stopping before the first device call. Full environments,
command argv and pins are in each staged/<line>/dry/PRELEASE.json.

This seat ran all three with **FORGE_GPU_DRY_RUN=1 only**: each forge78,
launcher78, inner null, GPU run false. Logs say `state: no run` and
`self_leasing: false`. STAGED_LINES.json records every dry log SHA256,
expected real exits and lead command. Lead finish receipts will be
staged/<line>/real/RESULT.json, including actual exits and GPU log SHA256;
COMPARE.json and INNER_RESULT.json hold underlying comparisons. Dry and
real output roots are separate; existing outputs are never reused silently.

## Module pins

H: ea7233854f0d67c60652a230c7ec8e2e7712c5164144a1259b6be231ba7a4e43
G: 5fcd179b98b0749af0d75414d35ae0fc8519bf200364c30c743904a90fc1a711
Paths and verified current hashes are in MODULES.json. No module rebuilt.

## Deviations

1. Own-script repair **1/3**: G4 classifier's startswith missed23 progress-
   prefixed branch lines. AMENDMENT_1.md records the regex correction;
   G4_initial.json, CPU_RUN.log and before/eng3f4_cpu_initial.py preserve
   the original classification/helper. Reclassified existing raw tests;
   no assertions, expected count, branch predicate or gate narrowed and no
   unchanged suite rerun. CPU_SUMMARY.log and CPU_SUMMARY_REPAIRED.log
   record PASS. CPU_RUN.log intentionally retains initial G4 false.
2. Read-only ls discovery of absent output/scratch and guessed summary
   paths returned2; paths resolved from live artifacts. No gate affected.
3. No other own-script repair, forbidden action or retained DELETE_ME move.
   Long foreground command resumed with 300000ms waits and no intervening
   status message. Temporary files/roots retained in SCRATCH_WRITTEN.txt.

## Prior art

Python subprocess environment isolation (PSF,2003+) and local F1 run_child
(2026): reuse child/environment mechanics; ours is per-row no-device
isolation. NVIDIA CUDA Runtime API cudaGetDevice (year unverified) and
P1T-FIX4/F1 (2026): test the existing current-device rule with exact equality.
Pytest hooks/parametrization (Krekel et al.,2004+) and local F3 r2 harness
(2026): reuse phase folding, split census, real CPU prefixes, per-ID parity
and configured expectations; ours is F4 branch registration and receipts.
SHA-256 (NIST,2001): reuse integrity pinning. No new numerical algorithm,
selection algorithm, data structure, proof technique or optimization.
No prior art known to me for this specific artifact/prose composition beyond
the named local harness. External literature **unverified — lead to check**:
Python subprocess env; pytest hook API; CUDA Runtime API cudaGetDevice;
NIST SHA-256. Code-site comments and LEDGER.md contain the same attributions.

## Every file written

FILES_WRITTEN.txt lists every workspace file, including this report and
manifests. SCRATCH_WRITTEN.txt lists every retained scratch file and root.
FILES.sha256 seals stable workspace outputs, excluding itself and active
shim transcript. The complete workspace list follows:

```text
artifacts/eng3f4/AMENDMENT_1.md
artifacts/eng3f4/BEFORE_F01.log
artifacts/eng3f4/BEFORE_F01.xml
artifacts/eng3f4/CPU_COMMANDS.json
artifacts/eng3f4/CPU_RUN.log
artifacts/eng3f4/CPU_SUMMARY.log
artifacts/eng3f4/CPU_SUMMARY_REPAIRED.log
artifacts/eng3f4/DIFF_SUMMARY.json
artifacts/eng3f4/DRY_g3-g1.log
artifacts/eng3f4/DRY_g3-g2.log
artifacts/eng3f4/DRY_gpu-tests.log
artifacts/eng3f4/FILES.sha256
artifacts/eng3f4/FILES_WRITTEN.txt
artifacts/eng3f4/G1.json
artifacts/eng3f4/G1_all.log
artifacts/eng3f4/G1_unset.log
artifacts/eng3f4/G2.json
artifacts/eng3f4/G2_engine.log
artifacts/eng3f4/G2_host.log
artifacts/eng3f4/G4.json
artifacts/eng3f4/G4_initial.json
artifacts/eng3f4/G4_power.log
artifacts/eng3f4/G5.json
artifacts/eng3f4/GIT_AFTER.json
artifacts/eng3f4/GIT_BEFORE.json
artifacts/eng3f4/INPUT_PINS.json
artifacts/eng3f4/LEDGER.md
artifacts/eng3f4/MODULES.json
artifacts/eng3f4/REGISTRATION.md
artifacts/eng3f4/REGISTRATION.sha256
artifacts/eng3f4/REPORT.md
artifacts/eng3f4/SCOPE.log
artifacts/eng3f4/SCOPE_AFTER.json
artifacts/eng3f4/SCOPE_BEFORE.json
artifacts/eng3f4/SCOPE_FINAL.log
artifacts/eng3f4/SCRATCH_WRITTEN.txt
artifacts/eng3f4/STAGED_LINES.json
artifacts/eng3f4/TOUCHED_IDS.json
artifacts/eng3f4/before/eng3f4_cpu_initial.py
artifacts/eng3f4/before/test_eng3_f3.py
artifacts/eng3f4/before/test_eng3_host.py
artifacts/eng3f4/g3/g3-cpu_H/junit.xml
artifacts/eng3f4/g3/g3-cpu_H/outcomes.json
artifacts/eng3f4/host/G1_all/RESULT.json
artifacts/eng3f4/host/G1_all/junit.xml
artifacts/eng3f4/host/G1_unset/RESULT.json
artifacts/eng3f4/host/G1_unset/junit.xml
artifacts/eng3f4/host/G2_host/RESULT.json
artifacts/eng3f4/host/G2_host/junit.xml
artifacts/eng3f4/host/G4_power/RESULT.json
artifacts/eng3f4/host/G4_power/junit.xml
artifacts/eng3f4/staged/g3-g1/dry/CPU_PREFIX.json
artifacts/eng3f4/staged/g3-g1/dry/GPU.log
artifacts/eng3f4/staged/g3-g1/dry/PRELEASE.json
artifacts/eng3f4/staged/g3-g1/dry/RESULT.json
artifacts/eng3f4/staged/g3-g1/dry/prefix_G.log
artifacts/eng3f4/staged/g3-g1/dry/prefix_H.log
artifacts/eng3f4/staged/g3-g2/dry/CPU_PREFIX.json
artifacts/eng3f4/staged/g3-g2/dry/GPU.log
artifacts/eng3f4/staged/g3-g2/dry/PRELEASE.json
artifacts/eng3f4/staged/g3-g2/dry/RESULT.json
artifacts/eng3f4/staged/g3-g2/dry/prefix_G.log
artifacts/eng3f4/staged/g3-g2/dry/prefix_H.log
artifacts/eng3f4/staged/gpu-tests/dry/CPU_PREFIX.json
artifacts/eng3f4/staged/gpu-tests/dry/GPU.log
artifacts/eng3f4/staged/gpu-tests/dry/PRELEASE.json
artifacts/eng3f4/staged/gpu-tests/dry/RESULT.json
artifacts/eng3f4/staged/gpu-tests/dry/prefix_H.log
artifacts/eng3f4/staged/launch-g3-g1.sh
artifacts/eng3f4/staged/launch-g3-g2.sh
artifacts/eng3f4/staged/launch-gpu-tests.sh
artifacts/eng3f4/tests.diff
logs/eng3f4_r1.log
scripts/eng3f4_close.py
scripts/eng3f4_cpu.py
scripts/eng3f4_host.py
scripts/eng3f4_scope.py
scripts/eng3f4_stage.py
scripts/eng3f4_suite.py
tensor_cuda/tests/test_eng3_f3.py
tensor_cuda/tests/test_eng3_host.py
```

Retained scratch:

```text
/tmp/eng3f4_bootstrap.py
/tmp/eng3f4_cpu
/tmp/eng3f4_make_harness.py
/tmp/eng3f4_stage
```
