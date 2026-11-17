# ENG3-F3 r2 ledger

- Start: SHIM-RUN 20261117T080246Z-3662740 (logs/eng3f3_r2.log).
- Read house mirror, project/local AGENTS, both orders, REV3 mechanism and
  r1 reports/tests/harness/receipts. Quick memory lookup had no relevant hits.
- pytest 9.0.2 and Python 3.12.3 checked with --version before use.
- SCOPE_BEFORE_r2.json: 19,166 metadata entries; hashes for 112 r1 artifact
  files, 31 product/order files, and both live modules. G/H pins match.
- REGISTRATION_r2.md sealed before gates. Acceptance/failure forks and exact
  24 exceptions inherited from r1, no post-result allowance.

## Deviations

- Initial read-only discovery commands lacked shell timeout wrappers;
  completed immediately. All commands after reading the F3 order use timeout.
- Read-only discovery named nonexistent tools/ and G1_H_final; rg/ls reported
  those missing paths. Correct paths read next; no mutation, no gate affected.
- Own-script repairs: 0/3 at registration.

## Prior art

Public tensor_empty/P1T-FIX4 (2026) and CUDA Runtime API cudaGetDevice: reuse
real device-query/error branching in the 24 amended assertions. NumPy nominal
size refusal and P1T-FIX2 (2026): cite existing F01 ruling in test docstrings.
ENG3-F2/F3 (2026), pytest hooks (2004+), SHA-256 (NIST 2001): reuse pinned
module imports, per-ID outcomes and phase folding. Ours: r2 receipt assembly
and explicit GPU overflow parity. External literature unverified — lead to
check: CUDA Runtime API cudaGetDevice; NumPy array is too big; pytest hooks.
No new numerical algorithm, selection rule, optimization or product behavior.

## Execution receipts

- G1_r2 host H: exit 0, 267 passed / 1 no-device GPU skip. All 24 amended
  rows passed RuntimeError/cudaGetDevice assertions. G2 host G: exit 1,
  237 passed / 30 failed / 1 skipped, exactly the registered 24 + six F04.
  Actual unexpected G exception types/messages and caught H messages are
  preserved per ID in host/*_r2/RESULT_r2.json.
- CPU suite fresh G/H runs completed: 3,368 IDs each, no outcome differences,
  no collection-file differences. Each pytest exit 1: 815 passed, 1,946
  failed, 349 error, 258 skipped. These are inherited regression outcomes,
  not a green engine suite.
- Own-script repair 1/3: comparison initially returned FAIL because the
  inherited metadata detector attributed the growing shim transcript
  logs/eng3f3_r2.log to tests. Progress/wait tool records grow that file.
  Preserve initial comparison/inner receipts; corrected comparison retains
  raw observed_writes, excludes only this active dispatch log, and still
  rejects every other write. Test results, IDs, errors and thresholds unchanged.
  Recompare existing raw suite receipts; no pytest rerun needed for attribution.
- Corrected engine comparison PASS; initial FAIL receipts retained as
  staged_r2/g3-cpu/{COMPARE,INNER_RESULT}_initial_r2.json. The original
  INNER_RESULT_r2.json also remains FAIL; INNER_RESULT_corrected_r2.json and
  COMPARE_r2.json record the attribution correction. G2_r2.json combines
  corrected engine parity with G2_HOST_COMPARE_r2.json.
- Both re-staged GPU lines: actual dry forge/launcher exits 78, inner null,
  real G/H host prefixes pass, GPU log says no run / self_leasing false.
  Lead's real expected forge/inner/launcher exits all 0. New GPU stage also
  requires 48 overflow rows + moved GPU index test all pass on both modules,
  with no exception for the 24 bare-cuda rows. Actual GPU work NOT RUN.
- Scope checks: all 112 r1 artifacts byte-identical; product files unchanged;
  only amended function executable branch and two new pinning docstrings
  changed. CPU executable path and moved killing test bodies unchanged.
  Live module and r1 test-receipt hashes for all three mutant kills match.
  G4_SCOPE_r2.json records the cwd census: no unexpected writes/deletions.
- Own-script repairs final count 1/3. No files deleted or moved to DELETE_ME;
  retained scratch /tmp/eng3f3_host_r2, /tmp/eng3f3_suite_r2,
  /tmp/eng3f3_stage_r2. No builds, GPU, git commands, or subagents.
