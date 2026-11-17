# ENG3-F3 r2 report — Amendment 1

G1 PASS. G3 PASS (r1 kills retained and verified). G4 PASS. G2 CPU PASS
with exactly 24 registered inherited exceptions. G2 GPU STAGED, NOT RUN.
Full acceptance remains pending the lead's GPU runs and narrow resumed
cross-model review; no promotion performed.

## Result and evidence

Suite-run evidence: G1_r2.json and host/G1_H_r2/RESULT_r2.json record
Build H pytest exit 0: **267 passed, 1 skipped**, 45 deselected by the unchanged
CPU selection. The skip is the moved GPU-index test on this no-device host.
None of the amended 24 rows skipped; every one asserted RuntimeError naming
cudaGetDevice and its hook against the real runtime.

Per-ID regression evidence: G2_HOST_COMPARE_r2.json records all 24 amended IDs,
their actual Build G OverflowError types and complete shape/dtype messages,
their Build H caught RuntimeError messages, and the original r1 G/H outcomes.
The same amended test tree ran against both modules: on G, those 24 tests
now fail because the early query behavior is absent; on H they pass. This is
the registered CPU-only exception, not a clean Build G pytest run. The other
**230 inherited host IDs are identical**. The 14 new/moved IDs retain their
r1 G/H outcomes, including the six intended F04 G-fail/H-pass rows.
Host G therefore records exit 1: 237 passed, 30 failed, 1 skipped. H records
exit 0: 267 passed, 1 skipped. Host census and collection-error sets agree.

Frozen engine-suite regression evidence: fresh G/H runs each exercised
**3,368 IDs**, with **zero per-ID differences** and no collection-file set
differences. Each pytest exit is 1: 815 passed, 1,946 failed, 349 error,
258 skipped. This establishes parity of inherited outcomes on this CPU
host; it does not claim a green engine suite. Raw outcomes are in
`g3_r2/g3-cpu_{G,H}/outcomes_r2.json`. The corrected comparison is
staged_r2/g3-cpu/COMPARE_r2.json. G2_r2.json combines engine and host parity.
No logs or timings were compared as regression outcomes.

G3 evidence is carried from r1, not rerun: all three mutant modules still
match the recorded SHA256s, all original test receipts are byte-identical,
and the moved killing-test bodies are unchanged. G3_r2.json points to the
original three kills. No new mutant or mutation coverage is claimed.

Scope evidence: G4_SCOPE_r2.json, SCOPE_CHECKS_r2.json and before/after cwd
censuses. All 112 r1 artifact files are byte-identical. Product/order hashes
and earlier build metadata are unchanged; no baseline file disappeared.
The audit is scoped to this worktree and recorded commands, not the global
filesystem. No product edit, rebuild, GPU use, git command, subagent,
deletion or DELETE_ME move occurred.

## Diff summary

`tests_r2.diff`: one existing parametrized function changed. Only its 24
bare-cuda rows query the real libcudart.so.12 cudaGetDevice. Query failure
requires RuntimeError/cudaGetDevice; query success executes the existing
OverflowError checks for the full shape and dtype. The 24 CPU rows retain
all their original executable statements and assertions. No other existing
test changed. Both new F01 pinning tests now cite ENG3R3-F01 and Amendment 1
A1.2 in their docstrings; their executable bodies are unchanged.
**kernels.cu remains unchanged.**

`harness_r2.diff`: three new _r2 Python helpers and two new _r2 launchers,
based on r1; all r1 scripts and launchers remain untouched. The helpers
preserve test census, options and outcome folding, isolate r2 receipts,
and capture exception messages per amended ID. GPU staging adds explicit
parity for all 48 overflow-message rows and the moved GPU-index test inside
each inherited split lease. All 24 amended rows must take the real device
branch and pass OverflowError/shape/dtype assertions identically on G/H.
No CPU exception is allowed on GPU. All GPU rows are **untested here**.

## Staged GPU lines and expected exits

Run exactly from
`/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_eng3`:

```bash
timeout 590s env FORGE_GPU_DRY_RUN=0 bash artifacts/eng3f3/staged_r2/launch-g3-g1_r2.sh
timeout 590s env FORGE_GPU_DRY_RUN=0 bash artifacts/eng3f3/staged_r2/launch-g3-g2_r2.sh
```

Expected each: **forge exit 0, inner comparison exit 0, launcher exit 0**.
Inherited split pytest exits are separately recorded and compared per ID.
Inside these launchers:

```bash
timeout 550s forge-gpu-run --min 30 --max 540 --label eng3f3-r2-g3-g1 -- timeout 510s nice -n 19 python3 -B -P scripts/eng3f3_stage_r2.py run g3-g1
timeout 550s forge-gpu-run --min 30 --max 540 --label eng3f3-r2-g3-g2 -- timeout 510s nice -n 19 python3 -B -P scripts/eng3f3_stage_r2.py run g3-g2
```

Each launcher explicitly establishes PATH, CUDA library path, visible devices,
package state, single-thread settings, temporary directory, hash seed and
bytecode settings, and clears inherited test controls. Both module hashes
are checked before the lease, then each real host CPU prefix passes and
stops before device allocation. Same module pins are checked in workers.

This seat invoked both launchers **only with FORGE_GPU_DRY_RUN=1**.
Both actual forge/launcher exits are 78, inner exit null, GPU run false.
Both GPU logs explicitly say `state: no run` and `self_leasing: false`.
STAGED_LINES_r2.json pins launchers/test files and records GPU log SHA256s.
The lead's real finish receipts will be
`staged_r2/g3-g1/RESULT_r2.json` and `staged_r2/g3-g2/RESULT_r2.json`, with
forge/inner exits and log SHA256. Actual GPU parity is not claimed.

## Module SHA256s

No module was rebuilt or replaced. Existing build receipts remain authoritative.

- G, source 60e28eb7: `5fcd179b98b0749af0d75414d35ae0fc8519bf200364c30c743904a90fc1a711`
  — tensor_cuda/build-eng3f2/source_G/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
- H, r1 source snapshot: `ea7233854f0d67c60652a230c7ec8e2e7712c5164144a1259b6be231ba7a4e43`
  — tensor_cuda/build-eng3f3/source_H/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
- f3m01_bare_cuda_no_query: `4e9b90866b86cf24c4209b83bcaf42052dd5738bb927d7a955c166e5db911a7f`
- f3m02_query_error_as_invalid: `91d53eae7c147ba4280a5ec6827da21c83fdbd03f8d844d2c981f26769b0b1c8`
- f1m02_malloc_unchecked: `264493c91b61bf03c8027082a62dc1cb1d86fa8c5381ee7605b391f6a257879b`

## Deviations

- Initial read-only discovery commands lacked timeout wrappers and completed
  immediately. Discovery also named nonexistent tools/ and G1_H_final; the
  actual paths were then read. No gate or write depended on those slips.
- **Own-script repair 1/3:** initial CPU engine comparison returned FAIL because
  the inherited metadata detector counted the growing dispatch transcript
  logs/eng3f3_r2.log as a test write. Progress/tool records during the yielded
  foreground gate caused that change. The corrected comparison excludes only
  this shim-owned log and preserves raw observed_writes. All other writes
  still fail. Raw suite receipts and initial FAIL comparison/inner receipts
  are retained; no test result, assertion, threshold or exception list changed.
  Recomparison, rather than rerunning unchanged tests, corrects attribution.
  Original INNER_RESULT_r2.json remains FAIL; the explicit
  INNER_RESULT_corrected_r2.json and COMPARE_r2.json record the correction.
- No other own-script repairs. Foreground runs used timeout; yielded exec
  sessions resumed with write_stdin yield_time_ms 300000. The tool relay used
  shorter waits for required progress updates, rather than silent background work.
- No deletes or DELETE_ME moves. Retained temporary roots are listed below.

## Prior art

CUDA Runtime API cudaGetDevice and local public tensor_empty/P1T-FIX4 (2026):
reuse the real current-device query/error branch; ours is amended assertion
placement, not a new selection algorithm. NumPy nominal-size refusal and
P1T-FIX2 (2026): cite the lead's existing F01 ruling, without a new byte rule.
ENG3-F2/F3 harness (2026), pytest hooks (2004+), SHA-256 (NIST 2001): reuse
pinned imports, phase folding and per-ID comparisons; ours is r2 exception
receipt assembly and explicit GPU overflow parity. r1 mutation prior art:
DeMillo, Lipton and Sayward (1978), unchanged tests and receipts. External
literature unverified — lead to check: CUDA Runtime API cudaGetDevice;
NumPy array is too big; pytest hooks; mutation testing 1978.
Annotations are at the test/helper sites and in LEDGER_r2.md. No new numerical
algorithm or optimization is implemented or proposed.

## Every file written

The complete workspace list follows and is also in FILES_WRITTEN_r2.txt.
FILES_r2.sha256 seals stable retained files, excluding itself and the active
shim transcript. Both amended tests, every new helper/launcher/receipt and
the shim transcript are included. No r1 file is omitted by replacement.
Temporary roots: `/tmp/eng3f3_host_r2`, `/tmp/eng3f3_suite_r2`,
`/tmp/eng3f3_stage_r2`; SCRATCH_WRITTEN_r2.txt records the retained contents.
Python/pytest automatic temporary files under /tmp may have been cleaned by
their tools; all non-cache retained workspace writes are inventoried.

```text
artifacts/eng3f3/FILES_WRITTEN_r2.txt
artifacts/eng3f3/FILES_r2.sha256
artifacts/eng3f3/G1_r2.json
artifacts/eng3f3/G1_r2.log
artifacts/eng3f3/G2_HOST_COMPARE_r2.json
artifacts/eng3f3/G2_cpu_r2.log
artifacts/eng3f3/G2_host_G_r2.log
artifacts/eng3f3/G2_r2.json
artifacts/eng3f3/G3_r2.json
artifacts/eng3f3/G4_SCOPE_r2.json
artifacts/eng3f3/LEDGER_r2.md
artifacts/eng3f3/ORDER_r2.md
artifacts/eng3f3/REGISTRATION_r2.md
artifacts/eng3f3/REGISTRATION_r2.sha256
artifacts/eng3f3/REPORT_r2.md
artifacts/eng3f3/SCOPE_AFTER_r2.json
artifacts/eng3f3/SCOPE_BEFORE_r2.json
artifacts/eng3f3/SCOPE_CHECKS_r2.json
artifacts/eng3f3/SCRATCH_WRITTEN_r2.txt
artifacts/eng3f3/STAGED_LINES_r2.json
artifacts/eng3f3/dry_g3-g1_r2.log
artifacts/eng3f3/dry_g3-g2_r2.log
artifacts/eng3f3/g3_r2/g3-cpu_G/junit_r2.xml
artifacts/eng3f3/g3_r2/g3-cpu_G/outcomes_r2.json
artifacts/eng3f3/g3_r2/g3-cpu_H/junit_r2.xml
artifacts/eng3f3/g3_r2/g3-cpu_H/outcomes_r2.json
artifacts/eng3f3/harness_r2.diff
artifacts/eng3f3/host/G1_H_r2/RESULT_r2.json
artifacts/eng3f3/host/G1_H_r2/junit_r2.xml
artifacts/eng3f3/host/G2_G_r2/RESULT_r2.json
artifacts/eng3f3/host/G2_G_r2/junit_r2.xml
artifacts/eng3f3/staged_r2/g3-cpu/COMPARE_initial_r2.json
artifacts/eng3f3/staged_r2/g3-cpu/COMPARE_r2.json
artifacts/eng3f3/staged_r2/g3-cpu/INNER_RESULT_corrected_r2.json
artifacts/eng3f3/staged_r2/g3-cpu/INNER_RESULT_initial_r2.json
artifacts/eng3f3/staged_r2/g3-cpu/INNER_RESULT_r2.json
artifacts/eng3f3/staged_r2/g3-cpu/suite_G_r2.log
artifacts/eng3f3/staged_r2/g3-cpu/suite_H_r2.log
artifacts/eng3f3/staged_r2/g3-g1/CPU_PREFIX_r2.json
artifacts/eng3f3/staged_r2/g3-g1/DRY_RESULT_r2.json
artifacts/eng3f3/staged_r2/g3-g1/GPU_r2.log
artifacts/eng3f3/staged_r2/g3-g1/prefix_G_r2.log
artifacts/eng3f3/staged_r2/g3-g1/prefix_H_r2.log
artifacts/eng3f3/staged_r2/g3-g2/CPU_PREFIX_r2.json
artifacts/eng3f3/staged_r2/g3-g2/DRY_RESULT_r2.json
artifacts/eng3f3/staged_r2/g3-g2/GPU_r2.log
artifacts/eng3f3/staged_r2/g3-g2/prefix_G_r2.log
artifacts/eng3f3/staged_r2/g3-g2/prefix_H_r2.log
artifacts/eng3f3/staged_r2/launch-g3-g1_r2.sh
artifacts/eng3f3/staged_r2/launch-g3-g2_r2.sh
artifacts/eng3f3/test_eng3_f3_before_r2.py
artifacts/eng3f3/test_eng3_host_before_r2.py
artifacts/eng3f3/tests_r2.diff
logs/eng3f3_r2.log
scripts/eng3f3_host_r2.py
scripts/eng3f3_stage_r2.py
scripts/eng3f3_suite_r2.py
tensor_cuda/tests/test_eng3_f3.py
tensor_cuda/tests/test_eng3_host.py
```
