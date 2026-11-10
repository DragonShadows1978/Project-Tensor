# ENG1-A2 — ACCEPT

Both mechanical fixes meet the registered acceptance criteria. Evidence classes:
CPU unit execution, CPU measurement-instrument checks, runtime read attribution,
byte identity and a completed-child CPU audit. R2-F01 is ready for the lead's
closure rerun of the same completeness check; no further review round. R2-F02
is fixed once and closed under the order's 11-03 rule.

Dispatch: `e73fafdacdf5d542bb6e9d6b614b90563f11bd38`. Immutable order:
`orders/ENG1_A2_REV2_FIXES.md`; immutable local registration: `REGISTRATION.md`.
Execution-only routing corrections are separately preserved in `AMENDMENTS.md`.

## Acceptance receipts

| Requirement | Result | Receipts |
|---|---|---|
| Complete source-reader list | PASS: 22 selectors / 23 IDs; 19 observed selectors, all 19 listed, **0 missing** | `SOURCE_READING_TESTS.md`, `added_source_readers.json`, `q2_classify_reads.json`, `completeness.json` |
| Stage census | PASS: **2110 comparison IDs**, 2133 candidate IDs, 23 excluded IDs; no collection errors | `stage_census.exit.json`, `stage_census.stdout`, `census.json`, `../tool/ids_a1_regression.txt` |
| New seal | PASS: 20 arms, none pending; old seals retained unchanged | `stage_seal.exit.json`, `../tool/staging_seal_a2.json`, `gates.json` |
| Audit line | PASS: CPU candidate_eng1 child 0, parent 0, timeout false, 47 passed | `audit.exit.json`, `audit.stdout`, `audit/candidate_eng1/attempt-0001/parent_result.json` |
| CPU subset | PASS: **47 passed / 47**, 109 deselected, exact IDs and candidate pin verified | `cpu.exit.json`, `cpu_result.json`, `cpu.xml`, `cpu_origin.json` |
| E3 fields on CPU | PASS: actual row return expression on 40 real emulation operand rows, including all 4 nonfinite rows; 7 finite-mask/pattern branch checks | `e3_rows.exit.json`, `e3_rows_cpu.json` |
| Scope and isolation | PASS: only test_eng1.py in required diff; only e3_distance/e3_measure definitions; src/ byte-identical | `scope.json`, `test_eng1.diff`, initial/final maps, `verify.exit.json`, `gates.json` |

New seal: `artifacts/eng1/tool/staging_seal_a2.json`.
SHA256: `ad630ee05f7a9db02aa1c52f95d1be4df6ea4f0447226740355d7f45be7a7564`.

## R2-F01: exclusion list and runtime completeness

`SOURCE_READING_TESTS.md` supersedes A1's table without changing it. The original
16 selectors and paths remain; REV2's six missing selectors (seven exact IDs)
are appended. The two parameterizations of test_compiled_d96 are individually
listed. The 14-script and 5-script helper-mediated read sets are expanded into
explicit paths from REV2's trace. `complete_list.py` credits those receipts.

Unchanged `stage.py` collects 156 ENG1 IDs, 47 CPU IDs, 155 non-E6 IDs, 53 terrain
IDs, 2133 candidate regression IDs and **2110** paired regression IDs. The 23
excluded exact IDs are recorded in `census.json`. No numerical test target or
expected exit was changed.

`q2_source_readers.py` and `q2_classify_reads.py` in this directory are
byte-identical to REV2's originals. `trace.py` redirects only their scratch path;
their __file__-relative outputs land in A2. Attribution, classification,
write/subprocess fencing, pytest plugins and test inputs remain unchanged.
The same A/B/C groups cover all 49 regression files and 2133 exact test IDs with
recorded outcomes. Classifier output (legacy key names retained):

```json
{"a1_count":22,"observed_readers":19,"in_both":19,"observed_not_in_a1":[],"blocked_events":36}
```

The three listed selectors observed only by name are:

| Selector | Reason |
|---|---|
| `test_apa_sp1_1.py::test_registration_and_every_existing_kernel_body_pinned` | Fails first with FileNotFoundError for `artifacts/apa_sp1/sp1_1_before/tensor_cuda__src__kernels.cu`, before the live source read. |
| `test_apa_sp3_a6.py::test_a6_bridge_all_old_kinds_future_edits_and_receipt_bytes` | Fails first with FileNotFoundError for `artifacts/apa_sp3/build/manifest.json`, before later source closure checks. |
| `test_apa_sp4g_a2.py::test_a2_source_pins_pre_branch_norm_rope_v_and_scale` | Passes; directly reads `/mnt/ForgeRealm/GraftRepository/core/gemma4_tc.py`, outside the worktree-only audit trace. |

`TRACE_COVERAGE.md` gives full IDs and evidence paths. The 36 refused operations
are 35 subprocess launches and one cache copy, as in REV2. All three trace pytest
exits are 1: A has 9 failed/139 passed; B has 7 failed/104 passed; C has 1481
failed/131 passed/214 skipped/48 errors. These are preserved read-census outcomes,
not a passing regression-suite claim. Completeness covers observed CPU reads;
source reads after an initial failing CUDA call remain unobserved. Imports,
traceback rendering, temporary tree walks and artifacts/orders reads are excluded
by the unchanged classifier; its pyproject.toml collection read is retained.

## R2-F02: nonfinite measurement fields

The row adds `nonfinite_pattern_equal` using exactly the prescribed NumPy
expression (converted to bool for JSON), and
`engine_distance_from_emulation_finite`, the maximum absolute difference over
jointly finite elements, or null when none exist. `e3_distance(...,
finite_only=True)` supplies the latter; its default retains the old all-elements
distance behavior. Existing nonfinite counts and original distance fields remain.
E3 stays measurement only, with the same accuracy targets and assertions.

The CPU instrument uses all 40 actual E3 emulation operand rows and evaluates
the live e3_measure return expression directly, bypassing only GPU tensor setup.
Its got operand is explicitly a copy of the CPU emulation, **not engine output**.
All rows record matching patterns; all-finite rows record distance 0; the four
all-nonfinite backward float16 near_eps rows record finite distance null:
LayerNorm 64/64, RMSNorm 64/64, BatchNorm1D 64/64, InstanceNorm2D 96/96.
Seven further branch checks exercise mixed finite pairs, signed infinities,
infinity versus NaN, displaced NaNs, matching all-nonfinite vectors, all-finite
vectors and an empty pair. Thus the change is exercised on the real emulation
data as well as discriminating boundary operands. Actual engine E3 pattern/distance
values remain unmeasured on CPU; no GPU E3 result is fabricated.

## Staging, seal and audit

`stage.py arms` refreshed exactly the 20 staged selection copies of this same
test_eng1.py; no other staged test, package or module changed. Its unmodified
seal routine verified equality of every staged test copy to the live file.
`staging.py` first writes its draft only to scratch, then publishes a new A2 seal
with the A2 order, dispatch hash, diff base and adapter/registration hashes.
The unchanged tool files, selectors, census files, registration files and complete
arm maps remain bound by the seal. The staged census filenames retain the
instrument-defined a1 suffix; earlier receipt directories are unchanged.

The audit uses unchanged start_gpu_run.py, run_gpu_line.py, run_gpu_tests.py and
forge_gpu_run_standin.py. A2's explicitly activated sitecustomize.py selects the
new seal/order and restricts the output guard to `artifacts/eng1/a2/audit`.
Every origin, module pin, sealed-file, exact-ID, JUnit, child and parent check
still runs. START, origin, invocation, child and parent receipts agree on the
new seal. `parent_result.json` was recorded after child exit and evaluates PASS;
the separate `audit.exit.json` records parent exit 0. Its publication hard-link
temporary is retained and listed in the file inventory.

## Scope, preserved failures and files

The required read-only command was run verbatim:

```sh
git diff 7cf46bd5 -- . ':!artifacts/eng1' ':!orders' ':!logs'
```

Only `tensor_cuda/tests/test_eng1.py` is touched: +15/-3, entirely in
e3_distance and the e3_measure row return. Full diff: `test_eng1.diff`.
The complete 296-file tensor_cuda before/after maps differ only at that file;
`tensor_cuda/src/` is byte-identical. Initial map SHA256:
`9ca809ee32c41925a632d570cf5e343461c86bb5eb24a352683aa5ba239d9e81`.
All 311 protected earlier receipts/old seals have unchanged bytes. All unrelated
tracked files and tool sources are unchanged. Staged arm maps differ only at
the 20 test_eng1.py selection copies. Baseline pin remains
`73ededa0e73357c8578e58e5eadd87df9f2f3b2d64d631eba32499f2fdb9523e`;
candidate pin remains
`3e6d5805fb8b89ebfeb2b32c173fd097ef1cfe4a01b3349dc177d4b0c8c0ad53`.

Two instrument setup failures are retained: initial trace A failed before
collection because TMPDIR lay outside its relocated fence; initial audit startup
refused a nested TMPDIR before launching a child. Only scratch routing was
corrected. `AMENDMENTS.md`, `q2_readers_A.*` and `audit_preflight_refused*` retain
the failures, including verbatim errors. The audit's refused startup receipts
were moved within A2, never deleted. No thresholds, engine code or sealed
instrument sources were changed to obtain a pass.

Every written deliverable and retained scratch file is itemized with SHA256 in
`FILES_WRITTEN.json` and `FILES_WRITTEN.md`; the inventory files themselves have
explicit self-hash exclusions. This includes files rewritten byte-identically by
stage.py, all refreshed staging copies, A2 receipts/scripts/maps/tables/report,
the JSON list, E3 test file and new seal. Scratch remains under
`/tmp/eng1_a2_work` and `/tmp/eng1_a2_audit`. Nothing needs moving to DELETE_ME.
No git writes, subagents, GPU execution, engine edits, other-test edits or
non-scratch deletes; no process outside this seat was stopped. Initial read-only
orientation commands used inherited environment; all collection/test/audit runs
used hidden CUDA, one-thread settings and foreground timeout supervision.

## Prior art

- REV2 (2026) q2_source_readers.py and q2_classify_reads.py: used unchanged,
  including attribution/classification and the six missing selectors/paths.
  PEP 578 (Steve Dower, 2019) supplies runtime audit hooks; pytest supplies
  runtest attribution and deselection. New here: mechanical list completion
  and path-only output routing.
- REV2 R2-F02 (2026) supplies the exact nonfinite-pattern expression and
  jointly finite distance requirement; NumPy supplies isfinite, where,
  array_equal(equal_nan=True), masking and max. New here: two census fields
  and an optional finite-only distance mode. No new engine algorithm.
- ENG1/GN1 A8 (2026) supplies sealed staging and completed-child supervision;
  GNU timeout/nice, Python subprocess/runpy/sys.settrace/sitecustomize and
  SHA256, NIST FIPS 180-4 (2015), are existing instruments. New here: A2
  order/seal/output selection; existing integrity/evaluation checks remain.

Existing REV2 prior-art annotations stand. External literature is
**unverified — lead to check**: search `PEP 578 Dower 2019`,
`NumPy array_equal equal_nan isfinite`, `FIPS 180-4 2015`.
