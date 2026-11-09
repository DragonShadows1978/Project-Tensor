# Review of all 27 tooling hunks

All frozen originals match accepted artifacts/gn1/a8 byte for byte; all 27 diff hunks reconstruct the live adapted files exactly (census_scope.json). Scope is the requested completed-child property, not GN1 contract-C equivalence.

| Hunk | File / header | Review |
|---|---|---|
| 1 | evidence.py @@ -1,29 +1,32 @@ | Header/import changes and removed GN1-only program binding; completed-child outcome checks remain. |
| 2 | evidence.py @@ -37,7 +40,7 @@ | RUNS scope only. |
| 3 | evidence.py @@ -45,37 +48,34 @@ | Per-arm source/module seals; original shared-engine design replaced. |
| 4 | evidence.py @@ -86,13 +86,19 @@ | START audit binding and ID normalization; no fabricated outcome. |
| 5 | evidence.py @@ -129,9 +135,7 @@ | JUnit dictionary initialization only. |
| 6 | evidence.py @@ -140,9 +144,6 @@ | Comment removal around duplicate failure/teardown entries; guard remains. |
| 7 | evidence.py @@ -153,14 +154,9 @@ | TMPDIR naming and deleted comments. |
| 8 | evidence.py @@ -186,8 +182,6 @@ | Timeout attribution comment removal only. |
| 9 | evidence.py @@ -208,36 +202,29 @@ | Remove GroupNorm-specific killer filters; introduce literal bracket matcher with registered mutants. |
| 10 | evidence.py @@ -247,58 +234,47 @@ | Adapt child arm/seal/census checks and baseline/mutant/candidate roles. JUnit equality and actual rc agreement retained. |
| 11 | evidence.py @@ -315,32 +291,8 @@ | Remove old runtime/program conjunctions from evaluate; final device classification now occurs in recompute_line. |
| 12 | evidence.py @@ -353,30 +305,16 @@ | Remove GN1-only package validation; retain exact baseline-passing ID comparison. |
| 13 | evidence.py @@ -387,95 +325,54 @@ | Completed-parent integrity, raw/normalized exit, ordered timestamps, dependencies and device evidence checked by scorer. |
| 14 | run_gpu_line.py @@ -1,7 +1,12 @@ | Parent docstring only. |
| 15 | run_gpu_line.py @@ -11,8 +16,8 @@ | Parent imports adapt evidence functions. |
| 16 | run_gpu_line.py @@ -29,21 +34,18 @@ | Parent environment, selected arm/suite and dependency changes. |
| 17 | run_gpu_line.py @@ -54,105 +56,81 @@ | Parent foreground subprocess.run returns before receipt construction/publication; actual exit and timeout retained. |
| 18 | run_gpu_tests.py @@ -1,7 +1,11 @@ | Child docstring only. |
| 19 | run_gpu_tests.py @@ -10,27 +14,22 @@ | Child imports/report exception fields adapted. |
| 20 | run_gpu_tests.py @@ -41,108 +40,74 @@ | Pinned arm preload and suite selection. Child returns pytest exit after writing JUnit/summary; D4 subprocess inheritance is R1-F02. |
| 21 | score_gpu.py @@ -1,45 +1,48 @@ | Scorer changes roles, control and census reporting; requires valid parent+child+JUnit and positive device evidence. |
| 22 | start_gpu_run.py @@ -1,4 +1,4 @@ | START docstring only. |
| 23 | start_gpu_run.py @@ -11,12 +11,12 @@ | START immutable record changes GN1 flags to ENG1 audit mode; retains seal/date bindings. |
| 24 | forge_gpu_run_standin.py @@ -1,13 +1,13 @@ | Stand-in environment naming; explicitly CPU audit only. |
| 25 | device_evidence.py @@ -1,4 +1,5 @@ | Device evidence docstring. |
| 26 | device_evidence.py @@ -98,7 +99,7 @@ | Device query pin is selected per arm. |
| 27 | gpu_lines.json @@ -1,811 +1,338 @@ | 22-line registration replaces GN1 labels and bounds; expected child exits remain separate from parent success. |

run_gpu_line.py:65–70 waits for the foreground subprocess, derives rc from returncode, and only then evaluates receipts and publishes parent_result.json at :108–130. evidence.py:229–249 requires the observed exit, matching child pytest_exit, exact collected IDs and JUnit outcomes; :278–280 cross-checks rc against recorded failures. recompute_line :320–342 refuses missing parent completion and validates start <= child_exited <= recorded. The parent never treats a mutant kill or a baseline expected failure as a child success; expected exit 1 is explicit. Device failures, missing tests, collection failures and unregistered skips are refused outside the regression role.

The existing audit scorer was executed unchanged against its original audit root. Only its attempted G5_SUMMARY.json write was redirected to rev1/audit_rescore.json to respect read-only receipts. It returned 1: candidate_eng1 passed in CPU mode, all other lines were absent, and audit data could not establish GPU acceptance.
