# A3 to A4 tooling hunks

The raw diff is changes_vs_a3.patch. Mechanical D1 prefix/order relocation is identified here separately from functional D2-D7 changes. Every hunk maps to the binding fixes or their required controls, staging and scope verification. Product/test changes: zero. No tolerance changes. Generated result trees are in the SHA maps.

| File | Raw hunk | Design |
|---|---|---|
| AMENDMENTS.md | `@@ -3,3 +3,6 @@` | D5 D6 |
| EQUIVALENCES.md | `@@ -13,3 +13,15 @@` | D6 |
| MUTANTS.md | `@@ -1,60 +1,62 @@` | D6 |
| REGISTRATION.md | `@@ -1 +1,3 @@` | D2 D3 D4 D5 D6 D7 |
| SEAL_IMPORT_TRACE.md | `@@ -1,29 +1,9 @@` | D5 |
| STAGED_GPU_COMMANDS.sh | `@@ -6,7 +6,7 @@` | D6 D7 |
| STAGED_GPU_COMMANDS.sh | `@@ -23,173 +23,178 @@` | D6 D7 |
| a4_controls.py | `@@ -0,0 +1,286 @@` | D2 D3 D4 D5 D6 D7 |
| arm_maps.json | `@@ -252,6 +252,28 @@` | D6 |
| build_mutants.py | `@@ -76,6 +76,11 @@` | D6 |
| cpu_checks.py | `@@ -79,6 +79,11 @@` | D6 |
| cpu_checks.py | `@@ -86,7 +91,7 @@` | D6 |
| equivalence_execution.py | `@@ -0,0 +1,34 @@` | D6 |
| evidence.py | `@@ -11,10 +11,11 @@` | D2 D3 D4 D5 D6 D7 |
| evidence.py | `@@ -34,7 +35,7 @@` | D2 D3 D4 D5 D6 D7 |
| evidence.py | `@@ -156,7 +157,7 @@` | D2 D3 D4 D5 D6 D7 |
| evidence.py | `@@ -187,6 +188,10 @@` | D2 D3 D4 D5 D6 D7 |
| evidence.py | `@@ -197,8 +202,13 @@` | D2 D3 D4 D5 D6 D7 |
| evidence.py | `@@ -221,19 +231,23 @@` | D2 D3 D4 D5 D6 D7 |
| evidence.py | `@@ -259,6 +273,10 @@` | D2 D3 D4 D5 D6 D7 |
| evidence.py | `@@ -278,14 +296,41 @@` | D2 D3 D4 D5 D6 D7 |
| evidence.py | `@@ -325,14 +370,13 @@` | D2 D3 D4 D5 D6 D7 |
| evidence.py | `@@ -376,5 +420,10 @@` | D2 D3 D4 D5 D6 D7 |
| extra_controls.py | `@@ -0,0 +1,58 @@` | D2 D3 D5 |
| gpu_lines.json | `@@ -9,7 +9,8 @@` | D6 D7 |
| gpu_lines.json | `@@ -21,7 +22,8 @@` | D6 D7 |
| gpu_lines.json | `@@ -33,7 +35,8 @@` | D6 D7 |
| gpu_lines.json | `@@ -46,7 +49,8 @@` | D6 D7 |
| gpu_lines.json | `@@ -59,7 +63,8 @@` | D6 D7 |
| gpu_lines.json | `@@ -72,7 +77,8 @@` | D6 D7 |
| gpu_lines.json | `@@ -85,7 +91,8 @@` | D6 D7 |
| gpu_lines.json | `@@ -98,7 +105,8 @@` | D6 D7 |
| gpu_lines.json | `@@ -111,7 +119,8 @@` | D6 D7 |
| gpu_lines.json | `@@ -124,7 +133,8 @@` | D6 D7 |
| gpu_lines.json | `@@ -137,7 +147,8 @@` | D6 D7 |
| gpu_lines.json | `@@ -150,7 +161,8 @@` | D6 D7 |
| gpu_lines.json | `@@ -163,7 +175,8 @@` | D6 D7 |
| gpu_lines.json | `@@ -176,7 +189,8 @@` | D6 D7 |
| gpu_lines.json | `@@ -189,7 +203,8 @@` | D6 D7 |
| gpu_lines.json | `@@ -202,7 +217,8 @@` | D6 D7 |
| gpu_lines.json | `@@ -215,7 +231,8 @@` | D6 D7 |
| gpu_lines.json | `@@ -228,7 +245,8 @@` | D6 D7 |
| gpu_lines.json | `@@ -241,7 +259,8 @@` | D6 D7 |
| gpu_lines.json | `@@ -254,7 +273,8 @@` | D6 D7 |
| gpu_lines.json | `@@ -267,7 +287,8 @@` | D6 D7 |
| gpu_lines.json | `@@ -280,7 +301,8 @@` | D6 D7 |
| gpu_lines.json | `@@ -293,7 +315,8 @@` | D6 D7 |
| gpu_lines.json | `@@ -306,7 +329,8 @@` | D6 D7 |
| gpu_lines.json | `@@ -319,7 +343,8 @@` | D6 D7 |
| gpu_lines.json | `@@ -332,7 +357,8 @@` | D6 D7 |
| gpu_lines.json | `@@ -345,7 +371,8 @@` | D6 D7 |
| gpu_lines.json | `@@ -358,7 +385,8 @@` | D6 D7 |
| gpu_lines.json | `@@ -371,7 +399,8 @@` | D6 D7 |
| gpu_lines.json | `@@ -384,7 +413,8 @@` | D6 D7 |
| gpu_lines.json | `@@ -397,7 +427,8 @@` | D6 D7 |
| gpu_lines.json | `@@ -410,7 +441,8 @@` | D6 D7 |
| gpu_lines.json | `@@ -423,7 +455,8 @@` | D6 D7 |
| gpu_lines.json | `@@ -436,7 +469,8 @@` | D6 D7 |
| gpu_lines.json | `@@ -449,7 +483,8 @@` | D6 D7 |
| gpu_lines.json | `@@ -462,7 +497,8 @@` | D6 D7 |
| gpu_lines.json | `@@ -475,7 +511,8 @@` | D6 D7 |
| gpu_lines.json | `@@ -488,7 +525,8 @@` | D6 D7 |
| gpu_lines.json | `@@ -501,7 +539,8 @@` | D6 D7 |
| gpu_lines.json | `@@ -514,7 +553,8 @@` | D6 D7 |
| gpu_lines.json | `@@ -527,7 +567,8 @@` | D6 D7 |
| gpu_lines.json | `@@ -540,7 +581,8 @@` | D6 D7 |
| gpu_lines.json | `@@ -553,7 +595,8 @@` | D6 D7 |
| gpu_lines.json | `@@ -566,7 +609,8 @@` | D6 D7 |
| gpu_lines.json | `@@ -579,7 +623,8 @@` | D6 D7 |
| gpu_lines.json | `@@ -592,7 +637,8 @@` | D6 D7 |
| gpu_lines.json | `@@ -605,7 +651,8 @@` | D6 D7 |
| gpu_lines.json | `@@ -618,7 +665,8 @@` | D6 D7 |
| gpu_lines.json | `@@ -631,7 +679,8 @@` | D6 D7 |
| gpu_lines.json | `@@ -644,7 +693,8 @@` | D6 D7 |
| gpu_lines.json | `@@ -657,7 +707,8 @@` | D6 D7 |
| gpu_lines.json | `@@ -670,7 +721,8 @@` | D6 D7 |
| gpu_lines.json | `@@ -683,7 +735,8 @@` | D6 D7 |
| gpu_lines.json | `@@ -696,7 +749,8 @@` | D6 D7 |
| gpu_lines.json | `@@ -709,7 +763,8 @@` | D6 D7 |
| gpu_lines.json | `@@ -722,6 +777,35 @@` | D6 D7 |
| hostile_forge.py | `@@ -29,6 +29,8 @@` | D2 D5 D6 |
| hostile_forge.py | `@@ -133,6 +135,8 @@` | D2 D5 D6 |
| hostile_scorer.py | `@@ -21,6 +21,7 @@` | D2 D7 |
| hostile_scorer.py | `@@ -32,10 +33,11 @@` | D2 D7 |
| hostile_scorer.py | `@@ -70,7 +72,7 @@` | D2 D7 |
| hostile_scorer.py | `@@ -175,12 +177,12 @@` | D2 D7 |
| mutants.json | `@@ -269,6 +269,32 @@` | D6 |
| order_commit.txt | `@@ -1 +1 @@` | D2 D3 D4 D5 D6 D7 |
| replay_driver.py | `@@ -0,0 +1,16 @@` | D2 D3 D4 D5 D6 D7 |
| rev2_controls.py | `@@ -31,7 +31,7 @@` | D2 D3 D7 |
| rev2_controls.py | `@@ -55,6 +55,7 @@` | D2 D3 D7 |
| rev2_controls.py | `@@ -66,10 +67,22 @@` | D2 D3 D7 |
| rev2_controls.py | `@@ -341,7 +354,7 @@` | D2 D3 D7 |
| rev2_forge.py | `@@ -28,6 +28,12 @@` | D2 D3 D4 D5 D6 |
| rev2_forge.py | `@@ -96,7 +102,28 @@` | D2 D3 D4 D5 D6 |
| rev2_forge.py | `@@ -114,10 +141,12 @@` | D2 D3 D4 D5 D6 |
| run_cpu_gate.py | `@@ -20,6 +20,9 @@` | D2 D3 D4 D5 D6 D7 |
| run_gpu_line.py | `@@ -40,7 +40,9 @@` | D2 D3 D4 D5 |
| run_gpu_line.py | `@@ -57,7 +59,7 @@` | D2 D3 D4 D5 |
| run_gpu_tests.py | `@@ -9,6 +9,14 @@` | D2 D4 D5 D6 |
| run_gpu_tests.py | `@@ -110,10 +118,12 @@` | D2 D4 D5 D6 |
| runner_controls.py | `@@ -11,7 +11,7 @@` | D2 D3 D7 |
| runner_controls.py | `@@ -30,6 +30,7 @@` | D2 D3 D7 |
| runner_controls.py | `@@ -37,13 +38,18 @@` | D2 D3 D7 |
| runner_controls.py | `@@ -97,7 +103,7 @@` | D2 D3 D7 |
| runner_controls.py | `@@ -118,7 +124,7 @@` | D2 D3 D7 |
| runner_controls.py | `@@ -126,7 +132,7 @@` | D2 D3 D7 |
| runner_controls.py | `@@ -148,7 +154,7 @@` | D2 D3 D7 |
| runtime_driver.py | `@@ -0,0 +1,7 @@` | D5 |
| runtime_evidence.py | `@@ -0,0 +1,78 @@` | D2 D4 D5 D6 |
| runtime_sentinel.py | `@@ -0,0 +1,2 @@` | D5 |
| seal_staging.py | `@@ -4,6 +4,7 @@` | D5 D6 |
| seal_staging.py | `@@ -16,16 +17,31 @@` | D5 D6 |
| trace_imports.py | `@@ -31,7 +31,7 @@` | D5 |
| trace_runtime.py | `@@ -0,0 +1,48 @@` | D5 |
| verify_scope.py | `@@ -1,39 +1,23 @@` | D2 D3 D4 D5 D6 D7 |
| verify_scope.py | `@@ -41,7 +25,6 @@` | D2 D3 D4 D5 D6 D7 |
