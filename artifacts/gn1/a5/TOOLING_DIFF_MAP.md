# Tooling hunk map

Raw diff: tooling_a4_to_a5.diff. Relocation-normalised diff: tooling_functional.diff. Every raw hunk is listed below. New files are CPU validation support and REV4 fixture preservation.

| File | Hunk | Design | Classification |
|---|---|---|---|
| STAGED_GPU_COMMANDS.sh | @@ -6,9 +6,13 @@ | D3 | implementation / validation |
| STAGED_GPU_COMMANDS.sh | @@ -23,178 +27,178 @@ | D2, D3 | implementation / validation |
| a4_controls.py | @@ -1,3 +1,4 @@ | D2 | implementation / validation |
| a4_controls.py | @@ -37,6 +38,8 @@ | D2 | implementation / validation |
| a4_controls.py | @@ -260,9 +263,9 @@ | D2, D4 | implementation / validation |
| evidence.py | @@ -15,7 +15,7 @@ | D2, D3, D4, D5 | implementation / validation |
| evidence.py | @@ -35,7 +35,7 @@ | D2, D3, D4, D5 | implementation / validation |
| evidence.py | @@ -157,7 +157,7 @@ | D2, D3, D4, D5 | implementation / validation |
| evidence.py | @@ -183,14 +183,17 @@ | D4 | implementation / validation |
| evidence.py | @@ -198,7 +201,7 @@ | D4 | implementation / validation |
| evidence.py | @@ -304,6 +307,8 @@ | D4 | implementation / validation |
| evidence.py | @@ -319,14 +324,13 @@ | D2 | implementation / validation |
| hostile_scorer.py | @@ -1,3 +1,4 @@ | D2 | implementation / validation |
| hostile_scorer.py | @@ -21,6 +22,8 @@ | D2 | implementation / validation |
| hostile_scorer.py | @@ -72,7 +75,7 @@ | D2 | implementation / validation |
| hostile_scorer.py | @@ -180,9 +183,9 @@ | D2 | implementation / validation |
| rev2_controls.py | @@ -1,3 +1,4 @@ | D2 | implementation / validation |
| rev2_controls.py | @@ -55,6 +56,8 @@ | D2 | implementation / validation |
| run_gpu_line.py | @@ -10,6 +10,7 @@ | D5 | implementation / validation |
| run_gpu_line.py | @@ -39,7 +40,12 @@ | D4, D5 | implementation / validation |
| run_gpu_line.py | @@ -53,45 +59,72 @@ | D3, D4, D5 | implementation / validation |
| runner_controls.py | @@ -1,3 +1,4 @@ | D2 | implementation / validation |
| runner_controls.py | @@ -30,6 +31,8 @@ | D2 | implementation / validation |
| runner_controls.py | @@ -84,7 +87,8 @@ | D2 | implementation / validation |
| seal_staging.py | @@ -17,7 +17,7 @@ | D2, D3, D4, D5 | mechanical relocation |
| seal_staging.py | @@ -28,7 +28,7 @@ | D2, D3, D4, D5 | mechanical relocation |
| seal_staging.py | @@ -36,7 +36,7 @@ | D2, D3, D4, D5 | mechanical relocation |
| verify_scope.py | @@ -16,7 +16,7 @@ | D2, D3, D4, D5 | mechanical relocation |
| a5_controls.py | @@ -0,0 +1,267 @@ | D2, D3, D4, D5 | implementation / validation |
| control_transport.py | @@ -0,0 +1,103 @@ | D2 | implementation / validation |
| cpu_driver_a5.py | @@ -0,0 +1,18 @@ | D2, D3, D4, D5 | implementation / validation |
| rev4_forge.py | @@ -0,0 +1,121 @@ | D2, D3, D4, D5 | implementation / validation |
