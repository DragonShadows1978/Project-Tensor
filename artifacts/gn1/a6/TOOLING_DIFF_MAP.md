# A5 to A6 tooling hunk map

Raw diff: tooling_a5_to_a6.diff; relocation-normalised: tooling_functional.diff. No numerical or mutation behavior changed.

| File | Hunk | Design | Classification |
|---|---|---|---|
| STAGED_GPU_COMMANDS.sh | @@ -1,204 +1,211 @@ | D1 | D1-D4 implementation or mandatory CPU control scaffolding |
| a4_controls.py | @@ -1,3 +1,5 @@ | D1 | D1-D4 implementation or mandatory CPU control scaffolding |
| a5_controls.py | @@ -1,3 +1,5 @@ | D1 | D1-D4 implementation or mandatory CPU control scaffolding |
| a5_controls.py | @@ -111,7 +113,7 @@ | D1 | D1-D4 implementation or mandatory CPU control scaffolding |
| a6_controls.py | @@ -0,0 +1,250 @@ | D1, D2, D3, D4 | D1-D4 implementation or mandatory CPU control scaffolding |
| evidence.py | @@ -12,10 +12,11 @@ | D1 | D1-D4 implementation or mandatory CPU control scaffolding |
| evidence.py | @@ -35,7 +36,7 @@ | D1 | D1-D4 implementation or mandatory CPU control scaffolding |
| evidence.py | @@ -84,6 +85,7 @@ | D1 | D1-D4 implementation or mandatory CPU control scaffolding |
| evidence.py | @@ -157,7 +159,7 @@ | D1 | D1-D4 implementation or mandatory CPU control scaffolding |
| evidence.py | @@ -327,10 +329,13 @@ | D1 | D1-D4 implementation or mandatory CPU control scaffolding |
| evidence.py | @@ -388,6 +393,7 @@ | D4 | D1-D4 implementation or mandatory CPU control scaffolding |
| evidence.py | @@ -395,6 +401,9 @@ | D3 | D1-D4 implementation or mandatory CPU control scaffolding |
| evidence.py | @@ -426,6 +435,7 @@ | D1 | D1-D4 implementation or mandatory CPU control scaffolding |
| extra_controls.py | @@ -1,3 +1,5 @@ | D1 | D1-D4 implementation or mandatory CPU control scaffolding |
| extra_controls.py | @@ -52,7 +54,7 @@ | D1 | D1-D4 implementation or mandatory CPU control scaffolding |
| forge_gpu_run_standin.py | @@ -11,7 +11,7 @@ | D1 | D1-D4 implementation or mandatory CPU control scaffolding |
| hostile_scorer.py | @@ -1,3 +1,5 @@ | D1 | D1-D4 implementation or mandatory CPU control scaffolding |
| hostile_scorer.py | @@ -75,7 +77,7 @@ | D1 | D1-D4 implementation or mandatory CPU control scaffolding |
| hostile_scorer.py | @@ -142,7 +144,7 @@ | D1 | D1-D4 implementation or mandatory CPU control scaffolding |
| hostile_scorer.py | @@ -183,9 +185,9 @@ | D1 | D1-D4 implementation or mandatory CPU control scaffolding |
| p116_probe.py | @@ -0,0 +1,36 @@ | D1 | D1-D4 implementation or mandatory CPU control scaffolding |
| program_binding.py | @@ -0,0 +1,89 @@ | D1 | D1-D4 implementation or mandatory CPU control scaffolding |
| rev2_controls.py | @@ -1,3 +1,5 @@ | D1 | D1-D4 implementation or mandatory CPU control scaffolding |
| rev2_controls.py | @@ -308,7 +310,7 @@ | D1 | D1-D4 implementation or mandatory CPU control scaffolding |
| rev4_forge.py | @@ -89,7 +89,7 @@ | D1 | mechanical A6 relocation |
| run_gpu_line.py | @@ -11,6 +11,7 @@ | D1 | D1-D4 implementation or mandatory CPU control scaffolding |
| run_gpu_line.py | @@ -30,6 +31,8 @@ | D1 | D1-D4 implementation or mandatory CPU control scaffolding |
| run_gpu_line.py | @@ -57,9 +60,18 @@ | D1, D2 | D1-D4 implementation or mandatory CPU control scaffolding |
| run_gpu_line.py | @@ -69,7 +81,10 @@ | D4 | D1-D4 implementation or mandatory CPU control scaffolding |
| run_gpu_line.py | @@ -79,7 +94,7 @@ | D2 | D1-D4 implementation or mandatory CPU control scaffolding |
| run_gpu_line.py | @@ -88,7 +103,7 @@ | D2 | D1-D4 implementation or mandatory CPU control scaffolding |
| run_gpu_line.py | @@ -99,21 +114,37 @@ | D1, D2, D4 | D1-D4 implementation or mandatory CPU control scaffolding |
| run_gpu_tests.py | @@ -71,6 +71,16 @@ | D1 | D1-D4 implementation or mandatory CPU control scaffolding |
| runner_controls.py | @@ -1,3 +1,5 @@ | D1 | D1-D4 implementation or mandatory CPU control scaffolding |
| score_gpu.py | @@ -16,15 +16,18 @@ | D1, D2 | D1-D4 implementation or mandatory CPU control scaffolding |
| seal_staging.py | @@ -5,6 +5,7 @@ | D1 | D1-D4 implementation or mandatory CPU control scaffolding |
| seal_staging.py | @@ -17,7 +18,7 @@ | D1 | D1-D4 implementation or mandatory CPU control scaffolding |
| seal_staging.py | @@ -28,15 +29,15 @@ | D1 | D1-D4 implementation or mandatory CPU control scaffolding |
| start_gpu_run.py | @@ -11,11 +11,11 @@ | D1 | D1-D4 implementation or mandatory CPU control scaffolding |
| verify_scope.py | @@ -16,7 +16,7 @@ | D1 | mechanical A6 relocation |
