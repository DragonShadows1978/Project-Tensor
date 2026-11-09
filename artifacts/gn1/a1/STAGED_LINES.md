# Staged lines (verbatim)

Run from artifacts/gn1/a1 with RUNS an explicit absolute fresh subdirectory of a1, reused across labels; TMPDIR=/tmp/gn1_a1_staged. No RUNS default. The shell writes/checks START.json before each line and changes into its tooling directory. GPU_RUN=forge-gpu-run for real G5; no device command ran this round. Every parent exit is expected 0, with child expected 0 on candidate focus/regression, child expected 1 on baseline focus/regression and all mutants. The second numeric argument below is the expected child exit. Each parent writes its receipt after child exit; collection, device, setup failures and timeouts cannot count as mutant kills.

```bash
timeout 350s python3 run_gpu_line.py baseline_focus 1 330 "$GPU_RUN" --min 250 --max 300 --label gn1-baseline_focus -- timeout --kill-after=10s 240s python3 run_gpu_tests.py baseline focus
timeout 350s python3 run_gpu_line.py candidate_focus 0 330 "$GPU_RUN" --min 250 --max 300 --label gn1-candidate_focus -- timeout --kill-after=10s 240s python3 run_gpu_tests.py candidate focus
timeout 950s python3 run_gpu_line.py baseline_regression 1 930 "$GPU_RUN" --min 850 --max 900 --label gn1-baseline_regression -- timeout --kill-after=10s 840s python3 run_gpu_tests.py baseline regression
timeout 950s python3 run_gpu_line.py candidate_regression 0 930 "$GPU_RUN" --min 850 --max 900 --label gn1-candidate_regression -- timeout --kill-after=10s 840s python3 run_gpu_tests.py candidate regression
timeout 290s python3 run_gpu_line.py M01 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-M01 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M01 mutation
timeout 290s python3 run_gpu_line.py M02 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-M02 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M02 mutation
timeout 290s python3 run_gpu_line.py M03 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-M03 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M03 mutation
timeout 290s python3 run_gpu_line.py M04 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-M04 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M04 mutation
timeout 290s python3 run_gpu_line.py M05 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-M05 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M05 mutation
timeout 290s python3 run_gpu_line.py M06 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-M06 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M06 mutation
timeout 290s python3 run_gpu_line.py M07 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-M07 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M07 mutation
timeout 290s python3 run_gpu_line.py M08 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-M08 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M08 mutation
timeout 290s python3 run_gpu_line.py M09 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-M09 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M09 mutation
timeout 290s python3 run_gpu_line.py M10 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-M10 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M10 mutation
timeout 290s python3 run_gpu_line.py M11 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-M11 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M11 mutation
timeout 290s python3 run_gpu_line.py M12 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-M12 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M12 mutation
timeout 290s python3 run_gpu_line.py M13 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-M13 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M13 mutation
timeout 290s python3 run_gpu_line.py M14 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-M14 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M14 mutation
timeout 290s python3 run_gpu_line.py M15 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-M15 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M15 mutation
timeout 290s python3 run_gpu_line.py M16 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-M16 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M16 mutation
timeout 290s python3 run_gpu_line.py M17 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-M17 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M17 mutation
timeout 290s python3 run_gpu_line.py M18 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-M18 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M18 mutation
```

CPU audit line actually run (exit 0):

```bash
TMPDIR=/tmp/gn1_a1_boot CUDA_VISIBLE_DEVICES= RUNS=/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_gn1/artifacts/gn1/a1/audit_cpu timeout 120s bash /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_gn1/artifacts/gn1/a1/STAGED_GPU_COMMANDS.sh --audit
```

The child ran 29 reference IDs, all passed; parent recorded child exit=0, timeout=false, audit=true, passed=true, after child exit. score_gpu.py is staged only; running it after all 22 actual device lines is expected exit 0.
