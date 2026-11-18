# STAGED_LINES.md

69 GPU lines staged. All 69 dry runs returned 78 with
`would_start: true` and zero lease attempts. G2_sink runs first.
Run from the worktree root as the GPU lead.

```bash
set -eu
cd /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_pf2f3
export CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
line_names=$(timeout 15s python3 -B -P -c 'import json; print(" ".join(json.load(open("artifacts/apa_pf2f3/LINES.json"))))')
for line in $line_names; do
  FORGE_GPU_DRY_RUN=0 timeout 660s python3 -B -P scripts/apa_pf2f3_line.py "$line" --lead
done
```

| Lines in order | Count | Min seconds | Coverage |
|---|---:|---:|---|
| G2_sink, G2_norm, G2_partition, G2_tc | 4 | 30, 60, 90, 60 | 16 sink, 16 norm, 44 partition, eight `tc` configurations |
| G1_public_00–03, G1_raw_00–03, G1_extra_00–01 | 10 | 120 | 752 inherited finite decode cases |
| G1_prefill_000–045 | 46 | 500 | 4,848 inherited finite prefill cases |
| G1_tc_00–03 | 4 | 120 | 224 finite BF16 `tc` cases across the reachable instantiations |
| G3_pf3m01, G3_pf3m02, G3_pf3m03 | 3 | 120 | Clean controls and targeted device failures for each mutant |
| G4_pytest | 1 | 120 | All 77 shipped tests with the F3 package on PYTHONPATH |
| G5_regression | 1 | 120 | 643 inherited P1 packed decode regression predicates |

Each launcher verifies the module, Python package, and registered input
SHA-256s before invoking `forge-gpu-run --min <n> --max 540 --margin 30`.
The launcher sets the CUDA runtime path, thread environment, scratch,
and device visibility. Full dry argv and environment are in
`gpu/*_dry.start.json`. Admission, exit 78 and log hashes are in
`gpu/*_dry.exit.json` and `DRY_RUNS.json`.

The launchers record the lead's exits and log SHA-256s in
`gpu/*_real.exit.json`. Test verdicts go in `results/<line>/`. G3
requires exit 1 from the inner mutant worker and exit 0 from the clean
control. The enclosing line exits successfully only when the complete
registered failure and control predicates match. The loop stops at the
first failed line.
