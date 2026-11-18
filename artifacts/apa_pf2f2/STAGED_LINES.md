# STAGED_LINES.md

**62 lines staged.** All 62 final dry runs returned 78 with
`would_start: true` and zero lease attempts. G2_sink is the first smoke.
Run from the worktree root as the GPU lead.

```bash
set -eu
cd /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_pf2f2
export CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
line_names=$(timeout 15s python3 -B -P -c 'import json; print(" ".join(json.load(open("artifacts/apa_pf2f2/LINES.json"))))')
for line in $line_names; do
  FORGE_GPU_DRY_RUN=0 timeout 660s python3 -B -P scripts/apa_pf2f2_line.py "$line" --lead
done
```

| Lines in order | Min seconds | Coverage |
|---|---:|---|
| G2_sink, G2_norm | 30 | The 16 sink and 16 raw-norm REV1 cases |
| G2_partition | 60 | 44 partition-carry cases, both dtypes and diagnostics settings, both shared G values and all bulk CAPs |
| G1_public_00–03, G1_raw_00–03, G1_extra_00–01 | 120 | 752 finite decode cases, including 512 public and 112 inherited native r2 cases |
| G1_prefill_000–045 | 500 | All 4,848 PF2 finite prefill cases with inherited case indices and splits |
| G3_bulk, G3_bulk_shared | 90 | F2 control, eight target failures on each revert, eight passing alternate-route controls per revert |
| G4_regression | 120 | P1-F2 LF2 clauses 1 and 2, plus P1-F3 overflow cases, on the packed decode paths |

Each launcher verifies script, registration, fixture and module SHA-256s
before admission. It sets the CUDA 12.6 runtime path, explicit device and
lease flags, TMPDIR, and six single-thread environment variables. G1 sets
Torch intra-op and inter-op threads to one in both arms. The launcher uses
`forge-gpu-run --min <n> --max 540 --margin 30` with a 500-second payload
timeout. Full argv and environments are in `gpu/*_dry.start.json`.

Lead exits and log SHA-256s go in `gpu/*_real.exit.json`; per-case verdicts
go in `results/<line>/`. The loop stops on any failed line. G3 expects exit
1 from the inner revert test and exit 0 from the enclosing launcher only
after the clean, target and alternate-route checks all pass.
