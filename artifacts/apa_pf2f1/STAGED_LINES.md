# STAGED_LINES.md

LEAD ONLY. Run from the worktree root below. This seat ran dry runs only.
There are 58 lines: G1_000..G1_045 cover all 4,848 finite cases once;
G2 has 16 sink and 16 norm cases; G4 covers 28 reachable combinations;
G5 tests both GPU mutants after clean F1 controls; G6 has 26 refusal and
24 memory rows. Exact indices, arguments and module pins: LINES.json.

Every launcher checks module, source, script, registration and shim hashes
before the lease. It sets PATH, LANG/LC_ALL, CUDA 12.6 LD_LIBRARY_PATH,
PYTHONPATH empty, TMPDIR=/tmp/pf2f1_gpu, PYTHONDONTWRITEBYTECODE=1,
TC_APA_SP=1, APA_PF2F1_LEASE=1 and all six BLAS/OpenMP thread variables to 1.
Only HOME, USER, LOGNAME, XDG_RUNTIME_DIR and DBUS_SESSION_BUS_ADDRESS are
inherited when present. CPU prechecks/dry runs have CUDA_VISIBLE_DEVICES
empty; real payloads explicitly use CUDA_VISIBLE_DEVICES=0. Torch intra-op
and inter-op thread counts are 1 where torch is used. Refusal/memory lines
also set LD_PRELOAD=/tmp/pf2f1_shim/libpf2count.so, checked against its pin.
Full per-line environments: gpu/*_dry.start.json and the launcher source.

Every GPU invocation uses forge-gpu-run --min <n> --max 540, inner timeout
500 seconds, nice -n 19 and python3 -B -P. G1 workers have a 460-second limit.
The foreground 600-second pauses below preserve the cooldown between lines.
Each launcher records the forge exit and log SHA in gpu/*_real.exit.json.

```bash
set -eu
cd /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_p1pf2
export CUDA_VISIBLE_DEVICES= TMPDIR=/tmp/pf2f1_lead PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
timeout 10s mkdir -p "$TMPDIR"
run() {
  timeout 660s env FORGE_GPU_DRY_RUN=0 python3 -B -P scripts/apa_pf2f1_line.py "$1" --lead --min "$2" --max 540
  timeout 610s sleep 600
}
for n in 000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045; do
  run "G1_$n" 500
done
run G2_sink 30
run G2_norm 30
run G4_coverage 120
run G5_sink_mutant 60
run G5_norm_mutant 60
run G6_refusals 60
run G6_memory_qwen3_1.7b_2048_2048 500
run G6_memory_qwen3_1.7b_8192_8192 500
run G6_memory_qwen3_1.7b_32768_2048 500
run G6_memory_qwen3_4b_2048_2048 500
run G6_memory_qwen3_4b_8192_8192 500
run G6_memory_qwen3_4b_32768_2048 500
timeout 120s python3 -B -P scripts/apa_pf2f1_lead_summary.py
```

Exit 75/76 is no experiment; 124 is timeout; none passes a gate. Stop on any
nonzero exit. Outputs use exclusive creation: retain prior attempt files by
moving them within artifacts/apa_pf2f1 before an authorized rerun; never delete
or overwrite a measurement. A measured timeout may require another registered
split; this seat measured no GPU timeout. No gate is narrowed.

All 58 final dry runs returned 78 before the lease (DRY_RUNS.json). The CPU
index mutant is already killed and SASS is checked; neither needs a GPU line.
GPU coverage remains 0/28 until G4_coverage runs; its result is COVERAGE_LEAD.json.
The CPU summary writes ACCEPTANCE_LEAD.json only after all receipts pass.
A separate Claude review still follows.
