# CPU command receipts

All seat shell commands explicitly set CUDA_VISIBLE_DEVICES empty and used timeout. Exact tool invocations, patches and outputs are retained in the authorized live transcript logs/apa_p1_a1_r1.log. Source snapshots and gate JSONs identify inputs/outputs; no git was invoked.

- Build (one attempt): `CUDA_VISIBLE_DEVICES= timeout 1790s env PYTHONDONTWRITEBYTECODE=1 python3 artifacts/apa_p1_a1/BUILD.py`.
- SASS dump (two read-only binaries, each timeout 850s inside outer 1790s): `python3 artifacts/apa_p1_a1/SASS.py`; raw-name result retained. Final exact-body comparison: `CUDA_VISIBLE_DEVICES= timeout 120s env PYTHONDONTWRITEBYTECODE=1 python3 artifacts/apa_p1_a1/SASS_COMPARE.py`.
- Scorer: `CUDA_VISIBLE_DEVICES= timeout 120s env PYTHONDONTWRITEBYTECODE=1 python3 artifacts/apa_p1_a1/TEST_SCORER.py` (log SCORER_TEST.log).
- G3 parity: `CUDA_VISIBLE_DEVICES= timeout 590s env PYTHONDONTWRITEBYTECODE=1 python3 artifacts/apa_p1_a1/G3.py` (per-arm pytest timeout 540s, cache disabled, basetemp under A1 and TMPDIR confined).
- Recurrence: layout.source_gate(INPUT_CENSUS.json); 3718 evaluations from A3 independent size/view models and port source expressions (RECURRENCE.json).
- Constructed readback: `CUDA_VISIBLE_DEVICES= timeout 590s env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 artifacts/apa_p1_a1/L48.py` (L48_READBACK.log).
- Freeze: `CUDA_VISIBLE_DEVICES= timeout 30s env PYTHONDONTWRITEBYTECODE=1 python3 artifacts/apa_p1_a1/STAGE.py`. Registration predates this invocation; no staged file existed before registration.
- Exact staged CPU rehearsal: `CUDA_VISIBLE_DEVICES= timeout 590s env PYTHONDONTWRITEBYTECODE=1 APA_P1_A1_CPU_REHEARSAL=1 APA_P1_A1_RUN_ROOT=artifacts/apa_p1_a1/cpu_staged_fixed PATH="$PWD/artifacts/apa_p1_a1/cpu_bin:$PATH" bash artifacts/apa_p1_a1/STAGED_GPU_LINES.sh` (CPU_STAGED_FIXED.log). Initial run in cpu_staged/ retained; STAGING_AMENDMENT_1.json explains CPU-plan-only correction.
- Final report-only re-score, cwd artifacts/apa_p1: `CUDA_VISIBLE_DEVICES= timeout 60s env PYTHONDONTWRITEBYTECODE=1 TMPDIR=/tmp/apa_p1_a1_rescore python3 ../apa_p1_a1/RESCORE.py --out ../apa_p1_a1/rescore_gpu1/final`.
- Final verification: `CUDA_VISIBLE_DEVICES= timeout 120s env PYTHONDONTWRITEBYTECODE=1 python3 artifacts/apa_p1_a1/VERIFY.py`.

GPU lines are staged only. The CPU seat never invoked the real forge-gpu-run. Every running foreground exec was immediately waited with yield_time_ms=300000; no background jobs or short wait polls.

Prior art: P1/DS1 recorded orchestration (Project-Tensor, 2026), standard bounded subprocess commands.
