# ENG5-S ledger (seat: Claude Opus 5.5; order orders/ENG5S_STATUS_CHECKS.md at 5660f4755)

All times UTC, 2026-11-24. Every command ran with CUDA_VISIBLE_DEVICES empty and TMPDIR=/tmp/eng5s_tmp.

| Time | Step | Receipt |
|---|---|---|
| 06:12 | L01 preflight: cmake 3.28.3, nvcc 12.6.85, gcc 13.3.0, pytest 9.0.2, Python 3.12.3 + NumPy 2.4.2, forge-gpu-run --status (exit 0), forge-gpu-lease-check present | preflight_L01.txt |
| 06:12 | S3: L82 census grep at fbd335ae7; all nine sites match the order's table (kernels.cu:108, :185, :773, :885-897, :898-908, :909-916, :918-924, :5395-5401; packed_prefill.cuh:244) | this ledger |
| 06:14 | Build B from `git archive fbd335ae7 tensor_cuda` (/tmp/eng5s_src_B) | build/BUILD_B.json |
| 06:15-06:18 | Nine status checks + helper (kernels.cu, packed_prefill.cuh, core.h); Build S | build/BUILD_S.json, source.diff |
| 06:21:31 | MUTANTS.json registered (s01-s04), before any mutant build or run | MUTANTS.json |
| 06:21-06:38 | Builds B_H, S_H (hooks arms for site 9), s01-s04; packages | build/*.json |
| 06:38-06:44 | CPU suite r1 (cpu_B, cpu_S): SLIP, basetemp parent missing (360 tmp_path tests errored in both arms) | h3/cpu_B, h3/cpu_S, G3_CPU_COMPARE_r1_basetemp_slip.json |
| 06:50 | G1 r1: 611 passed, 1 failed (emptylast int4_linear_fused: unwritten-memory result bytes; Build B against itself differs) | g1/G1_r1.xml, g1/UNWRITTEN_REPEAT_B_{1,2}.json |
| 07:00 | G1 r2 after the test's UNWRITTEN repair: 612 passed | g1/G1.xml, g1/G1_*.json |
| 07:12-07:13 | G2 verdict self-test: 97 checks, 0 failed; CPU drive of the G2 children (site 2 reached for real on the CPU lane) | gpu_dry/VERDICT_SELFTEST.json, gpu_dry/G2_CPU_DRIVE.json |
| 07:14 | 30 GPU lines staged, each dry run exit 78 | GPU_LINES.json, gpu_dry/<line>.json |
| 07:15-07:24 | CPU suite r2 (cpu_B_r2, cpu_S_r2) after the runner repair: 4,278 IDs each, 0 differences | G3_CPU_COMPARE.json |
| 07:14 | ENG5-F2 subset check: 9 statements, all in ENG5-F2, helper identical | F2_SUBSET_CHECK.json |

## Prior art (Prior Art Directive)
- Returned-status checking: the CUDA Runtime API's error model (each call returns its cudaError_t; cudaGetLastError
  returns and resets the thread's last error) and NVIDIA samples' checkCudaErrors. Taken: test the call's own status.
  Ours: the message form and never touching the slot in the helper (ENG4-F3). Code site: kernels.cu:107-117.
- Fault injection by library interposition: LFI (Marinescu and Candea, 2009; unverified, lead to check "LFI library
  fault injection interposition"); LD_PRELOAD + dlsym(RTLD_NEXT) as in ENG3's counter. Ours: multi-slot arming and the
  allocator injection. Code site: tensor_cuda/tests/support/eng5s_cuda_shim.c header.
- Mutation testing: DeMillo, Lipton and Sayward, 1978. Code site: scripts/eng5s_arms.py header.
- Differential testing of two builds: McKeeman, 1998 (unverified, lead to check). Code site: test_eng5s_status.py.
- No prior art known to me: the G1 row list, the B-versus-S rule, the hooks-arm routing of site 9, the subset check.
