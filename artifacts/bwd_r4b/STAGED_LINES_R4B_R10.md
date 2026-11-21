# BWD-R4b r10 staged lines (lead only)

Restaged: `g1-linear`, `g1-mfcc`, `r4bm03_mfcc_vjp_sign`, `g5-checkpoint`, `g5-plain`, `g-api`, `r4bm01_public_dw_transposed_kernel`, `r4bm05_public_end_padding_extent` and the four ref-at-wave commands. **Unaffected r4 lines** (keep their
r4 receipts, `STAGED_LINES_R4B_R4.md`): `g0-crop`, `g0-full`, `g3`, `g4-fresh-linear`, `g4-fresh-mfcc`, `g4-session-linear`, `g4-session-mfcc`, `r4bm02_wave_upstream_dropped`, `r4bm04_conv_post_bias_grad_dropped`, `wave-capture`.
Pin comparison (`PIN_COMPARISON_R4B_R10.json`): r4 pins changed ['scripts/bwd_r4b_stage.py', 'tc_signal/tests/test_bwd_r4b_public_conv_backward.py']; r5 pins changed
['scripts/bwd_r4b_stage.py', 'scripts/bwd_r4b_stage_r5.py', 'scripts/bwd_r4b_verify.py', 'tc_signal/tests/test_bwd_r4b_public_conv_backward.py']; r6 pins changed ['scripts/bwd_r4b_refwave_r6.py', 'scripts/bwd_r4b_stage_r6.py', 'scripts/bwd_r4b_verify.py', 'tc_signal/tests/test_bwd_r4b_public_conv_backward.py']; r7 pins changed
['scripts/bwd_r4b_refwave_r7.py', 'scripts/bwd_r4b_stage_r7.py', 'scripts/bwd_r4b_verify.py']; r8 pins changed ['scripts/bwd_r4b_gpu_r8.py', 'scripts/bwd_r4b_refwave_r8.py', 'scripts/bwd_r4b_stage_r8.py', 'scripts/bwd_r4b_verify.py']; r9 pins changed ['scripts/bwd_r4b_refwave_r9.py', 'scripts/bwd_r4b_stage_r9.py', 'scripts/bwd_r4b_verify.py']; none of the files the unaffected lines execute changed, and packages_r4 is
unchanged. G-API census: `G_API_CENSUS_R4B_R10.json` (103 IDs).

**Order.** (1) the four ref-at-wave commands (root `ref-at-wave-r10`; one attempt per mode per root: if any command
fails, rerun all four on a new root `ref-at-wave-r10-a2` (then -a3 ...) and pass `--ref-root ref-at-wave-r10-a2` to
`g1-mfcc` and `r4bm03`; the producer is deterministic at 3 BLAS threads, so its manifest must equal the sealed digest
634bf770...; the wrapper checks that and the bundle before it writes the manifest or a passing verdict);
(2) `g1-mfcc` and `r4bm03`; `g-api` before `r4bm01` and `r4bm05`; the rest are independent. G5's figure is a sampled
estimate (max of the sampled NVML per-process peak and R4a's basis), not a proven bound; bar 4.0 GB unchanged.
GPU lines: both address-space limits unlimited, MemoryMax = expected host peak + 1,000,000,000.

| Line | Item | Depends on | Expected seconds | Min / max seconds | Estimated device bytes | Expected peak host bytes | MemoryMax bytes |
|---|---|---|---:|---:|---:|---:|---:|
| `g1-linear` | G1 + G2 linear (r9 verifier: exact typed joins) | - | 120 | 240 / 540 | 2,000,000,000 | 5,500,000,000 | 6,500,000,000 |
| `g1-mfcc` | A1-1 G1 + A2-1 G2 MFCC (exact joins, single-attempt root) | wave-capture (r4, unaffected), ref-at-wave-r10 | 170 | 340 / 540 | 2,000,000,000 | 8,000,000,000 | 9,000,000,000 |
| `r4bm03_mfcc_vjp_sign` | mutant, caught by G1 (MFCC), r9 verifier | wave-capture (r4, unaffected), ref-at-wave-r10 | 140 | 280 / 540 | 2,000,000,000 | 7,000,000,000 | 8,000,000,000 |
| `g5-checkpoint` | G5 10 s sampled estimate, header values checked | - | 220 | 440 / 540 | 3,500,000,000 | 5,000,000,000 | 6,000,000,000 |
| `g5-plain` | G5 no-checkpoint arm (report only), header values checked | - | 180 | 360 / 540 | 9,100,000,000 | 5,000,000,000 | 6,000,000,000 |
| `g-api` | G-API: census, explicit JUnit totals, launch provenance | - | 240 | 480 / 540 | 1,500,000,000 | 3,500,000,000 | 4,500,000,000 |
| `r4bm01_public_dw_transposed_kernel` | mutant, caught by G-API (verified clean baseline) | g-api | 240 | 480 / 540 | 1,500,000,000 | 3,500,000,000 | 4,500,000,000 |
| `r4bm05_public_end_padding_extent` | mutant, caught by G-API (verified clean baseline) | g-api | 240 | 480 / 540 | 1,500,000,000 | 3,500,000,000 | 4,500,000,000 |

```bash
systemd-run --user --scope -p MemoryMax=6500000000 -p OOMPolicy=continue timeout 570s env FORGE_GPU_DRY_RUN=0 /usr/bin/python3 -B -P scripts/bwd_r4b_stage_r10.py g1-linear --min 240 --max 540
systemd-run --user --scope -p MemoryMax=9000000000 -p OOMPolicy=continue timeout 570s env FORGE_GPU_DRY_RUN=0 /usr/bin/python3 -B -P scripts/bwd_r4b_stage_r10.py g1-mfcc --min 340 --max 540
systemd-run --user --scope -p MemoryMax=8000000000 -p OOMPolicy=continue timeout 570s env FORGE_GPU_DRY_RUN=0 /usr/bin/python3 -B -P scripts/bwd_r4b_stage_r10.py r4bm03_mfcc_vjp_sign --min 280 --max 540
systemd-run --user --scope -p MemoryMax=6000000000 -p OOMPolicy=continue timeout 570s env FORGE_GPU_DRY_RUN=0 /usr/bin/python3 -B -P scripts/bwd_r4b_stage_r10.py g5-checkpoint --min 440 --max 540
systemd-run --user --scope -p MemoryMax=6000000000 -p OOMPolicy=continue timeout 570s env FORGE_GPU_DRY_RUN=0 /usr/bin/python3 -B -P scripts/bwd_r4b_stage_r10.py g5-plain --min 360 --max 540
systemd-run --user --scope -p MemoryMax=4500000000 -p OOMPolicy=continue timeout 570s env FORGE_GPU_DRY_RUN=0 /usr/bin/python3 -B -P scripts/bwd_r4b_stage_r10.py g-api --min 480 --max 540
systemd-run --user --scope -p MemoryMax=4500000000 -p OOMPolicy=continue timeout 570s env FORGE_GPU_DRY_RUN=0 /usr/bin/python3 -B -P scripts/bwd_r4b_stage_r10.py r4bm01_public_dw_transposed_kernel --min 480 --max 540
systemd-run --user --scope -p MemoryMax=4500000000 -p OOMPolicy=continue timeout 570s env FORGE_GPU_DRY_RUN=0 /usr/bin/python3 -B -P scripts/bwd_r4b_stage_r10.py r4bm05_public_end_padding_extent --min 480 --max 540
```

## ref-at-wave (lead CPU commands)

Expected peak host bytes 6,000,000,000 each (RLIMIT_AS 10 GiB inside, MemAvailable >= 22 GiB). Measured by the lead in r6: reference 26.4 s / 2.39 GB, fd parts 532 s and 533 s in parallel / 2.05 GB each, merge 21.5 s / 1.94 GB; the r7 merge adds the bundle check (r7 self-test: 71.3 s / 2.76 GB under load 15).

| Command | Expected seconds | Timeout |
|---|---:|---:|
| `reference` | 30 | 330 s |
| `fd-p0of2` | 540 | 840 s |
| `fd-p1of2` | 540 | 840 s |
| `merge` | 90 | 390 s |

```bash
systemd-run --user --scope -p MemoryMax=7000000000 -p OOMPolicy=continue timeout 330s nice -n 19 env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=3 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 PYTHONDONTWRITEBYTECODE=1 TMPDIR=/tmp/bwd_r4b_r10_ref_at_wave /usr/bin/python3 -B -P scripts/bwd_r4b_refwave_r10.py reference --root artifacts/bwd_r4b/ref-at-wave-r10 > logs/bwd_r4b_r10_ref_at_wave_reference.log 2>&1
systemd-run --user --scope -p MemoryMax=7000000000 -p OOMPolicy=continue timeout 840s nice -n 19 env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=3 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 PYTHONDONTWRITEBYTECODE=1 TMPDIR=/tmp/bwd_r4b_r10_ref_at_wave /usr/bin/python3 -B -P scripts/bwd_r4b_refwave_r10.py fd --part 0 --parts 2 --root artifacts/bwd_r4b/ref-at-wave-r10 > logs/bwd_r4b_r10_ref_at_wave_fd-p0of2.log 2>&1
systemd-run --user --scope -p MemoryMax=7000000000 -p OOMPolicy=continue timeout 840s nice -n 19 env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=3 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 PYTHONDONTWRITEBYTECODE=1 TMPDIR=/tmp/bwd_r4b_r10_ref_at_wave /usr/bin/python3 -B -P scripts/bwd_r4b_refwave_r10.py fd --part 1 --parts 2 --root artifacts/bwd_r4b/ref-at-wave-r10 > logs/bwd_r4b_r10_ref_at_wave_fd-p1of2.log 2>&1
systemd-run --user --scope -p MemoryMax=7000000000 -p OOMPolicy=continue timeout 390s nice -n 19 env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=3 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 PYTHONDONTWRITEBYTECODE=1 TMPDIR=/tmp/bwd_r4b_r10_ref_at_wave /usr/bin/python3 -B -P scripts/bwd_r4b_refwave_r10.py merge --root artifacts/bwd_r4b/ref-at-wave-r10 > logs/bwd_r4b_r10_ref_at_wave_merge.log 2>&1
```
