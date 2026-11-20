# BWD-R1 report

**RED on the launcher-invocation rule.** The build, host hook tests, host mutants, G6 per-ID comparison and G7 file-scope audit passed. The six CUDA lines are staged for the lead. The round has no GPU acceptance result.

## Diff summary and pins

Five product files changed, with 300 insertions and three replaced lines. The native `attach_vjp` node reuses output storage, snapshots registered revisions, validates all returned slots before accumulating, and releases its callback on execution or graph destruction. List additional tensors saved by the callback as frozen input slots. The binding and package export are additive. The fp32 matmul VJPs use a separate legacy-stream cuBLAS handle per device, a 32 MiB private workspace, atomics disabled and pedantic compute. The forward call and CUDA kernel source are byte-identical to the entry snapshot. Added three test modules and the gate runners. [Product diff](PRODUCT.diff) · [Build and source hashes](BUILD_candidate.json)

- Candidate module SHA-256 `78c2b99be540e6a925e708141090ab724bd1056f27fd469d175d216d1e0d0613`. The final restored build reproduced it.
- `TOLERANCES.json` SHA-256 `4f189cb9e675a02433625cda4b5436613c436e5611348d168f08a84f1b5c6f8d`, sealed `2026-11-18T20:28:13Z` before comparisons. [Seal](TOLERANCES.sha256)

## CPU results

- **G1 build passed.** CUDA 12.6.85, Release, sm_89, nice 19 and at most four build workers. [Build log](../../logs/bwd_r1_build_candidate.log)
- **G2 host passed.** 19 passed, seven CUDA cases staged. The host smoke used the inverse-reshape VJP. Other host cases covered ordered input slots, frozen saves, structural zeros, bypass, lifetime and exception paths, atomic validation, revision checks and re-entrancy. [Host log](../../logs/bwd_r1_host_hook.log)
- **G3 oracle checks passed.** 43 cells, 1,642 central finite-difference projections at three step sizes. Maximum ratio to the fp64 bound was 0.002958. Voice supplied add, constant div, LeakyReLU and Snake. Local NumPy supplied the others. The test plan includes BP1 seeded Kokoro geometries and an actual ONNX tail-input activation. Engine comparisons are staged. [Reference census](REFERENCE_CENSUS.json)
- **G4 host canary qualified.** TF32 operand rounding changes the fp64-reference gradients by up to 6.176× the matmul bound for dA and 5.951× for dB. Strict dispatch, five byte-identical repeats and the forward comparison to PROD are GPU gates. [Host witness](CANARY_REFERENCE.json)
- **G7 file-scope audit passed.** 19,411 protected files and seat Git administrative state matched. Zero GPU leases, explicit deletes or DELETE_ME moves. [Scope receipt](G7.json)

## Mutants

All five built incrementally. The three host killing tests ran on their pinned mutant builds. [Host mutant receipts](G5_CPU_MUTANTS.json)

| Mutant | Result | Edited path evidence |
|---|---|---|
| `r1m01_hook_drops_grad` | Killed, exit 1 | Inverse-reshape callback executed and `AssertionError: attached gradient was dropped`. |
| `r1m02_hook_slot_misaligned` | Killed, exit 1 | Ordered-slot callback executed and `Mismatched elements: 2 / 2 (100%)`. |
| `r1m03_revision_check_off` | Killed, exit 1 | Both stale callbacks executed after revision bumps. Each reported `Failed: DID NOT RAISE <class 'RuntimeError'>`. |
| `r1m04_matmul_tf32` | Built, staged | Kill requires `pedantic_compute=0` and the strict configuration error. |
| `r1m05_matmul_unordered_workspace` | Built, staged | Resetting the stream clears the workspace association. Kill requires `workspace_bytes=0` and the strict configuration error. |

## G6 whole CPU suite

**Per-ID PASS against production `3138cf54`.** All 3,864 existing outcomes matched. The only 77 differences were new BWD-R1 IDs, with 19 passed and 58 CUDA-skipped. Zero unexpected differences. Both MAINCHK3 suite runs completed with raw pytest exit 1 and the same existing 2,176 failed, 15 test errors and one collection error. PROD had 1,340 passed and 332 skipped. The candidate had 1,359 passed and 390 skipped. Each process had a 12 GiB address-space limit and the registered headroom. [Every ID](G6_PER_ID.json)

## LeakyReLU tie census

Voice's `x >= 0` branch has derivative 1 at zero. KokoroTC uses `x.maximum(x * slope)`. TensorCUDA's maximum VJP uses `a >= b` for the first input and `b > a` for the second. At either signed zero the raw slot derivatives are `(1, 0)`, giving the composition derivative 1 for slopes 0.2, 0.1 and 0.01. This is a source census. The raw maximum and composition runtime tests are staged. [Census](OPERAND_CENSUS.json)

## Deviations

- `forge-gpu-run --help` omitted `FORGE_GPU_DRY_RUN=1`. Argument parsing printed usage and exited before admission. The preflight and all lease-capable seat invocations used dry-run mode.
- A quoting error interrupted the first registration write. The partial draft and timestamp are retained. The sealed registration and tolerances preceded comparisons.
- One script repair addressed `ModuleNotFoundError: No module named 'bwd_r1_gate'` under Python `-P` before any mutant build. The closeout canary command also corrected an awk syntax error in its startup-memory check before running. [Ledger](LEDGER.md)

## Prior art

Linnainmaa (1970) and Baydin et al. (2018) supply reverse-mode differentiation. PyTorch `autograd.Function` and saved tensors (2019) and JAX `custom_vjp` (Bradbury et al., 2018) are custom VJP precedents. TensorCUDA's frozen revision-save slots, one-shot ownership and validation-before-accumulation are this integration. NVIDIA cuBLAS 12.6 (2024) supplies pedantic compute controls and workspace rules. Harris (2007) supplies fixed-order CUDA reduction context. The private backward handle, workspace and dispatch checks are this implementation. NVIDIA Ampere TF32 (2020) supplies the witness rounding format. Fornberg (1988), DeMillo–Lipton–Sayward (1978), McKeeman (1998) and MAINCHK3 (2026) supply centered differences, mutation testing and per-ID differential comparison. Maas–Hannun–Ng (2013) and Liu et al. (2020) supply LeakyReLU and Snake. The fixture choices and failure witnesses are specific to R1. Unverified — lead to check those names and the cuBLAS 12.6 reproducibility, `CUBLAS_COMPUTE_32F_PEDANTIC`, and `cublasSetWorkspace` documentation.

## Files written

The complete path inventory includes the five product files, three test modules, all BWD-R1 scripts, evidence and logs, frozen packages, build products and retained scratch files. The dispatch shim's pre-existing log is excluded. [Every file and symlink](FILES_WRITTEN.txt)

# Staged GPU lines for the lead

Run from the BWD-R1 worktree root. Run G4-strict before G5-m04 and G5-m05. Each command takes its own lease.

```bash
cd /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_bwd/WorkTree/tensor_fork_bwd_wt_r1
timeout 570s env FORGE_GPU_DRY_RUN=0 /usr/bin/python3 -B -P scripts/bwd_r1_stage.py G2-hook --min 45 --max 540
timeout 570s env FORGE_GPU_DRY_RUN=0 /usr/bin/python3 -B -P scripts/bwd_r1_stage.py G3-core --min 90 --max 540
timeout 570s env FORGE_GPU_DRY_RUN=0 /usr/bin/python3 -B -P scripts/bwd_r1_stage.py G4-strict --min 45 --max 540
timeout 570s env FORGE_GPU_DRY_RUN=0 /usr/bin/python3 -B -P scripts/bwd_r1_stage.py G4-forward --min 120 --max 540
timeout 570s env FORGE_GPU_DRY_RUN=0 /usr/bin/python3 -B -P scripts/bwd_r1_stage.py G5-m04 --min 35 --max 540
timeout 570s env FORGE_GPU_DRY_RUN=0 /usr/bin/python3 -B -P scripts/bwd_r1_stage.py G5-m05 --min 35 --max 540
```

Each launcher sets its full environment, including `CUDA_VISIBLE_DEVICES=0`, `CUDA_MODULE_LOADING=EAGER`, `NVIDIA_TF32_OVERRIDE=0` and scratch under `/tmp/bwd_r1_r1`. It checks module and package pins before requesting a lease. The forward line pins both modules. G3 also verifies the sealed tolerances and Voice input hashes. All leases have a 30-second margin and a 470-second inner timeout.

The four baseline lines expect forge and launcher exit 0. Each mutant expects pytest and forge exit 1 with the strict configuration error and its edited field in the dispatch trace. The launcher returns 0 for a confirmed kill. Save each real forge exit and log SHA-256 from `artifacts/bwd_r1/gpu_real_<line>.json`. Logs are at `logs/bwd_r1_gpu_real_<line>.log`. The forward line compares 72 raw-word arrays and writes `artifacts/bwd_r1/G4_FORWARD.json`.

For dry runs, change only `FORGE_GPU_DRY_RUN=0` to `FORGE_GPU_DRY_RUN=1`. All six returned 78 with `"would_start": true`, before lease acquisition. See `artifacts/bwd_r1/DRY_RUNS.json`.
