# BWD-R4b: stopped at the G1 floor rail (MFCC loss). Registration not sealed, nothing staged

I stopped the round before sealing, as the order requires. The floors were measured first, and for the MFCC loss, fp32 forward rounding alone puts the fp64 reference gradients within 10× of the G1 bars. That happens before any backward error is counted.

So `REGISTRATION_R4B.md` is neither written nor sealed, and no GPU line is staged or dry-run. G6, the package freeze, `MODULES_R4B.json` and the GPU worker, stage, suite and self-test scripts were not built or run, because they come after the rail.

Every other floor is far below its rail. The linear loss's G1 floor is 0, and G2's h vs h/2 floors are at most 1.0e-3 of ρ on both losses.

- **Ran:** 14:40:51Z (first line of `LEDGER_R4B.md`) to about 15:12Z.
- **Not done:** no GPU use, git write, delete, Voice write, or tensor_cuda or forward-kernel edit.
- **Report file:** the hook refused my full `REPORT_R4B.md` write with "Subagents should return findings as text". That is not the code-similarity hook, so per your rule the report content is this message. `REPORT_R4B.md` on disk is still the start stub. `LEDGER_R4B.md` is complete.

## Decision needed: the MFCC G1 reference

Under the order's definition, the reference is the fp64 chain run from the fp64 wave. A candidate with a perfect backward would already use about 63–83% of the style bar, and up to about 21% of a parameter's bar, from forward rounding alone. G1-MFCC would mostly measure forward fp32 error and the MFCC VJP's conditioning, not the backward. Three ways forward:
- **(A)** Keep the bars, and compare the MFCC arm against the fp64 backward fed `mfcc_loss_vjp` at the candidate's own fp32 wave. That isolates the backward the way the linear arm already is. The floor measured at the real wave would be reported beside it.
- **(B)** Keep the reference and re-derive the MFCC bars from the floor. That widens a bar, so only you can make it.
- **(C)** Keep both as written and accept that G1-MFCC may fail on forward rounding. The in-situ floor would attribute any failure.

I am not recommending one; it is your call.

## The floor that stopped the round (G1, MFCC loss)

`artifacts/bwd_r4b/host-reference/FLOORS_G1_R4B.json`, log `logs/bwd_r4b_host_reference.log` (rc 1, the registered stop).

**Method:** the fp64 backward fed `mfcc_loss_vjp` at an fp32 wave, compared with the same backward fed the cotangent at the fp64 wave, under R4a's G1 classes. The candidate's own fp32 wave exists only on the GPU, so I measured three CPU stand-ins and calibrated against ONNX Runtime (ORT):

| Stand-in wave | Wave rel L2 | dL/dwave rel | s_d floor (bar 1e-4) | Worst parameter / bar | Rows within 10× |
|---|---:|---:|---:|---:|---:|
| P1: fp64 wave rounded to fp32 | 2.52e-8 | 3.8e-6 | 1.0e-6 (0.010) | 0.0039 | 0 / 316 |
| P2: Voice's `ops.py` forward with FP64 set to float32 (in memory) | 2.15e-6 | 5.79e-5 | **6.29e-5 (0.63)** | 0.157 (`onnx::Conv_8019`) | **23** |
| P2k: P2's deviation scaled ×1.32 to ORT's | 2.84e-6 | 7.65e-5 | **8.31e-5 (0.83)** | 0.207 | **74** |

- **Calibration on the full LJ001-0004 input:** ORT's fp32 wave differs from the fp64 chain by 2.319e-6 relative L2 (BP1's G-FWD recorded 2.32e-6). The emulation differs by 1.754e-6. The MFCC VJP amplifies wave error about 27× at dL/dwave.
- **Rows within 10× under P2:** `s_d`, plus 22 parameters at 0.10–0.16 of their bar. These are generator `convs2.*.bias`, `noise_res.1.alpha2.*`, `ups.1.bias`, `noise_convs.1.bias` and `onnx::Conv_8019`.
- **Linear loss:** the floor is exactly 0 for every stand-in, because the candidate and the reference share the fp32 r.

## G2 floors (fp64 finite differences, both losses): far below the rail

Amendment 3's rules, on 8 slots: `s_d`, R4a's six, and conv_post `onnx::Conv_8064`. Each is probed along the aligned direction and its 3 largest-|g| coordinates. `s_d` also gets 8 seeded directions (PCG64 seed 0x593FB9A3), for 40 rows per loss. The crossing census covers all 13 LeakyReLU sites and the top_db clamp.

| | Linear | MFCC |
|---|---|---|
| Rows GREEN | 40 / 40 | 40 / 40 |
| Power (coordinate, dense) | 24/24, 16/16 | 24/24, 16/16 |
| A3-2 rows / rows halved for crossings | 0 / 16 | 0 / 13 |
| Worst h vs h/2 | 9.6e-7, which is 9.6e-4 of ρ | 1.0e-6, which is 1.0e-3 of ρ (`s_d` 1.4e-4 of ρ) |
| Plateau / bound, max-error / bound, \|fd\| / strength floor | ≤ 0.47, ≤ 0.50, ≥ 1.28 | ≤ 0.50, ≤ 0.54, ≥ 1.25 |

Receipts: `host-fd-{linear,mfcc}/FD_{PROBES,ORACLE,HOST_POWER,SUMMARY}_*_R4B.json` and `FD_DIRECTIONS_*_R4B.npz`. Logs: `logs/bwd_r4b_host_fd_*.log` (6 runs, all rc 0).

## Gate results (CPU parts)

| Item | Result | Receipt |
|---|---|---|
| L01 | python3 3.12.3, Voice venv python 3.12.3, nvcc 12.6.85, pytest 9.0.2. NVML no-op through ctypes: init rc 0, driver 595.99.02, `nvmlDeviceGetComputeRunningProcesses_v3` present. pynvml is in neither python. | `logs/bwd_r4b_l01_versions.log` |
| L02 smoke (CPU half) | **PASS.** fp64 full-chain d/ds_d for the linear loss: 128/128 nonzero, L2 0.4454287131, nonconstant. GPU half not staged. | `host-smoke/SMOKE_STYLE_LINEAR_R4B.json`, `logs/bwd_r4b_host_smoke.log` |
| G1 floors | **STOP**: MFCC within 10× (table above); linear 0 | `host-reference/FLOORS_G1_R4B.json` |
| G2 floors | far below the rail | `host-fd-*` |
| L101 zero census | **Literal rule fails.** All 29 of 29 of R4a's zero rows are nonzero in fp64 on both losses (L2 ≤ 9.6e-17 linear, ≤ 2.6e-16 MFCC; ≤ 1.45e-17 of the largest gradient). They exactly equal the small-norm rows, no row outside the set is zero, and the tail adds no zero tensor. The norm VJP's zero row-sum holds only to rounding, so "exactly zero" cannot hold in floating point. | `host-reference/REFERENCE_CENSUS_R4B.json` |
| G0, G1, G2 and G3 candidate, G4, G5, G-API | not staged (stop) | — |
| G6 | not run (stop) | — |
| Public-function host tests (smoke only, not a gate) | 27 passed, 61 GPU tests skipped, on a scratch package (`packages_r5d` plus my 2 tc_signal files, in `/tmp/bwd_r4b_probe/pkg`). The manifest census finds 16 BP1 and 36 BP2 conv geometries, equal to the tested labels. | the test file |

**Mutant table:** r4bm01 to r4bm04 are not built or run. Only the r4bm01 edit text is defined, in `scripts/bwd_r4b_common.py`.

**G6 per-ID summary:** not run.

**Staged lines with host peaks:** none.

## Ranking-eligible counts

`host-reference/RANKING_ELIGIBLE_R4B.json`:
- **Trained:** 315 (the 313 pre-tail tensors plus conv_post `w` and `b`).
- **Eligible:** 274, holding 53,241,978 elements.
- **Excluded:** 29 analytically zero, 12 identity-constant norms.
- **Each row gives:** name, shape, element count, status reason, and fp64 L2 per loss.
- **Frozen, not listed:** both iSTFT bases, `har`, the source spectrum, F0, N and `asr`.

## Diff summary and public signatures

**Tracked change:** `tc_signal/tc_signal/__init__.py`, +5 lines (the import and `__all__`).

**New files:**
- `tc_signal/tc_signal/_conv_backward.py`
- `tc_signal/tests/test_bwd_r4b_public_conv_backward.py`
- `scripts/bwd_r4b_common.py`
- `scripts/bwd_r4b_host.py`

No tensor_cuda file was touched.

The public signatures, so Voice can switch `tail_vjp.py`:
```
tc_signal.conv1d_backward(x, w, grad_out, *, stride=1, padding=0, dilation=1, groups=1, needs=(True, True, True)) -> (dx, dw, db)
tc_signal.conv_transpose1d_backward(x, w, grad_out, *, stride=1, padding=0, output_padding=0, dilation=1, groups=1, needs=(True, True, True)) -> (dx, dw, db)
```
- **Slots:** a slot not requested in `needs` comes back as `None`. Outputs are fresh fp32 TensorCUDA tensors with no autograd graph.
- **Refusals, all in Python, in this order:**
  1. Operands that are not TensorCUDA tensors. TensorCUDA always exports compact strides, so this is also the contiguity refusal.
  2. A dtype other than float32.
  3. Rank and positive extents, the attributes, groups, w's layout, and `grad_out` matching the forward output shape.
  4. `needs`.
  5. Device, last, so every other refusal can run with CUDA hidden.
- **Private entry:** `_conv_backward_run` is unchanged and still re-checks every operand.

## Scope checks

- **R4a receipts and scripts:** the content census covers `artifacts/bwd_r4/**` and the 25 `scripts/bwd_r4_*.py`, 363 files. Start and end lists are byte-identical (both sha256 `709fb535…`). Every R4b process also ran under a write guard that refuses writes to those paths.
- **Voice:** each of the 8 host runs read 78 files, with 0 changed and 0 write attempts. `tail_vjp.py` is still `aed3e3b9…`, and the Voice tree is clean.
- **Native builds match:** this branch's tensor_cuda sources equal the 47 sources core `430c829e…` was built from. Its tc_signal native sources equal R3's (`0d6f6ddf…`).

## Deviations

- The first pynvml probe in L01 had a shell-quoting slip; I repaired it in the same log.
- R4a's `scripts/bwd_r4_stage.py` and `scripts/bwd_r4_suite_r2.py` are untracked in R4a's worktree and absent here. So `bwd_r4_stage_r5*.py` and `bwd_r4_suite_r5*.py` cannot be imported on this branch, and R4b needs its own stage and suite scripts. Neither is written yet, because of the stop.
- The FD oracle ran in two parts per loss, then merged, to fit the 10-minute tool limit. The merge re-derives the probe list and requires identical part manifests and direction archives. No rule changed.
- The reference uses R4a's operands (DESIGN §6): the fp32 source spectrum and the fp32-stored r, converted to fp64. BP1's `chain()` uses fp64 for both.
- `REPORT_R4B.md` write refused by the hook; its content is this message.
- Script repairs: 0 of 3.

## Prior art

- **Reverse-mode differentiation:** Linnainmaa (1970) and Wengert (1964). Voice's BP1 fp64 pullbacks (2026) are used as they are.
- **Gradient checkpointing:** Chen et al. (2016). Not exercised before the stop.
- **MFCCs:** Davis and Mermelstein (1980), through Voice's `mfcc.py`.
- **STFT and iSTFT:** Griffin and Lim (1984). **iSTFTNet:** Kaneko et al. (2022), as Voice's tail.
- **NVML:** NVIDIA, through ctypes; only the no-op ran.
- **Finite differences and probe seeds:** centered differences after Higham (2002) and BP1's G-CHAIN; PCG64 (O'Neill, 2014); Amendments 2 and 3 reused unchanged.
- **Public functions:** convolution adjoints from Dumoulin and Visin (2016); the API shape from PyTorch's aten `convolution_backward` output mask and `torch.nn.grad.conv1d_*` (Paszke et al., 2019).
- **Write guard:** PEP 578 audit hooks.
- **What is mine:**
  - the floor method (three fp32 stand-ins with the ORT calibration);
  - the full-chain probe slots and the per-loss crossing census;
  - the public functions' TensorCUDA-only contract and refusal order.
- **Unverified, lead to check:** "Linnainmaa 1970", "Davis Mermelstein 1980", "Griffin Lim 1984", "Kaneko iSTFTNet 2022", "Chen sublinear memory 2016", "Higham 2002", "O'Neill PCG64", "Dumoulin Visin 2016", "aten convolution_backward output_mask", "PEP 578".

## Files written

Everything is in `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_bwd/WorkTree/tensor_fork_bwd_wt_r4b/`. The full list, including what was not written because of the stop, is in `artifacts/bwd_r4b/FILES_WRITTEN_R4B.txt`.

- **Code:**
  - `tc_signal/tc_signal/__init__.py` (+5)
  - `tc_signal/tc_signal/_conv_backward.py`
  - `tc_signal/tests/test_bwd_r4b_public_conv_backward.py`
  - `scripts/bwd_r4b_common.py`
  - `scripts/bwd_r4b_host.py`
- **Logs:** `logs/bwd_r4b_l01_versions.log`, `logs/bwd_r4b_host_smoke.log`, `logs/bwd_r4b_host_reference.log`, `logs/bwd_r4b_host_fd_{linear,mfcc}_p{0,1}of2.log`, `logs/bwd_r4b_host_fd_merge_{linear,mfcc}.log`
- **Ledger and censuses:**
  - `artifacts/bwd_r4b/LEDGER_R4B.md`
  - `artifacts/bwd_r4b/REPORT_R4B.md` (stub)
  - `artifacts/bwd_r4b/FILES_WRITTEN_R4B.txt`
  - `artifacts/bwd_r4b/CENSUS_START_R4B.sha256` and `CENSUS_END_R4B.sha256`
- **Receipts:** `artifacts/bwd_r4b/host-smoke/`, `host-reference/`, `host-fd-linear/`, `host-fd-mfcc/` and the four `host-fd-*-p*of2/` part directories.
- **Scratch left in /tmp:** `/tmp/bwd_r4b_probe/` (package copies for the host-test smoke) and `/tmp/bwd_r4b_host/`. Nothing went to DELETE_ME.

_Filed by the lead from the seat's final message; the seat's own REPORT_R4B.md write was refused by a hook._
