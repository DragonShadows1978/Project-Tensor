# BWD-R4b r3: Amendment 2's rail passes, registration sealed, 17 GPU lines staged and dry-run

The seat's part is complete. G6 passes against both baselines. Every GPU line dry-ran with exit 78, `"would_start": true`, no lease and no worker. Nothing has run on a GPU yet. The hook refused `REPORT_R4B.md` again, so this message is the report; the file is still the round-1 stub.

- **Start:** "r3 start 2026-11-19T15:21Z" in `LEDGER_R4B.md`.
- **Finish:** about 15:57Z.
- **Not done:** no GPU use, no git write, no delete, no Voice write, no tensor_cuda edit.

## A2-1 floor (r3 L02 smoke): passes

Run: `scripts/bwd_r4b_host.py a2-floor`, receipt `host-a2-floor/A2_FLOOR_R4B.json`, log `logs/bwd_r4b_host_a2_floor.log`.

The 40 A2-1 probes were selected from the A1-1 reference gradient on the P2 stand-in. On every row, the cotangent's fp32 cast moves the projection by at most 0.0020 of ρ (worst row: `s_d` random direction 4). No row is within 10×, so the rail holds with about 490× to spare.

**ref-at-wave, run in full on the P2 stand-in** (`selftest-ref-at-wave-P2`):

| Step | Wall time | Peak RSS |
|---|---:|---:|
| reference | 16.6 s | 2.38 GB |
| fd part 0 / part 1 (in parallel) | 376 s / 366 s | 2.06 GB each |
| merge | 9.8 s | 1.94 GB |

- **A2-1 oracle:** 40/40 GREEN, power 24/24 coordinate and 16/16 dense.
- **Its floors:** worst h vs h/2 floor 9.6e-4 of ρ; plateau/bound at most 0.48; |fd| at least 1.25× the strength floor.
- **Registered peak per command:** 6.0e9 bytes.

## Cotangent census (before sealing)

The operand census grep is in `logs/bwd_r4b_cotangent_census.log`. The full table is section 3 of `REGISTRATION_R4B.md`.

| Gate or line | Cotangent it uses | Governing rule | Gated? |
|---|---|---|---|
| g1-mfcc G1, r4bm03 | A1-1 reference at `wave_c`, computed by ref-at-wave itself | A1-1 | yes |
| g1-mfcc G2 | A2-1 oracle of ⟨g_c, wave(θ)⟩ at `wave_c` | A2-1 | yes |
| g1-mfcc end-to-end check | fp64-wave reference | A1-2 | report only |
| g1-mfcc round-1 FD comparison | round 1's fp64-wave FD | A2-2 | report only |
| ref-at-wave | derives g_c from `wave_c` itself | A1-1, A2-1 | producer |
| g3, g4-*-mfcc, g5-* | each run's own wave (candidate against candidate) | order G3/G4/G5 | yes, no fp64 reference |
| linear arm | shared fp32 r | order | yes |

No gated comparison uses the fp64-wave cotangent; only the two report-only arms do.

## Sealed registration

`artifacts/bwd_r4b/REGISTRATION_R4B.md`, sha256 `60cb94dbd83bae8f52ced97e537abc8514e26b53b832967d094178a0d7748675`. The seal file carries the date line `2026-11-19T15:56:23Z`, written by the same command as the sha256.

- **How it was written:** `scripts/bwd_r4b_register.py` reads every number from its receipt.
- **Contents:** Amendments 1 and 2, folded in. `AMENDMENT1_R4B.md` (`7532d2c4…`) and `AMENDMENT2_R4B.md` (`0dd5f5a2…`) are extracted verbatim from the orders.
- **Bars:** unchanged. G1 uses R4a's classes (`TOLERANCES_R4.json` `27052e94…`); G2 uses ρ of 1e-4 for style and 1e-3 for parameters with Amendment 3's rules.

**Floors recorded in the registration**

| Floor | Result | Within 10×? |
|---|---|---|
| A1-1 cast floor | s_d 2.69e-3 of its bar; worst parameter 1.69e-3 | none |
| A1-3 zero set | largest set row ≤ 1.45e-17 of the largest tensor; gap 1.74e12 to 6.52e12 | — |
| G2 h vs h/2, linear | worst 9.6e-4 of ρ | none |
| G2 h vs h/2, round-1 MFCC | worst 1.0e-3 of ρ | none |
| G2 h vs h/2, A2-1 | worst 9.6e-4 of ρ | none |
| A2-1 rail | worst 2.0e-3 of ρ | none |

## CPU gate results

| Item | Result | Receipt |
|---|---|---|
| G1-build | Packages frozen; no native rebuild. `MODULES_R4B.json` (`8a41dd19…`) maps the 73 engine source files to the builds the frozen modules came from. | `logs/bwd_r4b_freeze.log` |
| L02 smokes | Round 1 linear s_d, the A1-1 cast floor, and the A2-1 floor all pass. | `host-smoke/`, `host-a1-floors/`, `host-a2-floor/` |
| Worker self-test (CUDA hidden) | 18/18 checks pass on the final script bytes. | `selftest-worker-r4/SELFTEST_R4B.json`, `logs/bwd_r4b_selftest_worker_r4.log` |
| ref-at-wave self-test | Passes, timings above. | `selftest-ref-at-wave-P2/` |
| Ranking list | 274 of 315 tensors eligible (53,241,978 elements); 29 zero-set and 12 identity norms excluded. | `host-reference/RANKING_ELIGIBLE_R4B.json` |
| G6 | Passes against both baselines (below). | `G6_PER_ID_R4B.json` |
| Censuses | R4a's 363 files are identical at start and end. 20 Voice receipts show 0 changes and 0 write attempts. Staging pins re-checked at the end (`71ff0028…`). | `CENSUS_END_R3_R4B.sha256` |

The self-test covers the reference chain and its refusal of a different wave, the G1/G2 paths for both losses, the end-to-end and round-1 report paths, the NVML sampler on a fake library (2.07 ms median interval), the G5 operands, and the G-API pytest path (28 passed, 60 GPU tests skipped).

## Mutant table

Each GPU kill is a staged line. The CPU column applies the same edit to the fp64 chain and scores it with the same comparison code.

| Mutant | Edit | Caught by (GPU line) | CPU half |
|---|---|---|---|
| r4bm01_public_dw_transposed_kernel | `dw.flip([2])`, public path only, with an `R4BM01_EDIT` marker | G-API: a test_gpu failure with the marker, after a passing g-api | the flipped dw fails R3's conv bound on `conv1d_kokoro` (K = 3) |
| r4bm02_wave_upstream_dropped | zero upstream at the wave, counted | G1, linear | 316/316 rows fail |
| r4bm03_mfcc_vjp_sign | dL/dwave negated, counted | G1, MFCC, against the A1-1 reference | clean run 0 failures; mutant 287 rows fail |
| r4bm04_conv_post_bias_grad_dropped | tail_vjp wrapper returns db None, counted | G1, linear | exactly `conv_post.bias` fails |

## G6 per-ID summary

Each arm ran under RLIMIT_AS 12 GiB, started with at least 34.9 GB MemAvailable, and did not wait. Logs: `logs/bwd_r4b_g6_*.log`.

| Arm | Passed | Skipped | Failed | Error | Collection error |
|---|---:|---:|---:|---:|---:|
| r4a (`cd7d7d4d`) | 1347 | 626 | 2171 | 15 | 1 |
| prod (`3138cf54`) | 1222 | 339 | 2171 | 15 | 1 |
| candidate | 1374 | 687 | 2171 | 15 | 1 |

- **Against R4a (`cd7d7d4d`):** 88 changed IDs, all new `test_bwd_r4b_*` (27 passed, 61 skipped); 0 unexpected.
- **Against production (`3138cf54`):** exactly R4a's 412 r5d changes, with the same outcomes, plus the 88 new IDs; 0 unexpected, 0 missing.
  - r1: 52 passed, 88 skipped;
  - r2: 24 passed, 61 skipped (this includes R4a's renamed test);
  - r3: 38 passed, 129 skipped;
  - r4: 11 passed, 9 skipped;
  - r4b: 27 passed, 61 skipped.

## Staged lines

Pins: `STAGED_PINS_R4B.json`, 31 pins, sha256 `71ff0028…`. All 17 dry runs gave exit 78, `"would_start": [true]`, no lease and no worker (`dry_*_R4B.json`, `logs/bwd_r4b_dry_runs.log`).

| Line | Depends on | Expected s | Min / max | Expected peak host bytes |
|---|---|---:|---:|---:|
| g0-crop | — | 60 | 120 / 540 | 4.0e9 |
| g0-full | — | 90 | 180 / 540 | 4.5e9 |
| wave-capture | — | 45 | 90 / 540 | 3.5e9 |
| *ref-at-wave (4 lead CPU commands: reference, fd p0, fd p1, merge)* | wave-capture | 60 / 380 / 380 / 60 | timeouts 360 / 800 / 800 / 360 | 6.0e9 each |
| g1-linear | — | 120 | 240 / 540 | 5.0e9 |
| g1-mfcc | wave-capture, ref-at-wave | 150 | 300 / 540 | 6.5e9 |
| g3 | — | 120 | 240 / 540 | 5.0e9 |
| g4-session-linear | — | 120 | 240 / 540 | 4.5e9 |
| g4-session-mfcc | — | 120 | 240 / 540 | 4.5e9 |
| g4-fresh-linear | g4-session-linear | 70 | 140 / 540 | 4.5e9 |
| g4-fresh-mfcc | g4-session-mfcc | 70 | 140 / 540 | 4.5e9 |
| g5-checkpoint | — | 220 | 440 / 540 | 4.5e9 |
| g5-plain (report only) | — | 180 | 360 / 540 | 4.5e9 |
| g-api | — | 240 | 480 / 540 | 3.0e9 |
| r4bm01 | g-api | 240 | 480 / 540 | 3.0e9 |
| r4bm02 | — | 100 | 200 / 540 | 5.0e9 |
| r4bm03 | wave-capture, ref-at-wave | 120 | 240 / 540 | 6.5e9 |
| r4bm04 | — | 100 | 200 / 540 | 5.0e9 |

The full commands are in `STAGED_LINES_R4B.md`. Each GPU line runs in a scope with MemoryMax = its host peak + 1e9; each ref-at-wave command has MemoryMax 7e9.

**Running the A1-1 / A2-1 sequence:**
1. Run `wave-capture`. It requires the grad-off and grad-on forwards under both schedules to give byte-identical waves.
2. Run the four `ref-at-wave` CPU commands. They find `wave_c` through the newest passing wave-capture receipt and check its sha256.
3. Run `g1-mfcc` and `r4bm03`. Before the lease, each checks ref-at-wave's verdict and pins; in the worker, each checks its own forward wave against `wave_c`.

**Timing caveat:** the expected seconds are estimates scaled from R4a's measured lines. If a line runs past its lease, that is a lease-size question for you.

## Deviations

- **Script repairs:** 2 of 3, both in `scripts/bwd_r4b_selftest.py`. The refusal check ran outside its patch (FileNotFoundError), and my expected skip count was 61 instead of the actual 60. No gate or product code changed.
- **Change after the first passing self-test:** the r4bm01 line now runs the mutant package only and takes its clean run from the newest passing g-api receipt, because two pytest runs would not fit the lease. The self-test was rerun on the final bytes (`selftest-worker-r4`).
- **A2-2's "same probes":** read as round 1's fp64-wave MFCC probes, since that is the only set with an fp64-wave FD oracle. ref-at-wave records the shift on those probes.
- **G1-build:** no native recompile. Proof that the modules come from this branch is source equality, recorded in `MODULES_R4B.json`.
- **G5 sampler criterion:** median interval ≤ 5 ms, with the maximum reported.
- **Registration file:** written by `scripts/bwd_r4b_register.py`, not hand-typed.
- **G6 R4a arm:** built from `git archive cd7d7d4d` (read-only).

## Prior art

- **Gradient checkpointing:** Chen et al. (2016).
- **Reverse-mode differentiation:** Linnainmaa (1970).
- **MFCCs:** Davis and Mermelstein (1980), via Voice's `mfcc.py`.
- **STFT/iSTFT:** Griffin and Lim (1984).
- **iSTFTNet:** Kaneko et al. (2022).
- **NVML:** `nvmlDeviceGetComputeRunningProcesses_v3`, called through ctypes.
- **Centered differences and step checks:** Higham (2002), BP1 G-CHAIN, and R4a's Amendments 2 and 3.
- **PCG64:** O'Neill (2014).
- **Conv adjoints:** Dumoulin and Visin (2016).
- **Public API shape:** PyTorch's aten `convolution_backward` (Paszke et al., 2019).
- **Mutation testing:** DeMillo, Lipton and Sayward (1978).
- **Write guard:** PEP 578 audit hooks.
- **What is mine:**
  - the linearized-loss oracle at the candidate's wave and the receipt chain from the wave's sha256 to the GPU lines;
  - the per-process sampler with phase marks;
  - the four R4b mutants;
  - the floor method with fp32 stand-ins;
  - the source-equality build proof.
- **Unverified, lead to check:** each name above.

## Files written in r3

All under the worktree. The full list is appended to `artifacts/bwd_r4b/FILES_WRITTEN_R4B.txt`.

- **Scripts:**
  - `scripts/bwd_r4b_host.py` (modified);
  - new: `scripts/bwd_r4b_refwave.py`, `bwd_r4b_gpu.py`, `bwd_r4b_stage.py`, `bwd_r4b_selftest.py`, `bwd_r4b_suite.py`, `bwd_r4b_register.py`.
- **Amendment:** `artifacts/bwd_r4b/AMENDMENT2_R4B.md`.
- **Floor and self-test receipts:** `host-a2-floor/`, `selftest-ref-at-wave-P2/`, `selftest-worker{,-r2,-r3,-r4}/`.
- **G6:** `G6_{r4a,prod,candidate}_R4B.{json,xml}`, `G6_*_RUN_R4B.json`, `G6_PER_ID_R4B.json`.
- **Registration:** `REGISTRATION_R4B.md` and `REGISTRATION_R4B.sha256`.
- **Staging:** `STAGED_PINS_R4B.{json,sha256}`, `STAGED_LINES_R4B.md`, `STAGE_VOICE_MTIMES_R4B.json`, 17 `dry_*_R4B.json`.
- **Census and ledger:** `CENSUS_END_R3_R4B.sha256`, `LEDGER_R4B.md`, `FILES_WRITTEN_R4B.txt`.
- **Logs:** `logs/bwd_r4b_{host_a2_floor, selftest_refwave_*, selftest_worker*, g6_*, cotangent_census, register, stage_prepare, dry_runs}.log` and 17 `logs/bwd_r4b_dry_*_R4B.log`.
- **Scratch left in /tmp:** `/tmp/bwd_r4b_g6/`, `/tmp/bwd_r4b_selftest{,2,3,4}/`, 17 `/tmp/bwd_r4b_lease_*`, `/tmp/bwd_r4b_host/`, `/tmp/bwd_r4b_probe/`. Nothing went to DELETE_ME.

_Filed by the lead from the seat's final message; the seat's own REPORT_R4B.md write was refused by a hook._
