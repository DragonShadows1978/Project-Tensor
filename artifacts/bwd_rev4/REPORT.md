Filed by the lead from the reviewer's final message (the seat's write was refused). Text verbatim. Lead note: findings.json (final, sha256 08d56016) has 3 MAJOR, 6 MINOR, 5 NIT; the verdict line below says 4 MAJOR and 4 NIT, a miscount in the message. findings.json is authoritative.

# BWD-REV4: blind cross-model review of BWD-R4a (541faa2d)

Reviewer: Claude Opus 5.5 (the author was Codex). CPU only: no build, no git writes, nothing deleted. My tools refused the write of `REPORT.md`, so its full content is this message, as the order asks.

## Verdict

**No BLOCKER. 4 MAJOR, 6 MINOR, 4 NIT.**

- **The affine VJP is correct.**
  - My NumPy fp64 copy of the shipped rule matches central differences to 3e-10.
  - The GPU receipts show G-AFF passing 20 of 20 cells and the frozen dX bitwise equal on all 20.
  - r4m03, r4m04 and r2m03 were all killed.
- **The mirror is faithful to `model.py:878-939`.**
  - Both the 8-segment and the 31-segment mirrors compute the same expression graph as Voice's own `decoder` and `generator`.
  - Their fp64 outputs are bitwise equal to Voice's, all 313 names are reached, and r4m02 is detected.
- **Every count the lead reported reproduces from the receipts.**
- **Amendment 1 hid no defect.**
  - A3's diagnosis holds. The 90 byte-identical tensors are exactly the per-iteration parameters of resblocks 3 to 5, as v1 predicted before I read any receipt.
  - A2's bar makes G2's parameter probes almost powerless (F01, MAJOR).
- **The other two MAJORs block R4b and training, not R4a's numbers.**
  - F03: checkpoint replays its inputs with no revision check.
  - F04: the 12 "trainable" norm scales and biases are fixed ones and zeros, shared across 58 decoder sites and 12 predictor sites.

**findings.json hashes:**
- **v1**, sealed before any receipt was read: sha256 `70f732cf778dcdd16ecf3f96a698d03896834b36538804052222afaa14d1e821`. The date line, written by the same command, is `Thu Nov 19 04:01:10 AM EST 2026`; mtime 04:01:10.133. Files: `findings_v1.json`, `findings_v1.seal`.
- **final**: sha256 `08d5601664149a5204ca147a1b7107a90c08432efe6ade249f80b19b48509518`, 04:05:00 EST. File: `findings_final.seal`.

## Findings

Full arguments, lines and post-v1 evidence are in `findings.json`.

- **BWDR4R-F01 (MAJOR)**, `scripts/bwd_r4_gate.py:474-498`, probes at 52-59 and 372-386. A2 makes G2's parameter probes nearly vacuous, and G1 already implies them.
  - **Why the bar is loose.** A2's bar is 1e-3·‖g_ref‖·‖v‖. For a single coordinate of a large tensor, that bar is a large multiple of the probed value.
  - **Blind check, my own seed.** I took the bar divided by |g_i| over every entry of each probed tensor. The median is 2.07 for the ups.1 transposed-conv weight, 1.59 for encode.norm1.fc, 1.31 for resblocks.0 convs1.0 and 0.50 for noise_convs.1.
  - **Registered probes.** The bar is at least the candidate's projection on 9 of the 18 parameter probes, so a zero entry would pass. It is at least twice the projection on 7, so a sign flip would pass. The largest ratio is 18.57.
  - **Logical redundancy.** |cand_i − fd_i| ≤ ‖cand − ref‖ + |ref_i − fd_i|. G1 plus the oracle's own finite-difference check therefore implies A2, so G2 adds no independent evidence for parameters.
  - **No defect hidden.** The candidate's worst per-probe error is 1.06e-4 of its own projection. r3's misses were 1.4e-5 to 6.6e-5 relative and did not change with step size (the spread across steps is at most 1.5% of the error).
  - **Fix.** Use directions where the bound is tight: v = g_ref/‖g_ref‖, the largest-|g| entries, or dense random directions with a √N-scaled bar. Alternatively, use a per-probe bar relative to |fd| at G1's class. Any new bar must be registered before fresh data.
- **F02 (MINOR)**, `bwd_r4_gpu.py:288-289, 308-311, 372-394`; `bwd_r4_worker_r2.py:119-159`. The 31-segment schedule behind the 10 s figure is gated only by A3.
  - G0, G1 and G4 (session, fresh-a and fresh-b) all ran the 8-segment schedule.
  - My trace shows the 31-segment structure is identical, but its fp32 forward bits and its determinism are not gated.
  - Fix: run G0, G1 and G4 with fine=True, and pin the mirror to model.py.
- **F03 (MAJOR)**, `tensor_cuda/src/bindings.cpp:131-183` (line 155), compared with `autograd.cpp:142, 158-166`. Checkpoint replays whatever its saved inputs hold at backward time.
  - `attach_vjp` records each input's storage revision and refuses a changed input; `checkpoint_py` does not. An input mutated in place, through `mark_modified`, or by an optimizer step is replayed silently with the new values.
  - R4a's harness never mutates a checkpoint input, so its results stand.
  - r4m01 does not test this case: it perturbs the activation inside the replay function, not a saved input.
  - DESIGN §3 and §6 asked for exactly this guard.
  - Fix: record revisions in the checkpoint and raise before replay, with a test and a mutant. This is a core change and needs its own order.
- **F04 (MAJOR)**, Voice `model.py:552-566, 401-417`. The 12 instance-norm scale and bias "parameters" are ONNX identity constants.
  - All 6 scales are exactly ones and all 6 biases exactly zeros.
  - They serve 58 decoder sites, with each shared by 1, 1, 4, 4, 24 or 24 sites.
  - `kmodel.predictor.F0.0.norm1.norm.weight`, `kmodel.predictor.F0.1.norm2.norm.weight` and their biases are also the leaves of 12 predictor sites in F0.0 to F0.2 and N.0 to N.2, and `P()` returns one shared tensor per name.
  - Training or ranking them ties unrelated sites and moves the predictor.
  - Because the scale is the identity, the chain gates cannot see a dX that ignores the scale (difference 0.0); only G-AFF's fixtures can.
  - Fix: R4b freezes them or unties them per site, and excludes them from ranking.
- **F05 (MINOR)**, `tc_signal/tests/test_bwd_r4_norm_affine.py:21-37`. The new GPU test uses T=1, which makes x̂ zero.
  - A zero dscale, a batch-0-only dscale, r2m03 and a zero dX all pass it; only r4m03 and dbias errors fail it.
  - Fix: add a case with T≥3, N=2 and a non-identity scale, checked against an fp64 NumPy reference.
- **F06 (NIT)**, `tc_signal/tests/test_bwd_r2_paths.py:70-74`. The changed assertion expects a BufferError mentioning CUDA.
  - That error fires on both the tracked and the untracked path, so the test no longer checks routing.
  - It still catches a reinstated refusal, which is weakly meaningful.
  - Its name is stale and it duplicates the new host test. Fix: rename it and add the `out=` routing witness.
- **F07 (MINOR)**, `bwd_r4_gpu.py:106-134, 351-394`. A3 is justified but gives up ulp-level sensitivity to a stale replay.
  - The fan-in explanation is confirmed by the exact 90-tensor set. r4m01's perturbation is ±3/32, which is gross.
  - A replay that is wrong by less than about 1e-3 relative would pass.
  - Fix: add a bitwise replay-identity probe for each segment.
- **F08 (MINOR)**, `bwd_r4_gpu.py:23-65, 292-307, 427-453`. Non-pool memory is sampled device-wide at only 4 synchronized points, so transient allocations between them can be missed.
  - Reserved plus non-pool is the right footprint to compare with GRAPA.
  - Non-pool bytes: 1,065,615,360 at load, 1,315,176,448 after forward, 1,399,062,528 after backward.
  - Fix: poll per-process NVML. R4b re-measures with the tail VJP and with GRAPA resident.
- **F09 (MINOR)**, `bwd_r4_gate.py:326-364`. 29 of the 313 compared tensors are analytically zero and pass only through the small-norm absolute floor.
  - They are the conv1/convs1 biases, each feeding an instance norm. Their fp64 norms are at most 1.56e-14, and the candidate's L2 error is at most 1.71e-6 against a floor of 2.41e-4.
  - The R4_G1 log line reports `worst_parameter` as one of these rows, resblocks.3.convs1.0.bias, at relative L2 785,817,802.
  - The meaningful worst is encode.norm1.norm.bias at 4.93e-5, which is 0.049 of its bar.
  - Fix: report floor rows separately, and report the worst error-to-bound ratio.
- **F10 (NIT)**, `TOLERANCES_R4.json`. G1's bars suit an fp32 chain this deep, but they cannot see eps-scale or identity-scale VJP defects. That coverage rests on G-AFF, and the report should say so.
- **F11 (NIT)**, `tensor_cuda/src/autograd.cpp:172`. `attach_vjp` silently accepts None for an active slot, which is exactly what r4m04 does. Fix: an optional strict or debug mode for training.
- **F12 (NIT)**, `bwd_r4_gpu.py:90-97`. `rebuild_inputs` does not look inside tuples. This is harmless today. Fix: look inside tuples, or assert that every tensor was collected.
- **F13 (MINOR, post-v1)**, `bwd_r4_gpu.py:292-299`. A4's allocator trim released nothing in the 10 s run.
  - Free bytes (10,129,113,088) and reserved pool bytes (1,006,632,960) are identical after forward and after the trim, with 660,132,452 bytes live.
  - So the drop from 4.32 GB to 3.61 GB comes from the 31-segment schedule. R4b must not count on the trim.
- **F14 (NIT, post-v1)**, G6 in REPORT_R2. The new GPU test IDs were only ever skipped in the CPU suite. The same operands did pass on GPU inside G-AFF's structural cases. Fix: run tc_signal's GPU tests once on a lead line.

## Answers, with what was run or read

### 1. The affine VJP

I read every hunk of `git diff baea226b 541faa2d -- tc_signal`.

**Are dscale = Σ dy·x̂ and dbias = Σ dy right?** Yes.
- `checks/rev4_affine_vjp.py` is an op-by-op copy of the rule. Its worst scaled error against fp64 central differences is 2.9e-10 for x, 1.2e-10 for scale and 1.1e-10 for bias.
- x̂ replays the forward's shifted two-pass statistics. The fp32 replay is within 5.3e-7 of the fp64 x̂, including at offsets of 1e4.

**Is the reduction order fixed, with no atomics?** Yes.
- The time sum runs `reduce_lastdim_kernel`, which uses a fixed tree. The batch sum runs `reduce_kernel`, which is serial per output (`kernels.cu:1234-1301`).
- Neither uses atomics.

**Is the frozen-affine dX path byte-unchanged?** Yes.
- The diff keeps the same operations in the same order.
- G-AFF's `frozen_dx_bitwise` is true on 20 of 20 cells. The gated `_norms.py` (`641ff8c7…`) equals `git show 541faa2d`.

**Are the refusals intact?** Yes. The `out=` (ValueError), foreign-operand (TypeError) and missing-`attach_vjp` (RuntimeError) refusals are intact in `__init__.py:75-87`.

**Does the changed BWD-R2 assertion test something meaningful?** Only weakly; see F06.

### 2. The harness's fidelity to Voice's model

**Does the mirror follow the generator exactly?** Yes. Line by line against model.py:

| model.py | Mirror |
|---|---|
| 878-886, decoder | Voice's own `decoder()` in the 8-segment schedule; an inline copy of 880-885 in the 31-segment schedule |
| 908, LeakyRelu | lrelu0, then lrelu1 |
| 909, noise conv | On the constant spectrum |
| 910, noise_res | `resblock1` |
| 911, convT | ups[i] |
| 912-913, reflection pad | (1,0), at i=1 only |
| 914, Add_3 / Add_6 | `x + xs` |
| 916-918, resblocks | `((r0 + r1) + r2)` |
| 919, Div_1 / Div_2 | `/ C(div)` |

`checks/rev4_mirror_trace.py` runs Voice's real methods and the harness's real `mirror()` on a NumPy fp64 backend that tags every value with a shared expression node.
- Both schedules (8 and 31 checkpoints) give the same expression graph as Voice, with fp64 bitwise equality.
- All 313 names are reached, none missing and none extra.
- The r4m02 control loses 32 names, with a maximum difference of 10.48.

**Are the constants really constants?** Yes. `har`, the spectrum, F0, N and asr are constant leaves without grad.

**Is every on-path parameter a real leaf, with no stray per-use cast?** Yes. The harness asserts that all 313 are fresh fp32 leaves, `f32()` is the identity in fp32 residency, and the trace finds no derived copies.

**Would a harness that skipped or reordered a block still pass G0 and G1?**
- A skipped block fails both G0 and G1; r4m02 shows this, with 32 missing gradients.
- A reorder of commutative adds fails G0 on bits but passes G1, which is harmless.
- In the 31-segment schedule, fp32 bit-parity is not gated (F02).

### 3. The fp64 comparison

**Same upstream, same boundary, same crop?** Yes.
- The upstream is drawn from a PCG64(81073) standard normal, divided by √numel and stored as fp32 (upcast for the oracle). See `gate.py:285-304` and `gpu.py:281`.
- The boundary is the oracle's `istft_tail`, intercepted to return Div_2 (`gate.py:269-282`).
- One `case` object feeds both sides (`worker_r2.py:195-200`).

**Are all 313 names compared, with nothing dropped?** Yes. I recomputed 314 rows (313 plus style) with 0 failures and no error rows. The key sets must be equal, and a None or constant gradient is a FAIL.

**Is the four-name tail-only list right?** Yes, by check 2: `onnx::Conv_8064` (conv_post weight), `conv_post.bias`, `stft.weight_backward_real` and `stft.weight_backward_imag`.

**Are the G1 bars sound for an fp32 deep chain?** Yes, with the caveats in F09 and F10.
- The style error is 1.03e-5 against a bar of 1e-4.
- The worst parameter on a relative bar is 4.93e-5 against 1e-3.

### 4. The checkpointing

**Are the 31 segments right?** Yes: 5 decoder stages plus 2 upsampling paths plus 8 blocks of 3 iterations, which is 31. The trace counted 31 calls with the graph unchanged.

**Can a replay see stale or mutated inputs?** Not in R4a's harness. The engine does not prevent it (F03).

**Is `empty_cache()` safe at that point?** Yes. It runs after both libraries synchronize, and the trim only frees unused memory. It freed nothing in the 10 s run (F13).

**Is the "byte-identical within a mode, chain-class across modes" ruling sound?** Yes. The predicted 90-tensor set was confirmed, and G4 found 0 mismatches over 314 tensors in all three comparisons. Within-mode identity was only shown for the 8-segment schedule (F02).

### 5. Amendment 1

**Was re-registering A2 and A3 justified by the r3 evidence?** Yes.
- r3's finite-difference misses did not change with step size and sat at G1's precision class.
- The fan-in explanation is confirmed exactly: 72 named parameters plus the 18 weights `onnx::Conv_7942` to `8061`, which the ONNX graph maps one to one onto resblocks 3-5 convs1/convs2.

**Did it hide a defect?** No.
- A2's form strips G2 of its power (F01).
- A3 gives up ulp-level sensitivity to a stale replay (F07).

**Were the new bars and probes registered before the data?**
- Before r4's data: the probes at 08:21:17Z, the seal at 08:22:12Z, and the first r4 GPU line at 08:38:10Z.
- After r3's data, which the amendment intends.
- The probes are fresh: no parameter overlap with r2, and the maximum |cos| between old and new style directions is 0.20.

### 6. The 10 s measurement

**Does the reserved-plus-non-pool basis measure the footprint that competes with GRAPA?** Yes, with the sampling caveat in F08.
- 3,613,655,040 bytes against a bar of 4,000,000,000.
- Live basis 2,906,964,584 bytes; the plain arm 10,190,323,712 bytes.
- Forward 2.21 s, backward 0.896 s.

**Is the 10 s input built by a sound rule?** Yes. It repeats the reference in phase-aligned tiles: 212 to 400 frames, 424 to 800 F0/N frames, and 127,200 to 240,000 samples.

### 7. Tests and gates

**Do the new tests reach the code they name?** The host test does. The GPU test is degenerate (F05) and never ran as a pytest ID (F14).

**Do the mutants cover the riskiest edits?** r4m03, r4m04, r2m03, r4m01 (at 1,383.67× its bound) and r4m02 were all killed. Not covered by any registered mutant:
- a dX that ignores the scale (F04);
- dscale's batch reduction, which only G-AFF's N=2 cell would catch;
- the checkpoint revision check (F03).

**Does the harness write nothing in Voice's tree?** Yes.
- The mtime census covers 16 lines and 86 to 169 files read, with 0 changed and 0 write attempts.
- I also searched the Voice folders the harness reads for changes between 01:20 and 03:50. The only change is Voice's own commit 385b0c1 (`tail_vjp.py`, at 01:47:30). No `__pycache__` was created and the git tree is clean.
- My own runs: census over 76 to 80 files, 0 changed, 0 writes.

### 8. Anything else, and what R4b must do

1. Join Voice's tail VJP at Div_2 and gate the full chain to the waveform and the MFCC loss, including the 4 tail-only names.
2. Decide what happens to the 12 norm scales and biases (F04).
3. Add the checkpoint revision guard before any optimizer loop (F03).
4. Run G0, G1 and G4 on the 31-segment schedule (F02).
5. Replace A2's single-coordinate probes (F01).
6. Re-measure 10 s with the tail VJP and with GRAPA resident, without crediting the trim (F13).
7. Exclude the 29 analytically-zero biases from weight ranking (F09).

## Counts recomputed (L44)

Every reading matches the lead's:

| Reading | Recomputed |
|---|---|
| G1 style relative L2 | 1.0322924353585311e-5 |
| G1 compared / failures | 313 / 0 |
| G2 | 26 of 26 (worst error at 0.0206 of its bar) |
| G3 byte-identical / failures | 90 of 314 / 0 |
| r4m01 error-to-bound | 1,383.6713 |
| G5 reserved + non-pool | 3,613,655,040 |
| G5 live + non-pool | 2,906,964,584 |

## Evidence and files written

Everything is under `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_bwd/WorkTree/tensor_fork_bwd_wt_rev4/artifacts/bwd_rev4/`. All commands ran with `CUDA_VISIBLE_DEVICES=` empty, single-threaded BLAS, `nice -n 19`, and `timeout` of at most 1200 s.

- `findings_v1.json`, `findings_v1.seal`, `findings.json`, `findings_final.seal`.
- `checks/rev4_affine_vjp.py` with `.out`: rc 0.
- `checks/rev4_reference_crop.py` (rc 0 for both runs; RLIMIT_AS 8 GiB, peak RSS 1.58 GB, about 11 s each):
  - blind run: `rev4_reference_crop_seed20261119.json` and `.out`;
  - post-v1 run with the registered probes: `rev4_reference_crop_seed81073_registered.json` and `.out`.
- `checks/rev4_mirror_trace.py` with `.json` and `.out`: rc 0; RLIMIT_AS 10 GiB, peak RSS 1.93 GB, 17 s.
- `checks/rev4_receipts.py` with `.out`: rc 0.
- Scratch left in place: `/tmp/bwd_rev4_work/`, which holds the Voice census JSONs.

## Deviations

- **Check 1:** the "dscale from batch 0" variant was first coded as literal zeros. I fixed it to use the replayed x̂ and reran it (1 repair). The result did not change.
- **Check 2:** it read the probe manifest as a list. I made it read the dict after v1, before its post-v1 run (1 repair).
- **Disjointness one-liner:** it raised a NumPy shape ValueError. I fixed it with `ravel` (1 repair).
- **v1 line cite:** I corrected model.py line 556 to 552 before sealing v1.
- **Report write:** the write of `REPORT.md` was refused, so its content is this message.

## Prior art

- **Affine normalization VJP:** Ulyanov, Vedaldi & Lempitsky (2016), instance normalization; Ioffe & Szegedy (2015), dγ = Σ dy·x̂ and dβ = Σ dy. Review only.
- **Shifted two-pass variance:** Welford (1962); Chan, Golub & LeVeque (1983). Review only.
- **Centered differences:** Higham (2002). Mine: the T=1 distinguishability table.
- **The F01 projection bound:** Cauchy (1821) and Schwarz (1888); √N concentration of coordinate projections, e.g. Vershynin (2018). Mine: the census of bar against probed value. No prior art known to me for that exact census.
- **Checkpointing:** Chen et al. (2016); Griewank & Walther (2000), Revolve.
- **The F07 fan-in prediction:** floating-point non-associativity, Goldberg (1991) and Higham (2002). Mine: applying it to predict the exact byte-identical set.
- **The mirror census:** hash-consing and value numbering, Ershov (1958), Cocke (1970), Filliâtre & Conchon (2006). Mine: using it to compare the mirror's graph with the source's.
- **Allocator:** the CUDA stream-ordered allocator and `cudaMemPoolTrimTo` (NVIDIA docs). Review only.
- **The model:** Kokoro / StyleTTS 2 (Li et al., 2023), iSTFTNet (Kaneko et al., 2022), HiFi-GAN (Kong et al., 2020). Review only.

All of these are unverified, because there is no network here; the lead should check them. Search terms: "Ulyanov instance normalization 2016", "batch normalization backward gamma beta", "Chan Golub LeVeque variance", "Training Deep Nets with Sublinear Memory Cost", "Revolve Griewank Walther", "Filliatre Conchon hash-consing 2006", "Goldberg floating-point 1991", "cudaMemPoolTrimTo".
