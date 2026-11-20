# BWD-R3 convolution backward report

**Seat gates pass. GPU qualification and blind review are pending.** Both convolution VJPs and the four no-VJP refusals are implemented. The CPU comparison passed against accepted BWD-R2 at 9b76e58a and MAINCHK3 production at 3138cf54. All 31 GPU launchers returned 78 with would_start=true in dry-run mode and opened no lease. Device numerics, runtime, TF32 dispatch, determinism and the eight native mutant kills remain unverified.

## Diff summary

Seven tracked files changed, with 153 additions and 10 removals. Two new native files add 320 lines. The change adds the tracked convolution branches, private backward plan/run/config bindings, a sidecar CUDA translation unit, documentation, four test modules, copied fp64 support, five drivers, and gate evidence. G10 verified the forward kernel sources and tensor_cuda/ against the dispatch tree. The untracked forward branch bodies are preserved. [Full diff](DIFF.patch) · [Scope receipt](G10_SCOPE.json)

Voice supplies tensor bias at model.py lines 738 and 742. Each fixture with an omitted stored bias gets a zero tensor for the tensor-bias checks. The 12 BP1 small variants also exercise b=None. The raw BP1 and BP2 manifests match 3822b6a6… and bcabaf93… respectively. The stored fp64 key agreed with the fp64 oracle on the stored inputs rounded to fp32 for all 156 slot comparisons across the 52 bias-present fixture calls. [Shape and buffer census](OPERAND_CENSUS.json) · [CPU key comparison](../../logs/bwd_r3_geometry_worker.log)

## VJP implementation

Either convolution records when grad mode is enabled and x, w or b is an active TensorCUDA operand. A tracked call checks operands, allocates a fresh TensorCUDA value on their device, calls the same forward, marks the value modified and attaches the BWD-R1 hook. The callback captures x and w, plus geometry and allocation metadata. The hook retains and revision-checks x, w and b. With b=None the frozen TensorCUDA alias of x reserves slot 2 without a device allocation. The VJP allocates only gradients for active slots.

Let q be input time, t the upstream output time, k a tap, and s, d, pb the stride, dilation and left pad. Group-local input and output channel indices are c and o.

| Geometry class | dX derivation |
|---|---|
| Conv1d K1 with stride 1 and zero padding | Forward q=t, so dX[n,c,q] is the in-group dot of W[o,c,0] with G[n,o,q]. |
| Conv1d dense or grouped, stride 1, including dilations 1, 3 and 5 | Forward q=t−pb+k·d. Solve t=q+pb−k·d and gather W[o,c,k]·G[n,o,t] for 0≤t<T_out. This covers asymmetric end pads through the resolved extent. |
| Conv1d strided, including source stride 5 and the 22→256 K12 noise projection | Solve t=(q+pb−k·d)/s. Require a nonnegative numerator divisible by s and t<T_out. The stride-6 noise case uses t=(q+3−k)/6. |
| ConvTranspose1d grouped and BP1 boundary cases | Each forward input q reaches u=q·s−pb+k·d. Gather W[c,o,k]·G[n,o,u] for 0≤u<T_out. T_out includes both end pad and output padding. |
| ConvTranspose1d depthwise ×2 | Each channel uses its own taps at u=2q−1+k. With BP2's 40 input positions, q=39 and k=2 reach output bin 79. |
| ConvTranspose1d dense 10/20 and 6/12 | The mapped bins are 10q−5+k and 6q−3+k respectively, each with the resolved extent. All in-group output channels contribute. |
| ConvTranspose1d stride-5 real and imaginary tail | The mapped bins are 5q+k, with K20 and the full resolved tail extent. |

Conv1d dW packs X[n,c,t·s−pb+k·d] and multiplies by G. ConvTranspose1d dW packs G[n,o,q·s−pb+k·d] and multiplies by X. The fp32 weight rectangles merge in ascending (batch,time) order. Each GEMM lane reuses a 1,024-position pack and one group rectangle, each capped at 32 MiB. A fixed CUDA tree handles small groups and shapes outside the pack limits. dB sums all N and T, including bias-only bins, using fixed lane traversal and a fixed tree. The separate backward handle uses the legacy stream, 32 MiB private workspace, atomics disabled, pedantic math and CUBLAS_COMPUTE_32F_PEDANTIC. The call and configuration check read the same compute and algorithm members. These are source properties. [Native code](../../tc_signal/src/conv_backward.cu) · [Bindings](../../tc_signal/src/bindings.cpp) · [Tracked Python](../../tc_signal/tc_signal/__init__.py)

## Gates

| Gate | Result and evidence |
|---|---|
| L01 | Version checks recorded. Initial forge-gpu-run dry-run exit 78 and would_start=true. [Versions](../../logs/bwd_r3_l01_versions.log) · [Admission](../../logs/bwd_r3_l01_dry.log) |
| G1 | **PASS.** CUDA 12.6, SM89, nice 19 and -j4. Signal module SHA-256 29595097d7af880d76e13593e4e5479a5d705df735131327b2b33a5b2680def3. Core module SHA-256 4a0921493f9d379bd67239ebacca343c4ba1024bdfc667d22e87290c10f5325c. [Build](../../logs/bwd_r3_g1_build.log) · [Modules](G1_MODULES.json) · [Sources](G1_SOURCES.json) |
| G2 host | **PASS.** The tracked smoke and all 28 host cases passed. Tests cover x, w and b activation, refusal before foreign export, no_grad planner routing, and F10. [Smoke](../../logs/bwd_r3_g2_smoke_a2.log) · [Host](../../logs/bwd_r3_g2_host_a2.log) |
| G2 GPU | **Staged.** 16 hook, revision and DLPack tests. Seven forward guard lines compare production and candidate uint32 output digests and producer/allocator call traces for the fixture census and special bit patterns. [Dry receipts](GPU_DRY_RUNS.json) |
| G3 | **Staged.** All 16 BP1 and 36 BP2 cells with bias tensors, 12 biasless small variants, and three ordered-merge or cancellation/unit/zero checks. [Census](OPERAND_CENSUS.json) · [Dry receipts](GPU_DRY_RUNS.json) |
| G4-key | **PASS, answer key only.** Four cells × three slots. Each slot used eight seeded directions plus two boundary coordinates at h/2, h and 2h. [FD key](../../logs/bwd_r3_g4_key_a2.log) |
| G4-candidate | **Staged.** Candidate projection on the same four cells plus two exhaustive tiny cells. [Dry receipts](GPU_DRY_RUNS.json) |
| G5 | **Staged.** 10 in-process repeats and three fresh processes each for dense 10/20, depthwise ×2 and the real stride-5 tail. Each receipt includes dX/dW/dB byte digests, cuBLAS version and SM count. [Dry receipts](GPU_DRY_RUNS.json) |
| G6 | **Host sensitivity PASS, GPU staged.** Aligned dW fp64 surrogate rounds TF32 inputs far enough to differ by 6.36538904 times the sealed per-entry bound. The unmasked device line checks the true handle and strictly compares dW. [Host](../../logs/bwd_r3_g6_host_a2.log) · [Dry receipts](GPU_DRY_RUNS.json) |
| G7 | **Host PASS.** r3m08 reached the real planner and failed in the mutant. Eight isolated CUDA variants built and are staged. [Build](../../logs/bwd_r3_g7_build.log) · [Host](G7_HOST.json) · [Mutant pins](G7_MUTANTS.json) · [Dry receipts](GPU_DRY_RUNS.json) |
| G8 | **PASS vs both baselines.** MAINCHK3 phase folding, empty CUDA_VISIBLE_DEVICES, 12 GiB RLIMIT_AS and no optional build or fixture overrides. [Driver](../../logs/bwd_r3_g8_driver.log) · [Per-ID](g8_PER_ID.json) |
| G9 | **PASS.** Fresh tree with only candidate packages on PYTHONPATH. 29 passed, 99 skipped for missing fixture roots and 23 skipped for CUDA disabled. Process creation was kernel-blocked for the successful pytest run. [Final plain log](../../logs/bwd_r3_g9_plain_final.log) |
| G10 | **PASS.** Seven authorized tracked edit sites, 188 protected files, empty forbidden-string search and no staged Git changes. No file deleted, GPU lease taken or CUDA workload run. [Scope](../../logs/bwd_r3_g10_scope.log) · [Scope receipt](G10_SCOPE.json) |

The tolerance file was sealed before any comparison. SHA-256 8f6c52e38d8a653aaaf9c1b51453fa8031b1f50f59444a3bf436262fa4845f2b. Each entry uses abs(error) ≤ max(3e−6, 1.5e−5·S) + 1e−4·abs(reference), with S the fp64 absolute term mass. Exact structural zeros and raw uint32 comparisons have separate assertions. The FD bound is 1e−7 + 1e−5·abs(projected gradient). [Tolerance seal](TOLERANCES_R3.sha256) · [Registration](REGISTRATION.md) · [Registration seal](REGISTRATION.sha256)

## G7 mutant table

| Mutant | Reached edit and killing case | Result |
|---|---|---|
| r3m01_conv_dw_ignores_dilation | Remove dilation from the Conv1d pack index. BP2 conv1d_14 dW. | GPU staged |
| r3m02_conv_dx_reverses_tap | Reverse the Conv1d dX weight tap index. BP1 conv1d_small1 dX. | GPU staged |
| r3m03_conv_omits_bias_reduction | Replace dB with the first time bin. BP1 conv1d_small1 dB. | GPU staged |
| r3m04_convT_q_without_stride_pad | Drop stride and pad in the ConvTranspose1d dX gather. BP1 conv_transpose1d_small3 dX. | GPU staged |
| r3m05_convT_swaps_ingroup_weight_axes | Swap the in-group weight axes. BP1 conv_transpose1d_small3 dX. | GPU staged |
| r3m06_convT_trims_output_padding_bins | Drop the last resolved ConvTranspose1d output-padding bins in dX. BP2 depthwise ×2. | GPU staged |
| r3m07_tf32_at_backward_gemm_call | Set FAST_TF32 at the actual GemmEx call. Aligned dW must fail the numeric bound after the call succeeds. | GPU staged |
| r3m08_tracked_conv_allows_out | Pass None to the tracked out guard. The real planner is reached and raises BufferError: __dlpack_device__: TensorCUDA exports CUDA tensors only; this tensor is on cpu. | Host killed, exit 1 |
| r3m09_dw_tiles_merge_out_of_order | Reverse the time tiles. Three tiles contribute 2²⁴, 1 and −2²⁴. The ascending fp32 merge gives 0; the reversed merge gives 1. | GPU staged |

Native mutants print witnesses after the edited launch or call. A numeric bound failure or exact merge-witness failure is required. r3m07 additionally requires the strict handle configuration report. [Mutant table receipt](G7_MUTANT_TABLE.json) · [Host mutant trace](../../logs/bwd_r3_g7_host_r3m08_tracked_conv_allows_out.log)

## Memory and runtime

These are byte counts derived from actual fixture shapes and the candidate scratch API for an all-slot VJP. Workspace is persistent per device and shown where the case uses GEMM. The staged measurement includes all-slot, x-only and w-only variants. The probe records the synchronized native and end-to-end backward times and the CUDA pool peak increment above the live forward baseline, with private workspace accounted for separately.

| Real geometry | Gradient bytes | Pack bytes | Partial bytes | Active workspace bytes | Peak and runtime |
|---|---:|---:|---:|---:|---|
| BP1 dense Conv1d | 6,756,000 | 1,307,616 | 6,316,032 | 33,554,432 | GPU staged |
| BP2 1,090-input Conv1d | 13,572,416 | 523,200 | 13,393,920 | 33,554,432 | GPU staged |
| BP2 stride-6 K12 noise Conv1d | 693,848 | 844,800 | 270,336 | 33,554,432 | GPU staged |
| BP2 Conv1d K11, T=4,801 | 3,179,520 | 5,767,168 | 720,896 | 33,554,432 | GPU staged |
| BP1 dense ConvTranspose1d 10/20 | 11,355,136 | 8,683,520 | 10,485,760 | 33,554,432 | GPU staged |
| BP2 dense ConvTranspose1d 6/12 | 2,392,576 | 4,915,200 | 1,572,864 | 33,554,432 | GPU staged |
| BP2 depthwise ConvTranspose1d ×2 | 191,840 | 0 | 0 | 0 | GPU staged |
| BP1 real stride-5 tail | 1,120,288 | 81,920 | 880 | 33,554,432 | GPU staged |

## G8 per-ID summary

| Arm | Passed | Skipped | Failed | Setup errors | Collection errors |
|---|---:|---:|---:|---:|---:|
| Parent 9b76e58a | 1,298 | 488 | 2,171 | 15 | 1 |
| Production 3138cf54 | 1,222 | 339 | 2,171 | 15 | 1 |
| Candidate | 1,327 | 610 | 2,171 | 15 | 1 |

Against the parent, only 151 R3 IDs were added, 29 pass and 122 skip. Against production, only 376 IDs were added, including R1 52 pass / 88 skip, R2 24 pass / 61 skip, and R3 29 pass / 122 skip. Existing failures and other existing outcomes match exactly. **F10 list is empty.** [Per-ID matrix](g8_PER_ID.json) · [Parent log](../../logs/bwd_r3_g8_parent.log) · [Production log](../../logs/bwd_r3_g8_prod.log) · [Candidate log](../../logs/bwd_r3_g8_candidate.log)

## Deviations

- One edit invocation was rejected on an LSTM context mismatch and one on a JavaScript template syntax error. Neither wrote files. Both edits were resubmitted with corrected inputs.
- The first host driver stopped at NameError: name 'PYTHONPATH' is not defined before starting pytest. The harness quoting was repaired and the fresh a2 run passed. The first stage inventory omitted the separately collected configuration node. It stopped at an AssertionError before running any dry lines. The node was added and collection was rechecked. [Host attempt](../../logs/bwd_r3_host_driver.log) · [Stage attempt](../../logs/bwd_r3_prepare_worker.log)
- The final plain recheck was initially invoked with g9, a label the harness does not recognize. It ended with KeyError: 'g9'. The corrected g9_plain final run passed. [Mislabelled log](../../logs/bwd_r3_g9_final.log) · [Final plain log](../../logs/bwd_r3_g9_plain_final.log)
- The final plain run and stage collection used the final test sources. The intervening edits after G8 were confined to GPU-skipped test bodies and their imports. Test IDs and final plain outcomes match the G8 candidate R3 subset. A one-element dB is checked for a finite nonzero scalar because variation is undefined for a scalar.

## Prior art

- Dumoulin and Visin (2016) supply convolution arithmetic and the adjoint mapping. Chellapilla, Puri and Simard (2006) supply im2col plus GEMM. R3 chooses the common ONNX group mapping, batch/time tiles and single live group rectangle.
- Harris (2007) supplies the CUDA shared-memory reduction tree. R3 fixes the bias traversal, tile ownership, dispatch threshold and merge order.
- NVIDIA cuBLAS 12.6 (2024) supplies pedantic compute, stream/workspace reproducibility and the TF32-override contract. BWD-R1 (2026) supplies the checked-call member pattern. R3 owns the sidecar handle and the aligned dW canary.
- PyTorch convolution backward and saved-input autograd (Paszke et al., 2019) and BWD-R2 (2026) supply VJP and revision-check precedents. The biasless fixed slot and the sidecar integration are R3 choices.
- Fornberg (1988) supplies centered finite differences, DeMillo, Lipton and Sayward (1978) supply mutation testing, and MAINCHK3, System-Claude GW3 and BWD-R2 (2026) supply phase folding, lease dry runs and process auditing. R3 binds them to these fixtures, mutant edits and candidate-gradient projections.
- Unverified — lead to check Dumoulin Visin convolution arithmetic 2016, Chellapilla document processing im2col, Harris CUDA parallel reduction, cuBLAS 12.6 GemmEx pedantic TF32 reproducibility, PyTorch convolution_backward, Fornberg centered finite differences, and DeMillo Lipton Sayward mutation testing.

## Files written

[Complete file inventory](FILES_WRITTEN.txt) lists source edits, tests, drivers, gate artifacts, logs, binaries, and retained scratch. All receipts under artifacts/bwd_r3 and logs/bwd_r3_* are new except the read-only dispatch transcript. No fixture data was added to the worktree.

## BWD-R3 GPU lines

Run from the R3 worktree root in the order below. Each line verifies package and test SHA-256s before lease admission. It opens a unique log and receipt. The launcher sets CUDA_MODULE_LOADING=EAGER and NVIDIA_TF32_OVERRIDE=0, except g6-canary and r3m07, where the override is absent. The lease request is 500–540 s with a 30 s margin. The worker timeout is 475 s with 5 s kill grace. Durations below are estimates from fixture sizes and operation counts. Runtime is recorded by the lead.

| Line | Expected s | Cells / workers |
|---|---:|---|
| g2-paths | 30 | 16 pytest |
| g2-forward-bits | 25 | 1 forward |
| g2-forward-bp1-small | 25 | 12 forward |
| g3-bp1-small | 25 | 24 pytest |
| g2-forward-bp1-kokoro | 125 | 4 forward |
| g3-bp1-kokoro | 125 | 4 pytest |
| g2-forward-bp2-1 | 75 | 12 forward |
| g3-bp2-1 | 75 | 12 pytest |
| g2-forward-bp2-2 | 90 | 10 forward |
| g3-bp2-2 | 90 | 10 pytest |
| g2-forward-bp2-3 | 125 | 10 forward |
| g3-bp2-3 | 125 | 10 pytest |
| g2-forward-bp2-transpose | 60 | 4 forward |
| g3-bp2-transpose | 60 | 4 pytest |
| g3-special | 30 | 3 pytest |
| g4-candidate-small | 35 | 4 pytest |
| g4-candidate-real | 45 | 2 pytest |
| g5-dense | 95 | 1 repeat |
| g5-depthwise | 45 | 1 repeat |
| g5-tail | 50 | 1 repeat |
| g6-canary | 30 | 2 pytest |
| r3m01_conv_dw_ignores_dilation | 30 | 1 mutant |
| r3m02_conv_dx_reverses_tap | 30 | 1 mutant |
| r3m03_conv_omits_bias_reduction | 30 | 1 mutant |
| r3m04_convT_q_without_stride_pad | 30 | 1 mutant |
| r3m05_convT_swaps_ingroup_weight_axes | 30 | 1 mutant |
| r3m06_convT_trims_output_padding_bins | 30 | 1 mutant |
| r3m07_tf32_at_backward_gemm_call | 30 | 1 mutant |
| r3m09_dw_tiles_merge_out_of_order | 30 | 1 mutant |
| memory-conv1d | 150 | 12 measure |
| memory-convT | 150 | 12 measure |

```bash
timeout 570s env FORGE_GPU_DRY_RUN=0 /usr/bin/python3 -B -P scripts/bwd_r3_stage.py g2-paths --min 500 --max 540
timeout 570s env FORGE_GPU_DRY_RUN=0 /usr/bin/python3 -B -P scripts/bwd_r3_stage.py g2-forward-bits --min 500 --max 540
timeout 570s env FORGE_GPU_DRY_RUN=0 /usr/bin/python3 -B -P scripts/bwd_r3_stage.py g2-forward-bp1-small --min 500 --max 540
timeout 570s env FORGE_GPU_DRY_RUN=0 /usr/bin/python3 -B -P scripts/bwd_r3_stage.py g3-bp1-small --min 500 --max 540
timeout 570s env FORGE_GPU_DRY_RUN=0 /usr/bin/python3 -B -P scripts/bwd_r3_stage.py g2-forward-bp1-kokoro --min 500 --max 540
timeout 570s env FORGE_GPU_DRY_RUN=0 /usr/bin/python3 -B -P scripts/bwd_r3_stage.py g3-bp1-kokoro --min 500 --max 540
timeout 570s env FORGE_GPU_DRY_RUN=0 /usr/bin/python3 -B -P scripts/bwd_r3_stage.py g2-forward-bp2-1 --min 500 --max 540
timeout 570s env FORGE_GPU_DRY_RUN=0 /usr/bin/python3 -B -P scripts/bwd_r3_stage.py g3-bp2-1 --min 500 --max 540
timeout 570s env FORGE_GPU_DRY_RUN=0 /usr/bin/python3 -B -P scripts/bwd_r3_stage.py g2-forward-bp2-2 --min 500 --max 540
timeout 570s env FORGE_GPU_DRY_RUN=0 /usr/bin/python3 -B -P scripts/bwd_r3_stage.py g3-bp2-2 --min 500 --max 540
timeout 570s env FORGE_GPU_DRY_RUN=0 /usr/bin/python3 -B -P scripts/bwd_r3_stage.py g2-forward-bp2-3 --min 500 --max 540
timeout 570s env FORGE_GPU_DRY_RUN=0 /usr/bin/python3 -B -P scripts/bwd_r3_stage.py g3-bp2-3 --min 500 --max 540
timeout 570s env FORGE_GPU_DRY_RUN=0 /usr/bin/python3 -B -P scripts/bwd_r3_stage.py g2-forward-bp2-transpose --min 500 --max 540
timeout 570s env FORGE_GPU_DRY_RUN=0 /usr/bin/python3 -B -P scripts/bwd_r3_stage.py g3-bp2-transpose --min 500 --max 540
timeout 570s env FORGE_GPU_DRY_RUN=0 /usr/bin/python3 -B -P scripts/bwd_r3_stage.py g3-special --min 500 --max 540
timeout 570s env FORGE_GPU_DRY_RUN=0 /usr/bin/python3 -B -P scripts/bwd_r3_stage.py g4-candidate-small --min 500 --max 540
timeout 570s env FORGE_GPU_DRY_RUN=0 /usr/bin/python3 -B -P scripts/bwd_r3_stage.py g4-candidate-real --min 500 --max 540
timeout 570s env FORGE_GPU_DRY_RUN=0 /usr/bin/python3 -B -P scripts/bwd_r3_stage.py g5-dense --min 500 --max 540
timeout 570s env FORGE_GPU_DRY_RUN=0 /usr/bin/python3 -B -P scripts/bwd_r3_stage.py g5-depthwise --min 500 --max 540
timeout 570s env FORGE_GPU_DRY_RUN=0 /usr/bin/python3 -B -P scripts/bwd_r3_stage.py g5-tail --min 500 --max 540
timeout 570s env FORGE_GPU_DRY_RUN=0 /usr/bin/python3 -B -P scripts/bwd_r3_stage.py g6-canary --min 500 --max 540
timeout 570s env FORGE_GPU_DRY_RUN=0 /usr/bin/python3 -B -P scripts/bwd_r3_stage.py r3m01_conv_dw_ignores_dilation --min 500 --max 540
timeout 570s env FORGE_GPU_DRY_RUN=0 /usr/bin/python3 -B -P scripts/bwd_r3_stage.py r3m02_conv_dx_reverses_tap --min 500 --max 540
timeout 570s env FORGE_GPU_DRY_RUN=0 /usr/bin/python3 -B -P scripts/bwd_r3_stage.py r3m03_conv_omits_bias_reduction --min 500 --max 540
timeout 570s env FORGE_GPU_DRY_RUN=0 /usr/bin/python3 -B -P scripts/bwd_r3_stage.py r3m04_convT_q_without_stride_pad --min 500 --max 540
timeout 570s env FORGE_GPU_DRY_RUN=0 /usr/bin/python3 -B -P scripts/bwd_r3_stage.py r3m05_convT_swaps_ingroup_weight_axes --min 500 --max 540
timeout 570s env FORGE_GPU_DRY_RUN=0 /usr/bin/python3 -B -P scripts/bwd_r3_stage.py r3m06_convT_trims_output_padding_bins --min 500 --max 540
timeout 570s env FORGE_GPU_DRY_RUN=0 /usr/bin/python3 -B -P scripts/bwd_r3_stage.py r3m07_tf32_at_backward_gemm_call --min 500 --max 540
timeout 570s env FORGE_GPU_DRY_RUN=0 /usr/bin/python3 -B -P scripts/bwd_r3_stage.py r3m09_dw_tiles_merge_out_of_order --min 500 --max 540
timeout 570s env FORGE_GPU_DRY_RUN=0 /usr/bin/python3 -B -P scripts/bwd_r3_stage.py memory-conv1d --min 500 --max 540
timeout 570s env FORGE_GPU_DRY_RUN=0 /usr/bin/python3 -B -P scripts/bwd_r3_stage.py memory-convT --min 500 --max 540
```
