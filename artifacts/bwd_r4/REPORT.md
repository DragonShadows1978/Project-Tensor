# BWD-R4a report

**RED on the frozen candidate.** Voice BP1’s fp64 boundary VJP returned all 313 pre-tail gradients. The candidate’s InstanceNorm wrapper refuses to train 12 of those initializers. The host probe exercised the first decoder norm scale and bias and emitted “tc_signal: instance_norm1d: scale and bias must be frozen when recording gradients”. The refusal occurs before checkpoint replay or gradient collection. All 11 GPU commands passed admission checks in dry-run mode. GPU acceptance and blind review remain pending. [Refusal](../../logs/bwd_r4_g1_frozen.log) · [Registration](REGISTRATION.md) · [Ledger](LEDGER.md)

## Harness and product code

The harness builds Voice’s stock KokoroTC at fp32 residency and activates s_d and all 313 pre-tail initializers by ONNX name. Voice defines residency and P/f32 at model.py:36 and :401–424, decoder and resblk1d at :878–886 and :759–771, and conv/convT, AdaIN and Snake at :737–757. The mirror follows the generator’s LeakyReLU (:908), noise conv and residual block (:909–910), transposed conv and reflection pad (:911–913), Add_3/Add_6 (:914), and three resblock1 branches and ordered Add/Div merges (:915–919). It uses Voice’s resblock1 at :773–779 and checkpoints all eight generator residual blocks with explicit activation, style and unique weight inputs. The fixed signed upstream seed enters at Div_2 before :923. G0 uses the original generator’s keep_div2 tap (:937–938) with the same frozen source inputs. The CPU oracle exposes the existing Div_2 value at BP1’s tail dispatch and seeds its existing reverse tape. [Voice model](/mnt/ForgeRealm/Project-Voice/engine/kokoro_tc/model.py:878) · [Harness](../../scripts/bwd_r4_gate.py) · [GPU workers](../../scripts/bwd_r4_gpu.py)

Zero product files changed. The refusal is enforced in the frozen tc_signal wrapper. A source change would leave the pinned acceptance package unaffected, and a rebuild would need a new acceptance pin. The two local Voice control libraries were built from read-only sources with Voice’s published shipped flags for the native G0 control and capture tap. The stock loader census matched all 317 BP1 initializer values. [Norm wrapper](../../tc_signal/tc_signal/_norms.py#L60) · [Control build](../../logs/bwd_r4_voice_build.log) · [Stock metadata](../../logs/bwd_r4_stock.log)

## Gates

| Gate | Result | Evidence |
|---|---|---|
| L01 | Versions recorded. Admission-only probe returned 78 with would_start true. | [Versions](../../logs/bwd_r4_l01_versions.log) · [Build nvcc](../../logs/bwd_r4_nvcc126_version.log) · [Dry run](../../logs/bwd_r4_l01_dry.log) |
| G0 | Native bitwise parity staged. The fp64 archive comparison returned **0.5036856491 relative L2** on the 40-frame crop and **5.192514189e-7** on the full 212 frames. The archive slice includes full-utterance norm and convolution context. The crop missed the 1e-4 bar even in fp64. BP1 reports an observed Div_2 max absolute error and registers 1e-4 relative L2 for the waveform. R4 preregistered that bar for the requested archive cross-check. | [Archive comparison](../../logs/bwd_r4_g0_context.log) · [Shape census](../../logs/bwd_r4_shapes.log) · [Dry admissions](../../logs/bwd_r4_gpu_dry.log) |
| G1 | CPU oracle passed with **313** nonconstant parameter gradients and 128 nonzero style entries, style norm **9.9261668346**. Frozen candidate capability **RED**. **0/313** candidate arrays compared, so worst candidate relative L2 is unavailable. | [Oracle](../../logs/bwd_r4_g1_cpu.log) · [Refusal](../../logs/bwd_r4_g1_frozen.log) · [Per-parameter table](PER_PARAMETER.csv) |
| G2 | CPU fp64 **26/26**, eight style directions and 18 entries across six parameters at h, h/2 and 2h. One halving resolved a LeakyReLU sign crossing. Worst accepted projected error **0.389×** its DESIGN bound. Candidate projections staged and blocked by the frozen refusal. | [CPU log](../../logs/bwd_r4_g2_cpu.log) · [Projections](g2_cpu/FD_PROJECTIONS.json) · [Dry admissions](../../logs/bwd_r4_gpu_dry.log) |
| G3 | Checkpoint versus plain staged. Candidate gradient bytes unmeasured. | [Dry admissions](../../logs/bwd_r4_gpu_dry.log) |
| G4 | Three in-process graphs and two fresh processes staged. Candidate determinism unmeasured. | [Dry admissions](../../logs/bwd_r4_gpu_dry.log) |
| G5 | Both ten-second arms staged. Device peak, wall time and backward time unmeasured. | [Shape census](../../logs/bwd_r4_shapes.log) · [Dry admissions](../../logs/bwd_r4_gpu_dry.log) |
| G6 | Inapplicable. No product change triggered the CPU suite. | [Scope](../../logs/bwd_r4_g7_scope.log) |
| G7 | Scope checks passed. The mtime, inode and size census found zero changes across 169 Voice files read by the harness. Both compiler inputs have matching before/after stamps. Tracked diff from the dispatched HEAD is empty. Zero GPU leases, git writes, deletions and Voice suite runs. The L67 path deviation is below. | [Scope log](../../logs/bwd_r4_g7_scope.log) · [Census and receipts](G7_SCOPE.json) · [Compiler stamps](../../logs/bwd_r4_voice_build.log) |

Tolerance SHA-256 **27052e947b4704ec817ce5b962c94f7018d3143a2d802288fc888dfa002699be**. The G1 bars are 1e-4 relative L2 for style and 1e-3 per parameter. For parameter norms strictly below 1e-6 × the largest parameter-gradient norm, absolute L2 uses that same threshold. Style aggregates the deep fp32 AdaIN chain. Individual weights sum thousands of time positions with different packing and reduction orders. Relative error becomes ill-conditioned for very small reference norms. [Tolerances](TOLERANCES_R4.json) · [Seal](TOLERANCES_R4.sha256)

### Parameter census

The boundary reached 313 of Voice’s 317 names. The largest fp64 parameter-gradient norm was **241.170602686254** for onnx::ConvTranspose_7930. Twelve named scale and bias initializers trainable in BP1 are shared across InstanceNorm sites through ONNX Identity aliases. The four tail-only names are:

- kmodel.decoder.generator.conv_post.bias
- kmodel.decoder.generator.stft.weight_backward_imag
- kmodel.decoder.generator.stft.weight_backward_real
- onnx::Conv_8064

[Name census](g1_cpu/PARAMETERS.json) · [Comparison status and norms](PER_PARAMETER.csv)

### Memory and time

The 10 s fixture repeats captured ASR/F0/N/har along the time axis, takes the required prefix and recomputes the fixed source spectrum with Voice’s helper. It has 400 duration frames, 800 F0/N frames, L=240,000 and F=48,001.

| Arm | DESIGN live bytes | Measured device bytes | Measured wall / backward s | Expected work / lease min / max s |
|---|---:|---:|---:|---:|
| Checkpointed | 2.7–3.3 GB | Unmeasured | Unmeasured | 220 / 440 / 540 |
| Plain | 8.7–9.1 GB | Unmeasured | Unmeasured | 180 / 360 / 540 |

The checkpointed cap is 4.0 decimal GB. The plain arm uses a 9.5 GB admission budget. The staged worker records driver pool high-water marks and samples non-pool device-wide bytes at synchronized phase boundaries. Transient non-pool allocations between samples are outside that accounting. The estimates and dry admission provide no measured device fit or timing. [Design](../bwd_d/DESIGN.public.md) · [Reservations](STAGED_LINES.md)

### Mutants

| Mutant | Registered edit and check | Result |
|---|---|---|
| r4m01_checkpoint_replays_stale_input | Perturb the final block’s replay activation in a nonuniform time pattern. Require one edit, unchanged forward bytes and a G3 or G1 failure. | Staged, kill unverified |
| r4m02_skip_add_dropped | Omit stage-0 Add_3. Require one edit and both the G0 tap and G1 comparisons to fail. | Staged, kill unverified |

The refusal occurs upstream of both edits. The worker requires passing clean comparisons before counting a mutation kill. [Workers](../../scripts/bwd_r4_gpu.py) · [Receipt index](G7_SCOPE.json)

## Staging and deviations

The exact 11 commands, expected seconds and byte estimates are in the staging document. Each launcher checks the frozen packages and sealed inputs, sets CUDA_MODULE_LOADING=EAGER and NVIDIA_TF32_OVERRIDE=0, and opens fresh receipt and log paths. All 11 dry runs returned 78 with would_start true and neither a lease nor a GPU worker output directory. [Staged commands](STAGED_LINES.md) · [Admission log](../../logs/bwd_r4_gpu_dry.log) · [Staged pins](STAGED_PINS.json)

- The two control binaries initially landed under artifacts/bwd_r4/native/. They were moved intact into tc_signal/build-bwd-r4-voice/ to comply with L67. The build log retains their original hashes.
- The optional syscall census failed with “PTRACE_TRACEME: Operation not permitted”. G7 used the managed write boundary, audited Python reads with explicit records for native paths, compiler input stamps and read-only tracked diff. The initial pre-registration metadata import emitted a telemetry persistence warning. Registered runs kept telemetry state inside authorized scratch.
- The first ledger/report shell write interpreted quotation marks in the Markdown text and left both files incomplete. The files were corrected in place before closeout. No registered threshold or gate input changed.

## Prior art

- Linnainmaa (1970) and Voice BP1 (2026) supply reverse accumulation and the existing fp64 pullbacks. R4 exposes the value at the actual pre-tail dispatch. Higham (2002) and O’Neill (2014) supply centered differences and PCG64, applied here at three steps with per-site branch checks.
- Chen et al. (2016) and Griewank and Walther’s Revolve (2000) supply recomputation. R4 places checkpoints at Voice’s eight generator blocks and binds their aliasing weights as explicit operands.
- Kokoro (hexgrad, 2024–25), StyleTTS 2 (Li et al., 2023), HiFi-GAN (Kong et al., 2020) and iSTFTNet (Kaneko et al., 2022) supply the inherited graph. R4 contributes the generator mirror and the frozen-source tap control.
- NVIDIA CUDA 12.6 (2024), R3’s runtime gate (2026) and forge-gpu-run/GW3 (2026) supply the memory and admission primitives. R4 applies them to a complete decoder-side graph.
- Unverified — lead to check “Linnainmaa reverse accumulation 1970”, “Higham centered finite differences”, “O’Neill PCG64”, “Chen sublinear memory Griewank Walther revolve”, “Kokoro StyleTTS 2 HiFi-GAN iSTFTNet” and “cudaMemPoolAttrUsedMemHigh CUDA 12.6”.

[Complete file inventory](FILES_WRITTEN.txt)
