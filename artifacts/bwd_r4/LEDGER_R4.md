# BWD-R4a r4 implementation ledger

## Registration and diagnosis

- SHIM-RUN `20261119T081732Z-223756` from `logs/bwd_r4_r4.log:14`. Read the worktree order, local HOUSE_RULES, r3 mechanism report, r3 staging, sealed tolerance files and lead logs. Order SHA-256 `a8e66d612b0d05fc9fb49283b6debb1a003de0552687370d8f12c5c788fc8532`.
- R3 end-to-end receipts show G1 style relative L2 `1.0322948599702484e-5` and 313 compared parameters with zero failures, G2 18/26, G3 224/314 byte mismatches, G4 in-process reruns with zero mismatches, and G5 checkpoint reserved-plus-non-pool `4,318,298,112` / live-plus-non-pool `3,800,666,112` bytes. The four publication failures end with `OSError: [Errno 18] Invalid cross-device link`. The r3 traces and the A1 operand census show the old rename went from /tmp (device 2050) to this worktree (66305). R3 G2 and G3 scores remain the original results. [Census](../../logs/bwd_r4_r4_census_R4.log)
- Operand and registration preparation read the real BP1 graph and style shape and froze all 26 probe identities before any gates. Seed is the first 32 bits of the immutable order hash. The 18 coordinates and eight normalized vectors were checked against the recorded r1/r2 identities. [Probe manifest](FD_PROBES_R4.json) · [Preparation log](../../logs/bwd_r4_r4_register_R4.log)
- Sealed the lead's A2/A3 thresholds at 2026-11-19 08:22:12 UTC. SHA-256 `e615c23c40ab708ddd965b1f86599c9314eaeaa78f87c056a6f413ce2fd90f4f`. [Registration](AMENDMENT1_R4.md) · [Seal](AMENDMENT1_R4.sha256)

## Implementation

- Four authorized harness files edited. Round 4 stages worker output in /tmp, copies in bounded chunks, flushes and fsyncs each destination temporary file, renames within the worker directory, and fsyncs that directory. Existing receipts are retained.
- G2 uses independent centered differences from BP1 forward and per-slot norms from the separate BP1 VJP. G3 and r4m01 use the G1 per-tensor comparison helper, with the plain gradient defining relative scales and the largest parameter norm defining the floor. The mutant kill additionally requires a numeric error-to-bound witness at or above 10 and unchanged forward bytes.
- The fine schedule uses 31 checkpoints on A3 crop and G5 checkpoint. Its replay tape covers one generator residual iteration at a time. It checkpoints five decoder residual stages and two upsampling paths to reduce saved non-generator activations. A synchronized trim follows forward. The G5 breakdown keeps both phase highs and the same whole-run reserved-plus-observed-non-pool measure. G0, G1, G-AFF, G4 and r4m02 retain the r3 numerical paths and criteria.
- A5 stages the same line labels, lease inputs, host estimates and scope sizes. The live launcher records worker RSS and cgroup v2 scope memory.peak where available.

## Prior art

- POSIX.1-2017 supplies same-filesystem rename and fsync. A1 chooses a same-directory temporary file to publish the cross-device copy. Unverified — lead to check “POSIX rename EXDEV fsync”.
- Cauchy (1821), Schwarz (1888), Higham (2002), O'Neill (2014) and BP1 (2026) supply the projection inequality, centered differences, PCG64 and branch checks. A2 applies G1's relative L2 class to 26 freshly registered probes. A3 uses Project-Tensor BWD-R4a G1 (2026) for per-tensor bands and the floor with a plain fp32 reference. Unverified — lead to check “Cauchy Schwarz inequality”, “Higham centered differences”, “PCG O'Neill 2014”.
- Chen et al. (2016) supply activation recomputation. CUDA 12.6 (NVIDIA, 2024) supplies pool trimming and high-water accounting. A4 chooses the Voice graph boundaries and records phase highs around the trim. Linux cgroup v2 memory.peak (2022), systemd v231/v243 (2016/2019), and SystemOps forge-gpu-run/GW3 (2026) supply host measurement, scopes and dry admission. Unverified — lead to check “Training Deep Nets with Sublinear Memory”, “CUDA cudaMemPoolTrimTo”, “Linux cgroup v2 memory.peak”.

## CPU gate receipts

- A1 smoke PASS. Copied the real `MEASUREMENT.json` (1,113 bytes) and `GRADIENT_BYTES.npz` (213,052,946 bytes) through the publication helper from device 2050 to 66305. SHA-256 matched the r3 source and r4 scratch for both files, the scratch files remained, and a second publication refused the existing receipt. [Smoke log](../../logs/bwd_r4_r4_exdev_R4.log)
- BP1 fp64 gate PASS, 26/26 registered probes under the existing DESIGN §6 basis. Observed 12 pre-tail LeakyReLU sites. Candidate G2 remains for the GPU. [CPU gate log](../../logs/bwd_r4_r4_host_fd_R4.log) · [Oracle slopes](FD_ORACLE_R4.json)
- One pre-staging source correction made the shared norm calculation explicitly fp64 for G3's fp32 plain reference. G1 already supplies fp64 reference tensors. This changes no threshold or probe.
- Correction to the exact r3 G1 transcription above. The lead log gives `1.0322924353585311e-5`. [Lead G1](../../logs/bwd_r4_r3_gpu_g1-g2_20261119T081307749863Z-218676_R3.log#L8)

## Staging and verification

- Prepared a 54-entry pin map and staging sheet with CUDA hidden. Exactly four prior harness source hashes changed and five registration/oracle entries were added to the prior 49. Verified all 15 files in the frozen candidate and both sealed tolerance files. [Preparation log](../../logs/bwd_r4_r4_prepare_R4.log) · [Pins](STAGED_PINS_R4.sha256)
- Ran the 16 foreground real-admission dry calls with CUDA hidden. All exited 78, reported one true would_start and produced no lease or worker output. The per-line argv and scopes exactly match r3. Launcher preflight showed both address-space limits unlimited in all 16 cases. The verification indexed hashes, host estimates and scope limits. [Dry log](../../logs/bwd_r4_r4_dry_R4.log) · [Dry receipts](DRY_RUNS_R4.json) · [Verification](../../logs/bwd_r4_r4_verify_retry_R4.log)
- Verified the four-file diff against copies whose hashes match the r3 pins. Checked Voice mtimes and write-attempt counters in registration, oracle, preparation and all admissions. No live r4 GPU receipt or lease was created. [Diff](DIFF_R4.patch)
- G2 candidate projection, G3 crop equivalence, the r4m01 kill ratio, and G5 reserved footprint await GPU evidence. R3 G5 remains RED by 318,298,112 bytes. R3 measured 2,919,235,584 peak pool-reserved bytes and 1,399,062,528 maximum sampled non-pool bytes. [R3 memory log](../../logs/bwd_r4_r3_gpu_g5-checkpoint_20261119T081344513986Z-219175_R3.log#L6)

## Deviations

- One staging patch was rejected for an out-of-order context line. It wrote nothing and was reapplied. One pre-staging harness correction explicitly evaluates the shared norm in fp64 for A3.
- The first verification command looked for launcher preflight in the per-line child log and emitted `ValueError: not enough values to unpack (expected 1, got 0)`. Its shell tail printed the error and returned the cat status. The corrected verifier reads preflight from the aggregate launcher log, runs fail-fast, and passed against the original dry receipts. No gate was rerun. [First verifier](../../logs/bwd_r4_r4_verify_R4.log) · [Corrected verifier](../../logs/bwd_r4_r4_verify_retry_R4.log)
- The initial inventory read hit an output cap and queried r4 scratch before creation. Targeted reads recovered the source and operand census. ONNX Runtime used its in-memory telemetry identifier because it could not persist the ID during registration. No receipt or gate loss. [Registration log](../../logs/bwd_r4_r4_register_R4.log)
- Order-lint emitted warnings at dispatch. The shim started the seat. The scoped census, registration seal, dry decisions and inventory provide the execution record. [Shim log](../../logs/bwd_r4_r4.log#L1)
