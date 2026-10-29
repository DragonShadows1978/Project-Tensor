# Staged GPU pass rules (no GPU execution by seat)
Run GPU_COMMANDS.sh as written, in a clean lead/ receipt directory. Every label
is unique in this registration. Existing paths fail closed; preserve failed
runs and register a separate run directory/labels before retrying, never delete.
All inputs exist in inputs/ and hashes are in INPUTS.json. Staged package is
engine/tensor_cuda/tensor_cuda. Baseline is the immutable 69b154cd engine.
The command uses the current installed forge-gpu-run, with --min 300 --max 600;
the command itself is bounded by timeout --foreground 300s. Refused windows
are RED/missing, never a pass. No command here has been run on a GPU by seat.

L-G1: base/off uint16 BF16 and uint32 FP32 views equal bytes on all six tensors,
on GRAPA geometry and both small geometries. Includes train autograd (small),
all valid diagnostic variants except zero-output c; includes h/h_tf32 forwards. No masks or tolerances.
L-G2: on/off exact out,lse,thr on those cases; input bytes also equal.
L-G3: each on output/gradient matches float64 NumPy with tolerances frozen in
CALIBRATION.json BEFORE any selective-on capture. E is the pooled measured
base/off maxabs error per dtype and tensor across two small shapes/variants;
tolerance=max(8E,1e-5). Recompute calibration from raw base tensors and require
identity to the saved calibration. Every variant must give exactly zero off dK
and nonzero on dK at key 1, whose kq is all zero and threshold strictly positive.
This is the selected-only mutant's killing check. Variant c uses its deletion
ablation reference and contributes no zero tensors to bitwise gates.
L-G4: zthr=-1e6, all-selected fixtures, on/off all six tensors equal bytes.
Recheck stored thresholds are negative on all multi-key rows; the pinned
singleton dot must also meet the saved threshold. This proves every visible
pair selected without materializing a GRAPA-size pairwise mask.
The large fixture pins q(query 0) and kq(key 0) to .25 so the singleton bulk dot
is exact in either reduction order. No forward/threshold production code change.
L-G5: two fresh on captures equal bytes for all six tensors, deterministic mode.
Existing atomic variants can fail: TC_DETERMINISTIC does not control APA atomics.
A failure remains RED. This harness never changes dispatch to obtain equality.
L-G6: two kernel samples per arm after one warmup; Nsight CUDA records include
only apa_selective_bwd_kernel captured between cudaProfilerStart/Stop. Every
duration finite positive, exactly two kernel launches, report median and ratio.
CUDA-event backward wrapper samples are reported separately. No speed threshold;
P2 (1.3-2x) is a prediction only. Whole model/optimizer step unavailable from the
engine-only entry point; no registered checkpoint/optimizer fixture was supplied;
GRAPA times its step. Do not label backward-wrapper timing a training step.

Each capture's PASS only checks capture health. The gate verdict is recomputed
by verify from stored raw tensors/profiler records, not copied from that flag.
A GPU gate PASS additionally requires each dependency's governor TSV row to be
unique with outcome=ran and rc=0. If a command times out or does not run, it
cannot count even if partial files exist. Final GPU_RECEIPT.json contains PASS
for each L-G1..L-G6 and the combined PASS.

Work/time estimates (reasoning, no GPU timing evidence): small job processes
24 cases with <=4*17*19 pairs/head geometry, <=32 feature width: under ~10^8
scalar operations, estimated 2-25 s including CUDA cold start/I/O. Large job
has 16*4096*4097/2=134250496 visible query-key pairs, 128/64 feature widths:
roughly 1e11 scalar arithmetic operations and order 1e11 bytes of traffic per
backward, including up to ~2.58e10 BF16 atomics on. Estimate 10-120 s per capture.
Timing job: 1 warmup + 2 sampled backwards, one forward, Nsight export: estimate
30-240 s. These are scheduling estimates, not measured guarantees; 300 s is a
budget, and any timeout is RED. Jobs run sequentially, never concurrently.

Prior art: Bengio et al. 2013 STE; Dao et al. 2022 FlashAttention VJP;
NVIDIA CUDA events/Nsight Systems (years unverified - lead to check);
NIST FIPS 180-4 (2015) SHA-256. Integration/fixtures/receipt rules are ours.
