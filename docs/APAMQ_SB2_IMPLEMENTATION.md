# APAMQ-SB2 Fix-Round Receipt

Evidence state: **IMPLEMENTED / BUILT / CPU-CHECKED / GPU GATES PENDING LEAD**.
The sandbox has no CUDA device, so no GPU parity, sanitizer, memory, or timing
claim is made here.

## Found illegal-access bug

In the landed SB1 source (`b9a79a0`), `tensor_cuda/src/gemm_apa.cu:339-341`
performed an unchecked `atomicAdd` append into buffers allocated to the exact
scan-derived count. The count pass used `abs(score) >= threshold`; the append
pass used the non-equivalent inverse test `abs(score) < threshold` to reject.
For a non-finite score or threshold, both comparisons are false: the count pass
reserves no slot while the append pass proceeds, so `pair_rows[dst]` and
`pair_keys[dst]` write beyond the allocation. The following gather consumes the
corrupted index, which explains why the fault can surface later at the skinny
GEMM or a synchronization point and poison the CUDA context.

SB2 uses one positive-form, finite-only predicate for counting, debug masks,
and compaction. The appender is capacity-checked before either index write.
With `-DTC_APA_GEMM_DEBUG_ASSERTS=ON`, device assertions cover causal bounds,
append capacity, gathered row/key indices, chunk copies, and scatter indices.

## Compaction and synchronization

Each query sub-chunk preallocates `(row,key)` buffers to
`ceil(min(1, 2*Phi(-zthr))*valid_pairs)`, approximately twice the nominal
refine fraction. One device counter block contains the sub-chunk append cursor,
cumulative selected count, dropped count, and overflow flag. Atomic append
clamps at capacity. Sparse-route overflow increments both `dropped` and the
call overflow flag; public stats return
`(selected_pairs, valid_pairs, overflow_calls, dropped_pairs)`.

There is no count readback before or between compact/refine launches. One
four-word D2H readback occurs at the end of the native attention call, so the
synchronization count is O(1) in both query sub-chunks and refine work chunks.

## Transient policy

The query sub-chunk length is

```text
Lc = clamp(1, L, floor(288 MiB /
     (B * H * S * (14 + 8 * min(1, 2*Phi(-zthr))))))
```

The per-score estimate includes INT32 bulk, FP32 bulk, optional FP32 exact,
BF16 probabilities, and capacity-sized row/key indices. This is stricter than
the requested INT32+FP32-only 384 MiB bound and leaves room for call-global K
codes/scales, output, and the 32 MiB cuBLASLt workspace. The registered 64K
shape therefore never owns the original full-L INT32 and FP32 score matrices
simultaneously.

Sparse refinement keeps the SB1 gather plus strided-batched skinny BF16 GEMM
when its bounded Q/K/exact workspace is at most 128 MiB. Larger prefill chunks
use a dense BF16 QK GEMM with FP32 output, then blend only selected exact scores
over the FP32 bulk matrix. This fallback preserves numerical selection/blend
semantics and avoids an impossible multi-gigabyte capacity gather, but it is an
explicit architecture deviation from SB1's selected-only exact computation.
It must be judged by the lead's registered wall-time gates.

## Local checks

Build:

```bash
cd tensor_cuda
bash build.sh 89
```

CPU and skip-safe gates:

```bash
PYTHONPATH=tensor_cuda python3 -m pytest \
  tensor_cuda/tests/test_apamq_sb1.py -q -rs
```

Local result: `9 passed, 6 skipped`; all skips are CUDA-unavailable native
legs. New CPU checks cover the finite-only shared predicate, bounded overflow
accounting/stats surface, and registered-shape sub-chunk budget.

## Exact lead commands

Normal build and all gates/performance cells:

```bash
cd /mnt/ForgeRealm/wt/apamq-sb
bash tensor_cuda/build.sh 89
flock -w 7200 /tmp/forge-gpu.lock \
  bash tensor_cuda/run_apamq_sb2_lead.sh
```

Debug-assert build and memcheck of the previously failing cell:

```bash
cd /mnt/ForgeRealm/wt/apamq-sb
cmake -S tensor_cuda -B tensor_cuda/build \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DTC_APA_GEMM_DEBUG_ASSERTS=ON
cmake --build tensor_cuda/build -j
flock -w 7200 /tmp/forge-gpu.lock \
  env PYTHONPATH=tensor_cuda \
  compute-sanitizer --tool memcheck --error-exitcode=99 \
  python3 scripts/apamq_e1_sweep.py \
  --cell gemm_apa 1 512 prefill 16384
```

Restore the normal build afterward with
`-DTC_APA_GEMM_DEBUG_ASSERTS=OFF` (or `bash tensor_cuda/build.sh 89`).

## RED deviations

- GPU results remain pending; the sanitizer, all native gates, the prefill
  wall rails, transient-pool rail, and decode regression are unverified here.
- Large compact workspaces use the dense exact sub-chunk fallback described
  above. It preserves outputs but computes exact scores for unselected keys.
- The 288 MiB policy guarantees the score/index/probability estimate for every
  shape where one `(B,H,1,S)` plane fits that budget. Still larger planes need
  key-axis streaming, which is outside this L-sub-chunk-only order.
