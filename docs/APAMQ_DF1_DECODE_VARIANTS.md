# APAMQ-DF1/DF2 decode bandwidth-floor variants

Evidence status: implementation and CPU/build receipts are local; every CUDA
timing and equivalence gate is **LEAD-RUN / OPEN** because this sandbox has no
GPU. No performance number is claimed here.

## Registered target

Before lead measurement: `B=1,H=16,KVH=1,L=1,D=512,S=65536`, BF16,
bottom-right causal INT4 APA decode must be **at most 1.000 ms/token**. The
benchmark prints `PASS` or `MISS` without changing this rail.

## Stage timing design

`scripts/apamq_df1_bench.py` uses CUDA events placed in C++ immediately around
the four logical stages. It reports the mean `pack`, `stats`, `split`, and
`merge` times, their sum, plus a median whole-call CUDA-event time. V4's stats
stage contains the partition kernel and its tiny row reducer under one event
pair. The matrix is
`D={128,512}`, `S={8192,16384,32768,65536}`, with fixed
`B=1,H=16,KVH=1,L=1`, BF16 causal decode. Setup, input creation, workspace
creation, and host copies are outside the timed calls.
For V1 cells, the profiler rewinds only the derived valid-row marker before
each repetition, so the measured pack stage repacks exactly one appended row.

The code-resolved grid receipt at `rows=16` is:

| S | V4 stats P / partial blocks | V4 reduce blocks | baseline or V2-cached split P / blocks | V3 uncached split P / blocks | merge blocks |
|---:|---:|---:|---:|---:|---:|
| 8192 | 4 / 64 | 16 | 4 / 64 | 4 / 64 | 16 |
| 16384 | 4 / 64 | 16 | 8 / 128 | 4 / 64 | 16 |
| 32768 | 4 / 64 | 16 | 16 / 256 | 4 / 64 | 16 |
| 65536 | 4 / 64 | 16 | 32 / 512 | 4 / 64 | 16 |

V4 targets one 56-SM wave for stats independently of split. V3 does the same
for uncached split. When V2 removes the split-stage packed-K dot, that stage is
lighter and retains the 2,048-key width instead of collapsing from 32
partitions to four; this is the measured V2xV3 interaction fix.

## Variant flag matrix

| Variant | Selection | Default | Effect |
|---|---|---:|---|
| V-off | no workspace; env vars unset | on | Original whole-cache pack and 2,048-key split partitions |
| V1 | `workspace=tc.apa_int4_workspace(k, capacity=Smax)` | off | Persistent symmetric-7 codes/scales; packs only newly appended rows; call `workspace.reset()` after non-append mutation |
| V2 | `TC_APAMQ_DF_V2=1` | off | Stats writes its exact fp32 bulk scores; split consumes them instead of rereading/dequantizing packed K |
| V3 | `TC_APAMQ_DF_V3=1` | off | Sizes uncached split for one 56-SM wave; cached V2 split intentionally keeps more blocks |
| V4 | `TC_APAMQ_DF_V4=1` | off | Split-K stats partials plus tiny row reducer; V2 emits the bulk cache from the partial kernel |
| Combined | workspace plus any/all env vars | off | Orthogonal A/B composition |

`TC_APA_SELECTIVE_PATH=2` is set by the benchmark to hold the split-K family
constant. Existing public calls without `workspace` and without DF env vars
retain the V-off path.

V2 is deliberately a bulk-score-cache design, not a cooperative single-kernel
global barrier: stats writes `rows*S` fp32 scores and split reads that 4 MiB
cache at the registered shape. With V4, the one packed-K pass both creates all
stats partials and fills the cache; the reducer reads only two tiny fp32 arrays.
It removes the split stage's second packed-K dequantization walk while retaining
exact full-range threshold semantics.

The expected best DF2 combination is `v1_v2_v4` (and
`v1_v2_v3_v4` has the same registered-shape split plan because V2 overrides
the V3 four-part choice). This is a design expectation only; the unchanged
1.000 ms rail remains OPEN until lead measurement.

V1 storage is `capacity*(ceil(D/2)+4)` bytes per batch/KV head: 4.25 MiB for
`D=128,S=64K`, but 16.25 MiB for the registered `D=512,S=64K`. The order's
4 MiB figure applies to D=128 codes, not the D=512 target.

## Exact lead commands

Build this worktree with the authorized read-only pybind11 source:

```bash
cmake -S tensor_cuda -B tensor_cuda/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 -DFETCHCONTENT_SOURCE_DIR_PYBIND11=/mnt/ForgeRealm/Project-Tensor/tensor_cuda/build/_deps/pybind11-src && cmake --build tensor_cuda/build -j2
```

Run all sixteen V-off/V-on combinations across the registered matrix:

```bash
PYTHONPATH=tensor_cuda python3 scripts/apamq_df1_bench.py
```

Run only the decisive 64K target for quick iteration:

```bash
PYTHONPATH=tensor_cuda python3 scripts/apamq_df1_bench.py --d 512 --s 65536 --warmup 3 --repeats 20
```

Run the expected-best combination alone:

```bash
PYTHONPATH=tensor_cuda python3 scripts/apamq_df1_bench.py --variants v1_v2_v4 --d 512 --s 65536 --warmup 3 --repeats 20
```

Run FA1's 11 gates plus the seven DF GPU gates and CPU plan gate:

```bash
PYTHONPATH=tensor_cuda python3 -m pytest tensor_cuda/tests/test_apa_selective_int4.py tensor_cuda/tests/test_apamq_df1.py -q -rs
```

## Nsight Compute command list, one stage each

Each command runs one expected-best target call and filters one kernel family.
The summary row count is the launch-count receipt; the requested metrics cover
grid/block shape, achieved occupancy, and DRAM bytes/throughput.

```bash
ncu --target-processes application-only --kernel-name-base demangled --kernel-name 'regex:apa_int4_pack_rows_kernel' --launch-count 1 --metrics 'launch__grid_size,launch__block_size,sm__warps_active.avg.pct_of_peak_sustained_active,dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum,dram__bytes_write.sum' --print-summary per-kernel python3 scripts/apamq_df1_bench.py --ncu-once --variant v1_v2_v4 --ncu-d 512 --ncu-s 65536

ncu --target-processes application-only --kernel-name-base demangled --kernel-name 'regex:apa_selective_int4_stats_splitk_kernel' --launch-count 1 --metrics 'launch__grid_size,launch__block_size,sm__warps_active.avg.pct_of_peak_sustained_active,dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum,dram__bytes_write.sum' --print-summary per-kernel python3 scripts/apamq_df1_bench.py --ncu-once --variant v1_v2_v4 --ncu-d 512 --ncu-s 65536

ncu --target-processes application-only --kernel-name-base demangled --kernel-name 'regex:apa_selective_int4_stats_reduce_kernel' --launch-count 1 --metrics 'launch__grid_size,launch__block_size,sm__warps_active.avg.pct_of_peak_sustained_active,dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum,dram__bytes_write.sum' --print-summary per-kernel python3 scripts/apamq_df1_bench.py --ncu-once --variant v1_v2_v4 --ncu-d 512 --ncu-s 65536

ncu --target-processes application-only --kernel-name-base demangled --kernel-name 'regex:apa_selective_int4_split_kernel' --launch-count 1 --metrics 'launch__grid_size,launch__block_size,sm__warps_active.avg.pct_of_peak_sustained_active,dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum,dram__bytes_write.sum' --print-summary per-kernel python3 scripts/apamq_df1_bench.py --ncu-once --variant v1_v2_v4 --ncu-d 512 --ncu-s 65536

ncu --target-processes application-only --kernel-name-base demangled --kernel-name 'regex:apa_selective_merge_kernel' --launch-count 1 --metrics 'launch__grid_size,launch__block_size,sm__warps_active.avg.pct_of_peak_sustained_active,dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum,dram__bytes_write.sum' --print-summary per-kernel python3 scripts/apamq_df1_bench.py --ncu-once --variant v1_v2_v4 --ncu-d 512 --ncu-s 65536
```

Repeat any command with `--variant v2`, `v3`, `v4`, or `v1_v2_v3_v4` to profile the
selected implementation. `--launch-count 1` means one profiled matching
launch even if Nsight Compute replays it for multiple metric passes.

## Tolerances and state contract

V-on versus V-off fp32 decode equivalence is registered at
`rtol=3e-3, atol=3e-3`, matching FA1's split-K reassociation rail. V1 preserves
FA1 `roundf` symmetric-7 ties. V2 caches the same fp32 bulk scalar produced by
stats. V3 changes online-softmax partition association. V4 changes only fp32
stats partial-sum association and is gated both with and without V2 cache
emission. V1 callers must reset after overwrite, eviction, ring wrap, splice,
or any other non-append change.
