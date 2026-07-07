# Phase 0.4 — Register/Shared-Memory/Occupancy Receipts

Evidence class: kernel sweep (compile-time `ptxas -v`, no GPU run).

## Compile command

```
/usr/local/cuda-12.6/bin/nvcc -c src/kernels.cu -o <scratch>/kernels.o \
  -I include -arch=sm_89 -O3 -std=c++17 --use_fast_math -Xcompiler -fPIC -Xptxas -v
```

Run from `tensor_cuda/` (matches `build.sh`'s toolkit selection: `/usr/local/cuda-12.6`,
not the distro `/usr/bin/nvcc` at 12.0). Device-code-only `-c` compile of `src/kernels.cu`
standalone, output routed to
`/tmp/claude-1000/-home-vader/8e6895ec-3f98-4d60-acf1-5f24fb3c9128/scratchpad/ptxas_build/`
(existing `tensor_cuda/build/` and installed package untouched). Compilation succeeded
on the first attempt — `kernels.cu` only pulls in `tc/core.h` (self-contained: `<cstdint>`,
`<memory>`, `<string>`, `<tuple>`, `<vector>`) plus CUDA runtime/fp16/bf16 headers, so no
blocking include or missing dependency was hit; no workaround was needed. 238 kernel
entry-point instantiations compiled for `sm_89`.

## GPU model (RTX 4070 SUPER, Ada / SM 8.9)

56 SMs · 65,536 regs/SM · max 1536 threads/SM (48 warps/SM) · max 16 blocks/SM ·
100 KB shared/SM usable.

Occupancy = (limiting blocks/SM × warps/block) / 48 warps/SM. Register blocks/SM
computed by rounding regs/thread up to a warp's register block, then to a 256-register
scheduler allocation unit (Ada/Ampere convention), then floor(65536 / regs-per-block).
This is an analytical approximation, not `ncu`-measured achieved occupancy — treat as
a receipt for relative comparison, not a substitute for the Phase 0.1/0.2 hardware
measurement.

## Spill scan (all 238 instantiations, whole file)

**0 kernels show nonzero spill loads or spill stores** at `-O3`, `sm_89`. No spill
flags below.

## Receipts table

| kernel | dtype variants | regs/thread | static smem | spills | block size | warps/blk | limiting blocks/SM (reg/thr/smem/cap) | occupancy | limiter |
|---|---|---|---|---|---|---|---|---|---|
| causal_softmax_kernel | bf16/f16/f32 | 20 | 3072 B | 0/0 | 256 (`kT`) | 8 | 10 / 6 / 33 / 16 | **100.0%** | threads |
| apa_selective_kernel | bf16/f16/f32 × DMAX{64,128,256,512} | 40 | 0 | 0/0 | 128 | 4 | 12 / 12 / 16 / 16 | **100.0%** | regs & threads (tied) |
| apa_selective_sink_kernel | bf16/f16/f32 × DMAX{64,128,256,512} | 40 | 0 | 0/0 | 128 | 4 | 12 / 12 / 16 / 16 | **100.0%** | regs & threads (tied) |
| apa_selective_fwd_train_kernel | bf16/f16/f32 × DMAX{64,128,256,512} | 40 | 0 | 0/0 | 128 | 4 | 12 / 12 / 16 / 16 | **100.0%** | regs & threads (tied) |
| apa_selective_bwd_kernel | bf16 × DMAX{64,128,256,512} | 52 | 0 | 0/0 | 128 | 4 | 9 / 12 / 16 / 16 | **75.0%** | **regs** |
| apa_selective_bwd_kernel | f16 × DMAX{64,128,256,512} | 48 | 0 | 0/0 | 128 | 4 | 10 / 12 / 16 / 16 | **83.3%** | **regs** |
| apa_selective_bwd_kernel | f32 × DMAX{64,128,256,512} | 53 | 0 | 0/0 | 128 | 4 | 9 / 12 / 16 / 16 | **75.0%** | **regs** |
| apa_blend_softmax_kernel2 | bf16 | 19 | 1024 B | 0/0 | 256 | 8 | 10 / 6 / 100 / 16 | **100.0%** | threads |
| apa_blend_softmax_kernel2 | f16 | 18 | 1024 B | 0/0 | 256 | 8 | 10 / 6 / 100 / 16 | **100.0%** | threads |
| apa_blend_softmax_kernel2 | f32 | 27 | 1024 B | 0/0 | 256 | 8 | 8 / 6 / 100 / 16 | **100.0%** | threads |
| apa_blend_softmax_sink_kernel | bf16 | 19 | 1024 B | 0/0 | 256 | 8 | 10 / 6 / 100 / 16 | **100.0%** | threads |
| apa_blend_softmax_sink_kernel | f16 | 18 | 1024 B | 0/0 | 256 | 8 | 10 / 6 / 100 / 16 | **100.0%** | threads |
| apa_blend_softmax_sink_kernel | f32 | 26 | 1024 B | 0/0 | 256 | 8 | 8 / 6 / 100 / 16 | **100.0%** | threads |
| int4_gemv_kernel | bf16/f16/f32 | 39 | 0 (static)† | 0/0 | 256 | 8 | 6 / 6 / 16 / 16 | **100.0%** | regs & threads (tied) |
| int4_gemm_fused_kernel | bf16/f16/f32 | 38 | 2048 B | 0/0 | 256 (16×16 tile) | 8 | 6 / 6 / 50 / 16 | **100.0%** | regs & threads (tied) |
| intn_gemv_kernel | bf16/f16/f32 | 40 | 0 (static)† | 0/0 | 256 | 8 | 6 / 6 / 16 / 16 | **100.0%** | regs & threads (tied) |
| intn_gemm_fused_kernel | bf16/f16 | 40 | 2048 B | 0/0 | 256 (16×16 tile) | 8 | 6 / 6 / 50 / 16 | **100.0%** | regs & threads (tied) |
| intn_gemm_fused_kernel | f32 | 38 | 2048 B | 0/0 | 256 (16×16 tile) | 8 | 6 / 6 / 50 / 16 | **100.0%** | regs & threads (tied) |
| intn_dequant_t_kernel | bf16/f16/f32 | 26 | 0 | 0/0 | 256 (`kT`) | 8 | 8 / 6 / 16 / 16 | **100.0%** | threads |
| mxfp4_gemv_kernel | bf16/f16/f32 | 21 | 0 (static)† | 0/0 | 256 | 8 | 10 / 6 / 16 / 16 | **100.0%** | threads |
| mxfp4_gemm_kernel | bf16/f16/f32 | 38 | 2048 B | 0/0 | 256 (16×16 tile) | 8 | 6 / 6 / 50 / 16 | **100.0%** | regs & threads (tied) |
| rope_kernel | bf16/f16/f32 | 20 | 0 | 0/0 | 256 (`kT`) | 8 | 10 / 6 / 16 / 16 | **100.0%** | threads |
| rms_norm_kernel | 7 of 9 in/out combos | 15 | 2048 B | 0/0 | 256 (`kT`) | 8 | 16 / 6 / 50 / 16 | **100.0%** | threads |
| rms_norm_kernel | 2 of 9 in/out combos (f16→bf16, f16→f16) | 14 | 2048 B | 0/0 | 256 (`kT`) | 8 | 16 / 6 / 50 / 16 | **100.0%** | threads |
| unary_kernel (top elementwise) | bf16/f16/f32 | 12 | 0 | 0/0 | 256 (`kT`) | 8 | 16 / 6 / 16 / 16 | **100.0%** | threads |
| binary_kernel (top elementwise) | bf16/f16/f32 | 32 | 0 | 0/0 | 256 (`kT`) | 8 | 8 / 6 / 16 / 16 | **100.0%** | threads |
| scalar_kernel (top elementwise) | bf16/f16/f32 | 10 | 0 | 0/0 | 256 (`kT`) | 8 | 16 / 6 / 16 / 16 | **100.0%** | threads |

† `int4_gemv_kernel` / `intn_gemv_kernel` / `mxfp4_gemv_kernel` take their large
per-row staging buffer as **dynamic** shared memory (`shmem = K * sizeof(T)` bytes,
up to 96 KB via `cudaFuncSetAttribute(..., 96*1024)` for K=15360 rows per the
comment at kernels.cu:2117-2122). `ptxas -v` only reports **static** smem (0 B here);
the real per-block dynamic smem is data-dependent (up to 96 KB) and at large K becomes
the actual occupancy limiter (1 resident block/SM once dynamic smem exceeds 50 KB) —
not visible in this compile-time-only receipt. Flagging this as the one place where
the table's "100%/threads" verdict is compile-time-only and can be wrong at runtime
for large-K decode shapes; Phase 0.1/0.2 `ncu` occupancy numbers are authoritative there.

## Launch-site block sizes (grepped from kernels.cu, current tree)

- `kT = 256` (kernels.cu:211) — used by `causal_softmax_kernel`, `rms_norm_kernel`,
  `rope_kernel`, `intn_dequant_t_kernel`, and the generic elementwise
  (`unary_kernel`, `binary_kernel`, `scalar_kernel`, `fill_kernel`, `compare_*`, etc.)
  via `nblk(n), kT`.
- `apa_selective_kernel` / `_sink_kernel` / `_fwd_train_kernel` / `_bwd_kernel`:
  `threads = 128` (kernels.cu:1200, 1381, 1654, 1691), grid = `rows = B*H*L`.
  DMAX dispatched at 64/128/256/512 by `cap = max(D, VD)` (kernels.cu:1205-1222) —
  register count is identical across all 4 DMAX values per dtype (DMAX only changes
  a local array's compile-time bound, not live register pressure at the reported
  points), so occupancy is DMAX-invariant per dtype.
- `apa_blend_softmax_kernel2` / `apa_blend_softmax_sink_kernel`: fixed `256`
  (kernels.cu:1814, 1918).
- `int4_gemv_kernel` / `intn_gemv_kernel` / `mxfp4_gemv_kernel`: `threads = 256`
  (1D, 8 warps), grid = `ceil(N / (threads/32))` — decode (M==1) fast path only.
- `int4_gemm_fused_kernel` / `intn_gemm_fused_kernel` / `mxfp4_gemm_kernel`:
  `dim3 block(TC_I4_TILE, TC_I4_TILE)` with `TC_I4_TILE = 16` (kernels.cu:2010) →
  256 threads, tile path for M>1.

## Flags

- **Register-pressure flag:** `apa_selective_bwd_kernel` is the only focus kernel
  below 100% analytical occupancy — 75.0% (bf16/f32, 52-53 regs/thread) to 83.3%
  (f16, 48 regs/thread), limited by register file, not shared mem or block-count cap.
  This is the attention backward kernel (dq/dk/dv), consistent with it carrying the
  most live state (5 input pointers + lse/thr + 3 output pointers + per-DMAX
  accumulators) of the four `apa_selective*` kernels. No spills — the extra registers
  are used, not overflowing to local memory — so this is a real occupancy cost, not
  a correctness or spill problem.
- **No kernel below 25% occupancy.** Nothing else flagged.
- **No spills anywhere in the file** (0/238 instantiations, all kernels, not just
  the focus set).
- **Dynamic-smem blind spot** on the three `*_gemv_kernel`s (int4/intn/mxfp4): see
  † above — this compile-time receipt cannot see the data-dependent dynamic
  shared-memory allocation that the code itself flags (kernels.cu:2117-2122) as
  the thing that keeps large-K decode rows off the slow tile path. Phase 0.1/0.2
  hardware receipts should carry the real occupancy number for those shapes.
