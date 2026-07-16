# PAINT-CUDA-2 frozen engine gate register

Status: frozen before implementation on branch `paint-cuda-kernels` at
`a463024b55d610267dc0c4dc1c0811b670be369d`.

This leg is engine-only. It does not import or modify ColdCast. The NumPy
comparisons in `tensor_cuda/tests/test_paint_bake.py` vendor the operation order
from ColdCast `hy3d_tc/paint/bake.py:649-704,757-781`.

## Public contracts

```python
bake_back_project(
    atlas_positions_h_f32,
    view_f32,
    view_depth_f32,
    view_reliable_u8,
    view_cosine_f32,
    world_to_camera_f32,
    image_projection_f32,
    depth_threshold,
    *,
    threads=256,
) -> tuple[valid_u8, colors_f32, cosine_f32, depth_delta_f32]

bake_cosine_blend(
    view_colors_f32,
    view_cosine_f32,
    view_valid_u8,
    view_weights_f32,
    view_enabled_u8,
    *,
    threads=256,
) -> tuple[texture_f32, trust_f32, valid_u8]
```

`bake_back_project` launches one thread per atlas sample. Inputs are
`atlas_positions_h_f32[N,4]`, square `view_f32[H,H,C]`,
`view_depth_f32[H,H]`, `view_reliable_u8[H,H]`,
`view_cosine_f32[H,H]`, and two float32 4x4 matrices. Projection is the
frozen row-vector expression
`atlas_positions_h @ world_to_camera.T @ image_projection`. The output stays
in atlas-sample order: invalid colors and cosine are zero, while
`depth_delta_f32[N]` reports the absolute projected/winner-depth difference
used by the strict `< depth_threshold` verdict.

The square-view requirement deliberately freezes ColdCast's current
`indices = img_y * height + img_x` convention. Generalizing or silently
repairing that convention belongs to a separately registered change.

`bake_cosine_blend` consumes `[V,N,C]`, `[V,N]`, `[V,N]`, `[V]`, and `[V]`
inputs. Each output thread visits views in increasing dimension-0 order.
`view_enabled_u8` carries the caller's frozen view-level overlap/skip verdict;
disabled views are not read into the accumulation. The kernel computes
float32 `view_weight * power(cosine, 4)`, applies the strict `> 0` comparison,
then performs multiply and add as separately rounded operations. It returns
the normalized texture, unnormalized trust, and `trust > 1e-8` validity.
There are no cross-view atomics and launch shape is not semantic.

Both operations are detached, CUDA-only, and float32-only where named.
`threads` must be a warp multiple in `[32,1024]`.

## Scope boundary

Canny, reliability-mask construction, ESRGAN replay, Navier-Stokes inpaint,
and xatlas remain CPU/external in this leg. The scout receipt at
`ColdCast/logs/paint_cuda_0_r1.log:26756` reports that back-projection atlas
indices are unique, but indexed accumulation at ColdCast `bake.py:773-781`
does not move here: both public outputs remain aligned to the input atlas
samples. Therefore this leg introduces no duplicate-sensitive scatter and
needs no debug duplicate exception.

## Registered gates

| Gate | Frozen pass condition |
|---|---|
| G-VIS | On front-occluder, behind-occluder, grazing-cosine, exact-depth-tie, and threshold-boundary synthetic cases, CUDA and the vendored CPU oracle have identical verdicts on every non-tie sample. Exact-depth and threshold-boundary tie counts are reported separately; neither count has a threshold. |
| G-TEXEL | For samples with matching visibility on consistently wound fixtures, back-projected bilinear colors and the final ordered blend are bit-exact against the vendored CPU path. If a specific operation prevents exactness, the exact claim stops at that operation and the mismatch count, maximum ULP, and maximum absolute spread are reported without proposing a tolerance. |
| G-DET | All output buffers are byte-equal across five reruns at each of two launch configurations and byte-equal across configurations. |
| G-SUITE | The full engine suite adds no failure relative to the before count. The standing GroupNorm failure and the existing selector CLI collection exclusion are recorded explicitly. |

## Performance registration

Report synchronized kernel wall time for the compact fixture and for a
synthetic production material with 1,230,000-face provenance, six 2048x2048
views, and a 4096x4096 atlas. Face count is provenance for the upstream leg-1
raster; these kernels scale with valid atlas samples, views, and channels.
Report peak GPU memory against the registered 3.0 GiB rail. This leg registers
no performance threshold.

## Pre-implementation arithmetic probe

The existing generic CUDA `pow` primitive was compared with NumPy float32
`np.power(x, 4)` on 1,000,000 seeded values in `[0,1]`: 62,850 values differed,
with maximum spread one ULP / `5.960464477539063e-08`. This probe does not
change G-TEXEL and is not a threshold. The fused implementation must be
measured independently; if it uses the same native `powf` behavior, G-TEXEL
is RED specifically at the power operation while earlier exact subcontracts
remain eligible to pass.

## Execution receipt

Build: PASS, CUDA 12.6.85, Release sm_89. Generated flags:

```text
CUDA: -O3 -DNDEBUG -std=c++17 --generate-code=arch=compute_89,code=[compute_89,sm_89] -Xcompiler=-fPIC -Xcompiler=-fvisibility=hidden
CXX:  -O3 -DNDEBUG -std=gnu++17 -fPIC -fvisibility=hidden
```

| Gate | Result | Receipt |
|---|---|---|
| G-VIS | PASS | Synthetic front/behind occluders, grazing cosine, exact depth, and threshold-boundary cases: 4 exact-depth ties and 1 threshold-boundary tie reported; 3 non-tie samples, 0 non-tie verdict divergence, and 0 total verdict divergence. No tie threshold inferred. |
| G-TEXEL back-project | PASS | On 256 atlas samples from a consistently wound quad, valid, bilinear color, sampled cosine, and depth-delta buffers were byte-exact versus the vendored NumPy expression. A power-of-two cosine fixture also proved the ordered multiply/add/normalize blend byte-exact when power itself is exact. |
| G-TEXEL full blend | RED at native `powf` | On 65,537 samples across six frozen-order views, trust differed at 3,748 samples and final texture at 6,932 texels. Maximum spread: trust 2 ULP / `2.384185791015625e-07`; texture 4 ULP / `2.384185791015625e-07`. The registered exact test is a strict xfail at this named operation. No tolerance or replacement threshold is proposed. |
| G-DET | PASS | Every back-project and blend output buffer was byte-equal over five reruns at 64 threads, five reruns at 512 threads, and across both launch configurations. |
| G-SUITE | PASS relative to standing failure | Before: 293 passed, 18 skipped, 1 failed. After: 299 passed, 18 skipped, 1 xfailed, 1 failed. The before/after failure is the same `test_ext_phase7.py::test_norms_and_conv1d`; the xfail is the frozen G-TEXEL exact assertion stopped at native `powf`. `test_selector_accuracy.py` is the same excluded CLI script before and after. |

The `powf` result is an operation-level stop, not a general bake tolerance:
projection, visibility, fixed-tap bilinear interpolation, view order,
separately rounded weighted multiply/add, and normalization all retained their
exact subcontracts where exercised independently of the host/device libm
difference.

Performance uses resident inputs, steady-state transient pooling, explicit
CUDA synchronization, and excludes host input construction/upload. The
per-material figure is `6 * median back-project/view + median blend`.

| Workload | Back-project / view | Six-view blend | Per-material kernel wall |
|---|---:|---:|---:|
| Compact fixture: 1,852-face provenance, 6x64x64 views, 64x64 atlas, 30 repeats | median 0.012518 ms; mean 0.012736 ms; range 0.012313-0.015108 ms | median 0.013320 ms; mean 0.013856 ms; range 0.013074-0.026070 ms | 0.088431 ms |
| Synthetic production: 1,230,000-face provenance, 6x2048x2048 views, 4096x4096 atlas, 5 repeats | median 2.151375 ms; mean 2.154122 ms; range 2.143881-2.163478 ms | median 4.297350 ms; mean 4.304055 ms; range 4.293362-4.326916 ms | 17.205600 ms |

The face counts are provenance only for these kernels; upstream raster work
is measured by PAINT-CUDA-1. Production VRAM was measured with back-project
view inputs streamed and the six aligned blend inputs resident:

| VRAM measure | MiB |
|---|---:|
| Process after CUDA context, before benchmark inputs | 182 |
| Back-project resident inputs / peak with outputs | 524 / 878 |
| Blend resident inputs / peak with outputs | 1,818 / 2,106 |
| Peak whole-GPU increase over the pre-input baseline | 1,924 |
| Registered rail | 3,072 |

VRAM rail: PASS. The post-run GPU returned to 197 MiB used and 0% utilization.
