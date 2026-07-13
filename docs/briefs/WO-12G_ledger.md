# WO-12G ledger — object DDA grazing-angle corner leak

## 2026-07-12 — scope and pre-fix deterministic reproduction

This order is restricted to the additive `terrain_render` OBJECT overlay,
additive `tensor_cuda/tests/test_terrain_render.py` coverage, the two
`artifacts/wo12g_*.ppm` receipts, and this ledger.  `dda_raycast`, all terrain
kernels and their behavior, and APA are read-only rails.  No subagents, git
writes, network access, or pip actions are in scope.  The pre-existing
extension is `tensor_cuda/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so`
(2026-07-11 19:04:09 -0400, 7,824,480 bytes); it has not been rebuilt before
this baseline.

### Registered pre-fix fixture

The fixture is a fully solid `uint8[32,32,8]` object at world origin
`(64,-48,12)` and an otherwise empty terrain.  Camera target is its local
center, `(80,-32,16)`, at radius `160`, vertical FOV `20°`, and resolution
`640x480`.  The non-zero origin exercises the object-local ray transform.  The
registered sweep is six azimuths
`{0,15,30,45,60,75}°` times pitches `{2,4,8,15}°`; `2°` and `4°` are the
required sub-`5°` grazing cases.

For every pixel, the expected silhouette is an independently evaluated
float64 ray/closed-box interval from the engine's frozen float32 camera rays.
Only `t_exit > max(t_enter, 0)` is included, so tangencies are excluded.  A
pixel is a leak when that strict analytical-interior mask is true and the
pre-fix float32 object-DDA replica returns background.  The replica retains
the current object path's separate rounded multiply/add, one-coordinate-ULP
`nextafter`, half-open voxel conversion, and low-axis tie policy.

| azimuth | pitch | strict-interior pixels | pre-fix object hits | G-LEAK background pixels |
|---:|---:|---:|---:|---:|
| 0° | 2° | 23,500 | 23,500 | 0 |
| 0° | 4° | 26,224 | 26,224 | 0 |
| 0° | 8° | 31,212 | 31,208 | 4 |
| 0° | 15° | 39,698 | 39,662 | 36 |
| 15° | 2° | 25,715 | 25,715 | 0 |
| 15° | 4° | 28,356 | 28,356 | 0 |
| 15° | 8° | 33,405 | 33,397 | 8 |
| 15° | 15° | 41,800 | 41,769 | 31 |
| 30° | 2° | 28,247 | 22,756 | 5,491 |
| 30° | 4° | 30,826 | 25,236 | 5,590 |
| 30° | 8° | 35,863 | 30,439 | 5,424 |
| 30° | 15° | 44,206 | 39,214 | 4,992 |
| 45° | 2° | 29,068 | 25,600 | 3,468 |
| 45° | 4° | 31,652 | 28,246 | 3,406 |
| 45° | 8° | 36,690 | 33,482 | 3,208 |
| 45° | 15° | 45,012 | 42,178 | 2,834 |
| 60° | 2° | 28,247 | 22,620 | 5,627 |
| 60° | 4° | 30,826 | 25,303 | 5,523 |
| 60° | 8° | 35,863 | 30,515 | 5,348 |
| 60° | 15° | 44,206 | 39,182 | 5,024 |
| 75° | 2° | 25,715 | 25,715 | 0 |
| 75° | 4° | 28,356 | 28,356 | 0 |
| 75° | 8° | 33,405 | 33,397 | 8 |
| 75° | 15° | 41,800 | 41,767 | 33 |

Worst registered angle: `azimuth=60°`, `pitch=2°`, with `5,627` interior
background pixels.  This is the frozen `wo12g_before` receipt angle.

Evidence class: executable CPU float32 replica plus independent float64
silhouette projection.  The local sandbox cannot initialize an NVIDIA driver
(`nvidia-smi` reports it cannot communicate with the driver), so this is a
deterministic pre-build correctness receipt rather than a claimed live-GPU
run.

## 2026-07-12 — robust object-entry sampling

### Root cause and source correction

The fault is confined to `object_dda_first_hit` in the additive OBJECT overlay;
the terrain renderer and standalone `dda_raycast` remain byte-for-byte
untouched.  For an external ray, the old path separately rounded
`origin + direction * t_enter` and applied one coordinate `nextafter`.  At a
grazing face that rounded position can already be multiple coordinate ULPs past
the half-open upper boundary.  A single nudge then still floors to the exterior
cell.  The recorded failing trace is `legacy_floor [32, 2, 7]` for a
`0..31` x-axis, where the robust sample floors to `[31, 2, 7]`.

The object path now advances only its *initial-cell sample* from the exact
entry by

`min(1e-4, 0.5 * (t_exit - start_t))`.

It rejects zero-width external contacts, floors that bounded interior sample,
and uses the same sample time as the lower bound for the first `tMax` values.
The reported hit distance remains the exact `start_t`, so object-versus-terrain
depth ordering is unchanged.  The established lowest-axis exact-tie policy is
explicitly retained.  This is applied at the sole object DDA entry point, which
serves every object in the overlay loop.

### G-LEAK CPU receipt — full 24-pose sweep

The executable CPU replica retained the frozen legacy branch for the `before`
side and uses the new object-only robust branch for `after`.  The independent
silhouette remains float64 ray/closed-box math over frozen float32 camera rays.
All counts are 640x480; a leak is a strict-silhouette pixel with background
depth.

| azimuth | pitch | strict interior | before leaks | after leaks |
|---:|---:|---:|---:|---:|
| 0° | 2° | 23,500 | 0 | 0 |
| 0° | 4° | 26,224 | 0 | 0 |
| 0° | 8° | 31,212 | 4 | 0 |
| 0° | 15° | 39,698 | 36 | 0 |
| 15° | 2° | 25,715 | 0 | 0 |
| 15° | 4° | 28,356 | 0 | 0 |
| 15° | 8° | 33,405 | 8 | 0 |
| 15° | 15° | 41,800 | 31 | 0 |
| 30° | 2° | 28,247 | 5,491 | 0 |
| 30° | 4° | 30,826 | 5,590 | 0 |
| 30° | 8° | 35,863 | 5,424 | 0 |
| 30° | 15° | 44,206 | 4,992 | 0 |
| 45° | 2° | 29,068 | 3,468 | 0 |
| 45° | 4° | 31,652 | 3,406 | 0 |
| 45° | 8° | 36,690 | 3,208 | 0 |
| 45° | 15° | 45,012 | 2,834 | 0 |
| 60° | 2° | 28,247 | 5,627 | 0 |
| 60° | 4° | 30,826 | 5,523 | 0 |
| 60° | 8° | 35,863 | 5,348 | 0 |
| 60° | 15° | 44,206 | 5,024 | 0 |
| 75° | 2° | 25,715 | 0 | 0 |
| 75° | 4° | 28,356 | 0 | 0 |
| 75° | 8° | 33,405 | 8 | 0 |
| 75° | 15° | 41,800 | 33 | 0 |

G-LEAK is therefore zero for every registered pose in the executable CPU
model, including all twelve sub-5-degree cases.  The worst frozen angle remains
`azimuth=60°`, `pitch=2°`.

### Existing-object parity and receipts

The pre-existing G2 two-object fixture and all three G3 occlusion fixtures are
byte-exact between the frozen legacy object oracle and the corrected object
oracle for RGB, depth, and observable winner map.  No parity reference needed
an update: the robust sample changes only previously invalid external-entry
rounding, not correct behavior in those fixtures.

At the worst angle, the regenerated binary P6 receipts are both 640x480
(`921,615` bytes including header):

| receipt | SHA-256 |
|---|---|
| `artifacts/wo12g_before.ppm` | `0fb170d4c9bdb42ae30e43fd0f37fd01247fbe7ba0af6d2bb0ff224984c76c35` |
| `artifacts/wo12g_after.ppm` | `e8adbff1247e947690a682258eef2caf1e366546764e7549ebe4d45e2c30573e` |

They differ in exactly `5,627` pixels, matching the frozen before-leak count;
the after receipt has no strict-interior background pixels.

### Registered battery and host limitation

| gate | command/result |
|---|---|
| CPU G-LEAK + reference regression | `2 passed in 11.72s`; emitted `WO-12G CPU G-LEAK cases=24 worst_pre_fix=5627 post_fix_max=0` |
| Registered selection, with timing enabled | `2 passed, 7 skipped in 11.61s` |
| Terrain engine suite | `PYTHONPATH=. python3 -m pytest tests/test_terrain_render.py -q`: `14 passed, 39 skipped in 34.31s` |
| CUDA G-LEAK, CUDA object parity, and 100-frame +0.1ms timing | all invoked, but skipped by `_require_cuda()` because CUDA allocation fails with `no CUDA-capable device is detected` |

`nvidia-smi` independently reports that it cannot communicate with the NVIDIA
driver.  Thus the CPU gate and terrain suite are green, but this ledger does
not claim a live-GPU G-LEAK, byte-exact CUDA parity, or the `<=0.1900 ms`
objects timing receipt.  Those seven registered CUDA gates must be rerun on a
driver-enabled host after the final build; no performance value is fabricated.

The broad `tests` invocation also cannot serve as a green host receipt here:
GPU-default tests fail at `cudaMalloc`, and its collection includes the
standalone `test_selector_accuracy.py`, which interprets pytest's
`--collect-only` argument as an integer bit width.  Neither issue is in this
object-overlay order's edit scope.
