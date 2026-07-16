# PAINT-CUDA-1 frozen engine gate register

Status: frozen before implementation on branch `paint-cuda-kernels` at
`f181dfd88b18af2432f87cd389714881f3266eec`.

This leg is engine-only.  It does not import or modify ColdCast.  The NumPy
comparison in `tensor_cuda/tests/test_paint_raster.py` is a vendored
transcription of ColdCast's frozen `SoftwareRasterizer.rasterize_clip()`.

## Public contracts

```python
raster_winner_scatter_min(
    pixel_indices_i64, depth_keys_i64, face_ids_i64, pixel_count, *, threads=256
) -> tuple[winner_depth_i64, winner_face_i64]

raster_triangle_winners(
    clip_positions_f32, faces_i64, height, width, *, face_threads=256
) -> tuple[winner_depth_i64, winner_face_i64]

raster_winner_resolve(
    clip_positions_f32, faces_i64, winner_face_i64, *, pixel_threads=256
) -> tuple[resolved_depth_i64, barycentric_f32]

rasterize_clip(
    clip_positions_f32, faces_i64, height, width, *,
    face_threads=256, pixel_threads=256
) -> tuple[face_ids_i64, barycentric_f32, depth_keys_i64]
```

Winner ordering is the lexicographic minimum of signed int32 depth and
positive one-based face ID.  The implementation packs
`(uint32(depth) ^ 0x80000000, uint32(face_id))` into one unsigned 64-bit key
and applies atomic minimum.  Background unpacks as `(INT32_MAX, 0)`.  The
public engine has no uint64 dtype, so packed storage stays internal and the
public winner components are detached int64 tensors.

`threads`, `face_threads`, and `pixel_threads` are launch controls, not
semantic controls.  They must be warp multiples in `[32, 1024]`.  Candidate
depths must fit signed int32, candidate face IDs must be in
`[1, INT32_MAX]`, and candidate pixel indices must be in
`[0, pixel_count)`.

Coverage is exactly the frozen conjunction `alpha,beta,gamma in [0,1]` for
both windings, with no epsilon and no culling.  Shared-edge candidates may be
inclusive in both faces, as in the oracle; the packed winner leaves exactly
one final owner and lower face ID resolves an exact depth tie.

## Registered gates

| Gate | Frozen pass condition |
|---|---|
| G1 | Both windings cover identically.  A two-triangle shared edge has no final-owner gap or duplicate, and inclusive shared-edge ties select lower face ID. |
| G2 | Direct scatter and triangle overlap both honor signed int32 depth ordering; coplanar/exact-depth ties select lower one-based face ID. |
| G3 | Zero-area and edge-on faces emit no winner; exact ties, clipped bboxes, one covering subpixel triangle, and one non-covering subpixel triangle match the vendored oracle. |
| G4 | Face, depth, and barycentric buffers are byte-equal over five reruns at each of two launch configurations and equal across configurations. |
| G5 | Every won interior pixel (3x3 constant-face neighborhood) is bit-exact for face, depth, and barycentrics versus the vendored NumPy oracle.  Boundary divergence is reported per fixture as count and percent of boundary pixels; no boundary threshold is registered here. |
| G6 | The full engine suite adds no failure beyond the standing `test_ext_phase7.py::test_norms_and_conv1d` GroupNorm reshape failure.  `test_selector_accuracy.py` remains excluded because it is a CLI script that parses pytest argv during collection. |

## Performance registration

Report synchronized per-call wall time for the compact fixture and for a
synthetic 1,230,000-face, six-view, 2048x2048 workload.  This leg registers no
performance threshold.

## Execution receipt

Build: PASS, CUDA 12.6.85, Release sm_89.  Generated flags:

```text
CUDA: -O3 -DNDEBUG -std=c++17 --generate-code=arch=compute_89,code=[compute_89,sm_89] -Xcompiler=-fPIC -Xcompiler=-fvisibility=hidden
CXX:  -O3 -DNDEBUG -std=gnu++17 -fPIC -fvisibility=hidden
```

| Gate | Result | Receipt |
|---|---|---|
| G1 | PASS | Both windings have byte-stable final support; the inclusive shared edge has no final-owner gap and its exact ties belong to face 1. |
| G2 | PASS | Direct scatter is invariant to reversed candidate order and 64/256-thread launches; negative signed depth wins and exact depth ties select the lower face. |
| G3 | PASS | Zero-area, edge-on, exact-tie, clipped-bbox, covering-subpixel, and non-covering-subpixel cases all match their frozen expectations. |
| G4 | PASS | Face, depth, and barycentric bytes match across five reruns at `(64,128)` and five reruns at `(256,512)`, and across the two configurations. |
| G5 | PASS | All won 3x3-interior pixels are bit-exact for face, depth, and barycentrics versus the vendored NumPy oracle. |
| G6 | PASS relative to standing failure | Before: 282 passed, 18 skipped, 1 failed. After: 293 passed, 18 skipped, 1 failed. Both failures are `test_ext_phase7.py::test_norms_and_conv1d`; `test_selector_accuracy.py` is the same excluded CLI script before and after. |

Boundary measurement denominator is the fixture's covered boundary pixels
(covered union minus the oracle's won 3x3 constant-face interior).  No
boundary threshold is inferred.

| Fixture | Divergent boundary pixels | Boundary pixels | Percent |
|---|---:|---:|---:|
| shared_edge | 0 | 69 | 0.000000000% |
| reversed_winding | 9 | 73 | 12.328767123% |
| signed_overlap | 0 | 36 | 0.000000000% |
| degeneracies | 0 | 39 | 0.000000000% |
| perspective | 0 | 146 | 0.000000000% |
| random_overlap | 0 | 759 | 0.000000000% |

The reversed-winding delta is the one operation-level impossibility observed:
the input-order barycentric fp32/double sequence changes lattice-edge signs
when vertex order reverses, so it cannot simultaneously reproduce both
input-order oracle boundary masks and provide winding-stable shared-edge
support.  The producer canonicalizes vertex IDs for the coverage conjunction
only.  It retains caller order for depth FFMA and the resolve pass retains it
for signed area, beta/gamma FFMA, double-literal alpha, depth FFMA,
round-toward-zero quantization, perspective divisions, ordered weight sum,
reciprocal, and final multiplies.  This leaves every measured interior
operation bit-exact and exposes the nine boundary pixels above.

Performance (resident inputs, steady-state pooled allocations, explicit CUDA
synchronization, host upload excluded; no target):

| Workload | Result |
|---|---:|
| Synthetic fixture scale, 1,852 faces, 2048x2048, 10 repeats | median 0.539831 ms/call; mean 0.540037 ms; range 0.527647-0.549589 ms |
| Synthetic production scale, 1,230,000 faces, 2048x2048, 6 views | median 0.986711 ms/view; mean 0.985266 ms; range 0.968641-1.005481 ms; six-view total 5.911597 ms |
