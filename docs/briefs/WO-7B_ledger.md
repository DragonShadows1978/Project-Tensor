# WO-7B ledger — GPU-resident terrain render kernel

## Scope

Implemented only the additive `tensor_cuda.terrain_render` path in
Project-Tensor. The new path accepts a resident `uint8` voxel grid and resident
`uint8 (N,3)` palette, derives the frozen Scorch camera basis in O(1) host work,
then performs per-pixel ray generation, DDA traversal, gradient/AO/palette
shading, and RGB/depth writes in one CUDA launch. Existing `dda_raycast` and
all APA kernels were left unmodified.

The public Python call is:

```python
terrain_render(materials_u8_device, cam, light, palette, consts=None)
```

`cam` supplies position, look-at, world-up, vertical FOV, width, and height;
`light.direction` is normalized and interpreted as the light travel direction.
`consts.max_steps` is optional; all shading constants are frozen and rejected
if callers try to override them. Misses produce black RGB and depth `-1`.

## Registered results

Pending runtime gate execution on a stable CUDA driver. Build and static/API
validation receipts will be appended after the full verification pass.

## Registered results (LEAD-RECORDED 2026-07-11: agent sandbox had no GPU; wrapper clipped its report — lead built and ran all gates on host)

- Build: ./build.sh clean, exit 0.
- G1 parity: test_terrain_render_full_spec_numpy_parity_640x480 PASSED
  (kernel vs shipped NumPy reference); output contract (misses black,
  depth −1, frozen-constants rejection) PASSED.
- G2: full existing suite 197 passed — dda_raycast fixtures untouched
  and green. Pre-existing exceptions only: test_ext_phase7 norms fail
  (documented since WO-T1) and test_selector_accuracy.py collection
  error (CLI script, not a pytest module; predates this order).
- G3 stage timing (TC_RUN_TERRAIN_TIMING=1, 100 frames, 640×480):
  256×256×96: kernel 0.589ms + D2H 0.255ms = 0.898ms mean — GREEN.
  512×512×192: kernel 1.328ms + D2H 0.248ms = 1.638ms mean — GREEN
  (registered target ≤16ms beaten 9.8×; ≈610 fps terrain pass vs the
  current CPU pipeline's ~40ms flat / ~143ms shaded).
- Out-of-rails check: core/__init__.py in the tree is 2026-07-08
  Trinity-era leftover, not this order's; excluded from this commit.
