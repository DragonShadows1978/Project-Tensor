# WO-7B — GPU-resident terrain render kernel (tensor_cuda, engine-only)

You are an implementation agent on Project-Tensor (repo root = cwd).
This is an ENGINE order for Project-Scorch's renderer (a consumer of
this library) — but you touch ONLY this repo. Follow the WO-T1 pattern
(commit 55619a6, the `dda_raycast` op): additive-only public op,
pybind11 binding, engine tests. Existing ops (especially all APA
kernels) are untouchable. Append scope/results to your work in
docs/briefs/WO-7B_ledger.md (this repo has no central ledger for
Scorch orders; create that file).

## Why (measured)

Scorch's frame at 512×512×192 @640×480 is ~40ms, and the GPU DDA is a
minority of it: CPU NumPy builds+normalizes 307k rays every frame
(~11ms), shades hits on CPU (~17ms), and ships multi-MB hit buffers
D2H. The fix: one kernel that generates rays, marches, shades, and
returns the finished RGB frame.

## The op

`terrain_render(materials_u8_device, cam, light, palette, consts) ->
(rgb_u8 HxWx3, depth_f32 HxW)` where:

- `materials` is the device-resident u8 voxel grid (caller uploads
  once and reuses across frames, as with dda_raycast).
- `cam` = position, look_at, world_up, vertical_fov_degrees, width,
  height. RAY CONVENTION IS FROZEN and must match Scorch's
  `render/camera.py:camera_rays` exactly: pixel centers at i+0.5,
  screen_x = ((i+0.5)/W*2−1)*half_w, screen_y = (1−(j+0.5)/H*2)*half_h,
  half_h = tan(fov/2), half_w = half_h*W/H, basis from
  forward/right/up cross products, directions normalized. (Read that
  file read-only at /mnt/ForgeRealm/Project-Scorch/render/camera.py
  for the reference; do NOT modify anything there.)
- March: DDA identical in traversal semantics to the existing
  dda_raycast op (reuse its device logic; refactor-to-share is
  allowed if dda_raycast's outputs stay bit-identical — prove with
  its existing tests).
- On hit, shade with the FROZEN spec (same as Scorch WO-7A):
  1. normal = normalize(central-difference occupancy gradient over
     the 3^3 neighborhood); fallback to face normal when |g| < 1e-6.
  2. diffuse = clamp(dot(normal, -light_dir), ambient_floor, 1.0),
     ambient_floor = 0.35.
  3. AO: occ_frac = occupied/(26 neighbors); ao = clamp(1 − 0.6*occ_frac,
     0.4, 1.0). color *= diffuse * ao.
  4. palette jitter: h = splitmix64(x*0x9E3779B97F4A7C15 ^ y*0xBF58476D1CE4E5B9 ^ z*0x94D049BB133111EB)
     (64-bit, exact constants); jitter value ±8% and channel mix ±4%
     from h's top bits, applied to the per-material base color passed
     in `palette`. Deterministic: same (x,y,z,material) → same color.
- Miss: rgb = (0,0,0), depth = −1. Sky/fog stay the CALLER's job
  (they composite on CPU using the depth channel) — do not implement
  fog.

## Gates (registered NOW; red is red)

- G1 parity: ship a NumPy reference implementation of the full spec
  (ray gen + march + shade) in the test harness; kernel vs reference
  on ≥3 synthetic grids + one 256×256×96 noise-terrain fixture at
  640×480: ≥99.9% of pixels byte-exact per channel, remainder
  |Δ| ≤ 1, depth max rel err ≤ 1e-5. (WO-T1's parity precedent.)
- G2 dda_raycast unchanged: its existing tests pass bit-identical.
- G3 stage timing harness (CUDA events): report kernel / D2H / total
  for 256×256×96 and 512×512×192 grids at 640×480, mountains-like
  fixture, 100 frames. TARGET (registered): total ≤ 16ms at 512
  (≥60fps). If red, report the measured wall verbatim with the stage
  breakdown — a red target with a clean breakdown is a valid result.
- G4 engine suite green (pre-existing documented failures excepted,
  named explicitly in your report).

## Rails

Writable: tensor_cuda source (new op + binding + tests, additive),
docs/briefs/WO-7B_ledger.md. READ-ONLY: every existing op's behavior
(APA absolutely), /mnt/ForgeRealm/Project-Scorch/** (reference reads
only). No subagents, no git writes, no network, no pip. Report: gate
table verbatim, stage-timing table, files added.
