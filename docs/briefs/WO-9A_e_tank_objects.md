# WO-9A-e — voxel objects in terrain_render (engine half of in-scene tanks)

Implementation agent, Project-Tensor (repo root = cwd). Additive
engine order continuing the terrain_render line (ledgers in
docs/briefs/WO-7B/8A/8C/9B*). Same rails; append to
docs/briefs/WO-9A_e_ledger.md. Build the .so only at the END.

## Purpose

Scorch's tanks are billboard sprites — structurally dead at ground
level. They become first-class ray-traced objects. The CONSUMER
pre-bakes each tank (hull+turret+barrel at current quantized aim,
wreck variants, livery colors) into a small AXIS-ALIGNED u8 voxel
grid — the engine does NO rotations.

## The op change (frozen)

`terrain_render(..., objects=None)`: objects is a list (max 16) of
(grid_u8_device [X,Y,Z], origin_f32x3 world min-corner, palette_u8
[N,3] device). Per ray: after (or interleaved with) terrain traversal,
intersect the object's AABB; if entered, DDA the object grid in its
local frame (integer voxels, origin offset, same step semantics as
terrain); nearest hit across terrain+objects wins per pixel (depth
compare). Object shading: BLOCKY face normals (crisp machines),
diffuse+AO' where AO for object voxels uses the OBJECT grid's own
3^3 occupancy, palette from the object's own palette (no jitter, no
detail octaves on objects). Depth output = winning hit distance.
Misses unchanged. detail/density_filter/surface_mode apply to
TERRAIN only.

## Registered gates

- G1: objects=None (and empty list) byte-identical to HEAD — all
  existing fixtures pass unmodified.
- G2: NumPy reference parity for the objects path (2 synthetic
  objects incl. one overlapping terrain): >=99.9% byte-exact,
  |delta|<=1 remainder, depth exact-class.
- G3 occlusion correctness: fixture where an object is (a) fully in
  front of terrain, (b) half-buried behind a ridge, (c) fully
  occluded — per-pixel winner asserted against the reference; the
  half case must show partial visibility both directions.
- G4 perf: 4 objects of 24^3 @512x512x192 640x480: total <= +1.5ms
  over the no-objects smooth baseline. Red is red.
- G5: engine suite green (documented pre-existing exceptions only).
- G6 receipts: artifacts/wo9ae_{front,ridge,far}.ppm — a synthetic
  tank-ish object (use a simple hull+turret blob) at 12 voxels, half
  behind a ridge, and at 200 voxels.

## Rails

Writable: tensor_cuda source additive, tests additive,
artifacts/wo9ae_*.ppm, docs/briefs/WO-9A_e_ledger.md. APA + existing
op behavior untouchable; Scorch paths READ-ONLY (render/tank_model.py
may be READ for shape conventions). No subagents/git-write/network/
pip. Report: gate table + timing verbatim.
