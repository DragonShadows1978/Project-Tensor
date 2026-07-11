# WO-8C — density filter radius knob for smooth terrain (engine)

You are an implementation agent on Project-Tensor (repo root = cwd).
Continuation of WO-8A (docs/briefs/WO-8A_{smooth_terrain,ledger}.md,
commit 8bf5a1e). Additive-only, same rails; append to
docs/briefs/WO-8C_ledger.md.

## Why

Operator eyeball of smooth mode: sub-voxel facets gone, but
MULTI-VOXEL slope terracing (contour banding) survives — the 2×2×2
trilinear kernel can't melt staircase structure wider than a voxel.
Registered knob: pre-filter the occupancy field.

## The work

Add `density_filter` to terrain_render's smooth mode: 0 (current
behavior, default), 1 (3³ box filter over occupancy before trilinear
sampling), 2 (5³). Implement WITHOUT materializing a filtered copy of
the whole grid per frame if avoidable (sum-on-the-fly in the mixed
band, or a cached filtered field invalidated by grid revision — your
call; document the choice and its memory cost in the ledger).
Iso-level stays 0.5; normals from the filtered field's gradient;
material/AO from nearest solid voxel as in WO-8A. Blocky mode and
filter=0 byte-identical to HEAD.

## Registered gates

- G1: filter=0 byte-identical to WO-8A smooth (existing fixtures
  green unmodified); blocky untouched.
- G2: NumPy reference extended with the filter; kernel parity at
  640×480 noise fixture for filter∈{1,2}: ≥99.9% byte-exact,
  |Δ|≤1 remainder.
- G3: geometry sanity — 45° ramp fixture: max deviation of the
  filtered isosurface from the ideal plane DECREASES monotonically
  with filter radius (that is the whole point; report the three
  values).
- G4 perf: filter=2 total ≤ 12ms mean @512×512×192 640×480
  (registered; filter=0 is 2.84ms). Red is red.
- G5: A/B/C PPMs of one mountain-like fixture at filter 0/1/2 →
  artifacts/wo8c_filter{0,1,2}.ppm for the operator.
- G6: engine suite green (documented pre-existing exceptions only).

## Rails

Writable: tensor_cuda source additive, tensor_cuda/tests/
test_terrain_render.py (additive), artifacts/wo8c_*.ppm,
docs/briefs/WO-8C_ledger.md. Everything else read-only (APA
absolutely; all Scorch paths). No subagents/git-write/network/pip.
Report: gate table + timing table verbatim.
