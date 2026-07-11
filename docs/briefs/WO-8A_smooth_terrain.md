# WO-8A — smooth isosurface terrain mode in terrain_render (engine)

You are an implementation agent on Project-Tensor (repo root = cwd).
Continue the WO-7B op (see docs/briefs/WO-7B_render_kernel.md and
docs/briefs/WO-7B_ledger.md; commit 9d808f4). Additive-only, same
rails: existing ops (APA absolutely, dda_raycast) untouchable; append
scope/results to docs/briefs/WO-8A_ledger.md. Operator intent: "the
ground is still blocky and not smooth — the original was very smooth
graphics, that was part of what made the game special." Sim stays
voxels; SMOOTHNESS IS RENDER-SIDE ONLY.

## The work

Add `surface_mode` to `terrain_render`: `"blocky"` (default — the
current path, byte-identical) and `"smooth"`:

1. Density field: trilinear interpolation of occupancy (materials !=
   0 → 1.0) sampled at voxel centers. Iso-level 0.5.
2. March: keep the existing DDA through empty space, but when the ray
   enters any voxel whose 2×2×2 interpolation neighborhood contains
   both states, switch to fixed steps ≤ 0.5 voxel sampling the
   trilinear density; on the first sample ≥ 0.5, refine the crossing
   with ≥ 4 secant/bisection iterations. Exiting the mixed band with
   no crossing resumes DDA.
3. Normal = normalized central-difference gradient of the trilinear
   density at the hit (h = 0.5 voxel); this REPLACES the occupancy
   normal in smooth mode. Degenerate gradient falls back to the
   blocky normal.
4. Material/palette/AO: from the nearest solid voxel to the hit
   point; shading constants remain the canonical WO-7B spec.
5. Depth = distance to the refined crossing; miss semantics
   unchanged.

## Registered gates

- G1 blocky regression: surface_mode="blocky" (and the default call
  with no argument) byte-identical to HEAD — existing terrain tests
  + WO-7B parity fixtures pass unmodified.
- G2 smooth correctness (NumPy reference in tests, same style as
  WO-7B): (a) flat-plane fixture: hit z within 0.1 voxel of the
  analytic isosurface everywhere, normals within 2° of vertical;
  (b) sphere fixture (radius ≥ 12): surface distance error ≤ 0.15
  voxel RMS, normals within 5° of radial; (c) kernel vs NumPy smooth
  reference at 640×480: ≥ 99.9% byte-exact, remainder |Δ| ≤ 1,
  depth rel err ≤ 1e-4.
- G3 perf (TC_RUN_TERRAIN_TIMING=1 harness, extended with
  surface_mode): smooth mode total ≤ 8ms mean at 512×512×192
  @640×480 (blocky is 1.64ms; registered budget 5×). Red is red.
- G4 engine suite green (documented pre-existing exceptions only).
- G5 eyeball receipt: test-generated PPM pair blocky-vs-smooth of a
  noise-terrain fixture (same camera) written to
  artifacts/wo8a_{blocky,smooth}.ppm for the operator.

## Rails

Writable: tensor_cuda source additive (terrain.cu, bindings,
__init__.py), tensor_cuda/tests/test_terrain_render.py (additive),
artifacts/wo8a_*.ppm, docs/briefs/WO-8A_ledger.md. READ-ONLY:
everything else, all Scorch paths. No subagents/git-write/network/
pip. Report: gate table verbatim + stage timings.
