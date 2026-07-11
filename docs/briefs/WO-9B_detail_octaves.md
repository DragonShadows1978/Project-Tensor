# WO-9B — scale-agnostic ground detail (in-kernel procedural octaves)

Implementation agent, Project-Tensor (repo root = cwd). Additive
engine order continuing the terrain_render line (WO-7B/8A/8C ledgers
in docs/briefs/). Same rails as those orders; append to
docs/briefs/WO-9B_ledger.md. Build the .so only at the END.

## Operator intent

"Fidelity is pretty low the closer you zoom in — it should be almost
SVG, scale-agnostic: zoom in, it stays high fidelity." Geometry is
1-voxel; close-up surfaces render as featureless smooth patches.

## The work (frozen design)

Add `detail` (0=off default, 1=on) to terrain_render, smooth AND
blocky modes:

1. World-anchored fBm detail field: value-noise octaves derived from
   the existing splitmix64 hash law on lattice points of the WORLD
   coordinates of the hit (never screen space). Base octave
   wavelength 8.0 voxels, lacunarity 2.0, gain 0.5, octaves 0..6
   (finest wavelength 1/8 voxel).
2. Distance LOD: octave k contributes with weight
   w_k = clamp01((d_k − t)/d_k fade) — freeze: octave k is FULL at
   hit distance ≤ 24/2^k voxels and fades linearly to zero by
   2× that distance. No popping: contributions are continuous in
   distance.
3. Application: (a) NORMAL perturbation — gradient of the detail
   field rotates the shading normal by ≤ 22° max at full amplitude;
   (b) fine palette value jitter ±6% at the two finest active
   octaves. Material-aware amplitude scale: sand 0.5×, dirt 1.0×,
   rock 1.4×, scorched 1.2× (table, frozen).
4. Deterministic: same world+hit ⇒ same output, across runs and
   processes.

## Registered gates

- G1 default-off: detail=0 byte-identical to HEAD (all existing
  fixtures pass unmodified).
- G2 NumPy reference parity for detail=1 (same arithmetic class as
  prior orders): ≥99.9% byte-exact, remainder |Δ|≤1 at 640×480 on
  the noise fixture, near AND far cameras.
- G3 determinism: two-process byte-equal frames, detail=1.
- G4 LOD continuity: render a 200-step camera dolly toward a slope;
  assert max per-step frame difference stays below a smooth-motion
  band (no octave pop: per-step mean |Δ| < 3.0, max step change in
  mean |Δ| < 1.0). Report the curve.
- G5 perf: detail=1 total ≤ 6ms @512×512×192 640×480 CLOSE-UP camera
  (10 voxels off a slope, worst case for active octaves). Red is red.
- G6 engine suite green (documented pre-existing exceptions only).
- G7 receipts: artifacts/wo9b_{far,mid,near,macro}.ppm at distances
  {200, 60, 12, 4} with detail=1, same slope, smooth mode — the
  operator judges "SVG-ness" from the ladder.

## Rails

Writable: tensor_cuda source additive, tests additive,
artifacts/wo9b_*.ppm, docs/briefs/WO-9B_ledger.md. APA and existing
op behavior untouchable; Scorch read-only. No subagents/git-write/
network/pip. Report: gate table + LOD curve + timing verbatim.
