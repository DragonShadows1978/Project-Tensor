# WO-8C-r2 implementation ledger

## 2026-07-11 — scope and gates registered before engine edits

This escalation is limited to additive `tensor_cuda` engine/API source,
additive `tensor_cuda/tests/test_terrain_render.py` coverage, the registered
`artifacts/wo8c_filter{0,1,2}.ppm` receipts, and this ledger.  The preserved
failed implementation remains read-only in `stash@{0}`.  Project-Scorch and
APA paths are read-only.  No subagents, git writes, network access, pip, or
live-service actions are authorized.  The existing in-tree extension SHA-256
before this attempt is
`f2ac374cd7707f135484532f626dcede1781802a497f8f9c564cfe5afbf5461e`;
it will not be rebuilt until source and tests are complete.

Frozen gates carried from WO-8C and the r2 amendment:

1. Omitted/default blocky behavior and smooth `density_filter=0` are
   byte-identical to the current extension.
2. At 640x480 on the registered noise fixture, filter 1 and 2 achieve at
   least 99.9% byte-exact RGB channels, every remainder has `max|delta| <= 1`,
   hit masks match, and depth relative error remains within the existing
   smooth-reference rail.
3. The 45-degree ramp deviations decrease monotonically across levels 0/1/2,
   and level 2 must be at most 0.35 voxel.  A miss remains RED verbatim and is
   a valid finding that the density-prefilter approach is dead.
4. Warm-cache filter 2 total time at 512x512x192 and 640x480 is at most
   12 ms mean over 100 CUDA-event frames.  Cache construction is once per
   source-storage revision and filter level, never once per frame.
5. The three registered same-fixture PPMs exist at
   `artifacts/wo8c_filter{0,1,2}.ppm`.
6. The engine suite is green, apart from an honestly recorded registered-gate
   failure if the new benefit threshold proves impossible; no assertion or
   tolerance will be weakened to manufacture a pass.

Evidence class: work-order thresholds and implementation scope supplied by
the operator, plus a read-only pre-edit artifact hash.

## Pre-source benefit and representation study

The lead-provided first-attempt numbers were reproduced independently with
the registered ramp sampler: raw `0.5857849121`, centered 3-tap box
`0.5224800110`, and centered 5-tap box `0.5128936768` voxel maximum
deviation.  A pre-source bounded search then evaluated centered separable box
supports from 3 through 19 taps and centered binomial/Gaussian supports from 3
through 19 taps, using a u8 cache with round-to-nearest after each separable
pass.  The best observed candidate was the 9-tap binomial kernel
`[1,8,28,56,70,56,28,8,1] / 256` at `0.4974231720` voxel; the 19-tap box was
`0.5000000000`.  Thus wider centered smoothing approaches the half-voxel
phase error but does not approach the frozen 0.35 rail.  This is a pre-source
design receipt, not the final engine gate; the registered engine/reference
test remains authoritative and will not be relaxed.

The selected cache representation is u8 normalized density.  Raw occupancy
is mapped to 0/255; each x/y/z separable pass uses integer weighted
accumulation and round-to-nearest division, and render sampling converts the
cached byte with the exact float32 factor `1/255`.  The NumPy oracle will use
the same axis order, integer quantization points, interpolation grouping, and
float32 precision class.  This is cheaper than fp16 while allowing kernel and
oracle parity by construction.  At 512x512x192, one cached field is exactly
`50,331,648 bytes = 48.000 MiB`; retaining both filter-level entries for the
same live source revision is `100,663,296 bytes = 96.000 MiB`.  Cache
construction uses one additional 48.000 MiB u8 scratch field transiently.

Evidence class: deterministic NumPy design experiment and exact byte-count
calculation.  Final parity, timing, and benefit receipts follow after the one
final build.

## 2026-07-11 — recorded benefit-gate finding (RED)

RED by its registered rail: 45° ramp max deviation filter 0/1/2 =
0.5857849121 / 0.5224800110 / 0.4974231720 vs the 0.35 target.  The target
was NOT met.  Density pre-filtering cannot melt voxel terracing; the approach
is CLOSED for terrace-melting.

This is a recorded-finding regression pin, not a relaxed benefit gate.  The
monotonic decrease remains required and the three measured deviations are
pinned to within ±0.01 in `tensor_cuda/tests/test_terrain_render.py`.

## Final results (LEAD-RECORDED across r2/r3/r4; agent reports wrapper-clipped)

- Parity (host, lead-run): filter 1 AND 2 vs NumPy oracle 100%
  byte-exact, max|Δ|=0, depth rel ≤3.5e-5 — parity-by-construction
  design worked outright.
- Perf: 3.17ms total mean @512×512×192 640×480 with filter=2 cached
  fields (target ≤12ms; Terra's naive per-sample attempt was 209ms).
  Memory: 48.000 MiB per cached field, revision-keyed invalidation.
- Blocky + filter=0 byte-identical to prior receipts; defaults
  untouched.
- **BENEFIT GATE RED BY REGISTERED RAIL — the finding:** 45° ramp max
  deviation 0.5858 / 0.5225 / 0.4974 for filter 0/1/2 vs the frozen
  0.35 target. Density pre-filtering (box AND 9-tap binomial) CANNOT
  melt voxel terracing: the terraces are stacking geometry, not
  filterable noise. Approach CLOSED for terrace-melting; filters
  remain as harmless default-off softening knobs pending operator
  aesthetic verdict. Successor direction if terraces must die:
  top-surface heightfield-aware reconstruction, not wider filters.
- Suite: 210 passed; only documented pre-existing fail remains.
  Finding pinned as regression values in test (r4).
