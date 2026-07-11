# WO-9B implementation ledger

## 2026-07-11 — scope, frozen choices, and pre-edit receipt

This order is limited to additive `tensor_cuda` renderer/API source, additive
`tensor_cuda/tests/test_terrain_render.py` coverage, the registered
`artifacts/wo9b_{far,mid,near,macro}.ppm` receipts, and this ledger.  APA,
existing terrain behavior, and Project-Scorch remain untouched.  No
subagents, git writes, network access, pip, or live-service actions are in
scope.  The pre-edit extension SHA-256 is
`ebe4a7cf9a1f41789c7b67777b3ec044cd02c4156c95ff47891aa4fbaeddfda4`.
The extension will be rebuilt once, only after source, oracle, and receipt
tests are complete.

Frozen implementation choices for the supplied design:

1. `detail` is a strict integer selector: 0 is the default and dispatches to
   the untouched WO-7B/8A/8C kernels; 1 dispatches to detail-only kernels for
   blocky, smooth, and cached-smooth modes.
2. Each octave uses world-coordinate 3D value noise.  Lattice values derive
   from the existing unsigned-coordinate xor/multiply splitmix64 law; cubic
   Hermite interpolation supplies a continuous analytic gradient.  Wavelength
   is `8 / 2^k`, gain is `0.5^k`, and `k=0..6`.
3. The precise LOD interpretation is `full=24/2^k`,
   `weight=clamp((2*full-hit_distance)/full, 0, 1)`: full through `full`, then
   linearly continuous to zero at `2*full`.
4. Material amplitude multipliers follow the read-only Scorch material table:
   sand/material 3 = 0.5, dirt/1 = 1.0, rock/2 = 1.4, scorched/4 = 1.2
   (all other palette rows use 1.0).  The
   normal perturbation is tangent-projected and capped at 22 degrees after
   this scaling.  The two finest positive-LOD octaves supply a world-anchored
   palette-value jitter capped at plus/minus 6 percent after the same scaling.

Evidence class: operator-supplied work order plus read-only pre-edit artifact
hash.  Final gate receipts follow after the one final build.
