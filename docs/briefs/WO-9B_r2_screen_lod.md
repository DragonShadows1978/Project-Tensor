# WO-9B-r2 — screen-subtense LOD + judgeable receipts (continuation)

Implementation agent, Project-Tensor (repo root = cwd), continuing
WO-9B in the same tree (committed 98676ae). Read docs/briefs/
WO-9B_{detail_octaves,ledger}.md. Append to the ledger. Build LAST.

## Lead verdict on WO-9B

Mechanics all green (parity byte-exact, deterministic, 0.80ms
close-up). Aesthetic RED, cause = the LEAD BRIEF's frozen LOD
constants (acknowledged): "octave k full at <=24/2^k voxels" turns
all detail off beyond 48 voxels and requires sub-voxel proximity for
fine octaves — receipts show artificial scale-mesh at mid range and
soft blobs at near range, inverted from the operator's intent
("zoom in, it stays high fidelity"). Receipt fixture also back-lit
(unjudgeably dark).

## Amended design (frozen)

1. SCREEN-SUBTENSE LOD replaces the distance table: octave k is
   active where its wavelength lambda_k = 8/2^k voxels projects to
   >= 2.0 pixels at the hit: weight_k = clamp01(px(lambda_k)/2.0 - 1.0)
   capped at 1, where px(lambda) = lambda * (height/(2*tan(fov/2)))/depth.
   Coarsest octave (k=0) additionally never drops below weight 0.35
   (anchors the look at all ranges). Continuous in depth — no pops.
2. Amplitude/application constants unchanged (18-deg-capped normal
   rotation as implemented, +-6% fine palette jitter on the two
   finest ACTIVE octaves, material table unchanged).
3. Receipts regenerated JUDGEABLE: light direction toward the slope
   (front-lit), and a second ladder on the REAL mountains preset
   (import via the committed fixture-generation path used by earlier
   wo8a receipts) at distances {200, 60, 12, 4}:
   artifacts/wo9br2_{far,mid,near,macro}.ppm.

## Gates

- G1: detail=0 byte-identical to HEAD (existing fixtures unmodified).
- G2: parity vs updated NumPy reference, near+far, >=99.9% byte-exact
  |delta|<=1 remainder.
- G3: two-process determinism.
- G4: LOD continuity dolly — same smooth-motion band as WO-9B G4.
- G5: perf <=6ms close-up @512 (prior: 0.80ms; headroom exists).
- G6: SUBTENSE law test: at fixed fov/height, assert the exact set of
  active octaves at depths {4, 12, 60, 200} matches the formula.
- G7: engine suite green (documented pre-existing exceptions only).

## Rails

As WO-9B. Report: gate table + active-octave table at the four
depths + timing verbatim.
