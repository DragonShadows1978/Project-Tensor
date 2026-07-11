# WO-9A-e implementation ledger

## 2026-07-11 — scope, baseline, and frozen implementation split

This order is limited to additive `tensor_cuda` renderer/API source, additive
`tensor_cuda/tests/test_terrain_render.py` coverage, the registered
`artifacts/wo9ae_{front,ridge,far}.ppm` receipts, and this ledger.  APA and
Project-Scorch paths are untouched; `render/tank_model.py` was read only for
axis and scale conventions.  No subagents, git writes, network access, pip, or
live-service actions are in scope.

The pre-edit extension SHA-256 is
`0fec4dcedce8e1f79e73b90ba2d88223617f537576b3df55d4a56d98c60b4258`.
Before source edits, `python3 -m pytest tests/test_terrain_render.py -q`
reported `7 passed, 20 skipped in 20.30s`; all skips were CUDA gates because
the restricted sandbox reported `cudaMalloc failed: no CUDA-capable device is
detected`.  This is a baseline receipt, not a substitute for the final
GPU-enabled gate pass.  The extension will be rebuilt only once, after source,
oracle, receipt generation, and ledger setup are complete.

The no-object path remains the literal established five-argument C++ op and
existing terrain launch selection.  A non-empty object list takes an overload
which first invokes that same terrain path, then adds one CUDA overlay launch.
The launch receives a fixed-size by-value descriptor block (maximum 16), so it
does not allocate or upload a descriptor tensor per frame.  Each ray translates
the camera origin into each object's local frame, uses the established half-open
AABB/Amanatides-Woo policy, and replaces the terrain result only for a strictly
nearer object hit.  Object normals are the DDA entry face; AO counts the 26
neighbors in that object's own grid; color comes directly from that object's
palette with diffuse and AO only.  Terrain surface, density-filter, and detail
selectors therefore cannot affect object geometry or shading.

Evidence class: operator-supplied frozen order, read-only pre-edit artifact
hash and test receipt, plus source-level implementation registration.  Final
parity, occlusion, timing, suite, and artifact receipts follow after the sole
build.

## Pre-build oracle and receipt registration

The independent NumPy compositor was exercised before the extension build.
At the 192x144 two-object G2 fixture its winner counts are: miss `12576`,
terrain `14020`, object 1 `746`, object 2 `306`.  Object 2 physically overlaps
occupied terrain voxels; `231` of its `537` candidate rays are terrain-occluded
while the remaining `306` win, so the fixture covers both per-object palette
selection and depth competition rather than merely placing two unobstructed
AABBs in the frame.

The 192x144 G3 oracle counts are:

| case | object candidate rays | object wins | terrain wins over candidate |
|---|---:|---:|---:|
| front | 1027 | 1027 | 0 |
| half-ridge | 504 | 252 | 252 |
| fully occluded | 504 | 0 | 504 |

Thus the half case is an exact split in the registered test resolution, and
the fully hidden case still has a non-empty projected object candidate mask.
The actual CUDA test derives an object-winner bit from the depth change against
the same CUDA terrain-only frame and requires that bit to equal the oracle map
at every pixel.

The CPU-independent receipt test completed before build as `1 passed in
3.65s`; the three binary P6 files are 320x240 and 230415 bytes each.  Their
SHA-256 receipts are:

- `wo9ae_front.ppm`:
  `d633a6906645e017e72bbb230ad8dc0d6ced5e8905555ebcb2fd4a109784dafc`
- `wo9ae_ridge.ppm`:
  `70a309e8dae2bf15961797e722826a56b0efac2208f3f0bf52803032e2f9ab29`
- `wo9ae_far.ppm`:
  `685c4a5091036aa43656beacbba9a3890ec96ae44f77115051ed9d2663ed2c4e`

The front AABB begins exactly 12 world voxels from its camera along the view
axis; the far AABB begins exactly 200.  At receipt resolution the object winner
counts are front `19693`, ridge `675` (with `736` projected candidate rays
hidden by terrain), and far `2945`.  A post-edit/pre-build terrain test pass
reported `8 passed, 26 skipped in 23.44s`; the added pass is the receipt test,
and CUDA-dependent cases remain skipped for the same restricted-device reason.

Evidence class: executable NumPy oracle, exact integer winner masks, binary
artifact hashes, and pre-build pytest output.
