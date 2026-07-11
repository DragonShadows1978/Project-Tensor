# WO-9E implementation ledger

## 2026-07-11 — scope, baseline, and frozen implementation choices

This order is limited to additive `tensor_cuda` terrain-renderer/API source,
additive `tensor_cuda/tests/test_terrain_render.py` coverage, the registered
`artifacts/wo9e_{below,edge,overhead}.ppm` receipts, and this ledger.  APA and
all other existing operations are untouched.  No subagents, git writes,
network access, pip, or external service actions were used.

The pre-edit checkout was brief-only HEAD
`88a866b32bac99a5140a870bb91f0ca40bc94456`; the pre-edit extension SHA-256
was `e57a7924020e0faff131245fc576631d9d7f9a340f5e84bfb0c54a0fbf4793e1`.
Before source edits,
`python3 -m pytest tests/test_terrain_render.py -q` reported
`8 passed, 28 skipped in 23.75s`.  The skips were the registered CUDA gates:
the restricted runner reports
`cudaMalloc failed: no CUDA-capable device is detected`.  `nvidia-smi`
nevertheless identified an NVIDIA GeForce RTX 4070 SUPER with driver 595.71.05.
An approval request for the unsandboxed pre-edit GPU pass was rejected because
the host approval quota was exhausted; that restriction was not bypassed.

The precise implementation registered from the supplied design is:

1. `grounding` is a strict integer selector.  Zero is the default and launches
   only the literal established blocky/smooth/cached/detail terrain kernels.
   One adds a single output-side grounding kernel before the existing optional
   object overlay.
2. A terrain hit is a cut-face only when its true depth equals the recomputed
   external AABB `start_t` for the same frozen camera ray.  Low-axis tie policy
   matches traversal.  Both x and y walls qualify; only the lower z wall
   (`face_sign=-1`, z=0 underside) qualifies.  z=max remains landscape.  An
   entry through boundary air followed by an interior crossing cannot satisfy
   the equality and therefore remains ordinary terrain.
3. Bedrock uses face-flat RGB `(38,34,32)` and only the half-weight established
   diffuse term:
   `shade = 0.5 * clamp(dot(normal,-light),0.35,1)`.  Conversion is the existing
   round-to-nearest u8 rule.  Palette/hash jitter, detail octaves, and AO are
   absent.
4. On a terrain miss, a positive finite ray intersection with `z=z_horizon`
   becomes ground only when its x/y point lies outside the half-open terrain
   footprint.  The plane is two-sided and faces the ray origin.  Its frozen
   fog is `f=clamp((t-fog_start)/(fog_full-fog_start),0,1)` followed by the
   exact linear `lerp(bedrock, black_sky, f)`.  The output depth is the true
   plane `t`; inside-footprint, behind-camera, parallel, and above-horizon
   misses remain black with depth -1.

Public parameters are appended after the existing `objects` argument to retain
positional compatibility: `grounding=0`, `z_horizon=0.0`,
`fog_start=600.0`, and `fog_full=2400.0`.  Parameters are finite float32 values
and the fog range must satisfy `0 <= fog_start < fog_full`.

## Pre-build oracle and artifact registration

The independent NumPy grounding oracle, exact eight-case face classifier, and
receipt generator were run before the sole final build:

`2 passed in 1.51s`

The classifier covers x=0/x=max, y=0/y=max, z=0 underside, excluded z=max,
an external boundary-air then interior-solid crossing, and an origin-inside
interior crossing.  The three 320x240 binary P6 receipts use one bounded
160x160x64 mountain slab, `z_horizon=8`, and fog range 120/480.  Their masks
were:

| view | bedrock pixels | plane pixels | sky pixels |
|---|---:|---:|---:|
| below | 35655 | 26036 | 10250 |
| edge | 19400 | 22104 | 25405 |
| overhead | 0 | 0 | 0 |

The overhead grounded RGB and depth arrays are byte-identical to the ordinary
NumPy terrain render.  Each artifact is 230415 bytes.  Initial receipt hashes
will be refreshed below after the final pre-build oracle pass, because the
bedrock arithmetic was subsequently pinned to the literal half-weight diffuse
product described above.

The refreshed CPU-only pre-build command reported `4 passed in 1.46s` with no
warnings.  Final pre-build SHA-256 receipts are:

- `wo9e_below.ppm`:
  `80f987af4ef0143fc1c93123f350ec0879fe95577a760444b1f858c21e4566f8`
- `wo9e_edge.ppm`:
  `12572ce71d7b2f6f362a98df3a0a462be8dbbadb82903d839597f261a4383ea6`
- `wo9e_overhead.ppm`:
  `ca6c7b51ce36f46f2c3e62b01e061975cb8c704048fe3081d2c26eff5e46fde3`

With all source, oracle, tests, receipts, and ledger setup complete, the final
pre-build terrain suite reported `12 passed, 37 skipped in 22.93s`.  All 37
skips were CUDA-gated tests or opt-in timing tests under the same restricted
driver condition.  No build had been run at this point.
