# WO-9E — bedrock cut-faces + horizon (kill the floating-island look)

Implementation agent, Project-Tensor (repo root = cwd). Additive
engine order on terrain_render (ledgers docs/briefs/WO-7B..9A_e).
Same rails; append to docs/briefs/WO-9E_ledger.md. Build LAST.

## Operator problem

Free camera exposes the world slab from outside/below: boundary
cross-sections shade like landscape (sand bands, gradient lighting)
and the map floats in empty sky — no horizon, no under-world story.

## The work (frozen)

Add `grounding` (0=off default, 1=on) to terrain_render:

1. BEDROCK CUT-FACES: a hit whose face lies ON the grid boundary
   (x=0/max, y=0/max, z=0 planes — i.e., the ray entered the solid
   through the domain wall, not through an interior air-solid
   crossing) shades as BEDROCK: flat dark basalt (RGB 38,34,32),
   diffuse only at 0.5 weight, NO palette jitter, NO detail octaves,
   NO AO. Interior surfaces unchanged.
2. HORIZON GROUND PLANE: rays that MISS the grid but would cross the
   plane z = z_horizon (param, caller passes e.g. 8.0) outside the
   domain hit an infinite ground plane: shade as bedrock modulated by
   a distance fog ramp toward the sky color (fog_start/fog_full
   params, e.g. 600/2400 in voxels; exact linear lerp, frozen).
   Rays above the horizon plane keep sky-miss semantics (rgb 0,
   depth -1); plane hits return their true depth so the consumer's
   fog/compose stays consistent.

## Registered gates

- G1: grounding=0 byte-identical to HEAD (all fixtures unmodified).
- G2: NumPy reference parity for grounding=1: >=99.9% byte-exact,
  |delta|<=1, on a fixture viewed from outside-below AND from a
  normal overhead camera.
- G3 classification: synthetic fixture asserting exact face
  classification — interior crossings NEVER bedrock; domain-wall
  entries ALWAYS bedrock; z=0 underside bedrock.
- G4 perf: grounding=1 within +0.5ms of baseline @512 (it is a
  branch per hit + plane intersect per miss).
- G5: engine suite green (documented pre-existing exceptions only).
- G6 receipts: artifacts/wo9e_{below,edge,overhead}.ppm — the
  outside-below view that exposed the problem, an edge-on view, and
  a normal overhead view (must look unchanged).

## Rails

Writable: tensor_cuda source additive, tests additive,
artifacts/wo9e_*.ppm, docs/briefs/WO-9E_ledger.md. APA + existing op
behavior untouchable. No subagents/git-write/network/pip. Report:
gate table + timing verbatim.
