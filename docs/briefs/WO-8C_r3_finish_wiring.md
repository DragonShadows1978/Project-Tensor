# WO-8C-r3 — finish the r2 wiring (continuation, same tree)

Implementation agent, Project-Tensor, repo root = cwd. CONTINUATION of
WO-8C-r2 (docs/briefs/WO-8C_r2_density_filter_sol.md — its amendments
and ALL its gates still govern; read docs/briefs/WO-8C_r2_ledger.md,
which contains YOUR OWN registered design: centered 3-tap box for
filter 1, 9-tap binomial for filter 2, u8 cached fields built with
exact integer passes, 48.000 MiB per field at 512³ scale, NumPy
oracle same arithmetic by construction).

The prior run was wall-clock clipped MID-WIRING. Current tree state
(lead-verified): headers/bindings declare the new terrain_render
signature but the definition was never landed — the .so builds and
fails at import: undefined symbol _ZN2tc14terrain_renderERK... Your
first task is making the engine import again; your last task is the
full gate battery.

Order of work (build LAST, per r2 amendment 3):
1. Land the terrain_render implementation matching the new
   declaration (cached-field construction + revision-keyed
   invalidation per your registered design).
2. Tests: extend tests/test_terrain_render.py with the NumPy oracle
   for filters 1/2, parity cases, the 45° ramp benefit gate
   (filter 2 ≤ 0.35 voxel or verbatim red), cache-invalidation test
   (carve → next frame reflects it), and memory-cost assertion.
3. Build once, run everything you can (CUDA may be sandbox-blocked —
   report blocked gates as blocked; lead re-runs on host).

All WO-8C-r2 gates stand: blocky + filter=0 byte-identical, parity
≥99.9% / |Δ|≤1, perf ≤12ms @512, suite green, A/B/C PPMs regenerated.
Rails as r2. Report: gate table verbatim.
