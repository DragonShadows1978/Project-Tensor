# WO-8C-r4 — convert the red benefit gate to a recorded-finding pin (micro)

Implementation agent, Project-Tensor, repo root = cwd. The WO-8C-r2
benefit gate ran and came back RED by its registered rail: 45° ramp
max deviation filter 0/1/2 = 0.5857849121 / 0.5224800110 /
0.4974231720 vs the 0.35 target. Per the r2 brief that red IS the
finding: density pre-filtering cannot melt voxel terracing. The
ledger records the red verbatim (docs/briefs/WO-8C_r2_ledger.md —
append the finding there too).

Your change: in tests/test_terrain_render.py, convert the failing
0.35 assertion in test_terrain_render_smooth_density_filter_ramp_geometry
into a RECORDED-FINDING REGRESSION PIN: assert the three measured
deviations to within ±0.01 each AND keep the monotonic-decrease
assertion; comment block must state the 0.35 target, that it was NOT
met, and that the approach is CLOSED for terrace-melting (pointer to
the r2 ledger). Do not weaken any other assertion. Gate:
`python3 -m pytest tests/test_terrain_render.py -q` fully green
(CUDA may be sandbox-blocked; report blocked). Rails: that test file
+ docs/briefs/WO-8C_r2_ledger.md only. No subagents/git-write/
network/pip. Report the suite line verbatim.
