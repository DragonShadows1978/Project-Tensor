
## Results (LEAD-RECORDED; agent report wrapper-clipped, its sandbox had no GPU)

Lead-run on host after rebuild: G1 blocky byte-identical (7B fixtures
unmodified, green). G2 CUDA-vs-NumPy: flat 100% exact; sphere 100%
exact (surface RMS 0.122 voxel, normals 2.65° RMS); noise fixture
99.99989% exact, max |Δ| 1, depth rel 5.3e-5 — all within bars.
G3 timing: smooth 2.84ms mean @512×512×192 (budget 8ms, 2.8× under);
blocky unchanged 1.71ms. G4 suite 203 passed, 1 documented
pre-existing fail (phase7 norms). G5 fixture artifacts generated but
operator-grade A/B deferred to the Scorch wiring (synthetic noise
fixture illegible for aesthetics).
