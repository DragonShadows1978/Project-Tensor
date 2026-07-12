# WO-12G-r2 — finish the robust-entry fix (continuation)

Implementation agent, Project-Tensor (repo root = cwd). CONTINUATION
of WO-12G, same tree. Your diagnosis nailed it before the clip:
legacy entry nudge floors the boundary-exact hit into the
out-of-grid cell (trace: legacy_floor [32 2 7] on a 0..31 grid vs
robust_floor [31 2 7]) — sky leak. Finish: apply the robust entry
sampling across the object DDA path, then run the FULL registered
battery: G-LEAK (zero interior-background pixels, 24-pose sweep incl.
sub-5-degree grazing), objects parity fixtures byte-exact vs their
references (justify in ledger if correct behavior legitimately
changed any), perf within +0.1ms, engine suite green, before/after
receipts at the worst angle. Build the .so at the END. Report
verbatim.
