# Preparation amendment 002 — preserve best green g if comparator f is RED

Source inspection after passing CPU tests found that census gate_source tied
its route choice to prediction eligibility. If f was RED but g2 was green,
it would incorrectly choose fallback g1. Correct route selection now retains
the best green g on a complete cell, while the f-RED result keeps the whole
step TIMING-ONLY. Incomplete/no-green cells still use registered g1 fallback.
A synthetic regression test covers this case. No GPU gate has run.

Census verification accepts a registered ancestor of the verified source-only
kernel amendment chain; every new census receipt binds the current effective
chain. All original protocol/config/tolerance/prediction values stay immutable.
Prior art: hash chains (Haber and Stornetta 1991), taken; source-only census
provenance integration ours. Unverified — lead to check that title. The route
selection is the user's experimental rule; no prior art known to me for its
particular combination of gates.
