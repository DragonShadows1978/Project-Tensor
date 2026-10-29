# APA-DK-STE1 immutable registration
Order: user APA-DK-STE1, 2026-10-24; source f2c686e (user-supplied, no git invoked).
Only this worktree is writable. CUDA_VISIBLE_DEVICES is empty for every seat command.
No GPU, git, subagents, deletions, production changes or background jobs.
S-G1: census all train forward/backward dispatch and GRAPA path.
S-G2: float64 reference; central differences epsilon=1e-6, atol=2e-7 rtol=2e-5; dense/all-selected equality atol=2e-12.
S-G3: CUDA12.6 build, CPU import with once-only switch print (unset,0,1); reject all other strings.
S-G4: selected-only mutant killed by nonzero unselected key gradient / L-G3.
L-G1 six tensors exact bytes vs base on GRAPA and two small shapes; L-G2 forward exact bytes; L-G4 refine-all exact bytes; L-G5 repeated on exact bytes.
L-G3 tolerance protocol (registered before any measurements): measure base/off absolute max errors E for each output/gradient on the same small inputs; freeze tolerance per tensor as max(8*E, 1e-5) before on runs. Require max on absolute error <= frozen tolerance, and nonzero on dK for an entirely unselected key with exact zero off dK. Persist calibration before arm. A calibration or branch witness failure is RED, never a tolerance adjustment.
L-G6 CUDA-event backward off/on timings, same shape and configuration; require finite positive samples and report ratio, no speed acceptance bound. Whole trainer step only if model can be safely instantiated read-only.
Every GPU job: forge-gpu-run --max 600 --label UNIQUE, timeout --foreground 300s; raw full-width tensors, nonzero/nonconstant checks, verdict recomputed from raw; governor ran/rc0 required.
Prior art: Bengio, Leonard and Courville (2013), straight-through estimator; Dao et al. (2022), FlashAttention backward recomputation; both unverified - lead to check. SHA-256 manifests: NIST FIPS 180-4 (2015), unverified - lead to check. Implementation applies STE solely to key accumulation, retains existing attention arithmetic.
