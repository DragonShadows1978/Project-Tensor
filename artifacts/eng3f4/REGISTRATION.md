# ENG3-F4 registration (immutable; before all gates)

Plan: orders/ENG3_F4_REV4_TEST_FIXES.md, immutable; follows F3 r2.
Mechanism read: eng3rev3/artifacts/eng3_rev4/findings.json version 2 and order rulings.
Scope is the four REV4 test findings, 31 IDs listed in TOUCHED_IDS.json plus
removal of the unused overflow child. No product edit/rebuild/promotion.
Rules read: /mnt/ForgeRealm/HOUSE_RULES.md (mirror) and Project-Tensor/AGENTS.md.
No git commands, subagents, deletes, GPU leases, background waits.
Every shell command under timeout. Tool versions first: Python 3.12.3,
pytest 9.0.2. Module hashes verified in MODULES.json before this registration.

G1: pinned H, CUDA_VISIBLE_DEVICES='', ENG3_EXPECT_CUDA=0. Run both full
test files, skipping only collected host GPU rows (g2 and g1b cuda) and the
existing device-index skip. Expect 267 passed, 46 skipped, no failures/errors.
Also reproduce the six parent-dependent failures before editing with CVD
unset, then require the fixed six all pass with CVD unset (only child calls
CUDA and child sets CVD=''). Never skip the six no-device rows.

G2: run F3 r2's CPU selection (cpu or test_f1_gpu_device_index) on pinned H,
ENG3_EXPECT_CUDA=0, CVD empty. Require all 268 IDs and outcome/collection
sets identical to host/G1_H_r2/RESULT_r2.json; list the 31 touched IDs and
both outcomes even if identical. Rerun inherited F3 r2 CPU engine suite on
H, same files/options/exclusions and per-phase folding. Compare per ID to
eng3f3/g3_r2/g3-cpu_H/outcomes_r2.json; expect 3368 IDs, no differences and
same collection file set; inherited red counts remain explicit (815 pass,
1946 fail,349 error,258 skip; pytest exit 1). Compare outcomes, never timings.
No differences outside touched IDs; no unregistered exemptions. Test writes
must be empty after excluding only artifacts/eng3f4 and logs/eng3f4_* control
receipts. Temporary and pytest basetemp paths /tmp/eng3f4_*; cache disabled.

G3 staged only: three launchers explicitly establish environment, pin H
(and G for comparison lines) before lease, run real host CPU prefix with CVD
empty, then set CVD=0 and ENG3_EXPECT_CUDA=1 for forge-gpu-run --min 30 --max
540. One H-only line runs all 14 F3 IDs plus the exact 24 bare-cuda overflow
IDs (38 passed, zero skips, including all six no-device children and exact
current-device label). g3-g1/g3-g2 reuse F3 r2 split forms on G/H and run 49
host parity IDs (48 overflow + device-index); every bare-cuda record must
show registered 1, query-success overflow. No six no-device tests on G: that
is intentionally H-only. Split suite red outcomes compared per ID as before.
Real expected launcher/forge/inner exits 0. Seat invokes only dry=1; expected
forge/launcher 78, inner null, no lease. Record logs' SHA256 and exits for
lead; check test/helper/module pins before lease. Full environment recorded
by launcher. Lead runs commands unchanged from worktree root.

G4: pinned H, CVD='', ENG3_EXPECT_CUDA=1, exactly 24 bare-cuda IDs. Require
pytest exit 1 and exactly 24 failed, none passed/skipped/error; every failure
must have executed and printed ENG3F4_BRANCH expected=1 branch=overflow with
query success false. Fail at registered/probed mismatch before allocation.
G4 is expected negative evidence, never a passing pytest suite.

G5: metadata census before/after, SHA256 product pins, previous receipts and
module pins intact, exact allowed write set. Host file diff only inside
cuda conditional in test_f2_cpu_overflow_message, CPU statements unchanged.
F3 edits only four findings; no narrowing. No files removed; .git admin state
unchanged. Active shim log is separately reported; preexisting logs/eng3f4_r1.log
is expected to grow. Initial before snapshots remain immutable.

Repair ceiling: at most 3 own-script repairs, recorded in separate amendments
before reruns; never rewrite registration. Deadline: 3 hours from dispatch.
Prior art: Python subprocess environment isolation (Python/PSF, 2003+), pytest
parametrization/hooks (Krekel et al., 2004+), CUDA Runtime API cudaGetDevice
(NVIDIA; year unverified), local ENG3 F3 r2 harness and P1T-FIX4 (2026),
SHA-256 (NIST, 2001). Taken: subprocess env, exact equality assertions,
per-ID phase folding and pinned split harness; ours: F4 branch registration
and receipt assembly. No new numerical algorithm. External literature
unverified -- lead to check: Python subprocess env; pytest hooks; CUDA Runtime
API cudaGetDevice; NIST SHA-256. No prior art known to me for the specific
F4 artifact assembly beyond the cited local harness.
