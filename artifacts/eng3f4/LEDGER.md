# ENG3-F4 ledger (append only)

2026-11-17T08:29:35.488120+00:00 Registration sealed 701a401b5c40d93279c7f6337b73f0b07e06324cf242588d41c7ce4ec00d1a35. Rules, mechanism and baseline receipts read. Baseline snapshots and 31 touched IDs recorded. Python/pytest versions verified. G/H hashes verified. No gates run yet.

Prior art: Python subprocess env (2003+), pytest (2004+), NVIDIA CUDA Runtime API cudaGetDevice (year unverified), local ENG3 F3 r2/P1T-FIX4 (2026), NIST SHA-256 (2001). Reuse isolation, exact assertions, per-ID folding, pinned split harness. Ours: branch registration and F4 receipt assembly. External citations unverified -- lead to check search terms in REGISTRATION.md. No prior art known to me for the specific assembly beyond local harness.

Deviations so far: read-only ls discovery of absent artifacts/eng3f4 and /tmp/eng3f4_* returned exit 2; no gate affected.

Baseline reproduction: BEFORE_F01.log/xml, unset parent, pinned H, six failures on `assert None == ''` (pytest exit 1). F02/F03/F04 source findings read before edit. Applied exactly four fixes: per-row hidden-device child, exact current label, environment-registered branch with probe agreement, dead overflow child removed. Default branch uses environment convention (empty CVD => 0; otherwise 1), never query result. Explicit ENG3_EXPECT_CUDA binds both lanes. Prior art annotations added at each changed mechanism.

CPU gates: {"G1": true, "G2": true, "G4": false, "exits": {"G1_all": 0, "G1_unset": 0, "G2_host": 0, "G2_engine": 0, "G4_power": 1}}
Commands/full environments and log hashes: CPU_COMMANDS.json. Per-ID reference comparison: G2.json. Negative gate branch lines: G4.json.

CPU_RUN.log: G1 PASS (267 pass,46 GPU-only skip; unset six pass); G2 PASS (268 host IDs +3368 engine IDs identical to F3 r2; inherited red counts retained, no test writes). G4 initial summary false due to progress-prefixed branch parser. AMENDMENT_1.md registers own-script repair 1/3, preserving initial receipts. G4 raw tests already correctly fail all24 at the registered mismatch.

CPU gates: {"G1": true, "G2": true, "G4": true, "exits": {"G1_all": 0, "G1_unset": 0, "G2_host": 0, "G2_engine": 0, "G4_power": 1}}
Commands/full environments and log hashes: CPU_COMMANDS.json. Per-ID reference comparison: G2.json. Negative gate branch lines: G4.json.

AMENDMENT_1 reclassification from unchanged raw tests: CPU_SUMMARY_REPAIRED.log records G1/G2/G4 PASS. Initial G4.json and initial CPU script retained under before/. No test rerun or gate predicate change.

GPU staging prepared: three env -i launchers, explicit ENG3_EXPECT_CUDA=1, H-only 38-ID full F3+24 lane, inherited G/H g3-g1 and g3-g2 forms plus 49-row parity. INPUT_PINS.json seals helpers/tests/launchers/frozen split/module records. Real expected exits all0, dry exits78 with inner null. Prefixes exercise real host path and stop before GPU tests/first device call. Seat will invoke dry=1 only. Prior art: local F3 r2 prefix/split/per-ID harness (2026), pytest hooks (2004+), SHA-256 (NIST2001), subprocess env isolation (PSF2003+); ours F4 staging/receipts. External leads in helper comments.

CPU gates: {"G1": true, "G2": true, "G4": true, "exits": {"G1_all": 0, "G1_unset": 0, "G2_host": 0, "G2_engine": 0, "G4_power": 1}}
Commands/full environments and log hashes: CPU_COMMANDS.json. Per-ID reference comparison: G2.json. Negative gate branch lines: G4.json.

GPU dry receipts: gpu-tests, g3-g1, g3-g2 all forge/launcher78, inner null; all real CPU prefixes pass. GPU logs state no run, self_leasing false. Each SHA256 and exact lead line in STAGED_LINES.json. No GPU lease taken.

G5 scope PASS: only both authorized tests and dispatch log changed existing; permitted new scripts/artifacts only, zero missing/unexpected files, no git admin metadata changes, H live product sources and G/H module hashes match build receipts. Host edit confined exactly to cuda conditional, all original no-device assertions retained in child; memory child and all other F3 statements unchanged.

Ledger path correction: preserved initial G4 summary is G4_initial.json at artifact root; preserved initial CPU helper is before/eng3f4_cpu_initial.py. Both CPU_SUMMARY.log (first corrected classification) and CPU_SUMMARY_REPAIRED.log (identical later closure classification) are retained; CPU_RUN.log retains initial false summary. Only summary parsing rerun, no unchanged tests rerun.

Deviations final: one own-script repair, 1/3; progress prefixes caused G4 parser miss, corrected under AMENDMENT_1, raw 24 failures and initial negative summary retained. Read-only ls discovery returned2 for absent output/scratch paths and two guessed summary paths; no gates affected. No other repairs, no deletes/DELETE_ME moves, no git commands, subagents, rebuild or GPU runs. Long foreground CPU command yielded; resumed using 300000ms waits with no status messages between waits. Retained scratch is listed in SCRATCH_WRITTEN.txt.

Closure: REPORT.md, FILES_WRITTEN.txt, SCRATCH_WRITTEN.txt and FILES.sha256 written. Final G5 scope PASS; G1/G2/G4 PASS; G3 staged only. Acceptance pending lead real G3 receipts. No promotion.
