# ENG2-REV3 ledger

- Read house rules mirror, mechanism order and A3 seat final. No scope mismatch. Memory registry search had no relevant hit; no memory facts used.
- Created REPORT.md and findings.json before independent checks; registered immutable plan in REGISTRATION.md.
- Instruments first-use versions: Python 3.12.3, git 2.43.0, pytest 9.0.2, NumPy 2.4.2. Copied tool version pending discovery.
- D01: initial read-only bootstrap used TMPDIR=/tmp and omitted nice/thread controls. Re-read pending under mandated environment.
- Prior art: pytest project collection, Python project ast traversal; Kildall 1973 data flow and DeMillo/Lipton/Sayward 1978 mutation testing (unverified — lead to check); NIST FIPS 180-4 2015 hashing (unverified — lead to check). Standard methods reused; review-specific binding and classification only. See code comments and report.

- D01 re-read completed: house_rules_read.txt. D02: exploratory sed requested MANIFEST.sha256 within each run root, absent; canonical manifests are artifacts/eng{1,2}/lead/g5_1114/eng{1,2}_g5_MANIFEST.sha256 with relative run-root entries. No unverified G5 receipt parsed.
- Q1 complete: 156 to 146 for both arms; exact ten pins removed, none added. Four fresh CPU-only pytest collections; source.diff and Q1_ids.json.
- R01 (own script repair 1/3, Q2): KeyError nodeid; recorded report key is id. Fix key and re-run.
- R02 (own script repair 2/3, L60 classification): changed set must use recorded labels tc.nn.parameter and tc.tensor(device=cuda,0 bytes), not only shorthand. Preliminary 32 differences were changed-entry rows accidentally included; preserve Q3_L60.initial.json and correct label set.
- Q4 copied evidence.py has no --version handler: --version exits zero without output. Source hash recorded as version identifier. Original verify_seal and verify_arm passed 17 arms; 21 recorded dry-run labels checked.

- Q2 repaired run passed 31/31. All 93 parent/child/JUnit files matched their canonical run manifests before parsing; JUnit and child call outcomes agree.
- Q3 scan parsed all 2452 tracked Python files at pinned HEADs, zero parse failures, zero statically confirmed bad dtype sites. Retained 38 unresolved direct/alias calls and 408 other candidate receivers for follow-up. Need instance/module-field/runtime-dictionary receiver tracing and A3 spot checks. Voice HEAD advanced to caeb64d3; other heads match A3.
- D03 (spot-source inline script repair 1/3): requested nonexistent grapa/engine_native.py; fixed operand list, repeated source excerpt collection successfully as Q3_spot_sources.txt. No consumer content executed.
- L60 unchanged-row comparison leaves only two raw message differences: pybind object addresses. These will be retained and separately normalized for behavioral parity; no behavioral difference is implied.

- E2R3-F01 recorded: 16 omitted changed-entry sites (Voice 2, GRAPA 13, GRM 1); 15 supported literals, one unresolved self.dtype(t) in tripwire.py:655. All 48 A3 unresolved changed-entry sites retained in final independent census. Voice extra files identical at old A3 HEAD. Live tracked source comparison via git grep found no differences from pinned HEAD Python sources in any repository.

- R03 (supplement_checks.py repair 1/3): dtype_name source declaration is const char*, not std::string; initial source needle asserted. Corrected the literal needle; no product code changed; first transcript kept in supplement_run.txt.

- Completed 11 source spot checks across all three repositories; supported the seat readings with mutable/global/external-caller qualifications. Expanded L60 parity: top-level 54 unchanged rows and A2 162 unchanged rows agree after only pybind object-address normalization.
- Staged censuses, current empty pins and after-commit test hash across all 17 arms verified. Dry_4 corroborated: 4 governor-78 roots, 17 dependency refusals, all 21 S1 argv/bounds match registration.
- Prior art in final report: pytest (2004+), DeMillo/Lipton/Sayward (1978), Python ast/Kildall (1973), SHA-256/FIPS 180-4 (2015), all unverified literature leads; standard methods reused and review-specific bindings only.
- Final REPORT.md and findings.json written, one MINOR E2R3-F01. findings.json sha256 d044315222ae7b4259bfe3320be9176e98c8a1c8c76b532872464678d5d9c1c3. No product changes or deletes.
