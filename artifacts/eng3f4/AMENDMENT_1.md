# F4 own-script repair 1/3 — receipt parser only

Initial CPU run: G1 PASS, G2 PASS. G4 raw pytest exit 1, exactly 24 failures
at registered ENG3_EXPECT_CUDA=1/probe mismatch. Initial summary falsely
reports G4 FAIL because startswith misses 23 branch lines prefixed by pytest
progress characters (F). Preserve G4_initial.json and CPU_RUN.log unchanged.
Fix extraction to locate literal ENG3F4_BRANCH expected=[01] anywhere on each
line. No test/gate/expected count or branch predicate changed. Re-summarize
existing receipts; no suite rerun is justified. This is own-script repair 1/3.
Prior art: Python re extraction (PSF), inherited pytest -s progress format;
ours: parser correction. Unverified -- lead to check: pytest terminal progress.
