# Erratum (lead, 2026-10-07): the MF2.2 engine binary was overwritten

`artifacts/apa_mf2_2/verdict.json` names the MF2.2 engine as `bc7ed2a9…`.
- That binary was overwritten at 19:29 by a stray rebuild during Amendment 1 (the seat's deviation, `docs/APA_MF2_2_LEDGER.md` §A).
- The mis-built directory was moved to DELETE_ME (`20261007-193132_apa-mf2-2a_build`).
- Every MF2.2 receipt that used `bc7ed2a9` predates the overwrite.

The seat rebuilt MF2.2 from verified sources (kernels.cu `17c89f3c`; the other 18 sources equal the MF2.1 manifest).
- The rebuilt binary is `320932d9…`. The builds are not byte-reproducible, so it does not match `bc7ed2a9`.
- Its outputs equal the original's on all 2,376 census cells and on the identity and N1 sets (`logs/apa_mf2_2a_rebuild_mf2_2.log`).
- Read `bc7ed2a9` in MF2.2's receipts as "original MF2.2 build, superseded on disk by `320932d9`".
