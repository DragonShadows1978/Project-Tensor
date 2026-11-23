# ENG4 report (seat: Claude Opus 5.5; order orders/ENG4_MATMUL_CONTRACT.md, d4f42fd3)

Status: IN PROGRESS (kept current; the ledger artifacts/eng4/LEDGER.md has the receipts).

- Build B 20881ace0ef7f3fd1feb324f1f75a63589d523d0f943857de7564c5f0e297c40; Build E
  a875936c39b201e46e571040b13cbdd3779f99e9974fcfe341aff362cf910c82.
- G1 on E: 37 passed (13 GPU tests skipped); the same tests on B: 26 failed, 11 passed.
- Census: 0 product paths fix a refused pair (no M1 RED).
- L55 probe: B and E pass identical cuBLAS arguments on every accepted witness; B reads past B's buffer on ENG-MM1.
- Next: MUTANTS.json, mutant builds, G4 CPU per ID, GPU lines staged and dry-run.
