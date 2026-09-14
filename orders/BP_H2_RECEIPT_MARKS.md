# BP-H2 — campaign-receipt marks for the BP-KERNEL-3/4 harness tests on main (2026-09-14)

Seat: Opus 5 (opus-max). Lead: Fable 5.1. YOUR WRITABLE TARGET is `/mnt/ForgeRealm/wt/pt-bph2` (Project-Tensor worktree, branch `bp-h2`,
from `main` 0b8c643): edits under `tests/` and `docs/TESTS_CAMPAIGN_RECEIPTS.md` (append a "BP-H2 delta" section) AUTHORIZED. Nothing
under `tensor_cuda/`, `scripts/`, `artifacts/`. No git, no GPU, no subagents, never kill or signal a process you did not start.
Every pytest run: `PYTHONPATH=tensor_cuda python3 -m pytest -q -p no:cacheprovider tests/…`.

## Situation
BP-KERNEL-3 and BP-KERNEL-4 merged to main (0b8c643). The `campaign_receipt` marker and rules from BP-H1 exist in `tests/conftest.py`
and `docs/TESTS_CAMPAIGN_RECEIPTS.md` — read them first. On canonical `tests/` reads **1 failed, 122 passed, 6 skipped**:
`tests/test_bp_kernel_4.py::test_registration_pins_and_protocol` — expected class: binary-bound (the BP-KERNEL-4 registration pins the
pt-bk4 worktree's engine `.so` fingerprint; canonical is a different build). Verify the class yourself in isolation; the pinned gitignored
arrays are already provisioned in the canonical checkout (copy `artifacts/bp_kernel_1/*.npz`, `artifacts/bp_kernel_2/reference.npz`,
`artifacts/bp_kernel_3/census/receipt.json`, `artifacts/bp_kernel_4/{reference.npz,census/receipt.json}` from
`/mnt/ForgeRealm/Project-Tensor` into the same relative paths in your worktree BEFORE classifying — a provisionable file is not a receipt).
Also check `tests/test_bp_kernel_3.py` for any test that passes only because of provisioning or the worktree path (mark nothing that passes).

## Task
Mark the receipt(s) with `campaign_receipt(registration=…)`; no assertion changes; a genuine defect is pinned, not marked (say so).
Append the BP-H2 delta table to the doc.

## Done — paste verbatim
1. Delta table (node id, class, reason, registration). 2. Last lines of `tests/` default (0 failed) and `-m campaign_receipt` (every mark
reproduces). 3. `grep -n 'BP-H2' tests/*.py`. 4. Prior art line. 5. Model + effort; confirmations.
