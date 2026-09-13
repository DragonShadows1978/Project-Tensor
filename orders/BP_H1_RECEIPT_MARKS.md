# BP-H1 — campaign-receipt marks for the BP-KERNEL-1/2 harness tests on main (2026-09-13)

Seat: Opus 5 (opus-max). Lead: Fable 5.1. YOUR WRITABLE TARGET is `/mnt/ForgeRealm/wt/pt-bph1` (Project-Tensor worktree, branch
`bp-h1`, from `main` 17343ec+): edits under `tests/` and a new `docs/TESTS_CAMPAIGN_RECEIPTS.md` AUTHORIZED. Nothing under
`tensor_cuda/`, `scripts/`, `artifacts/`. No git, no GPU, no subagents, never kill or signal a process you did not start. Every
pytest run: `PYTHONPATH=tensor_cuda python3 -m pytest -q -p no:cacheprovider tests/…`.

## Situation
The BP-KERNEL-1 and BP-KERNEL-2 campaigns ran on worktrees (`wt/pt-bk1`, `wt/pt-bk2`) and their registrations pin source
sha256s, an engine-binary fingerprint and worktree-local artifacts. Merged onto `main`, **6 of 45** tests in
`tests/test_bp_kernel_1.py` / `tests/test_bp_kernel_2.py` fail: `test_source_drift_closed`,
`test_default_regions_unchanged_vs_pinned_main`, `test_dry_run_subprocess_precedence_and_schema` (bk1);
`test_registration_and_fail_closed`, `test_census_registration_and_fixed_protocol`, `test_schema_rejects_incomplete_or_cpu_claim`
(bk2 — verify the current list yourself). Expected causes: BP-KERNEL-1's registration pins `kernels.cu` at the bk1 state (main now
carries bk2's kernels.cu); BP-KERNEL-2's registration pins the pt-bk2 engine binary fingerprint (the canonical `.so` is a different
build); worktree-absolute paths. These are receipts of closed campaigns, valid at their registration sha — not defects.

## Task
1. Port GraftRepository's marker mechanism (read `/mnt/ForgeRealm/GraftRepository/tests/conftest.py` and the rules at the top of
   `/mnt/ForgeRealm/GraftRepository/docs/TESTS_CAMPAIGN_RECEIPTS.md` — read-only): a `campaign_receipt(registration=…)` marker,
   deselected by default, opt-in with `-m campaign_receipt` / `--campaign-receipts`. Create `tests/conftest.py` (Project-Tensor has
   none — check first; if one exists, extend it).
2. Reproduce each failure in isolation, classify (sha-bound / binary-bound / artifact-bound / genuine defect), mark receipts;
   a genuine defect gets pinned with no assertion change (say so). Do NOT mark a passing test. Do NOT loosen assertions.
3. `docs/TESTS_CAMPAIGN_RECEIPTS.md` (new): rules (copied and attributed), the BP-H1 delta table (module, node id, class, reason,
   registration path).

## Done — paste verbatim
1. The delta table. 2. Last lines of: the two BP modules default run (must be 0 failed), the same with `-m campaign_receipt`
(every mark reproduces), and `PYTHONPATH=tensor_cuda python3 -m pytest -q -p no:cacheprovider tensor_cuda/tests/test_ops_phase2.py
tensor_cuda/tests/test_attention_phase5.py tensor_cuda/tests/test_apa_phase6.py tensor_cuda/tests/test_checkpoint.py` (engine smoke,
GPU-free? if a test needs the device, report it as not-run here — no GPU in this seat). 3. `grep -n 'BP-H1' tests/*.py`.
4. Prior art (pytest marker idiom). 5. Model + effort; confirmations.
