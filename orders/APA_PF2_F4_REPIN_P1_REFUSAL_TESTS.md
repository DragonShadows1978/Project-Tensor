# APA-PF2-F4: re-pin the two P1 tests that still assert the packed-prefill refusal PF2 removed (Codex; tests only; CPU)

Follows: APA_PF2_F3_REV2_FIXES.md
ID reuse: PF2-F4 is the fourth fix round of the same APA-PF2 change, as F1 to F3 were

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_pf2f3`
(branch `apa-pf2f3`, at the commit carrying this order; PF2-F3's receipts are `497df5ab9`, and the docs `f480b661c`).
Edits and CPU runs are AUTHORIZED, limited to:
- `tensor_cuda/tests/test_apa_p1d_policy.py`, only `test_prefill_on_packed_keys_is_refused_naming_dense` (line 574) and
  the fixture helpers it alone uses;
- `tensor_cuda/tests/test_apa_p1f2_cpu.py`, only `test_r1_pinned_route_refusals_still_precede_any_device_call` and the
  `FakeSpo2` fake it uses;
- new `scripts/apa_pf2f4_*.py`, `artifacts/apa_pf2f4/` (new), `logs/apa_pf2f4_*`;
- scratch under `/tmp/pf2f4_*`. Point `TMPDIR` there.

**Read-only:**
- every product file (no rebuild; Build F3, `1036ec93…`, is the module);
- every other test;
- every other worktree, including MAINCHK3's receipts (`tensor_fork_apa_wt_mainchk3/artifacts/mainchk3/`);
- production.

Never walk `/`, `/mnt` or `/mnt/Shared` recursively.

## Why

MAINCHK3 (`15b5af202`) ran the engine's whole CPU suite on the PF2 promotion tree and on both parents. Two IDs pass on
production and fail on the merged tree and on the PF2 reviewed tip:
- `test_apa_p1d_policy.py::test_prefill_on_packed_keys_is_refused_naming_dense`;
- `test_apa_p1f2_cpu.py::test_r1_pinned_route_refusals_still_precede_any_device_call`. Its `FakeSpo2` has no
  `packed_prefill`, so the new route raises AttributeError.

Both are P1 tests that pin the refusal PF2 removed by design. PF2's rounds never re-pinned them, because PF2's gates
never ran the whole CPU suite: a lead order gap. Shipping as is turns two passing tests red.

## The ruling

Re-pin each test to the shipped behaviour (`docs/APA.md` §8 at `f480b661c`): prefill (more than one query row) on
packed keys runs `_spo2.packed_prefill`.
- **The first test** becomes a test that prefill on packed keys routes to the packed prefill kernel. Keep what it
  checks beyond the refusal, such as naming and recorded route, wherever that still applies. Rename it to say what it
  now checks, and record the old and new names.
- **The second test** keeps checking that every remaining route refusal precedes any device call. `FakeSpo2` gains a
  `packed_prefill` that records its call. The prefill row now expects that call, not a refusal.
- **Census:**
  - `grep -rn -i 'prefill' tensor_cuda/tests/*.py`, then read each hit that asserts a refusal, a `ValueError` or a
    dense pin for prefill on packed keys;
  - the other prefill hits (sp3, sp4g and the others) pin dense prefill contracts, unrelated to packed keys.

  Confirm each in `CENSUS.json`, and edit only the two named tests.

## Gates

- **G1:** both tests pass on CPU with Build F3 on the import path.
- **G2:** the whole CPU suite on Build F3, in MAINCHK3's harness form (its scripts imported by path, CPU, one thread),
  compared per test ID with MAINCHK3's PF2 arm. The only changed IDs are the two re-pinned tests (now passing) and, if
  renamed, their new IDs. Every other ID is the same.
- **G3, power:** with `packed_prefill` made to raise in `FakeSpo2` (in scratch, not in the shipped file), the second test
  fails. With the route reverted to refusing (in scratch), the first test fails.
- **G4, scope:** test edits only; no product change; no git writes; nothing deleted; no GPU.

**Registered acceptance:** ACCEPT when G1 to G4 pass. These are test re-pins, so there is no further review. The lead
carries the two files to the promotion tree and reruns them in MAINCHK3's three arms.

## Mechanism & scope

- **Measures:** whether the two inherited P1 tests now pin the shipped prefill route, with nothing else in the suite
  changing.
- **Mechanism doc:** `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_pf2f3/docs/APA.md`
  (§8, at `f480b661c`), with `tensor_fork_apa_wt_mainchk3/artifacts/mainchk3/NOISE.json`.
- **Exercises:** the two tests; the whole CPU suite on Build F3; the two scratch power checks.
- **Does not cover:**
  - product code;
  - GPU runs;
  - any other test.
- **Operand census:** the grep above; the two tests; `FakeSpo2`.
- **Baseline isolation:** MAINCHK3's PF2-arm CPU results are the per-ID reference.
- **Required evidence:**
  - the diff;
  - `CENSUS.json`;
  - G1 and G2 receipts;
  - G3's power receipts;
  - `REPORT.md`.
- **Forbidden configuration:**
  - any product edit;
  - editing any other test;
  - GPU use;
  - git writes;
  - deletes.

## Seat and rules

Codex gpt-6.2-sol, xhigh effort.

**Rules:** read `/mnt/Shared/HOUSE_RULES.md` and `/mnt/ForgeRealm/Project-Tensor/AGENTS.md` first. This order is
immutable.
- No git writes, no subagents, no background waits; every command under a `timeout`.
- **No deletes:** never `rm`, `rmdir`, `unlink`, `git clean`, `find -delete` or an inline delete call. Leave and list.
- **CPU jobs:** one thread, under their own `RLIMIT_AS` (the whole suite needed up to 24 GiB in MAINCHK3); never a
  systemd scope. Start only with at least 12 GB MemAvailable over the limit you set.
- **Wall:** 3 hours. Your final message is the report; also write `artifacts/apa_pf2f4/REPORT.md`.
- **Prior art:** follow `AGENTS.md`.

## Registration checklist (order-lint)

- **L01:** pytest and python3, each run with `--version` first.
- **L02:** G1 is the smoke.
- **L03:** n/a. No share is gated.
- **L09:** G2 compares per test ID and outcome. Logs and timings are never compared.
- **L16:** the cheapest pass would delete the refusal assertions. Excluded: G3's power checks require each re-pinned
  test to fail when the new route breaks.
- **L35:** no gate is narrowed.
- **L38:** only the two named tests and `FakeSpo2` change. List each changed assertion, old and new.
- **L40:** MAINCHK3's harness is imported from its own path. Its default writes are rebound into
  `artifacts/apa_pf2f4/` and listed in `COPY_DEPS.json`.
- **L41:** each power check executes the code path its test names.
- **L56:** (a) tool slips go under Deviations; (b) at most 3 repairs of your own scripts.
- **L58:** CPU commands run with `CUDA_VISIBLE_DEVICES` empty.
- **L65:** no deletes.
- **L82:** the edit sites are the two tests (`test_apa_p1d_policy.py:574` and the named test in
  `test_apa_p1f2_cpu.py`) and `FakeSpo2`, from the census above.
- **L99:** the lead's readings: MAINCHK3's two interaction IDs, with the outcomes quoted above (M3 and PF2 fail, PROD
  passes, both runs).
- **L109:** helpers run as `python3 -B -P`; none shares a standard-library module name.
- **L114:** CPU jobs are bounded by their own `RLIMIT_AS`; none needs systemd in the seat.

## Done

Your final message gives, verbatim:
- the diff;
- `CENSUS.json`'s summary;
- G1 to G4;
- the deviations;
- every file written.
