# APA-P1-S3 r2b lead ruling (2026-11-03): ACCEPT (G1 to G7 PASS) on the seat's receipts; P1 now parks under this week's plan

**The round:**
- **The order:** `orders/APA_P1_S3_R2B_PROMOTION_ITEMS.md` (6a224cf5).
- **The seat:** Codex, gpt-6.1-sol at xhigh, CPU only. Run `20261103T213333Z-2960328`, `SHIM-DONE exit=0`, 57 minutes.
- **The report:** `artifacts/apa_p1_s3_r2/REPORT.md`.

## What it built

**Tickets (S1).**
- `eb3::counters` and `eb4::counters` now sit in a device-keyed map, with separate merge and choose slots and the L2
  size cached per device.
- A recursive host mutex covers lookup, growth and launch submission, so growth cannot free a pointer before another
  thread submits it.
- The operands' device is set and restored around allocation and launch, and growth synchronizes that device only.
- The single-stream contract is stated at the code site.
- **Size of the change:** `eb3.cuh` +161/−20, `eb4.cuh` +45/−16.

**The early-counter test (S2).**
- A two-CTA handshake with NaN-poisoned partials, plus a GIL-releasing native test binding (`bindings.cpp` +21).
- Both compile only with `TC_APA_P1_TEST_HOOKS=1`. The default is 0, and every hook sits under
  `#if defined(TC_APA_P1_TEST_HOOKS) && TC_APA_P1_TEST_HOOKS`.
- The CPU structural check passes both correct predicates and rejects EB3's mutant 3 and EB4's analogue.

**The exact-tie test (S3).** EB4's lead fixture, AST-identical to its source, with references at full width and
guarded nonzero and non-constant.

**The design (S4):** `DEFAULT_SWITCH_DESIGN.md`, with the source and argument census across the repositories.

**The new tests:** `test_apa_p1_tickets.py`, `test_apa_p1_early_counter.py` and `test_apa_p1_exact_tie.py`. On the
CPU: 5 passed, 4 GPU tests skipped.

**Staging:** six GPU lines, each dry-run to 78.

**No regression:** 2,347 existing test IDs show no outcome changes.

## Open, recorded for when P1 resumes

**SASS of the default build.** The seat's SASS comparison used the test-enabled build: 744 of 778 kernels unchanged,
and 34 changed, all in the select families that carry the hook argument. The default build has the hooks compiled
out. Its kernels are expected to equal A1's, but that was not measured here. The next P1 round shows it, or the
promotion build does.

**Not re-run by the lead.** The lead read the diff's guards and the seat's receipts, but did not re-run the CPU tests
in this round. P1 parks (the owner's week plan, 11-03: GRAPA run 8 first, no new tracks), so the GPU
lines, the default-switch round, the blind red team and the promotion ticket all wait for Tensor to resume.

## Held back from the public commit

118 files: the build tree, `refs/` (three modules), `source_before/`, `cmake_source/`, `cpu_compiled/`, the
`.sass` dumps, `OUTCOMES_*.json` (each 3 MB or more), and files carrying a local home path. Every one is hashed in
`artifacts/APA_P1_S3_R2_HELD_BACK.sha256`.
