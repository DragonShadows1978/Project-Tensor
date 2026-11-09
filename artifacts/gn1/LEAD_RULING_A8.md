# Lead ruling on GN1 amendment A8: ACCEPT under contract C; GN1-REV6 next

- **Order:** `orders/GN1_A8_CONTRACT.md` (`6eb2f564`).
- **Run:** Codex seat, run `20261101T214732Z-3079897`, 2,650 s, `SHIM-DONE exit=0`.
- **Report:** `artifacts/gn1/a8/REPORT.md`.

**Verdict:** ACCEPT. G1 to G6 pass. P125 and P126 are HELD.

## Checked by the lead

- **G1:** `git diff --quiet cf922891` over `nn.py` and the test file gives rc 0. The frozen earlier-round maps are
  equal before and after (44,761 files, `a8/G1_scope.json`).
- **G6:** the honest 58-line CPU root passes every line check. Its only root error is "not a device run: every line
  requires positive child device evidence" (`a8/G6_staging.json`).
- **G4** (`a8/G4_summary.json`):
  - 290 of A6's 322 targets were replayed under contract C;
  - 32 were classified as edited-receipt history and not run;
  - 4 new live controls and 25 direct unit cases ran: 24 refusals, plus the one data-only accept case;
  - all 270 resulting control roots were refused.
- **D2:** the device-evidence fields come from A7's `device_evidence.py`, unchanged, with the source lines cited in
  the report. The tooling diff holds 17 D2 and D3 hunks across 9 files (`a8/tooling_hunks.json`).

## For GN1-REV6

- **The classification:** check the D4 classification (`a8/D4_classification.md`), in particular the 32 class (iii)
  targets.
- **The scorer:** check D2's acceptance path, and D3's demote-only modes.
- **The judgement:** against contract C.

## Prior art

As the seat's report. No new algorithm.
