# Trinity Nano Port — Implementation Ledger

Receipts for the Trinity Nano port. Plan: docs/TRINITY_NANO_PORT_PLAN.md
(immutable). Synthesis: docs/TRINITY_AFMOE_SYNTHESIS.md.

## 2026-07-08 (opening)

Work order opened (David: "Lets try that Arcee Trinity - Biggest version
that'll fit on the card"). Sizing receipt: Mini (26B-A3B, config fetched)
needs 13.8-14.6GB at INT4-with-scales vs 12,282 MiB card — not resident;
expert-streaming rejected (7-15ms/token PCIe tax + weeks of work). Nano
(6.4B-A1B, config fetched via Trinity-Nano-Preview; bare Trinity-Nano
401s) selected. Predictions T1-T4 frozen in plan before any code.

Next action: P0 architecture scout (dispatched: Grok, read-only).
