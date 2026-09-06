# APA-SP3 amendment 1 (lead, 2026-09-06) — card change

David: the MiniCPM3-4B receipts in `GraftRepository/docs/MiniCPM3-MLA_Results.md`
(standard 20.065, APA bulk-4 19.817, context ceilings 3,072 standard /
32,768 APA) were produced on an 8 GB RTX 3070 (sm_86). SP3 runs on the
12 GB RTX 4070 SUPER (sm_89).

Consequences, registered before any GPU run:
1. G0 parity tolerance ±0.01 applies to the in-process reproduction of
   the PROTOCOL. A drift vs 20.065 attributable to the card (different
   SM generation, kernel selection) is a card change, not a RED; the
   lead adjudicates by re-running arm A here and using the in-process
   A/B/C/D/E comparison on identical tokens as the result.
2. The 3,072 / 32,768 ceilings are 8 GB numbers and are NOT targets.
   Standard still cannot reach S=8192 (40 heads x 8192^2 fp32 scores =
   10.7 GB); the long rows remain APA-only. Report what fits on 12 GB.
The seat was dispatched before this amendment; it does not change the
order's mission, arms, or predictions.
