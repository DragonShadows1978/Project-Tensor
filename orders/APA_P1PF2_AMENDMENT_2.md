# APA-P1-PF2 Amendment 2: three named script fixes authorized (lead, 2026-11-17)

Follows: APA_P1PF2_PACKED_PREFILL.md (with Amendment 1)

r2's receipts are `ed4838e4`. The round used its three L56b repairs. Three more script faults stopped G4, G2 and G3
before any row was measured. Each is a defect in the seat's own harness, not in the product code or a gate. The lead
authorizes exactly these three edits, outside the L56b count:

1. `scripts/apa_p1pf2_g45.py:197`: rename the local `fake = tc.PackedRegisteredMargins(...)` to `fake_table` (and its
   uses), so it no longer hides the `fake()` helper at line 119.
2. `scripts/apa_p1pf2_g2.py:84-85`: count the upper-triangle tails head by head, instead of casting the whole mask to
   int32 in one `sum`. The count is unchanged; peak memory is one head's mask. The two completed 1.7B unit receipts
   stand, and the rerun may skip them.
3. `scripts/apa_p1pf2_q1.py`: call P1-Q's `H.register()` in `setup()` (line 43), before the `p1q` attention hook is
   named.

Nothing else changes: no gate, bar, product file, build or mutant. Any further script fault counts against a fresh
L56b allowance of 1 for this resume. Then, in order: G4, G2 (the 4B layers), G3 (time one window first, and project it
against the GPU hours left of the 5 h cap), and the speed line. The report follows the order's Done section.
