# APA-P1-PF2 r2 state (2026-11-17 ~01:55 EST): stopped at the L56b repair limit

Done in r2 (GPU lines total, r1 + r2: 1.33 h of 5 h):
- Mutant builds m07-m16 (m13 on PFH): all PASS. G7: 16/16 killed, 16/16 controls clean (`MUTANT_RESULTS.json`).
  A1.2: m12 fails A1.1's PF-I3 on 132/132 rows (`gpu/G7/MUTANT_RESULTS_A12.json`).
- G5 PASS (`gpu/G5_r2/RESULT.json`), G6 PASS (LF1, L2/L4, LF6a/b, PINS, SASS), G2 partial (1.7B L11, L22 pass).

Blocked: L56b allows 3 script repairs per round; r1 used 2, r2 used 1 (the allocation shim's symbol resolution).
Three more exceptions in this round's own scripts are NOT repaired; each needs the lead's authorization:
1. G4 (`scripts/apa_p1pf2_g45.py` g4_arm): the local `fake = tc.PackedRegisteredMargins('0' * 64, ...)` shadows the
   helper `fake()`; TypeError at the FK line. Fix: rename the local to `fake_table` (2 uses).
2. G2 (`scripts/apa_p1pf2_g2.py` unit): `mask_d[0].sum(0, dtype=torch.int32)` casts the whole [32,8192,8192] mask to
   int32 (8 GiB, OOM on the 12 GB card, 4B units only). Fix: count upper-triangle entries per head
   (`sum(int(torch.triu(mask[0, h]).count_nonzero()) - ...)`, or `torch.triu(mask[0, h], 1).count_nonzero()` per h).
   The two 1.7B unit receipts in gpu/G2 stand; rerun reuses them (the script skips existing unit files).
3. G3 (`scripts/apa_p1pf2_q1.py` setup): P1-Q's `apa_p1q_harness.register()` (the 'p1q' AttentionInterface hook) is
   not called. Fix: call `H.register()` in setup() after `H.tc = tc`.
Not run yet: the speed line (`scripts/apa_p1pf2_speed.py`, untested).

Resume after authorization: apply the three fixes, then run G4 (`apa_p1pf2_g45.py g4 --out gpu/G4_r2`), G2
(`apa_p1pf2_g2.py --out gpu/G2`), G3 (time window 0 of Qwen3-1.7B first, project against 3.67 h left, then all
windows of both models, then `apa_p1pf2_q1.py report --out gpu/G3`), and the speed line, each through
`scripts/apa_p1pf2_line.py`.
