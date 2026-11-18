# APA-P1-PF2 Amendment 1: PF-I3's tolerance (lead, 2026-11-16 16:15)

Follows: APA_P1PF2_PACKED_PREFILL.md

**r1 stopped correctly under L35.** PF-I3 registered τ = 4 × the largest all-refined value, capped at 1e-5. Even with
identical decisions (PF-I1, 21.9 billion decisions, 0 unequal), packed decode and dense prefill accumulate differently:
- decode sums fp32 chunks in double and then merges partials;
- prefill runs an fp32 online softmax with `__expf` and a per-key rescale;
- bf16 outputs are rounded at the final store.

The all-refined gap is up to 3.91e-5 (fp32, S 8,192) and 3.38e-3 (bf16). Both exceed the cap.
- **Lead fault:** the cap came from the design draft (§5.1), and I adopted it without checking the decode/prefill
  accumulation gap.
- **PF-I3's FAIL stands for r1**, as registered.

**The amendment, for the resumed round:**
- **A1.1, PF-I3.** τ per dtype is 4 × that dtype's largest all-refined relative row difference (prefill against
  packed decode), measured on PF-I1's grid. There is no cap.
  - The all-refined regime isolates accumulation order, which is the design's own definition of the tolerance's
    basis.
  - The values come from the all-refined runs already made, which r1's receipts hold: fp32 τ = 1.56e-4, bf16
    τ = 1.35e-2.
  - Rows above τ at the registered delta: 0 allowed.
- **A1.2, power.** Mutant `m12_unrefined_dropped` must fail A1.1's PF-I3. If it does not, the gate has no power, and
  the round stops RED.
- **A1.3, resume.** Continue from G1's PF-I3 verdict under A1.1, then run G2, G3, G4, G5, the rest of G6, G7 and the
  speed line as the order registers them. Every other gate, rule and budget is unchanged. The 5-hour GPU cap counts
  r1's 1.03 h.

**Registration checklist:**
- **L35:** A1.1 is the lead's amendment. It keeps PF-I3's registered principle (4× the all-refined accumulation gap)
  and removes the unchecked cap. A1.2 shows the gate keeps its power.
- **L104:** PF-I3 is per row, per dtype, with N = the rows of PF-I1's grid; slack τ as above; 0 rows above τ.
- **L99:** r1's report gives the per-dtype all-refined maxima: fp32 3.91e-5 (S 8,192), bf16 3.38e-3. G1's `RESULT.json`
  (`726d9a51…`) and the G1 parts hold them.
