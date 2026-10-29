# TX3-X5 implementation ledger (seat: Opus 5.5, Claude Agent-tool seat, CPU only)

Order: `orders/TX3_X5_APA_LEAN.md` (branch `x5-apa-lean`, commit b6c9b9c9). Append-only.

## 2026-10-28 09:28 EDT: start

- Read the order, `/mnt/Shared/HOUSE_RULES.md`, `AGENTS.md`, `lean/apa_certs/ApaCerts/Spec.lean`, `Check.lean`,
  `lakefile.lean`, and the mechanism doc (`/mnt/Shared/APA_Primer/APA_Primer.md` §0-§1.3, §4.1-§4.3, §7).
- Mechanism check (HOUSE_RULES §3): the ten statements formalize primer §4.1 (Q1 z-score rows, Q2 monotone SP rule),
  §1.3/§4.1 (superset), §4.2 (split-K containment and the bulk [2,0] / exact [2,10] / V [0,1] counterexample) and §4.3
  (δ = ln(1/ε) + 2·e_q certificate). The gate is the real `lake build ApaCerts.Check`. No scope mismatch.
- Toolchain inside `lean/apa_certs`: Lean 4.34.1 (commit 5045d00), Lake 5.0.0-src+5045d00. (Outside the project the
  elan default is 4.28.0; all commands run from `lean/apa_certs`, which pins v4.34.1.)
- Lead-file hashes before any work: `artifacts/x5/hashes_before.txt` (all five match the order's pins for Spec/Check).

## 09:29-09:31: environment checks (no build of any dependency)

- `lake env lean ApaCerts/Spec.lean`: exit 0, no messages, wall 21.4 s, peak RSS 6.75 GB.
- `lake build --no-build Mathlib` (refuses to build; it only checks traces): `All targets up-to-date (8923 jobs).`, exit
  0. So the copied Mathlib is current, and the D3 rebuild rail did not trigger.
- `lake build ApaCerts.Spec`: `✔ [8924/8924] Built ApaCerts.Spec (5.3s)`, exit 0. Only Spec was built, so that
  Proofs.lean can import it.

## 09:31: statement audit before any proof (D4 screen; evidence class: reasoning)

All five statements and five witnesses were judged TRUE as stated, so none is reported under D4.
- **1 (sp_skip_final):** ¬spRefined gives b_j < pmax_j − δ. Also pmax_j ≤ pmax_k for j ≤ k, and pmax_j ≤ rowMax.
- **2 (sp_mask_superset):** pmax_j ≤ rowMax.
  - Strictness witness: [0,1], δ = 0, key 0.
- **3 (splitk_contains):**
  - Containment: the local filter is a subset of the global one, so the local sup' ≤ the global sup'.
  - Counterexample: global scores [2,0], so the output is 1/(e²+1) ≈ 0.119. The singleton split refines every key,
    so its scores are [2,10] and its output is e¹⁰/(e²+e¹⁰) ≈ 0.99966. The difference is ≈ 0.8807 > 0.8.
- **4 (z_not_single_pass):**
  - Row [1,0,0,0]: mean 1/4, var 3/16, threshold 1/4 + z·√3/4. That is ≤ 1 iff z ≤ √3, with equality at z = √3.
  - Row [1,0,0,10]: the threshold is ≥ 11/4 > 1 for z ≥ 0.
  - The f-clause follows from A ∧ ¬B.
- **5 (sp2_certificate):** let m be the prefix argmax (m ≤ j). Then
  exact_j ≤ bulk_j + e_q < bulk_m − ln(1/ε) − e_q ≤ exact_m + ln ε ≤ max exact + ln ε,
  and exp is strictly monotone with exp(a + ln ε) = ε·exp a.

## 09:33-09:35: Proofs.lean iterations (`lake env lean ApaCerts/Proofs.lean`; logs in /tmp/x5_scratch/iter*.log)

- **iter1:** 4 errors, all in splitk_contains part 2:
  - `fin_cases` left beta-redexes, so `rw [if_pos …]` found no pattern;
  - an `nlinarith` needed a product hint;
  - norm_num's normal form has `(exp 2 + 1)⁻¹`, not `1/(exp 2 + 1)`;
  - `if_pos`/`if_neg` are deprecated in Lean 4.34.1 (core `Init/Core.lean:1184`: use `ite_eq_left`/`ite_eq_right`).
- **iter2:** the Edit was BLOCKED by the PreToolUse hook "AFTERIMAGE [POSTGRESQL]: You've written similar code
  before! … Retry your write now." So iter2 re-ran the unchanged file and gave the same errors. I retried the edit and
  it applied.
- **iter3:** `show` re-elaborated the `if` and failed to synthesize `Decidable (spRefined …)`: Spec uses
  `open Classical`. Fixed by `exact (ite_eq_left h).trans (by simp)` / `exact ite_eq_right h`, which unify with the
  definition's own instance.
- **iter4:** exit 0, no messages.
- Docstring corrections: the Mathlib file paths were checked with grep (`Finset.le_sup'` etc. in
  `Mathlib/Data/Finset/Lattice/Fold.lean`, `exp_one_gt_d9` in `Mathlib/Analysis/Complex/ExponentialBounds.lean`,
  `add_one_le_exp` in `Mathlib/Analysis/Complex/Exponential.lean`). One more hook block on this edit; the retry
  applied.

## 09:35: gate run 1 (Proofs.lean sha256 355fcfbf…9ce7)

- `lake build ApaCerts.Check`: exit 0, wall 16.93 s, peak RSS 6.88 GB.
- All 10 axiom lines read `[propext, Classical.choice, Quot.sound]`.
- 0 sorry, 0 warning, 0 error.
- The five lead-file hashes were unchanged.
- Receipts: `build_check_run1.log` (+ `_log.txt` copy), `build_log_scan_run1.txt`, `forbidden_token_grep_run1.txt`,
  `hashes_after_run1.txt`, `proofs_sha256_run1.txt`. The run-1 receipts were renamed with `mv` to `*_run1*`; nothing
  was deleted.

## 09:36-09:37: boundary instantiations added (operand census), then gate run 2 (FINAL)

- **Added** four `example`s at the boundaries Spec's comments name:
  - z = √3: key 0 of [1,0,0,0] is refined; its threshold equals 1 exactly;
  - z = 0: key 0 of [1,0,0,10] is not refined;
  - δ = 0 on a single-key row (n = 0): spRefined holds;
  - errq = 0: sp2_certificate applies.
- **Why:** this makes the boundaries visible as machine-checked instances. It adds no claim beyond the ten theorems.
- **Edits:** the PreToolUse hook blocked this edit twice. iter5 ran on the unchanged file (exit 0). The edit applied
  on the third attempt, and iter6 gave exit 0 with no messages.
- **Gate run 2** (final Proofs.lean, sha256 c29a3627…4856):
  - `lake build ApaCerts.Check` exited 0, wall 16.58 s, peak RSS 6.92 GB.
  - Only `ApaCerts.Proofs` (7.5 s) and `ApaCerts.Check` (4.2 s) were built, out of 8926 jobs.
  - All 10 axiom lines read `[propext, Classical.choice, Quot.sound]`.
  - The log scan found 0 sorry, 0 warning, 0 error.
  - The D1 token grep found 0 hits for every token.
  - `hashes_after.txt` matches `hashes_before.txt`.
  - `lake build --no-build Mathlib` afterwards: `All targets up-to-date (8923 jobs).`
- **Receipts:**
  - `build_check.log`, with a `build_check_log.txt` copy, because `*.log` is git-ignored (repo `.gitignore` line 8);
  - `build_log_scan.txt`, `forbidden_token_grep.txt`, `theorem_list.txt`;
  - `hashes_before.txt`, `hashes_after.txt`, `proofs_sha256.txt`.

## Prior art (Prior Art Directive)

- **Prefix/row-maximum lemmas:** Mathlib `Finset.le_sup'`, `Finset.sup'_le` and `Finset.exists_mem_eq_sup'`
  (`Mathlib/Data/Finset/Lattice/Fold.lean`). This is standard finite-supremum order theory; I added only the
  instantiation.
- **SP rule and its monotonicity:** BLASST's running-maximum criterion (Yuan, Shinn et al., arXiv 2512.12087, MLSys
  2026; primer §7, lead-verified [R32]), applied by APA to precision. I took the criterion as the lead stated it. Mine
  is only the Lean proof.
- **SP2 certificate:**
  - The δ = ln(1/ε) + 2·e_q derivation is SP2's (primer §4.3 [R06]).
  - BLASST's ln λ threshold is the same margin without 2·e_q.
  - The proof uses standard exp/log monotonicity: Mathlib `Real.exp_lt_exp`, `Real.exp_add`, `Real.exp_log` and
    `Real.log_inv`.
- **Numeric bounds:** Mathlib `Real.exp_one_gt_d9`, `Real.add_one_le_exp` and `Real.log_le_sub_one_of_pos`.
- **Counterexamples:** the z-score rows (SP1, primer §4.1 [R24]) and the split-K example (SP1.1, primer §4.2 [R26]).
  The proofs evaluate them directly.
- **The z-score impossibility argument itself:** no prior art known to me. Primer §7 lists it as APA's own.
- **Lean tactic technique:** none beyond standard Mathlib usage (linarith, nlinarith, positivity, norm_num,
  fin_cases). No prior art claimed.
