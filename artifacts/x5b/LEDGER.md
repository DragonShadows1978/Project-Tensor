# TX3-X5B implementation ledger (seat: Opus 5.5, Claude Agent-tool seat, CPU only)

Order: `orders/TX3_X5B_AMENDED_STATEMENTS.md` (amendment; `orders/TX3_X5_APA_LEAN.md` stays binding), branch
`x5-apa-lean`, commit 3fe7dcb0. Append-only.

## 2026-10-28 09:55 EDT: start

- Read the amendment, the X5 order, `/mnt/Shared/HOUSE_RULES.md`, `AGENTS.md`, the amended `Spec.lean`, `Check.lean`,
  the current `Proofs.lean`, and the mechanism doc `docs/TX3_X5_REV1_REPORT.md` (findings F1, F2).
- Mechanism check (HOUSE_RULES §3): the amended `splitk_contains_stmt` adds the receipt's sink-0 instance through
  `rowOutputWithSink` and keeps the sink-free conjunct; `sp_mask_superset_nv_stmt` adds the antecedent witness
  `∃ b δ j, finalRefined b δ j`. Both are the review's proposed restatements. The gate is the real
  `lake build ApaCerts.Check`. No scope mismatch.
- Lead-file hashes before any work: `artifacts/x5b/hashes_before.txt`. Spec.lean = `36aeba97…f8fe5` (the order's pin),
  Check.lean = `2de6d3b0…abed` (the order's pin). Proofs.lean before edit = `a899ae23…050d` (copy at
  `/tmp/x5b_Proofs_before.lean`).

## 09:56: statement audit before any proof (D4 screen; evidence class: reasoning)

- **splitk_contains, sink-0 conjunct:** global scores [2,0] (key 0 refined, key 1 skipped), V = [0,1], sink 0:
  output 1/(e²+e⁰+e⁰) = 1/(e²+2) ≈ 0.1065. Split-K singleton scores [2,10]: e¹⁰/(e²+e¹⁰+1) ≈ 0.99962. Difference
  ≈ 0.8931 > 0.8. TRUE. Matches the order's operand census and REV1's 0.8931123035.
- **splitk_contains, sink-free conjunct:** unchanged from X5 (≈ 0.8805 > 0.8). TRUE.
- **sp_mask_superset_nv, antecedent witness:** bulk [0,1], δ = 0, key 1: rowMax = 1 ≤ 1 = bulk_1. TRUE (REV1's
  suggested instance). Strictness conjunct unchanged from X5.
- None reported under D4.

## 09:56-09:57: environment checks (no dependency built)

- `lake build --no-build Mathlib`: `All targets up-to-date (8923 jobs).`, rc 0 (`nobuild_mathlib_before.txt`). The D3
  rebuild rail did not trigger.
- `lake build --no-build ApaCerts.Spec`: `target is out-of-date and needs to be rebuilt`, rc 3
  (`nobuild_spec_before.txt`). The Spec olean predated the amendment, so `lake env lean ApaCerts/Proofs.lean` would
  have checked against the OLD statements. Fixed by building Spec alone first.
- `lake build ApaCerts.Spec`: `✔ [8924/8924] Built ApaCerts.Spec (5.9s)`, rc 0, wall 9.71 s, peak RSS 6.80 GB
  (`build_spec_only.txt`). Only Spec was built.

## 09:57-09:58: Proofs.lean edits (`lake env lean ApaCerts/Proofs.lean`)

- **New helper** `rowOutputWithSink_two`: two-key expansion of `rowOutputWithSink` by `Fin.sum_univ_two`.
- **`sp_mask_superset_nv`:** now `⟨⟨![0,1], 0, 1, antecedent⟩, ⟨![0,1], 0, 0, sp, ¬final⟩⟩`. The antecedent proof
  unfolds `finalRefined rowMax`, applies `Finset.sup'_le`, and closes both cases with `fin_cases i <;> simp`. The
  strictness proof is unchanged.
- **`splitk_contains`:** the containment part is unchanged. In the output part, the exp bounds (e² > 7, e⁶ ≥ 7,
  e¹⁰ > 49·e²) moved before the goal rewrite so both conjuncts share them. The goal is rewritten with
  `rowOutputWithSink_two` and `rowOutput_two`, normalised by `simp only` (no `norm_num` on the goal, so the normal
  form is predictable), 0.8 is rewritten to 4/5, and the conjunction is split:
  - sink 0: split > 23/25 (from 2·e¹⁰ > 23·e² + 23) and global 1/(e²+1+1) < 1/9 (from e² > 7), so the difference
    > 182/225 > 4/5;
  - no sink: split > 49/50 and global 1/(e²+1) < 1/8, as in X5.
- **Comments:** prior-art notes added at the helper, at `sp_mask_superset_nv`, at §3 and in the file header.
- **iter1:** rc 0, one linter warning: unused simp argument `one_mul` (`iter1_lake_env_lean.txt`). Removed it.
- **iter2:** rc 0, no messages, wall 7.47 s (`iter2_lake_env_lean.txt`).
- Then the header comment was rewrapped to 120 columns (comment only), before the gate run.
- **No other theorem's proof changed** (`proofs_diff.txt`): `sp_skip_final`, `sp_skip_final_nv`, `sp_mask_superset`,
  `splitk_contains_nv`, `z_not_single_pass`, `z_not_single_pass_nv`, `sp2_certificate` and `sp2_certificate_nv` are
  byte-identical, and so are the four boundary `example`s.
- **Personal-name screen (L21):** in the comments I added, the softmax prior art is cited by title and arXiv id, not
  by author. One author citation that the X5 seat wrote at line 16 (BLASST) is still there. I left it, because
  changing it was outside C1; it is disclosed for the lead.

## 09:58-09:59: gate run (Proofs.lean sha256 6ee2f428…a5c7, 368 lines)

- **X5-G1:** `lake build ApaCerts.Check` rc 0, `Build completed successfully (8926 jobs).`, wall 16.37 s, peak RSS
  6.88 GB. Only `ApaCerts.Proofs` (7.5 s) and `ApaCerts.Check` (5.0 s) were built. Scan: sorry 0, warning 0,
  error 0 (`build_check.log`, copy `build_check_log.txt`; `build_log_scan.txt`).
- **X5-G2:** 10 axiom lines, all exactly `[propext, Classical.choice, Quot.sound]`; 0 names outside the allowed set.
- **X5-G3:** the five lead-file hashes are IDENTICAL before and after (`hashes_before.txt`, `hashes_after.txt`).
  The D1 token grep of Proofs.lean has 0 hits for every token, by word and by case-insensitive substring
  (`forbidden_token_grep.txt`).
- `lake build --no-build Mathlib` afterwards: `All targets up-to-date (8923 jobs).` (`nobuild_mathlib_after.txt`).
- **Verdict:** ACCEPT (all three gates pass for all ten theorems). None reported under D4.

## Prior art (Prior Art Directive)

- **`rowOutputWithSink_two` / sink-0 conjunct:** standard softmax normalization with one sink logit in the
  denominator, as in SP1.1's reference `dense_scores(..., sinks=...)` (primer §4.2 [R26]). The online-softmax merge
  adds the sink once ("Online normalizer calculation for softmax", arXiv 1805.02867, 2018; FlashAttention, arXiv
  2205.14135, 2022). Unverified — lead to check; search terms: "online normalizer calculation for softmax",
  "attention sink softmax denominator". What I took: the instance and the denominator semantics (the lead's
  statement). Mine: only the Lean expansion and the rational bounds 23/25 and 1/9.
- **Exp bounds:** Mathlib `Real.exp_one_gt_d9` and `Real.add_one_le_exp`, reused from X5. Standard; no prior art
  claimed.
- **`sp_mask_superset_nv` antecedent witness:** the instance REV1 F2 suggested (bulk [0,1], δ = 0, key 1; SP1 mask
  containment, primer §4.1 [R24]; BLASST running-max criterion, arXiv 2512.12087). The proof uses Mathlib
  `Finset.sup'_le`. No prior art known to me beyond that.
- **The other eight theorems:** unchanged; their prior art is as recorded in `artifacts/x5/LEDGER.md`.
