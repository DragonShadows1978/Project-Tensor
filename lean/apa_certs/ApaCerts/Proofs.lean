import ApaCerts.Spec

/-!
# APA certificate proofs (TX3 X5): seat-written proofs of the lead's statements

Every theorem here has exactly the type of its `<name>_stmt` in `ApaCerts/Spec.lean`; `ApaCerts/Check.lean` pins
each one. Helper lemmas come first.

**Prior art (Prior Art Directive, AGENTS.md).**
- The prefix-maximum facts (`le_prefixMax`, `prefixMax_le`, `prefixMax_mono`, `prefixMax_le_rowMax`,
  `exists_prefixMax_eq`, `localPrefixMax_le_prefixMax`) are thin wrappers over Mathlib's `Finset.le_sup'`,
  `Finset.sup'_le` and `Finset.exists_mem_eq_sup'` (Mathlib, `Mathlib/Data/Finset/Lattice/Fold.lean`): a supremum
  over a larger index set is larger, and a finite nonempty supremum in a linear order is attained. Standard order
  theory; nothing here is mine beyond the instantiation.
- The SP rule `bulk_j ≥ max(bulk[0..j]) − δ` and its monotonicity argument are BLASST's running-maximum criterion
  (Yuan, Shinn et al., arXiv 2512.12087, MLSys 2026), applied in APA to precision instead of block skipping (primer
  §7, lead-verified [R32]). What I took: the criterion as stated by the lead. What is mine: only the Lean proof.
- The SP2 certificate proof is the standard exp/log monotonicity argument (exp is strictly increasing,
  `exp (a + log ε) = ε · exp a` for `ε > 0`), plus the triangle bound `|bulk − exact| ≤ e_q` applied twice, once at
  the skipped key and once at the prefix argmax. The derivation of δ = ln(1/ε) + 2·e_q is SP2's (primer §4.3, [R06]);
  BLASST's ln λ threshold is the same quantity without the 2·e_q margin.
- The numeric bounds `exp 1 > 2.7182818283`, `exp 2 > 7`, `exp 6 ≥ 7`, `exp 10 > 49 · exp 2` use Mathlib's
  `Real.exp_one_gt_d9` (`Mathlib/Analysis/Complex/ExponentialBounds.lean`) and `Real.add_one_le_exp`
  (`Mathlib/Analysis/Complex/Exponential.lean`); the SP2 witness uses `Real.log_le_sub_one_of_pos`
  (`log x ≤ x − 1`). Standard bounds; no prior art beyond Mathlib is claimed for their use here.
- The z-score counterexample rows are the lead's / SP1's (primer §4.1, [R24]); the proof is direct evaluation.
- Split-K counterexample: SP1.1's (primer §4.2, [R26]); the proof is direct evaluation. After REV1 (X5B) it is
  checked both with the receipt's sink logit 0 (`rowOutputWithSink`, the standard softmax denominator plus one sink
  term; online-softmax merge per "Online normalizer calculation for softmax", arXiv 1805.02867, 2018, and
  FlashAttention, arXiv 2205.14135, 2022; unverified — lead to check) and without a sink. The bounds e² > 7 and
  e¹⁰ > 49·e² are reused for both.
- `sp_mask_superset_nv` antecedent witness (X5B, REV1 F2): bulk [0,1], δ = 0, key 1, the review's suggested instance;
  no prior art known to me beyond `Finset.sup'_le`.
-/

namespace ApaCerts

open Finset

/-! ## Helper lemmas: prefix and row maxima

Prior art: Mathlib `Finset.le_sup'` / `Finset.sup'_le` / `Finset.exists_mem_eq_sup'` (standard finite-supremum
order theory); no prior art known to me beyond that for these instantiations. -/

/-- Every key at or before `j` is at most the prefix maximum at `j`. -/
theorem le_prefixMax {n : ℕ} (b : Fin (n+1) → ℝ) {i j : Fin (n+1)} (hij : i ≤ j) :
    b i ≤ prefixMax b j := by
  unfold prefixMax
  exact Finset.le_sup' b (by simp [hij])

/-- The prefix maximum at `j` is at most any common upper bound of keys `0..j`. -/
theorem prefixMax_le {n : ℕ} (b : Fin (n+1) → ℝ) (j : Fin (n+1)) {c : ℝ}
    (h : ∀ i, i ≤ j → b i ≤ c) : prefixMax b j ≤ c := by
  unfold prefixMax
  exact Finset.sup'_le _ _ (fun i hi => h i (by simpa using hi))

/-- The running maximum never decreases (the monotonicity behind the SP rule; BLASST's running max). -/
theorem prefixMax_mono {n : ℕ} (b : Fin (n+1) → ℝ) {j k : Fin (n+1)} (hjk : j ≤ k) :
    prefixMax b j ≤ prefixMax b k :=
  prefixMax_le b j (fun _ hi => le_prefixMax b (hi.trans hjk))

/-- Every key is at most the row maximum. -/
theorem le_rowMax {n : ℕ} (b : Fin (n+1) → ℝ) (i : Fin (n+1)) : b i ≤ rowMax b := by
  unfold rowMax
  exact Finset.le_sup' b (Finset.mem_univ i)

/-- A prefix maximum is at most the final (row) maximum. -/
theorem prefixMax_le_rowMax {n : ℕ} (b : Fin (n+1) → ℝ) (j : Fin (n+1)) :
    prefixMax b j ≤ rowMax b :=
  prefixMax_le b j (fun i _ => le_rowMax b i)

/-- The prefix maximum is attained at some key `m ≤ j`. -/
theorem exists_prefixMax_eq {n : ℕ} (b : Fin (n+1) → ℝ) (j : Fin (n+1)) :
    ∃ m, m ≤ j ∧ prefixMax b j = b m := by
  unfold prefixMax
  obtain ⟨m, hm, h⟩ := Finset.exists_mem_eq_sup'
    (⟨j, by simp⟩ : (univ.filter (fun i : Fin (n+1) => i ≤ j)).Nonempty) b
  exact ⟨m, by simpa using hm, h⟩

/-- A partition-local prefix maximum is never above the global prefix maximum (primer §4.2). -/
theorem localPrefixMax_le_prefixMax {n : ℕ} (P : Finset (Fin (n+1))) (b : Fin (n+1) → ℝ) (j : Fin (n+1))
    (hj : j ∈ P) : localPrefixMax P b j hj ≤ prefixMax b j := by
  unfold localPrefixMax
  exact Finset.sup'_le _ _ (fun i hi => le_prefixMax b (Finset.mem_filter.mp hi).2)

/-- On a singleton partition the local prefix maximum is the key itself. -/
theorem localPrefixMax_singleton {n : ℕ} (b : Fin (n+1) → ℝ) (i : Fin (n+1)) :
    localPrefixMax {i} b i (mem_singleton_self i) = b i := by
  apply le_antisymm
  · unfold localPrefixMax
    refine Finset.sup'_le _ _ (fun x hx => ?_)
    have hx' : x = i := Finset.mem_singleton.mp (Finset.mem_filter.mp hx).1
    rw [hx']
  · unfold localPrefixMax
    exact Finset.le_sup' b (by simp)

/-- One row's output on a two-key row, written out. -/
theorem rowOutput_two (s V : Fin (1+1) → ℝ) :
    rowOutput s V = (Real.exp (s 0) * V 0 + Real.exp (s 1) * V 1) / (Real.exp (s 0) + Real.exp (s 1)) := by
  unfold rowOutput
  rw [Fin.sum_univ_two, Fin.sum_univ_two]

/-- One row's output with an attention sink on a two-key row, written out (X5B, amended statement F1). The sink
adds `exp sink` once to the denominator. Prior art: standard softmax normalization with a sink logit (SP1.1
reference `dense_scores(..., sinks=...)`, primer §4.2 [R26]); online-softmax merge ("Online normalizer calculation
for softmax", arXiv 1805.02867, 2018; FlashAttention, arXiv 2205.14135, 2022), unverified — lead to check: search
"online normalizer calculation for softmax", "attention sink softmax denominator". Mine: only the Lean expansion
via `Fin.sum_univ_two`. -/
theorem rowOutputWithSink_two (s V : Fin (1+1) → ℝ) (sink : ℝ) :
    rowOutputWithSink s V sink =
      (Real.exp (s 0) * V 0 + Real.exp (s 1) * V 1) / (Real.exp (s 0) + Real.exp (s 1) + Real.exp sink) := by
  unfold rowOutputWithSink
  rw [Fin.sum_univ_two, Fin.sum_univ_two]

/-! ## 1. `sp_skip_final` (primer §4.1, Q2). Prior art: BLASST running-max criterion (arXiv 2512.12087); the
monotone-prefix argument is `prefixMax_mono` + `prefixMax_le_rowMax` (Mathlib `Finset.sup'` order facts). -/

theorem sp_skip_final : sp_skip_final_stmt := by
  intro n b δ j k hskip
  unfold spRefined at hskip
  rw [not_le] at hskip
  refine ⟨fun hjk => ?_, ?_⟩
  · have h := prefixMax_mono b hjk
    linarith
  · unfold finalRefined
    rw [not_le]
    have h := prefixMax_le_rowMax b j
    linarith

/-- Witness: bulk [1, 0], δ = 0; key 1 is skipped (1 − 0 > 0). -/
theorem sp_skip_final_nv : sp_skip_final_nv_stmt := by
  refine ⟨![1, 0], 0, 1, ?_⟩
  unfold spRefined
  rw [not_le]
  have h0 := le_prefixMax (![1, 0] : Fin (1+1) → ℝ) (show (0 : Fin (1+1)) ≤ 1 by decide)
  simp at h0 ⊢
  linarith

/-! ## 2. `sp_mask_superset` (primer §1.3 / §4.1). Prior art: as §1 (BLASST criterion; Mathlib `Finset.sup'`). -/

theorem sp_mask_superset : sp_mask_superset_stmt := by
  intro n b δ j h
  unfold finalRefined at h
  unfold spRefined
  have h' := prefixMax_le_rowMax b j
  linarith

/-- Witnesses on the row bulk [0, 1], δ = 0 (amended after REV1 F2).
- Antecedent: key 1 is final-refined (row max 1 ≤ bulk 1), so the implication's hypothesis is satisfiable.
- Strictness: key 0 is SP-refined (prefix max 0) but not final-refined (max 1).
Prior art: the antecedent instance is the one REV1 F2 suggested (SP1 mask containment, primer §4.1 [R24]); the proof
is `Finset.sup'_le` plus case evaluation. No prior art known to me beyond that. -/
theorem sp_mask_superset_nv : sp_mask_superset_nv_stmt := by
  refine ⟨⟨![0, 1], 0, 1, ?_⟩, ⟨![0, 1], 0, 0, ?_, ?_⟩⟩
  · unfold finalRefined rowMax
    rw [sub_zero]
    refine Finset.sup'_le _ _ (fun i _ => ?_)
    fin_cases i <;> simp
  · unfold spRefined
    have h : prefixMax (![0, 1] : Fin (1+1) → ℝ) 0 ≤ 0 := by
      apply prefixMax_le
      intro i hi
      fin_cases i
      · simp
      · exact absurd hi (by decide)
    simpa using h
  · unfold finalRefined
    rw [not_le]
    have h1 := le_rowMax (![0, 1] : Fin (1+1) → ℝ) 1
    simp at h1 ⊢
    linarith

/-! ## 3. `splitk_contains` (primer §4.2). Containment: `localPrefixMax_le_prefixMax` (Mathlib `Finset.sup'_le`).
Counterexample: SP1.1's [R26], evaluated with `Real.exp_one_gt_d9` and `Real.add_one_le_exp`, once with the
receipt's sink logit 0 (`rowOutputWithSink_two`) and once without a sink (`rowOutput_two`). -/

/-- Global SP on bulk [2, 0], δ = 0: key 0 is refined. -/
theorem sp_ex_refined0 : spRefined (![2, 0] : Fin (1+1) → ℝ) 0 0 := by
  unfold spRefined
  have h : prefixMax (![2, 0] : Fin (1+1) → ℝ) 0 ≤ 2 := by
    apply prefixMax_le
    intro i _
    fin_cases i <;> simp
  simpa using h

/-- Global SP on bulk [2, 0], δ = 0: key 1 is skipped. -/
theorem sp_ex_skip1 : ¬ spRefined (![2, 0] : Fin (1+1) → ℝ) 0 1 := by
  unfold spRefined
  rw [not_le]
  have h0 := le_prefixMax (![2, 0] : Fin (1+1) → ℝ) (show (0 : Fin (1+1)) ≤ 1 by decide)
  simp at h0 ⊢
  linarith

theorem splitk_contains : splitk_contains_stmt := by
  refine ⟨fun n P b δ j hj => ?_, ?_⟩
  · have h := localPrefixMax_le_prefixMax P b j hj
    refine ⟨h, fun hr => ?_⟩
    unfold spRefined at hr
    linarith
  · -- Global SP scores: key 0 refined (exact 2), key 1 skipped (bulk 0).
    have hglob : mixedScores (![2, 0] : Fin (1+1) → ℝ) ![2, 10] (fun i => spRefined ![2, 0] 0 i) = ![2, 0] := by
      funext i
      fin_cases i
      · exact (ite_eq_left sp_ex_refined0).trans (by simp)
      · exact ite_eq_right sp_ex_skip1
    -- Split-K with singleton partitions: every key is refined, so the scores are the exact ones.
    have hsplit : mixedScores (![2, 0] : Fin (1+1) → ℝ) ![2, 10]
        (fun i => localPrefixMax {i} ![2, 0] i (mem_singleton_self i) - 0 ≤ ![2, 0] i) = ![2, 10] := by
      funext i
      have hc : localPrefixMax {i} (![2, 0] : Fin (1+1) → ℝ) i (mem_singleton_self i) - 0 ≤ ![2, 0] i := by
        rw [localPrefixMax_singleton, sub_zero]
      exact ite_eq_left hc
    -- Numeric bounds on exp, shared by both output conjuncts.
    have he1 : (2.7182818283 : ℝ) < Real.exp 1 := Real.exp_one_gt_d9
    have he2 : Real.exp 2 = Real.exp 1 * Real.exp 1 := by
      rw [← Real.exp_add]; norm_num
    have ha : (7 : ℝ) < Real.exp 2 := by rw [he2]; nlinarith
    have he6 : (7 : ℝ) ≤ Real.exp 6 := by
      have := Real.add_one_le_exp (6 : ℝ); linarith
    have he10 : Real.exp 10 = Real.exp 2 * (Real.exp 2 * Real.exp 6) := by
      rw [← Real.exp_add, ← Real.exp_add]; norm_num
    have hpos2 : 0 < Real.exp 2 := Real.exp_pos 2
    have h49 : (49 : ℝ) < Real.exp 2 * Real.exp 6 := by nlinarith
    have hc : 49 * Real.exp 2 < Real.exp 10 := by
      have := mul_lt_mul_of_pos_left h49 hpos2
      rw [he10]; linarith
    have h08 : (0.8 : ℝ) = 4 / 5 := by norm_num
    dsimp only
    rw [hglob, hsplit, rowOutputWithSink_two, rowOutputWithSink_two, rowOutput_two, rowOutput_two]
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one, Real.exp_zero, mul_zero, mul_one, zero_add]
    rw [h08]
    refine ⟨?_, ?_⟩
    · -- With sink logit 0 (the receipt's instance, REV1 F1): split e¹⁰/(e²+e¹⁰+1) ≈ 0.99962 against global
      -- 1/(e²+1+1) ≈ 0.1065. Bounds used: split > 23/25 and global < 1/9, so the difference > 182/225 > 4/5.
      have hA : (23 / 25 : ℝ) < Real.exp 10 / (Real.exp 2 + Real.exp 10 + 1) := by
        rw [lt_div_iff₀ (by positivity)]; linarith
      have hB : (1 : ℝ) / (Real.exp 2 + 1 + 1) < 1 / 9 := by
        rw [div_lt_div_iff₀ (by positivity) (by norm_num)]; linarith
      rw [lt_abs]
      left
      linarith
    · -- Without a sink: split e¹⁰/(e²+e¹⁰) against global 1/(e²+1); split > 49/50 and global < 1/8.
      have hA : (49 / 50 : ℝ) < Real.exp 10 / (Real.exp 2 + Real.exp 10) := by
        rw [lt_div_iff₀ (by positivity)]; linarith
      have hB : (1 : ℝ) / (Real.exp 2 + 1) < 1 / 8 := by
        rw [div_lt_div_iff₀ (by positivity) (by norm_num)]; linarith
      rw [lt_abs]
      left
      linarith

/-- Witness: bulk [2, 0], δ = 0, partition {0}, key 0 (refined globally, and a member of its partition). -/
theorem splitk_contains_nv : splitk_contains_nv_stmt :=
  ⟨![2, 0], 0, {0}, 0, mem_singleton_self 0, sp_ex_refined0⟩

/-! ## 4. `z_not_single_pass` (primer §4.1, Q1). Rows are SP1's [R24]; the proof evaluates the lead's real-number
transcription of the production z-score threshold. Prior art: no prior art known to me for the impossibility
argument itself (primer §7 lists it as APA's own); the sqrt facts are Mathlib's `Real.sqrt_mul`, `Real.sqrt_sq`. -/

/-- `zRefined` with its `let`s unfolded. -/
theorem zRefined_iff {n : ℕ} (b : Fin (n+1) → ℝ) (z : ℝ) (j : Fin (n+1)) :
    zRefined b z j ↔
      (∑ i, |b i|) / ((n : ℝ) + 1) +
        z * Real.sqrt (max ((∑ i, |b i| ^ 2) / ((n : ℝ) + 1) - ((∑ i, |b i|) / ((n : ℝ) + 1)) ^ 2) 0)
        ≤ |b j| :=
  Iff.rfl

theorem z_not_single_pass : z_not_single_pass_stmt := by
  intro z hz0 hz3
  have hA : zRefined (![1, 0, 0, 0] : Fin (3+1) → ℝ) z 0 := by
    rw [zRefined_iff]
    have hs1 : (∑ i, |(![1, 0, 0, 0] : Fin (3+1) → ℝ) i|) = 1 := by
      simp [Fin.sum_univ_four]
    have hs2 : (∑ i, |(![1, 0, 0, 0] : Fin (3+1) → ℝ) i| ^ 2) = 1 := by
      simp [Fin.sum_univ_four]
    have hN : ((3 : ℕ) : ℝ) + 1 = 4 := by norm_num
    have hv : max ((1 : ℝ) / 4 - (1 / 4) ^ 2) 0 = 3 / 16 := by
      rw [max_eq_left (by norm_num)]; norm_num
    have hb0 : |(![1, 0, 0, 0] : Fin (3+1) → ℝ) 0| = 1 := by simp
    rw [hs1, hs2, hN, hv, hb0]
    -- z · √(3/16) ≤ √3 · √(3/16) = √(9/16) = 3/4.
    have hst : Real.sqrt 3 * Real.sqrt (3 / 16) = 3 / 4 := by
      rw [← Real.sqrt_mul (by norm_num), show (3 : ℝ) * (3 / 16) = (3 / 4) ^ 2 by norm_num]
      exact Real.sqrt_sq (by norm_num)
    have hle : z * Real.sqrt (3 / 16) ≤ Real.sqrt 3 * Real.sqrt (3 / 16) :=
      mul_le_mul_of_nonneg_right hz3 (Real.sqrt_nonneg _)
    linarith
  have hB : ¬ zRefined (![1, 0, 0, 10] : Fin (3+1) → ℝ) z 0 := by
    rw [zRefined_iff, not_le]
    have hs1 : (∑ i, |(![1, 0, 0, 10] : Fin (3+1) → ℝ) i|) = 11 := by
      simp [Fin.sum_univ_four]; norm_num
    have hN : ((3 : ℕ) : ℝ) + 1 = 4 := by norm_num
    have hb0 : |(![1, 0, 0, 10] : Fin (3+1) → ℝ) 0| = 1 := by simp
    rw [hs1, hN, hb0]
    have hnn : 0 ≤ z * Real.sqrt (max ((∑ i, |(![1, 0, 0, 10] : Fin (3+1) → ℝ) i| ^ 2) / 4 - (11 / 4) ^ 2) 0) :=
      mul_nonneg hz0 (Real.sqrt_nonneg _)
    linarith
  exact ⟨hA, hB, fun f ⟨h1, h2⟩ => hB (h2.mp (h1.mpr hA))⟩

/-- Witness: z = 0 (the lower boundary). -/
theorem z_not_single_pass_nv : z_not_single_pass_nv_stmt :=
  ⟨0, le_refl 0, Real.sqrt_nonneg 3⟩

/-! ## 5. `sp2_certificate` (primer §4.3). Prior art: SP2's δ derivation [R06]; BLASST's ln λ threshold
(arXiv 2512.12087); standard exp/log monotonicity (`Real.exp_lt_exp`, `Real.exp_add`, `Real.exp_log`,
`Real.log_inv`); the attained prefix maximum is Mathlib's `Finset.exists_mem_eq_sup'`. -/

theorem sp2_certificate : sp2_certificate_stmt := by
  intro n bulk exact errq ε j hε herr hskip
  unfold spRefined at hskip
  rw [not_le] at hskip
  -- The prefix maximum is attained at some key m ≤ j.
  obtain ⟨m, _, hm⟩ := exists_prefixMax_eq bulk j
  rw [hm] at hskip
  have hj := abs_le.mp (herr j)
  have hmm := abs_le.mp (herr m)
  have hmax := le_rowMax exact m
  have hlog : Real.log (1 / ε) = - Real.log ε := by rw [one_div, Real.log_inv]
  rw [hlog] at hskip
  -- exact_j ≤ bulk_j + e_q < bulk_m + log ε − e_q ≤ exact_m + log ε ≤ max exact + log ε.
  have key : exact j < rowMax exact + Real.log ε := by
    linarith [hj.1, hj.2, hmm.1, hmm.2]
  calc Real.exp (exact j) < Real.exp (rowMax exact + Real.log ε) := Real.exp_lt_exp.mpr key
    _ = ε * Real.exp (rowMax exact) := by rw [Real.exp_add, Real.exp_log hε, mul_comm]

/-- Witness: bulk [10, 0], exact [9, 1], e_q = 1, ε = 1/2, key 1. Then δ = ln 2 + 2 ≤ 3 and 10 − δ > 0. -/
theorem sp2_certificate_nv : sp2_certificate_nv_stmt := by
  refine ⟨![10, 0], ![9, 1], 1, 1 / 2, 1, by norm_num, ?_, ?_⟩
  · intro i
    fin_cases i <;> norm_num
  · unfold spRefined
    rw [not_le]
    have h0 := le_prefixMax (![10, 0] : Fin (1+1) → ℝ) (show (0 : Fin (1+1)) ≤ 1 by decide)
    have hl : Real.log (1 / (1 / 2)) ≤ 1 := by
      have := Real.log_le_sub_one_of_pos (show (0 : ℝ) < 1 / (1 / 2) by norm_num)
      norm_num at this ⊢
      linarith
    simp at h0 ⊢
    linarith

/-! ## Boundary instantiations (order operand census)

Each universal theorem applied at a boundary named in Spec's comments. These add no claim beyond the ten theorems;
they only show the boundaries sit inside the proved statements. Prior art: none needed (direct instantiation). -/

/-- z exactly at √3: key 0 of [1,0,0,0] is still refined (its threshold equals |bulk_0| = 1 exactly). -/
example : zRefined (![1, 0, 0, 0] : Fin (3+1) → ℝ) (Real.sqrt 3) 0 :=
  (z_not_single_pass (Real.sqrt 3) (Real.sqrt_nonneg 3) le_rfl).1

/-- z exactly at 0: the threshold is the mean; key 0 of [1,0,0,10] is still not refined. -/
example : ¬ zRefined (![1, 0, 0, 10] : Fin (3+1) → ℝ) 0 0 :=
  (z_not_single_pass 0 le_rfl (Real.sqrt_nonneg 3)).2.1

/-- δ = 0 on a single-key row (n = 0): the final-max rule refines the only key, hence so does the SP rule. -/
example (b : Fin (0+1) → ℝ) : spRefined b 0 0 := by
  apply sp_mask_superset 0 b 0 0
  unfold finalRefined rowMax
  rw [sub_zero]
  refine Finset.sup'_le _ _ (fun i _ => ?_)
  fin_cases i
  exact le_rfl

/-- errq = 0 (exact bulk scores): the SP2 certificate still holds. -/
example (n : ℕ) (bulk exact : Fin (n+1) → ℝ) (ε : ℝ) (j : Fin (n+1)) (hε : 0 < ε)
    (h0 : ∀ i, |bulk i - exact i| ≤ 0) (hskip : ¬ spRefined bulk (Real.log (1 / ε) + 2 * 0) j) :
    Real.exp (exact j) < ε * Real.exp (rowMax exact) :=
  sp2_certificate n bulk exact 0 ε j hε h0 hskip

end ApaCerts
