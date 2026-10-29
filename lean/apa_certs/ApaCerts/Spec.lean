import Mathlib

/-!
# APA certificate statements (TX3 X5): lead-written, immutable

This file is the trust boundary. Each `<name>_stmt` states one claim from the APA primer
(`/mnt/Shared/APA_Primer/APA_Primer.md`). Each `<name>_nv_stmt` states that its hypotheses are satisfiable (a
non-vacuity witness). A seat proves `theorem <name> : <name>_stmt` and `theorem <name>_nv : <name>_nv_stmt` in
`ApaCerts/Proofs.lean`. Amended after the fidelity review TX3-X5-REV1: sink-aware split-K counterexample (F1) and
antecedent witness for `sp_mask_superset` (F2). `ApaCerts/Check.lean` (also lead-written) pins each proof to its statement and prints its axioms.

**Conventions.**
- One attention row has keys `0..n`, indexed by `Fin (n+1)`, so every row is nonempty.
- Indices are zero-based: the primer's "key 1" is key `0` here.
- `bulk` is a key's cheap (quantised) score, and `exact` its full-precision score.
- The definitions mirror production code where noted.
-/

namespace ApaCerts

open Finset

noncomputable section
open Classical

/-- Running (prefix) maximum of `b` over keys `0..j`: the SP rule's `max(bulk[0..j])`. -/
def prefixMax {n : ℕ} (b : Fin (n+1) → ℝ) (j : Fin (n+1)) : ℝ :=
  (univ.filter (fun i : Fin (n+1) => i ≤ j)).sup' ⟨j, by simp⟩ b

/-- Maximum of `b` over the whole row. -/
def rowMax {n : ℕ} (b : Fin (n+1) → ℝ) : ℝ :=
  univ.sup' univ_nonempty b

/-- Prefix maximum of `b` restricted to one split-K partition `P` that contains key `j`. -/
def localPrefixMax {n : ℕ} (P : Finset (Fin (n+1))) (b : Fin (n+1) → ℝ) (j : Fin (n+1)) (hj : j ∈ P) : ℝ :=
  (P.filter (fun i => i ≤ j)).sup' ⟨j, by simp [hj]⟩ b

/-- The SP refine rule (primer §4.1, Q2): refine key `j` iff `bulk_j ≥ max(bulk[0..j]) − δ`. -/
def spRefined {n : ℕ} (bulk : Fin (n+1) → ℝ) (δ : ℝ) (j : Fin (n+1)) : Prop :=
  prefixMax bulk j - δ ≤ bulk j

/-- The final-maximum rule: refine key `j` iff `bulk_j ≥ max(bulk) − δ`. -/
def finalRefined {n : ℕ} (bulk : Fin (n+1) → ℝ) (δ : ℝ) (j : Fin (n+1)) : Prop :=
  rowMax bulk - δ ≤ bulk j

/-- The scores the online softmax sees: the exact score where a key is refined, the bulk score elsewhere. -/
def mixedScores {n : ℕ} (bulk exact : Fin (n+1) → ℝ) (refined : Fin (n+1) → Prop) : Fin (n+1) → ℝ :=
  fun i => if refined i then exact i else bulk i

/-- One row's attention output for scalar values `V`. Every key stays in the single softmax denominator. -/
def rowOutput {n : ℕ} (scores V : Fin (n+1) → ℝ) : ℝ :=
  (∑ i, Real.exp (scores i) * V i) / (∑ i, Real.exp (scores i))

/-- One row's attention output with an attention sink of logit `sink`. The sink adds `exp(sink)` once to the
denominator and carries no value (SP1.1 reference `dense_scores(..., sinks=...)`,
`tensor_cuda/tests/apa_sp1_reference.py`; the online merge adds the sink once to the denominator). -/
def rowOutputWithSink {n : ℕ} (scores V : Fin (n+1) → ℝ) (sink : ℝ) : ℝ :=
  (∑ i, Real.exp (scores i) * V i) / (∑ i, Real.exp (scores i) + Real.exp sink)

/-- The two-pass kernel's z-score rule (production `apa_selective_kernel`, `tensor_cuda/src/kernels.cu`, at b4da646:
pass 1 accumulates `a = |bulk|` and `a*a`; `mean = total/cnt`; `var = total_sq/cnt − mean*mean`;
`thr = mean + zthr*sqrtf(fmaxf(var, 0))`; pass 2 refines iff `fabsf(bulk) >= thr`). Real arithmetic, no rounding. -/
def zRefined {n : ℕ} (b : Fin (n+1) → ℝ) (z : ℝ) (j : Fin (n+1)) : Prop :=
  let N : ℝ := n + 1
  let mean := (∑ i, |b i|) / N
  let var := (∑ i, |b i| ^ 2) / N - mean ^ 2
  mean + z * Real.sqrt (max var 0) ≤ |b j|

/-! ## 1. `sp_skip_final`: a skipped key stays skipped (primer §4.1, Q2: the SP rule is monotone, hence
exactly single-pass). A skip decided at key `j` is still a skip against every later prefix maximum, and against the
final maximum. -/
def sp_skip_final_stmt : Prop :=
  ∀ (n : ℕ) (b : Fin (n+1) → ℝ) (δ : ℝ) (j k : Fin (n+1)),
    ¬ spRefined b δ j → (j ≤ k → b j < prefixMax b k - δ) ∧ ¬ finalRefined b δ j

def sp_skip_final_nv_stmt : Prop :=
  ∃ (b : Fin (1+1) → ℝ) (δ : ℝ) (j : Fin (1+1)), ¬ spRefined b δ j

/-! ## 2. `sp_mask_superset`: the SP mask contains the final-maximum mask (primer §1 / §4.1: SP "replaced the
rule", and its mask is a superset of the final-max mask). The witness shows the containment can be strict. -/
def sp_mask_superset_stmt : Prop :=
  ∀ (n : ℕ) (b : Fin (n+1) → ℝ) (δ : ℝ) (j : Fin (n+1)),
    finalRefined b δ j → spRefined b δ j

def sp_mask_superset_nv_stmt : Prop :=
  (∃ (b : Fin (1+1) → ℝ) (δ : ℝ) (j : Fin (1+1)), finalRefined b δ j) ∧
  (∃ (b : Fin (1+1) → ℝ) (δ : ℝ) (j : Fin (1+1)), spRefined b δ j ∧ ¬ finalRefined b δ j)

/-! ## 3. `splitk_contains`: split-K selection contains the global SP rule, but output equality fails
(primer §4.2). A partition-local prefix maximum is never above the global one, so a key refined by the global
SP rule is refined in its partition. Counterexample: bulk [2,0], exact [2,10], V = [0,1], δ = 0, with an attention
sink of logit 0 as in the receipt (and also without a sink). Global SP (one partition) against split-K with
partitions {0} and {1}: the outputs differ by more than 0.8. -/
def splitk_contains_stmt : Prop :=
  (∀ (n : ℕ) (P : Finset (Fin (n+1))) (b : Fin (n+1) → ℝ) (δ : ℝ) (j : Fin (n+1)) (hj : j ∈ P),
      localPrefixMax P b j hj ≤ prefixMax b j ∧
      (spRefined b δ j → localPrefixMax P b j hj - δ ≤ b j)) ∧
  (let bulk : Fin (1+1) → ℝ := ![2, 0]
   let exact : Fin (1+1) → ℝ := ![2, 10]
   let V : Fin (1+1) → ℝ := ![0, 1]
   let globalScores := mixedScores bulk exact (fun i => spRefined bulk 0 i)
   let splitScores :=
     mixedScores bulk exact (fun i => localPrefixMax {i} bulk i (mem_singleton_self i) - 0 ≤ bulk i)
   -- The receipt's instance (SP1.1 report; `test_extra_refinement_changes_output...`): sink logit 0.
   (0.8 : ℝ) < |rowOutputWithSink splitScores V 0 - rowOutputWithSink globalScores V 0| ∧
   -- The same instance without a sink.
   (0.8 : ℝ) < |rowOutput splitScores V - rowOutput globalScores V|)

def splitk_contains_nv_stmt : Prop :=
  ∃ (b : Fin (1+1) → ℝ) (δ : ℝ) (P : Finset (Fin (1+1))) (j : Fin (1+1)), j ∈ P ∧ spRefined b δ j

/-! ## 4. `z_not_single_pass`: the z-score rule cannot run in one irrevocable pass (primer §4.1, Q1). The
rows [1,0,0,0] and [1,0,0,10] share the prefix [1], yet for every `z ∈ [0, √3]` key 0 is refined in the first and
not in the second. So no decision that sees only the prefix (here: key 0's bulk value) agrees with the rule on both
rows. -/
def z_not_single_pass_stmt : Prop :=
  ∀ z : ℝ, 0 ≤ z → z ≤ Real.sqrt 3 →
    zRefined (![1, 0, 0, 0] : Fin (3+1) → ℝ) z 0 ∧
    ¬ zRefined (![1, 0, 0, 10] : Fin (3+1) → ℝ) z 0 ∧
    ∀ f : ℝ → Prop,
      ¬ ((f 1 ↔ zRefined (![1, 0, 0, 0] : Fin (3+1) → ℝ) z 0) ∧
         (f 1 ↔ zRefined (![1, 0, 0, 10] : Fin (3+1) → ℝ) z 0))

def z_not_single_pass_nv_stmt : Prop :=
  ∃ z : ℝ, 0 ≤ z ∧ z ≤ Real.sqrt 3

/-! ## 5. `sp2_certificate`: the SP2 δ certificate (primer §4.3). Suppose every key's bulk error is at most
`errq` (the primer's `e_q`). If the SP rule with `δ = ln(1/ε) + 2·errq` skips key `j`, then its true weight is
below `ε` times the true maximum's weight: `exp(exact_j) < ε · exp(max exact)`. This is the true-weight form. -/
def sp2_certificate_stmt : Prop :=
  ∀ (n : ℕ) (bulk exact : Fin (n+1) → ℝ) (errq ε : ℝ) (j : Fin (n+1)),
    0 < ε →
    (∀ i, |bulk i - exact i| ≤ errq) →
    ¬ spRefined bulk (Real.log (1 / ε) + 2 * errq) j →
    Real.exp (exact j) < ε * Real.exp (rowMax exact)

def sp2_certificate_nv_stmt : Prop :=
  ∃ (bulk exact : Fin (1+1) → ℝ) (errq ε : ℝ) (j : Fin (1+1)),
    0 < ε ∧ (∀ i, |bulk i - exact i| ≤ errq) ∧ ¬ spRefined bulk (Real.log (1 / ε) + 2 * errq) j

end

end ApaCerts
