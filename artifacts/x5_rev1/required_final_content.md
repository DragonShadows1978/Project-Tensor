| Question | Statement | Verdict | Basis and lines read |
|---|---|---|---|
| S1 | sp_skip_final | FAITHFUL | Inclusive-prefix skip stays strict at every later prefix and final maximum. Spec 62–70; Proofs 97–119; Primer 99–102, 248; MUTATIONS 294. |
| S2 | sp_mask_superset | FAITHFUL | Final-max refinement implies prefix refinement; no equality of masks or outputs claimed. Spec 72–79; Proofs 121–145; Primer 99–102; MUTATIONS 295. |
| S3 | splitk_contains | MISMATCH | Containment and a valid sink-free counterexample are proved, but the receipt’s fixed sink-logit-zero counterexample is not encoded (F1). Spec 81–98; Proofs 147–214; Primer 259–261; SP1.1 report 9–13; test_apa_sp1_1.py 67–74. |
| S4 | z_not_single_pass | FAITHFUL | The registered shared-prefix obstruction for 0 ≤ z ≤ √3, including both endpoints. Spec 100–113; Proofs 216–262; Primer 247; MUTATIONS 298–299; SP1 registration 115. |
| S5 | sp2_certificate | FAITHFUL | Strict true-weight certificate with full-key error bound and the 2·errq margin. Spec 115–127; Proofs 264–299; Primer 274–284; SP2 derivation 9–47. |
| S6 | non-vacuity witnesses | INADEQUATE | S1 ADEQUATE; S2 INADEQUATE (F2); S3 ADEQUATE; S4 ADEQUATE; S5 ADEQUATE. |
| S7 | overall | REJECT | For recording all five original receipt claims verbatim. Four match; S3 certifies the sink-free variant. This does not dispute any checked proof or the qualitative split-K non-equality result. |

## Findings

**F1 — MAJOR: The split-K receipt uses sink logit 0; the Lean counterexample omits the sink.**

The receipt fixes sink=0, which contributes exp(0)=1 to the denominator. rowOutput has no sink term. Global/split outputs in Lean are 1/(exp(2)+1) and exp(10)/(exp(2)+exp(10)); in the receipt they are 1/(exp(2)+2) and exp(10)/(exp(2)+exp(10)+1). Differences are approximately 0.8804617278 and 0.8931123035 respectively. Both exceed 0.8. Thus the qualitative non-equality claim is faithfully demonstrated, but the exact receipt construction is not machine-checked by this statement. MAJOR is about faithful recording of that fixed instance, not a false theorem or kernel defect.

Evidence: Spec.lean:49–51,81–95; Proofs.lean:173–190; /mnt/ForgeRealm/Project-Tensor/docs/APA_SP1_1_REPORT.md:11–13; /mnt/ForgeRealm/Project-Tensor/tensor_cuda/tests/test_apa_sp1_1.py:67–74; /mnt/ForgeRealm/Project-Tensor/tensor_cuda/tests/apa_sp1_reference.py:37–50. Class: source inspection, exact arithmetic reasoning, CPU Decimal illustration.

Proposed restatement: Retain the universal containment conjunct and the fixed bulk [2,0], exact [2,10], V [0,1], delta=0 and singleton masks. Define rowOutputWithSink(scores,V,s) = (sum_i exp(scores_i)*V_i)/(sum_i exp(scores_i)+exp(s)); use rowOutputWithSink(...,0) for both globalOut and splitOut in the >0.8 conjunct. Retain the existing no-sink theorem as an additional variant, if desired. A source-site comment should credit SP1.1 R26 (2026) and the standard softmax/sink denominator; this is a transcription repair, with no new algorithm.

Prior art: APA SP1.1 (2026), R26; standard softmax normalization and online-softmax merge (Milakov and Gimelshein 2018; FlashAttention, Dao et al. 2022, via Primer section 7). Taken: receipt operands, masks and denominator semantics. New: identification of this transcription mismatch only. Literature not independently verified; unverified — lead to check: SP1.1 sink zero counterexample, online softmax Milakov Gimelshein 2018.

**F2 — MINOR: sp_mask_superset_nv witnesses strictness where the claim’s antecedent is false.**

The implication requires finalRefined b delta j. The nv statement instead requires spRefined b delta j AND NOT finalRefined b delta j. Its actual witness b=[0,1], delta=0, j=0 therefore makes the main implication vacuous at that key. This is a useful strict-containment witness, but it does not directly exercise the claim’s antecedent. This does not invalidate S2 or imply its antecedent is unsatisfiable; the same row at j=1 satisfies it.

Evidence: Spec.lean:74–79; Proofs.lean:130–145; Check.lean:15–16; MUTATIONS.md:304–308. Class: source inspection, Lean CPU probe.

Proposed restatement: Keep the strictness existential, and conjoin a separate existential over b:Fin 2→R, delta:R and j:Fin 2 satisfying finalRefined b delta j. For example, b=[0,1], delta=0, j=1 has finalRefined and spRefined true. A source-site comment should credit SP1’s final/prefix mask containment (R24, 2026; BLASST running-max criterion, Yuan/Shinn et al. 2025/2026); this only repairs witness coverage.

Prior art: APA SP1 (2026), R24, and X5 non-vacuity requirement, MUTATIONS 304–308; BLASST (Yuan, Shinn et al., arXiv 2512.12087, 2025/MLSys 2026) for the running-max criterion, via Primer section 7. Taken: existing mask definitions and antecedent-satisfiability requirement. New: diagnosis that strictness and antecedent coverage differ. Literature not independently verified; unverified — lead to check: BLASST running maximum criterion.

## Per-definition check

| Definition | Verdict | Check |
|---|---|---|
| prefixMax | FAITHFUL | Spec 25–27: nonempty finite supremum of b over i ≤ j; includes j; no artificial zero. For [-2,-3], prefixes are [-2,-2]. |
| rowMax | FAITHFUL | Spec 29–31: maximum of all eligible row keys; Fin(n+1) is nonempty, including n=0. No sink is a key in this maximum. |
| localPrefixMax | FAITHFUL | Spec 33–35: maximum over P ∩ {i ≤ j}; hj includes j and ensures nonemptiness. Arbitrary finite subsets generalize ordered partitions; no singleton-only restriction. Empty partitions have no j to decide. |
| spRefined | FAITHFUL | Spec 37–39: signed bulk ≥ inclusive prefix maximum − δ. Equality refines; negation gives strict skip. All real δ generalizes the receipt’s nonnegative margins. |
| finalRefined | FAITHFUL | Spec 41–43: signed bulk ≥ final row maximum − δ; distinct from both the z rule and the prefix rule. |
| mixedScores | FAITHFUL | Spec 45–47: exact on refined keys, bulk otherwise; all keys retained. |
| rowOutput | MISMATCH for the fixed R26 instance | Spec 49–51: correct scalar sink-free softmax, strictly positive denominator. A scalar output suffices for an output-inequality counterexample. Omits the sink-zero denominator contribution in the cited receipt (F1). |
| zRefined | FAITHFUL | Spec 53–60: moments of absolute bulk, population variance, max(var,0), mean+z·sqrt, inclusive threshold; N=n+1 is the count of eligible keys. |

## Z-rule comparison

| Part | Production/source lines | Comparison |
|---|---|---|
| Eligible keys | Kernel 1611–1617, 1627, 1638 | Spec 12, 56–59: interpret Fin(n+1) as exactly the eligible causal prefix or full row, n+1=s_max>0. Masked keys are not zero-filled into N. |
| Bulk and magnitudes | Kernel 1630–1635, 1641–1643 | b_i is the scaled bulk dot; moments use \|b_i\| and \|b_i\|², including negative scores by magnitude. |
| Reduction and count | Kernel 1646–1658 | Each key counted once; WCOOP lane-zero gate avoids 32-fold duplication. mean=Σ\|b\|/N; var=Σ\|b\|²/N−mean², population rather than sample variance. |
| Threshold | Kernel 1659 | mean + zthr·sqrt(max(var,0)) equals Spec 60 in real arithmetic. Variance is nonnegative mathematically; retaining the clamp matches the code. |
| Refine and blend | Kernel 1684–1695 and 1715–1725 | \|bulk\| ≥ threshold refines, with equality included; otherwise bulk is retained. Matches zRefined and mixedScores. |
| Endpoints | Spec 105–110; Proofs 306–312; CPU probe | For A=[1,0,0,0], μ=1/4, var=3/16, threshold=(1+z√3)/4. At z=0 it is 1/4; at z=√3 it is exactly 1, so key 0 refines at both. For B=[1,0,0,10], μ=11/4, var=283/16; threshold≥11/4>1 for z≥0, so key 0 skips. |
| Scope | MUTATIONS 298–299; SP1 registration 115 | The interval [0,√3] contains the receipted zthr=1.0364333894937898. At each fixed z and row length, any deterministic decision based only on the identical observed prefix induces the same f(1); both masks cannot be matched. The first three values coincide, so waiting through those zeros also cannot distinguish the rows. No unrestricted buffering impossibility or result for every real z is certified. |
