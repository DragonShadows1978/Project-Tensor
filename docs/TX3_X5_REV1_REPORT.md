# TX3-X5-REV1 statement fidelity review

REJECT for recording all five original receipt claims verbatim: S3 uses a sink-free counterexample instead of the receipt’s sink-logit-zero instance. The qualitative non-equality result remains proved. S2’s non-vacuity witness has a separate, minor coverage defect. No proof is alleged incorrect.

Seat: Codex, CPU only. The supplied order is the immutable plan; registration is `artifacts/x5_rev1/registration.md`. Baseline isolation: receipts, not statement comments. Evidence classes below distinguish live source inspection, reasoning, Lean instantiation and illustrative CPU arithmetic. No network, GPU, git, lake build, reviewed-file edits, deletions, or service actions.

Source abbreviations: Spec/Proofs/Check are `lean/apa_certs/ApaCerts/{Spec,Proofs,Check}.lean`. Primer is `/mnt/Shared/APA_Primer/APA_Primer.md`. MUTATIONS is `/mnt/ForgeRealm/reference/openai-math-notes/receipts/tx3_opus/MUTATIONS.md`. Kernel is `/mnt/ForgeRealm/Project-Tensor/tensor_cuda/src/kernels.cu`. SP1.1 report, tests, SP2 derivation and SP1 registration are in `/mnt/ForgeRealm/Project-Tensor/` at the paths in findings.json/source_excerpts.txt. Primer R24 resolves to SP1; R26 to SP1.1; R06 to SP2 derivation. Source line numbers refer to the live inputs snapshotted in this review.

Read in full: Spec 1–131, Proofs 1–329, Check 1–35. Read the requested Primer sections 1 and 4.1–4.3, plus its source/prior-art tables; mechanism-relevant production kernel passes 1 and 2 at 1580–1750. Spec SHA256 matches the supplied prefix: `1bf0a7d621912ca6e6a537e01041ea6de18e1304c00b5f89e78917ed97d8a01d`.

## S1 to S7

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

## Operand census and non-vacuity attacks

These are checks against the actual definitions, not inference from lead-written comments. The scratch Lean probe imports the reviewed compiled module without rebuilding; hashes of the source were recorded. Source inspection supplies the mathematical reading. CPU decimal numbers are illustrations only.

| Claim | Constructed instance and actual definition result | Vacuity/triviality attack and its disposition |
|---|---|---|
| S1 | b=[2,1,3], δ=0, j=1, k=2. prefixMax at j=2, at k=3, rowMax=3; 1<2 and 1<3. | With a one-key row and δ=0 no skip exists, so the implication alone would be vacuous. Actual nv uses [1,0], δ=0, j=1: 0<1, an active skip (Proofs 112–119). The two-key witness cannot give a proper later k, but the three-key instance does. |
| S2 | b=[0,1], δ=0, j=1: prefixMax=rowMax=1, so both masks refine. | At j=0 SP refines and final does not: useful strictness, but final⇒SP is vacuous there. That is precisely the actual nv witness (F2). Scratch Lean checks both its negative-antecedent type and the positive j=1 instance. |
| S3 | b=[2,0,1], δ=0, P={1,2}, j=1: global prefix=2, local prefix=0; global skips, local refines. At j=2 local prefix=1, global=2. P is not a singleton. The two-key output conjunct uses [2,0]/[2,10]/[0,1] and separate singletons, as described in F1. | P=univ gives local=global and could conceal reversed containment; arbitrary P in the statement includes the proper subset example. P=empty cannot supply hj; no key is excluded that would need a decision. V=[0,0] or exact=bulk would make outputs equal, but the actual fixed V=[0,1] and exact second score 10 avoid both. Actual nv uses b=[2,0], δ=0, P={0}, j=0, satisfying membership and global refinement (Proofs 212–214): ADEQUATE. |
| S4 | A=[1,0,0,0], B=[1,0,0,10], key 0. At both z=0 and z=√3, A refines and B skips, with exact formulas in the z comparison. | Contradictory bounds on z would make the universal statement vacuous. Actual nv chooses z=0, inside the closed interval (Proofs 260–262): ADEQUATE. Requiring the rows to differ only at unobserved key 3 ensures the f(1) impossibility is substantive, not a comparison of distinct prefixes. At z>√3 the same A first key skips, so this fixed pair does not certify that wider parameter range. |
| S5 | Actual nv: bulk=[10,0], exact=[9,1], errq=1, ε=1/2, j=1. Both absolute errors equal 1; δ=ln2+2≤3, and 0<10−δ. Conclusion exp(1)<exp(9)/2 has an active skip. For errq=0, bulk=exact=[1,0], ε=1, δ=0, j=1 gives an active skip and exp(0)<exp(1); scratch Lean checks the error and skip premises. | ε>1 would make exp(exact_j)<ε exp(max exact) trivially true even at the maximum. Actual nv uses ε=1/2, avoiding that shortcut. With ε≤1 and a single key, δ≥0 and no skip exists; nv uses two keys. ADEQUATE (Proofs 286–299). |

One-key boundary: n=0 means one key, not an empty row. For b=[5], δ=0, prefixMax=rowMax=5 and both masks refine by equality; zRefined also refines for every real z because variance=0. The scratch probe checks the SP/final case. No positive softmax denominator or maximum relies on an empty-set default.

## Quantifiers and strength

- S1 and S2 quantify over all real margins, not only δ≥0. Their order implications remain true; this is a valid generalization, not an extra hypothesis. S1 quantifies all later prefixes, with j≤k correctly guarding only the later-prefix conclusion; the final conclusion is unconditional after a skip. S2 is the contrapositive final-mask consequence, not online/offline output equality or z-mask equality.
- S3’s P is a finite subset, not a complete partition family. Membership and index order suffice to prove each local comparison, so allowing arbitrary P strengthens containment without requiring a cover, disjointness, or singleton partitions. Its output witness is an ideal scalar materialized softmax; it does not claim correctness of the production merge or any fixed production partition-size launcher. The sink setting differs from the receipt as F1 records.
- S4 fixes z before quantifying f; f may therefore be chosen for that z and fixed row length. Any such function of the observed prefix must have a single truth value at 1. The predicate formulation is at least as general as a Boolean refinement decision. The supported receipt uses zthr=1.0364333894937898, inside the formal interval. Neither unrestricted buffering nor all z values are established; no such expanded claim is approved.
- S5 requires ε>0. The receipt additionally describes ε≤1, but the theorem remains valid for ε>1; dropping the upper bound is a valid generalization. The all-key error premise and nonempty row imply errq≥0, so no explicit nonnegativity premise is missing. Over ℝ, all scores and errq are finite; NaN/infinity are not operands.
- S5 uses the inclusive prefix skip, exactly as SP2 derivation 23–38. MUTATIONS 300 uses a final bulk maximum shorthand, but the baseline is the receipt, not the proposal. Under the receipt’s error bound the same certificate also holds for a final-max skip (use a final bulk argmax); the statement correctly targets the production prefix decision. No replacement of true weights by bulk or mixed-score weights occurs.
- Normalized true weights w_j=exp(exact_j)/Z and w*=exp(max exact)/Z share a positive Z, so S5’s unnormalized inequality is equivalent after cancellation. An unchanged sink also cancels. This is why F1 is specific to the output counterexample, not SP2. Aggregate mass, perturbation, model quality and empirical universal error bounds are not among these five certificate claims.

## Prior art

F1 uses APA SP1.1 (2026), Primer R26 and its fixed counterexample; the denominator/merge convention is standard softmax and online softmax (Milakov and Gimelshein 2018; FlashAttention, Dao et al. 2022). F2 uses APA SP1 (2026), R24, the X5 witness requirement and BLASST’s running-max criterion (Yuan, Shinn et al., arXiv 2512.12087, 2025/MLSys 2026). Taken versus new, source-site comment recommendations, and literature-check leads appear in each finding and in the ledger. The z pair is SP1’s (2026); SP2’s 2·errq construction is R06’s. No new algorithm, selector or optimization is proposed. External literature was not independently checked; unverified — lead to check: “BLASST running maximum”, “Milakov Gimelshein online softmax 2018”.

## Verification, artifacts and limits

- `findings.json`: structured complete verdicts, findings, definition checks and z-rule comparison.
- `source_excerpts.txt`, `input_hashes.json`, `supplemental_input_hashes.json`: numbered source evidence and input identity.
- `probe_source.txt`: archival text of the new scratch-only Lean probe; no repository .lean file was modified or created.
- `lean_probe_attempt1.log` through `lean_probe_attempt4.log`: first three scratch tactic/type errors retained, final attempt exit 0 and empty output. These were probe-development errors, not failures of reviewed proofs. The command was `timeout 90s bash -c 'export PATH="$HOME/.elan/bin:$PATH"; lake env lean /tmp/x5_rev1_probes/Probe.lean'` from lean/apa_certs. No `lake build` ran.
- `splitk_outputs.json`: 70-digit Decimal illustrations of the two real formulas. These numbers are not a numerical-kernel test. Both real inequalities can hold while the encoded instance differs.
- `verification.json`: 144231 .lake files unchanged by size/mtime before/after probes, no input SHA256 changes. Full .lake contents were not rehashed; this is a write-detection check, not a fresh build certification.
- `registration.md` is immutable; `ledger.md` records execution, failures, findings and prior art. Scratch remains at `/tmp/x5_rev1_probes`; nothing was deleted or moved. The large .lake snapshot is an authorized artifact, not a modified build tree.

No remediation implemented: reviewed Lean files, lakefile, manifest and receipts remain unchanged. Not claimed fixed: F1 exact-receipt mismatch and F2 witness coverage. Proposed successor: add a sink-zero counterexample statement and an antecedent-satisfying S2 witness, then check those additions separately under lead authorization. This review makes no assertion about floating-point behavior, production kernel correctness, speed or model quality.
