# TX3-X5-REV2 statement fidelity review

APPROVE for recording the five precisely scoped claims and their five non-vacuity witnesses as machine-checked real-arithmetic APA claims. F1 and F2 are closed at the requested strength. No new or open findings.

Seat: Codex on CPU. The supplied order is the immutable plan; registration is `artifacts/x5_rev2/registration.md`. Evidence classes: source inspection, mathematical reasoning, scratch Lean elaboration, CPU Decimal illustration. Spec/Proofs/Check below mean `lean/apa_certs/ApaCerts/{Spec,Proofs,Check}.lean`. SP1.1 report/test/reference mean `/mnt/ForgeRealm/Project-Tensor/docs/APA_SP1_1_REPORT.md`, `/mnt/ForgeRealm/Project-Tensor/tensor_cuda/tests/test_apa_sp1_1.py`, and `apa_sp1_reference.py` in that tests directory. Primer is `/mnt/Shared/APA_Primer/APA_Primer.md`.

Read in full: Spec 1–143, Proofs 1–368, Check 1–35, the SP1.1 report 1–78, receipt test 1–158 and reference 1–191. Also read Primer 89–114, 238–299 and 454–483; supplemental partition reference 19–53; REV1 report/findings; X5B amended order. Source lines are retained in numbered excerpt artifacts. The two inherited findings below report closure, not current defects.

## S1 to S7

| Question | Statement | Verdict | Evidence |
|---|---|---|---|
| S1 | sp_skip_final | FAITHFUL | Unchanged. Spec 72–77; Proofs 118–137. |
| S2 | sp_mask_superset | FAITHFUL | Main claim unchanged; antecedent witness added. Spec 81–87; Proofs 141–171. |
| S3 | splitk_contains | FAITHFUL | Containment retained; sink-0 and sink-free gaps both required. Spec 54–58, 94–110; Proofs 194–253. |
| S4 | z_not_single_pass | FAITHFUL | Unchanged, including z=0 and z=√3. Spec 116–125; Proofs 267–301. |
| S5 | sp2_certificate | FAITHFUL | Unchanged full-key error premise and 2·errq margin. Spec 130–139; Proofs 307–339. |
| S6 | Each non-vacuity witness | ADEQUATE | S1 ADEQUATE; S2 ADEQUATE; S3 ADEQUATE; S4 ADEQUATE; S5 ADEQUATE. |
| S7 | All ten theorems for public recording | APPROVE | Both findings closed; no open findings. Check 13–35; passing Lean probe. |

## Findings

**F1 — CLOSED (original severity MAJOR): S3 omitted the receipt’s sink.** The amendment adds exp(sink) exactly once to the denominator, with no numerator contribution, and uses sink=0 in both outputs of the required >0.8 conjunct. Universal containment and the earlier sink-free gap are retained. Evidence: Spec 54–58, 94–107; Proofs 194–249; SP1.1 report 9–13; receipt test 67–74; reference 37–50; passing Lean probe and independent operand recomputation. Proposed restatement, now implemented: retain containment and require both the sink-0 and sink-free gaps for bulk [2,0], exact [2,10], V [0,1], δ=0 and singleton partitions. Prior art: SP1.1 R26 (2026), standard softmax sink normalization; review only, no new algorithm.

**F2 — CLOSED (original severity MINOR): S2’s witness missed its antecedent.** The amended conjunction requires an existential with finalRefined true and retains the strictness existential. Its positive witness is bulk [0,1], δ=0, key 1: rowMax=prefixMax=1, so finalRefined and spRefined both hold. Key 0 still witnesses strict containment. Evidence: Spec 81–87; Proofs 148–171; Check 15–16; independent Lean evaluation. Proposed restatement, now implemented: (∃ b δ j, finalRefined b δ j) ∧ (∃ b δ j, spRefined b δ j ∧ ¬finalRefined b δ j). Prior art: REV1 F2 (2026), SP1 R24 (2026), BLASST’s running-max criterion; review only, no new algorithm.

## F1 recomputation

Raw receipt operands: bulk [2,0], exact [2,10], V [0,1], δ=0, partitions {0}/{1}, sink logit 0. Inclusive prefix maxima give global mask [true,false] and split mask [true,true], hence mixed scores [2,0] and [2,10]. The sink contributes exp(0)=1 once and contributes zero to the numerator.

Using rowOutputWithSink:

- Global: (e²·0 + e⁰·1)/(e² + e⁰ + e⁰) = 1/(e²+2) ≈ **0.106506978919200751**.
- Split: (e²·0 + e¹⁰·1)/(e² + e¹⁰ + e⁰) = e¹⁰/(e²+e¹⁰+1) ≈ **0.999619282443397359**.
- Absolute gap ≈ **0.893112303524196608 > 0.8**.

The retained sink-free outputs are 0.119202922022117556 and 0.999664649869533522, with gap 0.880461727847415966. Operands were extracted from the original test by AST, not copied from the lead’s figures. Decimal values use 80 digits; Lean separately checks the exact formulas and both strict gap conjuncts.

## Baseline isolation and statement strength

The repeated NO git rule prevented running the requested `git diff bd261ad7 05e0ca75`. Instead, reconstructed all 131 original Spec lines from `artifacts/x5_rev1/source_excerpts.txt` and verified SHA256 `1bf0a7d621912ca6e6a537e01041ea6de18e1304c00b5f89e78917ed97d8a01d`, matching REV1’s report. The complete unified diff is `spec_vs_rev1.diff`. This establishes source identity relative to REV1’s archive; no git object or commit provenance was independently checked.

Every old definition body is byte-identical after removing comments except `splitk_contains_stmt` and `sp_mask_superset_nv_stmt`. `rowOutputWithSink` is the only added definition. Thus S1, S2-main, S4 and S5 retain their baseline definitions and dependencies. Their original `_nv` definitions also remain unchanged. S3’s universal containment and original operands remain intact. The old sink-free output claim is nested in the new conjunction rather than weakened.

A scratch Lean file imports the compiled Check module, reproduces the full current and REV1 Spec in separate namespaces, and proves 19 current-source/imported definition equalities plus 16 unchanged REV1/current equalities by rfl. It also checks new-S3 ⇒ old-S3 and new-S2-nv ⇒ old-S2-nv by conjunction projection. These checks connect the source review to the imported statements and demonstrate retention of the earlier strength; they do not assume that a sink helper’s mere existence fixes F1.

S1: nonempty inclusive prefixes, signed bulk, strict skip from negating an inclusive refine threshold; same skipped key is compared with every later prefix and final max. All real margins remain quantified.

S2: finalRefined implies spRefined, not equality of masks or outputs. Antecedent and strictness witnesses are separate existentials; they need not be the same key.

S3: arbitrary finite P containing j generalizes ordered partition containment; P intersect the inclusive global prefix contains j, so there is no empty-maximum default. No partition-cover hypothesis is needed for the per-key order result. The fixed output comparison uses a genuine covering pair of singletons and one sink after aggregation.

S4: for each fixed z in [0,sqrt 3], the common first prefix cannot support irrevocable agreement on both rows. This is not an impossibility with unrestricted buffering or a claim for every real z.

S5: every key has bounded bulk/exact error. Nonempty row implies errq>=0 from that premise. epsilon>0 is sufficient; allowing epsilon>1 generalizes the receipt without weakening it. The strict skipped-key conclusion concerns true weights, not bulk or mixed weights. A common positive softmax denominator (including an unchanged sink) cancels. No aggregate error or empirical universal quantization bound is asserted.

## S6 and one vacuity/triviality attack per statement

**sp_skip_final_nv — ADEQUATE.** bulk=[1,0], delta=0, j=1: 0<1 is an active skip. Sources: Spec 76–77; Proofs 130–137. Attack/check: A one-key delta=0 row has no skip and would make an implication vacuous. The actual witness has two keys. A separate [2,1,3], j=1, k=2 instance exercises a strictly later prefix (Lean-checked).

**sp_mask_superset_nv — ADEQUATE.** bulk=[0,1], delta=0; j=1 final-refines and prefix-refines; j=0 prefix-refines but does not final-refine. Sources: Spec 85–87; Proofs 148–171. Attack/check: Strictness alone makes the main implication antecedent false at j=0. New first conjunct and the independently evaluated j=1 witness close that hole.

**splitk_contains_nv — ADEQUATE.** bulk=[2,0], delta=0, P={0}, j=0 is a member and is globally refined. Output construction is fixed, without a conditional antecedent. Sources: Spec 109–110; Proofs 251–253. Attack/check: P=univ makes containment equality and could hide a reversed inequality. Separate bulk=[2,0,1], P={1,2}, j=1 gives local prefix 0<global prefix 2. Lean checks local=0. V=0 would make output equality trivial; actual V=[0,1] and changed second score avoid this.

**z_not_single_pass_nv — ADEQUATE.** z=0 lies in [0,sqrt 3]. The same pair is checked at z=sqrt 3 as well. Sources: Spec 124–125; Proofs 299–301, 346–352. Attack/check: Contradictory z bounds would vacuate the universal claim. z=0 and sqrt 3 satisfy actual bounds. A=[1,0,0,0] has threshold (1+z sqrt 3)/4, at most 1 on the interval; B=[1,0,0,10] has mean 11/4>1 and nonnegative added term. Same first input 1 forces any f to the same decision.

**sp2_certificate_nv — ADEQUATE.** bulk=[10,0], exact=[9,1], errq=1, epsilon=1/2, j=1. Both errors are exactly 1; delta=log 2+2<=3, so 0<10-delta is an active skip. Sources: Spec 137–139; Proofs 325–339. Attack/check: epsilon>1 could make the weight bound trivial. Actual witness uses epsilon=1/2 and satisfies every-key error bound and skip together; independent Lean evaluation checks all premises.

## Verification and claim boundary

Live toolchain: `leanprover/lean4:v4.34.1`, Lean commit `5045d0056413266e57c625dcd7c365b10e377c52`, installed in `$HOME/.elan`. Instrument from `lean/apa_certs`:

```sh
timeout 120s bash -c 'export PATH="$HOME/.elan/bin:$PATH"; lake env lean /tmp/x5_rev2_probes/Probe.lean'
```

Final probe exit 0, no warnings. It checks all ten proof types and prints exactly `{propext, Classical.choice, Quot.sound}` for each theorem. Its formula checks evaluate the real definitions; Decimal outputs are illustrations, not floating-point validation. The lead’s 2026-10-28 `lake build ApaCerts.Check` exit 0 is supplied evidence; no build was rerun here. Two earlier scratch probe attempts exited 1 due to tactic construction errors. Both full logs and source versions remain, with causes in the ledger. They are not failures of the reviewed proofs.

All 14 initially hashed input files are unchanged. Current hashes before and after:

| File | SHA256 |
|---|---|
| Spec.lean | `36aeba97e236f8cc575cfe9a9f3f8b32f31c4b93b828746e3f8eb87eed3f8fe5` |
| Proofs.lean | `6ee2f428deb59c1ac7dcbaeff2974fd0b439df2cb6e123fa41d72d212ec5a3c7` |
| Check.lean | `2de6d3b0a503d7b06090a376d67d8e7ebb27f581ebe78e26e61000d01d78abed` |

All 144231 observed .lake file paths/sizes/mtimes remained unchanged across the probes. This is metadata write detection, not a full content hash of .lake. No reviewed Lean file, lakefile, manifest or receipt was modified. No git command, network, GPU, build, subagent, background process, deletion or move was used. Scratch remains at `/tmp/x5_rev2_probes`; no trash was produced.

Public release wording must retain the exact scopes above: mask containment does not establish output equality; the split-K output result is the fixed scalar counterexample; the z-rule obstruction is for irrevocable prefix decisions on the stated interval; SP2 is conditional on every-key error bounds. This review covers statement meaning. It makes no new proof-correctness assessment or claims about floating-point behavior, production kernel correctness, speed, model quality or packaging. Not claimed fixed: nothing remains open within F1/F2; excluded subjects were not tested.

## Prior art

SP1/SP1.1/SP2 receipts (2026); BLASST (Yuan, Shinn et al., 2025/2026); online softmax (Milakov and Gimelshein, 2018); FlashAttention (Dao et al., 2022); Mathlib finite-supremum facts. Taken: existing rules, operands and normalization; new: independent fidelity review only. External literature unverified — lead to check: “BLASST running maximum”, “online normalizer calculation for softmax”. Source-site annotations are in the scratch probe and arithmetic driver, with matching ledger entries.

## Artifact ledger

- `registration.md`: immutable pre-gate criteria; `ledger.md`: commands, outcomes, prior art, failures and decisions.
- `findings.json`: complete verdicts, closures, witness attacks, scope and arithmetic values.
- `spec_vs_rev1.diff`, `rev1_spec_source.txt`, `baseline_identity.json`: hash-verified baseline comparison.
- `source_excerpts.txt`, `partition_reference_excerpts.txt`: numbered source evidence.
- `input_hashes_before.json`, `input_hashes_after.json`, `supplemental_input_hashes.json`: input identity.
- `probe_source.txt`, `probe_attempt1_source.txt`, `probe_attempt2_source.txt`, `lean_probe_attempt1.log`, `lean_probe_attempt2.log`, `lean_probe_attempt3.log`: all probe attempts.
- `recompute_outputs.py`, `splitk_outputs.json`: independent raw-operand recomputation.
- `lake_metadata_before.json`, `verification.json`: integrity and Lean instrument results.
- `final_response.txt`: table, full finding closure text and recomputation, verbatim for final handoff.
