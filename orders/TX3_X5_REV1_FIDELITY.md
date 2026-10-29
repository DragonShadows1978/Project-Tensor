# TX3-X5-REV1: cross-model review of statement fidelity. Do the five Lean statements say what the APA receipts claim?

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_x5`.
Edits there are AUTHORIZED, but only under `artifacts/x5_rev1/` and `docs/TX3_X5_REV1_*.md`. You may also write scratch
under `/tmp/x5_rev1_*`. This is a REVIEW: do not modify any `.lean` file, lakefile, manifest or receipt.

## Context

TX3-X5 machine-checked five APA claims in Lean 4.34.1 with Mathlib `d13f23b` (`a08e1d74`):
- `lake build ApaCerts.Check` passes;
- every axiom set is {propext, Classical.choice, Quot.sound};
- there are ten theorems: five claims and five non-vacuity witnesses.

The proofs are kernel-checked, so their correctness is not in question here. **The trust boundary is the
statements.** The lead wrote them in `lean/apa_certs/ApaCerts/Spec.lean` (sha256 `1bf0a7d6…`) before any proof.
A statement can be true yet not be the claim its receipt makes. Your job is to find any such gap.

## Mechanism & scope

- **Measures:** for each of the five statements, whether it formalizes the claim in its receipt faithfully and at full
  strength. Check:
  - **definitions:** `prefixMax`, `rowMax`, `localPrefixMax`, `spRefined`, `finalRefined`, `mixedScores`,
    `rowOutput` and `zRefined`;
  - **quantifiers and hypotheses:** not stronger than the receipt needs, and not weaker;
  - **the boundary conventions:** ≤ versus <, zero-based indices, nonempty rows;
  - **the z-score rule against the production kernel**, in real arithmetic;
  - **the SP2 certificate's form:** true weight versus bulk weight;
  - **the split-K counterexample's construction against the receipt's.**
- **Mechanism doc:** `/mnt/Shared/APA_Primer/APA_Primer.md` (§1 and §4.1-§4.3, the receipts). Also:
  - `/mnt/ForgeRealm/reference/openai-math-notes/receipts/tx3_opus/MUTATIONS.md` (X5's proposal);
  - `lean/apa_certs/ApaCerts/Spec.lean`.
- **Exercises:**
  - read `Spec.lean`, `Proofs.lean` and `Check.lean`, the primer sections, and the production kernel's z-score code
    (`/mnt/ForgeRealm/Project-Tensor/tensor_cuda/src/kernels.cu`, `apa_selective_kernel` pass 1 and pass 2, read-only);
  - CPU checks are allowed: you may elaborate small Lean probe files under `/tmp/x5_rev1_*`, using
    `export PATH=$HOME/.elan/bin:$PATH` and `lake env lean <file>` from `lean/apa_certs`. Do not run `lake build`
    targets that write into `.lake/`.
- **Does not cover:**
  - proof correctness (the kernel checked it);
  - floating-point behaviour (the statements are over ℝ, by design);
  - kernel speed or correctness.
- **Required evidence:**
  - per statement: FAITHFUL, WEAKER (with the receipt meaning it misses) or MISMATCH (with a concrete reading
    showing it), plus the lines read;
  - a per-definition check;
  - the z-rule comparison with kernel line numbers;
  - every finding with severity (BLOCKING / MAJOR / MINOR / NIT) and a proposed restatement.
- **Operand census:**
  - boundary cases: z at 0 and at √3, δ = 0, a single-key row, errq = 0, a partition that is not a singleton;
  - for each statement, one instance where it would be vacuous or trivially true if mis-stated, checked against the
    actual definition.
- **Baseline isolation:** the receipts are the baseline. Treat the lead's statement comments as claims to check.
- **Forbidden configuration:**
  - editing any `.lean` file, lakefile, manifest or receipt;
  - `lake build` into `.lake/`;
  - network;
  - GPU;
  - git writes.

## Questions

- **S1 to S5:** one per statement (`sp_skip_final`, `sp_mask_superset`, `splitk_contains`, `z_not_single_pass`,
  `sp2_certificate`). FAITHFUL / WEAKER / MISMATCH.
- **S6, the witnesses:** each `_nv` statement actually exercises its claim's hypotheses. ADEQUATE / INADEQUATE.
- **S7, overall:** APPROVE / APPROVE-WITH-NITS / REJECT, for recording these five claims as machine-checked.

## Registration checklist (order-lint)

- **L09:** identity is the Lean kernel's check of each proof against its statement. This review judges statement
  meaning, not identity.
- **L16:** the cheapest passing artifact is a set of true but weakened statements. That is exactly what this review
  hunts.
- **L21:** public remote. Home paths as `$HOME/...`. No raw quotes from people.
- **L34:** a Codex seat on CPU.
- **L55 (exact-rules-checked-by-construction):** each statement is an exact rule. Check one constructed instance per
  statement against its definitions (operand census).

## Rules

- Read `/mnt/Shared/HOUSE_RULES.md` and `/mnt/ForgeRealm/Project-Tensor/AGENTS.md` first. This order is immutable.
- NO subagents. No background waits. Every command under an explicit timeout. Never delete anything.
- Prior Art Directive: name the source behind each finding.
- **Write early:** create `docs/TX3_X5_REV1_REPORT.md` and `artifacts/x5_rev1/findings.json` within your first few
  steps.

## Done

Your final message contains, verbatim:
- the S1 to S7 table;
- every finding in full;
- the per-definition check;
- the z-rule comparison.
