# TX3-X5-REV2: cross-model re-review of the two amended Lean statements, and the whole set for public release

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_x5`.

**Writable:** edits there are AUTHORIZED, but only under `artifacts/x5_rev2/` and in `docs/TX3_X5_REV2_*.md`. Scratch
may also go under `/tmp/x5_rev2_*`.

This is a REVIEW. Do not modify any `.lean` file, the lakefile, the manifest or any receipt.

## Context

- **TX3-X5-REV1** (`docs/TX3_X5_REV1_REPORT.md`, `bd261ad7`) judged the five original statements:
  - S1, S2, S4 and S5: FAITHFUL.
  - S3 (`splitk_contains`): MISMATCH. The receipt's counterexample has a sink of logit 0 and the statement had none
    (F1, MAJOR).
  - S6 was INADEQUATE for S2's witness, which did not exercise its claim's antecedent (F2, MINOR).
  - Overall: REJECT.
- **The lead's amendment** (`3fe7dcb0`) changed only those two statements, as REV1 proposed:
  - a new `rowOutputWithSink`;
  - `splitk_contains_stmt` now proves the receipt's sink-0 instance as well as the sink-free one;
  - `sp_mask_superset_nv_stmt` now adds an antecedent witness.
- **The proofs were updated** (TX3-X5B, `05e0ca75`). `lake build ApaCerts.Check` passes, and every axiom set is
  {propext, Classical.choice, Quot.sound}. The lead rebuilt it on 2026-10-28.
- **Current sha256:**
  - `Spec.lean` `36aeba97e236f8cc575cfe9a9f3f8b32f31c4b93b828746e3f8eb87eed3f8fe5`;
  - `Proofs.lean` `6ee2f428deb59c1ac7dcbaeff2974fd0b439df2cb6e123fa41d72d212ec5a3c7`;
  - `Check.lean` `2de6d3b0a503d7b06090a376d67d8e7ebb27f581ebe78e26e61000d01d78abed`.
- **Why now:** these statements are about to be published in the public repository and cited by the paper's v2. The
  trust boundary is the statements, so the amended ones need their own cross-model check.

## Mechanism & scope

- **Measures:**
  - whether F1 and F2 are closed at full strength;
  - whether the amendment left S1, S2's main statement, S4 and S5 exactly as REV1 approved them;
  - whether the full set of ten theorems is fit to record as machine-checked APA claims.
- **Mechanism doc:** `docs/TX3_X5_REV1_REPORT.md` (the findings and the proposed restatements). Also:
  - `artifacts/x5_rev1/findings.json`;
  - `orders/TX3_X5B_AMENDED_STATEMENTS.md`;
  - `/mnt/Shared/APA_Primer/APA_Primer.md` §4.1-§4.3.
- **Exercises:**
  - read `git diff bd261ad7 05e0ca75 -- lean/apa_certs/ApaCerts/Spec.lean`, the full current `Spec.lean`,
    `Proofs.lean` and `Check.lean`, and the receipts behind S3:
    - `/mnt/ForgeRealm/Project-Tensor/docs/APA_SP1_1_REPORT.md`;
    - `/mnt/ForgeRealm/Project-Tensor/tensor_cuda/tests/test_apa_sp1_1.py`;
    - `/mnt/ForgeRealm/Project-Tensor/tensor_cuda/tests/apa_sp1_reference.py`;
  - you may elaborate small Lean probe files under `/tmp/x5_rev2_*` with `export PATH=$HOME/.elan/bin:$PATH` and
    `lake env lean <file>` from `lean/apa_certs`. Never run `lake build` targets that write into `.lake/`.
- **Does not cover:**
  - proof correctness (the Lean kernel checked it);
  - floating-point behaviour (the statements are over ℝ by design);
  - kernel speed or correctness;
  - the repo's packaging.
- **Required evidence:**
  - S1 to S5 marked FAITHFUL, WEAKER or MISMATCH, with the lines read;
  - S6 per witness;
  - S7 overall;
  - for F1: the receipt's exact operands and both output values, recomputed against `rowOutputWithSink`;
  - every finding with a severity and a proposed restatement.
- **Operand census:**
  - **The sink-0 instance:** bulk [2,0], exact [2,10], V [0,1], δ = 0, singleton masks. The expected outputs are
    global 1/(e²+2) and split e¹⁰/(e²+e¹⁰+1).
  - **S2's antecedent witness:** a key with `finalRefined` true.
  - **One check per statement** that it would be vacuous or trivially true if mis-stated.
- **Baseline isolation:** the receipts are the baseline. REV1's verdicts on S1, S2-main, S4 and S5 hold only if the diff
  leaves those statements byte-identical in meaning.
- **Forbidden configuration:**
  - editing any `.lean` file, the lakefile, the manifest or any receipt;
  - `lake build` into `.lake/`;
  - network;
  - GPU;
  - git writes.

## Questions

- **S1 to S5:** one per statement. FAITHFUL / WEAKER / MISMATCH.
- **S6:** each `_nv` statement exercises its claim's hypotheses. ADEQUATE / INADEQUATE.
- **S7:** APPROVE / APPROVE-WITH-NITS / REJECT, for recording all five claims as machine-checked in a public repository
  and a paper.

## Registration checklist (order-lint)

- **L01:** the instrument is `lake env lean` from `lean/apa_certs`, toolchain `leanprover/lean4:v4.34.1` in `$HOME/.elan`.
  The lead ran `lake build ApaCerts.Check` with it on 2026-10-28 (exit 0).
- **L14:** n/a. There is no gain gate; "+2" in the census is the denominator e²+2.
- **L44:** the F1 values are recomputed from the receipt's raw operands. Neither the lead's figures nor REV1's are
  trusted.
- **L09:** identity is the Lean kernel's check of each proof against its statement. This review judges statement
  meaning.
- **L16:** the cheapest passing artifact is a true but weakened amendment, for example a sink term that is present but
  never used in the conjunct. That is what this review hunts.
- **L21:** the remote is public. Home paths as `$HOME/...`. No raw quotes from people.
- **L34:** a Codex seat on CPU.
- **L55 (exact-rules-checked-by-construction):** check one constructed instance per amended statement against its
  definitions (operand census).

## Rules

- Read `/mnt/Shared/HOUSE_RULES.md` and `/mnt/ForgeRealm/Project-Tensor/AGENTS.md` first. This order is immutable.
- NO subagents. No background waits. Every command runs under an explicit timeout.
- **Never delete anything:** move unwanted files of your own to `artifacts/x5_rev2/trash/`. `/tmp` is exempt.
- **Never walk a whole disk:** no recursive walk of `/`, `/mnt` or `/mnt/Shared`, and nothing under
  `/mnt/Shared/ForgeRealm_Link`.
- Prior Art Directive: name the source behind each finding.
- **Write early:** create `docs/TX3_X5_REV2_REPORT.md` and `artifacts/x5_rev2/findings.json` within your first few
  steps.

## Done

Your final message contains, verbatim:
- the S1 to S7 table;
- every finding in full;
- the F1 recomputation.
