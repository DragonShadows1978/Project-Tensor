# TX3-X5: machine-check five APA certificates in Lean (proofs only; the statements are the lead's)

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_x5`
(branch `x5-apa-lean`, at the commit carrying this order). Edits and CPU runs are AUTHORIZED there, but only on these
paths:
- `lean/apa_certs/ApaCerts/Proofs.lean` (new; your proofs);
- `lean/apa_certs/.lake/` (Lake's build output for this project only);
- receipts under `artifacts/x5/`;
- scratch under `/tmp/x5_*`.

**Read-only:**
- `lean/apa_certs/ApaCerts/Spec.lean` and `lean/apa_certs/ApaCerts/Check.lean` (lead-written);
- `lakefile.lean`, `lean-toolchain` and `lake-manifest.json`;
- the copied packages under `lean/apa_certs/.lake/packages/` (do not rebuild or modify them);
- `~/.elan` (the shared toolchain, used read-only);
- `~/omath-lean` (never build there);
- every other path.

## Why

TX3's mutation pass proposed X5 (`/mnt/ForgeRealm/reference/openai-math-notes/receipts/tx3_opus/MUTATIONS.md`, X5):
turn the APA arc's "reasoning/proof" receipts into machine-checked statements.
- **Where the claims come from:** the APA primer (`/mnt/Shared/APA_Primer/APA_Primer.md` §4.1-§4.3) records five
  claims checked only by prose and unit tests:
  - the SP running-max rule is monotone, so it is exactly single-pass;
  - its mask is a superset of the final-max mask;
  - split-K selection contains the global rule, but the outputs can differ;
  - the z-score rule cannot run in one irrevocable pass;
  - the SP2 δ certificate.
- **Why X5 first:** the owner directed on 2026-10-28 that these ideas be tested first. X5 is the cheapest of the APA ideas: no GPU and
  no engine change.

The trust boundary is the statement. The lead wrote every statement before any proof, in
`lean/apa_certs/ApaCerts/Spec.lean` (sha256 `1bf0a7d621912ca6e6a537e01041ea6de18e1304c00b5f89e78917ed97d8a01d`).
- Each statement is matched to its primer line in the file's comments.
- The z-score definition mirrors the production kernel's arithmetic (`kernels.cu` at b4da646), with real numbers in
  place of floats.
- `Spec.lean` elaborates cleanly against the pinned Mathlib (lead run, 11 s).

## Decisions (binding; the lead chose these)

- **D1, proofs only.** Write `lean/apa_certs/ApaCerts/Proofs.lean`, beginning `import ApaCerts.Spec`, with these ten
  theorems in `namespace ApaCerts`, each with exactly its stated type:
  - `sp_skip_final : sp_skip_final_stmt` and `sp_skip_final_nv : sp_skip_final_nv_stmt`;
  - `sp_mask_superset` and `sp_mask_superset_nv`;
  - `splitk_contains` and `splitk_contains_nv`;
  - `z_not_single_pass` and `z_not_single_pass_nv`;
  - `sp2_certificate` and `sp2_certificate_nv`.

  Helper lemmas are allowed in `Proofs.lean`. No `sorry`, `admit`, `axiom`, `opaque`-based escape,
  `implemented_by`, `extern`, `unsafe`, `native_decide`, or `set_option` that weakens checking.
- **D2, the check.** `lake build ApaCerts.Check`, run in `lean/apa_certs`, must succeed with no `sorry` warning.
  - `Check.lean` (sha256 `2de6d3b0a503d7b06090a376d67d8e7ebb27f581ebe78e26e61000d01d78abed`) pins every proof to its
    statement with `example : <stmt> := <proof>`.
  - It prints each theorem's axioms. Only `propext`, `Classical.choice` and `Quot.sound` are allowed.
- **D3, offline and no rebuild.**
  - The packages are a copy of the house build of Mathlib `d13f23b`.
  - Run no `lake update`, no `lake exe cache get` and no network.
  - If Lake starts rebuilding Mathlib or any dependency (the first build of `ApaCerts` should take minutes, not
    hours), stop at once and report it; the copy is wrong.
  - Put `export PATH=$HOME/.elan/bin:$PATH` before the Lake commands.
- **D4, a statement that seems false or mis-stated.** If you believe a statement is false, or does not say what its
  primer receipt means, stop work on that one and report it with a counterexample or the reason. Never weaken,
  restate or work around a statement; only the lead edits `Spec.lean`.

## Mechanism & scope

- **Measures:** whether the five APA claims, as the lead stated them, hold, together with their non-vacuity witnesses,
  machine-checked by Lean 4.34.1 with Mathlib `d13f23b`.
- **Mechanism doc:** `/mnt/Shared/APA_Primer/APA_Primer.md` (§4.1-§4.3, the receipts the statements formalize).
  Also `lean/apa_certs/ApaCerts/Spec.lean` (the statements and their definitions).
- **Exercises:** `lake build ApaCerts.Check` in `lean/apa_certs`, and the axiom printout.
- **Does not cover:**
  - floating-point behaviour (the statements are over ℝ);
  - any kernel's correctness;
  - GPU execution;
  - the other APA ideas X3a and X4.
- **Required evidence:**
  - the full `lake build ApaCerts.Check` log, with its wall time;
  - the axiom printout;
  - a grep of `Proofs.lean` for every forbidden token in D1 (zero hits);
  - the sha256 of `Spec.lean`, `Check.lean`, `lakefile.lean`, `lean-toolchain` and `lake-manifest.json` before and
    after (unchanged);
  - `Proofs.lean` itself.
- **Operand census:** the ten theorems; each statement's boundary cases are named in its comment:
  - z exactly at 0 and at √3;
  - δ = 0;
  - a single-key row;
  - `errq` = 0.
- **Baseline isolation:** `Spec.lean` and `Check.lean` are the baseline, pinned by the hashes above. Mathlib and its
  dependencies are untouched.
- **Forbidden configuration:**
  - any edit outside the writable paths above;
  - any network access;
  - rebuilding Mathlib;
  - git writes;
  - GPU use.

## Gates

- **X5-G1:** `lake build ApaCerts.Check` exits 0, with no `sorry` warning.
- **X5-G2:** every printed axiom set is a subset of {`propext`, `Classical.choice`, `Quot.sound`}.
- **X5-G3:** the five lead-file hashes are unchanged. `Proofs.lean` has zero hits for the D1 forbidden tokens.

**Registered acceptance:** ACCEPT when X5-G1 to X5-G3 pass for all ten theorems. A statement reported under D4
counts as a result for that statement, not a pass.

## Lead predictions (registered here, before any proof)

- **P16:** all ten theorems are proved. The receipt's prior is high: each claim is a few lines of real analysis or
  order reasoning.

## Registration checklist (order-lint)

- **L01:** the build command is `cd lean/apa_certs && export PATH=$HOME/.elan/bin:$PATH && lake build ApaCerts.Check`.
- **L09:** identity means the Lean kernel's definitional check of each proof against its statement (`example`
  pinning in `Check.lean`). No file identity is claimed beyond the five lead-file hashes.
- **L15:** n/a. No art or shape bar; P16 is a prediction.
- **L16:**
  - **Weakened statements:** the cheapest passing artifact is a proof of weakened statements. They are excluded
    because the statements are lead-written, hash-pinned and checked by `Check.lean`.
  - **Vacuous hypotheses:** excluded by the ten non-vacuity witnesses.
  - **Escape hatches:** excluded by the axiom check and the token grep.
- **L20:** nothing is tuned.
- **L21:** public remote. Home paths as `$HOME/...`. No raw quotes from people.
- **L34:** a Claude seat on CPU. Lean elaboration is single-process, with peak memory a few GB.
- **L41:** the check goes through the real `lake build`.
- **L44:** the axiom list in the report is copied from the build log, not retyped.
- **L55 (exact-rules-checked-by-construction):** every claim here is an exact mathematical statement. Each is
  constructed by the lead, then checked by the Lean kernel; the witnesses construct instances where the hypotheses
  hold.

## Rules

- Read `/mnt/Shared/HOUSE_RULES.md` and `/mnt/ForgeRealm/Project-Tensor/AGENTS.md` first. This order is immutable.
- NO subagents. No background waits. Every command in the foreground under an explicit timeout. **Never delete
  anything.**
- No git. No GPU. No network.
- A failed gate is a result. Report it RED with receipts.
- **Prior Art Directive (AGENTS.md):** annotate each proof technique, or write "no prior art known to me". Candidates
  include:
  - Mathlib's `Finset.sup'` lemmas;
  - the BLASST criterion (arXiv 2512.12087) behind the SP rule;
  - the standard exp/log monotonicity bounds.
- **Write early:** create `artifacts/x5/FINAL_REPORT.md` and `artifacts/x5/LEDGER.md` within your first few steps.

## Done

Your final message contains, verbatim:
- X5-G1 to X5-G3 with receipts;
- the axiom printout;
- the list of the ten theorems proved, or any reported under D4;
- the hashes before and after;
- every file written.
