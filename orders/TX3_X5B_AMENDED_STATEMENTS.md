# TX3-X5B: amendment to X5. Prove the two statements the lead amended after the fidelity review

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_x5`
(branch `x5-apa-lean`, at the commit carrying this amendment). The writable paths are as in
`orders/TX3_X5_APA_LEAN.md`:
- `lean/apa_certs/ApaCerts/Proofs.lean`;
- `lean/apa_certs/.lake/build`;
- receipts under `artifacts/x5b/`;
- scratch `/tmp/x5b_*`.

`orders/TX3_X5_APA_LEAN.md` stays binding in every point not changed here: D1 to D4, the gates, the rules and the
forbidden tokens.

## Why

The fidelity review TX3-X5-REV1 (`docs/TX3_X5_REV1_REPORT.md`, `bd261ad7`) found two statement defects, both the
lead's.
- **F1, MAJOR:** the split-K receipt's counterexample has an attention sink of logit 0 in the softmax denominator, and
  the Lean statement had none.
- **F2, MINOR:** `sp_mask_superset_nv` did not exercise its claim's antecedent.

The lead amended `lean/apa_certs/ApaCerts/Spec.lean` (now sha256
`36aeba97e236f8cc575cfe9a9f3f8b32f31c4b93b828746e3f8eb87eed3f8fe5`; it elaborates cleanly):
- **New definition** `rowOutputWithSink scores V sink = (Σ exp(s_i)·V_i) / (Σ exp(s_i) + exp sink)`.
- **`splitk_contains_stmt`:** the output conjunct is now a conjunction. One conjunct is the receipt's instance with
  sink logit 0; the other is the same instance without a sink, which the earlier statement already proved.
- **`sp_mask_superset_nv_stmt`:** now the conjunction of an antecedent witness (`∃ b δ j, finalRefined b δ j`) and the
  earlier strictness witness.

Nothing else changed, and `Check.lean` is unchanged.

## Change (binding)

- **C1.** Update `Proofs.lean` so that `splitk_contains` and `sp_mask_superset_nv` prove the amended statements.
  Change no other theorem's proof unless the build forces it, and say why if it does. Then rerun X5-G1 to X5-G3 with
  the new `Spec.lean` hash in the hash check.

## Mechanism & scope

- **Measures:** whether the amended statements hold, machine-checked as in X5.
- **Mechanism doc:**
  `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_x5/docs/TX3_X5_REV1_REPORT.md`
  (the findings). Also `orders/TX3_X5_APA_LEAN.md`.
- **Exercises:** `lake build ApaCerts.Check` in `lean/apa_certs`.
- **Does not cover:** as X5.
- **Required evidence:** as X5, with the new `Spec.lean` hash before and after.
- **Operand census:** the sink-0 instance's two outputs: global 1/(e²+2) and split e¹⁰/(e²+e¹⁰+1). Their difference
  is about 0.893.
- **Baseline isolation:** the amended `Spec.lean` (`36aeba97…`) and the unchanged `Check.lean` (`2de6d3b0…`) are the
  baseline.
- **Forbidden configuration:** as X5.

## Gates

X5-G1 to X5-G3, unchanged, on the amended `Spec.lean`.

**Registered acceptance:** ACCEPT when all three pass for all ten theorems.

## Registration checklist (order-lint)

- **L01:** `cd lean/apa_certs && export PATH=$HOME/.elan/bin:$PATH && lake build ApaCerts.Check`.
- **L09:** as X5. The Lean kernel checks each proof against its statement.
- **L16:** as X5. The statements are lead-written and pinned.
- **L21:** public remote. No personal names or raw quotes in any file you write.
- **L34:** a Claude seat on CPU.
- **L55 (exact-rules-checked-by-construction):** the sink-0 instance's outputs are computed by hand above, and the
  kernel checks the proof.

## Rules

As X5: no git, no network, no GPU, no subagents, no deletes, no background waits, and every command under a timeout.
Write `artifacts/x5b/FINAL_REPORT.md` early.

## Done

Your final message contains, verbatim:
- X5-G1 to X5-G3 with receipts;
- the axiom printout;
- the hashes before and after;
- the files written.
