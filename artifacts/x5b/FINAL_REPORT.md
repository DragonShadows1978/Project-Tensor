# TX3-X5B final report: ACCEPT (X5-G1, X5-G2, X5-G3 green on all ten theorems, on the amended Spec.lean; none reported under D4)

Order: `orders/TX3_X5B_AMENDED_STATEMENTS.md`, an amendment to `orders/TX3_X5_APA_LEAN.md`, which stays binding.
Branch `x5-apa-lean`, 3fe7dcb0. Seat: Opus 5.5, a Claude Agent-tool seat, CPU only. Ledger: `artifacts/x5b/LEDGER.md`.

**C1 is done.**
- `splitk_contains` proves the amended statement: the containment conjunct, plus the output conjunct, which is now
  the sink-0 instance ∧ the sink-free instance.
- `sp_mask_superset_nv` proves the antecedent witness ∧ the strictness witness.
- No other theorem's proof changed. The build did not force any such change. `proofs_diff.txt` shows the
  eight others are byte-identical.

## Gates

Receipt classes: the build log is a suite run of the real `lake build`; the hashes and the grep are mechanical
checks.

- **X5-G1: PASS.**
  - Command: `cd lean/apa_certs && export PATH=$HOME/.elan/bin:$PATH && lake build ApaCerts.Check`.
  - Result: exit 0, `Build completed successfully (8926 jobs).`, wall 0:16.37, peak RSS 6.88 GB.
  - Built: `ApaCerts.Proofs` (7.5 s) and `ApaCerts.Check` (5.0 s). `ApaCerts.Spec` (amended, `36aeba97…`) had been
    built alone just before, in 5.9 s (`build_spec_only.txt`), because its olean predated the amendment
    (`nobuild_spec_before.txt`).
  - Log scan: sorry 0, warning 0, error 0.
  - Receipts: `artifacts/x5b/build_check.log` (copy `build_check_log.txt`, since `*.log` is git-ignored) and
    `build_log_scan.txt`.
- **X5-G2: PASS.**
  - All 10 printed axiom sets are exactly `[propext, Classical.choice, Quot.sound]`.
  - 0 axiom names fall outside the allowed set.
  - Receipt: `build_log_scan.txt`.
- **X5-G3: PASS.**
  - The five lead-file hashes are identical before and after, with the NEW Spec hash (`hashes_before.txt`,
    `hashes_after.txt`, which ends `IDENTICAL`).
  - `Proofs.lean` has 0 hits for each D1 token (sorry, admit, axiom, opaque, implemented_by, extern, unsafe,
    native_decide, set_option), both as whole words and as case-insensitive substrings. Receipt:
    `forbidden_token_grep.txt`.
- **D3 rail:** `lake build --no-build Mathlib` reported `All targets up-to-date (8923 jobs).` both before and after.
  Mathlib was never rebuilt.

## Axiom printout (copied from `build_check.log`)

```
info: ApaCerts/Check.lean:26:0: 'ApaCerts.sp_skip_final' depends on axioms: [propext, Classical.choice, Quot.sound]
info: ApaCerts/Check.lean:27:0: 'ApaCerts.sp_skip_final_nv' depends on axioms: [propext, Classical.choice, Quot.sound]
info: ApaCerts/Check.lean:28:0: 'ApaCerts.sp_mask_superset' depends on axioms: [propext, Classical.choice, Quot.sound]
info: ApaCerts/Check.lean:29:0: 'ApaCerts.sp_mask_superset_nv' depends on axioms: [propext, Classical.choice, Quot.sound]
info: ApaCerts/Check.lean:30:0: 'ApaCerts.splitk_contains' depends on axioms: [propext, Classical.choice, Quot.sound]
info: ApaCerts/Check.lean:31:0: 'ApaCerts.splitk_contains_nv' depends on axioms: [propext, Classical.choice, Quot.sound]
info: ApaCerts/Check.lean:32:0: 'ApaCerts.z_not_single_pass' depends on axioms: [propext, Classical.choice, Quot.sound]
info: ApaCerts/Check.lean:33:0: 'ApaCerts.z_not_single_pass_nv' depends on axioms: [propext, Classical.choice, Quot.sound]
info: ApaCerts/Check.lean:34:0: 'ApaCerts.sp2_certificate' depends on axioms: [propext, Classical.choice, Quot.sound]
info: ApaCerts/Check.lean:35:0: 'ApaCerts.sp2_certificate_nv' depends on axioms: [propext, Classical.choice, Quot.sound]
```

## Hashes (sha256), before and after: unchanged

Before (`hashes_before.txt`, 2026-10-28T09:55:17-04:00) and after (`hashes_after.txt`, 2026-10-28T09:59:21-04:00):

```
36aeba97e236f8cc575cfe9a9f3f8b32f31c4b93b828746e3f8eb87eed3f8fe5  ApaCerts/Spec.lean
2de6d3b0a503d7b06090a376d67d8e7ebb27f581ebe78e26e61000d01d78abed  ApaCerts/Check.lean
0a803dc189c5706ef6a36bec368b8646d80cc7f3ab042bdc8b305c023f1ea998  lakefile.lean
d5edba4e4b8faad9c1baeadb265716d20d03be4d1a2647dc5e35b0c0325bea7b  lean-toolchain
d6149cfcbe3285f26e854ef10ec14b9c2e3525d84a0ddb33f43807326f115bc1  lake-manifest.json
```

Proofs.lean: before `a899ae23299b5b37d506ded73e4d6a70d306da1b1fa48ce5819dd7d7f757050d` (X5's final file had
c29a3627; `artifacts/x5/lead_rebuild_after_comment_edit.txt` records a lead rebuild after a comment edit, which I take
as the source of a899ae23; not verified line by line, since X5's c29a3627 copy is not on disk), after
`6ee2f428deb59c1ac7dcbaeff2974fd0b439df2cb6e123fa41d72d212ec5a3c7` (368 lines).

## What the amended statements now check

**splitk_contains, output conjunct.** The fixed instance is bulk [2,0], exact [2,10], V [0,1], δ = 0.

| Variant | Global SP output | Split-K (singletons) output | Difference | Bounds the proof uses |
|---|---|---|---|---|
| Sink logit 0 (the receipt) | 1/(e²+2) ≈ 0.1065 | e¹⁰/(e²+e¹⁰+1) ≈ 0.99962 | ≈ 0.8931 | split > 23/25, global < 1/9, so the difference > 182/225 > 4/5 |
| No sink | 1/(e²+1) ≈ 0.1192 | e¹⁰/(e²+e¹⁰) ≈ 0.99966 | ≈ 0.8805 | split > 49/50, global < 1/8 |

**sp_mask_superset_nv.**
- Antecedent witness: bulk [0,1], δ = 0, key 1 is final-refined (row max 1 ≤ 1). This is the instance REV1 F2
  suggested.
- Strictness witness, unchanged: same row, key 0 is SP-refined but not final-refined.

## Scope

**Covers:** the ten statements exactly as written in the amended Spec.lean, over ℝ, checked by the Lean 4.34.1
kernel against Mathlib d13f23b.

**Does not cover** (as X5): floating point, any kernel's correctness, GPU, X3a and X4.

## Prior art

- **`rowOutputWithSink_two` and the sink-0 conjunct.**
  - Source: standard softmax normalization with one sink logit in the denominator, as in SP1.1's reference
    `dense_scores(..., sinks=...)` (primer §4.2 [R26]). The online-softmax merge adds the sink once ("Online
    normalizer calculation for softmax", arXiv 1805.02867, 2018; FlashAttention, arXiv 2205.14135, 2022).
  - Unverified; lead to check. Search terms: "online normalizer calculation for softmax", "attention sink softmax
    denominator".
  - Taken: the instance and the denominator semantics, which are the lead's statement.
  - Mine: only the Lean expansion and the rational bounds 23/25 and 1/9.
- **Exp bounds:** Mathlib `Real.exp_one_gt_d9` and `Real.add_one_le_exp`, reused from X5. Standard; no prior art
  claimed.
- **`sp_mask_superset_nv` antecedent witness:** the instance REV1 F2 suggested. Context: SP1 mask containment (primer
  §4.1 [R24]) and the BLASST running-max criterion (arXiv 2512.12087). The proof uses Mathlib `Finset.sup'_le`. No
  prior art known to me beyond that.
- **The other eight theorems** are unchanged, and their prior art is as in `artifacts/x5/FINAL_REPORT.md`.

The same annotations sit at the code sites in Proofs.lean: the file header, `rowOutputWithSink_two`,
`sp_mask_superset_nv` and the §3 comment.

## Files written

- `lean/apa_certs/ApaCerts/Proofs.lean` (edited).
- `lean/apa_certs/.lake/build/`, Lake output: ApaCerts.Spec, ApaCerts.Proofs and ApaCerts.Check rebuilt.
- In `artifacts/x5b/`:
  - `FINAL_REPORT.md`, `LEDGER.md`;
  - `hashes_before.txt`, `hashes_after.txt`;
  - `build_check.log`, `build_check_log.txt`, `build_log_scan.txt`;
  - `forbidden_token_grep.txt`, `theorem_list.txt`;
  - `proofs_sha256.txt`, `proofs_diff.txt`;
  - `build_spec_only.txt`;
  - `nobuild_mathlib_before.txt`, `nobuild_mathlib_after.txt`, `nobuild_spec_before.txt`;
  - `iter1_lake_env_lean.txt`, `iter2_lake_env_lean.txt`.
- Scratch, left in place (`/tmp` is exempt from the no-delete rule):
  - `/tmp/x5b_Proofs_before.lean` (the pre-edit copy);
  - `/tmp/x5b_scratch/`: iter1.log, iter2.log, build_spec.txt, nobuild_*_before.txt, proofs_diff.txt.

## Disclosures

- **Builds beyond the gate:**
  - `lake build ApaCerts.Spec`, needed because the Spec olean was stale;
  - `lake env lean ApaCerts/Proofs.lean` ×2;
  - `lake build --no-build` ×3 (Mathlib before and after, and Spec before), which only check and never build.

  All of them wrote only to `lean/apa_certs/.lake/build`.
- **Personal names (L21).** The comments I added cite the softmax prior art by title and arXiv id, not by author.
  One author citation is still in Proofs.lean at line 16: the BLASST authors, written by the X5 seat and already in
  the committed file. I left it because changing it was outside C1. The lead may want to screen it before a public
  push.
- **Report files.** A session-level note says not to write report .md files. The order requires `FINAL_REPORT.md`,
  and X5 requires a ledger, so I followed the order.
- **Not done:** no git, no network, no GPU, no subagents, no deletes, no background waits, and no edits outside the
  writable paths. Spec.lean, Check.lean, lakefile, toolchain and manifest were not edited.
