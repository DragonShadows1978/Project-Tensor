# TX3-X5 final report: ACCEPT (X5-G1, X5-G2, X5-G3 green on all ten theorems; none reported under D4)

Order: `orders/TX3_X5_APA_LEAN.md` (branch `x5-apa-lean`, b6c9b9c9). Seat: Opus 5.5 (a Claude Agent-tool seat), CPU
only. Ledger: `artifacts/x5/LEDGER.md`. P16 (all ten proved): HIT.

## Gates (receipts: build log class = suite run of the real `lake build`; hashes and grep = mechanical check)

- **X5-G1: PASS.**
  - Command: `cd lean/apa_certs && export PATH=$HOME/.elan/bin:$PATH && lake build ApaCerts.Check`.
  - Result: exit 0, `Build completed successfully (8926 jobs).`, wall 0:16.58, peak RSS 6.92 GB.
  - Only `ApaCerts.Proofs` (7.5 s) and `ApaCerts.Check` (4.2 s) were built. `ApaCerts.Spec` was built earlier in this
    seat (5.3 s) from the unchanged, hash-pinned Spec.lean.
  - Log scan: sorry 0, warning 0, error 0.
  - Receipts: `artifacts/x5/build_check.log` (copy `build_check_log.txt`) and `build_log_scan.txt`.
- **X5-G2: PASS.** All 10 printed axiom sets are exactly `[propext, Classical.choice, Quot.sound]`; 0 lines deviate.
  Receipt: `build_log_scan.txt`.
- **X5-G3: PASS.**
  - The five lead-file hashes are identical before and after: `hashes_before.txt` = `hashes_after.txt`.
  - `Proofs.lean` has 0 hits for each D1 token (sorry, admit, axiom, opaque, implemented_by, extern, unsafe,
    native_decide, set_option), and 0 case-insensitive hits. Receipt: `forbidden_token_grep.txt`.

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

## Theorems proved (10 of 10); reported under D4: none

`sp_skip_final`, `sp_skip_final_nv`, `sp_mask_superset`, `sp_mask_superset_nv`, `splitk_contains`,
`splitk_contains_nv`, `z_not_single_pass`, `z_not_single_pass_nv`, `sp2_certificate`, `sp2_certificate_nv`
(declaration lines in `theorem_list.txt`).

**Witnesses used:**

| Theorem | Witness |
|---|---|
| `sp_skip_final_nv` | bulk [1,0], δ = 0, key 1 |
| `sp_mask_superset_nv` | bulk [0,1], δ = 0, key 0 (shows strictness) |
| `splitk_contains_nv` | bulk [2,0], δ = 0, P = {0}, key 0 |
| `z_not_single_pass_nv` | z = 0 |
| `sp2_certificate_nv` | bulk [10,0], exact [9,1], e_q = 1, ε = 1/2, key 1 (non-trivial: e_q > 0, ε < 1) |

**Boundary instantiations** were added as `example`s in Proofs.lean, checked in the same build:
- z = √3, where key 0 of [1,0,0,0] sits exactly at its threshold 1;
- z = 0;
- δ = 0 on a single-key row (n = 0);
- errq = 0.

## Hashes (sha256), before and after: unchanged

```
1bf0a7d621912ca6e6a537e01041ea6de18e1304c00b5f89e78917ed97d8a01d  ApaCerts/Spec.lean
2de6d3b0a503d7b06090a376d67d8e7ebb27f581ebe78e26e61000d01d78abed  ApaCerts/Check.lean
0a803dc189c5706ef6a36bec368b8646d80cc7f3ab042bdc8b305c023f1ea998  lakefile.lean
d5edba4e4b8faad9c1baeadb265716d20d03be4d1a2647dc5e35b0c0325bea7b  lean-toolchain
d6149cfcbe3285f26e854ef10ec14b9c2e3525d84a0ddb33f43807326f115bc1  lake-manifest.json
```

Final `Proofs.lean`: `c29a3627aa0690d4c301d18e66ee570692eea538a56862d123d3b1f604dd4856` (329 lines).

## Scope (what this does and does not cover)

**Covers:** the five claims, exactly as the lead stated them over ℝ, plus their witnesses. The Lean 4.34.1 kernel
checked them against Mathlib d13f23b.

**Does not cover:**
- floating point;
- any kernel's correctness, or whether `zRefined` faithfully transcribes `kernels.cu` (that transcription is the
  lead's statement);
- GPU;
- X3a and X4.

**Statement-level notes.** These are not D4 findings; every statement is true as written.
- `z_not_single_pass`: the impossibility covers decisions that are functions of key 0's bulk value, as Spec's comment
  scopes it. Both rows have the same length, so knowing N does not help either.
- `sp2_certificate` is the per-key certificate. It says nothing about the aggregate bound, which primer §4.3 calls
  useless at N·ε ≥ 1.
- `sp_skip_final` is the monotonicity form ("a skip stays a skip"). Online-versus-offline mask equality follows from
  the definitions, because the rule is defined on prefix maxima.

## Prior art

- **Prefix/row-maximum lemmas:** Mathlib `Finset.le_sup'`, `Finset.sup'_le` and `Finset.exists_mem_eq_sup'`
  (`Mathlib/Data/Finset/Lattice/Fold.lean`). This is standard order theory; I added only the instantiation.
- **SP rule and its monotonicity:** BLASST running-maximum criterion (Yuan, Shinn et al., arXiv 2512.12087, MLSys 2026;
  primer §7 [R32], lead-verified), applied to precision. I took the criterion. Mine is only the Lean proof.
- **SP2 certificate:**
  - The δ derivation is SP2's (primer §4.3 [R06]).
  - BLASST's ln λ is the same margin without 2·e_q.
  - The proof uses standard exp/log monotonicity: Mathlib `Real.exp_lt_exp`, `Real.exp_add`, `Real.exp_log` and
    `Real.log_inv`.
- **Numeric bounds:** Mathlib `Real.exp_one_gt_d9`, `Real.add_one_le_exp` and `Real.log_le_sub_one_of_pos`.
- **Counterexamples:** SP1's z rows [R24] and SP1.1's split-K example [R26], evaluated directly.
- **The z-score impossibility argument:** no prior art known to me (primer §7 lists it as APA's own).

## Disclosures

- **Hook blocks:** the PreToolUse hook "AFTERIMAGE [POSTGRESQL]: You've written similar code before! … Retry your write
  now" blocked 4 Edit calls. Each retry applied. Two `lake env lean` runs (iter2, iter5) therefore re-checked an
  unchanged file. No effect on results.
- **Two gate runs.** Run 1 was on Proofs.lean `355fcfbf…`: all green. I then added the four boundary `example`s and
  ran run 2 on the final `c29a3627…`: all green. The run-1 receipts were renamed with `mv` to `*_run1*`; nothing was
  deleted.
- **Lake commands beyond the gate:**
  - `lake env lean` (Spec, and Proofs ×6);
  - `lake build ApaCerts.Spec`;
  - `lake build --no-build Mathlib` ×2, which only checks and never builds.

  All of them wrote only to `lean/apa_certs/.lake/build`. Mathlib was never rebuilt: 8923 jobs were up to date both
  before and after.
- **Ignored receipts:** `*.log` is git-ignored (repo `.gitignore` line 8). So `build_check.log` and
  `build_check_run1.log` also have `.txt` copies (`build_check_log.txt`, `build_check_run1_log.txt`).
- **Scratch:** scratch is in `/tmp/x5_scratch/` (iter1-6 logs and `time` outputs). It is left in place, since /tmp is
  exempt from the no-delete rule.
- **No-report-file note:** a session-level note says "do not write report .md files". The order requires
  `FINAL_REPORT.md` and `LEDGER.md`, so I followed the order.
- **Not done:** no git, no network, no GPU, no subagents, no deletes, and no edits outside the writable paths.
