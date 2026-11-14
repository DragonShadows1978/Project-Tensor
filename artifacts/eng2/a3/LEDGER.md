# ENG2-A3 ledger (seat: Claude Opus 5.5, Agent tool; order orders/ENG2_A3_ENG1_ROWS_AND_CONSUMERS.md, branch eng2 at c3452516)

Times EST, 2026-11-14 (file mtimes). No git writes; read-only `git grep`/`git ls-files`/`git show`/`git rev-parse` only
on the three consumer repositories (R2), as the order permits; no git at all on Project-Tensor. No GPU: CUDA hidden for
every CPU run; the device-mode dry run stops at the governor (FORGE_GPU_DRY_RUN=1, exit 78). Scratch: /tmp/eng2_a3_*
only. Nothing deleted. At most 2 CPU-busy processes (in practice 1), nice -n 10, one BLAS thread per test process; the
largest process (consumer census) peaked at 55 MB RSS, so no scope was needed.

## Setup
- 15:17 read the A3 order (sha256 f89fb040d06d009aafac755de5e96e6f1cab9bb8090b97b6c3dad88182170710), HOUSE_RULES.md,
  AGENTS.md, the parent order, A1 and A2 orders, the A2 ledger and its scripts, tool/stage.py and evidence.py.
- 15:21 L01 (`preflight_L01.txt`): Python 3.12.3, NumPy 2.4.2, pytest 9.0.2, git 2.43.0, forge-gpu-run --help rc 0;
  /mnt/ForgeRealm 97,652,899,840 B free; 24 GB available. No-op collection on the candidate arm (A1/A2's method):
  253/394 selected, rc 0.
- 15:22 before-map `worktree_before_a3.json` (15,559 files; artifacts/eng2/snapshot.py, unchanged) and pre-edit copies
  in `before/`. Seal before A3: a9175ff93acf6f8c71a76aadecef916c92077f1ac489993eaec5c6e281bdf28f (A2's; G5 ran under it).
- 15:22 pre-edit ID collection (`collect_eng1_ids.py`, the sealed generator's COLLECT text, prepend mode as the regnorm
  suite collects test_eng1.py): 156 IDs on both arms, equal; exactly 10 contain float64 == L38_PINS expected_changes.

## Registration (written 15:23:24, before the edit at 15:23:33 and before any gate ran; immutable from here)
- RG1 (L38) test_eng1.py IDs 156 -> 146 on both arms; the removed set == the 10 L38 pin IDs exactly; no ID added.
- RG2 censuses re-collected by tool/stage.py step_census (unchanged): ids_regnorm and ids_regnorm_candidate 578 -> 568,
  losing exactly the 10 IDs; ids_regdtype / _candidate (1779; test_eng1.py not in that suite), ids_eng2 / _candidate
  (394) and ids_audit (253) byte-identical to before/.
- RG3 mutation coverage: no registered killer pattern of ENG1's 16 or ENG2's 15 mutants matches any of the 10 removed IDs
  (ENG1 matcher artifacts/eng1/tool/evidence.matches; ENG2 matcher artifacts/eng2/tool/evidence.matches); every mutant
  keeps >= 1 registered killer pattern matching a post-A3 collected ID; in its G5 receipt (child_result.json and
  junit.xml, sha256 confirmed against the lead's manifest) >= 1 such remaining killer ID failed. Any mutant losing its
  only killer = RED (stop and report).
- RG4 CPU suite (A-G2's method, cpu_runs.py, `-k cpu`): test_eng2.py candidate 253/253 pass; baseline fails exactly A2's
  120 (a2/baseline_cpu.json failing set: the 96 registered F1 IDs + the 24 non-string sentinel IDs). test_eng1.py
  (47 CPU IDs before, 47 after: no float64 ID is a CPU ID) passes 47/47 on both arms.
- RG5 staging: tool/baseline_expected_failures.json (166), gpu_lines.json (21 lines), STAGED_GPU_COMMANDS.sh,
  regression_files.json, source_reading_tests.json byte-identical to before/; the 17 arms differ from before only in
  tests/test_eng1.py; a new seal is written (it changes: test_eng1.py, L38_PINS.json and the census files it pins).
- RG6 MF1 / MF2 on CPU through evaluate(): K1 (12/12, 48/48 killers, traced through hunks [63,63] / [75,75]), K2, K3,
  S1, S2 as A2.
- RG7 dry run: 21/21 labels ok (4 roots governor 78; 17 dependents refused at the dependency check; S1 78 x21), start
  records bound to the new seal. Audit route: staged rc 0 with 253 CPU IDs; score_gpu.py rc 1 with the designed audit
  error; 11/11 audit steps ok.
- RG8 scope: changed only test_eng1.py (the three lists and one comment line), L38_PINS.json (expected_changes only),
  tool census/collect/seal/staging files, arms' test_eng1.py; added only a3/** and the fresh dry/audit roots.

## R1 (diff: `A3_SOURCE.diff`)
- 15:23 test_eng1.py: `E2_DTYPES` (:180) loses "float64"; `E2_ENGINE_DTYPE` (:186) loses its "float64" mapping, which
  also removes the old trailing comment ("# F2: the engine has no float64; tc.tensor builds float32"); line 187 is the
  one new comment ("# ENG2-A3: the engine has no float64, so no float64 rows here; test_eng2.py tests the refusal of a
  float64 request."); `test_e2_pow_grad_bits`' parametrize (:266) becomes ["float32"]. Line count unchanged (894), so
  no other line moved. Nothing else changed (scope receipt: differing lines exactly [180, 186, 187, 266]).
- 15:23 after-edit collection: 146 IDs on both arms (`ids_test_eng1_after_*.txt`); `ids_test_eng1_diff.txt` = exactly
  the 10 pinned IDs removed, none added.
- 15:24 L38_PINS.json: expected_changes = {}; one "a3" record added (retired IDs, why, receipts); every other key equal.
- 15:26 mutation coverage (`mutation_coverage.py` -> `MUTATION_COVERAGE.json`, log `mutation_coverage.log`): 31/31
  mutants keep their registered killers; no pattern matches a removed ID (matcher controls: a constructed float64
  pattern matches 5 / 1 / 10 removed IDs, so "none" is not vacuous); every mutant's admitted G5 attempt has >= 1
  remaining killer that failed; child_result.json, junit.xml, parent_result.json sha256 equal to the manifest for all
  31; JUnit and summary outcomes agree ID by ID; the G5 scorer's actual killers are all among the remaining failed
  killers. Manifests: ENG1 a722e5b240971ba24587eb5970151e999140b0ec3b6936dd67edf70240fbb9c9 (1022 files; the eng1 run
  log's post_run_manifest line quotes the same hash), ENG2 e957c001948c783833018776174fe6232f372d75598e1fa0ace8a4485253d72a
  (1855 files; eng2_g5.log quotes the same hash). ENG1's run root read from the eng1 worktree (D2).

## R3 runs (in order)
- 15:27 `restage.py arms` (copy of a2/restage.py; diff `restage.diff`: A3 order in the seal list, a3/baseline_cpu.json,
  test_eng1.py checked as pre-A3 bytes == ENG2's before-map and differing only on lines 180/186/187/266): 17 arms
  rewritten, 11 files each, modules unchanged (baseline 477e81cb..., others f72af46b...). Log `restage_arms.log`.
- 15:27-15:29 `cpu_runs.py` (copy of a2/cpu_runs.py; diff `cpu_runs.diff`: ENG2_A3_FILE selects test_eng1.py):
  test_eng2.py baseline 120 failed / 253, candidate 0 / 253, MF1 12 (22 tests traced through :63), MF2 48 (148 traced
  through :75); test_eng1.py baseline 0 / 47, candidate 0 / 47. Log `cpu_runs.log`.
- 15:29-15:31 `restage.py census lines` (stage.step_census unchanged + A2's lines): eng2 394 (both arms), audit 253,
  regnorm 568 (both), regdtype 1779 (both); 21 lines, 166 registered baseline failures of 394.
- 15:31 `restage.py script seal`: script byte-identical; seal `tool/staging_seal.json`
  b424bdcfa0d01acb84cfe71eec21dc4597a40c2ebd77d99c9143aee4f17df826 (was a9175ff9...). Seal diff vs before: arms (each
  arm's tests/test_eng1.py only), registration_files (test_eng1.py, L38_PINS.json, a3/baseline_cpu.json replaces
  a2/baseline_cpu.json, the A3 order added), tool_files (ids_regnorm.txt, ids_regnorm_candidate.txt).
- 15:32 dry run `g4_dry.py dry_4` (copy, scratch paths only; diff `g4_dry.diff`) -> `G4_DRY_dry_4.json`, log
  `g4_dry_4.log`: 21/21 ok (4 roots: seal+arms verified, governor 78 not admitted, no child; 17 dependents refused at
  the dependency check; S1 78 x21).
- 15:32 audit route: `env -u PYTHONOPTIMIZE RUNS=.../tool/gpu_runs/audit_4 TMPDIR=/tmp/eng2_a3_audit timeout 590s
  bash artifacts/eng2/tool/STAGED_GPU_COMMANDS.sh --audit` -> staged rc 0 (`audit.*`); `score_gpu.py` -> rc 1,
  "audit run cannot establish GPU acceptance" (designed; `audit_score.*`). `audit_check.py` (diff `audit_check.diff`)
  -> `A3G3_AUDIT_RECEIPT.json`: 11/11 ok. (score_gpu.py ran without -B, as staged: tool/__pycache__ refreshed; exempt.)
- 15:33 `mutant_eval_cpu.py` (diff `mutant_eval_cpu.diff`; root audit_4/cpu_mutants) -> `MUTANT_EVAL_CPU.json`: MF1
  K1/K3 (12/12 killers via evaluate(), traced), MF2 K1/K3 (48/48), candidate K2, S1/S2 refused "no registered killer
  ID failed". all_ok True.
- 15:34 `a3g3_cpu_check.py` -> `A3_CPU_AND_CENSUS.json`: C1-C7 all ok (RG1, RG2, RG4, RG5 as registered).
- ~16:00 final seal re-verification (evidence.verify_seal + verify_arm for 17 arms, read-only): ok.

## R2 consumer census (`consumer_census.py` -> `CONSUMER_CENSUS.json`; tables `census_report.py`; control
`consumer_census_selftest.py` -> `CONSUMER_CENSUS_SELFTEST.json`; reading notes `unresolved_reading.py` ->
`UNRESOLVED_CHANGED_READING.json`)
- 15:38 first run: 507 files, 0 newly refused. 15:43 code-in-string parsing added (0 in-string sites).
- 15:45 self-test caught a classification slip (dtype=float read as traced-non-string; D3); fixed, both re-run.
- 15:46 a read of GraftRepository's main ref showed `tc.zeros(*shape, dtype=dt)` in core/deepseek_v2_lite_tc.py, which
  the census had missed: the file takes `tc`/`F` from core.mistral7b_tc (cross-file). Added a cross-file alias fixpoint
  over every tracked .py file, the `M.tc.<entry>` attribute rule, and a self-test case (D4).
- 15:54 that run flagged deepseek :115 as newly refused (`dt = dtype or BlockTC.COMPUTE_DTYPE` read as possibly None).
  Reading showed a false positive of the BoolOp rule (`a or b` never yields a falsy a); refined, self-test case added,
  re-run (D5).
- 15:59 final run: 512 files, 1693 sites, 0 parse failures; newly refused 0; unresolved on changed entries 48 (33
  `_causal_mask(..., <t>.dtype)`, 13 `BlockTC.COMPUTE_DTYPE`, 1 Voice t1_empty, 1 trinity receipt), every one read and
  annotated (reasoning) as a supported string; unresolved on unchanged entries 288 (Tensor.astype; ENG2 cannot refuse).
  Project-Voice HEAD moved during the round (72b90984 -> 56174afe -> 17846484, another session committing); the final
  census is at 17846484. GraftRepository's checkout is on lc1-wip (03f6e45c), not main (abde4102); the listing on main
  has the same 147 files, and the main-ref grep shows the same non-literal tc dtype sites.

## Gates
- A3-G1 PASS: R1 done (`A3_SOURCE.diff`, `ids_test_eng1_diff.txt`); `MUTATION_COVERAGE.json` 31/31 with a remaining
  failed killer; L38_PINS expected_changes {}; censuses re-collected (`A3_CPU_AND_CENSUS.json` C5-C6).
- A3-G2 (census, not pass/fail): `CONSUMER_CENSUS.json`, `UNRESOLVED_CHANGED_READING.json`.
- A3-G3 PASS: CPU suite (`A3_CPU_AND_CENSUS.json` C1-C4), MF1/MF2 via evaluate() (`MUTANT_EVAL_CPU.json`), seal
  b424bdcf..., dry run (`G4_DRY_dry_4.json`), audit (`A3G3_AUDIT_RECEIPT.json`).
- Scope PASS (`A3_SCOPE.json`, `worktree_after_a3.json`).
- Repairs (L56b): none on any gate.

## Deviations
- D1 (order fault): the order names `gpu_runs/dry_3` and `gpu_runs/audit_3` as new roots; both already hold A2's
  receipts (11:51-11:53). Writing there would edit receipts in place (forbidden), and g4_dry.py / mutant_eval_cpu.py
  require fresh roots. Used `gpu_runs/dry_4` and `gpu_runs/audit_4` (the RUNS guard requires roots under tool/).
- D2: ENG1's G5 run root is read from the eng1 worktree
  (/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_eng1/artifacts/eng1/tool/gpu_runs/g5_1114):
  this worktree holds only its git-tracked subset (child_result.json, forge.log, stdout.log are git-ignored here).
  The manifest copies in both worktrees are byte-identical (a722e5b2...).
- D3 (L56a): consumer-census tool slip (dtype=float classified traced-non-string, not non-string-literal); fixed and
  re-run. No effect on newly-refused (both classes refuse).
- D4: the census goes beyond the order's import listing: a "cross-file" group (13 GraftRepository files that take a
  tensor_cuda alias from another repo module; 9 listed files get such aliases too) and a "supplementary" group (12
  files that call `tc.<entry>(` with no alias bound). The cross-file rule added 101 changed-entry sites (GraftRepository
  changed-entry sites 77 -> 178, mostly tc.tensor / tc.zeros in core/*_tc.py) that the import listing alone missed.
- D5: BoolOp refinement after a false positive (deepseek :115); the site is listed as unresolved with its reading note.
- D6: R3 says test_eng1.py and test_eng2.py on CPU "as A-G2 ran them"; A-G2 ran test_eng2.py only. test_eng1.py's
  `-k cpu` subset (47 IDs) ran through the same child (cpu_runs.py, ENG2_A3_FILE=test_eng1.py).
- D7: L38_PINS.json gains an "a3" record next to the emptied expected_changes (the evidence tool reads only
  expected_changes); every other key is unchanged.

## Residuals (not fixed; outside this order)
- test_eng1.py :268 docstring still reads "float32 (and the float64 request, which the engine builds as float32)";
  stale after R1, but the order allows only the three lists and one comment.
- L38_PINS.json `sites_that_change` / `dtype_string_sites` keep the pre-A3 trace (noted in its "a3" record).
- Census limits: flow- and scope-insensitive alias binding; in-file tracing only (cross-file values are reading notes);
  star imports from repo modules not followed; untracked files and WorkTree/ not seen; non-.py files not parsed (a
  git grep of them for `tc.<factory>(... dtype` found only a patch receipt, docs and logs). tc_signal and other code
  inside Project-Tensor itself is not a consumer here.
- *.json receipts are git-ignored in this repo (lead: `git status --ignored` before add).
- Scratch kept (no deletes): /tmp/eng2_a3_{scratch,stage,cpu,dry,dry_s1,audit,mut,selftest_repo}.

## Prior art
- Mutation coverage per mutant (killing-test sets): DeMillo, Lipton and Sayward (1978); per-mutant killing tests as in
  PIT (unverified -- lead to check "PIT killingTests"). Taken: a mutant is killed by a failing registered test. Ours:
  the removed-ID intersection and the manifest-bound receipt check.
- Consumer census: static call-site discovery over Python ASTs, as in linters and codemod tools (Bowler, LibCST,
  pyupgrade; unverified -- lead to check); in-file constant tracing in the spirit of reaching definitions / constant
  propagation (Kildall 1973; Aho, Sethi and Ullman), deliberately crude. Taken: AST call matching and def-use tracing.
  Ours: the entry table, the argument classes, the cross-file alias fixpoint, the `a or b` falsy rule and the ENG2
  newly-refused rule.
- Manifest diffing for scope (BSD mtree(8), unverified) and SHA-256 (NIST FIPS 180-4) for the seal and receipts.
- Re-staging, dry run and audit reuse ENG2's sealed tool and A2's scripts (this repo, 2026); no new algorithm there.
