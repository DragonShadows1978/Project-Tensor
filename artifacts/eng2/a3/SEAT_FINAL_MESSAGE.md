# ENG2-A3 seat final message (copied verbatim by the lead)

**Verdict: ACCEPT.** A3-G1 PASS, A3-G3 PASS, scope PASS. A3-G2 (census, not pass/fail): 0 consumer call sites would be newly refused by ENG2. 48 sites on the entries ENG2 changes are unresolved, and my reading of the source says each passes a supported string. Another 288 unresolved sites are on entries ENG2 does not change, so it cannot refuse them. New seal sha256: **b424bdcfa0d01acb84cfe71eec21dc4597a40c2ebd77d99c9143aee4f17df826** (was a9175ff9…).

## The diff
`artifacts/eng2/a3/A3_SOURCE.diff` (sha256 75c7a81c…9679):
```
--- a/tensor_cuda/tests/test_eng1.py
+++ tensor_cuda/tests/test_eng1.py
@@ -177,14 +177,14 @@
 E2_P = [-1.5, -0.5, 0.5, 2.0, 3.0]
-E2_DTYPES = ["float16", "bfloat16", "float32", "float64"]
+E2_DTYPES = ["float16", "bfloat16", "float32"]
@@
-E2_ENGINE_DTYPE = {"float16": "float16", "bfloat16": "bfloat16", "float32": "float32",
-                   "float64": "float32"}  # F2: the engine has no float64; tc.tensor builds float32
+E2_ENGINE_DTYPE = {"float16": "float16", "bfloat16": "bfloat16", "float32": "float32"}
+# ENG2-A3: the engine has no float64, so no float64 rows here; test_eng2.py tests the refusal of a float64 request.
@@ -263,7 +263,7 @@
-@pytest.mark.parametrize("dtype", ["float32", "float64"])
+@pytest.mark.parametrize("dtype", ["float32"])
 def test_e2_pow_grad_bits(dtype, p):
```
- **Line positions:** the file is still 894 lines. Only lines 180, 186, 187 and 266 differ.
- **`L38_PINS.json`:** `expected_changes` is now `{}`. One `"a3"` record was added (the retired IDs, why, receipts). Every other key is unchanged.
- **ID diff (L38):** 156 → 146 IDs on both arms. The removed set is exactly the 10 pins: `test_e2_pow_grad[float64-p-1.5|p-0.5|p+0.5|p+2|p+3]` and `test_e2_pow_grad_bits[float64-p-1.5|p-0.5|p+0.5|p+2|p+3]`. No ID was added. Receipt: `ids_test_eng1_diff.txt`.

## Mutation-coverage table (A3-G1)
Receipt: `a3/MUTATION_COVERAGE.json`.
- **Manifests:** ENG1 is a722e5b240971ba24587eb5970151e999140b0ec3b6936dd67edf70240fbb9c9 (1022 files). ENG2 is e957c001948c783833018776174fe6232f372d75598e1fa0ace8a4485253d72a (1855 files). Each matches its run log's `post_run_manifest` line.
- **Receipt hashes:** for every mutant, child_result.json, junit.xml and parent_result.json match the manifest. JUnit and summary outcomes agree ID by ID.
- **Matcher controls:** a made-up float64 pattern matches 5, 1 and 10 of the removed IDs, so "none" below is a real result, not an empty check.

| side | mutant | registered | match removed | remaining patterns | G5 failed remaining killers | receipts sha ok | ok |
|---|---|---|---|---|---|---|---|
| eng1 | M01 | 3 | none | 3 | 3 (e.g. test_e1_refuse[three_neg1]) | True | True |
| eng1 | M02 | 3 | none | 3 | 15 (e.g. test_e1_infer[r1_p0]) | True | True |
| eng1 | M03 | 2 | none | 2 | 2 (e.g. test_e1_refuse[zero_product]) | True | True |
| eng1 | M04 | 3 | none | 3 | 3 (e.g. test_e1_refuse[negative_with_neg1]) | True | True |
| eng1 | M05 | 2 | none | 2 | 2 (e.g. test_e2_pow_grad[float16-p-0.5]) | True | True |
| eng1 | M06 | 2 | none | 2 | 10 (e.g. test_e2_pow_grad[bfloat16-p+0.5]) | True | True |
| eng1 | M07 | 2 | none | 2 | 2 (e.g. test_e2_pow_grad[float16-p-0.5]) | True | True |
| eng1 | M08 | 4 | none | 4 | 4 (e.g. test_e4_empty[rms_norm-last0-cpu]) | True | True |
| eng1 | M09 | 2 | none | 2 | 2 (e.g. test_e4_empty[apa_blend_softmax-last0-cpu]) | True | True |
| eng1 | M10 | 3 | none | 3 | 3 (e.g. test_e4_refuse[topk-k1-last0-cpu]) | True | True |
| eng1 | M11 | 2 | none | 2 | 8 (e.g. test_e5_refuse_cpu[avg_pool2d-oh_zero]) | True | True |
| eng1 | M12 | 1 | none | 1 | 4 (e.g. test_e5_refuse_cpu[avg_pool2d-ow_only]) | True | True |
| eng1 | M13 | 1 | none | 1 | 4 (e.g. test_e5_refuse_cpu[avg_pool2d-trunc_gap]) | True | True |
| eng1 | M14 | 1 | none | 1 | 1 (e.g. test_e6_terrain_writes_nothing) | True | True |
| eng1 | M15 | 1 | none | 1 | 1 (e.g. test_e2_pow_grad_nonpow2[bfloat16]) | True | True |
| eng1 | M16 | 1 | none | 1 | 4 (e.g. test_e5_refuse_cpu[avg_pool2d-ow_zero]) | True | True |
| eng2 | MF1 | 12 | none | 12 | 12 (e.g. test_f1_refuse_cpu[tensor-double-cpu]) | True | True |
| eng2 | MF2 | 48 | none | 48 | 48 (e.g. test_f1_factory_refuses_before_allocation_cpu[ones-double]) | True | True |
| eng2 | MN1_LN | 16 | none | 16 | 16 (e.g. test_n1_half_rule_bits[LayerNorm-bfloat16-near_eps-backward]) | True | True |
| eng2 | MN1_RMS | 16 | none | 16 | 16 (e.g. test_n1_half_rule_bits[RMSNorm-bfloat16-near_eps-backward]) | True | True |
| eng2 | MN1_BN | 16 | none | 16 | 16 (e.g. test_n1_half_rule_bits[BatchNorm1D-bfloat16-near_eps-backward]) | True | True |
| eng2 | MN1_IN | 16 | none | 16 | 16 (e.g. test_n1_half_rule_bits[InstanceNorm2D-bfloat16-near_eps-backward]) | True | True |
| eng2 | MN2_LN | 5 | none | 5 | 5 (e.g. test_n1_half_rule_bits[LayerNorm-bfloat16-near_eps-backward]) | True | True |
| eng2 | MN2_RMS | 4 | none | 4 | 4 (e.g. test_n1_half_rule_bits[RMSNorm-bfloat16-near_eps-backward]) | True | True |
| eng2 | MN2_BN | 6 | none | 6 | 6 (e.g. test_n1_batchnorm_running_stats[bfloat16-near_eps]) | True | True |
| eng2 | MN2_IN | 5 | none | 5 | 5 (e.g. test_n1_half_rule_bits[InstanceNorm2D-bfloat16-near_eps-backward]) | True | True |
| eng2 | MN3_LN | 11 | none | 11 | 11 (e.g. test_n1_half_rule_bits[LayerNorm-bfloat16-near_eps-backward]) | True | True |
| eng2 | MN3_RMS | 9 | none | 9 | 9 (e.g. test_n1_half_rule_bits[RMSNorm-bfloat16-near_eps-backward]) | True | True |
| eng2 | MN3_BN | 14 | none | 14 | 14 (e.g. test_n1_batchnorm_running_stats[bfloat16-near_eps]) | True | True |
| eng2 | MN3_IN | 11 | none | 11 | 11 (e.g. test_n1_half_rule_bits[InstanceNorm2D-bfloat16-near_eps-backward]) | True | True |
| eng2 | MN4_LN | 4 | none | 4 | 4 (e.g. test_n1_float32_bits[LayerNorm-near_eps-backward]) | True | True |

`red: [] all_ok True`. No mutant lost its only killer. The G5 scorer's actual killers are all among the remaining killers that failed.

## Consumer census (A3-G2)
Receipts: `a3/CONSUMER_CENSUS.json` (sha256 d9422aa8…200f) and `UNRESOLVED_CHANGED_READING.json`.
- **Self-test:** `CONSUMER_CENSUS_SELFTEST.json` is ok. It covers 18 constructed cases, one per argument class and resolution route, including a cross-file case.
- **Checkouts censused:**
  - Project-Voice is on main at 17846484. Its HEAD moved twice during the round because another session was committing.
  - GRAPA is on main at f22ad802.
  - GraftRepository's checkout is on **lc1-wip** (03f6e45c), not main (abde4102). Main has the same 147 listed files.
- **What was skipped:** WorkTree/ (0 tracked files there) and untracked files. Parse failures: 0.

| Repository | Entries | Class | Sites | Live | Receipt |
|---|---|---|---:|---:|---:|
| GRAPA-Native-LLM | changed | supported-literal | 251 | 43 | 208 |
| GRAPA-Native-LLM | changed | unresolved | 32 | 6 | 26 |
| GRAPA-Native-LLM | unchanged | non-string-literal | 357 | 149 | 208 |
| GRAPA-Native-LLM | unchanged | traced-non-string | 23 | 23 | 0 |
| GRAPA-Native-LLM | unchanged | traced-supported | 2 | 2 | 0 |
| GRAPA-Native-LLM | unchanged | unresolved | 184 | 28 | 156 |
| GraftRepository | changed | supported-literal | 163 | 147 | 16 |
| GraftRepository | changed | unresolved | 15 | 14 | 1 |
| GraftRepository | unchanged | non-string-literal | 481 | 445 | 36 |
| GraftRepository | unchanged | supported-literal | 25 | 25 | 0 |
| GraftRepository | unchanged | traced-non-string | 3 | 2 | 1 |
| GraftRepository | unchanged | traced-supported | 5 | 5 | 0 |
| GraftRepository | unchanged | unresolved | 104 | 83 | 21 |
| Project-Voice | changed | supported-literal | 19 | 19 | 0 |
| Project-Voice | changed | traced-supported | 1 | 1 | 0 |
| Project-Voice | changed | unresolved | 1 | 1 | 0 |
| Project-Voice | unchanged | non-string-literal | 12 | 12 | 0 |
| Project-Voice | unchanged | supported-literal | 15 | 15 | 0 |

- **"Changed" entries:** tc.tensor, zeros, ones, randn, rand, nn.parameter and functional._causal_mask.
- **"Unchanged" entries:** tc.empty, Tensor.astype, the out_dtype entries, the quantization.affine helpers and RegisteredMargins.lookup. The A2 census shows identical baseline and candidate behaviour for these.

Sites per entry:
- **GRAPA:** astype 566, _causal_mask 32, tc.tensor 251.
- **GraftRepository:** astype 613, _causal_mask 2, int4_dequant 2, kv_int4_unpack 3, tc.tensor 166, tc.zeros 10.
- **Voice:** astype 15, tc.empty 12, tc.tensor 20, tc.zeros 1.

Files and sites by group and role:

| Repository | Group | Role | Files | Sites |
|---|---|---|---:|---:|
| GRAPA | import | product / receipt / script / test | 14 / 289 / 25 / 8 | 38 / 598 / 95 / 62 |
| GRAPA | supplementary | script / test | 6 / 3 | 37 / 19 |
| GraftRepository | import | product / receipt / script / test | 19 / 2 / 27 / 99 | 293 / 39 / 57 / 241 |
| GraftRepository | cross-file | product / receipt / script | 4 / 2 / 7 | 31 / 36 / 18 |
| GraftRepository | supplementary | script | 2 | 81 |
| Voice | import | product / script | 3 / 1 | 30 / 5 |
| Voice | supplementary | test | 1 | 13 |

**Newly refused sites: none (0).**

**Unresolved sites on changed entries (48, full list).** Each was read and annotated; the reading is reasoning, not a mechanical trace.
- **Note A:** `<t>.dtype` of a tensor_cuda query or scores tensor. Tensor.dtype is always one of the six supported names, so it is supported.
- **Note B:** BlockTC.COMPUTE_DTYPE. Every literal assignment in GraftRepository is bfloat16, float16 or float32. Dynamic setters exist only in glc_p0_audit.py:161/166, trinity_nope_graft_width_sweep.py:573/575 and trinity_t1_floor_reverb.py:619/623. Likely supported, not proven.

```
Project-Voice engine/tensor_requests/p1t_tests/adapters/tensorcuda_stub.py:205 | product | tc.zeros | arg dtype  [reading: run_p1t.py:605 dt in ('float32','float16'); literals elsewhere: supported]
GRAPA-Native-LLM artifacts/cc5/mutation/{answer_span_shift,distance_off_by_one,fresh_index,quarter_boundary,repeat_count,resume_identity_bypass,separator_omission,source_swap_noop,spillover_omission,weight_cap_bypass}/grapa/attention.py:82 | receipt | _causal_mask | q.dtype  [A] (10 sites)
GRAPA-Native-LLM artifacts/cc5_integration/mutation/{adam_reset,branch_identity_bypass,snapshot_pin_bypass,source_index_shift,spillover_drop,update_bound}/grapa/attention.py:82 | receipt | _causal_mask | q.dtype  [A] (6 sites)
GRAPA-Native-LLM artifacts/cc5_integration/original_mutation/{answer_span_shift,distance_off_by_one,fresh_index,quarter_boundary,repeat_count,resume_identity_bypass,separator_omission,source_swap_noop,spillover_omission,weight_cap_bypass}/grapa/attention.py:82 | receipt | _causal_mask | q.dtype  [A] (10 sites)
GRAPA-Native-LLM grapa/attention.py:82 | product | _causal_mask | q.dtype  [A]
GRAPA-Native-LLM orders/seeds/attn_probe_cc9_lead.py:53 | product | _causal_mask | q.dtype  [A]
GRAPA-Native-LLM scripts/cc10_ablation.py:2048 | script | _causal_mask | scores.dtype  [A]
GRAPA-Native-LLM scripts/cc58_bind_anatomy.py:363 | script | _causal_mask | query.dtype  [A]
GRAPA-Native-LLM scripts/cc69_write_time.py:511 | script | _causal_mask | query.dtype  [A]
GRAPA-Native-LLM scripts/cc72_query_side.py:381 | script | _causal_mask | query.dtype  [A]
GraftRepository artifacts/grm_xm3/evidence/trinity_nano_tc.py:316 | receipt | tc.tensor | _compute_dtype()  [returns 'bfloat16', :165/:951]
GraftRepository core/deepseek_v2_lite_tc.py:115 | product | tc.zeros | dt = dtype or BlockTC.COMPUTE_DTYPE  [B]
GraftRepository core/gemma4_tc.py:139 | product | tc.zeros | cdt = BlockTC.COMPUTE_DTYPE  [B]
GraftRepository core/gemma4_tc.py:740 | product | _causal_mask | sc.dtype  [A]
GraftRepository core/minicpm3_tc.py:292 | product | tc.zeros | BlockTC.COMPUTE_DTYPE  [B]
GraftRepository scripts/gpt_oss20b_e1_diag.py:2525,2526 | script | tc.tensor | compute_dtype  [B]
GraftRepository scripts/gpt_oss20b_expert_e1.py:4030,4031,4218,4219,4382,4383,4497,4498 | script | tc.tensor | compute_dtype  [B]
```

**Unresolved sites on unchanged entries (288, all `.astype(...)` except where noted; ENG2 cannot refuse these).** R = receipt. A line number listed twice means two calls on that line.
```
GRAPA artifacts/cc5/mutation/<10 dirs as above>/grapa/{attention.py:111, codebooks.py:87, model_mla.py:95,96,97,108} R
GRAPA artifacts/cc5_integration/mutation/<6 dirs as above>/grapa/{attention.py:111, codebooks.py:87, model_mla.py:95,96,97,108} R
GRAPA artifacts/cc5_integration/original_mutation/<10 dirs as above>/grapa/{attention.py:111, codebooks.py:87, model_mla.py:95,96,97,108} R
GRAPA grapa/attention.py:111; grapa/codebooks.py:87; grapa/fwd_precision.py:446,448,448,466,469; grapa/model_mla.py:95,96,97,108 (product)
GRAPA scripts/cc10_ablation.py:2045,2127,2128,2128,2133; cc46_determinism.py:1264,1266,1268; cc46_grad_anatomy.py:606; cc60_unit_census.py:823; cc61_unknown_decision.py:644,793,1182,1184 (script)
GRAPA tests/fixtures/cc60_unit_census_sealed_d45329ae.py:689; tests/test_cc60_cpu.py:237,269 (test)
GRM artifacts/grm_scout_fix5/before/core/graft_arena.py:512,513,831,854,5409,5410 R; artifacts/grm_scout_fix8/before/core/graft_arena.py:512,513,831,854,5437,5438 R
GRM artifacts/grm_xm3/evidence/tensor_functional.py:41 R; artifacts/grm_xm3/evidence/trinity_nano_tc.py:48,52,80,103,131,177,259,262 R
GRM core/deepseek_v2_lite_tc.py:96,325,611,612; :209 (kv_int4_unpack out_dtype); core/gemma4_tc.py:292,425; :263,289 (kv_int4_unpack out_dtype)
GRM core/gpt_oss20b_tc.py:248,267,415,554,619,619,629,667,802,850,1004,1007,1161,1165,1307; core/graft_arena.py:523,524,842,865,5555,5556; core/kv_graft.py:123,124,229,230
GRM core/minicpm3_tc.py:114,161,176,256,302,324,354,355,376; core/mistral7b_tc.py:164,168,235,243,339,340,350,508,568,597,598,624; core/qwen35_tc.py:151; core/qwen3_tc.py:54; scribe/losses.py:27 (product)
GRM scripts/_diag_step_vs_manual_ab.py:292; gpt_oss20b_expert_e1.py:845,3661,3668,3739; gpt_oss20b_stream_forward_smoke.py:227,228,523; grm_cmc1_gpu_arms.py:534,565; grm_det1_3_snapshot.py:1445; grm_rs2_mount_read_gpu.py:323,736,852,856; trinity_nope_graft_width_sweep.py:593; trinity_t1_floor_reverb.py:640,668 (script)
GRM tests/gemma4_kv_quant_apa_ppl.py:66; gemma4_kv_quant_ppl.py:59; gemma4_kv_quant_refine_sweep.py:63; nc17/p4_gate_b_state.py:105; test_gpt_oss20b_scaffold.py:477; test_graft_e1_mla_round1.py:99,103 (test)
```
The same list, one line per file, prints with `python3 artifacts/eng2/a3/census_report.py unresolved_unchanged`.

## Gates and receipts
- **A3-G1 PASS.**
  - R1 is done (`A3_SOURCE.diff`, `ids_test_eng1_diff.txt`).
  - Mutation coverage holds 31 of 31 (`MUTATION_COVERAGE.json`).
  - L38 `expected_changes` is `{}`.
  - The censuses were re-collected by the unchanged `stage.step_census`. ids_regnorm and ids_regnorm_candidate went from 578 to 568 and lost exactly the 10 pins. ids_regdtype (1779), ids_eng2 (394) and ids_audit (253) are byte-identical. All 7 collect files have rc 0 and no errors (`A3_CPU_AND_CENSUS.json` C5–C6).
- **A3-G3 PASS.**
  - **CPU suite, test_eng2.py:** the candidate passes 253 of 253. The baseline fails 120, equal to A2's set and equal to the 96 registered F1 IDs plus the 24 non-string sentinel IDs.
  - **CPU suite, test_eng1.py `-k cpu`:** 47 of 47 pass on both arms. These are the same 47 IDs as before the edit (C1–C4).
  - **MF1 and MF2 through evaluate():** MF1 kills 12 of 12 and MF2 kills 48 of 48, every killer traced through its hunk ([63,63] and [75,75]). K2, K3, S1 and S2 are ok (`MUTANT_EVAL_CPU.json`).
  - **Staging:** baseline_expected_failures (166), gpu_lines (21), the staged script, regression_files and source_reading_tests are byte-identical. The arms differ only in `tests/test_eng1.py` (17 of 17). The seal re-verifies with all 17 arms.
  - **Dry run (`G4_DRY_dry_4.json`):** 21 of 21 labels ok. The 4 root labels reach the governor (exit 78); the 17 dependents are refused at their dependency check; S1 gives 78 for all 21.
  - **Audit (`A3G3_AUDIT_RECEIPT.json`):** 11 of 11 steps ok. The staged run exits 0; score_gpu.py exits 1 with "audit run cannot establish GPU acceptance", as designed.
- **Scope PASS** (`A3_SCOPE.json`).
  - Changed outside a3/ and the arms: test_eng1.py (lines 180/186/187/266 only), L38_PINS.json, collect_{baseline,candidate}_regnorm.json, ids_regnorm(_candidate).txt and staging_seal.json.
  - Added: a3/ (88 files), gpu_runs/dry_4 (13 files), gpu_runs/audit_4 (33 files).
  - Removed: none. No product code changed.
- **Repairs:** none on any gate.

## Deviations
- **D1 (order fault):** the order names `gpu_runs/dry_3` and `gpu_runs/audit_3` as new roots, but both already hold A2's receipts. Writing there would edit receipts in place, and the scripts need fresh roots. I used `dry_4` and `audit_4`, which are outside the order's literal writable list.
- **D2:** I read ENG1's G5 run root from the eng1 worktree (`…/tensor_fork_apa_wt_eng1/artifacts/eng1/tool/gpu_runs/g5_1114`). This worktree lacks the git-ignored child_result.json and logs. The manifest copies in both worktrees are byte-identical.
- **D3 (L56a):** the census self-test caught a classification slip in my tool (`dtype=float` was classed as traced rather than a direct non-string). I fixed it and re-ran.
- **D4:** the census goes beyond the order's import listing. It adds a cross-file alias step (for example, GraftRepository's `from core.mistral7b_tc import F, tc`) and a supplementary `tc.<entry>(` group. The cross-file step added 101 changed-entry sites that the import listing alone missed.
- **D5:** I refined the `a or b` rule after a false positive at deepseek:115, which the tool had read as possibly None.
- **D6:** R3 says to run both test files "as A-G2 ran them", but A-G2 ran test_eng2.py only. I ran test_eng1.py's `-k cpu` subset through the same child.
- **D7:** L38_PINS.json carries one added `"a3"` record next to the emptied `expected_changes`.

**Residuals:**
- The docstring at test_eng1.py:268 still mentions "the float64 request". I left it because the order allows edits only to the three lists and one comment.
- Census limits: aliases are bound without regard to flow or scope, values are traced within one file only, star imports from repo modules are not followed, and non-.py files are not parsed.
- Scratch is kept under `/tmp/eng2_a3_*` (nothing deleted).

## Prior art
- **Per-mutant killing sets:** DeMillo, Lipton and Sayward (1978); PIT's killingTests (unverified, lead to check: "PIT mutation killingTests"). Taken: a mutant counts as killed when a registered test fails. Mine: intersecting the killer sets with the removed IDs, and binding each check to the manifest-verified receipts.
- **Census method:** call-site discovery over Python ASTs, as in linters and codemod tools (Bowler, LibCST, pyupgrade), plus crude in-file constant tracing in the reaching-definitions tradition (Kildall 1973; Aho, Sethi and Ullman). All unverified, lead to check: "Bowler codemod", "LibCST call site", "Kildall 1973 constant propagation". Mine: the entry table, the argument classes, the cross-file alias fixpoint, the `a or b` rule and the newly-refused rule.
- **Scope and seal:** manifest diffing (BSD mtree(8), unverified) and SHA-256 (NIST FIPS 180-4).
- **Re-staging, dry run and audit:** ENG2's sealed tool and A2's scripts, reused. No new algorithm.

Files are in `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_eng2/`:
- **Changed:**
  - `tensor_cuda/tests/test_eng1.py`
  - `artifacts/eng2/L38_PINS.json`
  - `artifacts/eng2/tool/{staging_seal.json, ids_regnorm.txt, ids_regnorm_candidate.txt, collect_*.json}`
  - `artifacts/eng2/tool/arms/*/tensor_cuda/tests/test_eng1.py`
- **New:**
  - `artifacts/eng2/a3/` (LEDGER.md, scripts, receipts)
  - `artifacts/eng2/tool/gpu_runs/dry_4/`
  - `artifacts/eng2/tool/gpu_runs/audit_4/`
