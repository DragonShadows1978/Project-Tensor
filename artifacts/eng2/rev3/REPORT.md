# ENG2-REV3 — A3 narrow cross-model review

**Completed: one MINOR census-completeness finding; no BLOCKER or MAJOR finding.** The test edit, retained mutant kills, staging seal and recorded dry-run routing check sound. No confirmed newly refused consumer call site was found, but A3’s consumer inventory is incomplete. This is a CPU/static/recorded-receipt review; it does not establish GPU acceptance or authorize promotion.

Reviewed before `e2f166b8`, after `10738fe3`; mechanism: `orders/ENG2_A3_ENG1_ROWS_AND_CONSUMERS.md`; review order: `orders/ENG2_REV3_A3_REVIEW.md`. Registration is in `REGISTRATION.md`; work and repair history in `LEDGER.md`. All deliverables are inside `artifacts/eng2/rev3/`; retained scratch is `/tmp/eng2_rev3_20261114/`. No product/consumer edits, GPU use, git writes, subagents or deletes.

## Finding (verbatim)

E2R3-F01 — MINOR — artifacts/eng2/a3/consumer_census.py:556: File selection omits 16 changed-entry call sites, including GRAPA grapa/tripwire.py:655 with unresolved self.dtype(t). The independent census finds 498 sites and 49 unresolved, versus A3’s 482 and 48. Reproduction: run consumer_scan.py and finish_consumer.py; omitted_by_a3 in Q3_consumer_final.json lists all 16. Suggested fix: census every tracked Python file, resolve field/namespace receivers, and update the site lists and counts. No newly refused call site was confirmed.

Evidence class: independent static census plus source reasoning. The import/cross-file/bare-`tc` file selection at `consumer_census.py:556–568` excludes files whose only engine access uses fields or namespaces. For the new unresolved site, `EngineDevice.__init__` stores the constructor’s `tc` at `grapa/tripwire.py:630`; `grapa/train.py:758` imports tensor_cuda and `:1195` passes it into `EngineDevice`. `tripwire.py:643` returns `str(t.dtype)`, which supports the expected engine-tensor path but remains unresolved for arbitrary injected objects. The other 15 omitted sites pass supported literals. Both extra Voice files are byte-identical at A3’s recorded HEAD, so these are historical omissions, not checkout drift. Finding details and suggested fix: `findings.json`.

All 16 omitted sites (the explicit dtype is shown):

| Repository | File:line | Dtype argument | Role |
|---|---|---|---|
| GRAPA-Native-LLM | `grapa/tripwire.py:655` | `self.dtype(t)` | product |
| GRAPA-Native-LLM | `scripts/cc27_tiny.py:223` | `'int64'` | script |
| GRAPA-Native-LLM | `scripts/cc27_tiny.py:224` | `'float32'` | script |
| GRAPA-Native-LLM | `scripts/cc28_tiny.py:270` | `'int64'` | script |
| GRAPA-Native-LLM | `scripts/cc28_tiny.py:271` | `'float32'` | script |
| GRAPA-Native-LLM | `scripts/cc29_tiny.py:298` | `'int64'` | script |
| GRAPA-Native-LLM | `scripts/cc29_tiny.py:299` | `'float32'` | script |
| GRAPA-Native-LLM | `scripts/cc30_tiny.py:371` | `'int64'` | script |
| GRAPA-Native-LLM | `scripts/cc30_tiny.py:372` | `'float32'` | script |
| GRAPA-Native-LLM | `scripts/cc67_tiny.py:584` | `'int64'` | script |
| GRAPA-Native-LLM | `scripts/cc67_tiny.py:585` | `'float32'` | script |
| GRAPA-Native-LLM | `scripts/cc68_tiny.py:420` | `'int64'` | script |
| GRAPA-Native-LLM | `scripts/cc68_tiny.py:421` | `'float32'` | script |
| GraftRepository | `scripts/grm_rs4_ceiling_gpu.py:512` | `'float32'` | script |
| Project-Voice | `engine/kokoro_tc/tests/test_gpu_residency.py:62` | `'float16'` | test |
| Project-Voice | `engine/kokoro_tc/unit_dose.py:76` | `'float32'` | product |

## Q1 — the edit: checked sound

Ran `git diff e2f166b8 10738fe3 -- tensor_cuda` and `git diff --name-only` independently (`source.diff`, `commands.jsonl`). Only `tensor_cuda/tests/test_eng1.py` changed: lines 180, 186, 187 and 266, removing float64 from E2_DTYPES, its mapping and the bits-test parameter list, and changing the explanatory comment. No other tensor_cuda file changed.

`review_checks.py q1` used `git show <commit>:tensor_cuda/tests/test_eng1.py` into isolated scratch and ran four fresh `python3 -m pytest --collect-only -q -p no:cacheprovider --confcutdir=... --rootdir=...` collections: both reviewed commits against both staged package arms, with CUDA hidden and plugin autoload disabled. Exact normalized-ID comparisons exclude logs/timing.

| Package arm | Before | After | Removed | Added |
|---|---:|---:|---:|---:|
| baseline | 156 | 146 | 10 | 0 |
| candidate | 156 | 146 | 10 | 0 |

The removed sets are identical and equal exactly to the old L38 expected_changes keys:

- `test_eng1.py::test_e2_pow_grad[float64-p+0.5]`
- `test_eng1.py::test_e2_pow_grad[float64-p+2]`
- `test_eng1.py::test_e2_pow_grad[float64-p+3]`
- `test_eng1.py::test_e2_pow_grad[float64-p-0.5]`
- `test_eng1.py::test_e2_pow_grad[float64-p-1.5]`
- `test_eng1.py::test_e2_pow_grad_bits[float64-p+0.5]`
- `test_eng1.py::test_e2_pow_grad_bits[float64-p+2]`
- `test_eng1.py::test_e2_pow_grad_bits[float64-p+3]`
- `test_eng1.py::test_e2_pow_grad_bits[float64-p-0.5]`
- `test_eng1.py::test_e2_pow_grad_bits[float64-p-1.5]`

`supplement_checks.py` also checked the stored staging censuses at both commits: regnorm and regnorm_candidate each 578→568, losing only those ten IDs; regdtype (1779), eng2 (394) and audit (253) byte-identical. All 17 sealed arms carry the after-commit test-file hash, and current L38 expected_changes is empty. Receipts: `Q1_ids.json`, all four `collect_*.txt` and `ids_*.txt`, `Q1_staged_censuses.json`, `Q1_run.txt`, `supplement_run_repaired.txt`. Evidence class: CPU collection and exact file/ID comparison, not test execution.

## Q2 — mutation coverage: checked sound

Ran independent `review_checks.py q2`, without executing the A3 mutation-coverage script or re-executing any mutant. ENG1 registry has 14 original records plus the two registered amendments in `artifacts/eng1/a1/MUTANTS.json`; ENG2 has 15 records. The matcher treats pytest parameter brackets literally and only `*` as a wildcard. Positive controls against actual retired IDs match 5, 1 and 10 IDs. Every pattern has zero removed-ID matches; every mutant retains registered killers that failed on its own admitted G5 line. Child call outcomes and JUnit agree by test ID; parent actual_killers are among the retained failing registered tests.

Before parsing each G5 input, verified its SHA-256 against that run’s canonical manifest. The manifests are stored under the lead receipt directories, not inside the run roots:

- eng1: `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_eng2/artifacts/eng1/lead/g5_1114/eng1_g5_MANIFEST.sha256`; SHA-256 `a722e5b240971ba24587eb5970151e999140b0ec3b6936dd67edf70240fbb9c9`; 1022 manifest entries; 48 verified inputs. Run root: `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_eng1/artifacts/eng1/tool/gpu_runs/g5_1114`.
- eng2: `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_eng2/artifacts/eng2/lead/g5_1114/eng2_g5_MANIFEST.sha256`; SHA-256 `e957c001948c783833018776174fe6232f372d75598e1fa0ace8a4485253d72a`; 1855 manifest entries; 45 verified inputs. Run root: `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_eng2/artifacts/eng2/tool/gpu_runs/g5_1114`.

93 files verified in total: parent_result.json, child_result.json and junit.xml for each of 31 mutants. Full per-file expected/actual hashes and the complete registered patterns, matches and retained failed IDs are in `Q2_mutants.json`.

| Side | Mutant | Patterns | Removed matches | Retained failed killers | Example retained failed killer |
|---|---|---:|---:|---:|---|
| eng1 | M01 | 3 | 0 | 3 | `test_e1_refuse[three_neg1]` |
| eng1 | M02 | 3 | 0 | 15 | `test_e1_infer[r1_p0]` |
| eng1 | M03 | 2 | 0 | 2 | `test_e1_refuse[zero_product]` |
| eng1 | M04 | 3 | 0 | 3 | `test_e1_refuse[negative_with_neg1]` |
| eng1 | M05 | 2 | 0 | 2 | `test_e2_pow_grad[float16-p-0.5]` |
| eng1 | M06 | 2 | 0 | 10 | `test_e2_pow_grad[bfloat16-p+0.5]` |
| eng1 | M07 | 2 | 0 | 2 | `test_e2_pow_grad[float16-p-0.5]` |
| eng1 | M08 | 4 | 0 | 4 | `test_e4_empty[rms_norm-last0-cpu]` |
| eng1 | M09 | 2 | 0 | 2 | `test_e4_empty[apa_blend_softmax-last0-cpu]` |
| eng1 | M10 | 3 | 0 | 3 | `test_e4_refuse[topk-k1-last0-cpu]` |
| eng1 | M11 | 2 | 0 | 8 | `test_e5_refuse_cpu[avg_pool2d-oh_zero]` |
| eng1 | M12 | 1 | 0 | 4 | `test_e5_refuse_cpu[avg_pool2d-ow_only]` |
| eng1 | M13 | 1 | 0 | 4 | `test_e5_refuse_cpu[avg_pool2d-trunc_gap]` |
| eng1 | M14 | 1 | 0 | 1 | `test_e6_terrain_writes_nothing` |
| eng1 | M15 | 1 | 0 | 1 | `test_e2_pow_grad_nonpow2[bfloat16]` |
| eng1 | M16 | 1 | 0 | 4 | `test_e5_refuse_cpu[avg_pool2d-ow_zero]` |
| eng2 | MF1 | 12 | 0 | 12 | `test_f1_refuse_cpu[tensor-double-cpu]` |
| eng2 | MF2 | 48 | 0 | 48 | `test_f1_factory_refuses_before_allocation_cpu[ones-double]` |
| eng2 | MN1_LN | 16 | 0 | 16 | `test_n1_half_rule_bits[LayerNorm-bfloat16-near_eps-backward]` |
| eng2 | MN1_RMS | 16 | 0 | 16 | `test_n1_half_rule_bits[RMSNorm-bfloat16-near_eps-backward]` |
| eng2 | MN1_BN | 16 | 0 | 16 | `test_n1_half_rule_bits[BatchNorm1D-bfloat16-near_eps-backward]` |
| eng2 | MN1_IN | 16 | 0 | 16 | `test_n1_half_rule_bits[InstanceNorm2D-bfloat16-near_eps-backward]` |
| eng2 | MN2_LN | 5 | 0 | 5 | `test_n1_half_rule_bits[LayerNorm-bfloat16-near_eps-backward]` |
| eng2 | MN2_RMS | 4 | 0 | 4 | `test_n1_half_rule_bits[RMSNorm-bfloat16-near_eps-backward]` |
| eng2 | MN2_BN | 6 | 0 | 6 | `test_n1_batchnorm_running_stats[bfloat16-near_eps]` |
| eng2 | MN2_IN | 5 | 0 | 5 | `test_n1_half_rule_bits[InstanceNorm2D-bfloat16-near_eps-backward]` |
| eng2 | MN3_LN | 11 | 0 | 11 | `test_n1_half_rule_bits[LayerNorm-bfloat16-near_eps-backward]` |
| eng2 | MN3_RMS | 9 | 0 | 9 | `test_n1_half_rule_bits[RMSNorm-bfloat16-near_eps-backward]` |
| eng2 | MN3_BN | 14 | 0 | 14 | `test_n1_batchnorm_running_stats[bfloat16-near_eps]` |
| eng2 | MN3_IN | 11 | 0 | 11 | `test_n1_half_rule_bits[InstanceNorm2D-bfloat16-near_eps-backward]` |
| eng2 | MN4_LN | 4 | 0 | 4 | `test_n1_float32_bits[LayerNorm-near_eps-backward]` |

Result: 31/31 retain a recorded G5 kill. Receipts: `Q2_mutants.json`, `Q2_run_repaired.txt`; initial schema-slip transcript is retained in `Q2_run.txt`. Evidence class: manifest-bound recorded G5 results plus independent set matching. Fresh GPU kills under the new seal remain the lead’s task.

## Q3 — consumer census: zero confirmed refusals; inventory correction needed

Ran own `consumer_scan.py` over every tracked Python file listed by `git ls-files`, loading each pinned HEAD file through `git show`, parsing with Python ast, resolving imports/re-exports/assignment aliases, and conservatively tracing explicit dtype values in lexical scopes. Retained all unresolved calls and unbound receivers. `finish_consumer.py` then resolved 55 field/namespace/runtime-dictionary receiver candidates by source reading and compared the results to A3. It also used `git grep -n -I -e ^ -- *.py` to confirm all current tracked Python text matches the pinned HEAD sources (no source differences in any repository). No consumer module was imported or executed. Parse failures: zero. All seven entry types were considered, including those with zero calls.

| Repository | Pinned HEAD | Tracked Python | Explicit dtype sites | Supported | Unresolved | Confirmed newly refused |
|---|---|---:|---:|---:|---:|---:|
| Project-Voice | `caeb64d3898b1d2d361d1f4351cae77169dd97bb` | 768 | 23 | 22 | 1 | 0 |
| GRAPA-Native-LLM | `f22ad802118849ffd601301e49fdbe1b26326649` | 1018 | 296 | 263 | 33 | 0 |
| GraftRepository | `03f6e45c7f39a171dafc459236e2f4ff27d04279` | 666 | 179 | 164 | 15 | 0 |

Thus “0 newly refused call sites” is supported as **zero statically confirmed bad string/non-string dtype arguments**, with 49 unresolved explicit-dtype sites retained; it is not proof that arbitrary runtime values or external callers cannot be refused. A3 reports 482 explicit-dtype changed-entry sites and 48 unresolved; the corrected inventory is 498 and 49. The Voice HEAD advanced from A3’s 17846484 to caeb64d3, with no tracked-Python working-copy differences; the omitted Voice operands were separately checked at the historical revision. GRAPA and GRM HEADs match A3. GRM remains on the A3-reviewed lc1-wip snapshot; no claim about a different main snapshot is made.

Role counts (all roles receive the same classification; paths under p1t_tests are counted as tests here):

| Repository | Role | Supported | Unresolved | Sites |
|---|---|---:|---:|---:|
| Project-Voice | product | 13 | 0 | 13 |
| Project-Voice | script | 0 | 0 | 0 |
| Project-Voice | test | 9 | 1 | 10 |
| Project-Voice | receipt | 0 | 0 | 0 |
| GRAPA-Native-LLM | product | 12 | 3 | 15 |
| GRAPA-Native-LLM | script | 38 | 4 | 42 |
| GRAPA-Native-LLM | test | 5 | 0 | 5 |
| GRAPA-Native-LLM | receipt | 208 | 26 | 234 |
| GraftRepository | product | 116 | 4 | 120 |
| GraftRepository | script | 6 | 10 | 16 |
| GraftRepository | test | 26 | 0 | 26 |
| GraftRepository | receipt | 16 | 1 | 17 |

Entry counts:

| Repository | tensor | zeros | ones | randn | rand | nn.parameter | functional._causal_mask |
|---|---:|---:|---:|---:|---:|---:|---:|
| Project-Voice | 22 | 1 | 0 | 0 | 0 | 0 | 0 |
| GRAPA-Native-LLM | 263 | 1 | 0 | 0 | 0 | 0 | 32 |
| GraftRepository | 167 | 10 | 0 | 0 | 0 | 0 | 2 |

**Original unresolved-site spot checks: 11 checked, across every repository, including live and receipt code.** Each supports the seat’s source reading with the qualifications below; none demonstrates a newly refused call. `Q3_spot_checks.json` contains exact source excerpts, file hashes and source-chain references for each. `Q3_engine_dtype_source.json` binds the Tensor.dtype reading to reviewed-commit C++ source: the property returns dtype_name, whose six named cases are supported strings.

| Repository / file:line | Independent source reading |
|---|---|
| Project-Voice `engine/tensor_requests/p1t_tests/adapters/tensorcuda_stub.py:205` | holds for recorded harness callers; dtype is an unguarded external parameter, so retained unresolved |
| GRAPA-Native-LLM `grapa/attention.py:82` | holds for tensor_cuda q: q.dtype is an engine dtype string; call uses q.device and tensor_cuda matmul |
| GRAPA-Native-LLM `artifacts/cc5/mutation/answer_span_shift/grapa/attention.py:82` | holds on the same tensor_cuda query path in the tracked mutant receipt copy |
| GRAPA-Native-LLM `scripts/cc10_ablation.py:2048` | holds: scores is produced by tensor_cuda matmul and tensor arithmetic, so scores.dtype is an engine string |
| GraftRepository `artifacts/grm_xm3/evidence/trinity_nano_tc.py:316` | holds: _compute_dtype returns BlockTC.COMPUTE_DTYPE, with in-file bfloat16 default and loader assignment |
| GraftRepository `core/deepseek_v2_lite_tc.py:115` | qualified reading holds: dtype or BlockTC.COMPUTE_DTYPE; registered callers use supported values; external override remains unresolved |
| GraftRepository `core/gemma4_tc.py:139` | qualified reading holds: cdt is shared compute dtype, assigned bfloat16 by loader; mutable global remains unresolved |
| GraftRepository `core/gemma4_tc.py:740` | holds: sc is tensor_cuda matmul result reshaped without changing dtype; sc.dtype is an engine string |
| GraftRepository `core/minicpm3_tc.py:292` | qualified reading holds: shared compute dtype is float16 by default and bfloat16 in loader; mutable global retained unresolved |
| GraftRepository `scripts/gpt_oss20b_e1_diag.py:2525` | qualified reading holds: runtime BlockTC.COMPUTE_DTYPE feeds compute_dtype; runtime loader imports tensor_cuda; mutable setting retained unresolved |
| GraftRepository `scripts/gpt_oss20b_expert_e1.py:4030` | qualified reading holds: runtime dictionary supplies tc and shared BlockTC dtype; no claim about arbitrary setters |

**Unchanged-entry L60 parity: checked sound within the recorded probes.** `Q3_L60_parity.json` compares all recorded result/exception fields, excluding source-site metadata and normalizing only process-specific Tensor object addresses in pybind TypeError messages. The order’s top-level L60 files contain 54 unchanged-entry rows: tc.empty, Tensor.astype and three dequant out_dtype entries. They match behaviorally; the two raw differences are object addresses. Those top-level files do not contain the affine helper probes. The expanded A2 L60 files cited by A3 contain those helpers and RegisteredMargins.lookup: all 162 unchanged-entry rows match behaviorally; six raw differences are object addresses. This verifies the stated finite census, not all possible arguments.

All 49 unresolved explicit-dtype changed-entry sites (including the newly found tripwire site):

| Repository | File:line | Dtype expression |
|---|---|---|
| GRAPA-Native-LLM | `artifacts/cc5/mutation/answer_span_shift/grapa/attention.py:82` | `q.dtype` |
| GRAPA-Native-LLM | `artifacts/cc5/mutation/distance_off_by_one/grapa/attention.py:82` | `q.dtype` |
| GRAPA-Native-LLM | `artifacts/cc5/mutation/fresh_index/grapa/attention.py:82` | `q.dtype` |
| GRAPA-Native-LLM | `artifacts/cc5/mutation/quarter_boundary/grapa/attention.py:82` | `q.dtype` |
| GRAPA-Native-LLM | `artifacts/cc5/mutation/repeat_count/grapa/attention.py:82` | `q.dtype` |
| GRAPA-Native-LLM | `artifacts/cc5/mutation/resume_identity_bypass/grapa/attention.py:82` | `q.dtype` |
| GRAPA-Native-LLM | `artifacts/cc5/mutation/separator_omission/grapa/attention.py:82` | `q.dtype` |
| GRAPA-Native-LLM | `artifacts/cc5/mutation/source_swap_noop/grapa/attention.py:82` | `q.dtype` |
| GRAPA-Native-LLM | `artifacts/cc5/mutation/spillover_omission/grapa/attention.py:82` | `q.dtype` |
| GRAPA-Native-LLM | `artifacts/cc5/mutation/weight_cap_bypass/grapa/attention.py:82` | `q.dtype` |
| GRAPA-Native-LLM | `artifacts/cc5_integration/mutation/adam_reset/grapa/attention.py:82` | `q.dtype` |
| GRAPA-Native-LLM | `artifacts/cc5_integration/mutation/branch_identity_bypass/grapa/attention.py:82` | `q.dtype` |
| GRAPA-Native-LLM | `artifacts/cc5_integration/mutation/snapshot_pin_bypass/grapa/attention.py:82` | `q.dtype` |
| GRAPA-Native-LLM | `artifacts/cc5_integration/mutation/source_index_shift/grapa/attention.py:82` | `q.dtype` |
| GRAPA-Native-LLM | `artifacts/cc5_integration/mutation/spillover_drop/grapa/attention.py:82` | `q.dtype` |
| GRAPA-Native-LLM | `artifacts/cc5_integration/mutation/update_bound/grapa/attention.py:82` | `q.dtype` |
| GRAPA-Native-LLM | `artifacts/cc5_integration/original_mutation/answer_span_shift/grapa/attention.py:82` | `q.dtype` |
| GRAPA-Native-LLM | `artifacts/cc5_integration/original_mutation/distance_off_by_one/grapa/attention.py:82` | `q.dtype` |
| GRAPA-Native-LLM | `artifacts/cc5_integration/original_mutation/fresh_index/grapa/attention.py:82` | `q.dtype` |
| GRAPA-Native-LLM | `artifacts/cc5_integration/original_mutation/quarter_boundary/grapa/attention.py:82` | `q.dtype` |
| GRAPA-Native-LLM | `artifacts/cc5_integration/original_mutation/repeat_count/grapa/attention.py:82` | `q.dtype` |
| GRAPA-Native-LLM | `artifacts/cc5_integration/original_mutation/resume_identity_bypass/grapa/attention.py:82` | `q.dtype` |
| GRAPA-Native-LLM | `artifacts/cc5_integration/original_mutation/separator_omission/grapa/attention.py:82` | `q.dtype` |
| GRAPA-Native-LLM | `artifacts/cc5_integration/original_mutation/source_swap_noop/grapa/attention.py:82` | `q.dtype` |
| GRAPA-Native-LLM | `artifacts/cc5_integration/original_mutation/spillover_omission/grapa/attention.py:82` | `q.dtype` |
| GRAPA-Native-LLM | `artifacts/cc5_integration/original_mutation/weight_cap_bypass/grapa/attention.py:82` | `q.dtype` |
| GRAPA-Native-LLM | `grapa/attention.py:82` | `q.dtype` |
| GRAPA-Native-LLM | `grapa/tripwire.py:655` | `self.dtype(t)` |
| GRAPA-Native-LLM | `orders/seeds/attn_probe_cc9_lead.py:53` | `q.dtype` |
| GRAPA-Native-LLM | `scripts/cc10_ablation.py:2048` | `scores.dtype` |
| GRAPA-Native-LLM | `scripts/cc58_bind_anatomy.py:363` | `query.dtype` |
| GRAPA-Native-LLM | `scripts/cc69_write_time.py:511` | `query.dtype` |
| GRAPA-Native-LLM | `scripts/cc72_query_side.py:381` | `query.dtype` |
| GraftRepository | `artifacts/grm_xm3/evidence/trinity_nano_tc.py:316` | `_compute_dtype()` |
| GraftRepository | `core/deepseek_v2_lite_tc.py:115` | `dt` |
| GraftRepository | `core/gemma4_tc.py:139` | `cdt` |
| GraftRepository | `core/gemma4_tc.py:740` | `sc.dtype` |
| GraftRepository | `core/minicpm3_tc.py:292` | `BlockTC.COMPUTE_DTYPE` |
| GraftRepository | `scripts/gpt_oss20b_e1_diag.py:2525` | `compute_dtype` |
| GraftRepository | `scripts/gpt_oss20b_e1_diag.py:2526` | `compute_dtype` |
| GraftRepository | `scripts/gpt_oss20b_expert_e1.py:4030` | `compute_dtype` |
| GraftRepository | `scripts/gpt_oss20b_expert_e1.py:4031` | `compute_dtype` |
| GraftRepository | `scripts/gpt_oss20b_expert_e1.py:4218` | `compute_dtype` |
| GraftRepository | `scripts/gpt_oss20b_expert_e1.py:4219` | `compute_dtype` |
| GraftRepository | `scripts/gpt_oss20b_expert_e1.py:4382` | `compute_dtype` |
| GraftRepository | `scripts/gpt_oss20b_expert_e1.py:4383` | `compute_dtype` |
| GraftRepository | `scripts/gpt_oss20b_expert_e1.py:4497` | `compute_dtype` |
| GraftRepository | `scripts/gpt_oss20b_expert_e1.py:4498` | `compute_dtype` |
| Project-Voice | `engine/tensor_requests/p1t_tests/adapters/tensorcuda_stub.py:205` | `dtype` |

Receipts: `Q3_consumer_final.json` (all 498 sites, full unresolved and omission lists), `Q3_consumer_scan.json` (initial inventory, default-only calls, all other candidate receivers and per-file hashes), `Q3_git_commands.json`, `Q3_scan_run.txt`, `Q3_finish_run.txt`, `Q3_spot_checks.json`, `Q3_extra_sources.txt`, `Q3_spot_sources.txt`, `Q3_engine_dtype_source.json`, and `Q3_L60_parity.json`. Source copies remain in scratch. Limits: untracked files and WorkTree paths are excluded; arbitrary reflection/eval, dynamic import strings, external calls and mutable-global values cannot be exhaustively proven by this static pass.

## Q4 — seal and staging: checked sound

Ran `review_checks.py q4`, importing the copied tool’s original `evidence.verify_seal()` and calling its original `verify_arm(arm, seal)` on every arm. This recomputed order/tool/registration digests, exact arm file maps, symlink absence and engine pins. All 17 arms passed, arms_pending is empty. Seal SHA-256: `b424bdcfa0d01acb84cfe71eec21dc4597a40c2ebd77d99c9143aee4f17df826`. The copied evidence module has no --version implementation: asking --version exits zero with no output; its source SHA-256 is recorded as its identifier in `Q4_seal.json`.

Independently compared `a3/G4_DRY_dry_4.json` with the sealed gpu_lines.json: exactly 21 unique registered labels, matching arm, suite, inner timeout, governor label, min/max and margin. `supplement_checks.py` corroborated original dry_4 root parent_result.json and stdout.log: 4 roots reached dry governor exit 78 without starting a child; 17 dependents had their registered dependency refusals and no attempt. Each of all 21 S1 argv routes returned 78. This is recorded dry-run routing evidence, not GPU acceptance. No staged line was executed by this review. Receipts: `Q4_seal.json`, `Q4_run.txt`, `Q4_dry_corroborated.json`, `supplement_run_repaired.txt`.

## Commands and execution controls

Every independent check used this foreground prefix (timeouts 60/120 seconds except the 540-second full consumer pass):

```sh
TMPDIR=/tmp/eng2_rev3_20261114 CUDA_VISIBLE_DEVICES= \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 timeout <seconds>s nice -n 10 python3 <script> [argument]
```

Independent invocations: `review_checks.py q1`, `review_checks.py q2` (one repaired schema lookup), `consumer_scan.py`, `finish_consumer.py`, `review_checks.py l60` (label correction), `review_checks.py q4`, and `supplement_checks.py` (one repaired source needle). Instrument receipts: `preflight_and_final_checks.txt` (Python 3.12.3, pytest 9.0.2, NumPy 2.4.2, git 2.43.0). Q1 exact subprocess commands and their full outputs are in `commands.jsonl`; the full all-file git command list and output digests are in `Q3_git_commands.json`; source outputs are retained in scratch and source hashes/call excerpts in the census. Resource-heavy execution was not needed; the consumer scan streams one AST at a time and no job was designed to exceed 1GB. At most two CPU-busy processes; no background command launched. When the exec tool yielded a running command, waited again with yield_time_ms=300000 without intervening status messages.

## Deviations and repairs

- D01: initial house-rule/memory reads used TMPDIR=/tmp and omitted nice/thread controls, before any check. Repeated the house-rule read under required controls (`house_rules_read.txt`); all independent checks used the required environment. Memory search had no relevant hit and supplied no review facts.
- D02: exploratory reads assumed MANIFEST.sha256 lived in each G5 run root; neither path exists. Located the canonical lead manifests and used only those, checking every G5 input before parsing.
- R01: review_checks.py q2 initially requested report[nodeid]; real receipt schema uses report[id]. One schema repair, then 31/31 passed; first transcript retained.
- R02: initial L60 label set omitted the exact recorded tc.nn.parameter and CUDA tensor labels, causing changed rows to appear in the unchanged comparison. Corrected the labels, preserved Q3_L60.initial.json, then retained raw object-address differences and separately checked behavioral parity. No product or historical receipt was edited.
- D03: spot-source inline script requested nonexistent grapa/engine_native.py. Removed that operand from the review script and reran successfully; no consumer content executed.
- R03: supplement_checks.py searched for std::string dtype_name; the declaration is const char*. Corrected that source needle once and reran successfully. Both transcripts retained. No script exception required more than one repair; review_checks.py also had the separately recorded L60 classification correction.
- Instrument note: NumPy version was read from numpy.__version__; the copied evidence module accepts --version as an ignored argument and emits no version. The exact tool source digest is recorded. No gate was run.

## Prior art

- Pytest collection: pytest project (Krekel and contributors, 2004 onward), unverified — lead to check “pytest history collection”. Taken: fresh test-ID collection. Review-specific: exact before/after/arm comparison and L38 binding.
- Mutation kill-set review: DeMillo, Lipton and Sayward (1978), unverified — lead to check “mutation testing 1978”; Python standard-library shell-pattern semantics. Taken: a registered failing test witnesses a mutant kill, with literal pytest brackets and star wildcards. Review-specific: retired-ID exclusion and independent manifest binding for all 31 mutants. No new mutation algorithm.
- Static census: Python project ast module; Kildall (1973) data-flow tradition, unverified — lead to check “Kildall 1973 data flow analysis”. Taken: syntax traversal, conservative constant/value and alias propagation. Review-specific: all-file seven-entry boundary census, receiver follow-up, role counts and omission comparison. Proposed fix uses the same standard AST inventory method; no novelty claimed.
- Integrity: SHA-256, NIST FIPS 180-4 (2015), unverified — lead to check “NIST FIPS 180-4 SHA-256”. Taken: content digests and manifest matching. Review-specific: binding each consumed G5 receipt and staged arm to its registered bytes.
- Seal verification: existing ENG2 evidence.py reused unchanged; no new technique. Source spot checks and address-normalized behavioral comparison are ordinary review methods; no specific additional prior art known to me.

## Deliverable integrity

`findings.json` SHA-256: `d044315222ae7b4259bfe3320be9176e98c8a1c8c76b532872464678d5d9c1c3`. Final status: complete, one open MINOR census finding. No NIT raised; no additional review round requested. No files moved to DELETE_ME; nothing deleted. Retained scratch: `/tmp/eng2_rev3_20261114/`. Promotion and a fresh GPU G5 run under the new seal are outside this review.
