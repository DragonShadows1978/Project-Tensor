# ENG2-REV2 — RED: two MINOR residual gaps

No BLOCKER, MAJOR or NIT. The original membership bypass and three tooling defects are fixed; two literal completeness requirements remain open. No newly introduced product regression was detected in this narrow CPU/source review. The factory ordering and missing census entry predate A1. Not claimed fixed: the full pre-side-effect refusal guarantee and all-entry census.

Baseline `bc02d3a1`; reviewed A1 `1a4beed1`; review head `4ee41ad4b3036bbb901206b0eb4ce1b15aca5608`. N1 semantics, C++ defects, GPU execution and promotion are excluded. Findings are not product edits: everything outside rev2 is read-only.

## Findings

### E2R2-F01 — MINOR: Factory shape conversion still precedes the non-string dtype guard

File: `tensor_cuda/tensor_cuda/__init__.py:70`.

Reproduction: python3 -B artifacts/eng2/rev2/dtype_boundaries.py: for zeros, ones, randn and rand, pass Shape([2,3]) with dtype=None, where Shape is a list subclass whose __iter__ records a call and draws once from NumPy. All four execute shape.__iter__ and advance RNG before raising the registered TypeError; zero device calls occur.

Argument: Lines 70-71 normalize a tuple/list shape before line 72 validates dtype. Even a plain list allocates its tuple before refusal. Ordinary integer-shape calls are fixed, but the stated before-any-conversion/allocation/random-draw guarantee is incomplete. The shape conversion is pre-existing, not a newly introduced regression; MINOR because this residual requires the shape-normalization path, and the observed RNG draw is in the supplied iterator.

Suggested fix: Move the non-string dtype check ahead of shape normalization in make, retaining the registered TypeError and existing supported-dtype behavior; add a shape-iterator sentinel check.

Receipts: `dtype_boundaries.py`, `dtype_boundaries.json`, `dtype_boundaries.log`.

### E2R2-F02 — MINOR: The all-entry census still omits RegisteredMargins.lookup

File: `artifacts/eng2/dtype_census.py:43`.

Reproduction: python3 -B artifacts/eng2/rev2/dtype_boundaries.py and dtype_signature_inventory.json: tensor_cuda/tensor_cuda/apa_sp2.py:53 defines RegisteredMargins.lookup(..., dtype, ...), but neither 337-row census contains it. The exact baseline and candidate method bodies both raise ValueError("apa_sp2: table covers fp32 at scale=1/sqrt(D) only") for float64, double, int32 and not_a_dtype before reading the table.

Argument: A1 adds all four omissions named by REV1, but the order requires every dtype-string entry under tensor_cuda/tensor_cuda. This additional public metadata-validation entry accepts a dtype string and is absent. It is a census completeness gap, not a tensor fallback or a newly introduced product defect. The refusal probe uses a table-read sentinel; it does not validate calibration data or attention execution.

Suggested fix: Add RegisteredMargins.lookup to the census with its baseline and candidate refusal rows, identifying its metadata-validation role and float32-only contract; leave its product behavior unchanged.

Receipts: `dtype_signature_inventory.json`, `dtype_boundaries.py`, `dtype_boundaries.json`, `census_baseline.json`, `census_candidate.json`.

## Questions and answers

1. PARTIAL. The original dtype-membership bypass is fixed: probe_edges.py and dtype_boundaries.py refused all 36 ordinary-shape non-string cases before dtype hashing/equality, data conversion, allocation, RNG draws or mask cache lookup. Factory shape normalization still precedes refusal (E2R2-F01). census_review.py checked 35 supported/default source pairs; dtype_boundaries.py checked 90 plain-str/NumPy-str/plain-subclass pairs and six nonconstant mask pairs. All six supported names retain their host paths and casts. name=True is safe at its three internal call sites: it checks string type, then each caller validates membership before data work. It is not a standalone supported-name validator. str subclasses, including numpy.str_, are acceptable under the explicitly requested isinstance(dtype, str) contract; arbitrary overridden subclass methods are not a promise of builtin-str behavior.

Receipts: `edge_probes.log`, `dtype_boundaries.json`, `census_review.json`.

2. PASS. Re-ran copied probe_edges.py, which is the command recorded by REV1 edge_probes.command.json; no separate edge_probes.py exists. Every negative was refused, and the old unstructured positive was correctly refused for missing JUnit data. Replayed A1 P0 through copied tool_checks.py: all 10 real candidate pin failures were accepted. P1-P7 were refused. The trusted receipt must contain exactly one failed call, exception_class == TypeError, and the exact registered JUnit message; another class, altered message, or extra failure cannot pass these checks. This claim assumes the sealed producer and authentic receipts, not forged class names or forged XML.

Receipts: `pin_cases.json`, `TOOL_CHECKS.json`, `tool_checks.log`.

3. PASS. probe_edges.py and tool_checks.py confirmed all 15 killer sets self-match and match exactly their registered IDs, with no outsiders over all 298 census IDs in normalized and arm-prefixed forms. Fresh cpu_runs.py runs produced MF1 12/12 and MF2 48/48 registered failures; all 60 killers traced lines 63/75 respectively. mutant_eval_cpu.py replayed A1 receipts through unchanged evaluate() with the declared 157-ID CPU-census substitution: both killed; S1 and S2 refused with "no registered killer ID failed".

Receipts: `killer_matching.json`, `TOOL_CHECKS.json`, `arms_review.json`, `MUTANT_EVAL_CPU.json`.

4. PASS. Copied cross_arm_review.py freshly collected 298 IDs per arm and emitted 35 f1bytes and 20 p302bits rows through the real _record function. The exact scorer loop reported IDs equal True, zero differing hashes and no census errors. tool_checks.py rejected a changed hash, duplicate and missing row, plus extra-ID and wrong-kind controls. final_bindings.py confirmed normalized ID/hash maps equal REV1 and A1. These are freshly reproduced constructed hash payloads, not GPU-output byte-equality evidence.

Receipts: `cross_arm_review.json`, `TOOL_CHECKS.json`, `final_bindings.json`.

5. PARTIAL. census_review.py reproduced all 337 rows per arm, including the four entries named by REV1; the original 201 rows remain a prefix. The all-entry claim is still incomplete because RegisteredMargins.lookup is absent (E2R2-F02). dtype_boundaries.py verified all six _causal_mask names against baseline on a nonconstant 2x3 mask: float16/bfloat16 request casts; float32/int64/bool/uint8 retain the existing float32 bias. Unsupported names and non-strings refuse before cache lookup. The three affine helpers retain NumPy dtype semantics and their baseline outcomes.

Receipts: `census_review.json`, `dtype_signature_inventory.json`, `dtype_boundaries.json`, `supplemental_checks.json`.

6. PASS within this CPU/source scope, subject to the two residual gaps above. scope_and_arms.py compared git diff 9452ef61 1a4beed1 with AG3_SCOPE.json: all 63 modified paths match; 217 additions are authorized receipts plus the dispatch order. Product hunks are confined to the named functions, test_eng2.py is additions-only, and N1 bytes are unchanged. Copied cpu_runs.py passed 157/157 candidate CPU tests with CUDA hidden. Re-executing the exact make_arm recipe in scratch for baseline, candidate, MF1 and MF2 matched all 11 files per arm, including module hashes and mutant edits; A1 deviation 2 was faithful. No GPU acceptance or promotion is claimed.

Receipts: `scope_review.json`, `scope_full.diff`, `arms_review.json`, `candidate_pytest.log`.

## Commands and outputs

Every run used `TMPDIR=/tmp/eng2_rev2_review`, `CUDA_VISIBLE_DEVICES=''`, `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=NUMEXPR_NUM_THREADS=1`, `PYTHONDONTWRITEBYTECODE=1`, `timeout 590s nice -n 19`; each script ran in the foreground. No pytest cache was enabled. No git writes, subagents, deletes or GPU work occurred. The stdout/stderr and command/environment/time/exit JSON are paired by run name. `run_check.py` wraps each command without changing its outcome.

| Run / receipt prefix | Command after `python3 artifacts/eng2/rev2/run_check.py <prefix>` | Output |
|---|---|---|
| edge_probes | `python3 -B artifacts/eng2/rev2/probe_edges.py` | UserString refused; four unstructured pin scenarios refused; 15 self-matching sets |
| cross_arm | `python3 -B artifacts/eng2/rev2/cross_arm_review.py` | 298 collected each; 35/20 matched; errors=[] |
| tool_checks | `python3 -B artifacts/eng2/rev2/tool_checks.py` | P0-P7, M1-M3, C0-C5 all ok |
| census_review | `python3 -B artifacts/eng2/rev2/census_review.py` | 337 rows per arm; 35 supported/default pairs, 17 nontrivial byte comparisons |
| cpu_suite | `python3 -B artifacts/eng2/rev2/cpu_runs.py candidate MF1 MF2` | 157 passed; MF1 12 failures; MF2 48 failures |
| mutant_evaluation | `python3 -B artifacts/eng2/rev2/mutant_eval_cpu.py` | 12/12 and 48/48 killed/traced; two survivors refused |
| scope_and_arms | `python3 -B artifacts/eng2/rev2/scope_and_arms.py` | Initial harness AssertionError, retained; D02 |
| scope_and_arms_r1 | same command after repair | 63 modifications match; 4 arms x 11 files identical |
| dtype_boundaries | `python3 -B artifacts/eng2/rev2/dtype_boundaries.py` | 36 refusal cases; 90 supported-string pairs; six mask pairs; both findings reproduced |
| final_bindings | `python3 -B artifacts/eng2/rev2/final_bindings.py` | Fresh row maps equal REV1/A1; 20 input SHA-256 bindings |

The CPU wrapper's child interpreter returns 0 after writing its receipt even when pytest returns 1; the mutant verdicts use the recorded `pytest_rc` and per-ID failures, never that wrapper exit. Full pytest output is retained in candidate_pytest.log, MF1_pytest.log and MF2_pytest.log. Mutation trace maps in arms_review.json bind each counted killer to the hunk line. No N1 test algorithm was re-reviewed.

Copies are review-owned and each adaptation is in a `.py.diff` receipt. Output roots were rebound; old 262 collection expectations were updated to the registered 298; fixed scorer globals were loaded; census comparison was rebound to A1/1a4beed1. A1 P0 and staged mutant receipts were re-scored read-only; fresh candidate/MF1/MF2 CPU executions are separate receipts. Baseline-passing pin outcomes in the P0 checker remain A1's declared constructed baseline side; the candidate failures are real sealed-child receipts.

Census inputs: 16 recorded entry labels, 337 rows per arm, original 201-row prefix preserved after object-address normalization. final_bindings.json records all 152 changed baseline/candidate rows with both outcomes. Metadata lookup refusal is a source-extracted control-flow probe with a table-read sentinel, not a calibration-data gate. Arrays compared for host-byte checks are explicitly nonzero and nonconstant; constant factory cases are checked by unchanged source/cast behavior, with no byte-proof claim. supplemental_checks.json verifies the seeded and one-draw RNG states are also nonzero/nonconstant.

## Deviations and limits

- D01: One functions.wait used 1000 ms while a 300000-ms process wait was pending; the following wait used 300000. No status message or extra process poll intervened; results unaffected.
- D02: scope_and_arms.py repair 1 of 3: positional AST keys mislabeled an unchanged main guard after insertion. Replaced them with AST-content keys and re-ran successfully. Initial failure retained in scope_and_arms.log; success in scope_and_arms_r1.log.

There is no `edge_probes.py` in REV1; edge_probes.command.json identifies probe_edges.py as its producer. This naming clarification did not omit a scenario. NumPy has no module --version CLI: its version was read from __version__; Python 3.12.3, NumPy 2.4.2, pytest 9.0.2, git 2.43.0.

Scratch retained, with no deletion or DELETE_ME moves: `/tmp/eng2_rev2_review/`, including `recipe_rebuild/` and pin variants in `tool_checks/`. Review deliverables remain under `artifacts/eng2/rev2/`. The command log and complete diff are receipts; no approval or promotion action was taken.

## Prior art

Reused REV1/A1 probes; differential testing (McKeeman, 1998), mutation testing (DeMillo, Lipton and Sayward, 1978), pytest/JUnit and CPython tracing, Python/POSIX fnmatch, SHA-256 (NIST FIPS 180-4). Literature unverified — lead to check. New contribution: scoped boundary cases, exhaustive ID comparisons and receipt binding; no novel algorithm. For validation-before-side-effects and complete signature census, no specific prior art known to me.

Search leads: "McKeeman 1998 Differential Testing for Software"; "DeMillo Lipton Sayward 1978 Hints on Test Data Selection"; "pytest junitxml reprcrash message"; "Python fnmatchcase literal brackets"; "NIST FIPS 180-4". Prior-art notes appear in review script comments and LEDGER.md. The proposed product/census fixes were not implemented because those paths are read-only.

## Integrity

`findings.json` SHA-256: `453010211095d1b856321f40b34c8b6b2cf6e77a7f3c824b4d8176dcaa8a772d`.
