# ENG2-REV1 — RED

Review complete: **2 BLOCKER, 2 MAJOR, 1 MINOR; no NITs.** Correct the scoring blockers before relying on G5. Product code and copied tooling were not edited. CPU emulation does not establish GPU results.

Baseline bc02d3a1; candidate 2237299b; review HEAD c309e53ce2747f6acc9d40223cd53e0c3b9835e7. Source and arm identities verified in static_tool_review.json.

## Findings

### E2R-F01 — MAJOR: Non-string dtype objects equal to supported names bypass refusal

Location: `tensor_cuda/tensor_cuda/__init__.py:52`.

Reproduction: probe_edges.py accepts collections.UserString("float32") in all five entries. nonstring_random_review.py reaches data conversion in tensor and advances the real NumPy RNG in randn/rand; the C++ boundary is intercepted, so no device call occurs.

Argument: Set/dict membership tests equality and hash, not isinstance(dtype, str). The _np_dtype docstring and review require every non-string to be refused before conversion/allocation/random draws.

Suggested fix: Validate isinstance(dtype, str) before the _DEVICE_CAST_DTYPE membership test in both tensor and make, with the registered TypeError; add non-string tests including UserString and unhashable objects.

Receipts: `edge_probes.log`, `nonstring.json`, `nonstring_random_review.log`, `nonstring_random_review.json`.

### E2R-F02 — MAJOR: L38 accepts an unrelated failure containing the registered TypeError text

Location: `artifacts/eng2/tool/evidence.py:338`.

Reproduction: python3 artifacts/eng2/rev1/probe_edges.py: wrapped_other_reason and wrong_type_exact_text return reasons=[] for a pinned ID with exception_class=AssertionError. A plain unrelated AssertionError is correctly refused.

Argument: The pin branch checks a substring in the complete pytest longrepr, without checking the terminal exception class/message. Quoted text in another failure is sufficient.

Suggested fix: Require the call-phase terminal exception to be TypeError with the registered exact message, using structured exception data; reject other exception classes and extra failures. Add the constructed negative cases.

Receipts: `edge_probes.log`, `pin_cases.json`.

### E2R-F03 — BLOCKER: All registered mutant killer IDs fail the matcher

Location: `artifacts/eng2/tool/evidence.py:229`.

Reproduction: python3 artifacts/eng2/rev1/probe_edges.py: all 15 mutants have self_matching=0 across all registered killers.

Argument: matches() strips tensor_cuda/tests/test_eng2.py:: from the observed ID but compares it to registry patterns retaining that prefix. evaluate() therefore reports no registered killer ID failed even when a real registered killer fails and executes the mutation.

Suggested fix: Normalize both observed IDs and registered patterns consistently, preserving literal parameter brackets; verify every registered killer self-matches and exercise a real CPU MF1/MF2 receipt through evaluate().

Receipts: `edge_probes.log`, `killer_matching.json`.

### E2R-F04 — BLOCKER: Cross-arm byte scoring compares unnormalized arm-specific test IDs

Location: `artifacts/eng2/tool/score_gpu.py:46`.

Reproduction: python3 artifacts/eng2/rev1/cross_arm_review.py: collect 262 real IDs per arm, call the actual _record with identical constructed hashes, then execute the unchanged scoring loop. Both f1bytes (35/35 rows) and p302bits (20/20 rows) report IDs equal False and every baseline row differing.

Argument: test_eng2.py:57 stores request.node.nodeid, including artifacts/eng2/tool/arms/baseline or candidate. score_gpu.py:46-47 keys by that raw string, so no corresponding row can match.

Suggested fix: Normalize row test_id with evidence.norm_id on both sides before comparing, reject duplicate normalized IDs, and verify the expected 35/20 row census. Keep hashes unchanged.

Receipts: `cross_arm_review.log`, `cross_arm_review.json`, `baseline_collection.log`, `candidate_collection.log`.

### E2R-F05 — MINOR: The all-entry dtype census omits Python entries and a residual private fallback

Location: `artifacts/eng2/dtype_census.py:45`.

Reproduction: python3 artifacts/eng2/rev1/census_review.py: functional._causal_mask(0,0,"cpu",dtype) returns float32 for float64, double, int32, not_a_dtype and None. Three quantization.affine scale_dtype/out_dtype entries are also absent from L60 JSON; they use NumPy conversion semantics.

Argument: The existing 201 rows reproduce, but are not an exhaustive census of tensor_cuda/tensor_cuda. The ledger mentions the private mask assumption without emitting its required per-input rows. Its callers currently pass Tensor.dtype, so unsupported-name reachability is limited to direct private calls.

Suggested fix: Complete the census with the omitted Python entries and their baseline outcomes. Guard the private mask against unsupported/non-string dtype arguments before constructing a tensor, or obtain an explicit scope exception for that internal-only contract; preserve valid NumPy dtype support in affine helpers.

Receipts: `census_review_r1.log`, `census_review.json`.

## Answers

### 1. F1 refusal

Unsupported ordinary strings are refused before conversion/allocation/draws; the claim for every non-string is false (E2R-F01). The six supported dtypes and float64-data default preserve their source paths. Ran cpu_review.py (candidate 121 passed, baseline 60 failed/61 passed) and census_review.py (201 rows per arm reproduced, 35 supported/default host-path pairs, 17 nonconstant byte comparisons). Spot-checked tensor, randn, nn.parameter, empty and Tensor.astype. A private silent fallback remains in functional._causal_mask; the census is incomplete (E2R-F05). Device bytes remain for G5.

### 2. N1 forwards

Checked: half/bfloat16 statistics use float32, half affine parameters upcast, and output casts back once. final_checks.py traced 20 input/parameter-dtype combinations. norm_review.py executed the actual forward source: six float32 class/mode operation-sequence diffs are empty, including BatchNorm train/eval. Its float32 running update remains (1-m)*running + m*batch. Ten emulated input/parameter-gradient rows retain their leaf dtype, consistent with ops.cpp:1072-1078 casting each VJP to src. No GPU gradient claim.

### 3. Tests

Checked: E3 is imported unchanged; byte comparison to bc02d3a1 passed. Both added groups are sound for their registered cases. completion_checks.py independently bounded all six running-statistics cases within the registered tolerances (maximum bound/tolerance: mean 0.861680, variance 0.678618). Actual-source emulation reproduces 46 baseline N1 failures and zero candidate failures; union with the 60 real CPU F1 failures equals all 106 registered IDs exactly. cross_arm_review.py collected 262 IDs per arm.

### 4. Mutants

Checked: static_tool_review.py validated all 15 as exactly one registered edit, with matching hunk lines and otherwise identical arm files; final_checks.py matched every killer set to its prediction. Four traced spot checks: MF1 has 12 real CPU failures/12 hunk hits; MF2 has 48/48; MN2_RMS executes line 305 and differs in 15/64 forward elements; MN3_BN executes line 533 and differs in 28/64. The latter two execute actual source/test code with CPU arithmetic emulation. GPU kills remain for G5; the matcher currently blocks scoring (E2R-F03).

### 5. Tool changes

Parent-after-child receipts, device-mode seal verification and dependency checks are preserved: read run_gpu_line.py:34-44,68-107; verified the live seal, all 17 arms, all 12 COPY_DEPS records and the existing parent timestamps; recomputed unsatisfied device-mode prerequisites. Exactly ten IDs are pinned. edge_probes.py constructs an unrelated AssertionError that is refused, but another AssertionError containing the expected text is accepted (E2R-F02). All 15 killer sets fail matching (E2R-F03). Real collection/_record plus the exact scorer loop rejects identical constructed hashes for 35 F1 and 20 float32 rows (E2R-F04).

### 6. G3 SASS

Confirmed by sass_review.py using CUDA 12.6 cuobjdump -sass on both pinned modules: 1,005,817 lines per module, 28 module-ID substitutions each, every other byte unchanged, identical normalized SHA-256 7b2980117ad34db3cac8145412f8ea08b78aa888f8eecc0921b5cb8da1163fd8. This confirms the device-code comparison, not binary identity or GPU execution.

## Commands and outputs

Every run used `TMPDIR=/tmp/eng2_rev1_bootstrap CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1`, `nice -n 19` and foreground `timeout` under ten minutes. CPU tests disable pytest cache; scratch is under the authorized /tmp prefix. Full command text, exit status and elapsed time are in `<receipt>.command.json`; complete stdout/stderr are in `<receipt>.log`. `run_review.py` writes its receipt after its child completes.

| Receipt | Command/check | Output |
|---|---|---|
| final_versions | python3/NumPy/pytest/cuobjdump/git versions; read-only git diff of read-only targets | Python 3.12.3, NumPy 2.4.2, pytest 9.0.2, cuobjdump 12.6.77, git 2.43.0; diff empty |
| edge_probes | python3 artifacts/eng2/rev1/probe_edges.py | F01/F02/F03 reproduced; see deviation for the original nonempty spy |
| nonstring_random_review | python3 artifacts/eng2/rev1/nonstring_random_review.py | tensor converts UserString operand; randn/rand draw; C++ boundary suppressed |
| cpu_review | python3 artifacts/eng2/rev1/cpu_review.py | candidate 121 pass; baseline 60 fail/61 pass; MF1 12 and MF2 48 failures, all traced |
| norm_review_r1 | python3 artifacts/eng2/rev1/norm_review.py | 106 N1 IDs per arm emulated: baseline 46 fail, candidate 0; empty op diff; two norm mutants killed; 10 gradient dtype rows |
| census_review_r1 | python3 artifacts/eng2/rev1/census_review.py | 201 rows/arm match after object-address normalization; 35 source pairs; omitted entries reproduced |
| static_tool_review_r1 | python3 artifacts/eng2/rev1/static_tool_review.py | live seal + 17 arms; 12 dependency records; 15 single edits; source pins/E3; prerequisites refuse |
| cross_arm_review | python3 artifacts/eng2/rev1/cross_arm_review.py | actual 262-ID collections; exact scoring loop rejects equal constructed hashes, 35 and 20 rows |
| completion_checks | python3 artifacts/eng2/rev1/completion_checks.py | full dtype-argument AST inventory; six independently bounded BN cases; 106 failure union; C++ non-string refusals; parent timestamps |
| final_checks | python3 artifacts/eng2/rev1/final_checks.py | all killer sets match predictions; 20 half source traces; omitted-entry sources match baseline; lookup rejects |
| sass_review | python3 artifacts/eng2/rev1/sass_review.py | exact normalized SASS equality; pinned raw module and SASS hashes reproduced |

Read receipts (`source_reads`, `tool_reads`, `focused_reads`, `evidence_reads`, `tape_reads`, `norm_setup_reads`, `collection_reads`) retain the file/line reads used for reasoning. Their complete disk logs remain available even where terminal output was truncated. No claim relies on truncated terminal text.

## Complete dtype-entry classification

Four string columns are `float64`, `double`, `int32`, `not_a_dtype`; non-string examples are named explicitly. The same baseline behavior applies to additional entries because final_checks.py compared their full source bytes to bc02d3a1. Zero-byte engine operands only were used for census runs. No array equality is inferred from empty or constant tensors.

| Entry and source | Baseline four strings | Candidate four strings | Non-string behavior |
|---|---|---|---|
| tensor (__init__.py:58); zeros/ones/randn/rand via make (:69) | all silently float32 | all TypeError | np.float32/np.dtype: baseline float32, candidate TypeError; UserString(float32): accepted (F01) |
| nn.parameter (nn.py:17) | all silently float32 | all TypeError | forwards to tensor; same behavior |
| empty (__init__.py:171) | all RuntimeError unknown dtype | unchanged | np.float32/np.dtype rejected by pybind TypeError |
| Tensor.astype (bindings.cpp:1893) | all RuntimeError unknown dtype | unchanged | np.float32/np.dtype rejected by pybind TypeError |
| int4_dequant (:1127), intn_dequant (:1144), kv_int4_unpack (:1179), out_dtype | all RuntimeError unknown dtype | unchanged | None rejected by pybind TypeError, before kernels |
| functional._causal_mask (functional.py:35,68-73) | all silently float32 | unchanged | None returns float32; private callers use Tensor.dtype |
| GapMargins.lookup (apa_sp2.py:53-57) | all ValueError (uncertified dtype) | unchanged | None ValueError; this is a certificate lookup, not a tensor constructor |
| quantize_affine_per_group scale_dtype (affine.py:142,170-171) | float64, float64, int32, TypeError | unchanged | np.float32 handled; None maps to NumPy float64 |
| dequantize_affine_per_group out_dtype (affine.py:178,198) | float64, float64, int32, TypeError | unchanged | np.float32 handled; None maps to NumPy float64 |
| dequantize_symmetric_per_group out_dtype (affine.py:201,218) | float64, float64, int32, TypeError | unchanged | np.float32 handled; None maps to NumPy float64 |
| _np_dtype (__init__.py:43) | absent; inline get fallback | internal strict map, ordinary unknown string TypeError | hash/equality-alias non-string can pass (F01) |

`Tensor.to` is a device-only binding, not a dtype-string entry. C++ `dtype_from_string` (kernels.cu:50-58) explicitly refuses unknown names and keeps its registered aliases. NumPy affine conversion is an actual host dtype conversion, not an engine float32 fallback. General C++ defects remain outside this review.

## Norm and tolerance reasoning

The actual forward function ASTs were extracted from the baseline commit, candidate source and pinned mutants; comments were not used as evidence of op order. The CPU adapter executes these functions with ENG1's E3 primitive arithmetic, including its reduction order and VJPs. `float32_op_sequences.json` contains both traces and `float32_op_sequences.diff` is empty. Arithmetic output comparisons occur in the same process with one thread. Nonzero/nonconstant checks precede finite comparisons; an entirely nonfinite baseline gradient is classified as a failure, not a numerical pass.

In the half paths, all mean/variance calls are float32, affine parameters are converted when needed, and one output cast targets the original dtype. `final_checks.json` records 20 class/input/parameter combinations. `ops.cpp:1072-1078` returns the input on a same-dtype cast and casts the backward contribution to the saved source dtype on a genuine cast. Consequently input and half parameter gradients return to the corresponding leaf dtype. The CPU tape confirms this for ten class/dtype combinations; device arithmetic remains untested.

BatchNorm uses float32 running tensors initialized at nn.py:513-514; updates at :535-536 remain `(1-m)*running + m*batch` with biased batch variance, as in the baseline. Half batch statistics now arrive in float32; this intentional change avoids NumPy float16 momentum multiplication. The float32 train/eval op traces are unchanged.

The half-rule test uses the same engine operations on separate leaves, with matching dtype/shape and nonconstant reference checks. It tests the specified composition rather than imposing a looser E3 numerical threshold. Parameters in the inherited helper are constants, so its input-gradient comparison is not perturbed by cross-graph parameter accumulation.

Running-statistics tolerances are sufficient for the six registered first-update fixtures: N=16, m=0.1, initial mean=0 and variance=1. The comment's n*u expression is first-order shorthand, not a universal repeated-update/cancellation theorem. An independent finite-case bound uses gamma_15 for the serial sum, exact /16 scaling on these normal operands, `alpha=(1+u)^3*(1+gamma_15)-1`, and `d_var <= alpha*var + (1+alpha)*d_mean^2`. The actual float32 representations of 0.1 and 0.9 bound the update. All six conservative bounds are below the registered tolerances: mean ratio <=0.861679078; variance ratio <=0.678617611. No tolerance was tuned or changed.

## Mutation reasoning

All 15 arms match a single scoped old-to-new substitution exactly, compile as Python, and have no other changed arm file. MN1 omits the output cast (four classes; dtype assertions kill it). MN2 omits bfloat16 upcasting (four classes; registered bfloat16 composition checks kill it). MN3 leaves the selected class's statistics in the input dtype (four classes; registered composition/numerical/statistics checks distinguish it). MN4_LN adds a float16 round trip on the float32 path (float32 bit checks distinguish it). MF1/MF2 restore their respective silent fallback. These are valid non-equivalent mutants; a same-dtype no-op cast was not used as MN4.

Four L41 spot checks have executed-hunk receipts: `MF1_cpu.json`, `MF2_cpu.json`, `norm_mutant_traces.json`. Every registered killer agrees with the author's predicted failure set; baseline/candidate source emulation and the two additional norm traces support the predictions. No device mutant kill is claimed. F03 prevents their existing scorer from recognizing them until fixed.

## SASS receipt

Base module SHA-256: `477e81cb21eb13eb2e7a639515edf34a61face790dd5106d62da2b8f01113a54`.
Candidate module SHA-256: `f72af46b50923800875f2f1237fe279efb04c72404d844569a684bf4167609a7`.
Only `_[0-9a-f]{8}_[0-9]{7}` was replaced by `_MODID`; no path, whitespace or instruction normalization. The raw and normalized hashes reproduce G3_sass_host.json exactly. Raw and normalized texts remain under `/tmp/eng2_rev1_bootstrap/{base,cand}{,.normalized}.sass`; compact receipt: `sass_comparison.json`.

## Deviations and residuals

- NumPy has no `python3 -m numpy --version` entry point; version read from `numpy.__version__` instead.
- One read command included an unmatched `build/*.py` glob; relevant source files were subsequently read directly. The read command's final exit masked that intermediate rg error; no evidence claim depends on the glob.
- Original F01 factory spy returned a nonempty array with `device=cpu`; the engine's from_host path attempted CUDA and refused with `CUDA error at from_host: no CUDA-capable device is detected`. CUDA was hidden and no GPU work executed. The spy was repaired to use a zero-byte result. The independent random-order probe intercepts the C++ boundary entirely. This is a reviewer-tool slip, not a G5 result.
- norm_review.py repair 1: its liveness guard stopped on an entirely nonfinite baseline gradient. The rerun classifies that as a failure and retains nonzero/nonconstant checks for finite comparisons.
- static_tool_review.py repair 1: handled the explicitly binned e6_record.py dependency as registered absence rather than attempting to hash it.
- census_review.py repair 1: normalized only Python object addresses in two pybind TypeError messages when comparing old/new census receipts.
- No script used more than one repair. Original failed logs remain on disk. No registered gate was run and no tolerance/acceptance rule was altered.
- Scratch retained: `/tmp/eng2_rev1_bootstrap` (SASS dumps and pytest scratch). No files deleted, moved to DELETE_ME, or written outside the authorized review/scratch directories. No git writes or subagents.
- GPU execution, GPU bytes/gradients, device mutant kills, the recorded C++ defects, host machine-code equivalence and promotion are not established here. Findings are not claimed fixed; fixes are outside this seat's writable target. No NITs require an extra round.

## Prior art

Differential testing: McKeeman (1998); mutation testing: DeMillo, Lipton and Sayward (1978); reverse-mode tape: Wengert (1964), through ENG1 E3; rounding bounds: Higham (2002); mixed precision: Micikevicius et al. (2018); SASS normalization: reproducible-builds/diffoscope practice (year unverified). External references are unverified — lead to check these names/titles. Existing methods were reused for ENG2-specific probes; no novel algorithm is claimed. Proposed fixes reuse Python type validation and the existing evidence.norm_id; no specific additional prior art known to me for these local corrections.

## Findings digest

`findings.json` SHA-256: `1938c6e080a4d729c1a214ba0ebf041eb282f1b18307c4c0145fc0643db71a67`.
