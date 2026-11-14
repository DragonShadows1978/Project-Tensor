# Prior art: order's evidence-led review format; no prior art known to me for this prose layout.
# No new algorithm; assemble already-recorded facts and bounded reasoning.
import json
from pathlib import Path
rev=Path(__file__).resolve().parent
f=json.loads((rev/'findings.json').read_text());N=json.loads((rev/'numerics.json').read_text())
out="""# ENG1-REV1 — REJECT

Two MAJOR findings, two MINOR coverage findings and one NIT. The E1/E2/E4/E5/E6 source treatments are correct in their stated domain by source reasoning and the CPU checks below. The measurement/harness package needs correction before it establishes the proposed G5 claims. No product, test, frozen receipt or build was edited. GPU execution, performance and promotion remain outside this review.

## Findings

"""
for r in f['findings']:
 out+=f"### {r['id']} — {r['severity']} — {r['title']}\n\n"
 out+=f"Location: {r['file']}:{r['line']}.\n\nReproduction: {r['reproduction']}\n\nReasoning: {r['argument']}\n\nSuggested fix: {r['suggested_fix']}\n\nReceipts: "+', '.join('['+p+']('+p+')' for p in r['receipts'])+'.\n\n'
out+="""## 1. Correctness of each fix

**E1 checked.** For a valid contiguous NDArray, the loop rejects every negative extent except a single -1. With one -1 it requires positive known product and exact divisibility, then inserts numel/known. Without -1 it retains the old element-count check. A zero-element source and positive known product correctly infer 0; zero known product is ambiguous and refused. These are PyTorch's documented inference rules. Storage ownership is still copied and only metadata is replaced. Binding bindings.cpp:1879 forwards the int64 vector unchanged.

semantics.py compared all 120 constructed empty targets (negative/zero/positive extents, ranks 1–3) with CPU PyTorch 2.11.0+cu130: no shape/acceptance differences. Four huge-zero targets also agree on acceptance/refusal; overflow exception classes/messages differ, which the order does not require to match. Nonempty integer rules follow from source; GPU storage-write verification remains G5. General PyTorch noncontiguous view constraints do not apply to this engine's contiguous NDArray representation.

The reported 30 caller count is wrong (R1-F05). [CALLER_CENSUS.md](CALLER_CENSUS.md) checks **37 engine calls on 32 text lines**, plus nine NumPy calls, by expression and element-count identity. quant.py's rotation and blocked-row-sum reshapes are included. Every previously valid nonnegative explicit shape takes the same final comparison and preserves result/storage. [CORE_CALLERS.md](CORE_CALLERS.md) also itemizes every C++ metadata reshape.

**E2 checked.** Half inputs and upstream g convert exactly to fp32, and ew_pow/nmuls/nmul compute g32*(p*x32^(p-1)), casting the completed product back once. All registered p and p-1 are exactly fp32. Float32 executes the old expression verbatim at ops.cpp:872; its primitive kernels and accumulation are unchanged. Identity to the baseline therefore follows by source construction. Independent uint32 round-order witness: **0x407b3880 versus 0x407b387f**. The frozen bit tests recompose engine primitives, rather than read baseline output; this oracle is not independent, but the unchanged-line and scope argument supplies the baseline identity proof. Device bit comparisons remain G5.

**E4 checked.** RMS constructs x.shape/out_dtype; its Tensor wrapper supplies x.dtype. Causal and APA softmax construct the score shape/dtype. Each zero guard precedes division and returns without launch. APA's earlier rank-one Lchunk division already uses (S>0?S:1). Topk checks k<0 or k>S at kernels.cu:5380 before output-shape construction, allocation or launch; outputs have input-prefix+[k], input value dtype, and int64 indices. Candidate/C_COPY/C_UBSAN pass all CPU empty/refusal IDs. C_UBSAN covers division ordering for all four entries.

D1 preserves S<L refusal: (2,3,0) still refuses and S=0 is accepted only with L=0. D2 is right: sink rows=B*H*L has no host division by S, so it is unchanged; it may still launch for S=0 with positive rows. The order only required changing the sink if it shared the division.

CORE_CALLERS.md names each direct caller across package/core/scripts/external source/benchmarks/tests, and callback/interposer sites. Nonempty numerical paths are unchanged; no intentional reliance on prior UB/garbage was found. Invalid routing k now refuses as designed. The existing topk regression uses k=3,S=5. Scalar/rank-invalid entry validation is inherited and not claimed fixed. Expected empty behavior comes from the order's explicit design and PyTorch's same-shape softmax/normalization and topk size-k/int64/range contract; CPU probes corroborate, rather than define, it.

**E5 checked.** All five direct make_dims callers are itemized: im2col:158, col2im:166, avgpool2d:173, avgpool2d_bwd:180 and maxpool2d:187. All call it before allocation/launch. maxpool2d_bwd consumes the valid forward's saved argmax and does not directly call it. With positive s, floor(A/s)+1>0 iff A>=0, so every candidate-accepted geometry retains its old quotient. The **old accepted set changes**: -s<A<0 formerly passed and now refuses. H=3,k=4,s=2,p=0 changes old OH=1 to floor OH=0/refusal. A finite 15,120 one-axis census found 1,369 such exclusions and zero common-accepted output changes. Invalid strides/padding and int32 range limits are inherited and not claimed repaired. R1-F04 concerns the isolated width-zero test.

**E6 checked.** ASTs are identical after restoring the six fixture arguments and eight destination expressions. Image computations/assertions are unchanged. The real candidate E6 witness passes; nested writers report **5 passed, 1 CUDA-hidden skip**, with empty/unchanged target maps. An exact M14 arm copied under rev1 fails its registered E6 witness after the nested tests succeed: one **921,615-byte** .ppm appears. No frozen arm was allowed to write into itself.

Receipts: [semantics.json](semantics.json), [census_scope.json](census_scope.json), [cpu_gates.json](cpu_gates.json), individual cpu_*.xml/log, [e6_record.json](e6_record.json), [m14_probe/e6_record.json](m14_probe/e6_record.json).

## 2. Independent references and differentiation

Value references do not use engine y or dx. E1 reshapes an independent NumPy sequence; E2 differentiates a scalar power; E3 evaluates normalization/VJP in float64; E5 builds independent padded windows/scatters. Reading back stored operands x,w,b,g/r correctly accounts for dtype quantization and does not make the operation reference depend on its output. E2's bit-composition oracle is the explicit exception discussed above.

Independently, x^p=exp(p*log(x)) for x>0 gives **d/dx sum(w*x^p)=w*p*exp((p-1)*log(x))**. numerics.py uses this expression, rather than the registered np.power gradient, and checks all five exponents and stored dtypes with centered differences.

For RMSNorm, r=sqrt(sum(x_j^2)/n+eps), y_i=w_i*x_i/r, hence **J_ij=w_i*(delta_ij/r-x_i*x_j/(n*r^3))**. The independently formed J^T*g matches the registered VJP for regular/near_eps in both half dtypes, with a separate centered-difference coordinate check per case. No checked path crosses a kink: the power operands stay positive and the RMS denominator stays positive.

"""
out+=f"Maximum E2 relative central-difference discrepancy: {max(r['finite_difference_max_rel'] for r in N['power']):.9g}. Maximum RMS Jacobian/reference absolute discrepancy: {max(r['jacobian_max_abs'] for r in N['rms_jacobian']):.9g}. Maximum selected-coordinate relative discrepancy: {max(r['relative_error'] for r in N['rms_jacobian']):.9g}.\n\n"
out+="""Receipts: [numerics.py](numerics.py), [numerics.json](numerics.json), [numerics.stdout](numerics.stdout). The 43 CPU IDs also recheck the supplied hand references.

## 3. Tolerances and honest-error bounds

E1/E4/refusals/E6 use exact shape, value, message, exit or map equality: zero numerical slack. E5 has positive inputs/upstreams, so no sum/scatter cancellation. At most nine additions plus division are bounded by gamma_10=10*u32/(1-10*u32)=5.96046803e-7, below 2^-20=9.53674316e-7. Max-pool forward and im2col are exact on these stored operands.

Using the registered CUDA **4-ulp powf assumption**, fp32 power-chain error is at most (1+4*2^-23)*(1+u32)^2-1 = **5.96046508e-7**, below its 2^-20 tolerance. Including final dtype rounding yields **4.88877588e-4** (float16) and **3.90684837e-3** (bfloat16), below registered 4.89234924e-4 and 3.90720367e-3. Final half-subnormal spacing covers final underflow. All registered widened intermediates fit fp32. These bounds use stored operands and registered exponents/ranges; they are not general arbitrary-input bounds.

**E3's derivation is unsound (R1-F01).** n*u*max|reference| is not an absolute bound through cancellation or subnormal intermediates. A half square near 1e-6 has absolute half-spacing 2^-25, about 3% relative before inverse normalization, not 2^-11. A final single subnormal allowance cannot cover that amplified error. Mean rounding near offset 0.5 is amplified by 1/sqrt(var+eps); its leading normalized absolute scale is 0.05459 (float16) or 0.43673 (bfloat16), absent from the op-count argument.

error_bounds.py supplies conservative outward absolute intervals including fp32 reduction error, each stored intermediate, scalar rounding, powf error and cancellation. Worst bound/tolerance case: **InstanceNorm2D/bfloat16/near_eps bound 2.09914196 versus tolerance 0.0474236961** (44.26x). RMSNorm/float16/near_eps has bound **0.0636486764 versus 0.00843157815**; nearest-even CPU emulation already exhibits **0.0156582687** error. These are reasoning bounds and IEEE emulation, not measured GPU failures.

There is no finite honest error bound for every registered half norm backward graph: RMSNorm/float16/near_eps has a pow VJP about **-2.27e6 to -2.56e6**. Returning that half gradient correctly overflows before later x/16 scaling, although the ideal final input VJP is finite (max 1444.88). E2 still casts its completed result as required. This composed-norm range limitation is **not claimed fixed**; E3 intentionally measures it. A fixed accuracy target can report these errors, but must not be described as a guaranteed rounding bound.

[MUTANT_BOUNDS.md](MUTANT_BOUNDS.md) and [mutant_bounds.json](mutant_bounds.json) give **every M01–M14 bound and witness call path**. M05/M07 force infinity against finite tolerances (~0.43723 and ~0.47118 on displayed witnesses). M06 at x=7 has error at least **1528.6 times** tolerance over all ten half IDs, allowing powf and rounding uncertainty. All other mutants violate an exact behavioral comparison. None of the 14 can hide behind a loose tolerance; R1-F03 is a different bfloat16-only early-cast decision.

Receipts: [error_bounds.json](error_bounds.json), [backward_bound.json](backward_bound.json), [coverage_gaps.json](coverage_gaps.json), [numerics.json](numerics.json).

## 4. Mutants and controls

M01–M14 each consist of exactly one registered replacement at one occurrence. M01–M13 reconstructed pristine/mutated/restored source hashes and arm module pins match their build receipts. M14's Python copy is byte-identical to the registered one-line edit. M03's pre-run amendment is compile-correct and retains its decision. Static call paths prove each witness reaches the edited site (per-mutant table); actual CPU results corroborate all CPU-observable edits. M09 makes optimizer-hidden UB observable through UBSan, with a passing unmutated control.

Own frozen CPU results:
- Candidate, C_COPY, C_UBSAN: **43 passed** each.
- Baseline: **29 failed, 14 passed**, matching CPU registration.
- M01/M02/M03/M04/M08/M09/M10/M11/M12/M13: **1/4/1/1/2/1/1/8/4/4** failing IDs, all registered killers.
- M14 exact copied arm: its **one registered E6 witness fails** on the actual added .ppm.
- M05–M07 need nonempty CUDA computation; own CPU error bounds predict their registered kills, which remain G5.

Unpinned decisions: bfloat16-only early-cast rounding (R1-F03), and OH>0/OW=0 refusal (R1-F04). Allocation-before-refusal placement is source-proved, not observed with an allocation hook. Noninteger reshape/mismatch refusals have exact tests without dedicated mutants. C_UBSAN covers division ordering for all four E4 entries, though M09 mutates only APA. E3 numerical accuracy is intentionally measured rather than gated.

## 5. Staged lines and tooling

The 22 labels/suites/expected exits match test intent, subject to R1-F02:
- baseline_eng1 child 1, **54 exact failed IDs**: E1 24, E2 5, E4 4, E5 20, E6 1.
- candidate_eng1 child 0; E3 reports measurements without asserting their accuracy result.
- Four regression/terrain lines allow child 0 or 1, then require no baseline-passing ID to lose its pass; terrain additionally requires baseline writes/candidate no writes.
- C_COPY/C_UBSAN child 0.
- Each M01–M14 child 1, with registered killer and passing candidate on that ID; the final scorer also requires C_UBSAN for M09.
- Parent exit 0 means the **actual child outcome matches registration**, not that every child passed. Bounds are per-label in the frozen JSON/script; no staged GPU line was executed here.

run_gpu_line.py:65–70 waits on foreground subprocess.run and derives actual returncode before evaluating/publishing its completion receipt. evidence.py:229–249 binds exit to child pytest_exit and exact collected-ID/JUnit outcomes; :278–280 checks exit/failure consistency. recompute_line:320–342 refuses absent parent completion and validates start <= child_exited <= recorded. Missing/extra IDs, collection errors, device-pattern failures and non-regression skips cannot establish a pass. Desired labels do not synthesize child outcomes.

[TOOLING_REVIEW.md](TOOLING_REVIEW.md) reviews all **27 hunks**. Nine saved originals match GN1 A8; in-memory application of every hunk reconstructs adapted live files exactly. GN1-specific program/runtime-contract conjunctions were removed, so this is not full GN1 contract-C equivalence. The requested completed-child exit/JUnit property remains. R1-F02 weakens what regression evidence exercises, rather than fabricating its exit/JUnit results.

The frozen scorer was run on original **tool/audit_final as it stands**. Only its attempted summary write was redirected to rev1/audit_rescore.json, preserving every evaluation/input byte and path. It returned **1**: candidate_eng1 passed as **CPU**, other lines were absent, and audit/nondevice data were explicitly refused as G5 acceptance.

Receipts: [scorer.exit.json](scorer.exit.json), [scorer.stdout](scorer.stdout), [audit_rescore.json](audit_rescore.json), [census_scope.json](census_scope.json).

## 6. Arms

One CPU process per arm performs staged preloading and collects all **2,133 exact regression IDs** from 49 real files. All loaded tensor_cuda submodules/extensions remain under their own arm before/after collection. Baseline module: **73ededa0e73357c8578e58e5eadd87df9f2f3b2d64d631eba32499f2fdb9523e**. Candidate: **3e6d5805fb8b89ebfeb2b32c173fd097ef1cfe4a01b3349dc177d4b0c8c0ad53**. Both match independent pins. Seal checking rejects arm symlinks.

This proves in-process collection isolation. Source-reading tests inspect candidate repo files in both arms. A fresh child loses sys.path/sys.modules preloading: the independent probe exits **1, ModuleNotFoundError: No module named 'tensor_cuda'**. A different inherited environment could import an installed candidate/production copy. Thus D4's unconditional candidate-in-both-children description is incorrect (R1-F02).

Receipts: [arm_baseline.json](arm_baseline.json), [arm_candidate.json](arm_candidate.json), [d4_child.json](d4_child.json), associated stdout/exit receipts.

## 7. Scope, missing requirements and premise corrections

Across **10,573 tracked hashes**, exactly four allowed source/test paths differ, and live initial bytes match the frozen after map. source.diff equals read-only git diff from 1680764d. Every hunk is within seven named C++ functions or E6 fixture/write lines. The new test file is authorized; nn.py is unchanged.

- **F1 correct**, re-derived from baseline source: no topk bound check, sentinel value/index -1 for iterations beyond S. S=0 also has host division UB, so the reported garbage bytes are not defined behavior.
- **F2 correct**: dtype enum lacks float64 and factory fallback drops the request (Q8).
- **F3 correct with scope boundary**: floor removes previously admitted negative-gap geometries. Candidate-accepted outputs are unchanged; the old accepted set is not unchanged.
- **F4 correct**: tracked identical-byte rewrites evade git status. Stat/mtime detection matters. Own copied M14 emits the same .ppm hash 618e9a71... as the frozen receipt.
- **F5 correct**: six fixture signature changes are necessary; AST comparison proves no image/assertion change.
- **F6 correct for supplied artifacts**: originals equal A8 and program_binding.py is absent. Completed-child property survives; full old contract conjunctions do not.
- **D1/D2 correct**: preserve causal refusal; sink has no host S division.
- **D3 correct**: only E6 is deselected for C++ mutant/control suites; M14 retains it.
- **D4 partly correct**: in-process preload works and source-path limitations are real; subprocess binding is absent/environment-dependent.

The absent float64 path is honestly disclosed, not a real float64 test. E3 measurement results/device mutants/GPU bits remain G5. Required missing validation is named by the findings; no unrelated product edit is proposed.

## 8. F2: conversion site and smallest honest follow-up

At **tensor_cuda/tensor_cuda/__init__.py:48**, _NP_DTYPE.get(dtype,np.float32) substitutes fp32 for an unknown key including "float64" **before C++ sees the array**. The analogous factory fallback is :60. The enum in include/tc/core.h:30 has no float64; dtype_from_string at kernels.cu:50–58 would refuse it, but tensor() bypasses that parser. Both CPU probes observe an empty float64 request becoming float32.

The smallest honest repair is an explicit unsupported-dtype **refusal before np.asarray**, including float64, with the same policy in _factory. Prefer clear TypeError/ValueError over silently reducing precision. A warning explicitly naming float64-to-float32 conversion is a compatibility-preserving alternative, but still returns less precision and may be missed; it is not float64 support. Also inspect bindings.cpp:77–97 in that later round: numpy_dtype has another fallback, while the body does not perform the force-cast promised by its comment. That separate from_numpy/binding behavior was not executed or claimed fixed here.

**Advice only. No product fix made.**

## Reproduction, files and limits

Use run.py LABEL python3 -B SCRIPT with TMPDIR=/tmp/eng1_rev1_scratch, CUDA_VISIBLE_DEVICES empty and numerical thread variables at 1. The supervisor applies nice 19/timeout 550s, waits in foreground, then saves actual exit/timeout plus stdout/stderr. Every command remained below ten minutes. Initial map completion was observed as exit 0, 11,161 files; final comparison is recorded below.

Executed: frozen CPU selection, candidate E6 and copied-M14 E6, import/real collection probes, independent references, numerical/interval bounds, source/hash/diff census, and frozen audit scorer. Not executed/claimed: GPU G5, device bit identity, device mutation kills, device norm census, other engine operations, performance or promotion.

All outputs are under rev1, scratch under /tmp/eng1_rev1_scratch. The copied M14 package/module/tests and one .ppm are deliberate artifacts. **Nothing was deleted or moved to DELETE_ME.** review_files.json inventories all deliverables.

## Prior art

Sources also appear at computation sites and in LEDGER.md:
- Installed PyTorch 2.11.0 documentation/source: reshape/view inference, topk and pooling contracts. Taken: documented semantics; independent cases corroborate. Public doc leads: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.view.html, https://docs.pytorch.org/docs/stable/generated/torch.reshape.html, https://docs.pytorch.org/docs/stable/generated/torch.topk.html. Network versions were not fetched.
- IEEE 754 and Higham (2002), Accuracy and Stability of Numerical Algorithms: relative/absolute rounding and gamma_n. Moore (1966), Interval Analysis: outward propagation. Taken: established bounds; ours: registered-operand evaluations.
- Fornberg (1988), finite-difference formulas: centered differences. Taken: smooth scalar/VJP checks.
- Zhang/Sennrich (2019), RMSNorm; Ioffe/Szegedy (2015), Batch Normalization; Ba/Kiros/Hinton (2016), Layer Normalization; Wu/He (2018), Group Normalization. Taken: definitions; RMS Jacobian independently differentiated.
- Linnainmaa (1970), reverse-mode computational graphs: ordinary chain-rule accounting of stored intermediates and range limits.
- DeMillo/Lipton/Sayward (1978), mutation testing: one edit, registered witness and control comparisons. Ours: uncovered-decision witnesses.
- GN1 A8 completed-child supervision; Python import caching/subprocess environment; pytest/JUnit; GNU timeout; SHA-256, NIST FIPS180-4 (2015). Existing evidence formats, no new hashing/supervision algorithm.

Bibliographic details and CUDA 12.6 powf 4-ulp literature are **unverified — lead to check** the named authors/titles and "CUDA 12.6 single precision powf 4 ulp accuracy". This review adds scoped call audits, independent fixed-input calculations and findings.
"""
(rev/'REPORT.md').write_text(out)
print('REPORT.md bytes',len(out.encode()))
