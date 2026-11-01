"""Compose evidence-bounded A1 report from completed gate receipts.
Prior art: round-one GN1 evidence/ledger format; no new algorithm.
"""
from pathlib import Path
import datetime
import hashlib
import json
import re
TOOL=Path(__file__).resolve().parent
ROOT=TOOL.parents[2]
def data(f):return json.loads((TOOL/f).read_text())
def write(f,s):(TOOL/f).write_text(s)
for gate in ("G1_scope.json","G2_cpu.json","G3_pin.json","G4_staging.json"):assert data(gate)["passed"]
summary=data("G2_arithmetic_summary.json")
meta=data("test_id_tolerances.json")
new=[r for r in meta if r["status"]=="new"]
changed=[r for r in meta if r["status"]=="changed"]
# P85 is the original all-FP32 lane. The new FP32/FP16 mixed-parameter
# witnesses are explicit in the census rather than hidden in that count.
census={}
for row in meta:
    if "input_dtype" not in row:continue
    key=(row["input_dtype"],row["parameter_dtype"])
    census[key]=census.get(key,0)+1
write("dtype_configuration_census.json",json.dumps([dict(input_dtype=k[0],parameter_dtype=k[1],ids=v) for k,v in sorted(census.items())],indent=2)+"\n")
dtypes="""# Dtype and parameter configurations (source, collection and reasoning evidence)

| Input | Parameter leaves | Output | x gradient | weight/bias gradients | Coverage |
|---|---|---|---|---|---|
| float32 | float32 | float32 | float32 | float32 | Original 98 IDs, byte-identical math statements, no casts |
| float16 | float16 | float16 | float16 | float16 | Original census, including restored singleton backward IDs |
| bfloat16 | bfloat16 | bfloat16 | bfloat16 | bfloat16 | Original census |
| float16 | module's untouched float32 | float16 | float16 | float32 | Mixed and low-variance, forward/backward |
| float16 | float16 via Module.half() | float16 | float16 | float16 | Mixed and low-variance, forward/backward |
| bfloat16 | module's untouched float32 | bfloat16 | bfloat16 | float32 | Mixed and low-variance, forward/backward |
| bfloat16 | float16 via Module.half() | bfloat16 | bfloat16 | float16 | Mixed and low-variance, forward/backward |
| float32 | float16 via Module.half() | float32 | float32 | float16 | Two new M17 witnesses, outside original all-FP32 lane |

Assertions are staged, not observed GPU results. Every gradient dtype must equal its own leaf's dtype, even when it differs from x. Cast backward is source-verified at ops.cpp:1060-1063. Integer/bool normalization and float64 engine computation are not supported, as established in round one's source census; NumPy float64 is reference only. Actual collected input/parameter counts are dtype_configuration_census.json. There are 100 engine IDs with FP32 inputs: 98 original FP32/FP32 IDs plus two new FP32/FP16 mixed witnesses. P85 refers to preservation of the original all-FP32 lane, not a claim that those new mixed witnesses do not exist.
"""
write("DTYPE_CENSUS.md",dtypes)
tol="""# Registered A1 tolerances

Evidence class: pre-device arithmetic/finite-census reasoning. REGISTRATION.md and tolerances_registered.json are immutable and precede every gate. No GPU data was collected. For represented operands and float32 compute, use round one's |z-r|<=A+R|r| with A=R=3e-5 forward, A=R=3e-4 backward. One destination rounding yields atol=(1+u)A+s, rtol=(1+u)R+u, including subnormals. Here u16=2^-11, s16=2^-25; ubf=2^-8, sbf=2^-134. FP32 has no extra rounding allowance. Each parameter gradient uses its own leaf dtype's budget, not x's.

| Destination | Forward atol | Forward rtol | Backward atol | Backward rtol |
|---|---:|---:|---:|---:|
"""
for dtype,v in data("tolerances_registered.json").items():
    tol += f'| {dtype} | {v["forward"][0]!r} | {v["forward"][1]!r} | {v["backward"][0]!r} | {v["backward"][1]!r} |\n'
tol += """
All low-precision budgets are no looser than round one. The original 98 FP32 IDs keep their original tolerances. The ordinary census uses round one's conditioning. New low-variance fixtures have extent eight and exact zero represented centers, |projection|<=1/2, |w|<=1.5, |q|<=.75 and invstd<=317; 20*u32*.75*317<2.84e-4 is the conservative finite-fixture FP32 backward floor calculation (within 3e-4), with relative allowance for nonzero results. It is an engineering budget for the census, not a universal error proof.

CPU oracle comparisons: binary-exact hand values use (0,0); original float64 central differences use (2e-7,2e-7), h=2^-18; new low-variance differences use (2e-7,2e-7), h=2^-25. The small h squared relative to var+eps (<9e-11) and float64 subtraction support that budget. These checks all passed before the staged CPU audit. Dtype strings, IDs and SHA256 use exact equality. NaN/inf always fail.

Prior art: IEEE 754 (1985/2019), Higham (1996/2002), classical central differences (Fornberg 1988 lead), round-one GN1 budgets. Literature unverified — lead to check named titles/terms. New contribution: this finite-census rounding derivation and fixture registration.
"""
write("TOLERANCES.md",tol)
overflow=data("overflow_arithmetic.json")
text="""# Overflow arithmetic (CPU float64 and represented-storage reasoning)

The engine's pow backward calls ew_pow(a.data(), p-1), ops.cpp:854-859; p=-.5 materializes base^-1.5 in a's dtype before multiplication. FP16 max=65504. Therefore base<65504^(-2/3)=0.0006153961657226473 exceeds its finite range. With default eps=1e-5, variance<0.0006053961657226473 is in this domain.

Nominal variance 1e-4: (1.1e-4)^(-1.5)=866784.1720414475 > 65504.
Nominal variance 1e-6: (1.1e-5)^(-1.5)=27410122.234342143 > 65504.
Singleton variance 0: (1e-5)^(-1.5)=31622776.60168379 > 65504.

The represented variances differ slightly from the decimal targets. Every new low-variance ID, its represented variance, and M15's stepwise stored var/base/power are in overflow_arithmetic.json. Default and .half() affine configuration and forward/backward do not change variance. Unique dtype/target rows follow; every group has this same variance.

| Input dtype | Target variance | Represented variance | Float64 inverse cube | M15 stored var+eps | M15 inverse cube before storage | Actual M15 FP16 overflow |
|---|---:|---:|---:|---:|---:|---|
"""
seen=set()
for r in overflow["rows"]:
    key=(r["dtype"],r["target_variance"])
    if key in seen:continue
    seen.add(key)
    text += f'| {key[0]} | {key[1]:g} | {r["represented_variances"][0][0]:.17g} | {r["invcube_float64"][0][0]:.17g} | {r["M15_stored_base"][0][0]:.17g} | {r["M15_power_before_storage"][0][0]:.17g} | {r["actual_M15_overflows"]} |\n'
text += """
BF16 itself has sufficient exponent range: the >65504 comparison for BF16-represented operands is a hypothetical FP16 range statement, not a claim that BF16 overflows. All eight new FP16 forward/backward/configuration IDs have an FP16-overflowing M15 backward intermediate. Arithmetic proves that intermediate range failure; an actual test kill remains G5. D1's FP32 storage keeps these powers representable. Prior art: IEEE 754 range/rounding, round one's AMENDMENT_01 and ops.cpp mechanism; literature unverified — lead to check.
"""
write("OVERFLOW.md",text)
# Concise verbatim IDs with shared exact tolerance/derivation keys.
keys="""# Exact new/changed IDs and tolerances

All IDs below are verbatim pytest node IDs. The prefix is tensor_cuda/tests/test_groupnorm_gn1.py::. F32/F16/B16 mean the output dtype, with each gradient using its own leaf's matching dtype entry. Entries give (atol,rtol). Forward checks also execute inside backward IDs. Exact dtype strings and finite arrays are required for every engine comparison.

| Key | Forward (atol,rtol) | Backward (atol,rtol) | Derivation |
|---|---|---|---|
| F32 | (3e-5,3e-5) | (3e-4,3e-4) | Original FP32 budget, unchanged |
| F16 | (3.0044450759887697e-5,0.0005182958984375) | (0.00030017628669738765,0.000788427734375) | Represented input; FP32 budget; one rounding u=2^-11,s=2^-25 |
| B16 | (3.01171875e-5,0.0039363671875) | (0.00030117187499999995,0.004207421875) | Represented input; FP32 budget; one rounding u=2^-8,s=2^-134 |
| FD | (2e-7,2e-7) | n/a | Float64 central differences h=2^-25; truncation/subtraction derivation in REGISTRATION.md |

For F16/B16: A'= (1+u)A+s, R'=(1+u)R+u, A=R=3e-5 forward / 3e-4 backward. Table rows identify output / x-gradient / parameter-gradient keys. A dash means no gradient assertion in that forward ID. All IDs and leaf-specific numeric tolerances also appear in test_id_tolerances.json. The original 98 float32/float32 IDs are unchanged and are not listed here.

| ID | Status | Output / x grad / parameter grads |
|---|---|---|
"""
for r in [*changed,*new]:
    if "reference" in r["id"]:
        assignment="FD"
    else:
        k={"float32":"F32","float16":"F16","bfloat16":"B16"}
        assignment=k[r["input_dtype"]]+" / "+(k[r["input_dtype"]]+" / "+k[r["parameter_dtype"]] if "x_grad" in r["comparisons"] else "- / -")
    keys += f'| `{r["id"]}` | {r["status"]} | {assignment} |\n'
write("IDS_VERBATIM.md",keys)
# Every staged child command as it appears in the actual shell script.
script=(TOOL/"STAGED_GPU_COMMANDS.sh").read_text()
command_lines=[line.strip() for line in script.splitlines() if "python3 run_gpu_line.py" in line]
assert len(command_lines)==22
staged="""# Staged lines (verbatim)

Run from artifacts/gn1/a1 with RUNS an explicit absolute fresh subdirectory of a1, reused across labels; TMPDIR=/tmp/gn1_a1_staged. No RUNS default. The shell writes/checks START.json before each line and changes into its tooling directory. GPU_RUN=forge-gpu-run for real G5; no device command ran this round. Every parent exit is expected 0, with child expected 0 on candidate focus/regression, child expected 1 on baseline focus/regression and all mutants. The second numeric argument below is the expected child exit. Each parent writes its receipt after child exit; collection, device, setup failures and timeouts cannot count as mutant kills.

```bash
"""+"\n".join(command_lines)+"\n```\n\nCPU audit line actually run (exit 0):\n\n```bash\nTMPDIR=/tmp/gn1_a1_boot CUDA_VISIBLE_DEVICES= RUNS="+str(TOOL/"audit_cpu")+" timeout 120s bash "+str(TOOL/"STAGED_GPU_COMMANDS.sh")+" --audit\n```\n\nThe child ran 29 reference IDs, all passed; parent recorded child exit=0, timeout=false, audit=true, passed=true, after child exit. score_gpu.py is staged only; running it after all 22 actual device lines is expected exit 0.\n"
write("STAGED_LINES.md",staged)
# Anticipate the final inventory outputs so it includes itself and the report.
for name in ("files_written.txt","files_written.json","DONE_VERBATIM.md","GATES.json"):
    path=TOOL/name
    if not path.exists():path.write_text("")
files=[ROOT/"tensor_cuda/tensor_cuda/nn.py",ROOT/"tensor_cuda/tests/test_groupnorm_gn1.py"]
files += sorted(f for f in TOOL.rglob("*") if f.is_file() or f.is_symlink())
relative=[str(f.relative_to(ROOT)) for f in files]
write("files_written.txt","\n".join(relative)+"\n")
write("files_written.json",json.dumps([dict(path=str(f.relative_to(ROOT)),kind="symlink" if f.is_symlink() else "file",target=str(f.resolve().relative_to(ROOT)) if f.is_symlink() else None) for f in files],indent=2)+"\n")
done="## Done\n\nNo deletes, no moves to DELETE_ME, and nothing for the lead to bin. Existing round-one artifacts, arms and CUDA-12.0 retained build remain read-only historical evidence. Partial A1 staging was completed in place; first failed receipts remain for audit. Scratch directories /tmp/gn1_a1_boot and /tmp/gn1_a1_cpu remain; Python cache/scratch only is exempt but nothing was deleted. Every file written, including new module symlinks (not module copies), is listed below and in files_written.txt/json.\n\n```text\n"+"\n".join(relative)+"\n```\n"
write("DONE_VERBATIM.md",done)
predictions={"P84":{"status":"NOT HELD","meaning":"not established, not refuted: no GPU cases or M15 kill ran; lead G5 pending","cpu_evidence":"budgets strictly tighter than round one; overflow arithmetic supports predicted M15 intermediate failure"},"P85":{"status":"HELD","meaning":"98 original float32/float32 IDs and tolerances preserved; byte-identical math statement sequence, no casts","additional_mixed_witnesses":"two new float32-input/half-parameter IDs are separately counted; total FP32-input engine IDs now 100"}}
gates=dict(verdict="ACCEPT",registered_acceptance="G1-G4",G1=data("G1_scope.json"),G2=data("G2_cpu.json"),G3=data("G3_pin.json"),G4=data("G4_staging.json"),G5="NOT RUN; lead after clean cross-model review",predictions=predictions,completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
write("GATES.json",json.dumps(gates,indent=2)+"\n")
report="""# GN1 amendment A1 report

ACCEPT under registered G1-G4 acceptance. G5 NOT RUN: actual device forward/backward, regression outcomes and mutant kills are not claimed. Evidence classes: source/byte identity, CPU NumPy unit tests and central differences, arithmetic, hidden-CUDA imports/collection, and one CPU staging audit. No engine rebuild, C++/CUDA source change, GPU use, git write, deletion or production edit.

## Gates

- G1 PASS — only GroupNorm in nn.py and the designated test changed. 7,593 other current tracked files and 445 existing round-one/build files are byte-identical. Relative to 0e1dea3, common product files differ only in GroupNorm plus the new test; relative to 50b3418f, GroupNorm changes are D1 only. Orders/receipt metadata are scoped separately and preserved. Receipts: G1_scope.json, scope_pin_checks.json/log, tracked_before.json, tracked_after.json, base_sha256.json, round1_sha256.json, readonly_before.json, readonly_after.json, changes_vs_base.patch, changes_vs_round1.patch, float32_identity.json.
- G2 PASS — 29 CPU oracle checks (two exact examples, three original differences, 24 new low-variance differences) plus refusal pass. Both arms import the pinned module and collect identical 351 GN1 / 358 focus / 414 regression IDs with CUDA hidden. 98 original FP32/FP32 IDs and tolerances unchanged. Recomputed all new low-variance range comparisons, with eight actual FP16 M15-overflow operands. Receipts: G2_cpu.json, reference_checks_preloaded.json/log, cpu_candidate_mutation_references.json, collect_baseline_focus.json/log, collect_candidate_focus.json/log, collect_baseline_regression.json/log, collect_candidate_regression.json/log, ids_baseline_focus.txt, ids_candidate_focus.txt, ids_baseline_regression.txt, ids_candidate_regression.txt, arithmetic_checks.json/log, overflow_arithmetic.json, test_id_tolerances.json.
- G3 PASS — pinned module c9b7dc610421010a8b11e8e3af74d1a9ad99928dabb98835aa6c265754daf8ce; source-map SHA256 98ef59cd48200fb4af38caf9dd017022f0ad032b7f42082dfd1574506b470a5c, 72 unchanged source inputs, unchanged existing build tree. All 20 arms use symlinks to that same module. Baseline Python bytes exactly match 0e1dea3. No rebuild. Receipts: G3_pin.json, engine_pin.json, engine_sources_original.json, engine_sources_after.json, scope_pin_checks.json/log.
- G4 PASS — 22 bounded staged lines, each expected child/parent exit registered; M01-M18 re-derived on D1, no equivalence or actual kill claimed; CPU audit child=0, parent=0, timeout=false, 29 reference calls passed. Receipts: G4_staging.json, staging_checks.json/log, STAGED_GPU_COMMANDS.sh, gpu_lines.json, mutants.json, MUTANTS.md, mutant_validation.json, staging_seal.json, audit_cpu/START.json, audit_cpu/candidate_focus/parent_result.json, audit_cpu/candidate_focus/child_result.json, audit_cpu/candidate_focus/stdout.log.

## D1 and float32 identity

The input is differentiably cast to FP32 for half/BF16; all statistics, normalization and affine operations compute in FP32. Non-FP32 parameter casts retain each leaf's source dtype in backward; final output returns to the original input dtype. Original FP32/FP32 inputs return through the original affine statement before all parameter/output casts. float32_identity.json compares the reached math statement source segments byte-for-byte, ordered SHA256 d1c87a381d0a74e4b48fe8e03ed28c124b89e8bb061ec3619ae6b17f6e241991. Indentation/control guards are added, not tensor arithmetic. The constructor is unchanged from round one.

### nn.py diff against round one (verbatim)

```diff
"""+(TOOL/"nn_vs_round1.diff").read_text()+"```\n\n"+tol+"\n"+dtypes+"\n"+text+"\n"
report += "## Test IDs\n\n52 new IDs include two restored singleton backward IDs, eight mixed-parameter IDs, 16 low-variance device IDs, two FP32-input/half-parameter witnesses, and 24 CPU low-variance finite-difference IDs. The 194 existing low-precision IDs now use the pre-registered tighter budgets. All 299 old IDs are preserved; none is skipped or xfailed. Original FP32/FP32 numeric tolerances/IDs remain unchanged. Exact strings and per-leaf tolerances follow.\n\n"+keys+"\n"+(TOOL/"MUTANTS.md").read_text()+"\n"+staged
report += """
## Predictions

- P84 NOT HELD — not established, not refuted. This is a G5 prediction: actual FP16/BF16 pass outcomes and M15's observed killer require GPU execution, which this round forbids. CPU arithmetic and strictly tighter registered budgets support the mechanism but cannot substitute for that outcome.
- P85 HELD — 98 original FP32/FP32 IDs, original tolerances, byte-identical reached tensor-math statements and no casts. Two newly required FP32-input/half-parameter witnesses are separately counted; total FP32-input engine IDs are 100. They exercise different parameter dtypes from the unchanged original lane.

## Failures retained and residuals

Preparation attempt 1 failed with AssertionError: ('M02', '(var + self.eps).pow(-0.5)', 5), because the generator counted matching text across nn.py instead of just GroupNorm. The generator was restricted to the class and partial artifacts completed; no change outside the class was applied. The first CPU smoke recorded 29 passed and one refusal import failure: ImportError: cannot import name 'nn' from 'tensor_cuda' (unknown location). inspect_cpu.py now preloads/asserts the real staged package before pytest, matching round one's GPU child. The corrected smoke passes; original failure receipt reference_checks.json/log is retained. No tolerance or reference was changed in either correction.

Not claimed fixed outside GroupNorm: engine FP16 pow-backward materialization; BatchNorm1D/InstanceNorm2D; -1 reshape inference; GPU correctness, regression results and mutant kills; performance. G5 and independent cross-model review remain the lead's next steps. Actual mutant killers are null until valid completed device receipts exist. No equivalence proofs are claimed.

## Prior art

PyTorch GroupNorm accumulation in FP32 for half normalization (living implementation/version unverified — lead to check: PyTorch group_norm acc_type half); Micikevicius et al. (2018), Mixed Precision Training (unverified — lead to check title), for FP32 compute with low-precision storage. Wu and He (2018), Group Normalization, for population statistics and contiguous grouping; PyTorch GroupNorm for refusal and biased variance; round-one GN1 implementation/reference/scaffolding reused. IEEE 754 (1985/2019) for range and one-rounding bounds; Higham (1996/2002), Accuracy and Stability of Numerical Algorithms, for gamma_n; classical central differences (Fornberg 1988, unverified lead to check); DeMillo, Lipton and Sayward (1978), Hints on Test Data Selection, for mutation testing. SHA256 (NIST 2001), pytest hooks (Krekel et al. 2004+), GNU timeout and Python subprocess for pinning/synchronous receipt plumbing. Literature not checked through a network; named leads are for lead verification. A1's contribution is D1's dtype wiring, these finite fixture/error-budget registrations, and extending the existing mutation/staging instruments.

"""+done
write("REPORT.md",report)
with (TOOL/"LEDGER.md").open("a") as f:
    f.write("- G1/G3: scope_pin_checks.json/log exit 0, no timeout. 7593 other tracked files, 445 existing artifact/build files and 72 pinned source inputs unchanged; same pinned module across 20 arms.\n")
    f.write("- G4: staged --audit command returned 0. audit_cpu/candidate_focus/parent_result.json records exit 0, timeout false and passed true after child exit; 29 CPU oracle calls passed. staging_checks.json/log exit 0. 22 device lines and 18 mutants remain NOT RUN.\n")
    f.write("- Final synthesis: ACCEPT for G1-G4 only. P84 NOT HELD (unestablished; G5 pending, not refuted). P85 HELD for original 98 FP32/FP32 IDs; explicitly report two new FP32-input/FP16-parameter witnesses, total FP32-input engine IDs 100. No GPU or kills claimed.\n")
    f.write("- Prior art: PyTorch half normalization and Micikevicius et al. (2018); round-one Wu/He (2018), PyTorch refusal/population variance, Higham, IEEE 754, central differences/Fornberg lead and DeMillo/Lipton/Sayward mutation scaffolding; SHA256/pytest/GNU timeout/Python subprocess. Literature unverified — lead to check named terms. D1 wiring, finite fixture/rounding budgets and staging extensions are A1's work.\n")
    f.write("- Every file written: files_written.txt/json (includes source/test, artifacts and module symlinks). No deletes, moves or files for the lead to bin. Scratch /tmp/gn1_a1_boot and /tmp/gn1_a1_cpu left in place. Final report contains exact diff, every new/changed ID with budget derivation, overflow arithmetic, dtype census, M01-M18 and staged lines.\n")
print(json.dumps(dict(verdict="ACCEPT",G1="PASS",G2="PASS",G3="PASS",G4="PASS",files_written=len(files),new_ids=len(new),changed_ids=len(changed),report_bytes=len(report.encode()))))
