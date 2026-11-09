"""Render the A2 required delivery from receipts; no gates or GPU execution.
Prior art: A1 report rendering; GN1 evidence-class/reporting requirements.
"""
import datetime
import json
from pathlib import Path
import numpy as np
import pytest

TOOL=Path(__file__).resolve().parent
ROOT=TOOL.parents[2]
read=lambda name:json.loads((TOOL/name).read_text())
def put(name,text):
    (TOOL/name).write_text(text)
assert pytest.__version__=="9.0.2"
put("L01_instruments.json",json.dumps(dict(passed=True,numpy=np.__version__,pytest=pytest.__version__,
    scripts=["../a1/run_gpu_line.py","../rev1/hostile_scorer.py","../rev1/shim_trace.py","../rev1/gn_emulate.py"],
    module=read("engine_pin.json")["module"],verified_CPU_only=True),indent=2)+"\n")
gates=read("GATES.json");g1=read("G1_scope.json");g2=read("G2_cpu.json");g3=read("G3_pin.json")
g4=read("G4_hostile.json");g5=read("G5_mutants.json");tol=read("tolerance_bounds.json")
lines=read("gpu_lines.json");mutants=read("mutants.json")
assert gates["acceptance"]=="ACCEPT" and g4["passed"] and g5["passed"]
fixes=[
("R1-F01","Case-insensitive session-wide device veto; real assertion/refusal kill required.","evidence.py; G4_hostile.json H2_REV1_exact/H2c_REV1_exact/H2_pattern_00–11"),
("R1-F02","Bind child arm/suite, nn/engine/seal/root and JUnit outcomes ID by ID; exclude test_reference_* and unregistered exception classes.","G4_hostile.json H3/H3b/H4, Junit_one_ID_disagreement, Reference_only_kill, Wrong_nn_hash, Wrong_engine_hash, One_byte_engine_copy"),
("R1-F03","Scorer re-verifies START/seal/root, child/JUnit, expected exits, forge log and stored verdicts; recomputes dependencies.","G4_hostile.json H1/H1b, Stored_parent_tamper, Expected_exit_tamper, START_seal_tamper; evidence.py"),
("R1-F04","Split all sites, including both affine branches and both mean/variance expressions. One runnable expression per mutant.","build_mutants.py; G5_mutants.json; mutants/*/nn.diff and trace.json"),
("R1-F05","Both distinct parameter dtype configurations get forward/backward cases; V1/V2 assert-fail; V3 has a post-D5 equivalence proof.","G2_shim.json; mutants/V1/trace.json; mutants/V2/trace.json; mutants/V3/trace.json; EQUIVALENCES.md"),
("R1-F06","Positive integer/bool checks before divisibility; rank/channel and input/weight/bias dtype refusals before tensor ops.","nn_vs_0c6a4244.diff; native_validation.json (35 cases); G2_shim.json; pytorch_behavior.json"),
("R1-F07","Candidate regression permits exits 0/1; old passes must still pass and each candidate failure must match a baseline failure class.","gpu_lines.json; G4_hostile.json D6_normal, D6_regression_legacy_error and negative D6 controls"),
("R1-F08","Numbered not-admitted 75/76 attempts do not consume a label; scorer lists attempts and selects last admitted.","G4_hostile.json H5_retry_75/H5_retry_76; hostile_runs/H5_retry_*/"),
("R1-F09","Replace multiplier prose with per-op first-order bound and final-rounding composition; preserve every number.","TOLERANCES.md; derive_tolerances.py; tolerance_bounds.json (324 IDs)"),
("R1-F10","43 engine byte copies, built and child-start hash checked; deterministic arm regeneration from order commit + D5/mutant recipes + pin.","regenerate_arms.py; regeneration_verified.json; arm_maps.json; G3_pin.json"),
("R1-F11","Read runner’s final nine-field log; record ran/timeout/window-closed/no-window/no-lock plus exit. Missing log stays missing.","G4_hostile.json D10_timeout/D10_window_closed/H6_hang_after_summary; parent_result.json forge_outcome"),
]
fix_table="| Finding | Fix | Receipt |\n|---|---|---|\n"+"".join(f"| {a} | {b} | {c} |\n" for a,b,c in fixes)
put("FINDINGS_FIXED.md","# Per-finding fixes\n\nAll R1-F01 to R1-F11 fixed for the registered CPU acceptance; independent REV2 pending.\n\n"+fix_table)
control_table="| Control | Target | Observed result | Receipt |\n|---|---|---|---|\n"
for i,c in enumerate(g4["controls"]):
    name=c["control"];target=c["target"]
    outcome=("refusal rejected; retry admitted" if name.startswith("H5_retry") else
             "admitted" if name in ("Full_valid_synthetic","Registered_refusal_positive","D6_normal","D6_regression_legacy_error","D6_dual_failure") else "rejected")
    control_table+=f"| {name} | {target} | {outcome}; PASS | G4_hostile.json controls[{i}] |\n"
put("G4_CONTROLS.md","# G4 controls\n\n48/48 PASS through the real sealed parent/scorer. CPU forge stand-in only. See HOSTILE_ADAPTATIONS.md for each REV1 adaptation.\n\n"+control_table)
mut_table="| Mutant | Witness (prefix below) | Edited → reached lines | CPU outcome | Trace |\n|---|---|---|---|---|\n"
for m in mutants:
    row=next(r for r in g5["mutants"] if r["id"]==m["id"])
    mid=m["id"];witness=m["predicted_witness"].split("::")[1]
    edited=row["edited_lines"];reached=row["reached_lines"]
    outcome="equivalent (V3 proof)" if row["equivalence"] else "AssertionError"
    mut_table+=f"| {mid} | `{witness}` | {edited} → {reached} | {outcome} | [trace](mutants/{mid}/trace.json) |\n"
put("MUTANTS.md","# A2 mutant table\n\nPrior art: DeMillo/Lipton/Sayward (1978), mutation adequacy; Offutt/Voas RIP model, unverified — lead to check. All 41 load, differ and reach their single edit site. No multiline edit exception is needed. Witness prefix: `tensor_cuda/tests/test_groupnorm_gn1.py::`. 40 CPU assertion kills with explicit NumPy engine stand-in; one V3 equivalence. No actual GPU kills claimed.\n\n"+mut_table)
stage_table="| Label | Arm/suite | Expected child exits | Expected parent | Inner / max / outer (s) |\n|---|---|---|---|---|\n"
for l in lines:
    stage_table+="| {} | {}/{} | {} | 0 | {} / {} / {} |\n".format(l["label"],l["arm"],l["suite"],l["expected_child_exits"],l["inner"],l["maximum"],l["outer"])
put("STAGED_LINES.md","# Staged lines and expected exits\n\nEvery line runs from A2’s own directory through forge-gpu-run and foreground timeout. This seat ran only --audit. Lease refusals are not admitted lines and may be retried.\n\n"+stage_table+"\n```bash\n"+(TOOL/"STAGED_GPU_COMMANDS.sh").read_text()+"```\n")
idsdoc="# Native collected IDs (GPU hidden)\n\nExact IDs from pytest collection; focus is 388 per arm, regression 444 per arm, mutation 381, audit 29.\n"
for name in ("baseline_focus","candidate_focus","baseline_regression","candidate_regression","mutation","audit"):
    idsdoc+="\n## "+name+"\n\n```text\n"+(TOOL/("ids_"+name+".txt")).read_text()+"```\n"
put("COLLECTED_IDS.md",idsdoc)
pt=read("pytorch_behavior.json")
ptdoc="# Live PyTorch behavior\n\nCPU observation of PyTorch "+pt.get("version","unverified — lead to check")+"; A2 requirements govern the product even where PyTorch differs. No GPU queried or allocated.\n\n| Case | PyTorch result |\n|---|---|\n"
for name,r in pt.get("results",{}).items():
    result=r.get("exception_class", "accepted")
    if r["outcome"]=="accepted":result="accepted constructor" if name.startswith("num_") else "accepted"
    ptdoc+=f"| {name} | {result} |\n"
ptdoc+="\nExact exception messages and operands: pytorch_behavior.json. Both mixed configurations refuse during forward; there is no backward outcome to claim.\n"
put("PYTORCH_BEHAVIOR.md",ptdoc)
start=datetime.datetime.fromisoformat(read("started.json")["utc"])
now=datetime.datetime.now(datetime.timezone.utc)
elapsed=(now-start).total_seconds()
report="""# GN1 A2 report

**ACCEPT for the registered CPU gates: G1–G6 PASS.** All 11 REV1 findings have scoped fixes and receipts. P86 HELD; P87 HELD. Independent GN1-REV2 and the lead’s GPU run remain pending. No GPU execution, engine rebuild, C++/CUDA change, git write, subagent or deletion occurred. CPU stand-in assertions are not device kills or GPU numerical evidence.

Order: `orders/GN1_A2_REV1_FIXES.md`, immutable commit `13b8f77b8c3363593b256f9e60b7df197a5f8b19`. Registration: REGISTRATION.md; execution amendments: AMENDMENT_EXECUTION_NOTES.md; modifications and preparation corrections: LEDGER.md. The final content seal is staging_seal.json; previous seals and CPU attempts are retained as historical receipts. Every command used an explicit timeout below 20 minutes and TMPDIR under /tmp/gn1_a2_*, with CUDA hidden for CPU execution and thread counts at one.

## Gates

| Gate | Verdict | Evidence and scope |
|---|---|---|
| G1 | PASS | G1_scope.json; changes_vs_base.patch; changes_vs_a1.patch; nn_vs_0c6a4244.diff. Only GroupNorm and the test file differ in product scope. 7,753 tracked hashes: only the two authorized product/test files changed. 702 frozen GN1/A1/REV1 files match before/after; maps have identical SHA-256. |
| G2 | PASS | G2_cpu.json; native_cpu_runs.json; cpu_candidate_mutation_references.json: 29 references + 14 constructor refusals + divisibility refusal pass natively (44 calls). native_validation.json: 35 actual-class metadata-only guard cases. G2_shim.json: all 381 real Python test IDs pass with explicit NumPy stand-in, including new refusals and both mixed forward/backward cases. Both native arms import and collect (388 focus, 444 regression per arm), CUDA hidden. Full IDs: COLLECTED_IDS.md and ids_*.txt. |
| G3 | PASS | G3_pin.json; engine_sources_verified.json; arm_maps.json; regeneration_verified.json. Original pin/source map unchanged, all 72 source hashes match; all 43 A2 engine byte copies match pin; no symlinks/builds. |
| G4 | PASS | G4_hostile.json; G4_CONTROLS.md; hostile_runs/ and hostile_run.log: 48 controls through real sealed parent/scorer. HOSTILE_ADAPTATIONS.md lists each REV1 adaptation. |
| G5 | PASS | G5_mutants.json; mutants.json; mutants/*/trace.json; MUTANTS.md. All 41 single-site mutants load, differ and reach. 40 CPU assertion failures; V3 equivalence proof in EQUIVALENCES.md. Actual device kills remain unclaimed. |
| G6 | PASS | G6_staging.json; STAGED_GPU_COMMANDS.sh; gpu_lines.json; staging_seal.json; audit_cpu/. 45 staged labels, expected exits registered; real audit child 29 references passed, exit 0; parent passed after child exit, forge outcome ran. |

Frozen-map SHA-256: `29d5b007d3ad8243367b8ae85e1166b3a4b1e4fa62d0d4b82daca4c473e072ee` before and after. Engine module SHA-256: `c9b7dc610421010a8b11e8e3af74d1a9ad99928dabb98835aa6c265754daf8ce`. Source-map SHA-256: `98ef59cd48200fb4af38caf9dd017022f0ad032b7f42082dfd1574506b470a5c`.

## Per-finding fix table

"""+fix_table+"\n## nn.py diff against 0c6a4244\n\n```diff\n"+(TOOL/"nn_vs_0c6a4244.diff").read_text()+"```\n\n## G4 control table\n\n"+control_table+"\n## Mutant table with witness traces\n\nWitness prefix is `tensor_cuda/tests/test_groupnorm_gn1.py::`; traces are real Python line events with explicit CPU stand-in. M14’s definition line is traced during import and its test calls the constructor without eps. Every other mutant reaches its expression during its test. Every mutant is exactly one replacement expression; no multi-line exception is needed.\n\n"+mut_table+"\n## D8 bounds and margins\n\n"+(TOOL/"TOLERANCES.md").read_text().removeprefix("# A2 tolerance derivations (D8)\n")+"\n## PyTorch comparison\n\n"+ptdoc.removeprefix("# Live PyTorch behavior\n")+"\n## Staged lines and expected exits\n\n"+stage_table+"\nExact staged script (no GPU line executed by this seat):\n\n```bash\n"+(TOOL/"STAGED_GPU_COMMANDS.sh").read_text()+"```\n"
report+="""
## Regeneration

`regenerate_arms.py --verify-seal` computes every package byte from the immutable
order commit, the registered D5 insertions and single-site mutant recipe, and
the module pin. A1 mutant definitions are read from that commit using read-only
git. Baseline package bytes come from 0e1dea3. It copies the pinned module,
checks SHA-256 while building, refuses existing different bytes/symlinks, and
compares the regenerated maps with the seal. The child repeats arm/module
checks before importing and asserts the engine is its own copy. Two receipts (initial
and final) cover deterministic agreement (no rebuild).
The script and recipes are the lead’s commit record; arms are generated local
artifacts and must not be committed. This seat performs no git writes.

Reproduction from the worktree (after obtaining the pinned module):

```bash
TMPDIR=/tmp/gn1_a2_reproduce CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout 90s python3 artifacts/gn1/a2/regenerate_arms.py --verify-seal
```

## Predictions

- P86 HELD — all 48 final controls meet their targets; H2c, H3b and H6 stay rejected. H5 refuses each lease-denied attempt and admits its valid retry; it does not reject the retry.
- P87 HELD — every A1 tolerance number is exactly retained. FP32 compute bound peaks at 0.9073368234728002 of budget. Final destination rounding leaves the smaller, explicitly recorded BF16 margin; no budget was widened.

## Prior art

Wu and He (2018), Group Normalization, for the existing population-statistic
formula (untouched); PyTorch GroupNorm for validation context, with concrete
CPU behavior verified at version 2.11.0+cu130 (A2’s positive-int/refusal policy
and mixed dtype acceptance are order requirements, not claims of PyTorch
parity). DeMillo, Lipton and Sayward (1978), mutation adequacy; Offutt/Voas RIP
model, for single-site mutants and witness reach. Wilkinson (1960s) and Higham
(1996/2002), for first-order running error recurrences. IEEE 754 (1985/2019),
for rounding/underflow; NIST SHA-256 (2001), for seals; pytest/JUnit and GNU
timeout/Python subprocess, for reports and foreground supervision. Meszaros
(2007), fault-injection test stubs; Karpathy micrograd (2020), expression-graph
reverse mode used only in the CPU stand-in; classical central differences,
Fornberg (1988), for retained reference checks. Bibliographic details and
CUDA powf 4-ulp accuracy are unverified — lead to check named titles/authors
and CUDA 12.6 Math API powf maximum ulp error. V3’s elementary case analysis:
no specific prior art known to me. A2 contributes the order-specific guards,
receipt bindings, attempt records, single-site recut and finite-census bound
wiring. Prior art is also recorded at code sites and in LEDGER.md.

## Failures retained and limits

The first stand-in import failed with `AttributeError: module 'tensor_cuda' has no attribute 'tensor'` because pytest replaced a module lacking
package metadata; the exact quoted exception is in G2_shim.json.attempt1 and cpu_candidate_shim.log.attempt1. The corrected shim
uses package metadata and the real suite passes. Initial tolerance JSON
serialization failed on NumPy int64 indices; indices now serialize as Python
ints. Two unmatched apply_patch contexts and one inline documentation quoting
`SyntaxError: invalid decimal literal` changed no files. The first hostile run was RED only because its
missing-ID control expected a later reason string; the exact reasons were `JUnit/summary outcomes disagree ID by ID` and `baseline-passing IDs no longer pass`; its control had expected `missing test calls`. That
expectation was corrected to the registered D6 rule. Earlier hostile
runs and seals are retained; the final 48 controls all passed under the final
seal. A final scorer rerun after restoring deliberately tampered receipts
accepts the full synthetic run (`full_synthetic_restored.log`); that acceptance
is CPU harness sensitivity, never GPU acceptance.

Not claimed fixed or established: GPU values/gradients, real device mutant
kills, the real regression outcome, FP16-POW1, reshape(-1), other layers,
performance, or independent review. The lead runs GN1-REV2 before any GPU run.

## Done

Every workspace file written is listed exactly in files_written.txt/json;
the two product/test paths and every A2 arm/module/mutant/receipt/tool are
included. DONE_VERBATIM.md embeds the report and that exhaustive list. All
files under A2 are intentional deliverables or retained provenance; no other
workspace path was edited. No deletion or DELETE_ME move occurred.

Leave-and-list for the lead: `/tmp/gn1_a2_boot` is scratch (including the saved
one-byte-control engine copy and outside-guard directory). Generated A2 arms
are retained and are not for commit; the lead may bin them after review, then
regenerate from the script. The .attempt* trees and previous seals
remain as historical receipts, not active runs. Nothing was silently removed.
"""
report+=f"\nWall-clock since A2 start snapshot: {elapsed:.1f} s (budget 7200 s); every single command bound <1200 s.\n"
put("REPORT.md",report)
# Create planned final receipt before census so every final written path is listed.
if not (TOOL/"delivery_verify.json").exists():put("delivery_verify.json",json.dumps(dict(status="PENDING final read-back"))+"\n")
paths=["tensor_cuda/tensor_cuda/nn.py","tensor_cuda/tests/test_groupnorm_gn1.py"]
paths += [str(p.relative_to(ROOT)) for p in sorted(TOOL.rglob("*")) if p.is_file()]
paths += ["artifacts/gn1/a2/REPORT.md","artifacts/gn1/a2/DONE_VERBATIM.md","artifacts/gn1/a2/files_written.txt","artifacts/gn1/a2/files_written.json"]
paths=sorted(set(paths))
scratch=Path("/tmp/gn1_a2_boot")
scratch_files=sorted(str(p) for p in scratch.rglob("*") if p.is_file())
put("files_written.txt","\n".join(paths)+"\n")
put("files_written.json",json.dumps(dict(workspace_files=paths,scratch_files=scratch_files,scratch_root=str(scratch),deletes=[],DELETE_ME_moves=[]),indent=2)+"\n")
put("DONE_VERBATIM.md",report+"\n## Every file written\n\n```text\n"+"\n".join(paths)+"\n```\n\nScratch files left for the lead to bin (under the exempt /tmp root):\n\n```text\n"+"\n".join(scratch_files)+"\n```\n")
print(json.dumps(dict(report="REPORT.md",verbatim="DONE_VERBATIM.md",files=len(paths),elapsed_seconds=elapsed,acceptance="ACCEPT",G1_to_G6="PASS")))
