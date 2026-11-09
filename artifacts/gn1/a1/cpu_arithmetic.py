"""A1 represented-operand census, overflow arithmetic, exact ID/tolerance audit.
Prior art: IEEE 754 round-to-nearest; Higham gamma_n; round-one GN1 oracle.
This is CPU arithmetic/source evidence, not emulated engine correctness.
"""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import numpy as np

TOOL=Path(__file__).resolve().parent
ROOT=TOOL.parents[2]
spec=importlib.util.spec_from_file_location("gn1_a1_reference",ROOT/"tensor_cuda/tests/test_groupnorm_gn1.py")
ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
registered=json.loads((TOOL/"tolerances_registered.json").read_text())
assert json.loads(json.dumps(ref.TOLERANCES))==registered
old_spec=importlib.util.spec_from_file_location("gn1_round1_reference",TOOL/"test.round1.py")
old=importlib.util.module_from_spec(old_spec);old_spec.loader.exec_module(old)
assert ref.TOLERANCES["float32"]==old.TOLERANCES["float32"]
for dtype in ("float16","bfloat16"):
    for lane in ("forward","backward"):
        assert all(a<=b for a,b in zip(ref.TOLERANCES[dtype][lane],old.TOLERANCES[dtype][lane]))
focus=(TOOL/"ids_candidate_focus.txt").read_text().splitlines()
regression=(TOOL/"ids_candidate_regression.txt").read_text().splitlines()
assert focus==(TOOL/"ids_baseline_focus.txt").read_text().splitlines()
assert regression==(TOOL/"ids_baseline_regression.txt").read_text().splitlines()
ids=[x for x in focus if "test_groupnorm_gn1.py::" in x]
old_ids=(TOOL.parent/"ids_candidate.txt").read_text().splitlines()
f32old=[x for x in old_ids if "float32" in x]
f32new=[x for x in ids if "float32" in x]
assert f32old==f32new
assert set(old_ids)<=set(ids)
for name,seq in [("ids_focus.txt",focus),("ids_regression.txt",regression),("ids_mutation.txt",ids),("ids_audit.txt",[x for x in ids if "reference" in x])]:
    (TOOL/name).write_text("\n".join(seq)+"\n")
baseline_fails=[x for x in ids if "reference" not in x and not x.endswith("::test_constructor_defaults")]
baseline_fails += ["tensor_cuda/tests/test_ext_phase7.py::test_norms_and_conv1d"]
(TOOL/"baseline_expected_failures.json").write_text(json.dumps(baseline_fails,indent=2)+"\n")
mutants=json.loads((TOOL/"mutants.json").read_text())
assert [m["id"] for m in mutants]==[f"M{i:02d}" for i in range(1,19)]
assert all(m["predicted_witness"] in ids for m in mutants)

# Byte comparison of the reached FP32 math statements, with indentation removed.
# No mock executes numerical work: this proves source path identity only.
def forward(text):
    cls=next(n for n in ast.parse(text).body if isinstance(n,ast.ClassDef) and n.name=="GroupNorm")
    return next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=="forward")
round1=(TOOL/"nn.round1.py").read_text()
candidate=(ROOT/"tensor_cuda/tensor_cuda/nn.py").read_text()
a,b=forward(round1),forward(candidate)
a_seq=[ast.get_source_segment(round1,n) for n in a.body]
b_seq=[]
for n in b.body:
    if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=="dtype":continue
    if isinstance(n,ast.If):
        condition=ast.unparse(n.test)
        if condition=="dtype in ('float16', 'bfloat16')":continue
        assert condition=="dtype == 'float32' and self.weight.dtype == 'float32' and (self.bias.dtype == 'float32')",condition
        b_seq += [ast.get_source_segment(candidate,z) for z in n.body]
        break
    b_seq.append(ast.get_source_segment(candidate,n))
assert a_seq==b_seq,(a_seq,b_seq)
sequence_bytes="\n".join(a_seq).encode()
identity=dict(float32_ids=len(f32new),ids_unchanged=True,tolerances_unchanged=True,math_statements=a_seq,math_statement_bytes_identical=True,math_sha256=hashlib.sha256(sequence_bytes).hexdigest(),no_casts_reached=True,proof="x dtype float32 skips upcast; all-float32 condition returns before parameter/output casts")
(TOOL/"float32_identity.json").write_text(json.dumps(identity,indent=2)+"\n")

rows=[]
for dtype in ("float16","bfloat16"):
    for variance in (1e-4,1e-6):
        values,w,b,projection=ref.low_variance_inputs(variance)
        x=ref.represented(values,dtype)
        v=x.reshape(2,3,8).var(axis=2)
        assert np.all(v>0) and np.all(np.abs(v/variance-1)<.01)
        invcube=(v+1e-5)**(-1.5)
        assert np.all(invcube>65504)
        # Arithmetic for M15's storage dtype at each var and eps operation.
        centered=x-x.reshape(2,3,8).mean(axis=2,keepdims=True).reshape(2,3,1,1).repeat(2,axis=2).reshape(2,6,1)
        squares=ref.represented(centered*centered,dtype).reshape(2,3,8)
        summed=ref.represented(squares.sum(axis=2),dtype)
        m15_var=ref.represented(summed/8,dtype)
        m15_base=ref.represented(m15_var+1e-5,dtype)
        m15_power=m15_base**(-1.5)
        assert np.all(m15_power>65504)
        for configuration in ("default","half"):
            for lane in ("forward","backward"):
                node=f"tensor_cuda/tests/test_groupnorm_gn1.py::test_low_variance[{lane}-var{variance:g}-{configuration}-{dtype}]"
                # pytest labels use explicit exponent spelling, not Python g format.
                node=node.replace("var0.0001","var1e-4").replace("var1e-06","var1e-6")
                assert node in ids,node
                rows.append(dict(id=node,dtype=dtype,parameters="float32" if configuration=="default" else "float16 via Module.half",target_variance=variance,represented_variances=v.tolist(),invcube_float64=invcube.tolist(),M15_stored_variance=m15_var.tolist(),M15_stored_base=m15_base.tolist(),M15_power_before_storage=m15_power.tolist(),float16_limit_exceeded=True,actual_M15_overflows=(dtype=="float16"),note="BF16 has sufficient exponent range; comparison to 65504 is a hypothetical FP16 storage bound" if dtype=="bfloat16" else "FP16 pow backward intermediate overflows on storage"))
result=dict(float16_max=65504,base_threshold=65504**(-2/3),variance_threshold_at_default_eps=65504**(-2/3)-1e-5,singleton_power=(1e-5)**(-1.5),rows=rows)
(TOOL/"overflow_arithmetic.json").write_text(json.dumps(result,indent=2)+"\n")

# Every ID records all comparisons, including forward checks inside backward IDs.
metadata=[]
for node in ids:
    entry=dict(id=node,status="new" if node not in old_ids else "changed" if ("float16" in node or "bfloat16" in node) else "unchanged")
    if "reference" in node:
        budget=[0,0] if "hand_values" in node else [2e-7,2e-7]
        entry.update(comparisons={"reference":budget},derivation="binary-exact arithmetic" if "hand_values" in node else "central differences: h=2^-25 low variance / 2^-18 original, truncation plus float64 subtraction; REGISTRATION.md")
    elif node.endswith("::test_constructor_defaults") or node.endswith("::test_nondivisible_refusal"):
        entry.update(comparisons={"exact":[0,0]},derivation="exact constructor/dtype/refusal equality")
    else:
        dtype="bfloat16" if "bfloat16" in node else "float16" if "float16" in node else "float32"
        pdtype="float32" if "-default-" in node and ("mixed_parameters" in node or "low_variance" in node) else "float16" if ("-half-" in node or "fp32_input_half_parameters" in node) else dtype
        backward="test_backward_values" in node or "[backward" in node
        comparisons={"output":ref.TOLERANCES[dtype]["forward"]}
        if backward:
            comparisons.update(x_grad=ref.TOLERANCES[dtype]["backward"],weight_grad=ref.TOLERANCES[pdtype]["backward"],bias_grad=ref.TOLERANCES[pdtype]["backward"])
        entry.update(input_dtype=dtype,parameter_dtype=pdtype,comparisons=comparisons,derivation="represented inputs; FP32 budget A=R=3e-5 forward / 3e-4 backward; destination rounding (1+u)*A+s, (1+u)*R+u; u16=2^-11,s16=2^-25; ubf=2^-8,sbf=2^-134; FP32 leaves unchanged",dtype_assertions="output=input; every gradient=its leaf, exact strings; all arrays finite")
    metadata.append(entry)
(TOOL/"test_id_tolerances.json").write_text(json.dumps(metadata,indent=2)+"\n")
changed=[e for e in metadata if e["status"]!="unchanged"]
md="# New and changed A1 IDs (verbatim)\n\nEvery comparison is abs(actual-reference)<=atol+rtol*abs(reference), all arrays finite. Dtypes compare exact strings.\n\n"
for row in changed:
    md += "- `"+row["id"]+"`: "+json.dumps(row["comparisons"])+"; "+row["derivation"]+".\n"
(TOOL/"NEW_CHANGED_IDS.md").write_text(md)
summary=dict(groupnorm_ids=len(ids),focus_ids=len(focus),regression_ids=len(regression),audit_ids=len([x for x in ids if "reference" in x]),new_ids=len(set(ids)-set(old_ids)),changed_existing_ids=sum(e["status"]=="changed" for e in metadata),float32_ids=len(f32new),overflow_cases=len(rows),float16_overflow_cases=sum(r["actual_M15_overflows"] for r in rows))
(TOOL/"G2_arithmetic_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary))
print(json.dumps({"base_threshold":result["base_threshold"],"nominal_1e-4_power":(1e-4+1e-5)**(-1.5),"nominal_1e-6_power":(1e-6+1e-5)**(-1.5)}))
