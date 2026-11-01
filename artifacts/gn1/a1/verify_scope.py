"""Exact byte/scope and pinned-module gates; read-only git, no engine calls.
Prior art: SHA256 (NIST 2001) content identity; Python AST source boundaries.
Reuses round-one source maps and the order's G1/G3 rules, no novel algorithm.
"""
from pathlib import Path
import ast
import difflib
import hashlib
import json
import subprocess

TOOL=Path(__file__).resolve().parent
ROOT=TOOL.parents[2]
sha=lambda p:hashlib.file_digest(open(p,"rb"),"sha256").hexdigest()
def read(name):return json.loads((TOOL/name).read_text())
def write(name,obj):(TOOL/name).write_text(json.dumps(obj,indent=2)+"\n")
before=read("tracked_before.json")
after={f:sha(ROOT/f) if (ROOT/f).is_file() else None for f in before}
write("tracked_after.json",after)
changed=[f for f in before if before[f]!=after[f]]
allowed=["tensor_cuda/tensor_cuda/nn.py","tensor_cuda/tests/test_groupnorm_gn1.py"]
assert sorted(changed)==sorted(allowed),changed
assert all(after[f]==before[f] for f in before if f not in allowed)
readonly=read("readonly_before.json")
readonly_after={f:sha(ROOT/f) for f in readonly}
write("readonly_after.json",readonly_after)
assert readonly==readonly_after
text=(ROOT/allowed[0]).read_text()
def outside_class(src):
    cls=next(n for n in ast.parse(src).body if isinstance(n,ast.ClassDef) and n.name=="GroupNorm")
    lines=src.splitlines(keepends=True)
    return "".join(lines[:cls.lineno-1]),"".join(lines[cls.end_lineno:])
for name in ("base","round1"):
    old=(TOOL/("nn."+name+".py")).read_text()
    assert outside_class(old)==outside_class(text),name
    diff="".join(difflib.unified_diff(old.splitlines(True),text.splitlines(True),fromfile=name+"/tensor_cuda/tensor_cuda/nn.py",tofile="candidate/tensor_cuda/tensor_cuda/nn.py"))
    (TOOL/("nn_vs_"+name+".diff")).write_text(diff)
    mapping=read(name+"_sha256.json")
    # Order/receipt additions are metadata, separately pinned unchanged above.
    product=lambda f:not f.startswith(("artifacts/","orders/","logs/"))
    deltas=[f for f in mapping if product(f) and (f not in after or mapping[f]!=after[f])]
    additions=[f for f in after if product(f) and f not in mapping]
    expected_deltas=allowed if name=="round1" else [allowed[0]]
    assert sorted(deltas)==sorted(expected_deltas),(name,deltas)
    assert additions==([] if name=="round1" else [allowed[1]]),(name,additions)
    write("scope_vs_"+name+".json",dict(product_differences=deltas,product_additions=additions,metadata_paths_excluded=["artifacts/","orders/","logs/"],unchanged_common_product_files=sum(product(f) and f in after and mapping[f]==after[f] for f in mapping)))
# Literal read-only git diff receipts, including the test addition.
for rev,name in [("0e1dea3","base"),("50b3418f","round1")]:
    diff=subprocess.check_output(["git","diff",rev,"--",*allowed],cwd=ROOT)
    (TOOL/("changes_vs_"+name+".patch")).write_bytes(diff)
identity=read("float32_identity.json")
assert identity["math_statement_bytes_identical"] and identity["no_casts_reached"]
# Constructor is exactly round one's constructor; only forward got D1.
a=ast.parse((TOOL/"nn.round1.py").read_text());b=ast.parse(text)
def constructor(t):
    cls=next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name=="GroupNorm")
    return next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=="__init__")
assert ast.dump(constructor(a))==ast.dump(constructor(b))
write("G1_scope.json",dict(passed=True,tracked_files=len(before),changed=changed,other_tracked_identical=len(before)-len(changed),readonly_files_unchanged=len(readonly),groupnorm_only=True,round1_constructor_unchanged=True,round1_forward_change="D1 only; original FP32 math sequence byte-identical",receipts=["tracked_before.json","tracked_after.json","base_sha256.json","round1_sha256.json","readonly_before.json","readonly_after.json","scope_vs_base.json","scope_vs_round1.json","changes_vs_base.patch","changes_vs_round1.patch","float32_identity.json"]))
pin=read("engine_pin.json")
module=ROOT/pin["module"]
assert sha(module)==pin["sha256"]
original=TOOL.parent/"engine_sources_after.json"
assert sha(original)==pin["source_map_sha256"]
assert sha(TOOL/"engine_sources_original.json")==pin["source_map_sha256"]
source_map=json.loads(original.read_text())
sources={f:sha(Path(f) if Path(f).is_absolute() else ROOT/f) for f in source_map}
assert source_map==sources
write("engine_sources_after.json",sources)
for arm in (TOOL/"arms").iterdir():
    p=arm/"tensor_cuda"/module.name
    assert p.is_symlink() and p.resolve()==module.resolve()
    assert sha(p)==pin["sha256"]
    if arm.name=="baseline":
        expected=read("base_sha256.json")
        for f in (arm/"tensor_cuda").rglob("*.py"):
            rel="tensor_cuda/tensor_cuda/"+str(f.relative_to(arm/"tensor_cuda"))
            assert sha(f)==expected[rel],rel
write("G3_pin.json",dict(passed=True,rebuild=False,module=pin["module"],sha256=sha(module),source_map_sha256=sha(original),source_count=len(sources),build_tree_unchanged=all(readonly[f]==readonly_after[f] for f in readonly if f.startswith("tensor_cuda/build-gn1/")),arms=20,baseline_package_matches="0e1dea3 SHA256 map",same_module_symlink_every_arm=True))
print(json.dumps(dict(G1="PASS",G3="PASS",tracked=len(before),unchanged=len(before)-len(changed),readonly=len(readonly),source_inputs=len(sources))))
