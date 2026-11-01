"""Stage A1 isolated packages and one-decision mutants, no engine build.
Prior art: round-one GN1 scaffolding; DeMillo/Lipton/Sayward (1978) mutation
operators; SHA256 NIST (2001). Unverified literature leads named in registration.
"""
from pathlib import Path
import ast
import difflib
import hashlib
import json
import os
import shutil
import subprocess

TOOL = Path(__file__).resolve().parent
ROOT = TOOL.parents[2]
ROUND = TOOL.parent
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_text() == text, f"existing different artifact: {path}"
        return
    with path.open("x") as stream:
        stream.write(text)

candidate = (ROOT / "tensor_cuda/tensor_cuda/nn.py").read_text()
for name in ("run_cpu_gate.py", "run_gpu_line.py", "run_gpu_tests.py", "start_gpu_run.py", "score_gpu.py", "forge_gpu_run_standin.py"):
    text = (ROUND / name).read_text().replace("TOOL.parents[1]", "TOOL.parents[2]")
    text = text.replace("resolve().parents[2]", "resolve().parents[3]")
    text = text.replace('root / "artifacts/gn1"', 'root / "artifacts/gn1/a1"')
    text = text.replace('"/tmp/gn1_cpu"', '"/tmp/gn1_a1_cpu"')
    text = text.replace('TMPDIR=str(runs / "tmp")', 'TMPDIR=os.environ["TMPDIR"]')
    text = text.replace('"GN1_GROUPNORM_FIX.md"', '"GN1_A1_HALF_PRECISION.md"')
    text = text.replace('orders/GN1_GROUPNORM_FIX.md', 'orders/GN1_A1_HALF_PRECISION.md')
    text = text.replace('if len(data["collected_ids"]) != 5:', 'if sorted(data["collected_ids"]) != sorted((TOOL / "ids_audit.txt").read_text().splitlines()):')
    text = text.replace('audit did not run exactly five CPU oracle IDs', 'audit CPU oracle census mismatch')
    text = text.replace('str(output / "pytest_tmp")', 'str(Path(os.environ["TMPDIR"]) / (label + "_pytest"))')
    # A run root must stay in the authorized artifact tree; all scratch in /tmp.
    needle = 'runs = Path(os.environ["RUNS"]).resolve()'
    if needle in text and name != "score_gpu.py":
        text = text.replace(needle, needle + '\n    assert runs.is_relative_to(TOOL) and runs != TOOL, "RUNS must be under a1"' if name != "start_gpu_run.py" else needle + '\nassert runs.is_relative_to(TOOL) and runs != TOOL, "RUNS must be under a1"')
    write(TOOL / name, text)
    if name == "forge_gpu_run_standin.py":
        (TOOL / name).chmod(0o755)
# Root expression in old CPU runner is different from the shared TOOL convention.
assert 'parents[3]' in (TOOL / "run_cpu_gate.py").read_text()
pin = json.loads((ROUND / "engine_pin.json").read_text())
write(TOOL / "engine_pin.json", json.dumps(pin, indent=2) + "\n")
module = ROOT / pin["module"]
assert sha(module) == pin["sha256"]
source_map = ROUND / "engine_sources_after.json"
assert sha(source_map) == pin["source_map_sha256"]
write(TOOL / "engine_sources_original.json", source_map.read_text())

mutants = json.loads((ROUND / "mutants.json").read_text())
affine = "g * self.weight.reshape(wshape) + self.bias.reshape(wshape)"
local_affine = "g * weight.reshape(wshape) + bias.reshape(wshape)"
param_block = '        weight = self.weight if self.weight.dtype == "float32" else self.weight.astype("float32")\n        bias = self.bias if self.bias.dtype == "float32" else self.bias.astype("float32")'
new_specs = [
    ("M15", "input_dtype_compute", [
        ('if dtype in ("float16", "bfloat16"):', 'if False and dtype in ("float16", "bfloat16"):'),
        (param_block, param_block.replace('"float32"', 'dtype'))],
     "test_low_variance[backward-var1e-6-half-float16]"),
    ("M16", "output_not_cast_back", [
        ('return output if output.dtype == dtype else output.astype(dtype)', 'return output')],
     "test_mixed_parameters[forward-default-float16]"),
    ("M17", "parameters_not_cast_for_fp32_input", [
        (param_block, '        if dtype == "float32":\n            weight, bias = self.weight, self.bias\n        else:\n' + "\n".join('    ' + line for line in param_block.splitlines()))],
     "test_fp32_input_half_parameters[forward]"),
    ("M18", "input_upcast_without_parameter_casts", [
        (param_block, '        weight, bias = self.weight, self.bias')],
     "test_mixed_parameters[forward-half-float16]"),
]
for row in mutants:
    row["edits"] = [[row.pop("old"), row.pop("new")]]
    if row["id"] in ("M07", "M08", "M09"):
        replacement = {"M07":"g + self.bias.reshape(wshape)", "M08":"g * self.weight.reshape(wshape)", "M09":"g * self.bias.reshape(wshape) + self.weight.reshape(wshape)"}[row["id"]]
        row["edits"] = [[affine, replacement], [local_affine, replacement.replace("self.", "")]]
for mid, name, edits, witness in new_specs:
    mutants.append(dict(id=mid, name=name, edits=edits, expected_child_exit=1,
                        predicted_witness="tensor_cuda/tests/test_groupnorm_gn1.py::"+witness))

packages = sorted((ROOT / "tensor_cuda/tensor_cuda").rglob("*.py"))
packages = [p for p in packages if "__pycache__" not in p.parts]
assert len(packages) == 8
validation=[]
for arm in ["baseline", "candidate"] + [m["id"] for m in mutants]:
    package = TOOL / "arms" / arm / "tensor_cuda"
    package.mkdir(parents=True, exist_ok=True)
    for source in packages:
        rel = source.relative_to(ROOT / "tensor_cuda/tensor_cuda")
        target = package / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if arm == "baseline":
            data = subprocess.check_output(["git", "show", "0e1dea3:tensor_cuda/tensor_cuda/" + str(rel)])
        else:
            data = source.read_bytes()
        target.write_bytes(data)
    if not (package / module.name).exists():
        (package / module.name).symlink_to(module)
    assert (package / module.name).resolve() == module.resolve()
    if arm.startswith("M"):
        row = next(m for m in mutants if m["id"] == arm)
        lo, hi = candidate.index("class GroupNorm("), candidate.index("\n\nclass InstanceNorm2D")
        text = candidate[lo:hi]
        for old, new in row["edits"]:
            assert text.count(old) == 1, (arm, old, text.count(old))
            text = text.replace(old, new, 1)
        text = candidate[:lo] + text + candidate[hi:]
        ast.parse(text)
        assert text[:text.index("class GroupNorm(")] == candidate[:candidate.index("class GroupNorm(")]
        assert text[text.index("\n\nclass InstanceNorm2D"):]==candidate[candidate.index("\n\nclass InstanceNorm2D"):]
        (package / "nn.py").write_text(text)
        target = TOOL / "mutants" / arm
        write(target / "nn.py", text)
        diff = "".join(difflib.unified_diff(candidate.splitlines(True), text.splitlines(True), fromfile="candidate/nn.py", tofile=arm+"/nn.py"))
        write(target / "nn.diff", diff)
        row.update(sha256=sha(package / "nn.py"), actual_killer=None, status="NOT RUN; lead G5")
        validation.append(dict(id=arm, syntax=True, groupnorm_only=True, semantic_decisions=1, replacements=len(row["edits"])))
write(TOOL / "mutants.json", json.dumps(mutants, indent=2)+"\n")
write(TOOL / "mutant_validation.json", json.dumps(validation, indent=2)+"\n")
md = "# A1 mutant registration\n\nPrior art: DeMillo, Lipton and Sayward (1978), unverified — lead to check. Re-derived on D1. One semantic decision per mutant, including both affine branches or dtype plumbing where necessary. No GPU executions or actual kills claimed. Shape/dtype errors during a test call are valid structural kills; device, collection or setup failures and timeouts are invalid. Every actual killer must be recorded by the parent after child exit. No equivalences claimed.\n\n| ID | Decision | Child / parent expected | Predicted witness (not observed) |\n|---|---|---|---|\n"
for row in mutants:
    md += f'| {row["id"]} | {row["name"]} | 1 / 0 | `{row["predicted_witness"]}` |\n'
write(TOOL / "MUTANTS.md", md)
# Census from the actual test AST, not a stale hand-maintained file list.
files=[]
for file in sorted((ROOT / "tensor_cuda/tests").glob("test_*.py")):
    tree=ast.parse(file.read_text())
    imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
    if any((isinstance(n,ast.ImportFrom) and (n.module=="tensor_cuda" and any(a.name=="nn" for a in n.names) or n.module=="tensor_cuda.nn")) or (isinstance(n,ast.Import) and any(a.name=="tensor_cuda.nn" for a in n.names)) for n in imports):
        files.append(str(file.relative_to(ROOT)))
assert files == json.loads((ROUND / "nn_import_test_files.json").read_text())
write(TOOL / "nn_import_test_files.json", json.dumps(files, indent=2)+"\n")
lines=json.loads((ROUND / "gpu_lines.json").read_text())
for n in range(15,19):
    lines.append(dict(label=f"M{n:02d}",arm=f"M{n:02d}",suite="mutation",expected_child_exit=1,inner=180,maximum=240,outer=270,expected_parent_exit=0))
write(TOOL / "gpu_lines.json",json.dumps(lines,indent=2)+"\n")
script = (ROUND / "STAGED_GPU_COMMANDS.sh").read_text().replace("18 labels", "22 labels").replace("all 18 labels", "all 22 labels")
script = script.replace('export TMPDIR="$RUNS/tmp" PYTHONDONTWRITEBYTECODE=1', 'export TMPDIR="${TMPDIR:-/tmp/gn1_a1_staged}" PYTHONDONTWRITEBYTECODE=1\n[[ "$TMPDIR" == /tmp/gn1_a1_* ]] || { echo "TMPDIR must be /tmp/gn1_a1_*" >&2; exit 64; }')
start = script.index('  M14)')
end = script.index('  *)', start)
block=script[start:end]
script = script[:end]+"".join(block.replace("M14",f"M{n:02d}") for n in range(15,19))+script[end:]
write(TOOL / "STAGED_GPU_COMMANDS.sh",script)
print(json.dumps(dict(arms=20,mutants=len(mutants),staged_lines=len(lines),regression_files=len(files),pin=pin["sha256"])))
