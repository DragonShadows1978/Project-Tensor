"""Single edit-site mutation registration and deterministic source transformation.
Prior art: DeMillo, Lipton and Sayward (1978); Offutt/Voas RIP model (1990s,
unverified — lead to check). A2 splits every independent site, including
same-line mean/variance expressions; all replacements are runnable expressions.
"""
import ast
import difflib
import json
import subprocess
from pathlib import Path

TOOL = Path(__file__).resolve().parent
PREFIX = "tensor_cuda/tests/test_groupnorm_gn1.py::"
F32 = "test_forward_values[float32-r3-n2-g2-default]"
F16 = "test_forward_values[float16-r3-n2-g2-default]"


def specs():
    rows = []
    def add(mid, name, old, new, witness=F32, equivalence=None):
        rows.append(dict(id=mid, name=name, old=old, new=new,
                         predicted_witness=PREFIX+witness, equivalence=equivalence,
                         expected_child_exit=0 if equivalence else 1, actual_gpu_killer=None))
    commit = (TOOL / "order_commit.txt").read_text().strip()
    a1 = json.loads(subprocess.check_output(
        ["git", "show", commit+":artifacts/gn1/a1/mutants.json"], cwd=TOOL.parents[2], text=True))
    for row in a1:
        mid = row["id"]
        if mid in ("M04", "M06", "M07", "M08", "M09", "M15", "M17", "M18"):
            continue
        old, new = row["edits"][0]
        add(mid, row["name"], old, new, row["predicted_witness"].split("::")[1])
    add("M04m", "wrong_mean_axis", "mean = g.mean([2], True)", "mean = g.mean([1], True)")
    add("M04v", "wrong_variance_axis", "var = g.var([2], True)", "var = g.var([1], True)")
    add("M06in", "interleaved_input", "x.reshape([N, self.G, (C // self.G) * math.prod(spatial)])",
        "x.reshape([N, C // self.G, self.G] + spatial).permute([0, 2, 1] + list(range(3, 3 + len(spatial)))).reshape([N, self.G, (C // self.G) * math.prod(spatial)])")
    add("M06out", "interleaved_output", "g = g.reshape([N, C] + spatial)",
        "g = g.reshape([N, self.G, C // self.G] + spatial).permute([0, 2, 1] + list(range(3, 3 + len(spatial)))).reshape([N, C] + spatial)")
    for mid, name, replacements in (
        ("M07", "weight_omitted", ("g + self.bias.reshape(wshape)", "g + bias.reshape(wshape)")),
        ("M08", "bias_omitted", ("g * self.weight.reshape(wshape)", "g * weight.reshape(wshape)")),
        ("M09", "affine_swapped", ("g * self.bias.reshape(wshape) + self.weight.reshape(wshape)", "g * bias.reshape(wshape) + weight.reshape(wshape)"))):
        add(mid+"fast", name+"_fast", "g * self.weight.reshape(wshape) + self.bias.reshape(wshape)", replacements[0])
        add(mid+"cast", name+"_cast", "g * weight.reshape(wshape) + bias.reshape(wshape)", replacements[1], F16)
    add("M15x", "input_not_upcast", "if dtype in (\"float16\", \"bfloat16\"):", "if False:", F16)
    for operand in ("weight", "bias"):
        expr = f"self.{operand} if self.{operand}.dtype == \"float32\" else self.{operand}.astype(\"float32\")"
        add("M15"+operand[0], "parameter_to_input_dtype_"+operand, expr,
            f"self.{operand} if self.{operand}.dtype == dtype else self.{operand}.astype(dtype)", F16)
        add("M17"+operand[0], "fp32_parameter_cast_omitted_"+operand, expr,
            f"self.{operand} if dtype == \"float32\" else ("+expr+")",
            "test_distinct_parameter_dtypes[forward-"+("w16-b32" if operand=="weight" else "w32-b16")+"]")
        add("M18"+operand[0], "parameter_cast_omitted_"+operand, expr, f"self.{operand}",
            "test_mixed_parameters[forward-half-float16]")
    add("V1", "bias_fast_check_dropped", "and self.bias.dtype == \"float32\"", "and True",
        "test_distinct_parameter_dtypes[forward-w32-b16]")
    add("V2", "weight_fast_check_dropped", "and self.weight.dtype == \"float32\"", "and True",
        "test_distinct_parameter_dtypes[forward-w16-b32]")
    add("V3", "upcast_non_fp32", "if dtype in (\"float16\", \"bfloat16\"):", "if dtype != \"float32\":",
        F16, "EQUIVALENCES.md#v3")
    for arg, short in (("num_groups", "g"), ("num_channels", "c")):
        for kind, old, new, witness in (
            ("bool", f"isinstance({arg}, bool)", "False", "true"),
            ("int", f"not isinstance({arg}, int)", "False", "float"),
            ("positive", f"{arg} <= 0", "False", "negative")):
            add("D5_"+short+"_"+kind, "remove_"+arg+"_"+kind, old, new,
                f"test_positive_integer_refusal[{witness}-{arg}]")
    add("D5_rank", "rank_check_removed", "if x.ndim < 2:", "if False:", "test_shape_refusal[rank1]")
    add("D5_channels", "channel_check_removed", "if x.shape[1] != self.C:", "if False:", "test_shape_refusal[channels]")
    for operand, expr in (("input", "x"), ("weight", "self.weight"), ("bias", "self.bias")):
        add("D5_"+operand+"_dtype", operand+"_dtype_check_removed",
            f"if {expr}.dtype not in (\"float32\", \"float16\", \"bfloat16\"):", "if False:",
            f"test_dtype_refusal[int64-{operand}]")
    return sorted(rows, key=lambda m:m["id"])


def mutate(candidate, row):
    lo, hi = candidate.index("class GroupNorm("), candidate.index("\n\nclass InstanceNorm2D")
    part = candidate[lo:hi]
    assert part.count(row["old"]) == 1, (row["id"], "nonunique edit")
    result = candidate[:lo] + part.replace(row["old"], row["new"], 1) + candidate[hi:]
    ast.parse(result)
    assert result != candidate
    return result


if __name__ == "__main__":
    print(json.dumps(specs(), indent=2))
