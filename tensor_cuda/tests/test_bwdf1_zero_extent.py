"""BWD-F1 (orders/BWDF1_ZERO_EXTENT_AND_AUDIT_FIXES.md): zero extents in broadcasting, the dispatch guard, reduce_to,
and the core H3 messages. Every test here runs on the CPU with CUDA_VISIBLE_DEVICES empty (host tensors, ENG3).

G1 (Z2 refusals): through the public API, every refusal of 0 against an extent above 1, in each operand position
and at ranks 1-4 (and across ranks), raises RuntimeError naming both shapes, with no CUDA call (ENG3's LD_PRELOAD
counter, tests/support/cudart_count.c, compiled at test time; positive control: a device tensor makes CUDA calls).
Ops: every binary op in artifacts/bwdf1/CENSUS.json (add, sub, mul, div, maximum, minimum, gt, ge, lt, le, eq),
where (cond / x / y position), masked_fill (mask / a), expand, einsum, axpy_; dtypes float32, float16, bfloat16.
Z2 0-against-1: each op returns NumPy's shape (np.broadcast_shapes), with no CUDA call (the output is empty, so no
launch); Build B launches over a non-empty output instead (m01 kills this test).
Z2 dispatch guard (m02): the host-only block of kernels.cu between BWDF1-HOST-BEGIN and BWDF1-HOST-END is compiled
with the host C++ compiler as its exact text and driven directly: an operand with zero elements under an output with
elements is refused, naming the operand and both shapes. A source check shows each dispatch (ew_binary, compare,
where_nd) calls it before its output allocation. No public path reaches the guard once the shape rule is right.
Z3 (m03, CPU part): the gradient of a [4, 1] operand under a [4, 0] output reaches reduce_sum (without a device its
launch fails with the no-device CUDA error). The CPU reshape failure ("element count mismatch") belongs to m03 on
Build E's corrected forward (BWD-F2 R10, review BWDF1R1-F10): Build B never reaches this backward on the CPU, its
forward already launches over the max-rule output [4, 1] and fails there. Build B's old `> 1` predicate is shown by
its source and by the G3 GPU line, not by this row. The GPU lines (G3) check the zeros themselves.
H3 (core): attach_vjp's value error and matmul backward's first refusal name the operand(s) and shapes.

Environment: BWDF1_PKG_PARENT (the directory holding the tensor_cuda package; default this tests directory's
parent) and BWDF1_KERNELS_CU (default ../src/kernels.cu) select the build and the source under test (mutant runs).

Prior art: NumPy's broadcasting rules (NumPy documentation, "Broadcasting"; np.broadcast_shapes is the reference);
ENG3's LD_PRELOAD + dlsym call counter (ltrace-style interposition); PyTorch's sum_to_size for the reduce_to rule;
DeMillo, Lipton and Sayward (1978), mutation testing (these tests are written to kill m01-m04 in
artifacts/bwdf1/MUTANTS.json). Ours: the case census, the exact-text host harness of the guard.
"""
import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_PARENT = os.environ.get("BWDF1_PKG_PARENT", os.path.dirname(HERE))
KERNELS_CU = os.environ.get("BWDF1_KERNELS_CU", os.path.join(os.path.dirname(HERE), "src", "kernels.cu"))
SHIM = os.path.join(HERE, "support", "cudart_count.c")
sys.path.insert(0, PKG_PARENT)
import tensor_cuda as tc  # noqa: E402

DTYPES = ("float32", "float16", "bfloat16")
BIN_OPS = ("add", "sub", "mul", "div", "maximum", "minimum", "gt", "ge", "lt", "le", "eq")

# 0 against an extent above 1: (shape with the zero, shape with the larger extent), each at ranks 1-4 and across
# ranks; every pair is also run with the operands swapped (both positions).
REFUSE_PAIRS = [
    ((0,), (3,)),
    ((4, 0), (4, 3)),
    ((0, 3), (2, 3)),
    ((2, 0, 5), (2, 3, 5)),
    ((2, 3, 0), (1, 1, 4)),
    ((0, 1, 1, 1), (2, 1, 1, 1)),
    ((1, 2, 3, 0), (5,)),
    ((0,), (4, 3)),
    ((3, 0), (2, 1, 5)),
]
# 0 against 1 (NumPy: 0), and the other zero classes that NumPy accepts.
ZERO_PAIRS = [
    ((4, 0), (4, 1)),
    ((0,), (1,)),
    ((2, 0, 3), (2, 1, 3)),
    ((1, 0), (3, 1)),
    ((0, 1, 4), (1, 5, 1)),
    ((0,), (4, 1)),
    ((4, 0), (4, 0)),
    ((2, 1, 0), (1,)),
]


def _shape_text(s):
    return "[" + ", ".join(str(int(e)) for e in s) + "]"


def refuse_cases():
    cases = []
    for z, big in REFUSE_PAIRS:
        for first, second in ((z, big), (big, z)):
            for dt in DTYPES:
                for op in BIN_OPS:
                    cases.append({"id": f"{op}-{dt}-{first}-{second}", "op": op, "dtype": dt,
                                  "shapes": [list(first), list(second)], "names": [first, second]})
                cases.append({"id": f"masked_fill_mask-{dt}-{first}-{second}", "op": "masked_fill", "dtype": dt,
                              "shapes": [list(first), list(second)], "names": [first, second]})
                cases.append({"id": f"expand-{dt}-{first}-{second}", "op": "expand", "dtype": dt,
                              "shapes": [list(first), list(second)], "names": [first, second]})
                cases.append({"id": f"axpy_-{dt}-{first}-{second}", "op": "axpy_", "dtype": dt,
                              "shapes": [list(first), list(second)], "names": [first, second]})
            for pos in ("cond", "x", "y"):  # where: the zero (or the large extent) in each operand position
                other = second
                shapes = {"cond": list(other), "x": list(other), "y": list(other)}
                shapes[pos] = list(first)
                cases.append({"id": f"where_{pos}-float32-{first}-{second}", "op": "where", "dtype": "float32",
                              "shapes": [shapes["cond"], shapes["x"], shapes["y"]], "names": [first, second]})
    # einsum: index j has extents 0 and 3 (each operand position); 2-D and 3-D terms
    for eq, sa, sb in (("ij,jk->ik", (2, 0), (3, 4)), ("ij,jk->ik", (2, 3), (0, 4)),
                       ("bij,bjk->bik", (2, 1, 0), (2, 3, 4)), ("bij,bjk->bik", (0, 1, 2), (3, 2, 4))):
        cases.append({"id": f"einsum-float32-{eq}-{sa}-{sb}", "op": "einsum", "dtype": "float32", "eq": eq,
                      "shapes": [list(sa), list(sb)], "names": [sa, sb]})
    return cases


def axpy_size_cases():
    """BWD-F2 R10 (BWDF1R1-F10): axpy_'s destination-size refusals, labelled apart from the zero-against-larger
    broadcast refusals (their shapes ARE broadcast-compatible: an update with fewer elements than param, which only a
    zero extent in `other` can give)."""
    return [{"id": f"axpy_fewer-float32-{p}-{o}", "op": "axpy_", "dtype": "float32", "shapes": [list(p), list(o)],
             "names": [p, o], "fewer": True}
            for p, o in (((4, 1), (4, 0)), ((1,), (0,)), ((2, 1, 3), (2, 0, 3)))]


def zero_cases():
    cases = []
    for a, b in ZERO_PAIRS:
        # BWD-F2 R10 (review BWDF1R1-F10): an equal pair is generated once (both directions were the same case, and
        # its 39 IDs were duplicated).
        for first, second in (((a, b), (b, a)) if a != b else ((a, b),)):
            for dt in DTYPES:
                for op in BIN_OPS:
                    cases.append({"id": f"{op}-{dt}-{first}-{second}", "op": op, "dtype": dt,
                                  "shapes": [list(first), list(second)]})
                cases.append({"id": f"masked_fill-{dt}-{first}-{second}", "op": "masked_fill", "dtype": dt,
                              "shapes": [list(first), list(second)]})
            for pos in ("cond", "x", "y"):
                shapes = {"cond": list(second), "x": list(second), "y": list(second)}
                shapes[pos] = list(first)
                cases.append({"id": f"where_{pos}-float32-{first}-{second}", "op": "where", "dtype": "float32",
                              "shapes": [shapes["cond"], shapes["x"], shapes["y"]]})
    for src, target in (((4, 1), (4, 0)), ((1,), (0,)), ((1, 3), (0, 3)), ((2, 1, 1), (2, 0, 5))):
        cases.append({"id": f"expand-float32-{src}-{target}", "op": "expand", "dtype": "float32",
                      "shapes": [list(target), list(src)]})
    return cases


REFUSE = refuse_cases()
AXPY_SIZE = axpy_size_cases()
ZERO = zero_cases()

# The child runs every case with the counter preloaded and prints one JSON object. The op table is shared text.
_CHILD = r'''
import ctypes, json, os, sys
sys.path.insert(0, os.environ["BWDF1_PKG_PARENT"])
import numpy as np
import tensor_cuda as tc
L = ctypes.CDLL(None)
L.eng3_cudart_calls.restype = ctypes.c_ulong
cases = json.loads(open(os.environ["BWDF1_CASES"]).read())
C = tc._C

def host(shape, dt, rg=False, fill=1.0):
    return tc.tensor(np.full(shape, fill, np.float32), device="cpu", dtype=dt, requires_grad=rg)

def run(c):
    dt, s = c["dtype"], c["shapes"]
    op = c["op"]
    if op in ("add", "sub", "mul", "div"):
        a, b = host(s[0], dt), host(s[1], dt)
        return {"add": lambda: a + b, "sub": lambda: a - b, "mul": lambda: a * b, "div": lambda: a / b}[op]()
    if op in ("maximum", "minimum", "gt", "ge", "lt", "le", "eq"):
        a, b = host(s[0], dt), host(s[1], dt)
        return getattr(a, op)(b)
    if op == "where":
        return tc.where(host(s[0], dt), host(s[1], dt), host(s[2], dt))
    if op == "masked_fill":   # shapes [mask, a]
        return host(s[1], dt).masked_fill(host(s[0], dt), 2.0)
    if op == "expand":        # shapes [target, source]
        return host(s[1], dt).expand(list(s[0]))
    if op == "einsum":
        return tc.einsum(c["eq"], host(s[0], dt), host(s[1], dt))
    if op == "axpy_":         # shapes [param, other]
        p = host(s[0], dt)
        C.axpy_(p, host(s[1], dt), 0.5)
        return p
    if op == "attach_vjp_value":
        return tc.attach_vjp([0.0], [host([3], dt, True)], lambda dy, m: (dy,))
    if op == "matmul_backward_host":
        a, b = host(s[0], dt, True), host(s[1], dt, True)
        y = tc.matmul(a, b)
        y.backward()
        return y
    raise ValueError(op)

out = {}
base = L.eng3_cudart_calls()
for c in cases:
    before = L.eng3_cudart_calls()
    try:
        r = run(c)
        res = {"raised": None, "shape": list(r.shape), "dtype": str(r.dtype)}
    except BaseException as e:  # noqa: BLE001
        res = {"raised": type(e).__name__, "message": str(e)}
    res["calls"] = L.eng3_cudart_calls() - before
    out[c["id"]] = res
case_calls = L.eng3_cudart_calls() - base
try:
    tc.tensor([1.0], device="cuda")   # positive control: a device tensor
    control = "no error"
except RuntimeError as e:
    control = str(e)
print(json.dumps({"results": out, "case_calls": case_calls,
                  "control_calls": L.eng3_cudart_calls() - base - case_calls, "control": control}))
'''


def _run_child(tmp_path_factory, cases, tag):
    compiler = next((p for n in ("cc", "gcc", "clang") if (p := shutil.which(n))), None)
    if compiler is None:
        pytest.skip("no C compiler for the CUDA-call counter shim")
    d = tmp_path_factory.mktemp("bwdf1_" + tag)
    shim = d / "cudart_count.so"
    cmd = [compiler, "-shared", "-fPIC", "-O2", "-o", str(shim), SHIM, "-ldl"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, (cmd, r.stderr)
    cf = d / "cases.json"
    cf.write_text(json.dumps(cases))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_PRELOAD=str(shim), BWDF1_PKG_PARENT=PKG_PARENT,
               BWDF1_CASES=str(cf))
    p = subprocess.run([sys.executable, "-B", "-P", "-c", _CHILD], env=env, capture_output=True, text=True,
                       timeout=600)
    assert p.returncode == 0, p.stderr[-3000:]
    res = json.loads(p.stdout.strip().splitlines()[-1])
    assert res["control_calls"] > 0, res   # the counter sees CUDA calls in this process
    rec = os.environ.get("BWDF1_RECORD_DIR")
    if rec:
        os.makedirs(rec, exist_ok=True)
        with open(os.path.join(rec, f"g1_{tag}.json"), "w") as f:
            json.dump({"cases": cases, **res}, f, indent=1)
    return res


@pytest.fixture(scope="module")
def refuse_results(tmp_path_factory):
    return _run_child(tmp_path_factory, REFUSE + AXPY_SIZE, "refuse")


@pytest.fixture(scope="module")
def zero_results(tmp_path_factory):
    return _run_child(tmp_path_factory, ZERO, "zero")


@pytest.mark.parametrize("case", REFUSE, ids=[c["id"] for c in REFUSE])
def test_cpu_g1_refusal_names_both_shapes_no_cuda_call(refuse_results, case):
    r = refuse_results["results"][case["id"]]
    assert r["raised"] == "RuntimeError", r
    msg = r["message"]
    for s in case["names"]:
        assert _shape_text(s) in msg, (msg, case["names"])
    assert msg.startswith("shapes not broadcastable"), msg
    assert r["calls"] == 0, r   # refused before any CUDA call


@pytest.mark.parametrize("case", AXPY_SIZE, ids=[c["id"] for c in AXPY_SIZE])
def test_cpu_g1_axpy_size_refusal(refuse_results, case):
    """axpy_ refuses an update with fewer elements than param (compatible shapes; BWD-F2 R10 labels these apart)."""
    r = refuse_results["results"][case["id"]]
    assert r["raised"] == "RuntimeError", r
    for s in case["names"]:
        assert _shape_text(s) in r["message"], (r["message"], case["names"])
    assert "fewer elements than param" in r["message"], r["message"]
    assert r["calls"] == 0, r


@pytest.mark.parametrize("case", ZERO, ids=[c["id"] for c in ZERO])
def test_cpu_g1_zero_against_one_numpy_shape(zero_results, case):
    r = zero_results["results"][case["id"]]
    assert r["raised"] is None, r
    want = list(np.broadcast_shapes(*[tuple(s) for s in case["shapes"]]))
    assert r["shape"] == want, (r, want)
    assert r["dtype"] == case["dtype"], r
    assert 0 in r["shape"], r
    assert r["calls"] == 0, r   # an empty output: no launch, no CUDA call


# ------------------------------------------------------------------------------------- the guard's host harness
_HARNESS = r'''
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>
using Shape = std::vector<int64_t>;
namespace h {
%BLOCK%
}
static void guard(const char* label, const char* op, Shape out, std::initializer_list<const char*> names,
                  std::initializer_list<Shape> shapes) {
  std::vector<Shape> keep(shapes);
  std::vector<const Shape*> ptrs;
  for (auto& s : keep) ptrs.push_back(&s);
  try {
    if (ptrs.size() == 2) h::require_readable_operands(op, out, names, {ptrs[0], ptrs[1]});
    else h::require_readable_operands(op, out, names, {ptrs[0], ptrs[1], ptrs[2]});
    std::printf("{\"case\": \"%s\", \"raised\": null}\n", label);
  } catch (const std::runtime_error& e) {
    std::printf("{\"case\": \"%s\", \"raised\": \"%s\"}\n", label, e.what());
  }
}
static void bshape(const char* label, Shape a, Shape b) {
  try {
    Shape o = h::broadcast_shape(a, b);
    std::printf("{\"case\": \"%s\", \"shape\": %s}\n", label, h::bcast_shape_text(o).c_str());
  } catch (const std::runtime_error& e) {
    std::printf("{\"case\": \"%s\", \"raised\": \"%s\"}\n", label, e.what());
  }
}
int main() {
  guard("ew_binary-a-empty", "ew_binary", {4, 1}, {"a", "b"}, {{4, 0}, {4, 1}});
  guard("ew_binary-b-empty", "ew_binary", {4, 1}, {"a", "b"}, {{4, 1}, {4, 0}});
  guard("compare-b-empty-rank1", "compare", {3}, {"a", "b"}, {{3}, {0}});
  guard("where-x-empty", "where", {2, 3}, {"cond", "x", "y"}, {{2, 3}, {0}, {2, 3}});
  guard("where-y-empty", "where", {2, 3}, {"cond", "x", "y"}, {{2, 3}, {2, 3}, {2, 0, 3}});
  guard("ok-empty-out", "ew_binary", {4, 0}, {"a", "b"}, {{4, 0}, {4, 1}});
  guard("ok-scalar", "ew_binary", {}, {"a", "b"}, {{}, {}});
  guard("ok-nonempty", "compare", {4, 3}, {"a", "b"}, {{4, 3}, {4, 1}});
%BSHAPE%
  return 0;
}
'''
BSHAPE_PAIRS = [((4, 0), (4, 1)), ((4, 1), (4, 0)), ((0,), (1,)), ((4, 0), (4, 3)), ((4, 3), (4, 0)), ((3,), (4, 3)),
                ((4, 1), (4, 3)), ((2, 1, 0), (1,)), ((5, 1, 3), (1, 4, 1)), ((), (2, 0)), ((2, 3), (3, 2))]


def _host_block():
    src = open(KERNELS_CU).read()
    m = re.search(r"// BWDF1-HOST-BEGIN.*?\n(.*?)// BWDF1-HOST-END", src, re.S)
    assert m, f"BWDF1-HOST block not found in {KERNELS_CU}"
    return m.group(1)


@pytest.fixture(scope="module")
def harness(tmp_path_factory):
    compiler = next((p for n in ("c++", "g++", "clang++") if (p := shutil.which(n))), None)
    if compiler is None:
        pytest.skip("no C++ compiler for the host harness")
    d = tmp_path_factory.mktemp("bwdf1_harness")
    bs = "\n".join(f'  bshape("{a}|{b}", {{{", ".join(map(str, a))}}}, {{{", ".join(map(str, b))}}});'
                   for a, b in BSHAPE_PAIRS)
    (d / "h.cpp").write_text(_HARNESS.replace("%BLOCK%", _host_block()).replace("%BSHAPE%", bs))
    r = subprocess.run([compiler, "-std=c++17", "-O1", "-o", str(d / "h"), str(d / "h.cpp")], capture_output=True,
                       text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-3000:]
    p = subprocess.run([str(d / "h")], capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
    return {row["case"]: row for row in map(json.loads, p.stdout.splitlines())}


@pytest.mark.parametrize("label,operand,shape,out", [
    ("ew_binary-a-empty", "a", "[4, 0]", "[4, 1]"),
    ("ew_binary-b-empty", "b", "[4, 0]", "[4, 1]"),
    ("compare-b-empty-rank1", "b", "[0]", "[3]"),
    ("where-x-empty", "x", "[0]", "[2, 3]"),
    ("where-y-empty", "y", "[2, 0, 3]", "[2, 3]"),
])
def test_cpu_guard_refuses_empty_operand_under_nonempty_output(harness, label, operand, shape, out):
    msg = harness[label]["raised"]
    assert msg is not None, harness[label]
    assert f"operand {operand} {shape} has no elements but the output {out} has elements" in msg, msg


@pytest.mark.parametrize("label", ["ok-empty-out", "ok-scalar", "ok-nonempty"])
def test_cpu_guard_accepts_readable_operands(harness, label):
    assert harness[label]["raised"] is None, harness[label]


@pytest.mark.parametrize("a,b", BSHAPE_PAIRS, ids=[f"{a}|{b}" for a, b in BSHAPE_PAIRS])
def test_cpu_broadcast_shape_text_equals_numpy(harness, a, b):
    row = harness[f"{a}|{b}"]
    try:
        want = list(np.broadcast_shapes(a, b))
    except ValueError:
        want = None
    if want is None:
        assert row.get("raised", "").startswith("shapes not broadcastable: "), row
        assert _shape_text(a) in row["raised"] and _shape_text(b) in row["raised"], row
    else:
        assert row.get("shape") == want, (row, want)


@pytest.mark.parametrize("fn", ["NDArray ew_binary(", "NDArray compare(", "NDArray where_nd("])
def test_cpu_each_dispatch_calls_the_guard_before_allocating(fn):
    src = open(KERNELS_CU).read()
    start = src.index(fn)
    body = src[start:src.index("\n}\n", start)]
    guard_at = body.find("require_readable_operands(")
    alloc_at = body.find("NDArray out(")
    assert guard_at != -1 and alloc_at != -1 and guard_at < alloc_at, (fn, guard_at, alloc_at)


# --------------------------------------------------------------------------------------------------- Z3 (CPU)
def test_cpu_z3_zero_axis_gradient_reaches_the_sum():
    """x [4, 0] + y [4, 1]: y's gradient sums the zero-extent axis. Without a device that sum's launch fails with the
    no-device CUDA error. m03 (the `> 1` rule on Build E's corrected forward) skips the axis and raises reshape's
    element-count mismatch on the host; Build B fails earlier, in its forward launch (BWD-F2 R10, BWDF1R1-F10). The
    zeros themselves are checked on the GPU (G3)."""
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        pytest.skip("CPU lane only (CUDA_VISIBLE_DEVICES must be empty)")
    x = tc.tensor(np.zeros((4, 0), np.float32), device="cpu", requires_grad=True)
    y = tc.tensor(np.ones((4, 1), np.float32), device="cpu", requires_grad=True)
    z = x + y
    assert tuple(z.shape) == (4, 0)
    with pytest.raises(RuntimeError) as e:
        z.backward()
    assert "element count mismatch" not in str(e.value), str(e.value)
    assert "CUDA error at reduce" in str(e.value), str(e.value)


# --------------------------------------------------------------------------------------------------- H3 (core)
H3_CASES = [
    {"id": "attach_vjp_value", "op": "attach_vjp_value", "dtype": "float32", "shapes": [],
     "want": "attach_vjp: value must be a TensorCUDA Tensor, got list"},
    {"id": "matmul_backward_host_k0", "op": "matmul_backward_host", "dtype": "float32", "shapes": [[2, 0], [0, 3]],
     "want": "matmul backward: expected same-device CUDA fp32 matrices (left operand [2, 3] float32 on cpu, "
             "right operand [0, 3] float32 on cpu)"},
    {"id": "matmul_backward_host_n0", "op": "matmul_backward_host", "dtype": "float32", "shapes": [[2, 3], [3, 0]],
     "want": "matmul backward: expected same-device CUDA fp32 matrices (left operand [2, 0] float32 on cpu, "
             "right operand [3, 0] float32 on cpu)"},
]


@pytest.fixture(scope="module")
def h3_results(tmp_path_factory):
    return _run_child(tmp_path_factory, H3_CASES, "h3")


@pytest.mark.parametrize("case", H3_CASES, ids=[c["id"] for c in H3_CASES])
def test_cpu_h3_core_messages_name_operands(h3_results, case):
    r = h3_results["results"][case["id"]]
    assert r["raised"] == "RuntimeError", r
    assert r["message"] == case["want"], r
