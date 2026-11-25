"""BWD-F1 L60 (order) and Amendment 1 L60: artifacts/bwdf1/BOUNDARY.json.

  python3 -B -P scripts/bwdf1_boundary.py

For each refusal this round adds: the call, the refusal, each term of its predicate with the G1 rows that trigger it,
Build E's outcome (artifacts/bwdf1/g1/records_E) and Build B's behaviour for the same operands: on the CPU where
Build B makes no CUDA call before it (artifacts/bwdf1/g1/records_B; the copy-op rows run here on Build B, each in its
own process because some of them crash Build B's process), otherwise the source line or the GPU line that records it.
Prior art: none needed. Ours: the table.
"""
import collections
import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tensor_cuda/tests"))
os.environ.setdefault("BWDF1_PKG_PARENT", "/tmp/bwdf1_src_E/tensor_cuda")
import test_bwdf1_copy_ops as C  # noqa: E402  (the row table only; its module import loads no engine)

REC = ROOT / "artifacts/bwdf1/g1"


def load(arm, name):
    return json.loads((REC / f"records_{arm}" / name).read_text())


def b_copy_rows():
    """Each copy-op row on Build B in its own process (no counter; Build B may crash)."""
    out = {}
    for rid, op, setup, call, _ in C.ROWS:
        code = ("import sys, numpy as np\nsys.path.insert(0, '/tmp/bwdf1_src_B/tensor_cuda')\nimport tensor_cuda as tc\n"
                "def H(shape, dt='float32'):\n    return tc.tensor(np.ones(shape, np.uint8 if dt == 'uint8' else "
                "np.float32), device='cpu', dtype=dt)\n"
                "def C(shape, dt='float32'):\n    return tc.tensor(np.zeros(shape, np.float32), device='cuda', dtype=dt)\n"
                f"{setup}\ntry:\n    with tc.no_grad():\n        r = {call}\n    print('RETURNED', getattr(r, 'shape', r))\n"
                "except BaseException as e:\n    print('RAISED', type(e).__name__, str(e)[:200])\n")
        p = subprocess.run([sys.executable, "-B", "-P", "-c", code], capture_output=True, text=True, timeout=60,
                           env=dict(os.environ, CUDA_VISIBLE_DEVICES="", TMPDIR="/tmp/bwdf1_tmp"))
        line = (p.stdout.strip().splitlines() or [""])[0]
        out[rid] = {"exit": p.returncode, "stdout": line,
                    "stderr_tail": p.stderr.strip().splitlines()[-1][:200] if p.returncode not in (0,) and p.stderr.strip()
                    else ""}
    return out


def main():
    eref, bref = load("E", "g1_refuse.json"), load("B", "g1_refuse.json")
    ezero, bzero = load("E", "g1_zero.json"), load("B", "g1_zero.json")
    ets, bts = load("E", "g1_ts.json"), load("B", "g1_ts.json")
    eh3, bh3 = load("E", "g1_h3.json"), load("B", "g1_h3.json")
    ecopy = load("E", "g1_copy_ops.json")
    bcopy = b_copy_rows()

    def summarize(ids, rec):
        c = collections.Counter()
        for i in ids:
            r = rec["results"].get(i, {})
            msg = r.get("message") or ""
            key = (r.get("raised"), msg.split(" (")[0].split(":")[0] + (":" + msg.split(":")[1][:40] if ":" in msg
                                                                         else ""), r.get("calls", 0) > 0)
            c[str(key)] += 1
        return dict(c)

    by_op = collections.defaultdict(list)
    for c in C.ROWS:
        by_op[c[1]].append(c[0])
    refuse_by_op = collections.defaultdict(list)
    for c in eref["cases"]:
        refuse_by_op[c["op"]].append(c["id"])
    doc = {"order": "BWD-F1 L60 + Amendment 1 L60", "z2": [], "z2_guard": {}, "z3": {}, "h1": {}, "h3": {}, "c2": []}
    for op, ids in sorted(refuse_by_op.items()):
        site = {"add": "ew_binary", "maximum": "ew_binary", "gt": "compare", "where": "where_nd (broadcast_shape3)",
                "masked_fill": "ops::masked_fill -> tc::broadcast_shapes, before NDArray::full",
                "expand": "broadcast_to, before NDArray::zeros", "einsum": "functional.py einsum per-index check",
                "axpy_": "axpy_ (broadcast_shape, then the element-count check)"}.get(op, "ew_binary / compare")
        doc["z2"].append({
            "call": f"public {op}", "refusal": "RuntimeError 'shapes not broadcastable: <both shapes> ...' (axpy_ "
                                               "fewer-elements rows: '... fewer elements than param')",
            "site": site,
            "predicate_terms": ["two aligned extents differ and neither is 1 (0 against an extent above 1, either "
                                "operand, ranks 1-4 and across ranks)"] + (["axpy_: numel(broadcast(param, other)) < "
                                                                             "numel(param)"] if op == "axpy_" else []),
            "rows": len(ids), "row_ids_file": "artifacts/bwdf1/g1/records_E/g1_refuse.json",
            "build_E": summarize(ids, eref), "build_B_cpu": summarize(ids, bref)})
    doc["z2_zero_against_one"] = {
        "call": "every census op on 0 against 1 (NumPy: 0)", "build_E": summarize(list(ezero["results"]), ezero),
        "build_B_cpu": summarize(list(bzero["results"]), bzero),
        "build_B_note": "on the CPU Build B allocates the max-rule output and launches (the no-device CUDA error); "
                        "on a device it reads the empty operand: GPU lines g2_direct_B, g2_f01_B, g2_f03_B (memcheck)"}
    doc["z2_guard"] = {"call": "require_readable_operands (ew_binary, compare, where_nd), host block BWDF1-HOST",
                       "refusal": "RuntimeError '<op>: operand <name> <shape> has no elements but the output <shape> "
                                  "has elements; refused before any launch'",
                       "predicate_terms": ["some operand has an extent 0", "the output has no extent 0"],
                       "rows": "tensor_cuda/tests/test_bwdf1_zero_extent.py::test_cpu_guard_* (host harness, 8 rows)",
                       "build_B": "no guard (source: kernels.cu:1148-1166 at fbd335ae7 launches over out.numel())"}
    doc["z3"] = {"call": "reduce_to (accumulate_grad, expand backward, matmul shared-B)",
                 "change": "sums axes with target 1 and source != 1 (was > 1); not a refusal",
                 "rows": "test_cpu_z3_zero_axis_gradient_reaches_the_sum; GPU lines g3_B / g3_E / g3_m03_reduce_gt1",
                 "build_B": "reshape: element count mismatch (CPU, host; recorded in records_B via the B test run)"}
    doc["h1"] = {"call": "tracked ops with frozen inputs and out= requiring grad (tc_signal _recording)",
                 "refusal": "RuntimeError 'tc_signal: <op>: out cannot require grad'",
                 "predicate_terms": ["grad mode on", "no input requires grad", "an out= TensorCUDA destination requires grad"],
                 "build_E": {k: v for k, v in ets["results"].items() if k.endswith("grad_out_rg")},
                 "build_B_cpu": {k: v for k, v in bts["results"].items() if k.endswith("grad_out_rg")}}
    doc["h3"] = {"build_E": eh3["results"], "build_B_cpu": bh3["results"],
                 "tc_signal_backward_entry": {"build_E": {k: v for k, v in ets["results"].items() if "-" in k and
                                                          not k.split("-")[-1] in ("rg", "frozen")}}}
    for op, ids in by_op.items():
        doc["c2"].append({"op": op, "rows": [{"id": i, "terms_from_row": next(c[4] for c in C.ROWS if c[0] == i),
                                              "build_E": ecopy["results"][i], "build_B_cpu_own_process": bcopy[i]}
                                             for i in ids]})
    (ROOT / "artifacts/bwdf1/BOUNDARY.json").write_text(json.dumps(doc, indent=1))
    crashed = {k: v for k, v in bcopy.items() if v["exit"] != 0}
    print("BOUNDARY.json written; Build B copy rows that crashed their process:", json.dumps(crashed, indent=0)[:1500])


if __name__ == "__main__":
    main()
