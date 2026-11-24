"""ENG4-F3 L48: drive each staged line's post-device steps (readback, judges, verdicts) on constructed returns, on the
CPU, and write artifacts/eng4/f3/POST_DEVICE_STEPS.json (each step with its expected and observed terminal label).

  CUDA_VISIBLE_DEVICES= PYTHONPATH=<F3 package parent> python3 -B -P scripts/eng4_f3_postdevice.py

Nothing here runs a staged line or its step function, and nothing reads, sets or clears CUDA_VISIBLE_DEVICES (L58).
The line scripts are imported only for their pure functions (judge_cell, verdict, judge, junit_outcome, lease_guard)
and for g3z's run_cells on host operands (the real F3 readback of every zero-size cell, no device).

Prior art: negative-control testing of a comparator (each verdict path driven by a constructed input that must reach
it); ENG4-F2's eng4_f2_postdevice.py (this repo). No other prior art known to me.
"""
import copy
import importlib.util
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = "/tmp/eng4f3_tmp/postdevice"
STEPS = []


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


g3z = load("eng4_f3_g3z", "scripts/eng4_f3_g3z.py")
pend = load("eng4_f3_pending", "scripts/eng4_f3_pending.py")
line = load("eng4_f3_gpu_line", "scripts/eng4_f3_gpu_line.py")


def step(sid, ln, what, expected, observed, detail=None):
    STEPS.append({"step": sid, "line": ln, "what": what, "expected": expected, "observed": observed,
                  "ok": expected == observed, "detail": detail})
    print(("ok " if expected == observed else "MISMATCH ") + sid, expected, observed, flush=True)


def g3z_steps():
    import tensor_cuda as tc
    host_rows = g3z.run_cells(tc, device="cpu")  # the real readback of every zero-size cell, on host operands
    v = sorted({g3z.judge_cell(r)[0] for r in host_rows})
    step("g3z_host_readback_all_zero_cells", "g3z_F3", "run_cells on host operands (F3's K = 0 and empty paths, "
         "each dtype) then judge_cell", "PASS", v[0] if len(v) == 1 else str(v), {"cells": len(host_rows)})
    k0 = next(r for r in host_rows if "_K0-" in r["id"] and r["dtype"] == "float16")
    emp = next(r for r in host_rows if "_batch0-" in r["id"] and r["dtype"] == "bfloat16")
    for sid, mutate, want in [
        ("g3z_k0_nonzero", lambda r: r.update(hex="0100" + r["hex"][4:]), "FAIL"),
        ("g3z_empty_with_bytes", None, "FAIL"),
        ("g3z_wrong_shape", lambda r: r.update(shape=r["shape"][:-1] + [r["shape"][-1] + 1]), "FAIL"),
        ("g3z_wrong_tensor_dtype", lambda r: r.update(tensor_dtype="float32"), "FAIL"),
        ("g3z_raised", lambda r: r.update(raised={"type": "RuntimeError", "message": "x"}), "FAIL")]:
        r = copy.deepcopy(emp if sid == "g3z_empty_with_bytes" else k0)
        if sid == "g3z_empty_with_bytes":
            r["hex"] = "00000000"
        else:
            mutate(r)
        step(sid, "g3z_F3", "judge_cell on a constructed bad return", want, g3z.judge_cell(r)[0])
    bwd0 = g3z.bwd0
    grads = []
    for c in bwd0.cases():
        host = np.float16 if c["dtype"] == "float16" else np.float32
        gb = np.zeros(c["b"], host)
        grads.append(dict(c, child_exit=0,
                          a_grad={"shape": c["a"], "dtype": c["dtype"], "host_dtype": str(np.dtype(host)),
                                  "elements": 0, "hex": ""},
                          b_grad={"shape": c["b"], "dtype": c["dtype"], "host_dtype": str(np.dtype(host)),
                                  "elements": int(gb.size), "hex": gb.tobytes().hex()}))
    for sid, cells, gr, want in [("g3z_verdict_all_pass", host_rows, grads, "PASS"),
                                 ("g3z_verdict_one_grad_bit", host_rows, None, "FAIL"),
                                 ("g3z_verdict_missing_cell", host_rows[:-1], grads, "FAIL")]:
        if gr is None:
            gr = copy.deepcopy(grads)
            raw = bytearray(bytes.fromhex(gr[7]["b_grad"]["hex"]))
            raw[-1] ^= 1
            gr[7]["b_grad"]["hex"] = raw.hex()
        d = os.path.join(TMP, sid)
        os.makedirs(d, exist_ok=True)
        json.dump({"module_sha256": "constructed", "cells": cells, "gradients": gr},
                  open(os.path.join(d, "g3z.json"), "w"))
        g3z.verdict(d)
        step(sid, "g3z_F3", "verdict on a constructed g3z.json", want,
             json.load(open(os.path.join(d, "VERDICT.json")))["verdict"])


def pending_steps():
    oom = {"raised": {"type": "RuntimeError", "message": "cudaMalloc failed: out of memory"}}
    rows = {
        "F2_forward_raises_fill": {"arm": "F2", "variant": "forward", "make_pending": oom,
                                   "k0_matmul": {"raised": {"type": "RuntimeError",
                                                            "message": "CUDA error at matmul zero fill: out of memory"}}},
        "F3_forward_returns_zeros": {"arm": "F3", "variant": "forward", "make_pending": oom,
                                     "k0_matmul": {"returned": [2, 3]},
                                     "drain": {"raised": {"type": "RuntimeError",
                                                          "message": "CUDA error at sync: out of memory"}},
                                     "readback": {"returned": [[0, 0, 0], [0, 0, 0]]}},
        "F2_gradient_raises_fill": {"arm": "F2", "variant": "gradient", "make_pending": oom,
                                    "backward": {"raised": {"type": "RuntimeError", "message": "CUDA error at matmul "
                                                            "backward zero gradient: out of memory"}}},
        "F3_gradient_zero_b": {"arm": "F3", "variant": "gradient", "make_pending": oom,
                               "backward": {"returned": [0, 2, 4]},
                               "readback": {"returned": {"shape": [3, 4], "u32": [[0] * 4] * 3}}},
    }
    bad = {
        "F2_forward_returned": dict(rows["F2_forward_raises_fill"], k0_matmul={"returned": [2, 3]}),
        "F3_forward_raised_fill": dict(rows["F3_forward_returns_zeros"],
                                       k0_matmul=rows["F2_forward_raises_fill"]["k0_matmul"]),
        "F3_forward_nonzero": dict(rows["F3_forward_returns_zeros"], readback={"returned": [[0, 1, 0], [0, 0, 0]]}),
        "no_pending_made": dict(rows["F3_forward_returns_zeros"], make_pending={"returned": [524288, 524288]}),
        "F3_gradient_nonzero": dict(rows["F3_gradient_zero_b"],
                                    readback={"returned": {"shape": [3, 4], "u32": [[0, 0, 0, 7]] + [[0] * 4] * 2}}),
    }
    for k, r in rows.items():
        step(f"pending_judge_{k}", "pending_F2_F3", "judge on a constructed child record", "PASS", pend.judge(r)[0])
    for k, r in bad.items():
        step(f"pending_judge_{k}", "pending_F2_F3", "judge on a constructed bad child record", "FAIL",
             pend.judge(r)[0])
    for sid, recs, want in [("pending_verdict_pass", list(rows.values()), "PASS"),
                            ("pending_verdict_F3_raised", [rows["F2_forward_raises_fill"], bad["F3_forward_raised_fill"]],
                             "FAIL")]:
        d = os.path.join(TMP, sid)
        os.makedirs(d, exist_ok=True)
        json.dump({"rows": recs}, open(os.path.join(d, "pending.json"), "w"))
        pend.verdict(d)
        step(sid, "pending_F2_F3", "verdict on constructed records", want,
             json.load(open(os.path.join(d, "VERDICT.json")))["verdict"])


def own_steps():
    d = os.path.join(TMP, "own")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "j.xml")
    open(p, "w").write('<testsuites><testsuite><testcase name="a"/><testcase name="b"><failure message="x"/>'
                       '</testcase><testcase name="c"><skipped/></testcase></testsuite></testsuites>')
    step("own_junit_outcome", "own_F3", "junit_outcome counts on a constructed junit file",
         json.dumps({"passed": 1, "failed": 1, "skipped": 1}, sort_keys=True),
         json.dumps(line.junit_outcome(p), sort_keys=True))
    step("own_junit_missing", "own_F3", "junit_outcome when pytest wrote no file", "missing",
         line.junit_outcome(os.path.join(d, "none.xml")))
    step("lease_guard_outside_forge", "all F3 lines", "lease_guard() in this CPU process: the step would refuse",
         "False", str(line.lease_guard()))


def main():
    os.makedirs(TMP, exist_ok=True)
    g3z_steps()
    pending_steps()
    own_steps()
    out = os.path.join(ROOT, "artifacts/eng4/f3/POST_DEVICE_STEPS.json")
    rec = {"order": "ENG4-F3 L48", "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"), "steps": STEPS,
           "count": len(STEPS), "all_ok": all(s["ok"] for s in STEPS)}
    json.dump(rec, open(out, "w"), indent=1)
    print("POST_DEVICE_STEPS", len(STEPS), "all ok" if rec["all_ok"] else "MISMATCHES")
    return 0 if rec["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
