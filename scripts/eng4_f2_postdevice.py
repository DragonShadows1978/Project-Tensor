"""ENG4-F2 L48: drive every post-device step of g3_F2, bwd0_B, bwd0_F2 and mut_m10_f2 (readback, comparators,
verdict code) on constructed returns, on the CPU, and write artifacts/eng4/f2/POST_DEVICE_STEPS.json (each step with
its expected and observed terminal label).

  CUDA_VISIBLE_DEVICES= PYTHONPATH=<F2 package parent> python3 -B -P scripts/eng4_f2_postdevice.py

Nothing here runs a staged line or its step function, and nothing reads, sets or clears CUDA_VISIBLE_DEVICES (L58):
the comparators run as the lines' after-commands do (scripts/eng4_f1_g3_compare.py, unchanged, as a child process),
the bwd0 verdict and recorder run in process (record_case on host operands only), and the mutant kill classifier is
imported from scripts/eng4_f2_gpu_line.py as a function (classify_kill, junit_case).

Prior art: no prior art known to me beyond negative-control testing of a comparator (each verdict path driven by a
constructed input that must reach it).
"""
import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = "/tmp/eng4_tmp/f2_postdevice"


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


g3 = load("eng4_g3", "scripts/eng4_g3.py")
bwd0 = load("eng4_f2_bwd0", "scripts/eng4_f2_bwd0.py")
line = load("eng4_f2_gpu_line", "scripts/eng4_f2_gpu_line.py")
STEPS = []


def step(sid, line_name, what, expected, observed, detail=None):
    STEPS.append({"step": sid, "line": line_name, "what": what, "expected": expected, "observed": observed,
                  "ok": expected == observed, "detail": detail})
    print(("ok " if expected == observed else "MISMATCH ") + sid, expected, observed, flush=True)


# ---------------------------------------------------------------- readback (the engine's .numpy() per dtype)
def readback():
    import tensor_cuda as tc
    for dtype in ("float32", "float16", "bfloat16"):
        for shape in ((3, 17, 19), (0, 17, 19), (3, 0, 19), (4, 64, 96)):
            t = tc.tensor(np.ones(shape, np.float32), dtype=dtype, device="cpu")
            x = t.numpy()
            want = np.float16 if dtype == "float16" else np.float32
            ok = x.dtype == want and x.shape == shape
            step(f"readback_{dtype}_{'x'.join(map(str, shape))}", "g3_F2/bwd0_*",
                 "host tensor .numpy(): float16 stays float16, float32 and bfloat16 come back as float32 (the "
                 "widening the comparators expect), shape kept, empty included", "PASS", "PASS" if ok else "FAIL",
                 {"dtype": str(x.dtype), "shape": list(x.shape)})


# ---------------------------------------------------------------- g3 compare (eng4_f1_g3_compare.py, unchanged)
def g3_dirs(tag, mutate=None):
    """Constructed B and F2 run directories with the real grid's cells: nonzero cells get seeded nonconstant values
    (identical on both sides), zero-size cells get NumPy's result on F2 and nonzero garbage on B (B's K = 0 output
    was unwritten memory)."""
    base = os.path.join(TMP, "g3", tag)
    out = {}
    for side in ("B", "F2"):
        d = os.path.join(base, side)
        os.makedirs(d, exist_ok=True)
        cells, arrs = [], {}
        for c in g3.cells():
            a_s, b_s = c["a"], c["b"]
            b_log = tuple(b_s[:-2]) + ((b_s[-1], b_s[-2]) if c["trans_b"] else tuple(b_s[-2:]))
            if c["class"] in ("a", "b"):
                shape = np.matmul(np.zeros(a_s, np.float32), np.zeros(b_log, np.float32)).shape
            else:
                shape = tuple(a_s[:-1]) + (b_log[-1],)
            host = np.float16 if c["dtype"] == "float16" else np.float32
            rng = np.random.default_rng(c["seed"])
            if not c["zero_size"]:
                x = rng.standard_normal(shape).astype(host)
            elif side == "B":
                x = rng.standard_normal(shape).astype(host)  # unwritten memory stand-in
            else:
                x = np.zeros(shape, host)
            arrs[c["id"]] = np.ascontiguousarray(x)
            row = dict(c, a=list(a_s), b=list(b_s), a_sha256="in", b_sha256="in", tensor_dtype=c["dtype"],
                       out_shape=list(shape), host_dtype=str(x.dtype))
            if c["zero_size"]:
                row["numpy_matmul_shape"] = list(np.matmul(np.zeros(a_s, np.float32),
                                                           np.zeros(b_log, np.float32)).shape)
            cells.append(row)
        if mutate and side == "F2":
            mutate(cells, arrs)
        np.savez(os.path.join(d, "outputs.npz"), **arrs)
        json.dump({"module_sha256": f"constructed-{side}", "cells": cells}, open(os.path.join(d, "manifest.json"), "w"))
        out[side] = d
    return out


def run_g3_compare(tag, mutate=None):
    d = g3_dirs(tag, mutate)
    res = os.path.join(TMP, "g3", tag, "compare.json")
    p = subprocess.run([sys.executable, "-B", "-P", os.path.join(ROOT, "scripts/eng4_f1_g3_compare.py"), d["B"],
                        d["F2"], res], capture_output=True, text=True, timeout=300)
    rec = json.load(open(res))
    return rec["verdict"], rec["failed"], p.returncode


def g3_steps():
    largest = "a_medium-bfloat16-T-a0.5"

    def flip(cells, arrs):
        x = arrs[largest].copy()
        x.reshape(-1).view(np.uint32)[0] ^= 1
        arrs[largest] = x

    def k0_nonzero(cells, arrs):
        x = arrs["b_K0-float16"].copy()
        x.reshape(-1)[0] = np.float16(1.0)
        arrs["b_K0-float16"] = x

    def empty_shape(cells, arrs):
        arrs["c_batch0-bfloat16"] = np.zeros((0, 17, 18), np.float32)

    def constant(cells, arrs):
        arrs["b_rank2-float32-N-a1.0"] = np.full_like(arrs["b_rank2-float32-N-a1.0"], 2.0)

    def dtype(cells, arrs):
        arrs["a_small-float16-N-a1.0"] = arrs["a_small-float16-N-a1.0"].astype(np.float32)

    cases = [("g3_identical_with_numpy_zero_cells", None, "PASS", []),
             ("g3_bitflip_largest_cell", flip, "FAIL", [largest]),
             ("g3_k0_cell_nonzero", k0_nonzero, "FAIL", ["b_K0-float16"]),
             ("g3_empty_cell_wrong_shape", empty_shape, "FAIL", ["c_batch0-bfloat16"]),
             ("g3_nonzero_cell_constant_L19", constant, "FAIL", ["b_rank2-float32-N-a1.0"]),
             ("g3_host_dtype_mismatch", dtype, "FAIL", ["a_small-float16-N-a1.0"])]
    for sid, mut, want, want_failed in cases:
        verdict, failed, rc = run_g3_compare(sid, mut)
        observed = verdict if (verdict == "PASS" and not failed) or failed == want_failed else f"{verdict}:{failed}"
        step(sid, "g3_F2", "scripts/eng4_f1_g3_compare.py on constructed B/F2 runs (123 cells: zero-size, empty and "
             "the largest cells included)", want, observed, {"failed": failed, "exit": rc})


# ---------------------------------------------------------------- bwd0 recorder and verdict
def bwd0_steps():
    import tensor_cuda as tc
    rows = []
    for c in bwd0.cases():
        r = dict(c, child_exit=0)
        r.update(bwd0.record_case(tc, c, "cpu"))
        rows.append(r)
    host_lp = [r for r in rows if r["dtype"] != "float32"]
    host_f32 = [r for r in rows if r["dtype"] == "float32"]
    v = sorted({bwd0.judge_row(r)[0] for r in host_lp})
    step("bwd0_record_and_judge_host_f16_bf16", "bwd0_F2", "record_case on host float16/bfloat16 (the real F2 "
         "backward, readback of both gradients) then judge_row: every case zeros of B's shape, A empty",
         "PASS", v[0] if len(v) == 1 else str(v), {"cases": len(host_lp)})
    v = sorted({bwd0.judge_row(r)[0] for r in host_f32})
    step("bwd0_record_and_judge_host_f32_raise_path", "bwd0_F2", "record_case on host float32: the fp32 backward "
         "needs CUDA operands, so the raise branch of the recorder and judge_row's FAIL on a raise are driven",
         "FAIL", v[0] if len(v) == 1 else str(v),
         {"cases": len(host_f32), "message": host_f32[0].get("backward_raised")})

    def verdict_of(tag, rows_, mode):
        d = os.path.join(TMP, "bwd0", tag)
        os.makedirs(d, exist_ok=True)
        json.dump({"module_sha256": "constructed", "rows": rows_}, open(os.path.join(d, "bwd0.json"), "w"))
        bwd0.verdict(d, mode)
        return json.load(open(os.path.join(d, "VERDICT.json")))["verdict"]

    good = []
    for c in bwd0.cases():  # constructed device returns: zeros of B's shape and dtype, A empty
        host = np.float16 if c["dtype"] == "float16" else np.float32
        gb = np.zeros(c["b"], host)
        good.append(dict(c, child_exit=0, out_shape=c["a"][:-1] + [c["b"][-2] if c["trans_b"] else c["b"][-1]],
                         a_grad={"shape": c["a"], "dtype": c["dtype"], "host_dtype": str(np.dtype(host)),
                                 "elements": 0, "hex": "", "sha256": hashlib.sha256(b"").hexdigest()},
                         b_grad={"shape": c["b"], "dtype": c["dtype"], "host_dtype": str(np.dtype(host)),
                                 "elements": int(gb.size), "hex": gb.tobytes().hex(),
                                 "sha256": hashlib.sha256(gb.tobytes()).hexdigest()}))
    step("bwd0_verdict_F2_all_zero", "bwd0_F2", "verdict F2 on 24 constructed zero-gradient returns (largest B "
         "included)", "PASS", verdict_of("all_zero", good, "F2"))
    bad = copy.deepcopy(good)
    raw = bytearray(bytes.fromhex(bad[5]["b_grad"]["hex"]))
    raw[0] ^= 1
    bad[5]["b_grad"]["hex"] = raw.hex()
    step("bwd0_verdict_F2_one_bit", "bwd0_F2", "one nonzero bit in one B gradient", "FAIL",
         verdict_of("one_bit", bad, "F2"))
    bad = copy.deepcopy(good)
    for k in ("a_grad", "b_grad"):
        bad[0].pop(k)
    bad[0]["backward_raised"] = {"type": "RuntimeError", "message": "reshape: element count mismatch"}
    step("bwd0_verdict_F2_raised", "bwd0_F2", "a backward that raised (Build B / m10 shape)", "FAIL",
         verdict_of("raised", bad, "F2"))
    bad = copy.deepcopy(good)
    bad[1]["b_grad"]["shape"] = [3, 4, 1]
    step("bwd0_verdict_F2_wrong_shape", "bwd0_F2", "B gradient with the wrong shape", "FAIL",
         verdict_of("wrong_shape", bad, "F2"))
    bad = copy.deepcopy(good)
    bad[2]["b_grad"]["dtype"] = "float32" if bad[2]["dtype"] != "float32" else "float16"
    step("bwd0_verdict_F2_wrong_dtype", "bwd0_F2", "B gradient with the wrong dtype", "FAIL",
         verdict_of("wrong_dtype", bad, "F2"))
    bad = copy.deepcopy(good)
    bad[3]["b_grad"].update(elements=0, hex="")
    step("bwd0_verdict_F2_empty_b_grad", "bwd0_F2", "an empty B gradient", "FAIL", verdict_of("empty_b", bad, "F2"))
    bad = copy.deepcopy(good)
    bad[4]["child_exit"] = -11
    step("bwd0_verdict_F2_child_died", "bwd0_F2", "a child process that died", "FAIL",
         verdict_of("died", bad, "F2"))
    mixed = copy.deepcopy(good)
    mixed[0].pop("b_grad")
    mixed[0]["backward_raised"] = {"type": "RuntimeError", "message": "reshape: element count mismatch"}
    step("bwd0_verdict_B_records", "bwd0_B", "verdict B records every case (raised or gradients), no gate",
         "RECORDED", verdict_of("B_mode", mixed, "B"))


# ---------------------------------------------------------------- mut_m10_f2 kill classification
def mut_steps():
    d = os.path.join(TMP, "mut")
    os.makedirs(d, exist_ok=True)
    mid = "m10_f2_empty_batch_raise"
    junits = {
        "failed_named": '<testsuites><testsuite><testcase name="t"><failure message="Failed: zero-gradient check: a '
                        'shared B ... raised: reshape: element count mismatch"/></testcase></testsuite></testsuites>',
        "failed_other": '<testsuites><testsuite><testcase name="t"><failure message="AssertionError: other"/>'
                        '</testcase></testsuite></testsuites>',
        "passed": '<testsuites><testsuite><testcase name="t"/></testsuite></testsuites>',
        "skipped": '<testsuites><testsuite><testcase name="t"><skipped message="CUDA disabled"/></testcase>'
                   '</testsuite></testsuites>',
    }
    expect = {"failed_named": (0, True), "failed_other": (0, False), "passed": (0, False), "skipped": (0, False)}
    for k, xml in junits.items():
        p = os.path.join(d, f"{k}.xml")
        open(p, "w").write(xml)
        outcome, msg = line.junit_case(p)
        row = {"pytest_exit": 1 if outcome == "failed" else 0, "outcome": outcome, "failure_message": msg}
        r = line.classify_kill(row, f"ENG4-MUTANT {mid} executed\n", mid, ["zero-gradient check"])
        step(f"mut_m10_classify_{k}", "mut_m10_f2", "junit_case + classify_kill on a constructed junit file",
             "KILLED" if expect[k][1] else "SURVIVED", "KILLED" if r["killed"] else "SURVIVED",
             {"outcome": outcome, "marker_seen": r["marker_seen"]})
    outcome, msg = line.junit_case(os.path.join(d, "missing.xml"))
    r = line.classify_kill({"pytest_exit": -11, "outcome": outcome, "failure_message": msg}, "", mid,
                           ["zero-gradient check"])
    step("mut_m10_classify_process_died", "mut_m10_f2", "no junit and a negative exit (a signal)", "KILLED",
         "KILLED" if r["killed"] else "SURVIVED", {"outcome": outcome})
    step("mut_m10_lease_guard_outside_forge", "all F2 lines", "lease_guard() called in this CPU process (no "
         "forge-gpu-run ancestor): the step refuses before touching the environment", "False",
         str(line.lease_guard()))


def main():
    os.makedirs(TMP, exist_ok=True)
    readback()
    g3_steps()
    bwd0_steps()
    mut_steps()
    out = os.path.join(ROOT, "artifacts/eng4/f2/POST_DEVICE_STEPS.json")
    rec = {"order": "ENG4-F2 L48", "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "steps": STEPS, "all_ok": all(s["ok"] for s in STEPS), "count": len(STEPS)}
    json.dump(rec, open(out, "w"), indent=1)
    print("POST_DEVICE_STEPS", len(STEPS), "all ok" if rec["all_ok"] else "MISMATCHES")
    return 0 if rec["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
