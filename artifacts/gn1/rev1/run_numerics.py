#!/usr/bin/env python3
"""GN1-REV1 numerics driver: oracle independence (Q2), honest error vs
tolerance (Q3), mutant error vs witness tolerance (Q3/Q4).

CPU only. Imports the registered test file READ-ONLY for its input
generators and TOLERANCES; every value/bound here is the reviewer's own.
Prior art: see gn_emulate.py (Wu and He 2018; Higham 2002 ch. 3; IEEE 754;
DeMillo, Lipton and Sayward 1978 for mutation analysis).
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import gn_emulate as E  # noqa: E402

spec = importlib.util.spec_from_file_location("gn1_tests", ROOT / "tensor_cuda/tests/test_groupnorm_gn1.py")
T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
TOL = T.TOLERANCES
rep = T.represented


def cases():
    """Every registered value-checked ID, with its represented operands."""
    out = []
    for dtype in T.DTYPES:
        for shape in T.SHAPES:
            for G in (1, 2, 6):
                for eps in (1e-5, 0.125):
                    v, w, b, p = T._inputs(shape)
                    lab = f"{dtype}-r{len(shape)}-n{shape[0]}-g{G}-" + ("default" if eps == 1e-5 else "eps0125")
                    for kind in ("forward_values", "backward_values"):
                        out.append(dict(id=f"test_{kind}[{lab}]", x=rep(v, dtype), w=rep(w, dtype), b=rep(b, dtype),
                                        p=rep(p, dtype), G=G, eps=eps, xd=dtype, wd=dtype, bd=dtype,
                                        backward=kind.startswith("backward")))
    for dtype in T.DTYPES:
        eps = 2. ** -12 if dtype == "float32" else 0.25
        amp = 2. ** -6 if dtype == "float32" else 0.5
        v = np.tile([-amp, amp], 24).reshape(2, 6, 4) + np.repeat([0., .25, .5], 2).reshape(1, 6, 1)
        _, w, b, p = T._inputs(v.shape)
        for bw in (False, True):
            out.append(dict(id=f"test_near_epsilon[{'backward' if bw else 'forward'}-{dtype}]", x=rep(v, dtype),
                            w=rep(w, dtype), b=rep(b, dtype), p=rep(p, dtype), G=3, eps=eps, xd=dtype, wd=dtype,
                            bd=dtype, backward=bw))
    for dtype in ("float16", "bfloat16"):
        for conf in ("default", "half"):
            v, w, b, p = T._inputs((2, 6, 4))
            if conf == "default":
                w, b, wd = np.ones(6), np.zeros(6), "float32"
            else:
                w, b, wd = rep(w, "float16"), rep(b, "float16"), "float16"
            for bw in (False, True):
                out.append(dict(id=f"test_mixed_parameters[{'backward' if bw else 'forward'}-{conf}-{dtype}]",
                                x=rep(v, dtype), w=w, b=b, p=rep(p, dtype), G=3, eps=1e-5, xd=dtype, wd=wd, bd=wd,
                                backward=bw))
            for var in (1e-4, 1e-6):
                v, w0, b0, p = T.low_variance_inputs(var)
                if conf == "default":
                    w, b, wd = np.ones(6), np.zeros(6), "float32"
                else:
                    w, b, wd = rep(w0, "float16"), rep(b0, "float16"), "float16"
                for bw in (False, True):
                    vid = "var1e-4" if var == 1e-4 else "var1e-6"
                    out.append(dict(id=f"test_low_variance[{'backward' if bw else 'forward'}-{vid}-{conf}-{dtype}]",
                                    x=rep(v, dtype), w=w, b=b, p=rep(p, dtype), G=3, eps=1e-5, xd=dtype, wd=wd,
                                    bd=wd, backward=bw))
    v, w, b, p = T._inputs((2, 6, 4))
    for bw in (False, True):
        out.append(dict(id=f"test_fp32_input_half_parameters[{'backward' if bw else 'forward'}]", x=rep(v, "float32"),
                        w=rep(w, "float16"), b=rep(b, "float16"), p=rep(p, "float32"), G=3, eps=1e-5, xd="float32",
                        wd="float16", bd="float16", backward=bw))
    return out


def ratio(actual, ref, atol, rtol):
    actual = np.asarray(actual, np.float64); ref = np.asarray(ref, np.float64)
    if not np.all(np.isfinite(actual)):
        return float("inf")
    return float(np.max(np.abs(actual - ref) / (atol + rtol * np.abs(ref))))


def main():
    res = dict(oracle={}, honest={}, bound={}, mutants={}, extra_mutants={})
    allc = cases()
    res["n_value_ids"] = len(allc)
    # ---------------- Q2: oracle independence and agreement
    worst_oracle = 0.0
    for cs in allc:
        y1, (dx1, dw1, db1) = T.groupnorm_reference(cs["x"], cs["G"], cs["w"], cs["b"], cs["eps"], cs["p"])
        y2, dx2, dw2, db2 = E.ref_loops(cs["x"], cs["G"], cs["w"], cs["b"], cs["eps"], cs["p"])
        for a, bb in ((y1, y2), (dx1, dx2), (dw1, dw2), (db1, db2)):
            worst_oracle = max(worst_oracle, float(np.max(np.abs(a - bb) / (1e-300 + 1 + np.abs(bb)))))
    res["oracle"]["max_rel_diff_test_oracle_vs_reviewer_loops"] = worst_oracle
    # own central-difference check of the reviewer's loop VJP on one case
    cs = [c for c in allc if c["id"] == "test_backward_values[float32-r4-n2-g2-default]"][0]
    x, w, b, p = (np.array(cs[k]) for k in "xwbp")
    _, dx, dw, db = E.ref_loops(x, 2, w, b, 1e-5, p)
    h = 2. ** -20
    fd = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        xp = x.copy(); xm = x.copy(); xp[idx] += h; xm[idx] -= h
        fd[idx] = (np.sum(E.ref_loops(xp, 2, w, b, 1e-5, p)[0] * p) - np.sum(E.ref_loops(xm, 2, w, b, 1e-5, p)[0] * p)) / (2 * h)
    res["oracle"]["reviewer_fd_vs_loop_vjp_max_abs"] = float(np.max(np.abs(fd - dx)))
    res["oracle"]["fd_case"] = cs["id"]
    # one hand-checkable case: x=[[1,3]] G=1 eps=0 -> mean 2, var 1, xhat [-1,1]; p=[1,0], w=[1,1]
    y, dxh, dwh, dbh = E.ref_loops(np.array([[1., 3.]]), 1, np.array([1., 1.]), np.array([0., 0.]), 0.0, np.array([[1., 0.]]))
    res["oracle"]["hand_case"] = dict(y=y.tolist(), dx=dxh.tolist(), dw=dwh.tolist(), db=dbh.tolist(),
                                      expected=dict(y=[[-1.0, 1.0]], dx=[[0.0, 0.0]], dw=[-1.0, 0.0], db=[1.0, 0.0]))
    # ---------------- Q3: honest error (engine-graph emulation, float32 compute, 3 orders) and running bound
    worst = {}
    for cs in allc:
        dt = cs["xd"]
        ref = T.groupnorm_reference(cs["x"], cs["G"], cs["w"], cs["b"], cs["eps"], cs["p"])
        y_ref, grads = ref
        r_emp = 0.0
        for order in ("seq", "rev", "pair"):
            y, dx, dw, db, _ = E.engine_graph(cs["x"], cs["G"], cs["w"], cs["b"], cs["eps"], cs["p"], "float32", order,
                                              out_dt=dt, w_dt=cs["wd"], b_dt=cs["bd"])
            fa, fr = TOL[dt]["forward"]
            r = ratio(y, y_ref, fa, fr)
            if cs["backward"]:
                for got, exp, ld in ((dx, grads[0], dt), (dw, grads[1], cs["wd"]), (db, grads[2], cs["bd"])):
                    ba, br = TOL[ld]["backward"]
                    r = max(r, ratio(got, exp, ba, br))
            r_emp = max(r_emp, r)
        # running bound of the float32 graph against the FLOAT32 budget A+R|ref| (A1's premise)
        B = E.running_bound(cs["x"], cs["G"], cs["w"], cs["b"], cs["eps"], cs["p"])
        rb = float(np.max(B["ey"] / (3e-5 + 3e-5 * np.abs(y_ref))))
        if cs["backward"]:
            rb = max(rb, float(np.max(B["edx"] / (3e-4 + 3e-4 * np.abs(grads[0])))),
                     float(np.max(B["edw"] / (3e-4 + 3e-4 * np.abs(grads[1])))),
                     float(np.max(B["edb"] / (3e-4 + 3e-4 * np.abs(grads[2])))))
        # float32-only emulated error against the float32 budget (A1's premise), worst of 3 orders
        r32 = 0.0
        for order in ("seq", "rev", "pair"):
            y, dx, dw, db, _ = E.engine_graph(cs["x"], cs["G"], cs["w"], cs["b"], cs["eps"], cs["p"], "float32", order,
                                              out_dt="float32", w_dt="float32", b_dt="float32")
            r32 = max(r32, ratio(y, y_ref, 3e-5, 3e-5))
            if cs["backward"]:
                r32 = max(r32, ratio(dx, grads[0], 3e-4, 3e-4), ratio(dw, grads[1], 3e-4, 3e-4), ratio(db, grads[2], 3e-4, 3e-4))
        worst[cs["id"]] = (r_emp, rb, r32)
    top = sorted(worst.items(), key=lambda kv: -kv[1][1])[:12]
    res["honest"]["max_emulated_ratio"] = max(v[0] for v in worst.values())
    res["honest"]["argmax_emulated"] = max(worst.items(), key=lambda kv: kv[1][0])[0]
    res["bound"]["max_running_bound_ratio_vs_fp32_budget"] = top[0][1][1]
    res["bound"]["top12_by_bound"] = [(k, v[0], v[1], v[2]) for k, v in top]
    res["honest"]["max_fp32_only_emulated_ratio_vs_fp32_budget"] = max(v[2] for v in worst.values())
    res["honest"]["argmax_fp32_only"] = max(worst.items(), key=lambda kv: kv[1][2])[0]
    res["bound"]["columns"] = "id, emulated ratio vs destination tolerance, running-bound ratio vs fp32 budget, fp32-only emulated ratio vs fp32 budget"
    # ---------------- mutants on their registered witnesses (exact arithmetic, ct=float64)
    wit = json.loads((ROOT / "artifacts/gn1/a1/mutants.json").read_text())
    wmap = {}
    for m in (wit if isinstance(wit, list) else wit.get("mutants", [])):
        mid = m.get("id") or m.get("mutant")
        wmap[mid] = m.get("predicted_witness") or m.get("witness")
    byid = {("tensor_cuda/tests/test_groupnorm_gn1.py::" + c["id"]): c for c in allc}
    for mid in ("M01", "M02", "M03", "M04", "M05", "M12", "M13"):
        wid = wmap.get(mid)
        cs = byid.get(wid)
        if cs is None:
            res["mutants"][mid] = dict(witness=wid, note="witness not a value ID")
            continue
        y_ref, grads = T.groupnorm_reference(cs["x"], cs["G"], cs["w"], cs["b"], cs["eps"], cs["p"])
        try:
            y, dx, dw, db, _ = E.engine_graph(cs["x"], cs["G"], cs["w"], cs["b"], cs["eps"], cs["p"], "float64",
                                              mutant=mid, out_dt=cs["xd"], w_dt=cs["wd"], b_dt=cs["bd"])
            fa, fr = TOL[cs["xd"]]["forward"]
            res["mutants"][mid] = dict(witness=wid, forward_ratio=ratio(y, y_ref, fa, fr),
                                       max_abs_err=float(np.max(np.abs(y - y_ref))),
                                       honest_ratio_on_witness=worst[cs["id"]][0])
        except Exception as exc:  # structural kill
            res["mutants"][mid] = dict(witness=wid, raises=f"{type(exc).__name__}: {exc}")
    # M15 (compute in input dtype) on its witness, float16 arithmetic
    wid = wmap.get("M15"); cs = byid[wid]
    y_ref, grads = T.groupnorm_reference(cs["x"], cs["G"], cs["w"], cs["b"], cs["eps"], cs["p"])
    y, dx, dw, db, big = E.engine_graph(cs["x"], cs["G"], cs["w"], cs["b"], cs["eps"], cs["p"], "float16",
                                        out_dt="float16", w_dt="float16", b_dt="float16")
    res["mutants"]["M15"] = dict(witness=wid, pow_m15_max=big, dx_finite=bool(np.all(np.isfinite(dx))),
                                 dw_finite=bool(np.all(np.isfinite(dw))))
    # Hypothetical single-decision mutants the registered list does not contain
    # (a) centering omitted; (b) 'bfloat16' dropped from the upcast set (bf16 computed in bf16)
    kills = {"nocenter": [], "bf16_not_upcast": []}
    for cs in allc:
        y_ref, grads = T.groupnorm_reference(cs["x"], cs["G"], cs["w"], cs["b"], cs["eps"], cs["p"])
        fa, fr = TOL[cs["xd"]]["forward"]
        y, *_ = E.engine_graph(cs["x"], cs["G"], cs["w"], cs["b"], cs["eps"], cs["p"], "float64", mutant="nocenter",
                               out_dt=cs["xd"], w_dt=cs["wd"], b_dt=cs["bd"])
        if ratio(y, y_ref, fa, fr) > 1:
            kills["nocenter"].append(cs["id"])
        if cs["xd"] == "bfloat16":
            y, dx, dw, db, _ = E.engine_graph(cs["x"], cs["G"], cs["w"], cs["b"], cs["eps"], cs["p"], "bfloat16",
                                              out_dt="bfloat16", w_dt=cs["wd"], b_dt=cs["bd"])
            r = ratio(y, y_ref, fa, fr)
            if cs["backward"]:
                for got, exp, ld in ((dx, grads[0], "bfloat16"), (dw, grads[1], cs["wd"]), (db, grads[2], cs["bd"])):
                    ba, br = TOL[ld]["backward"]
                    r = max(r, ratio(got, exp, ba, br))
            if r > 1:
                kills["bf16_not_upcast"].append((cs["id"], r))
    res["extra_mutants"] = {k: dict(n_killing_ids=len(v), first=v[:5]) for k, v in kills.items()}
    out = HERE / "numerics_results.json"
    out.write_text(json.dumps(res, indent=2, default=float) + "\n")
    print(json.dumps(res, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
