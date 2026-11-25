"""BWD-F1 staged GPU lines: the shared table, the arm loader, the witness cases and the post-device verdicts.

Imported by the step scripts only after their forge-gpu-lease-check line, and by scripts/bwdf1_gpu_gate.py and
scripts/bwdf1_gpu_post.py (CPU, no engine import: the gate and the post-device code use only NumPy and files).
The engine is imported only inside load_arm(), which only step scripts call.

Witness cases (G2 direct and G3): every binary op in artifacts/bwdf1/CENSUS.json on [4, 0] against [4, 1] (both
operand orders; where in each operand position; masked_fill; expand; axpy_), float32 / float16 / bfloat16. Expected:
NumPy's output shape (np.broadcast_shapes) and, for differentiable ops, each gradient zeros of its operand's shape
(np.zeros, compared bitwise as float32 after an exact widening). Controls: one nonzero-shape case per op ([4, 3]
against [4, 1], seeded), saved for a bitwise Build B / Build E comparison (L19 nonzero / non-constant first).

Prior art: NumPy's broadcasting rules and empty sums; the BWD-AUD1 witness scripts (this repo) whose loader API this
keeps; ENG4's staged-line pattern (gate before the lease, write-once directories). Ours: the case table.
"""
import hashlib
import importlib.util
import json
import os
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
GPU = ROOT / "artifacts/bwdf1/gpu"
PKGS = {"tc": ("tensor_cuda", "_tensor_cuda.cpython-312-x86_64-linux-gnu.so"),
        "ts": ("tc_signal", "_tc_signal.cpython-312-x86_64-linux-gnu.so")}
DTYPES = ("float32", "float16", "bfloat16")
SEED = 20261124


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def pkg_dir(pkg, arm):
    name, _ = PKGS[pkg]
    return ROOT / name / "build-bwdf1" / arm / "pkg"


def check_arm(pkg, arm):
    """L103: the frozen package's module and the build copy must both equal the build receipt."""
    name, so = PKGS[pkg]
    rec = json.loads((ROOT / f"artifacts/bwdf1/build/BUILD_{pkg}_{arm}.json").read_text())
    for p in (ROOT / name / "build-bwdf1" / arm / "module" / so, pkg_dir(pkg, arm) / name / so):
        got = sha(p)
        if got != rec["module_sha256"]:
            raise SystemExit(f"bwdf1: {p} sha256 {got} != BUILD_{pkg}_{arm}.json {rec['module_sha256']}; refusing")
    return rec["module_sha256"]


def load_arm(arm, pkg="tc"):
    """Import the arm's frozen package (step scripts only, inside the lease)."""
    name, _ = PKGS[pkg]
    got = check_arm(pkg, arm)
    sys.path.insert(0, str(pkg_dir(pkg, arm)))
    mod = __import__(name)
    assert pathlib.Path(mod.__file__).resolve().parent == (pkg_dir(pkg, arm) / name).resolve(), mod.__file__
    return mod, got


# ------------------------------------------------------------------------------------------------ witness cases
BIN = ("add", "sub", "mul", "div", "maximum", "minimum")
CMP = ("gt", "ge", "lt", "le", "eq")


def cases():
    out = []
    for dt in DTYPES:
        for op in BIN + CMP:
            for sa, sb in (((4, 0), (4, 1)), ((4, 1), (4, 0))):
                out.append({"id": f"{op}-{dt}-{sa}-{sb}", "op": op, "dtype": dt, "shapes": [sa, sb],
                            "grad": op in BIN})
        for pos in ("cond", "x", "y"):
            sh = {"cond": (4, 1), "x": (4, 1), "y": (4, 1)}
            sh[pos] = (4, 0)
            out.append({"id": f"where_{pos}-{dt}", "op": "where", "dtype": dt,
                        "shapes": [sh["cond"], sh["x"], sh["y"]], "grad": True})
        out.append({"id": f"masked_fill-{dt}", "op": "masked_fill", "dtype": dt, "shapes": [(4, 0), (4, 1)],
                    "grad": True})
        out.append({"id": f"expand-{dt}", "op": "expand", "dtype": dt, "shapes": [(4, 0), (4, 1)], "grad": True})
        out.append({"id": f"axpy_-{dt}", "op": "axpy_", "dtype": dt, "shapes": [(4, 0), (4, 1)], "grad": False})
    return out


def controls():
    return [dict(c, id="control-" + c["id"], control=True,
                 shapes=[tuple(3 if e == 0 else e for e in s) for s in c["shapes"]]) for c in cases()]


def data(shape, dt, salt, positive=False):
    rng = np.random.default_rng(SEED + salt)
    v = rng.standard_normal(shape).astype(np.float32)
    if positive:
        v = np.abs(v) + 0.5
    return v


def readback(t):
    return None if t is None else t.astype("float32").numpy()


def run_case(tc, c):
    """Forward + backward of one case; returns a JSON-able row and (for controls) the arrays."""
    dt, s, op = c["dtype"], c["shapes"], c["op"]
    leaf = lambda shape, salt, rg, pos=False: tc.tensor(data(shape, dt, salt, pos), device="cuda", dtype=dt,
                                                        requires_grad=rg)
    row = {"id": c["id"], "op": op, "dtype": dt, "shapes": [list(x) for x in s]}
    arrays = {}
    try:
        if op in BIN or op in CMP:
            a, b = leaf(s[0], 1, c["grad"]), leaf(s[1], 2, c["grad"], op == "div")
            y = {"add": lambda: a + b, "sub": lambda: a - b, "mul": lambda: a * b, "div": lambda: a / b}.get(
                op, lambda: getattr(a, op)(b))()
            leaves = {"a": a, "b": b} if c["grad"] else {}
            want = np.broadcast_shapes(tuple(s[0]), tuple(s[1]))
        elif op == "where":
            cond = tc.tensor((data(s[0], dt, 3) > 0).astype(np.float32), device="cuda", dtype=dt)
            x, z = leaf(s[1], 4, True), leaf(s[2], 5, True)
            y = tc.where(cond, x, z)
            leaves = {"x": x, "y": z}
            want = np.broadcast_shapes(*[tuple(v) for v in s])
        elif op == "masked_fill":   # shapes [mask, a]
            mask = tc.tensor((data(s[0], dt, 6) > 0).astype(np.float32), device="cuda", dtype=dt)
            a = leaf(s[1], 7, True)
            y = a.masked_fill(mask, 2.0)
            leaves = {"a": a}
            want = np.broadcast_shapes(tuple(s[0]), tuple(s[1]))
        elif op == "expand":        # shapes [target, source]
            a = leaf(s[1], 8, True)
            y = a.expand(list(s[0]))
            leaves = {"a": a}
            want = np.broadcast_shapes(tuple(s[0]), tuple(s[1]))
        elif op == "axpy_":         # shapes [param, other]
            p = leaf(s[0], 9, False)
            tc._C.axpy_(p, leaf(s[1], 10, False), 0.5)
            y, leaves, want = p, {}, tuple(s[0])
        else:
            raise ValueError(op)
        tc.synchronize()
        row["out_shape"] = list(y.shape)
        row["want_shape"] = list(want)
        arrays["out"] = readback(y)
        if leaves:
            y.sum().backward()
            tc.synchronize()
            row["grads"] = {}
            for k, v in leaves.items():
                g = readback(v.grad)
                row["grads"][k] = None if g is None else {"shape": list(g.shape), "want": list(v.shape),
                                                          "nonzero_bits": int(np.count_nonzero(
                                                              np.ascontiguousarray(g).view(np.uint32)))}
                arrays["grad_" + k] = g
        row["result"] = "OK"
    except Exception as e:  # noqa: BLE001
        row["result"] = "ERROR"
        row["error"] = f"{type(e).__name__}: {e}"
    return row, arrays


def case_verdict(row):
    """Build E's bar for one zero-extent case: NumPy's shape, every gradient defined and zeros of its operand's shape."""
    if row.get("result") != "OK":
        return "FAIL: " + row.get("error", "no result")
    if row["out_shape"] != row["want_shape"]:
        return f"FAIL: output shape {row['out_shape']} != NumPy {row['want_shape']}"
    for k, g in (row.get("grads") or {}).items():
        if g is None:
            return f"FAIL: gradient of {k} undefined"
        if g["shape"] != g["want"]:
            return f"FAIL: gradient of {k} shape {g['shape']} != operand {g['want']}"
        if g["nonzero_bits"]:
            return f"FAIL: gradient of {k} has {g['nonzero_bits']} non-zero elements"
    return "PASS"


def memcheck_summary(text):
    """Parse compute-sanitizer's last 'ERROR SUMMARY: N error(s)' line; None when absent."""
    n = None
    for line in text.splitlines():
        if "ERROR SUMMARY:" in line:
            try:
                n = int(line.split("ERROR SUMMARY:")[1].split()[0])
            except (IndexError, ValueError):
                n = None
    return n


def bitwise_pair(b, e):
    """L09/L19: same dtype and shape, each side non-zero and non-constant at full width, then equal bits."""
    if b is None or e is None:
        return "FAIL: missing side"
    if b.dtype != e.dtype or b.shape != e.shape:
        return f"FAIL: dtype/shape {b.dtype}{b.shape} vs {e.dtype}{e.shape}"
    ub, ue = np.ascontiguousarray(b).view(np.uint32), np.ascontiguousarray(e).view(np.uint32)
    for name, u in (("B", ub), ("E", ue)):
        if u.size == 0 or not np.any(u) or np.all(u == u.reshape(-1)[0]):
            return f"VACUOUS: side {name} is empty, all zero or constant"
    return "PASS" if np.array_equal(ub, ue) else f"FAIL: {int(np.count_nonzero(ub != ue))} elements differ"
