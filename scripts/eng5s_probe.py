"""ENG5-S copy of the eng5 branch's scripts/eng5_probe.py (read-only there): catalogue scripts/eng5s_catalogue.py, shim
tensor_cuda/tests/support/eng5s_cuda_shim.c, scratch /tmp/eng5s_*. Otherwise unchanged. ENG5 text follows.

ENG5 CPU probe (G1 / H1 / BOUNDARY): call every catalogued entry with host operands in a child interpreter that has
ENG3's CUDA runtime counter (tensor_cuda/tests/support/cudart_count.c) and ENG5's cuBLAS/remaining-runtime counter
(tensor_cuda/tests/support/eng5_cuda_shim.c, built with -DENG5_WITH_ENG3_COUNTER) preloaded, and record per call its
outcome (result summary or exception) and the CUDA call count it made.

Usage (worktree root, CUDA hidden):
  python3 -B -P scripts/eng5_probe.py <package parent dir> <out.json> [--modes host,empty,one] [--only key,key]

Modes: host = every tensor operand on the host with the catalogue's nonzero shapes; empty = every operand's first
extent set to 0; emptylast = every operand's last extent set to 0; one = only the first operand's first extent set to 0 and every other operand's first extent set to
1 (broadcastable), for the entries whose empty rule is "any".

The parent never imports the engine. The child prints one JSON line per call; a child that dies is restarted after the
entry it died in (recorded as "child_died").

Prior art: LD_PRELOAD interposition counting (ENG3's counter, this repo); subprocess isolation per run (PSF). No other
prior art known to me.
"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUPPORT = os.path.join(ROOT, "tensor_cuda", "tests", "support")
SHIM_DIR = "/tmp/eng5s_shim"  # ENG5-F2: scratch prefix overridable (default unchanged)


def build_shims():
    os.makedirs(SHIM_DIR, exist_ok=True)
    eng3 = os.path.join(SHIM_DIR, "cudart_count.so")
    eng5 = os.path.join(SHIM_DIR, "eng5_count.so")
    subprocess.run(["gcc", "-O1", "-fPIC", "-shared", "-mgeneral-regs-only", "-o", eng3,
                    os.path.join(SUPPORT, "cudart_count.c"), "-ldl"], check=True, timeout=120)
    subprocess.run(["gcc", "-O1", "-fPIC", "-shared", "-DENG5_WITH_ENG3_COUNTER", "-I/usr/local/cuda-12.6/include",
                    "-o", eng5, os.path.join(SUPPORT, "eng5s_cuda_shim.c"), "-ldl"], check=True, timeout=120)
    return eng3, eng5


CHILD = r'''
import ctypes, hashlib, json, sys, traceback
sys.path.insert(0, sys.argv[1])
sys.path.insert(0, sys.argv[2])
import numpy as np
import tensor_cuda as tc
import eng5s_catalogue as cat
L = ctypes.CDLL(None)
L.eng3_cudart_calls.restype = ctypes.c_ulong
L.eng3_cudart_count.restype = ctypes.c_ulong
L.eng3_cudart_name.restype = ctypes.c_char_p
L.eng5_extra_calls.restype = ctypes.c_ulong
L.eng5_count.restype = ctypes.c_ulong
L.eng5_name.restype = ctypes.c_char_p
def snap():
    d = {}
    for i in range(L.eng3_cudart_n()):
        c = L.eng3_cudart_count(i)
        if c: d[L.eng3_cudart_name(i).decode()] = c
    for i in range(33):
        c = L.eng5_count(i)
        if c: d[L.eng5_name(i).decode()] = c
    return d
def total():
    return L.eng3_cudart_calls() + L.eng5_extra_calls()
NP = {"float32": np.float32, "float16": np.float16, "int64": np.int64, "uint8": np.uint8, "bool": np.bool_}
def make(shape, dtype, seed):
    rng = np.random.default_rng(seed)
    if dtype == "bfloat16":
        a = rng.standard_normal(shape).astype(np.float32)
        return tc._C.tensor(a, "cpu", False).astype("bfloat16")
    if dtype in ("int64", "uint8"):
        a = rng.integers(0, 2, size=shape).astype(NP[dtype])
    elif dtype == "bool":
        a = rng.integers(0, 2, size=shape).astype(np.bool_)
    else:
        a = rng.standard_normal(shape).astype(NP[dtype])
    return tc._C.tensor(np.ascontiguousarray(a), "cpu", False)
def summary(r, depth=0):
    if isinstance(r, tc.Tensor):
        s = {"shape": list(r.shape), "dtype": r.dtype, "device": r.device}
        if r.device == "cpu":
            a = r.astype("float32").numpy() if r.dtype == "bfloat16" else r.numpy()
            s["sha256"] = hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
        return s
    if isinstance(r, (tuple, list)) and depth < 3:
        return [summary(x, depth + 1) for x in r]
    return {"type": type(r).__name__}
mode = sys.argv[3]
keys = sys.argv[4].split(",")
for key in keys:
    row = cat.ROWS[key]
    seed = [20261123]
    first = [True]
    def f(name, shape, dtype):
        shape = list(shape)
        if mode == "empty" and shape:
            shape[0] = 0
        if mode == "emptylast" and shape:
            shape[-1] = 0
        if mode == "one" and shape:
            shape[0] = 0 if first[0] else 1
        first[0] = False
        seed[0] += 1
        return make(tuple(shape), dtype, seed[0])
    print(json.dumps({"key": key, "start": True}), flush=True)
    before, b0 = snap(), total()
    out = {"key": key, "mode": mode}
    try:
        r = row["call"](f)
        b1 = total(); after = snap()
        out["outcome"] = "ok"
        out["result"] = summary(r)
    except BaseException as ex:
        b1 = total(); after = snap()
        out["outcome"] = "raised"
        out["exc_type"] = type(ex).__name__
        out["message"] = str(ex)[:600]
    out["cuda_calls"] = b1 - b0
    out["cuda_calls_by_function"] = {k: after.get(k, 0) - before.get(k, 0) for k in after
                                     if after.get(k, 0) != before.get(k, 0)}
    print(json.dumps(out), flush=True)
'''


def run(pkg_parent, keys, mode, eng3, eng5):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", TC_APA_SP="1", LD_PRELOAD=eng3 + ":" + eng5, PYTHONDONTWRITEBYTECODE="1",
               TMPDIR=os.environ.get("TMPDIR", "/tmp/eng5s_tmp"))
    rows, todo = [], list(keys)
    while todo:
        p = subprocess.run([sys.executable, "-B", "-P", "-c", CHILD, pkg_parent, os.path.join(ROOT, "scripts"), mode,
                            ",".join(todo)], env=env, capture_output=True, text=True, timeout=1500,
                           cwd="/tmp/eng5s_tmp")
        done, started = set(), None
        for line in p.stdout.splitlines():
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("start"):
                started = d["key"]
            else:
                rows.append(d)
                done.add(d["key"])
                started = None
        if started is not None:  # the child died inside this entry
            rows.append({"key": started, "mode": mode, "outcome": "child_died", "returncode": p.returncode,
                         "stderr_tail": p.stderr[-600:]})
            done.add(started)
        elif p.returncode != 0:
            raise SystemExit(f"child failed outside an entry rc={p.returncode}: {p.stderr[-2000:]}")
        todo = [k for k in todo if k not in done]
    return rows


def main():
    pkg_parent, out = sys.argv[1], sys.argv[2]
    modes = (sys.argv[sys.argv.index("--modes") + 1] if "--modes" in sys.argv else "host,empty,emptylast,one").split(",")
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import eng5s_catalogue as cat
    only = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else None
    eng3, eng5 = build_shims()
    res = {"package_parent": pkg_parent, "env": {"CUDA_VISIBLE_DEVICES": ""}, "rows": []}
    for mode in modes:
        keys = [k for k, r in cat.ROWS.items() if r["call"] is not None and (only is None or k in only)]
        if mode == "one":
            keys = [k for k in keys if cat.ROWS[k]["empty_rule"] == "any" and len(cat.ROWS[k]["ops"]) > 1]
        res["rows"] += run(pkg_parent, keys, mode, eng3, eng5)
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print("rows", len(res["rows"]))


if __name__ == "__main__":
    main()
