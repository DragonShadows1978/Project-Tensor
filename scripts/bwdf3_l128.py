"""BWD-F3 L128 / P5: sha256 snapshots of the files the arm-swapping suite runner writes temporarily and restores.

  python3 -B -P scripts/bwdf3_l128.py snap <name>        # artifacts/bwdf1/f3/l128/<name>.json (refuses if it exists)
  python3 -B -P scripts/bwdf3_l128.py cmp <before> <after>   # every path's sha256 equal -> PASS (prints and appends to
                                                             # artifacts/bwdf1/f3/l128/CHECKS.json)

The set (orders/BWDF3_REV2_FIXES.md writable set, L128): every file under tensor_cuda/src/ and tensor_cuda/include/,
tensor_cuda/tensor_cuda/*.py, tc_signal/tc_signal/__init__.py, tc_signal/src/bindings.cpp, tc_signal/README.md, and the
two installed modules (tensor_cuda/tensor_cuda/_tensor_cuda*.so, tc_signal/tc_signal/_tc_signal*.so). Each snapshot
also checks the engine sources against Build F3's receipted source map (artifacts/bwdf1/f3/build/BUILD_tc_F3.json and
BUILD_ts_F3.json): the restored tree must be the one Build F3 was built from. Prior art: content-hash manifests
(reproducible-builds practice); no other prior art known to me.
"""
import datetime
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "artifacts/bwdf1/f3/l128")
MODS = ["tensor_cuda/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so",
        "tc_signal/tc_signal/_tc_signal.cpython-312-x86_64-linux-gnu.so"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def paths():
    out = []
    for root in ("tensor_cuda/src", "tensor_cuda/include"):
        for d, dirs, names in os.walk(os.path.join(ROOT, root)):
            dirs[:] = sorted(x for x in dirs if x != "__pycache__")
            out += [os.path.relpath(os.path.join(d, n), ROOT) for n in sorted(names)]
    pk = os.path.join(ROOT, "tensor_cuda/tensor_cuda")
    out += sorted(os.path.relpath(os.path.join(pk, n), ROOT) for n in os.listdir(pk) if n.endswith(".py"))
    out += ["tc_signal/tc_signal/__init__.py", "tc_signal/src/bindings.cpp", "tc_signal/README.md"] + MODS
    return out


def snap(name):
    p = os.path.join(D, name + ".json")
    if os.path.exists(p):
        raise SystemExit(f"{p} exists")
    os.makedirs(D, exist_ok=True)
    files = {f: sha(os.path.join(ROOT, f)) for f in paths()}
    src_check = {}
    for pkg in ("tc", "ts"):
        rec = json.load(open(os.path.join(ROOT, f"artifacts/bwdf1/f3/build/BUILD_{pkg}_F3.json")))
        bad = {k: (files.get(k), v) for k, v in rec["sources"].items() if k in files and files[k] != v}
        src_check[pkg] = {"compared": sum(k in files for k in rec["sources"]), "differ_from_F3_source_map": bad,
                          "module_is_F3": files[MODS[0 if pkg == "tc" else 1]] == rec["module_sha256"]}
    doc = {"name": name, "time": datetime.datetime.now(datetime.timezone.utc).isoformat(), "n_files": len(files),
           "files": files, "F3_source_map_check": src_check}
    json.dump(doc, open(p, "w"), indent=1)
    print("snap", name, len(files), json.dumps({k: {kk: (vv if kk != "differ_from_F3_source_map" else len(vv))
                                                       for kk, vv in v.items()} for k, v in src_check.items()}))


def cmp(a, b):
    A = json.load(open(os.path.join(D, a + ".json")))["files"]
    B = json.load(open(os.path.join(D, b + ".json")))["files"]
    diff = {k: [A.get(k), B.get(k)] for k in sorted(set(A) | set(B)) if A.get(k) != B.get(k)}
    res = {"before": a, "after": b, "time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "n_paths": len(set(A) | set(B)), "differences": diff, "verdict": "PASS" if not diff else "FAIL"}
    cp = os.path.join(D, "CHECKS.json")
    rows = json.load(open(cp)) if os.path.exists(cp) else []
    rows.append(res)
    json.dump(rows, open(cp, "w"), indent=1)
    print("cmp", a, b, res["verdict"], "paths", res["n_paths"], "differences", len(diff))


if __name__ == "__main__":
    {"snap": lambda: snap(sys.argv[2]), "cmp": lambda: cmp(sys.argv[2], sys.argv[3])}[sys.argv[1]]()
