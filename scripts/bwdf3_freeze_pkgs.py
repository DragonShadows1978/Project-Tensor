"""BWD-F3: a copy of scripts/bwdf2_freeze_pkgs.py reading artifacts/bwdf1/f3/build and /tmp/bwdf3_src_<arm>.
BWD-F2 (a copy of scripts/bwdf1_freeze_pkgs.py reading this round's receipts and /tmp/bwdf3_src_<arm>). BWD-F1: freeze each staged arm's Python package with its receipted module into the build directory, so the staged
GPU lines read neither /tmp nor the worktree's (swappable) package folder.

  python3 -B -P scripts/bwdf3_freeze_pkgs.py <pkg tc|ts> <arm> [<arm> ...]

Copies /tmp/bwdf3_src_<arm>/<name>/<name>/ (no __pycache__) to <name>/build-bwdf1/<arm>/pkg/<name>/ (refuses if it
exists), then checks the copied module's sha256 against artifacts/bwdf1/f3/build/BUILD_<pkg>_<arm>.json and each copied
Python file against the receipt's source map. Prior art: ENG4's frozen pkg/ copies (this repo). Ours: nothing else.
"""
import hashlib
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKGS = {"tc": ("tensor_cuda", "_tensor_cuda.cpython-312-x86_64-linux-gnu.so"),
        "ts": ("tc_signal", "_tc_signal.cpython-312-x86_64-linux-gnu.so")}


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


pkg = sys.argv[1]
name, so = PKGS[pkg]
for arm in sys.argv[2:]:
    rec = json.load(open(os.path.join(ROOT, f"artifacts/bwdf1/f3/build/BUILD_{pkg}_{arm}.json")))
    src = f"/tmp/bwdf3_src_{arm}/{name}/{name}"
    dst = os.path.join(ROOT, name, "build-bwdf1", arm, "pkg", name)
    if os.path.exists(dst):
        raise SystemExit(f"{dst} exists")
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    assert sha(os.path.join(dst, so)) == rec["module_sha256"], (arm, "module")
    n = 0
    for d, _, files in os.walk(dst):
        for f in files:
            if f.endswith(".py"):
                key = name + "/" + os.path.relpath(os.path.join(d, f), os.path.dirname(dst))
                assert sha(os.path.join(d, f)) == rec["sources"][key], (arm, key)
                n += 1
    print("FROZEN", pkg, arm, dst, "module", rec["module_sha256"], "py files checked", n)
