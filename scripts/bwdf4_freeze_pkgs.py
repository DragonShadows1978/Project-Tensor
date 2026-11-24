"""BWD-F4: freeze each arm's tensor_cuda Python package with its receipted module into the build directory
(tensor_cuda/build-bwdf4/<arm>/pkg/tensor_cuda), so G1's rows and the mutant runs read neither /tmp nor the
worktree's (swappable) package folder. A copy of scripts/bwdf3_freeze_pkgs.py (tensor_fork_apa_wt_bwdf1), tensor_cuda
only, reading artifacts/bwdf1/f4/build and /tmp/bwdf4_src_<arm>. Refuses if the destination exists; checks the copied
module's sha256 and each copied Python file against the receipt's source map. Prior art: ENG4's frozen pkg/ copies
(this repo). Ours: nothing else.

  python3 -B -P scripts/bwdf4_freeze_pkgs.py <arm> [<arm> ...]
"""
import hashlib
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAME, SO = "tensor_cuda", "_tensor_cuda.cpython-312-x86_64-linux-gnu.so"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


for arm in sys.argv[1:]:
    rec = json.load(open(os.path.join(ROOT, f"artifacts/bwdf1/f4/build/BUILD_tc_{arm}.json")))
    src = f"/tmp/bwdf4_src_{arm}/{NAME}/{NAME}"
    dst = os.path.join(ROOT, NAME, "build-bwdf4", arm, "pkg", NAME)
    if os.path.exists(dst):
        raise SystemExit(f"{dst} exists")
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    assert sha(os.path.join(dst, SO)) == rec["module_sha256"], (arm, "module")
    n = 0
    for d, _, files in os.walk(dst):
        for f in files:
            if f.endswith(".py"):
                key = NAME + "/" + os.path.relpath(os.path.join(d, f), os.path.dirname(dst))
                assert sha(os.path.join(d, f)) == rec["sources"][key], (arm, key)
                n += 1
    print("FROZEN", arm, dst, "module", rec["module_sha256"], "py files checked", n)
