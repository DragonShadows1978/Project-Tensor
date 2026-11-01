#!/usr/bin/env python3
"""GN1-REV1 Q6/L09: arm isolation by sha256 maps (symlinks resolved) and one
import probe per arm class, replicating run_gpu_tests.py's sys.path recipe
exactly (sys.path[0] = tool dir, then insert(0, arm), insert(1, ROOT)).
CPU only; imports with bytecode writing disabled; writes only under rev1/.
Prior art: content addressing by SHA-256 (NIST FIPS 180-2, 2002). Ours: wiring.
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
A1 = ROOT / "artifacts/gn1/a1"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

PROBE = r'''
import sys, json, hashlib
from pathlib import Path
TOOL, ROOT, arm = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
package = TOOL / "arms" / arm / "tensor_cuda"
sys.path[0] = str(TOOL)            # what `python3 run_gpu_tests.py` gives
sys.path.insert(0, str(package.parent))
sys.path.insert(1, str(ROOT))
import tensor_cuda as tc
import tensor_cuda.functional as F
sys.path.insert(0, str(ROOT / "scripts"))   # what test_apa_om135_imm.py does at import
import tensor_cuda as tc2
import inspect
src = inspect.getsource(tc.nn.GroupNorm.forward)
print(json.dumps(dict(arm=arm, init=str(Path(tc.__file__).resolve()), nn=str(Path(tc.nn.__file__).resolve()),
    functional=str(Path(F.__file__).resolve()), C=str(Path(tc._C.__file__).resolve()),
    same_after_scripts_path=tc2 is tc, nn_sha=hashlib.sha256(Path(tc.nn.__file__).read_bytes()).hexdigest(),
    minus_one_reshape="-1])" in src, upcast="astype(\"float32\")" in src,
    sys_path_head=sys.path[:4])))
'''


def main():
    out = {}
    seal = json.loads((A1 / "staging_seal.json").read_text())
    arms = sorted(p.name for p in (A1 / "arms").iterdir())
    maps = {}
    for arm in arms:
        pkg = A1 / "arms" / arm / "tensor_cuda"
        files = sorted(str(p.relative_to(pkg)) for p in pkg.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
        maps[arm] = {f: sha((pkg / f).resolve()) for f in files}
        maps[arm]["_symlinks"] = {str(p.relative_to(pkg)): os.readlink(p) for p in pkg.rglob("*") if p.is_symlink()}
    out["arm_count"] = len(arms)
    cand = maps["candidate"]
    diffs = {}
    for arm in arms:
        diffs[arm] = sorted(f for f in set(cand) | set(maps[arm]) if f != "_symlinks" and cand.get(f) != maps[arm].get(f))
    out["files_differing_from_candidate"] = diffs
    out["seal_matches_disk"] = {arm: all(maps[arm].get(f) == h for f, h in seal["arms"][arm].items()) and
                                set(seal["arms"][arm]) == set(k for k in maps[arm] if k != "_symlinks")
                                for arm in arms}
    # baseline python files equal 0e1dea3's blobs; candidate nn.py equals HEAD's tracked nn.py
    base_ok = {}
    for f in maps["baseline"]:
        if f.endswith(".py"):
            blob = subprocess.run(["git", "-C", str(ROOT), "show", f"0e1dea3:tensor_cuda/tensor_cuda/{f}"],
                                  capture_output=True).stdout
            base_ok[f] = hashlib.sha256(blob).hexdigest() == maps["baseline"][f]
    out["baseline_py_equal_0e1dea3"] = base_ok
    out["candidate_nn_equals_worktree_nn"] = cand["nn.py"] == sha(ROOT / "tensor_cuda/tensor_cuda/nn.py")
    head_nn = subprocess.run(["git", "-C", str(ROOT), "show", "HEAD:tensor_cuda/tensor_cuda/nn.py"], capture_output=True).stdout
    out["candidate_nn_equals_HEAD_nn"] = cand["nn.py"] == hashlib.sha256(head_nn).hexdigest()
    pin = json.loads((A1 / "engine_pin.json").read_text())
    out["module_sha_equals_pin"] = sha(ROOT / pin["module"]) == pin["sha256"]
    out["so_symlink_targets"] = sorted({v for m in maps.values() for v in m["_symlinks"].values()})
    probes = {}
    for arm, cvd in (("baseline", ""), ("candidate", ""), ("M01", ""), ("candidate", "")):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=cvd, PYTHONDONTWRITEBYTECODE="1")
        key = arm if arm not in probes else "audit(candidate, CUDA hidden, GN1_AUDIT=1)"
        if key.startswith("audit"):
            env["GN1_AUDIT"] = "1"
        p = subprocess.run([sys.executable, "-B", "-c", PROBE, str(A1), str(ROOT), arm], cwd=str(A1), env=env,
                           capture_output=True, text=True, timeout=120)
        probes[key] = dict(exit=p.returncode, out=json.loads(p.stdout) if p.returncode == 0 else p.stderr[-800:])
    out["import_probes"] = probes
    (HERE / "arms_probe_results.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=1)[:7000])


if __name__ == "__main__":
    main()
