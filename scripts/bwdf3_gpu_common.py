"""BWD-F3 staged GPU lines: the arm table and loader (a copy of scripts/bwdf2_gpu_common.py with this round's arms and
outputs), re-using BWD-F1's witness cases and verdicts.

Imported by the step scripts only after their `forge-gpu-lease-check --require-held` line, and by
scripts/bwdf3_gpu_lines.py and scripts/bwdf3_gpu_post.py (CPU; no engine import: they use files and NumPy only).
The engine is imported only inside load_arm(), which only step scripts call.

Arms and their receipts (L103: the frozen package's module and the build copy must equal the receipt's sha256):
  B, E, m03_reduce_gt1   -> artifacts/bwdf1/build/BUILD_<pkg>_<arm>.json (BWD-F1's builds, read-only)
  F2, n0*                -> artifacts/bwdf1/f2/build/BUILD_<pkg>_<arm>.json (BWD-F2's builds, read-only)
  F3, p0*                -> artifacts/bwdf1/f3/build/BUILD_<pkg>_<arm>.json (this round's builds)
Everything else (cases, controls, data, run_case, case_verdict, memcheck_summary, bitwise_pair) is BWD-F1's
scripts/bwdf1_gpu_common.py, unchanged. Outputs go under artifacts/bwdf1/f3/gpu/.
Prior art: BWD-F1's, BWD-F2's and ENG4's staged-line patterns (this repo). Ours: the arm table.
"""
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import bwdf1_gpu_common as G1  # noqa: E402

GPU = ROOT / "artifacts/bwdf1/f3/gpu"
PKGS = G1.PKGS
DTYPES, SEED = G1.DTYPES, G1.SEED
cases, controls, data, readback = G1.cases, G1.controls, G1.data, G1.readback
run_case, case_verdict, memcheck_summary, bitwise_pair = G1.run_case, G1.case_verdict, G1.memcheck_summary, \
    G1.bitwise_pair
BWDF1_ARMS = ("B", "E", "m03_reduce_gt1")


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def receipt(pkg, arm):
    if arm in BWDF1_ARMS:
        d = "artifacts/bwdf1/build"
    elif arm == "F2" or arm.startswith("n0"):
        d = "artifacts/bwdf1/f2/build"
    else:
        d = "artifacts/bwdf1/f3/build"
    return ROOT / d / f"BUILD_{pkg}_{arm}.json"


def pkg_dir(pkg, arm):
    return ROOT / PKGS[pkg][0] / "build-bwdf1" / arm / "pkg"


def check_arm(pkg, arm):
    name, so = PKGS[pkg]
    rec = json.loads(receipt(pkg, arm).read_text())
    for p in (ROOT / name / "build-bwdf1" / arm / "module" / so, pkg_dir(pkg, arm) / name / so):
        got = sha(p)
        if got != rec["module_sha256"]:
            raise SystemExit(f"bwdf3: {p} sha256 {got} != {receipt(pkg, arm).name} {rec['module_sha256']}; refusing")
    return rec["module_sha256"]


def load_arm(arm, pkg="tc"):
    """Import the arm's frozen package (step scripts only, inside the lease)."""
    name, _ = PKGS[pkg]
    got = check_arm(pkg, arm)
    sys.path.insert(0, str(pkg_dir(pkg, arm)))
    mod = __import__(name)
    assert pathlib.Path(mod.__file__).resolve().parent == (pkg_dir(pkg, arm) / name).resolve(), mod.__file__
    return mod, got
