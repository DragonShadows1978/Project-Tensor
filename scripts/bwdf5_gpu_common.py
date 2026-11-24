"""BWD-F5 staged GPU lines: the arm table and loader (a copy of scripts/bwdf3_gpu_common.py with this round's arms and
outputs; BWD-F1's witness data helper is re-used unchanged).

Imported by the step script only after its `forge-gpu-lease-check --require-held` line, and by
scripts/bwdf5_gpu_lines.py, scripts/bwdf5_gpu_post.py and scripts/bwdf5_p2_run.py (CPU; no engine import: they use
files and NumPy only). The engine is imported only inside load_arm(), which only the step script calls.

Arms (P2 only) and where each is read from. L103: the frozen package's module and the build copy must both equal the
receipt's sha256 before a line starts, and the loader re-checks before the import.
  B  -> receipt artifacts/bwdf1/build/BUILD_tc_B.json;      modules in the BWD-F1 worktree (read-only):
        <sibling worktree tensor_fork_apa_wt_bwdf1>/tensor_cuda/build-bwdf1/B/{module,pkg}
  F2 -> receipt artifacts/bwdf1/f2/build/BUILD_tc_F2.json;  modules in the same sibling worktree: .../build-bwdf1/F2
  F4 -> receipt artifacts/bwdf1/f4/build/BUILD_tc_F4.json;  modules in THIS worktree: tensor_cuda/build-bwdf4/F4
Outputs go under artifacts/bwdf1/f5/gpu/.
Prior art: BWD-F3's loader (this repo), copied; changed: the arm table only. No other prior art known to me.
"""
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import bwdf1_gpu_common as G1  # noqa: E402

GPU = ROOT / "artifacts/bwdf1/f5/gpu"
PKGS = G1.PKGS
data = G1.data
SIBLING = ROOT.parent / "tensor_fork_apa_wt_bwdf1"  # BWD-F1's worktree: its Build B / Build F2 modules (read-only)
ARMS = {  # arm -> (receipt relative to ROOT, build directory holding module/ and pkg/)
    "B": ("artifacts/bwdf1/build/BUILD_tc_B.json", SIBLING / "tensor_cuda/build-bwdf1/B"),
    "F2": ("artifacts/bwdf1/f2/build/BUILD_tc_F2.json", SIBLING / "tensor_cuda/build-bwdf1/F2"),
    "F4": ("artifacts/bwdf1/f4/build/BUILD_tc_F4.json", ROOT / "tensor_cuda/build-bwdf4/F4"),
}


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def receipt(pkg, arm):
    return ROOT / ARMS[arm][0]


def pkg_dir(pkg, arm):
    return ARMS[arm][1] / "pkg"


def check_arm(pkg, arm):
    name, so = PKGS[pkg]
    rec = json.loads(receipt(pkg, arm).read_text())
    for p in (ARMS[arm][1] / "module" / so, pkg_dir(pkg, arm) / name / so):
        got = sha(p)
        if got != rec["module_sha256"]:
            raise SystemExit(f"bwdf5: {p} sha256 {got} != {receipt(pkg, arm).name} {rec['module_sha256']}; refusing")
    return rec["module_sha256"]


def load_arm(arm, pkg="tc"):
    """Import the arm's frozen package (step script only, inside the lease)."""
    name, _ = PKGS[pkg]
    got = check_arm(pkg, arm)
    sys.path.insert(0, str(pkg_dir(pkg, arm)))
    mod = __import__(name)
    assert pathlib.Path(mod.__file__).resolve().parent == (pkg_dir(pkg, arm) / name).resolve(), mod.__file__
    return mod, got
