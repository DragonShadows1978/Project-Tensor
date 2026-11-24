"""BWD-F3 copy (of BWD-F2's copy) of BWD-AUD1's staged-line loader (via BWD-F1's copy) with only its module pin changed: tensor_cuda from
this worktree's frozen Build F3 package (tensor_cuda/build-bwdf1/F3/pkg), after checking its sha256 against its build
receipt (scripts/bwdf3_gpu_common.py receipt(); L103). Same API as the audit's (check, load). Inside a lease only; it
never changes CUDA_VISIBLE_DEVICES. Prior art: the audit's loader (this repo); none other known to me."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[5] / 'scripts'))
import bwdf3_gpu_common as G  # noqa: E402

ARM = 'F3'


def check(pkg):
    got = G.check_arm('tc', ARM)
    return G.pkg_dir('tc', ARM) / 'tensor_cuda', got


def load(pkg='tensor_cuda'):
    return G.load_arm(ARM)
