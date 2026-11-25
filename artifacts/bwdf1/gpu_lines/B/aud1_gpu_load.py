"""BWD-F1 copy of BWD-AUD1's staged-line loader with only its module pin changed: tensor_cuda from this worktree's
frozen Build B package (tensor_cuda/build-bwdf1/B/pkg), after checking its sha256 against
artifacts/bwdf1/build/BUILD_tc_B.json (L103). Same API as the audit's (check, load). Inside a lease only; it never
changes CUDA_VISIBLE_DEVICES. Prior art: the audit's loader (this repo); none other known to me."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4] / 'scripts'))
import bwdf1_gpu_common as G  # noqa: E402

ARM = 'B'


def check(pkg):
    got = G.check_arm('tc', ARM)
    return G.pkg_dir('tc', ARM) / 'tensor_cuda', got


def load(pkg='tensor_cuda'):
    return G.load_arm(ARM)
