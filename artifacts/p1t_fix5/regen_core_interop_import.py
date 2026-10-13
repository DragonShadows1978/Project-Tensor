"""P1T-FIX5 copy of P1T-FIX4's regen_core_interop_import.py; only W (the tree used for the read-only `git show 62c8d9f`)
is retargeted to this worktree. Original docstring follows.

P1T-FIX4: regenerate the IMPORT half of the round-4 reviewer's core_interop_cpu.cpp from a given core tree.

The probe embeds verbatim 62c8d9f slices of core source. Its import mode (the V4-03 witness: real CUDA runtime, no
usable device, a zero-size device-0 capsule) calls tensor_from_dlpack_capsule from the embedded slice, so on its own it
can only ever test 62c8d9f. This script replaces exactly that one function's text (from "Tensor
tensor_from_dlpack_capsule(py::object obj) {" up to the blank line before "// Amendment 1: device-side empty") with the
same region of <tree>/tensor_cuda/src/bindings.cpp, after asserting that the probe's region equals 62c8d9f's
(git show 62c8d9f:tensor_cuda/src/bindings.cpp) byte for byte. Self-check: regenerating from 62c8d9f reproduces the probe
file byte for byte (printed). The export (SPY) half is NOT regenerated: the after export path calls a kernels.cu helper
(cudaPointerGetAttributes label proof) that the probe's kernels slice does not contain, and the real-module probes and
the LD_PRELOAD shim tests decide V4-01 on the built binary instead.
Usage: regen_core_interop_import.py <probe.cpp> <tree> <out.cpp>
No prior art claimed (same method as FIX3's regen_lifecycle_slice.py).
"""
import subprocess
import sys
from pathlib import Path

START = 'Tensor tensor_from_dlpack_capsule(py::object obj) {\n'
END = '\n// Amendment 1: device-side empty'
W = '/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix5'


def region(text):
    i = text.index(START)
    j = text.index(END, i)
    assert text.count(START) == 1
    return i, j


def main():
    probe_path, tree, out = sys.argv[1:4]
    probe = Path(probe_path).read_text()
    before = subprocess.run(['git', '-C', W, 'show', '62c8d9f:tensor_cuda/src/bindings.cpp'], capture_output=True,
                            text=True, check=True).stdout
    pi, pj = region(probe)
    bi, bj = region(before)
    assert probe[pi:pj] == before[bi:bj], 'the probe region is not 62c8d9f verbatim'
    self_check = probe[:pi] + before[bi:bj] + probe[pj:]
    print('self-check (regenerating from 62c8d9f reproduces the probe):', self_check == probe)
    src = (Path(tree) / 'tensor_cuda/src/bindings.cpp').read_text()
    si, sj = region(src)
    Path(out).write_text(probe[:pi] + src[si:sj] + probe[pj:])
    print(f'wrote {out}: region {pj - pi} -> {sj - si} chars from {tree}')


if __name__ == '__main__':
    main()
