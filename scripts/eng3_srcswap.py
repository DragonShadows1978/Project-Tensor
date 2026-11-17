"""ENG3: put Build B's or Build E's engine sources (kernels.cu, bindings.cpp, core.h) into the worktree, so the suite's
source-reading tests (byte pins of kernels.cu etc.) see the sources of the build whose module is installed.
B = artifacts/eng3/before/ (2478a4d bytes), E = artifacts/eng3/E_src/ (this round's final sources).
Usage: python3 -B -P scripts/eng3_srcswap.py B|E   (prints the sha256 of each installed file)
Prior art: no prior art known to me."""
import hashlib, os, shutil, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = {"B": "artifacts/eng3/before", "E": "artifacts/eng3/E_src"}[sys.argv[1]]
DST = {"kernels.cu": "tensor_cuda/src/kernels.cu", "bindings.cpp": "tensor_cuda/src/bindings.cpp",
       "core.h": "tensor_cuda/include/tc/core.h"}
for name, dst in DST.items():
    s, d = os.path.join(ROOT, SRC, name), os.path.join(ROOT, dst)
    shutil.copyfile(s, d)
    h = [hashlib.sha256(open(p, "rb").read()).hexdigest() for p in (s, d)]
    assert h[0] == h[1]
    print(sys.argv[1], dst, h[1])
