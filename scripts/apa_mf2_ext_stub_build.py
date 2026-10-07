"""APA-MF2 stub harness build: compile a given apa.cpp + scripts/apa_mf2_ext_stub_kernels.cpp
as the C++-only torch extension `apa_attention_cuda` into OUT_DIR.

usage: python3 scripts/apa_mf2_ext_stub_build.py <apa.cpp> <out_dir>

Labeled harness: it runs apa.cpp's own code (masks, thresholds, tiling, backward) on
CUDA tensors through ATen; apa_kernels.cu is replaced by an ATen transcription. It is
NOT the shipped extension and never counts as a G4 build PASS. No nvcc is involved,
so torch's CUDA major-version check (cpp_extension.py:544-545) does not apply.
Prior art: torch.utils.cpp_extension (PyTorch). Script: mine.
"""
import hashlib
import os
import shutil
import sys

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    src, out = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
    os.makedirs(out, exist_ok=True)
    tmp = os.path.join(out, "build_tmp")
    # copy sources next to the build so setuptools sees relative paths
    srcdir = os.path.join(out, "src")
    os.makedirs(srcdir, exist_ok=True)
    shutil.copyfile(src, os.path.join(srcdir, "apa.cpp"))
    shutil.copyfile(os.path.join(ROOT, "scripts/apa_mf2_ext_stub_kernels.cpp"),
                    os.path.join(srcdir, "apa_stub_kernels.cpp"))
    os.chdir(out)
    ext = CppExtension("apa_attention_cuda", ["src/apa.cpp", "src/apa_stub_kernels.cpp"],
                       extra_compile_args=["-O3", "-std=c++17"])
    setup(name="apa_mf2_ext_stub", ext_modules=[ext],
          cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},
          script_args=["build_ext", "--build-lib", out, "--build-temp", tmp])
    h = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
    so = [f for f in os.listdir(out) if f.startswith("apa_attention_cuda") and f.endswith(".so")]
    print("STUB BUILT", out, so, "apa.cpp sha256", h(src))


if __name__ == "__main__":
    main()
