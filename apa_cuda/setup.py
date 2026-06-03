"""Build script for the APA-Quant CUDA extension.

Build/install from this directory::

    pip install -e .

Requires a CUDA toolkit (nvcc) and a CUDA-enabled PyTorch. Set TORCH_CUDA_ARCH_LIST
to target a specific GPU, e.g. for an RTX 3070 (Ampere, sm_86)::

    TORCH_CUDA_ARCH_LIST="8.6" pip install -e .
"""

import os

from setuptools import find_packages, setup

try:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PyTorch is required to build this extension. Install torch first."
    ) from exc

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

ext = CUDAExtension(
    name="apa_attention_cuda",
    sources=[
        os.path.join("csrc", "apa.cpp"),
        os.path.join("csrc", "apa_kernels.cu"),
    ],
    extra_compile_args={
        "cxx": ["-O3", "-std=c++17"],
        "nvcc": ["-O3", "--use_fast_math"],
    },
)

setup(
    name="apa-attention-cuda",
    version="0.1.0",
    description="APA-Quant attention: drop-in PyTorch SDPA with C++/CUDA kernels",
    packages=find_packages(include=["apa_attention", "apa_attention.*"]),
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension},
    install_requires=["torch>=2.0", "numpy"],
    python_requires=">=3.9",
)
