"""APA-Quant attention as a PyTorch C++/CUDA extension.

A drop-in replacement for ``torch.nn.functional.scaled_dot_product_attention``
backed by hand-written CUDA kernels plus cuBLAS GEMMs, so APA can be compared
head-to-head with PyTorch attention without Python-interpreter overhead.

Usage::

    from apa_attention import apa_scaled_dot_product_attention as apa_sdpa
    out = apa_sdpa(q, k, v, is_causal=True)            # drop-in for F.sdpa

    from apa_attention import apa_quant_attention
    out = apa_quant_attention(q, k, v, bulk_bits=2, refine_percentile=0.15)
"""

from .ops import (
    apa_quant_attention,
    apa_scaled_dot_product_attention,
)
from .quant_tables import build_tables

__all__ = [
    "apa_quant_attention",
    "apa_scaled_dot_product_attention",
    "build_tables",
]

__version__ = "0.1.0"
