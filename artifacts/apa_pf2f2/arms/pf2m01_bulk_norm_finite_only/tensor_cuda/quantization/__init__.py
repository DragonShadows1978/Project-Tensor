"""Reusable quantization math for Project-Tensor.

CUDA kernels should only know packed layouts and fast dequant rules. This
package owns the slower, inspectable math: scale selection, reference packing,
calibration helpers, and future AWQ/GPTQ/HQQ/EXL2 policy code.
"""

from .affine import (
    AffineQuantizedWeights,
    dequantize_affine_per_group,
    dequantize_symmetric_per_group,
    pack_lowbit,
    packed_width,
    qmax,
    quantize_affine_per_group,
    symmetric_offset,
    unpack_lowbit,
)

__all__ = [
    "AffineQuantizedWeights",
    "dequantize_affine_per_group",
    "dequantize_symmetric_per_group",
    "pack_lowbit",
    "packed_width",
    "qmax",
    "quantize_affine_per_group",
    "symmetric_offset",
    "unpack_lowbit",
]
