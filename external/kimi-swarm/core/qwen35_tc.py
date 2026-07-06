"""core/qwen35_tc.py — Shared helpers across Qwen-class and Gemma ports.

HostEmbedding (CPU-side weight, GPU gather) and _cast are used by
the Gemma4 runner and other model adapters.
"""
from __future__ import annotations

import numpy as np

import tensor_cuda as tc
from core.mistral7b_tc import BlockTC


def _cast(t):
    """Cast a Tensor to the current compute dtype (BlockTC.COMPUTE_DTYPE)."""
    dt = BlockTC.COMPUTE_DTYPE
    return t if t.dtype == dt else t.astype(dt)


class HostEmbedding:
    """Embedding that keeps weights on host (CPU), gathers on GPU.

    Avoids pinning a 262K x D vocabulary matrix in GPU memory —
    the embedding table lives in host RAM and only the needed rows
    are copied to GPU per forward pass.
    """

    def __init__(self):
        self.weight = None          # (vocab_size, hidden_dim) numpy array on host

    def __call__(self, input_ids_np):
        """input_ids_np: (B, L) int64 numpy array -> (B, L, D) Tensor on GPU."""
        if self.weight is None:
            raise RuntimeError("HostEmbedding weight not set — call load_weights first")
        # Gather rows on host, transfer contiguous block to GPU
        rows = self.weight[input_ids_np]          # (B, L, D) on host
        return tc.tensor(np.ascontiguousarray(rows),
                         device="cuda", dtype=BlockTC.COMPUTE_DTYPE)

    @property
    def shape(self):
        return self.weight.shape if self.weight is not None else (0, 0)
