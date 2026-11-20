"""Gradient byte hashing on all four in-scope Kokoro operations."""

import hashlib

import numpy as np
import pytest

import tc_signal as ts
import tensor_cuda as tc
from bwd_r2_support.cases import (
    OPS, SLOTS, fixture_root, gpu_args, load_cell, nonconstant, reference_inputs, require_gpu, signal_forward,
)


def gradient_hashes(op, repetitions):
    row, data = load_cell(1, op, "kokoro")
    args, seed = reference_inputs(row, data, rounded=True)
    slots = SLOTS.get(op, ("x",))
    dy = tc.tensor(seed)
    digests = []
    for _ in range(repetitions):
        leaves = gpu_args(tc, op, args, active=True, omit_affine=op in OPS[2:])
        y = signal_forward(tc, ts, op, leaves, row["attrs"])
        (y * dy).sum().backward()
        grad_hash = {}
        for slot in slots:
            gradient = leaves[slot].grad.numpy()
            nonconstant(gradient)
            assert gradient.dtype == np.float32
            grad_hash[slot] = hashlib.sha256(gradient.view(np.uint32).tobytes()).hexdigest()
        digests.append(grad_hash)
    assert all(record == digests[0] for record in digests)
    return digests


@pytest.mark.parametrize("op", OPS)
def test_gpu_grad_bytes(op):
    fixture_root(1)
    require_gpu()
    print(f"{op} 10 reruns {gradient_hashes(op, 10)}", flush=True)
