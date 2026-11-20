"""Fixture conformance, fixed-order witnesses and strict precision."""
from dataclasses import replace
from itertools import product
import os

import numpy as np
import pytest
import tc_signal as ts

from bwd_r3_support.cases import (BP1, BP2, SLOTS, allowed, answer, compare, exact, fixture,
                                  gradients, require_gpu, synthetic)


def check(case, *, sparse=False, slots=SLOTS):
    expected, mass = answer(case)
    result = gradients(case, tuple(slot in slots for slot in SLOTS))
    for i, slot in enumerate(SLOTS):
        if slot in slots:
            compare(case.label, slot, result[i], expected[i], mass[i], sparse=sparse)
    return result


@pytest.mark.parametrize("label", BP1)
def test_gpu_bp1(label):
    case = fixture("bp1", label)
    require_gpu()
    check(case)


@pytest.mark.parametrize("label", BP2)
def test_gpu_bp2(label):
    case = fixture("bp2", label)
    require_gpu()
    check(case)


@pytest.mark.parametrize("label", tuple(label for label in BP1 if "small" in label))
def test_gpu_biasless(label):
    case = fixture("bp1", label, bias=False)
    require_gpu()
    check(case)


@pytest.mark.parametrize("op,lane", product(("conv1d", "conv_transpose1d"), ("gemm", "tree")))
def test_gpu_tiled_dw_lanes(op, lane):
    # BWD-REV3 (Claude, 2026) identified these two native coverage gaps.
    # This test pins the reach of each geometry before comparing all slots.
    require_gpu()
    transposed = op == "conv_transpose1d"
    groups = 16 if lane == "tree" else 2
    case = synthetic(op, f"{op}-{lane}-tiled", n=2, channels=16, width=1049, kernel=3,
                     groups=groups, pads=(1, 1),
                     **({"stride": 2, "output_padding": 1} if transposed else {}))
    time = case.x.shape[-1] if transposed else case.seed.shape[-1]
    per_group = case.w.size // groups
    assert case.x.shape[0] == case.seed.shape[0] == 2 and time > 1024
    if transposed:
        assert case.native_attrs[:4] == (2, 1, 1, 1)
    scratch = ts._C._conv_backward_scratch(transposed, case.x.shape, case.w.shape,
                                           case.seed.shape, case.native_attrs)
    if lane == "gemm":
        assert case.native_attrs[-1] == 2 and per_group > 64
        assert scratch == (16 // groups * 3 * 1024, per_group)
    else:
        assert case.native_attrs[-1] == case.x.shape[1] == case.seed.shape[1] and per_group == 3
        assert scratch == (0, 0)
    print(f"R3_TILED_DW {case.label} n=2 groups={groups} time={time} coefs={per_group} scratch={scratch}",
          flush=True)
    check(case)


@pytest.mark.parametrize("op", ("conv1d", "conv_transpose1d"))
def test_gpu_cancel_unit_and_structural_zeros(op):
    require_gpu()
    case = synthetic(op, "unit-and-cancellation", n=2, channels=4, width=7, kernel=3, stride=3, dilation=2,
                     groups=2, pads=(4, 1), **({"output_padding": 2} if op != "conv1d" else {}))
    x, w, b, g = case.x, case.w, case.b, case.seed
    x.fill(0)
    x[:, 0, ::2] = 1
    w.fill(0)
    w[0, 0, 0] = 1
    w[1, 0, 0] = -1
    w[2, 0, 0] = 1
    g.fill(0)
    g[:, :2, :,] = 1
    g[1] = -g[0]
    g[0, 2, ::2] = 1
    result = check(case, sparse=True)
    expected, mass = answer(case)
    for slot, got, exp, scale in zip(SLOTS, result, expected, mass):
        assert (scale == 0).any() or slot == "b", slot
        np.testing.assert_array_equal(got, exp.astype(np.float32))
    assert any(((exp == 0) & (scale > 0)).any() for exp, scale in zip(expected, mass))
    assert any((abs(exp) == 1).any() for exp in expected)


def canary():
    width = 128
    case = synthetic("conv1d", "tf32-canary", n=1, channels=128, width=width, kernel=1)
    i, j = np.indices((128, width))
    bump = np.float32(1 + 3 * 2.0 ** -13)
    return replace(case, x=(np.exp2((i + j) % 3) * bump)[None].astype(np.float32),
                   seed=(np.exp2((i + 2 * j) % 3) * bump)[None].astype(np.float32))


def rounded_tf32(x):
    # NVIDIA Ampere TF32 (2020), RNE rounding to ten fraction bits.
    bits = x.view(np.uint32)
    return ((bits + np.uint32(0xFFF) + ((bits >> 13) & 1)) & np.uint32(0xFFFFE000)).view(np.float32)


def test_host_canary_gap():
    case = canary()
    expected, mass = answer(case)
    rounded, _ = answer(replace(case, x=rounded_tf32(case.x), seed=rounded_tf32(case.seed)))
    ratio = (abs(rounded[1] - expected[1]) / allowed(expected[1], mass[1])).max()
    print(f"R3_CANARY tf32/fp32 dW gap={ratio:.9g} bounds", flush=True)
    assert ratio >= 4


def test_gpu_canary():
    require_gpu()
    assert "NVIDIA_TF32_OVERRIDE" not in os.environ
    test_gpu_configuration()
    test_host_canary_gap()
    check(canary(), slots=("w",))


def test_gpu_configuration():
    require_gpu()
    cfg = ts._C._conv_backward_config(0)
    print(f"R3_CUBLAS {cfg}", flush=True)
    assert all(cfg[k] is True for k in ("legacy_stream", "pedantic_compute", "pedantic_math",
                                        "atomics_not_allowed", "host_pointers", "default_algo"))
    assert cfg["workspace_bytes"] == cfg["scratch_cap_bytes_per_buffer"] == 32 * 2**20
    assert cfg["tile_time"] == 1024 and cfg["cublas_version"] > 0 and cfg["sm_count"] > 0


@pytest.mark.parametrize("variant", ("gemm", "tree", "gemm-batch-tail", "tree-batch-tail"))
def test_gpu_tile_merge(variant):
    require_gpu()
    # BWD-R3 (2026) cancellation witness, extended from BWD-REV3's review.
    # A single nonzero product per partial makes only the merge order matter.
    tree = variant.startswith("tree")
    batches = 2 if variant.endswith("batch-tail") else 1
    groups = 16 if tree else 1
    width = 1024 + 3 if batches == 2 else 3 * 1024
    case = synthetic("conv1d", f"ordered-merge-{variant}", n=batches, channels=16,
                     width=width, kernel=1, groups=groups)
    case.x.fill(0)
    case.seed.fill(0)
    locations = ((0, 0), (0, width - 1), (1, 0)) if batches == 2 else ((0, 0), (0, 1024), (0, 2048))
    for (batch, time), term in zip(locations, (2**24, 1, -(2**24))):
        case.x[batch, 0, time] = term
        case.seed[batch, 0, time] = 1
    last_batch, last_time = batches - 1, width - 1 if batches == 2 else 0
    case.x[last_batch, 1, last_time] = 2
    witness_channel = 1 if tree or batches == 2 else 0
    case.seed[last_batch, witness_channel, last_time] = 1
    col, part = ts._C._conv_backward_scratch(False, case.x.shape, case.w.shape, case.seed.shape, case.native_attrs)
    assert (col, part) == ((0, 0) if tree else (16 * 1024, 16 * 16))
    _, dw, _ = gradients(case, (False, True, False))
    target = np.zeros(case.w.shape, np.float32)
    target[witness_channel, 0 if tree else 1, 0] = 2
    print(f"R3_MERGE ordered witness {variant} actual={dw[0, 0, 0]}", flush=True)
    assert dw.size and np.ptp(dw) > 0 and np.ptp(target) > 0
    exact(dw, target)
