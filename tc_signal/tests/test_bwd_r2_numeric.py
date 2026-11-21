"""Elementwise GPU gradient gates against the sealed Voice reference."""

import numpy as np
import pytest

import tc_signal as ts
import tensor_cuda as tc
from bwd_r2_support.cases import (
    CASES, SLOTS, exact, fixture_root, gpu_args, identity_affine, load_cell, nonconstant,
    oracle_call, reference_inputs, require_gpu, signal_forward, tolerances,
)
from bwd_r2_support.oracle import instance_norm1d, instance_norm1d_vjp


def compare(slot, actual, expected, kind, *, sum_abs=None, source="fp64 replay"):
    nonconstant(expected)
    nonconstant(actual)
    assert actual.dtype == np.float32 and actual.shape == expected.shape
    limit = tolerances()[kind]
    atol = limit["atol"] if sum_abs is None else np.maximum(
        limit["atol_floor"], limit["sum_abs_factor"] * sum_abs)
    err = np.abs(actual.astype(np.float64) - expected)
    bound = atol + limit["rtol"] * np.abs(expected)
    worst = np.unravel_index(np.argmax(err / bound), err.shape)
    relative = np.divide(err, np.abs(expected), where=expected != 0, out=np.zeros_like(err))
    nrmse = np.linalg.norm(err.ravel()) / np.linalg.norm(expected.ravel())
    print(f"{source} gradient {slot} shape={actual.shape} max_abs={err.max():.9g} "
          f"max_rel={relative.max():.9g} normalized_l2={nrmse:.9g} "
          f"worst_index={worst} worst_ratio={err[worst] / bound[worst]:.9g}", flush=True)
    assert np.all(err <= bound), f"gradient {slot} exceeded {kind} at {worst} ({source})"


def adain_sum_abs(args, attrs, seed):
    normalized = instance_norm1d(args["x"], args["scale"], args["bias"], **attrs)
    dg = (seed * normalized).sum(axis=-1)
    dbeta = seed.sum(axis=-1)
    dh = np.abs(np.concatenate((dg, dbeta), axis=1))
    return {"s": dh @ np.abs(args["W"]), "W": dh.T @ np.abs(args["s"]), "b": dh.sum(axis=0)}


def assert_row_sum(dx, x, scale, h, eps):
    centered = (x - x[..., :1])
    centered = centered - centered.mean(axis=-1, keepdims=True)
    rstd = 1 / np.sqrt((centered * centered).mean(axis=-1) + eps)
    h_scale = 1 if scale is None else scale[None, :]
    bound = tolerances()["row_sum"]["sum_abs_factor"] * np.sum(np.abs(h), axis=-1) * np.abs(h_scale) * rstd
    total = dx.sum(axis=-1, dtype=np.float64)
    assert np.all(np.abs(total) <= bound), "instance norm dX row sum exceeded fp32 roundoff"


@pytest.mark.parametrize(("bp", "op", "case"), CASES, ids=[f"bp{b}-{op}-{c}" for b, op, c in CASES])
def test_gpu_voice_vjps(bp, op, case):
    fixture_root(bp)
    require_gpu()
    row, data = load_cell(bp, op, case)
    args, seed = reference_inputs(row, data, rounded=True)
    if op in ("instance_norm1d", "adain") and (case == "kokoro" or bp == 2):
        assert identity_affine(args)
        forms = (False, True)
    else:
        forms = (identity_affine(args),)
    for omit_affine in forms:
        print(f"BP{bp} {op} {case} affine={'None' if omit_affine else 'tensors' if 'scale' in args else 'n/a'}",
              flush=True)
        check_vjps(op, row, data, args, seed, omit_affine=omit_affine)


def check_vjps(op, row, data, args, seed, *, omit_affine):
    attrs = row["attrs"]
    nonconstant(seed)
    slots = SLOTS.get(op, ("x",))
    key = dict(zip(slots, oracle_call(op, args, attrs, seed)))
    leaves = gpu_args(tc, op, args, active=True, omit_affine=omit_affine)
    y = signal_forward(tc, ts, op, leaves, attrs)
    assert y.requires_grad and y.shape == seed.shape
    (y * tc.tensor(seed)).sum().backward()
    raw_args, raw_seed = reference_inputs(row, data)
    replay_sums = adain_sum_abs(args, attrs, seed) if op == "adain" else {}
    stored_sums = adain_sum_abs(raw_args, attrs, raw_seed) if op == "adain" else {}
    for slot in slots:
        assert leaves[slot].grad is not None, f"missing gradient {slot}"
        actual = leaves[slot].grad.numpy()
        kind = ("matmul" if slot != "x" else
                "instance_norm_dx" if op in ("instance_norm1d", "adain") else "nearest_reflection")
        compare(slot, actual, key[slot], kind, sum_abs=replay_sums.get(slot))
        compare(slot, actual, data["vjp_" + slot], kind, sum_abs=stored_sums.get(slot), source="Voice NPZ")
        if slot == "x" and op == "instance_norm1d":
            assert_row_sum(actual, args["x"], args["scale"], seed, attrs["eps"])


def dx_of(forward, x, seed):
    leaf = tc.tensor(x, requires_grad=True)
    y = forward(leaf)
    (y * tc.tensor(seed)).sum().backward()
    return y.numpy(), leaf.grad.numpy()


@pytest.mark.parametrize("op", ("upsample_nearest1d", "reflection_pad1d"))
def test_gpu_exact_integer(op):
    require_gpu()
    x = np.arange(-8, 8, dtype=np.float32).reshape(2, 2, 4)
    if op == "upsample_nearest1d":
        forward, attrs = lambda t: ts.upsample_nearest1d(t, 2), {"factor": 2}
    else:
        forward, attrs = lambda t: ts.reflection_pad1d(t, (3, 3)), {"pads": (3, 3)}
    out = oracle_call(op, {"x": x}, attrs)
    seed = np.arange(out.size, dtype=np.float32).reshape(out.shape) % 13 - 6
    seed[0, 0] = 0.0
    expected, = oracle_call(op, {"x": x}, attrs, seed)
    _, actual = dx_of(forward, x, seed)
    nonconstant(expected)
    nonconstant(actual)
    exact(actual, expected.astype(np.float32))
    exact(actual[0, 0], np.zeros(4, dtype=np.float32))


def test_gpu_nearest_ascending_order():
    require_gpu()
    x = np.array([1., 2.], dtype=np.float32).reshape(1, 1, 2)
    seed = np.array([2**24, 1, -2**24, 2, -3, 0, 0, -1], dtype=np.float32).reshape(1, 1, 8)
    _, dx = dx_of(lambda t: ts.upsample_nearest1d(t, 4), x, seed)
    exact(dx, np.array([[[2, -4]]], dtype=np.float32))
    _, dx = dx_of(lambda t: ts.upsample_nearest1d(t, 1), x, x)
    exact(dx, x)


def test_gpu_reflection_direct_left_right_order():
    require_gpu()
    x = np.arange(4, dtype=np.float32)
    seed = np.array([0, 0, 1, 0, 2**24, 0, 5, 0, -2**24, 0], dtype=np.float32)
    _, dx = dx_of(lambda t: ts.reflection_pad1d(t, (3, 3)), x, seed)
    exact(dx, np.array([0, 0, 0, 5], dtype=np.float32))
    _, dx = dx_of(lambda t: ts.reflection_pad1d(t, (0, 0)), x, x)
    exact(dx, x)


def test_gpu_norm_constant_and_large_offset():
    require_gpu()
    eps = 1e-5
    x = np.array([[[13, 13, 13, 13], [16384, 16384.004, 16384.03125, 16384.015625]]], dtype=np.float32)
    g = np.array([[[2, -2, 1, -5], [-1, 3, 4, -2]]], dtype=np.float32)
    _, dx = dx_of(lambda t: ts.instance_norm1d(t, eps=eps), x, g)
    ref, _, _ = instance_norm1d_vjp(x, np.ones(2), np.zeros(2), g, eps=eps)
    compare("x", dx, ref, "instance_norm_dx")
    assert_row_sum(dx, x.astype(np.float64), None, g, eps)
    y, dx = dx_of(lambda t: ts.instance_norm1d(t, eps=eps), x[:, :1, :], np.ones_like(x[:, :1, :]))
    exact(y, np.zeros_like(y))
    exact(dx, np.zeros_like(dx))


@pytest.mark.parametrize("eps", (1e-5, 0.))
def test_gpu_norm_singleton_and_singularity(eps):
    require_gpu()
    x = np.array([[[3.], [7.]]], dtype=np.float32)
    y, dx = dx_of(lambda t: ts.instance_norm1d(t, eps=eps), x, x)
    if eps:
        exact(y, np.zeros_like(y))
        exact(dx, np.zeros_like(dx))
    else:
        assert np.isnan(y).all() and np.isnan(dx).all()
