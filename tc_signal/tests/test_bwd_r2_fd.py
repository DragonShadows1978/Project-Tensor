"""Finite differences of Voice's fp64 forwards for the answer key and candidate."""

import numpy as np
import pytest

from bwd_r2_support.cases import (
    OPS, SLOTS, fixture_root, gpu_args, load_cell, nonconstant, oracle_call, reference_inputs, require_gpu,
    signal_forward, tolerances,
)


def tiny_cell(op):
    x = np.array([[[-.4, .25, .7, -.1], [.6, .45, -.3, .8]]], dtype=np.float64)
    args = {"x": x}
    if op == "upsample_nearest1d":
        return args, {"factor": 2}
    if op == "reflection_pad1d":
        return args, {"pads": (2, 3)}
    args.update(scale=np.array([.8, 1.3]), bias=np.array([.2, -.1]))
    if op == "adain":
        args.update(s=np.array([[.3, -.6]]), W=np.arange(8).reshape(4, 2) / 7 - .7,
                    b=np.array([-.1, .2, .4, -.3]))
    return args, {"eps": 1e-5}


def central_differences(op, args, attrs, seed, slot, direction, scale, multipliers):
    # Fornberg (1988), central finite differences. Use the same Voice fp64
    # objective at each of the three registered steps.
    h = np.finfo(np.float64).eps ** (1 / 3)
    value = args[slot]
    for multiplier in multipliers:
        step = h * multiplier
        plus = oracle_call(op, {**args, slot: value + step * direction}, attrs)
        minus = oracle_call(op, {**args, slot: value - step * direction}, attrs)
        p_hi = scale * np.sum(plus * seed)
        p_lo = scale * np.sum(minus * seed)
        yield multiplier, (p_hi - p_lo) / (2 * step)


def verify_projection(op, args, attrs, seed, slot, direction, expected, scale):
    rule = tolerances()["finite_difference"]
    measure = []
    for multiplier, actual in central_differences(op, args, attrs, seed, slot, direction, scale,
                                                 rule["step_multipliers"]):
        bound = rule["atol"] + rule["rtol"] * abs(expected)
        measure.append(abs(actual - expected) / bound)
        assert np.isfinite(actual) and abs(actual - expected) <= bound, (
            f"{op} {slot} at h*{multiplier}: FD {actual}, VJP {expected}, bound {bound}")
    return max(measure)


def edge_coordinates(op, slot, shape, attrs):
    coords = {tuple(0 for _ in shape), tuple(n - 1 for n in shape)}
    if slot == "x" and op == "reflection_pad1d":
        length = shape[-1]
        left, right = attrs["pads"]
        for index in (1, left, left + 1, length - 2 - right, length - 1 - right, length - 2):
            if 0 <= index < length:
                coords.add((0,) * (len(shape) - 1) + (index,))
    return sorted(coords)


def check_cell(op, args, attrs, seed, *, real):
    rule = tolerances()["finite_difference"]
    scale = rule["real_objective_scale" if real else "tiny_objective_scale"]
    slots = SLOTS.get(op, ("x",))
    rng = np.random.default_rng(20261118)
    refs = oracle_call(op, args, attrs, seed)
    nonconstant(seed)
    for slot, ref in zip(slots, refs):
        nonconstant(ref)
        shape = ref.shape
        coordinates = edge_coordinates(op, slot, shape, attrs) if real else list(np.ndindex(shape))
        worst = 0
        for coord in coordinates:
            direction = np.zeros(shape, dtype=np.float64)
            direction[coord] = 1
            worst = max(worst, verify_projection(op, args, attrs, seed, slot, direction, scale * ref[coord], scale))
        directions = 8 if real else 0
        for _ in range(directions):
            direction = rng.choice((-1.0, 1.0), size=shape) / np.sqrt(ref.size)
            projection = scale * np.sum(ref * direction)
            worst = max(worst, verify_projection(op, args, attrs, seed, slot, direction, projection, scale))
        print(f"FD {'Kokoro' if real else 'tiny'} {op} {slot} shape={shape} "
              f"coordinates={len(coordinates)} directions={directions} max_bound_ratio={worst:.9g}", flush=True)


@pytest.mark.parametrize("op", OPS)
def test_host_tiny_fd(op):
    args, attrs = tiny_cell(op)
    forward = oracle_call(op, args, attrs)
    i = np.arange(forward.size, dtype=np.float64)
    g = (np.cos(i * 1.7) + i / forward.size).reshape(forward.shape)
    check_cell(op, args, attrs, g, real=False)


@pytest.mark.parametrize("op", OPS)
def test_host_kokoro_fd(op):
    row, data = load_cell(1, op, "kokoro")
    args, seed = reference_inputs(row, data, rounded=True)
    check_cell(op, args, row["attrs"], seed, real=True)


def check_candidate_projections(op, args, attrs, seed, slot, grad, case):
    # PyTorch gradcheck's analytical/numerical comparison (Paszke et al.,
    # 2019) and randomized Jacobian probes (JAX check_grads, 2018). Here we
    # use candidate VJPs, eight seeded directions and an independent forward.
    rule = tolerances()["finite_difference"]
    scale = rule["real_objective_scale" if case == "kokoro" else "tiny_objective_scale"]
    multipliers = rule["step_multipliers"]
    rng = np.random.default_rng(20261118)
    seen = set()
    projections, differences = [], []
    while len(seen) < 8:
        signs = rng.choice((-1.0, 1.0), size=grad.shape)
        packed = np.packbits(signs > 0).tobytes()
        if packed in seen:
            continue
        seen.add(packed)
        direction = signs / np.sqrt(grad.size)
        projections.append(scale * np.sum(grad * direction, dtype=np.float64))
        differences.append([value for _, value in
                            central_differences(op, args, attrs, seed, slot, direction, scale, multipliers)])
    projected = np.asarray(projections)
    fd = np.asarray(differences)
    nonconstant(projected)
    for column in fd.T:
        nonconstant(column)
    bound = rule["atol"] + rule["rtol"] * np.abs(projected[:, None])
    ratios = np.abs(fd - projected[:, None]) / bound
    worst = np.unravel_index(np.argmax(ratios), ratios.shape)
    for index, (projection, steps) in enumerate(zip(projected, fd)):
        print(f"candidate FD {case} {op} {slot} direction={index} projection={projection:.12g} "
              f"central_differences={steps.tolist()} ratios={ratios[index].tolist()}", flush=True)
    print(f"candidate FD {case} {op} {slot} shape={grad.shape} directions={len(seen)} "
          f"max_bound_ratio={ratios[worst]:.9g}", flush=True)
    assert np.all(ratios <= 1), (
        f"{case} {op} {slot} direction {worst[0]} at h*{multipliers[worst[1]]}: "
        f"FD {fd[worst]}, candidate VJP {projected[worst[0]]}, bound {bound[worst[0], 0]}")


@pytest.mark.parametrize("case", ("small1", "kokoro"))
@pytest.mark.parametrize("op", OPS)
def test_gpu_candidate_fd(op, case):
    fixture_root(1)
    require_gpu()
    import tc_signal as ts
    import tensor_cuda as tc

    row, data = load_cell(1, op, case)
    args, seed = reference_inputs(row, data, rounded=True)
    nonconstant(seed)
    leaves = gpu_args(tc, op, args, active=True)
    y = signal_forward(tc, ts, op, leaves, row["attrs"])
    assert y.requires_grad and y.shape == seed.shape
    (y * tc.tensor(seed)).sum().backward()
    for slot in ("x", "s") if op == "adain" else ("x",):
        assert leaves[slot].grad is not None, f"missing candidate gradient {slot}"
        grad = leaves[slot].grad.numpy()
        assert grad.dtype == np.float32 and grad.shape == args[slot].shape
        nonconstant(grad)
        check_candidate_projections(op, args, row["attrs"], seed, slot, grad, case)
