"""Centered fp64 differences against key slots and actual candidate slots."""
import numpy as np
import pytest

from bwd_r3_support.cases import TOL, fixture, gradients, reference, require_gpu, synthetic

FD = (("bp1", "conv1d_small3"), ("bp1", "conv_transpose1d_small3"),
      ("bp2", "conv1d_12"), ("bp2", "conv_transpose1d_01"))


# Fornberg (1988) centered finite differences. R3 projects both the
# stored or actual GPU VJP and the same fixed fp64 objective.
def directions(value, rng, count, exhaustive):
    for _ in range(count):
        vector = rng.standard_normal(value.shape)
        yield vector / np.linalg.norm(vector.ravel())
    positions = range(value.size) if exhaustive else (0, value.size - 1)
    for pos in positions:
        vector = np.zeros_like(value)
        vector.flat[pos] = 1
        yield vector


def probe(case, gradients_, *, exhaustive=False):
    policy = TOL["finite_difference"]
    rng = np.random.default_rng(policy["direction_seed"])
    seed = case.seed.astype(np.float64)
    inputs = tuple(a.astype(np.float64) for a in case.inputs)
    scale = 1 / np.sqrt(seed.size)
    forward = getattr(reference, case.op)

    def phi(xs):
        return (forward(*xs, **case.attrs) * seed).sum() * scale

    for slot, (value, gradient) in enumerate(zip(inputs, gradients_)):
        assert gradient.shape == value.shape and gradient.size > 0 and np.isfinite(gradient).all()
        h = np.finfo(np.float64).eps ** (1 / 3) * max(1.0, np.sqrt(np.mean(value**2)))
        worst, floor = 0.0, 0.0
        for i, direction in enumerate(directions(value, rng, policy["directions_per_slot"], exhaustive)):
            expected = (gradient.astype(np.float64) * direction).sum() * scale
            bound = policy["absolute"] + policy["relative"] * abs(expected)
            estimates = []
            for factor in policy["multipliers"]:
                step = h * factor
                plus, minus = list(inputs), list(inputs)
                plus[slot], minus[slot] = value + step * direction, value - step * direction
                estimates.append((phi(plus) - phi(minus)) / (2 * step))
            error = max(abs(result - expected) for result in estimates)
            plateau = max(estimates) - min(estimates)
            worst, floor = max(worst, error / bound), max(floor, plateau)
            assert max(error, plateau) <= bound, (case.label, slot, i, expected, estimates, bound)
        print(f"R3_FD {case.label} slot={slot} directions={i+1} step={h:.9g} "
              f"scale={scale:.9g} plateau_max={floor:.9g} bound_ratio_max={worst:.9g}", flush=True)


@pytest.mark.parametrize("bp,label", FD)
def test_host_fd_key(bp, label):
    case, key = fixture(bp, label, dtype=np.float64, key=True)
    probe(case, key)


@pytest.mark.parametrize("bp,label", FD)
def test_gpu_fd_candidate(bp, label):
    case = fixture(bp, label)
    require_gpu()
    probe(case, gradients(case))


@pytest.mark.parametrize("op", ("conv1d", "conv_transpose1d"))
def test_gpu_fd_all_tiny_coordinates(op):
    require_gpu()
    case = synthetic(op, "tiny", n=1, channels=2, width=4, kernel=2, stride=2, pads=(1, 0))
    probe(case, gradients(case), exhaustive=True)
