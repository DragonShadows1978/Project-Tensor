"""InstanceNorm affine-only routing and optional-slot VJPs."""

import numpy as np
import pytest

import tc_signal as ts
import tensor_cuda as tc
from bwd_r2_support.cases import exact, require_gpu


@pytest.mark.parametrize("roles", [("scale",), ("bias",), ("scale", "bias")])
def test_host_norm_affine_only_reaches_planner(roles):
    x = tc.tensor([[[1., -2.], [4., 1.]]], device="cpu")
    affine = {role: tc.tensor([2., -1.], device="cpu", requires_grad=True) for role in roles}
    with pytest.raises(BufferError, match="CUDA"):
        ts.instance_norm1d(x, **affine)
    with pytest.raises(ValueError, match="out= is unsupported when recording gradients"):
        ts.instance_norm1d(x, out=x, **affine)


@pytest.mark.parametrize("roles", [("scale",), ("bias",), ("x", "scale", "bias")])
def test_gpu_norm_affine_optional_slots(roles):
    require_gpu()
    x = tc.tensor([[[1.], [-2.]], [[4.], [1.]]], requires_grad="x" in roles)
    affine = {role: tc.tensor([2., -1.], requires_grad=True) for role in roles if role != "x"}
    seed = tc.tensor([[[2.], [-3.]], [[5.], [1.]]])
    y = ts.instance_norm1d(x, **affine)
    assert y.requires_grad
    (y * seed).sum().backward()
    if "scale" in roles:
        exact(affine["scale"].grad.numpy(), np.zeros(2, dtype=np.float32))
    if "bias" in roles:
        exact(affine["bias"].grad.numpy(), np.array([7., -2.], dtype=np.float32))
    if "x" in roles:
        assert x.grad.shape == x.shape and np.all(x.grad.numpy() == 0)
    else:
        assert x.grad is None


# BWD-R4a r5b item 4: T >= 3, N = 2, non-identity scale and bias, every slot trainable, against an fp64 NumPy
# reference. The bar is the sealed G-AFF criterion (artifacts/bwd_r4/TOLERANCES_R4A_R2.json "g_aff", DESIGN section 6):
# |got - ref| <= atol + rtol * |ref|, with dx atol 5e-5 and rtol 3e-4, and for scale and bias
# atol = max(2e-6, 8e-6 * S_c) and rtol 5e-5, S_c the per-channel fp64 sum of |dy * xhat| (scale) or |dy| (bias).
# Prior art: Ulyanov et al. (2016) instance normalization; Ioffe and Szegedy (2015) the normalization and affine chain
# rule; Higham (2002) centered differences for the reference's own host check. Ours: these operands and this test.
G_AFF = dict(dx_atol=5e-5, dx_rtol=3e-4, affine_atol_floor=2e-6, affine_atol_per_abs_term_sum=8e-6,
             affine_rtol=5e-5)
EPS = 1e-5
RNG_SEED = 20261119


def checked(a):
    # Non-empty, finite and nonconstant before any comparison (L19).
    assert a.size > 0 and np.isfinite(a).all() and np.min(a) < np.max(a), "empty, nonfinite or constant array"


def affine_operands():
    rng = np.random.Generator(np.random.PCG64(RNG_SEED))
    x = rng.standard_normal((2, 3, 7)).astype(np.float32) * np.float32(1.5) + np.float32(0.25)
    scale = np.array([1.75, -0.6, 2.4], dtype=np.float32)
    bias = np.array([0.3, -1.2, 0.85], dtype=np.float32)
    dy = rng.standard_normal((2, 3, 7)).astype(np.float32)
    return x, scale, bias, dy


def norm_fp64(x, scale, bias, eps=EPS):
    x = x.astype(np.float64)
    mean = x.mean(-1, keepdims=True)
    var = ((x - mean) ** 2).mean(-1, keepdims=True)
    xhat = (x - mean) / np.sqrt(var + eps)
    return xhat * scale.astype(np.float64)[None, :, None] + bias.astype(np.float64)[None, :, None], xhat, var


def vjp_fp64(x, scale, bias, dy, eps=EPS):
    _, xhat, var = norm_fp64(x, scale, bias, eps)
    dy = dy.astype(np.float64)
    h = dy * scale.astype(np.float64)[None, :, None]
    dx = (h - h.mean(-1, keepdims=True) - xhat * (h * xhat).mean(-1, keepdims=True)) / np.sqrt(var + eps)
    dscale = (dy * xhat).sum((0, 2))
    dbias = dy.sum((0, 2))
    term_sums = dict(scale=np.abs(dy * xhat).sum((0, 2)), bias=np.abs(dy).sum((0, 2)))
    return dict(x=dx, scale=dscale, bias=dbias), term_sums


def within_g_aff(name, got, ref, term_sums):
    checked(got)
    checked(ref)
    assert got.shape == ref.shape
    if name == "x":
        bound = G_AFF["dx_atol"] + G_AFF["dx_rtol"] * np.abs(ref)
    else:
        bound = np.maximum(G_AFF["affine_atol_floor"], G_AFF["affine_atol_per_abs_term_sum"] * term_sums[name])
        bound = bound + G_AFF["affine_rtol"] * np.abs(ref)
    error = np.abs(got.astype(np.float64) - ref)
    assert np.all(error <= bound), f"{name}: worst error/bound {float(np.max(error / bound))}"
    return float(np.max(error / bound))


def test_host_affine_fp64_reference_matches_centered_differences():
    # The fp64 reference VJP itself, checked against centered differences of the fp64 forward on the same operands,
    # so the GPU test below compares against a verified reference.
    x, scale, bias, dy = (a.astype(np.float64) for a in affine_operands())
    grads, _ = vjp_fp64(x, scale, bias, dy)
    rng = np.random.Generator(np.random.PCG64(RNG_SEED + 1))
    operands = dict(x=x, scale=scale, bias=bias)
    for name, base in operands.items():
        checked(grads[name])
        for _ in range(4):
            v = rng.standard_normal(base.shape)
            v /= np.linalg.norm(v)
            h = np.cbrt(np.finfo(np.float64).eps) * max(1.0, float(np.linalg.norm(base)))
            slopes = []
            for step in (h, h / 2, 2 * h):
                plus = dict(operands, **{name: base + step * v})
                minus = dict(operands, **{name: base - step * v})
                fp = np.vdot(dy, norm_fp64(plus["x"], plus["scale"], plus["bias"])[0])
                fm = np.vdot(dy, norm_fp64(minus["x"], minus["scale"], minus["bias"])[0])
                slopes.append((fp - fm) / (2 * step))
            exact_projection = float(np.vdot(grads[name], v))
            assert abs(exact_projection) > 1e-3
            for slope in slopes:
                assert abs(slope - exact_projection) <= 1e-7 + 1e-5 * abs(exact_projection)


def test_gpu_norm_affine_fp64_reference_n2_t7():
    require_gpu()
    x_np, scale_np, bias_np, dy_np = affine_operands()
    assert x_np.shape[0] == 2 and x_np.shape[-1] >= 3
    assert not np.all(scale_np == 1) and not np.all(bias_np == 0)
    x = tc.tensor(x_np, requires_grad=True)
    scale = tc.tensor(scale_np, requires_grad=True)
    bias = tc.tensor(bias_np, requires_grad=True)
    y = ts.instance_norm1d(x, scale, bias, eps=EPS)
    assert y.requires_grad
    y_ref, _, _ = norm_fp64(x_np, scale_np, bias_np)
    within_g_aff("x", y.numpy(), y_ref, None)  # the forward value, under the dx class
    (y * tc.tensor(dy_np)).sum().backward()
    grads, term_sums = vjp_fp64(x_np, scale_np, bias_np, dy_np)
    for name, leaf in (("x", x), ("scale", scale), ("bias", bias)):
        assert leaf.grad is not None, f"{name}: gradient missing"
        within_g_aff(name, leaf.grad.numpy(), grads[name], term_sums)


# BWD-R4a r5d (re-check F05 residual): scale-only and bias-only affine cases at N = 2, T = 7 (>= 3), every supplied slot
# and x trainable, against the same fp64 NumPy reference (the missing slot is the identity: scale 1, bias 0) and the
# same sealed G-AFF criterion.
@pytest.mark.parametrize("supplied", ("scale", "bias"))
def test_gpu_norm_affine_single_slot_fp64_reference_n2_t7(supplied):
    require_gpu()
    x_np, scale_np, bias_np, dy_np = affine_operands()
    assert x_np.shape[0] == 2 and x_np.shape[-1] >= 3
    ref_scale = scale_np if supplied == "scale" else np.ones_like(scale_np)
    ref_bias = bias_np if supplied == "bias" else np.zeros_like(bias_np)
    x = tc.tensor(x_np, requires_grad=True)
    slot = tc.tensor(scale_np if supplied == "scale" else bias_np, requires_grad=True)
    y = ts.instance_norm1d(x, **{supplied: slot}, eps=EPS)
    assert y.requires_grad
    y_ref, _, _ = norm_fp64(x_np, ref_scale, ref_bias)
    within_g_aff("x", y.numpy(), y_ref, None)
    (y * tc.tensor(dy_np)).sum().backward()
    grads, term_sums = vjp_fp64(x_np, ref_scale, ref_bias, dy_np)
    assert x.grad is not None and slot.grad is not None
    within_g_aff("x", x.grad.numpy(), grads["x"], term_sums)
    within_g_aff(supplied, slot.grad.numpy(), grads[supplied], term_sums)
