"""gated_delta_step exactness vs a numpy reference of the HF recurrence
(l2norm inside, q scale, decay-first delta rule), including in-place
state mutation and multi-step sequential consistency."""
import numpy as np
import pytest

import tensor_cuda as tc


def _ref_step(q, k, v, a, b, A_neg, dtb, S):
    B, Hk, Dk = q.shape
    H, Dv = v.shape[1], v.shape[2]
    rep = H // Hk
    out = np.zeros((B, H, Dv), np.float32)
    for bb in range(B):
        for h in range(H):
            kk = k[bb, h // rep].astype(np.float64)
            qq = q[bb, h // rep].astype(np.float64)
            kk = kk / np.sqrt((kk * kk).sum() + 1e-6)
            qq = qq / np.sqrt((qq * qq).sum() + 1e-6) / np.sqrt(Dk)
            av = a[bb, h] + dtb[h]
            sp = max(av, 0.0) + np.log1p(np.exp(-abs(av)))
            alpha = np.exp(A_neg[h] * sp)
            beta = 1.0 / (1.0 + np.exp(-b[bb, h]))
            St = S[bb, h].astype(np.float64)
            St *= alpha
            kv = St.T @ kk
            delta = (v[bb, h] - kv) * beta
            St += np.outer(kk, delta)
            out[bb, h] = (St.T @ qq).astype(np.float32)
            S[bb, h] = St.astype(np.float32)
    return out


@pytest.mark.parametrize("B", [1, 2])
def test_gated_delta_step_exact(B):
    rng = np.random.default_rng(0)
    Hk, H, Dk, Dv = 16, 32, 128, 128
    q = rng.standard_normal((B, Hk, Dk)).astype(np.float32)
    k = rng.standard_normal((B, Hk, Dk)).astype(np.float32)
    v = rng.standard_normal((B, H, Dv)).astype(np.float32)
    a = rng.standard_normal((B, H)).astype(np.float32)
    b = rng.standard_normal((B, H)).astype(np.float32)
    A_neg = (-np.exp(rng.uniform(0, 2, H))).astype(np.float32)
    dtb = rng.standard_normal(H).astype(np.float32)
    S0 = (rng.standard_normal((B, H, Dk, Dv)) * 0.1).astype(np.float32)

    S_dev = tc.tensor(S0.copy())
    outs = []
    S_ref = S0.copy()
    for step in range(3):                      # sequential consistency
        o = tc.gated_delta_step(
            tc.tensor(q + step), tc.tensor(k - step), tc.tensor(v),
            tc.tensor(a), tc.tensor(b), tc.tensor(A_neg), tc.tensor(dtb),
            S_dev)
        outs.append(o.numpy())
        ref = _ref_step(q + step, k - step, v, a, b, A_neg, dtb, S_ref)
        np.testing.assert_allclose(outs[-1], ref, rtol=2e-4, atol=2e-4)
    # in-place state mutation matches the reference trajectory
    np.testing.assert_allclose(S_dev.numpy(), S_ref, rtol=2e-4, atol=2e-4)


def test_gated_delta_step_rejects_bad_dtype():
    z32 = tc.tensor(np.zeros((1, 16, 128), np.float32))
    bad = tc.tensor(np.zeros((1, 32, 128), np.float32)).astype("bfloat16")
    st = tc.tensor(np.zeros((1, 32, 128, 128), np.float32))
    g = tc.tensor(np.zeros((1, 32), np.float32))
    hvec = tc.tensor(np.zeros(32, np.float32))
    with pytest.raises(RuntimeError):
        tc.gated_delta_step(z32, z32, bad, g, g, hvec, hvec, st)
