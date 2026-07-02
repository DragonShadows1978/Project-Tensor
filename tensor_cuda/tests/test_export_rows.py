"""cache export gates for GRM arena payload slicing.

export_rows is the raw payload export primitive. export_rope_rows is the
positional payload export primitive: slice rows and apply forward/inverse RoPE
in one CUDA launch.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc
from tensor_cuda import functional as F


def _rope_tables(t, d):
    inv = 1.0 / (1e4 ** (np.arange(0, d, 2, np.float32) / d))
    ang = np.arange(t, dtype=np.float32)[:, None] * inv[None, :]
    emb = np.concatenate([ang, ang], -1)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def _pair_swap_np(x):
    *pre, d = x.shape
    return x.reshape(pre + [d // 2, 2]).swapaxes(-1, -2).reshape(x.shape)


def run():
    f_np = np.arange(1 * 2 * 9 * 4, dtype=np.float32).reshape(1, 2, 9, 4)
    f = tc.tensor(f_np)
    with tc.no_grad():
        f_out = tc.export_rows(f, 2, 3, 4)
    np.testing.assert_array_equal(f_out.numpy(), f_np[:, :, 3:7])

    u_np = (np.arange(1 * 1 * 8 * 5, dtype=np.uint8).reshape(1, 1, 8, 5)
            + 3)
    u = tc.tensor(u_np, dtype="uint8")
    with tc.no_grad():
        u_out = tc.export_rows(u, 2, 2, 3)
    np.testing.assert_array_equal(u_out.numpy(), u_np[:, :, 2:5])

    rng = np.random.default_rng(1)
    x_np = rng.standard_normal((1, 3, 8, 64)).astype(np.float32)
    cs_np, sn_np = _rope_tables(16, 64)
    x = tc.tensor(x_np)
    cs, sn = tc.tensor(cs_np), tc.tensor(sn_np)
    with tc.no_grad():
        x_rot = tc.rope_apply(x, cs, sn, 0)
        exp_inv = tc.export_rope_rows(
            x_rot, cs, sn, 2, 2, 4, 2, inverse=True)
        exp_fwd = tc.export_rope_rows(
            x, cs, sn, 2, 2, 4, 2, inverse=False)
        ref_fwd = F.apply_rotary(
            x.slice(2, 2, 4), cs.slice(0, 2, 4), sn.slice(0, 2, 4))
    np.testing.assert_allclose(exp_inv.numpy(), x_np[:, :, 2:6],
                               rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(exp_fwd.numpy(), ref_fwd.numpy(),
                               rtol=1e-6, atol=1e-6)

    # DeepSeek MLA's stored RoPE component uses the pair-swapped layout.
    x_ds_np = rng.standard_normal((1, 1, 7, 64)).astype(np.float32)
    x_ds = tc.tensor(x_ds_np)
    with tc.no_grad():
        rot_ds = tc.rope_apply(x_ds, cs, sn, 0, pair_swap=True)
        inv_ds = tc.export_rope_rows(
            rot_ds, cs, sn, 2, 1, 4, 1, inverse=True, pair_swap=True)
    ref_ds = F.apply_rotary(
        tc.tensor(_pair_swap_np(rot_ds.numpy()[:, :, 1:5])),
        cs.slice(0, 1, 4), sn.slice(0, 1, 4) * -1.0)
    np.testing.assert_allclose(inv_ds.numpy(), ref_ds.numpy(),
                               rtol=1e-6, atol=1e-6)

    raw_np = np.arange(1 * 9 * 5, dtype=np.float32).reshape(1, 9, 5)
    raw = tc.tensor(raw_np)
    with tc.no_grad():
        raw_pair, rope_pair = tc.export_row_pair(
            raw, x_rot, cs, sn, 1, 2, 3, 3, 4, 3, inverse=True)
    np.testing.assert_array_equal(raw_pair.numpy(), raw_np[:, 3:7])
    np.testing.assert_allclose(rope_pair.numpy(), x_np[:, :, 3:7],
                               rtol=1e-6, atol=1e-6)

    raw2_np = raw_np + 100.0
    x2_np = x_np * 0.5
    raw2 = tc.tensor(raw2_np)
    x2 = tc.tensor(x2_np)
    with tc.no_grad():
        x2_rot = tc.rope_apply(x2, cs, sn, 0)
        raws, ropes = tc.export_row_pairs(
            [raw, raw2], [x_rot, x2_rot], cs, sn, 1, 2, [2, 4], [2, 2],
            3, 2, inverse=True)
    assert len(raws) == 2 and len(ropes) == 2
    np.testing.assert_array_equal(raws[0].numpy(), raw_np[:, 2:5])
    np.testing.assert_array_equal(raws[1].numpy(), raw2_np[:, 4:7])
    np.testing.assert_allclose(ropes[0].numpy(), x_np[:, :, 2:5],
                               rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(ropes[1].numpy(), x2_np[:, :, 2:5],
                               rtol=1e-6, atol=1e-6)

    try:
        tc.export_rows(f, 2, 3, 4)
        raise AssertionError("grad guard missing")
    except RuntimeError:
        pass
    try:
        tc.export_row_pair(raw, x_rot, cs, sn, 1, 2, 3, 3, 4, 3,
                           inverse=True)
        raise AssertionError("pair grad guard missing")
    except RuntimeError:
        pass
    try:
        tc.export_row_pairs([raw], [x_rot], cs, sn, 1, 2, [3], [3], 4, 3,
                            inverse=True)
        raise AssertionError("pairs grad guard missing")
    except RuntimeError:
        pass

    print("EXPORT_ROWS GATE: PASS")


if __name__ == "__main__":
    run()
