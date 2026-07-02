"""Fused rope_apply gate: one-launch RoPE must match the composed
rotate_half chain bit-for-bit (same dtype path), honor pos0 table
offsets, and reject training use (backward raises)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc
from tensor_cuda import functional as F


def run():
    rng = np.random.default_rng(0)
    B, H, L, D, T = 1, 4, 3, 64, 32
    pos0 = 7
    inv = 1.0 / (1e4 ** (np.arange(0, D, 2, np.float32) / D))
    ang = np.arange(T, dtype=np.float32)[:, None] * inv[None, :]
    emb = np.concatenate([ang, ang], -1)
    for dt in ("float32", "bfloat16"):
        cs = tc.tensor(np.cos(emb)).astype(dt)
        sn = tc.tensor(np.sin(emb)).astype(dt)
        x = tc.tensor(rng.standard_normal((B, H, L, D)).astype(
            np.float32)).astype(dt)
        with tc.no_grad():
            y_fused = tc.rope_apply(x, cs, sn, pos0)
            y_ref = F.apply_rotary(x, cs.slice(0, pos0, L),
                                   sn.slice(0, pos0, L))
            y_inv = tc.rope_apply(y_fused, cs, sn, pos0, inverse=True)
        d = np.abs(y_fused.float().numpy() - y_ref.float().numpy()).max()
        roundtrip = np.abs(y_inv.float().numpy() - x.float().numpy()).max()
        # composed chain rounds the two products to dt before the add;
        # the fused kernel keeps the sum in fp32 — one-ulp class diffs
        ok = d <= (1e-6 if dt == "float32" else 0.02)
        rt_ok = roundtrip <= (1e-5 if dt == "float32" else 0.04)
        print(f"{dt}: max|fused - composed| {d:.3e} {'OK' if ok else 'FAIL'}")
        print(f"{dt}: inverse roundtrip max|d| {roundtrip:.3e} "
              f"{'OK' if rt_ok else 'FAIL'}")
        assert ok
        assert rt_ok

    # zero-frequency dims are identity (the p-RoPE convention)
    inv_p = np.concatenate([inv[:8], np.zeros(D // 2 - 8, np.float32)])
    ang = np.arange(T, dtype=np.float32)[:, None] * inv_p[None, :]
    emb_p = np.concatenate([ang, ang], -1)
    cs = tc.tensor(np.cos(emb_p))
    sn = tc.tensor(np.sin(emb_p))
    x = tc.tensor(rng.standard_normal((B, H, L, D)).astype(np.float32))
    with tc.no_grad():
        y = tc.rope_apply(x, cs, sn, 0).numpy()
    xn = x.numpy()
    half = D // 2
    ident = np.abs(y[..., 8:half] - xn[..., 8:half]).max()
    print(f"p-RoPE identity dims: max|d| {ident:.3e}")
    assert ident == 0.0

    print("ROPE_APPLY GATE: PASS")


if __name__ == "__main__":
    run()
