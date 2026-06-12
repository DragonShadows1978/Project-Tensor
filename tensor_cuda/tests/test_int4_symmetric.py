"""Symmetric-8 INT4 (q4_0 import) gate: an EMPTY zeros tensor selects
z = -8*s in all three int4 paths (two-stage dequant, fused tile GEMM,
GEMV). Exactness is checked against the q4_0 grid itself: w = s*(q-8)
must round-trip bit-exactly through dequant, and both matmul paths must
agree with a numpy fp32 reference on those exact weights.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc


def make_q4_0(N, K, group=32, seed=0):
    rng = np.random.default_rng(seed)
    q = rng.integers(0, 16, size=(N, K), dtype=np.uint8)
    s = (rng.random((N, K // group), dtype=np.float32) * 0.02 + 1e-4
         ).astype(np.float16)
    w = ((q.reshape(N, -1, group).astype(np.float32) - 8.0)
         * s.astype(np.float32)[:, :, None]).reshape(N, K)
    packed = (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8)
    return packed, s, w


def run():
    N, K, G = 96, 256, 32
    packed_np, s_np, w_ref = make_q4_0(N, K, G)
    packed = tc.tensor(packed_np, dtype="uint8")
    scales = tc.tensor(s_np, dtype="float16")
    empty_z = tc.tensor(np.zeros((0,), np.float16), dtype="float16")

    # [1] dequant exactness (fp32 out): symmetric path == grid
    w_kn = tc.int4_dequant(packed, scales, empty_z, G, out_dtype="float32")
    d = np.abs(w_kn.numpy().T - w_ref).max()
    assert d == 0.0, f"dequant symmetric mismatch: max|d| {d}"
    print(f"[1] dequant exact on q4_0 grid: max|d| {d}")

    # [2] two-stage linear vs numpy (fp32 x)
    x = np.random.default_rng(1).standard_normal((4, K)).astype(np.float32)
    y_ref = x @ w_ref.T
    y2 = tc.int4_linear(tc.tensor(x), packed, scales, empty_z, G).numpy()
    e2 = np.abs(y2 - y_ref).max() / (np.abs(y_ref).max() + 1e-9)
    assert e2 < 1e-5, f"two-stage rel err {e2}"
    print(f"[2] two-stage linear rel err {e2:.2e}")

    # [3] fused tile GEMM (M=4) and [4] GEMV (M=1)
    y3 = tc.int4_linear_fused(tc.tensor(x), packed, scales, empty_z, G).numpy()
    e3 = np.abs(y3 - y_ref).max() / (np.abs(y_ref).max() + 1e-9)
    assert e3 < 1e-5, f"fused tile rel err {e3}"
    print(f"[3] fused tile GEMM rel err {e3:.2e}")
    y4 = tc.int4_linear_fused(tc.tensor(x[:1]), packed, scales,
                              empty_z, G).numpy()
    e4 = np.abs(y4 - y_ref[:1]).max() / (np.abs(y_ref[:1]).max() + 1e-9)
    assert e4 < 1e-5, f"GEMV rel err {e4}"
    print(f"[4] GEMV rel err {e4:.2e}")

    # [5] asymmetric path unchanged (regression): real zeros tensor
    z_np = (np.random.default_rng(2).random((N, K // G), dtype=np.float32)
            * 0.01).astype(np.float16)
    w_asym = (np.repeat(s_np.astype(np.float32), G, 1)
              * ((packed_np[:, :, None] >> np.array([0, 4]))
                 .reshape(N, K) & 0x0F).astype(np.float32)
              + np.repeat(z_np.astype(np.float32), G, 1))
    zeros = tc.tensor(z_np, dtype="float16")
    w_kn2 = tc.int4_dequant(packed, scales, zeros, G, out_dtype="float32")
    d5 = np.abs(w_kn2.numpy().T - w_asym).max()
    assert d5 == 0.0, f"asym regression: max|d| {d5}"
    print(f"[5] asymmetric path regression: max|d| {d5}")

    print("INT4 SYMMETRIC GATE: PASS")


if __name__ == "__main__":
    run()
