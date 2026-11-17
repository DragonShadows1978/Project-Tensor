"""ENG3 exploration probe (GPU, Build B, device tensors only; not a gate): records the device cast bits the E3 host
rules must reproduce, above all the NaN outputs of the cvt instructions, whose behaviour the CUDA headers delegate to
PTX (cuda_fp16.hpp:550 cvt.rn.f16.f32, :709 cvt.f32.f16; cuda_bf16.hpp:475 cvt.rn.bf16.f32 on sm_80+, :588
mov.b32 {0,h} for bf16->f32 below sm_90). Output: artifacts/eng3/probe/device_probe.npz.

Usage (inside forge-gpu-run): python3 -B -P scripts/eng3_probe_device.py

Prior art: no prior art known to me for the probe itself; it measures the PTX cvt semantics (CUDA PTX ISA, cvt).
"""
import os
import sys

sys.path.insert(0, os.path.abspath("tensor_cuda"))
import numpy as np  # noqa: E402
import tensor_cuda as tc  # noqa: E402

out = {}
rng = np.random.default_rng(20261116)

# float16: every bit pattern
h_all = np.arange(65536, dtype=np.uint32).astype(np.uint16)
th = tc.tensor(h_all.view(np.float16), device="cuda", dtype="float16")
out["h_all_in"] = th.numpy().view(np.uint16)
out["h_all_to_f32"] = th.astype("float32").numpy().view(np.uint32)
out["h_all_to_bf16_as_f32"] = th.astype("bfloat16").numpy().view(np.uint32)
out["h_all_to_u8"] = th.astype("uint8").numpy()

# float32: specials (NaN payloads, signalling NaNs), every bf16-pattern (low 16 bits zero), random patterns
nans = np.array([0x7FC00000, 0xFFC00000, 0x7F800001, 0x7FBFFFFF, 0x7FC01234, 0xFFFFE000, 0x7FFFFFFF, 0x7FA00000,
                 0x7F810000, 0x7FC0FFFF, 0xFF812345], dtype=np.uint32)
bf_patterns = (np.arange(65536, dtype=np.uint32) << 16).astype(np.uint32)
rand = rng.integers(0, 2**32, size=200000, dtype=np.uint64).astype(np.uint32)
f_in = np.concatenate([nans, bf_patterns, rand])
tf = tc.tensor(f_in.view(np.float32), device="cuda", dtype="float32")
out["f_in"] = tf.numpy().view(np.uint32)
out["f_to_f16"] = tf.astype("float16").numpy().view(np.uint16)
tb = tf.astype("bfloat16")
out["f_to_bf16_as_f32"] = tb.numpy().view(np.uint32)
out["f_to_u8"] = tf.astype("uint8").numpy()
# bf16 sources made on the device from the patterns above: bf16 -> f16 / u8
out["bf_to_f16"] = tb.astype("float16").numpy().view(np.uint16)
out["bf_to_u8"] = tb.astype("uint8").numpy()

# uint8: every value
u_all = np.arange(256, dtype=np.uint8)
tu = tc.tensor(u_all, device="cuda", dtype="uint8")
out["u_to_f32"] = tu.astype("float32").numpy().view(np.uint32)
out["u_to_f16"] = tu.astype("float16").numpy().view(np.uint16)
out["u_to_bf16_as_f32"] = tu.astype("bfloat16").numpy().view(np.uint32)

os.makedirs("artifacts/eng3/probe", exist_ok=True)
np.savez("artifacts/eng3/probe/device_probe.npz", **out)
print("module", tc._C.__file__)
for k, v in out.items():
    print(k, v.dtype, v.shape)
