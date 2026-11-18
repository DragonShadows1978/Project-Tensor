"""P1-REV1 seeded operands (Project-Tensor, 2026), copied verbatim."""
import numpy as np

def rnd_bf16(x):
    u = np.asarray(x, np.float32).view(np.uint32)
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return u.view(np.float32)


def make_inputs(seed, B, H, KVH, S, D, plant=True, dup=False):
    rng = np.random.default_rng(seed)
    G = H // KVH
    q = rnd_bf16(rng.standard_normal((B, H, 1, D)))
    k = rnd_bf16(0.5 * rng.standard_normal((B, KVH, S, D)))
    v = rnd_bf16(rng.standard_normal((B, KVH, S, D)))
    if plant:
        for b in range(B):
            for kh in range(KVH):
                qd = q[b, kh * G, 0]
                # score q.big*scale ~= 25 for head kh*G (scale = 1/sqrt(D)); background scores ~ N(0, 0.5)
                big = rnd_bf16(qd / np.linalg.norm(qd) * (25.0 * np.sqrt(D) / np.linalg.norm(qd)))
                k[b, kh, 5] = big
                if dup:
                    k[b, kh, 300] = big
                    k[b, kh, 301] = big
    return q, k, v
