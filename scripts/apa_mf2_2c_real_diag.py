"""APA-MF2.2c G4: the review's real/Qwen3-1.7B/L16/bfloat16/L3 row, one engine per process.
Records the SDPA output bytes and compares with (a) the review's fp64 oracle (exact operands) and
(b) a bf16-storage-emulating oracle: scores rounded to bf16 as the bf16 matmul stores them, causal
softmax in fp64 on those scores, weights rounded to bf16, PV in fp64, output rounded to bf16.
usage: apa_mf2_2c_real_diag.py <arm> <out.json>. Mine."""
import hashlib, json, sys
import numpy as np, tensor_cuda as tc
from tensor_cuda import functional as F
CAP = "/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_spq3/artifacts/apa_spq3/captures/Qwen3-1.7B/"


def bf16(x):
    x = np.asarray(x, np.float32).copy(); u = x.view(np.uint32)
    u += np.uint32(0x7FFF) + ((u >> 16) & 1)          # round-to-nearest-even to bf16
    u &= np.uint32(0xFFFF0000); return x


def decode(a): return (a.astype(np.uint32) << 16).view(np.float32)[None]


arm, out = sys.argv[1], sys.argv[2]
q, k, v = (decode(np.load(CAP + f"L16_{n}.npy", mmap_mode="r")[:1, -3:, :] if n == "q" else np.load(CAP + f"L16_{n}.npy", mmap_mode="r")[:1]) for n in "qkv")
D = q.shape[-1]; S = k.shape[-2]; L = 3; scale = D ** -0.5
xs = [tc.tensor(x, dtype="bfloat16") for x in (q, k, v)]
with tc.no_grad():
    o = F.scaled_dot_product_attention(*xs, is_causal=True).float().numpy()
q64, k64, v64 = (x.astype(np.float64)[0, 0] for x in (q, k, v))
vis = np.arange(S)[None, :] <= (S - L + np.arange(L))[:, None]
def attn(s, wround=None):
    s = np.where(vis, s, -np.inf); m = s.max(-1, keepdims=True); w = np.exp(s - m); w /= w.sum(-1, keepdims=True)
    if wround: w = wround(w).astype(np.float64)
    return w @ v64
s64 = q64 @ k64.T * scale
exact = attn(s64)
s32 = (q.astype(np.float32)[0, 0] @ k.astype(np.float32)[0, 0].T * np.float32(scale)).astype(np.float32)
emu = bf16(attn(bf16(s32).astype(np.float64), bf16)).astype(np.float64)
idx = [(1, 39), (1, 62)]
ok = (np.abs(o - exact) <= 0.02 + 0.06 * np.abs(exact))
rec = {"arm": arm, "engine_sha256": hashlib.sha256(open(tc._C.__file__, "rb").read()).hexdigest(),
       "functional_sha256": hashlib.sha256(open(F.__file__, "rb").read()).hexdigest(),
       "output_sha256": hashlib.sha256(o.astype(np.float32).tobytes()).hexdigest(),
       "outside_review_tolerance_vs_fp64": int((~ok).sum()), "elements": int(o.size),
       "max_abs_err_vs_fp64": float(np.abs(o - exact).max()), "max_abs_err_vs_bf16_emulation": float(np.abs(o - emu).max()),
       "emulation_vs_fp64_max_abs": float(np.abs(emu - exact).max()),
       "at_review_indices": {f"[0,0,{r},{c}]": {"engine": float(o[0, 0, r, c]), "fp64": float(exact[r, c]), "bf16_emulation": float(emu[r, c])} for r, c in idx},
       "score_abs_max_visible_row1": float(np.abs(np.where(vis, s64, 0))[1].max())}
json.dump(rec, open(out, "x"), indent=1); print(json.dumps(rec, indent=1))
