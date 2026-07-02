"""SELECTOR-ACCURACY bench: which key-quantizer gives the most faithful
'which K/V matter' selection, PER HEAD, across percentiles?

The APA bulk scan IS the selector: it ranks keys by |Q . quant(K)^T| and
refines the top-r%. Its selection is only as good as quant(K) preserves
the true Q.K inner product. So 'is the selection accurate?' == 'does the
quantizer preserve the score ranking?'. This measures exactly that.

Ground truth: full-precision |Q . K^T| ranking (the true important keys).
Candidates:
  - turbo : the engine's _quantize_keys (per-head random rotation +
            Lloyd-Max codebook on the unit-sphere Cartesian key)
  - polar : PolarQuant reference (asymmetric n-bit radius / m-bit angle
            on RoPE dimension-pairs; post-RoPE keys)
Metric, PER HEAD, at percentile p:
  - recall@p   : of the true top-p% keys, what frac does the candidate's
                 top-p% also contain? (did we KEEP the keys that matter)
  - mass@p     : softmax-mass-weighted recall (did we keep the keys that
                 hold the ATTENTION, not just the count)
  - spearman   : rank correlation of candidate score vs true score

Synthetic-but-realistic keys: per-head Gaussian with channel outliers +
RoPE applied (PolarQuant assumes post-RoPE). Swap in real Gemma global-
layer K/Q dumps later for the deployed number.

  python3 tests/test_selector_accuracy.py [bits]
"""
import math
import sys

import numpy as np

sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
import tensor_cuda as tc                                    # noqa: E402
from tensor_cuda.quant import _tables, _quantize_keys       # noqa: E402

H, S, D = 16, 512, 128            # heads, keys, head-dim (Gemma global D=512;
L = 64                            # 128 here for a quick first pass)
PERCENTILES = [0.05, 0.10, 0.15, 0.20, 0.30]
BITS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
np.random.seed(0)


def rope(x, base=10000.0):
    """Apply RoPE to (H,S,D) — pairs (i, i+D/2), the standard interleave."""
    Hh, Ss, Dd = x.shape
    half = Dd // 2
    pos = np.arange(Ss)[:, None]
    inv = base ** (-np.arange(0, half) / half)
    ang = pos * inv[None, :]               # (S, half)
    cos = np.cos(ang)[None]                # (1,S,half)
    sin = np.sin(ang)[None]
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)


def polar_quant(k, n_bits, m_bits):
    """PolarQuant reference on post-RoPE keys (H,S,D). Pairs (j, j+D/2)
    as (x,y); quantize radius (n bits, no zero-point, per-channel scale)
    and angle (m bits over (-pi,pi]). Returns dequantized key (H,S,D)."""
    Hh, Ss, Dd = k.shape
    half = Dd // 2
    x, y = k[..., :half], k[..., half:]            # the RoPE pairs
    r = np.sqrt(x * x + y * y)                      # (H,S,half) radius
    th = np.arctan2(y, x)                           # (-pi, pi]
    # radius: n-bit per-channel (per (head,dim)) symmetric, no zero-point
    rmax = np.abs(r).max(axis=1, keepdims=True) + 1e-8     # (H,1,half)
    rq = np.round(r / rmax * (2**n_bits - 1)) / (2**n_bits - 1) * rmax
    # angle: m-bit uniform over (-pi, pi]
    aq = np.round((th + math.pi) / (2*math.pi) * (2**m_bits - 1))
    aq = aq / (2**m_bits - 1) * (2*math.pi) - math.pi
    xq, yq = rq * np.cos(aq), rq * np.sin(aq)
    return np.concatenate([xq, yq], -1)


def turbo_quant(k, bits):
    """Engine TurboQuant via _quantize_keys (per-head rotation+codebook)."""
    kt = tc.tensor(k[None].astype(np.float32))     # (1,H,S,D)
    R, CB, BND = _tables(D, bits, H, True, "cuda")
    return _quantize_keys(kt, R, CB, BND).float().numpy()[0]


def recall_and_mass(true_score, cand_score, p):
    """Per head: recall@p and softmax-mass@p of true-top-p kept by cand-top-p."""
    kth = max(1, int(round(p * S)))
    recalls, masses, spears = [], [], []
    for h in range(H):
        ts, cs = true_score[h], cand_score[h]      # (L,S) per query row
        rec_l, mass_l, sp_l = [], [], []
        for q in range(ts.shape[0]):
            tt = np.argsort(-ts[q])[:kth]           # true top-p keys
            cc = np.argsort(-cs[q])[:kth]           # candidate top-p keys
            keep = np.intersect1d(tt, cc, assume_unique=False)
            rec_l.append(len(keep) / kth)
            # mass: fraction of TRUE softmax mass (over true top-p) that cand kept
            w = np.exp(ts[q] - ts[q].max()); w = w / w.sum()
            mass_l.append(w[keep].sum() / max(w[tt].sum(), 1e-9))
            # spearman over all keys
            sp_l.append(np.corrcoef(
                np.argsort(np.argsort(ts[q])),
                np.argsort(np.argsort(cs[q])))[0, 1])
        recalls.append(np.mean(rec_l)); masses.append(np.mean(mass_l))
        spears.append(np.mean(sp_l))
    return np.array(recalls), np.array(masses), np.array(spears)


# build per-head keys with channel outliers, then RoPE; queries fp
def make_kq():
    k = np.random.randn(H, S, D).astype(np.float32)
    # inject channel outliers (a few dims per head with large magnitude)
    for h in range(H):
        oc = np.random.choice(D, 4, replace=False)
        k[h][:, oc] *= np.random.uniform(4, 9, size=4)[None, :]
    q = np.random.randn(H, L, D).astype(np.float32)
    return rope(k), rope(q)


k_rope, q_rope = make_kq()
scale = 1.0 / math.sqrt(D)
true_score = np.einsum("hld,hsd->hls", q_rope, k_rope) * scale   # (H,L,S) GT

# candidates (4-bit budgets: polar m4n? to match ~BITS total per pair-ish)
k_turbo = turbo_quant(k_rope, BITS)
# polar: split BITS between radius+angle. for BITS=4 use n=2,m=4 (~3-bit eq,
# the paper's m4n2 low-bit point) and also n=4,m=4 (full 4-bit-ish) to bracket.
variants = {
    f"turbo_{BITS}b": k_turbo,
    "polar_n2m4": polar_quant(k_rope, 2, 4),
    "polar_n4m4": polar_quant(k_rope, 4, 4),
    "polar_n3m5": polar_quant(k_rope, 3, 5),
}

print(f"\nSELECTOR ACCURACY | H={H} S={S} D={D} L={L} | bits={BITS}", flush=True)
print(f"{'candidate':12s} {'pctl':>5s} {'recall':>8s} {'mass':>8s} "
      f"{'spear':>8s} {'recall_min_head':>16s}", flush=True)
print("-" * 64, flush=True)
for name, kq in variants.items():
    cand_score = np.einsum("hld,hsd->hls", q_rope, kq) * scale
    for p in PERCENTILES:
        rec, mass, sp = recall_and_mass(true_score, cand_score, p)
        # report mean over heads AND the worst head (retrieval heads matter)
        print(f"{name:12s} {p:5.2f} {rec.mean():8.4f} {mass.mean():8.4f} "
              f"{sp.mean():8.4f} {rec.min():16.4f}", flush=True)
    print(flush=True)
print("DONE", flush=True)
