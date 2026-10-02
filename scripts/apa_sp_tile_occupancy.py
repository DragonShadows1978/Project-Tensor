"""APA-SP tile statistics for the kernel-optimization proposal (CPU only).

Evidence class: CPU emulation on the SP1 synthetic generator (q, k ~ N(0,1),
kq = k + 0.1*N(0,1); scripts/apa_sp1_gpu.py:84) with SP1 frozen deltas from
artifacts/apa_sp1/calibration.json. This establishes nothing about real-model
attention, whose scores are structured (sinks, locality); it answers three
design questions for docs/APA_SP_KERNEL_OPT_PROPOSAL.md:

  1. How many tensor-core tiles contain zero refined keys (could skip the
     exact QK^T MMA)?
  2. How many key tiles raise a row's running bulk maximum (need the full
     in-tile prefix scan; the rest can use the carry alone)?
  3. For decode, how much extra refinement does SP1.1's partition-local
     prefix cost versus SP1's whole-row prefix?

Prior art: running-max selection criterion — BLASST (Yuan et al., arXiv
2512.12087), reproduced by APA-SP1. Records of an iid sequence grow like
H_n ~ ln n (Chandler 1952; Renyi 1962) — unverified, lead to check
"record values iid expected number harmonic". Tile shapes are the
FlashAttention-2 (Dao 2023) warp/CTA tiles. Tabulation code is ours.

Usage: python3 scripts/apa_sp_tile_occupancy.py  (numpy only, ~2 min)
"""
from __future__ import annotations

import numpy as np

F = np.float32
TILES = [(16, 16), (16, 64), (64, 64), (128, 64)]


def masks(q, kq, scale, delta, causal):
    S = kq.shape[0]
    sel = np.zeros((q.shape[0], S), bool)
    raised = np.zeros((q.shape[0], S // 64), bool)
    for r0 in range(0, q.shape[0], 512):
        bulk = (q[r0:r0 + 512] @ kq.T) * scale
        rows = np.arange(r0, min(q.shape[0], r0 + 512))[:, None]
        vis = np.arange(S)[None, :] <= rows if causal else np.ones_like(bulk, bool)
        bulk = np.where(vis, bulk, -np.inf).astype(F)
        pm = np.maximum.accumulate(bulk, axis=1)
        sel[r0:r0 + 512] = (bulk >= pm - F(delta)) & vis
        # A 64-key tile needs the full in-tile scan only if it raises the
        # carry (tile max > running max entering the tile).
        tile_max = bulk.reshape(len(rows), S // 64, 64).max(-1)
        carry_in = np.concatenate(
            [np.full((len(rows), 1), -np.inf, F), pm[:, 63:-1:64]], axis=1)
        raised[r0:r0 + 512] = (tile_max > carry_in) & vis[:, ::64]
    vis_all = np.tri(q.shape[0], S, dtype=bool) if causal else np.ones((q.shape[0], S), bool)
    return sel, raised, vis_all


def prefill(S, D, causal, delta, heads, seed):
    rng = np.random.default_rng(seed)
    scale = F(1 / np.sqrt(D))
    empty = {t: [0, 0] for t in TILES}
    n_sel = n_vis = n_raise = n_tiles = w_raise = w_tiles = 0
    for _ in range(heads):
        q = rng.standard_normal((S, D), dtype=F)
        k = rng.standard_normal((S, D), dtype=F)
        kq = k + F(0.1) * rng.standard_normal(k.shape, dtype=F)
        sel, raised, vis = masks(q, kq, scale, delta, causal)
        n_sel += sel.sum(); n_vis += vis.sum()
        n_raise += raised.sum(); n_tiles += vis[:, ::64].sum()
        # A warp owns 16 rows in lockstep: it pays the scan if any row raises.
        w_raise += raised.reshape(S // 16, 16, -1).any(1).sum()
        w_tiles += vis[:, ::64].reshape(S // 16, 16, -1).any(1).sum()
        for br, bc in TILES:
            m = sel.reshape(S // br, br, S // bc, bc).any(axis=(1, 3))
            v = vis.reshape(S // br, br, S // bc, bc).any(axis=(1, 3))
            empty[(br, bc)][0] += (v & ~m).sum(); empty[(br, bc)][1] += v.sum()
    print(f"prefill S={S} D={D} causal={int(causal)} delta={delta} heads={heads}")
    print(f"  refined/visible = {n_sel / n_vis:.4f}")
    for t, (e, n) in empty.items():
        print(f"  {t[0]}x{t[1]} tiles with zero refined keys: {e}/{n}")
    print(f"  64-key row-tiles that raise the running bulk max: "
          f"{n_raise}/{n_tiles} = {n_raise / n_tiles:.4f}; "
          f"16-row warp tiles with any raising row: {w_raise}/{w_tiles} = "
          f"{w_raise / w_tiles:.4f}")


def decode(S, D, delta, draws, seed, part=2048):
    rng = np.random.default_rng(seed)
    scale = F(1 / np.sqrt(D))
    glob = local = 0
    for _ in range(draws):
        q = rng.standard_normal((1, D), dtype=F)
        k = rng.standard_normal((S, D), dtype=F)
        kq = k + F(0.1) * rng.standard_normal(k.shape, dtype=F)
        bulk = ((q @ kq.T) * scale).astype(F)[0]
        glob += (bulk >= np.maximum.accumulate(bulk) - F(delta)).sum()
        b = bulk.reshape(-1, part)
        local += (b >= np.maximum.accumulate(b, axis=1) - F(delta)).sum()
    print(f"decode S={S} D={D} delta={delta} draws={draws}: refined fraction "
          f"whole-row prefix (SP1) = {glob / (draws * S):.4f}, "
          f"2048-key partition prefix (SP1.1) = {local / (draws * S):.4f}")


if __name__ == "__main__":
    prefill(2048, 64, True, 1.75, heads=2, seed=0)
    prefill(2048, 64, False, 2.0, heads=2, seed=1)
    prefill(8192, 128, True, 2.25, heads=1, seed=2)
    decode(8192, 128, 2.45, draws=64, seed=3)
    decode(32768, 128, 2.75, draws=32, seed=4)
