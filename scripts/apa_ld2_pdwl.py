"""APA-LD2 detector additions on top of APA-LD1's PDWL core (scripts/apa_ld1_pdwl.py, imported, not edited).

(1) STAGGERED FRONTIER CUTS (part A, small leaks).
    LD1's cut form tags V positions j >= t (or j < t, window mirror) and requires rows whose TRUE visible
    set misses the tagged set to be bitwise invariant and every other row to change. One cut tests ONE
    row boundary, so LD1's two cuts cannot see a leak confined to one row (a single tile edge).
    LD2 tests EVERY row boundary of the true visibility -- t = hi_r + 1 for r = -1..L-2 (causal frontier)
    and t = lo_r (window left edge, mirror side) -- and packs the cuts one per (batch, kv-head) slab per
    run: V slabs are independent operands of a correct kernel, so B*KVH boundaries are tested per run.
    Runs = 1 + ceil(#cuts / (B*KVH)) (+1 repeat run for the determinism noise floor).
    Slabs that carry no cut in a run are all-invariant controls.

(2) ZERO-WEIGHT STATISTICS TAG, "ZWS" (part B, selection-statistics masks).
    A V-tag cannot see a statistics-only mask: the masked keys stay visible. ZWS tags K instead, on
    "ghost" keys whose softmax weight is EXACTLY zero both before and after tagging, so the tag reaches
    the output ONLY through the refine statistics (mean + z*std of |bulk| over the statistics set):
      * operands: every query row and every ordinary key share a direction e (scores ~ +0.86*A); ghost
        keys (positions >= t_g) are anti-aligned (scores ~ -0.86*A). With A = 80 every ghost score is
        > 100 below the row max, so exp(s - m) underflows to exactly 0 (fp32 expf / __expf ftz) in every
        kernel, before and after the tag. Checked per cell by a V-tag on the ghost block (must be
        invariant on every row) on the untagged and the tagged K.
      * tag: K (and the fused kernel's kq) at ghost positions >= t scaled by 2^E (exact).
      * correct kernel: a row whose TRUE statistics set (= its visible set) contains a tagged ghost sees
        |bulk_ghost| grow by 2^E, so mean + z*std jumps above every ordinary |bulk|: every ordinary key
        that was refined (ranking score) falls back to its bulk score -> the row MUST change, when an
        fp64 emulation of the definition (from the exact operands the kernel sees; keys inside the tie
        band may take either score, bounded by an output interval) proves the output moves by more than 4 ulps + a score-error allowance. A row whose
        visible set contains no tagged position MUST be bitwise invariant (row-local kernel, identical
        row operands; ghosts are masked or zero-weight).
      * statistics-mask bug (fd32b6d quant.py:174 top-left stats; CUDA stats range j < i+1): for rows
        i < t <= S-L+i the tagged ghost is visible but outside the buggy statistics set, so the selection
        is unchanged and, the ghost weight being 0, the row output is bitwise unchanged: a must-change
        row that does not change -> "blind" violation -> ALARM.

Prior art (Prior Art Directive; external items unverified -- lead to check; search terms: "Goldberg 1991
floating-point power of two scaling exact", "Chen Cheung Yiu 1998 metamorphic testing", "DeMillo Lipton
Sayward 1978", "FlashAttention-2 bottom-right causal mask", "group testing Dorfman 1943", "IEEE 754
exp underflow flush to zero"):
  - OpenAI (2026) openai/math #107 Prop 3.1 / #135 via the TX2 Opus M4a dossier (MUTATIONS.md M4) and
    APA-LD1: TAKEN -- exact 2^E tags, untag/invariance bitwise oracle, the cut predicate, LD1's operand
    regime, pre/postcheck and provable tagged-mass rule (reused unchanged through apa_ld1_pdwl).
  - Goldberg (1991): power-of-two scaling is exact absent over/underflow. TAKEN: the survival argument.
  - Chen, Cheung & Yiu (1998), metamorphic testing: two runs of the same kernel as the oracle. TAKEN.
  - Dorfman (1943) group testing / parallel independent probes: related idea for packing several tests
    into one run; not taken verbatim.
  - Dao (2023) FlashAttention-2: the bottom-right convention that the frontier cuts check.
  - MINE (APA-LD2): the per-slab staggered packing of every row boundary; the zero-weight ghost
    construction that isolates the statistics path (exact softmax underflow as the separator); the
    ghost-block V-tag precheck; the must-change rule by fp64 emulation with a sound output interval over the
    uncertain base selections; the sensitivity-maximising V for ZWS.
"""
from __future__ import annotations

import math
import time

import numpy as np

import apa_ld1_pdwl as P

# ------------------------------------------------------------------ (A) staggered frontier cuts


def frontier_cuts(lo, hi, S):
    """Every row boundary of the TRUE visibility (rows identical across slabs is asserted):
    ('ge', hi_r + 1) for the causal side, r = -1..L-2 (t = hi_0 makes every row read exactly the tagged
    frontier key), and ('lt', lo_r) for the window side."""
    l0, h0 = lo[0, 0], hi[0, 0]
    if not (np.all(lo == l0) and np.all(hi == h0)):
        raise ValueError("frontier_cuts: rows differ across slabs")
    cuts = set()
    for t in [int(h0[0])] + [int(x) + 1 for x in h0[:-1]]:
        if 1 <= t <= S - 1:
            cuts.add((t, "ge"))
    for x in l0:
        if 1 <= int(x) <= S - 1:
            cuts.add((int(x), "lt"))
    return sorted(cuts, key=lambda c: (c[1], c[0]))


def stagger_plan(cuts, B, KVH):
    """Pack cuts one per (b, g) slab per run. Returns a list of {(b, g): (t, side)}."""
    slabs = [(b, g) for b in range(B) for g in range(KVH)]
    plan = []
    for r0 in range(0, len(cuts), len(slabs)):
        chunk = cuts[r0:r0 + len(slabs)]
        plan.append({slabs[k]: chunk[k] for k in range(len(chunk))})
    return plan


def stagger_cut_check(run, X, lo, hi, n_den, plan, rep, dtype, sigma, *, repeat=True, slack=P.QUANT_SLACK):
    """Cut predicate with a per-slab cut. X: (B, KVH, S, C) operand; output (B, H, L, C'); head h reads
    slab h // rep. lo/hi/n_den: (B, H, L) TRUE visible interval and denominator count."""
    p = P.DT[dtype]
    E = p["E"]
    vmin, vmax = 2.0 ** p["vexp"], 2.0 * 2.0 ** p["vexp"]
    need = 2.0 ** -(p["mant"] - 2)
    t0 = time.perf_counter()
    base = run(X)
    ok, why = P.postcheck(dtype, base)
    res = dict(mode="stagger_cut", runs=1 + len(plan), ncuts=int(sum(len(a) for a in plan)))
    if not ok:
        res.update(verdict="REFUSED-operand", why=why, seconds=round(time.perf_counter() - t0, 4))
        return res
    noise = int(np.count_nonzero(P.bits(run(X)) != P.bits(base))) if repeat else None
    bb = P.bits(base)
    B, H, L = lo.shape
    g_of_h = np.arange(H) // rep
    tl = tb = ti = nmc = nmi = 0
    first, first_cut, leak_rows, blind_rows = None, None, set(), set()
    for assign in plan:
        Xt = X.copy()
        T = np.full((B, H), -1, np.int64)
        GE = np.zeros((B, H), bool)
        for (b, g), (t, side) in assign.items():
            if side == "ge":
                Xt[b, g, t:] *= np.float32(2.0 ** E)
            else:
                Xt[b, g, :t] *= np.float32(2.0 ** E)
            hs = np.where(g_of_h == g)[0]
            T[b, hs] = t
            GE[b, hs] = side == "ge"
        Y = run(Xt)
        changed = np.any(P.bits(Y) != bb, axis=-1)
        Tb, ge, has = T[..., None], GE[..., None], (T >= 0)[..., None]
        a = np.where(ge, np.maximum(lo, Tb), lo)
        b_ = np.where(ge, hi, np.minimum(hi, Tb - 1))
        n_t = np.where(has, np.maximum(0, b_ - a + 1), 0)
        touched = n_t > 0
        mass_lb = n_t * math.exp(-2.0 * slack * sigma) / np.maximum(1, n_den)
        mass_ok = mass_lb * (2.0 ** E - 1.0) * vmin / vmax >= need
        must_inv, must_chg = ~touched, touched & mass_ok
        leak, blind = changed & must_inv, (~changed) & must_chg
        tl += int(leak.sum())
        tb += int(blind.sum())
        ti += int((touched & ~mass_ok).sum())
        nmc += int(must_chg.sum())
        nmi += int(must_inv.sum())
        viol = leak | blind
        if viol.any():
            for idx in np.argwhere(leak):
                leak_rows.add(int(idx[2]))
            for idx in np.argwhere(blind):
                blind_rows.add(int(idx[2]))
            if first is None:
                f = np.argwhere(viol)[0]
                first = [int(x) for x in f]
                first_cut = list(assign[(int(f[0]), int(g_of_h[f[1]]))])
    nv = tl + tb
    res.update(verdict="ALARM" if nv else "PASS", violations=nv, leak_rows=tl, blind_rows=tb, indeterminate_rows=ti,
               must_change_rows=nmc, must_invariant_rows=nmi, first_violation=first, first_violation_cut=first_cut,
               leak_row_ids=sorted(leak_rows)[:64], blind_row_ids=sorted(blind_rows)[:64], repeat_noise=noise,
               deterministic_observed=(noise == 0) if repeat else None, seconds=round(time.perf_counter() - t0, 4))
    return res


# ------------------------------------------------------------------ (B) zero-weight statistics tag (ZWS)

ZWS = dict(
    A=80.0,            # |q|*|k|*scale: ordinary scores ~ +0.86 A, ghost scores ~ -0.86 A
    rho_q=0.3,         # query spread around the shared direction e
    rho_k=0.5,         # key spread around +-e
    E={"float32": 24, "bfloat16": 24, "float16": 7},   # K-tag exponent (fp16: tagged |score| <= max/4)
    tau=2.0 ** -12,    # base-selection tie band, relative to the row's max |bulk| (fp32 dot/stat error << tau)
    ulps=4,            # must-change margin: 4 output ulps ...
    score_err=1e-3,    # ... + 1e-3 * (V range width): kernel-vs-fp64 score/exp error allowance
)


def zws_qk(rng, B, H, KVH, L, S, D, scale, dtype, t_g, A=ZWS["A"]):
    """q (B,H,L,D), k (B,KVH,S,D); keys >= t_g are ghosts (anti-aligned). Rounded to the dtype grid."""
    e = rng.standard_normal(D)
    e /= np.linalg.norm(e)
    r = math.sqrt(A / scale)
    q = e + ZWS["rho_q"] * rng.standard_normal((B, H, L, D)) / math.sqrt(D)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True) * r
    k = e + ZWS["rho_k"] * rng.standard_normal((B, KVH, S, D)) / math.sqrt(D)
    k = k / np.linalg.norm(k, axis=-1, keepdims=True) * r
    k[:, :, t_g:] *= -1.0
    return P.round_to(q, dtype), P.round_to(k, dtype)


def zws_cuts(L, S):
    """ghost block start t_g and the K-tag cuts (each >= t_g): t_g itself (every row that sees a ghost
    sees a tagged one) and the interior cut (S-L) + L//2 (rows below it must stay invariant)."""
    t_g = S - max(L, 3)
    if t_g < 1:
        # APA-REGT1 B04 fix (the one retune the port order allows). The registered rule above gives t_g <= 0 at
        # L = S (APA-LD2 cell B04, the square control), so row 0 saw only ghost keys, its ghost weight was not 0,
        # and the ghost precheck refused the cell (REFUSED-operand). Corrected ghost block = the one APA-LD2's
        # exploratory rerun used (scripts/apa_ld2_explore.py @ cb2fac1; PASS on main and on PYSTAT at L = S = 33):
        # the block starts at row L//2's key position, so every row keeps >= 1 ordinary key, and the cuts are
        # {t_g, t_g + L//4}. Every cell with t_g >= 1 (every registered cell except B04) is unchanged.
        t_g = (S - L) + L // 2
        return t_g, sorted({t_g, min(S - 1, t_g + L // 4)})
    cuts = sorted({t_g, min(S - 1, max(t_g, (S - L) + L // 2))})
    return t_g, cuts


def zws_precheck(dtype, k, E, A=ZWS["A"]):
    """operand validity: tagged |k| and tagged |score| bound within max/4; returns (ok, why)."""
    mx = P.DT[dtype]["maxv"]
    kt = float(np.max(np.abs(k))) * 2.0 ** E
    st = 1.5 * A * 2.0 ** E
    if kt > mx / 4 or st > mx / 4:
        return False, f"tagged |k| {kt:.3g} or |score| bound {st:.3g} > max/4"
    return True, f"tagged |k| <= {kt:.3g}; tagged |score| <= {st:.3g}"


def _ulp(x, mant):
    x = np.maximum(np.abs(x), 1e-300)
    return 2.0 ** (np.floor(np.log2(x)) - (mant - 1))


def zws_emulate(q, kq, k, V, lo, hi, scale, z, rep, score_round, bh_list):
    """fp64 emulation of the APA refine DEFINITION (population mean / centered variance of |bulk| over the
    visible set, sel = |bulk| >= mean + z*std; refined -> ranking score q.k, else bulk q.kq) for the
    (b, h) slices in bh_list. Returns per slice: selection data for the must-change interval rule.
    score_round: callable rounding a score array to the kernel's score storage dtype."""
    out = {}
    S = k.shape[2]
    j = np.arange(S)
    for b, h in bh_list:
        g = h // rep
        Q = q[b, h].astype(np.float64)
        bulk = scale * (Q @ kq[b, g].astype(np.float64).T)
        rank = scale * (Q @ k[b, g].astype(np.float64).T)
        vis = (j[None, :] >= lo[b, h][:, None]) & (j[None, :] <= hi[b, h][:, None])
        a = np.where(vis, np.abs(bulk), 0.0)
        cnt = vis.sum(1, keepdims=True)
        mean = a.sum(1, keepdims=True) / cnt
        var = (np.where(vis, a - mean, 0.0) ** 2).sum(1, keepdims=True) / cnt
        thr = mean + z * np.sqrt(var)
        amax = a.max(1, keepdims=True)
        sel = vis & (a >= thr)
        unc = vis & (np.abs(a - thr) <= ZWS["tau"] * amax)
        out[(b, h)] = dict(bulk=score_round(bulk), rank=score_round(rank), vis=vis, sel=sel, unc=unc,
                           V=V[b, g].astype(np.float64))
    return out


def _row_interval(d):
    """per row (L, VD) output interval [y_lo, y_hi] over EVERY assignment of the uncertain keys (each may take
    its ranking or its bulk score); certain keys take sel ? ranking : bulk. V >= 0 and w > 0, so
    N_lo / D_hi <= y <= N_hi / D_lo is sound."""
    vis, sel, unc = d["vis"], d["sel"], d["unc"]
    r = np.where(vis, d["rank"], -np.inf)
    bk = np.where(vis, d["bulk"], -np.inf)
    m = np.maximum(r, bk).max(1, keepdims=True)
    wr, wb = np.exp(r - m), np.exp(bk - m)
    wc = np.where(sel, wr, wb)
    wlo = np.where(unc, np.minimum(wr, wb), wc)
    whi = np.where(unc, np.maximum(wr, wb), wc)
    V = d["V"]
    return (wlo @ V) / whi.sum(1, keepdims=True), (whi @ V) / wlo.sum(1, keepdims=True)


def zws_must_change(em_base, em_tag, dtype):
    """Per (b, h): (L,) bool must-change, (L,) bool indeterminate, (L,) separation. The base and tagged
    outputs are bounded by intervals that cover every assignment of their uncertain keys; must-change iff
    the intervals are separated, in some column, by more than ulps * ulp(output) + score_err * (V width)."""
    p = P.DT[dtype]
    width = 2.0 ** p["vexp"]
    res = {}
    for key, db in em_base.items():
        blo, bhi = _row_interval(db)
        tlo, thi = _row_interval(em_tag[key])
        sep = np.maximum(np.maximum(blo - thi, tlo - bhi), 0.0).max(1)
        ymax = np.maximum(np.abs(bhi), np.abs(thi)).max(1)
        thr = ZWS["ulps"] * _ulp(ymax, p["mant"]) + ZWS["score_err"] * width
        must = sep > thr
        res[key] = (must, ~must, sep)
    return res


def zws_sensitive_v(em_base, B, KVH, S, VD, rep, dtype, rng):
    """V on LD1's exact grid, chosen per key so that the tag-induced selection collapse moves the output one
    way: v = top of the grid for keys the emulated base selection refines on most rows of the slab's first
    head (their weight falls after the collapse: ranking -> bulk, and the 2-bit bulk is shrunk ~0.88x), bottom
    of the grid otherwise; plus a small grid jitter in the low bits so columns differ. (A first version keyed
    on sign(rank - bulk), which is + for almost every key, gave a near-constant V: dry-run finding, pre-seal.)"""
    vexp = P.DT[dtype]["vexp"]
    V = P.make_v(rng, (B, KVH, S, VD), dtype)
    for (b, h), d in em_base.items():
        if h % rep:
            continue
        g = h // rep
        nvis = np.maximum(1, d["vis"].sum(0))
        hi_ = (d["sel"].sum(0) / nvis) > 0.5
        n = rng.integers(0, 8, size=(S, VD))
        V[b, g] = np.where(hi_[:, None], (1.0 + (120 + n) / 128.0), (1.0 + n / 128.0)) * 2.0 ** vexp
    return V.astype(np.float32)


# =================================================================================================
# Engine adapter + ZWS cell runner (APA-REGT1 port), LINE SUBSETS of two fork files @ cb2fac1 (APA-LD2); every
# retained line is verbatim, lines were only removed:
#   * class Attn2 from scripts/apa_ld2_entries.py: the injected-(q, k) constructor, run_kv and engine_kq (the ZWS
#     plumbing). Removed: the win_off1 / tile_edge mutant wrappers (_mask_t, _grm_g4_sliding, tile_rows, _sub and
#     the tile_edge branch of run) and make_zattn (it imported apa_ld1_run.GEO): the ported tests use only the
#     offby1 mutant, which lives in Attn._kv_layout (scripts/apa_ld1_pdwl.py).
#   * zws_cell from scripts/apa_ld2_run.py: unchanged (the campaign runner around it is not ported).
# The fork files imported apa_ld1_entries as EN and this module as P2; both now live in modules already imported
# here, so the aliases below keep those lines verbatim (EN -> apa_ld1_pdwl, which carries Attn / up / down /
# expand_heads / tc / F / Q since the port; P2 -> this module).
# Prior art: this module's header (ZWS, staggered cuts); the adapters are APA-LD1/LD2's own (no prior art known
# to me for them).
# =================================================================================================
import sys as _sys  # noqa: E402

EN = P
P2 = _sys.modules[__name__]


class Attn2(EN.Attn):
    """LD1 Attn with the LD2 mutants and optional injected (q, k)."""

    def __init__(self, *a, inject=None, **kw):
        self._inject = inject
        super().__init__(*a, **kw)

    def _prep(self):
        if self._inject is not None:
            self.q, self.k = self._inject
        super()._prep()

    # ---- ZWS: run with a replacement K (and kq for the fused kernels), V given
    def run_kv(self, k_np, kq_np, V):
        old = (self.k_t, getattr(self, "kq_t", None))
        rep, how = self._heads_for_kernel()
        self.k_t = EN.up(EN.expand_heads(k_np, rep, how), self.dtype)
        if kq_np is not None and hasattr(self, "kq_t"):
            self.kq_t = EN.up(EN.expand_heads(kq_np, rep, how), self.dtype)
        try:
            return self.run(V)
        finally:
            self.k_t = old[0]
            if old[1] is not None:
                self.kq_t = old[1]

    def engine_kq(self, k_np):
        """the quantized keys the kernel actually uses for its statistics (exact values):
        apa_sel: the kq operand (LD1 _prep: chunked Q._quantize_keys over KVH heads, rounded to dtype);
        apa_quant: Q._quantize_keys of the head-expanded key in the operand dtype (quant.py:290)."""
        if self.family == "apa_quant":
            rep, how = self._heads_for_kernel()
            kx = EN.expand_heads(k_np, rep, how)
            R, CB, BND = EN.Q._tables(self.D, 2, self.H, True, "cuda")
            return EN.down(EN.Q._quantize_keys(EN.up(kx, self.dtype), R, CB, BND)), kx
        R, CB, BND = EN.Q._tables(self.D, 2, self.KVH, True, "cuda")
        parts = []
        for s0 in range(0, k_np.shape[2], 8192):
            parts.append(EN.down(EN.Q._quantize_keys(EN.up(k_np[:, :, s0:s0 + 8192], "float32"), R, CB, BND)))
        return P.round_to(np.concatenate(parts, axis=2), self.dtype), k_np


def zws_cell(e, t_g, cuts, rng, bh):
    """run the ZWS tag on one cell (operands injected). Returns the result row."""
    dt, rep = e.dtype, e.rep
    E = P2.ZWS["E"][dt]
    t0 = time.perf_counter()
    ok, why = P2.zws_precheck(dt, e.k, E)
    if not ok:
        return dict(verdict="REFUSED-operand", why=why)
    z = EN.Q._norm_ppf(1.0 - 0.15)
    lo, hi, _ = e.intervals()
    B, H, L = lo.shape
    bh = bh or [(b, h) for b in range(B) for h in range(H)]
    fused = e.family != "apa_quant"
    round_s = (lambda x: x) if fused else (lambda x: P.round_to(x.astype(np.float32), dt).astype(np.float64))
    kq0, kx0 = (e.kq, e.k) if fused else e.engine_kq(e.k)
    em0 = P2.zws_emulate(e.q, kq0, kx0, np.zeros((B, kx0.shape[1], e.S, e.VD)), lo, hi, e.scale, z,
                         1 if not fused else rep, round_s, bh)
    V = P2.zws_sensitive_v(em0, B, e.KVH, e.S, e.VD, rep, dt, rng)
    Vx = EN.expand_heads(V, rep, "repeat") if not fused else V
    for key in em0:
        b, h = key
        em0[key]["V"] = Vx[b, h if not fused else h // rep].astype(np.float64)
    kq_in = getattr(e, "kq", None)
    base = e.run_kv(e.k, kq_in, V)
    pok, pwhy = P.postcheck(dt, base)
    if not pok:
        return dict(verdict="REFUSED-operand", why=pwhy)
    noise = int(np.count_nonzero(P.bits(e.run_kv(e.k, kq_in, V)) != P.bits(base)))
    bb = P.bits(base)
    Ev = P.DT[dt]["E"]
    Vg = V.copy()
    Vg[:, :, t_g:] *= np.float32(2.0 ** Ev)
    pre_base = int(np.count_nonzero(np.any(P.bits(e.run_kv(e.k, kq_in, Vg)) != bb, axis=-1)))
    out = dict(mode="zws", t_g=t_g, cuts=cuts, E=E, precheck=why, repeat_noise=noise,
               ghost_vtag_changed_rows_untagged=pre_base, per_cut=[])
    runs = 3
    tl = tb = ti = nmc = nmi = 0
    first = None
    for t in cuts:
        kt = e.k.copy()
        kt[:, :, t:] *= np.float32(2.0 ** E)
        kqt = None
        if kq_in is not None:
            kqt = kq_in.copy()
            kqt[:, :, t:] *= np.float32(2.0 ** E)
        Y = e.run_kv(kt, kqt, V)
        runs += 1
        changed = np.any(P.bits(Y) != bb, axis=-1)
        touched = hi >= t                                 # row's TRUE visible (= statistics) set holds a tagged ghost
        kq1, kx1 = (kqt, kt) if fused else e.engine_kq(kt)
        em1 = P2.zws_emulate(e.q, kq1, kx1, np.zeros((B, kx1.shape[1], e.S, e.VD)), lo, hi, e.scale, z,
                             1 if not fused else rep, round_s, bh)
        for key in em1:
            em1[key]["V"] = em0[key]["V"]
        mc = P2.zws_must_change(em0, em1, dt)
        must_chg = np.zeros_like(touched)
        indet = touched.copy()
        dmin = []
        for (b, h), (m, ind, dm) in mc.items():
            must_chg[b, h] = m & touched[b, h]
            indet[b, h] = touched[b, h] & ~m
            dmin += [float(x) for x in dm[touched[b, h] & np.isfinite(dm)]]
        must_inv = ~touched
        leak, blind = changed & must_inv, (~changed) & must_chg
        nl, nb = int(leak.sum()), int(blind.sum())
        tl, tb, ti = tl + nl, tb + nb, ti + int(indet.sum())
        nmc, nmi = nmc + int(must_chg.sum()), nmi + int(must_inv.sum())
        viol = leak | blind
        if viol.any() and first is None:
            first = [int(x) for x in np.argwhere(viol)[0]] + [int(t)]
        # ghost weight zero also under the tag (V-tag on the ghost block of the tagged run)
        if t == cuts[0]:
            Vy = P.bits(Y)
            pre_tag = int(np.count_nonzero(np.any(P.bits(e.run_kv(kt, kqt, Vg)) != Vy, axis=-1)))
            runs += 1
            out["ghost_vtag_changed_rows_tagged"] = pre_tag
        band = touched & (np.arange(L)[None, None, :] < t)     # rows a top-left statistics set would blind
        out["per_cut"].append(dict(t=int(t), leak_rows=nl, blind_rows=nb, must_change_rows=int(must_chg.sum()),
                                   must_invariant_rows=int(must_inv.sum()), indeterminate_rows=int(indet.sum()),
                                   changed_rows=int(changed.sum()), touched_rows=int(touched.sum()),
                                   band_rows_must_change=int((band & must_chg).sum()),
                                   dmin_min=(min(dmin) if dmin else None), dmin_median=(float(np.median(dmin)) if dmin else None)))
    if out["ghost_vtag_changed_rows_untagged"] or out.get("ghost_vtag_changed_rows_tagged"):
        verdict = "REFUSED-operand"
        out["why"] = "ghost weights not exactly zero (V-tag on the ghost block changed rows)"
    else:
        verdict = "ALARM" if (tl + tb) else "PASS"
    out.update(verdict=verdict, violations=tl + tb, leak_rows=tl, blind_rows=tb, must_change_rows=nmc,
               must_invariant_rows=nmi, indeterminate_rows=ti, first_violation=first, runs=runs,
               seconds=round(time.perf_counter() - t0, 4))
    return out
