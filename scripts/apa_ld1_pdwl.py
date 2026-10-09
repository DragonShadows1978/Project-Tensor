"""APA-LD1: the power-of-two degeneration-weight leak detector (PDWL), harness core.

Detector (TX2 Opus M4a / GPT-6 Astra M04, openai/math #107 Prop 3.1 and #135):
  * a kernel linear in an operand X (V in attention; dO for the dV leg; B in a batched GEMM) is run on
    X tagged by EXACT powers of two keyed to a label (KV head, batch, slab, bank, key-position cut);
  * the output is untagged by the label it SHOULD have read and compared BITWISE with the untagged run;
  * a term read from the wrong label keeps a factor 2^(+-E) != 1 and cannot cancel.
Why bitwise is legitimate: IEEE round-to-nearest commutes with scaling by 2^E (every product, sum,
fma, division by an untagged denominator and every cast keeps its significand) as long as no operand
or result leaves the NORMAL range of its format. The operand-validity precheck/postcheck below
refuses operands for which that is not guaranteed (subnormal products, overflow, near-zero outputs).

Run schedule (Walsh bits): for each bit r of the label, one run tags {bit r = 1} and one tags
{bit r = 0}, positive exponents only (Opus's ledger: negative tags pushed fp16 into subnormals);
plus the untagged reference run: 2*ceil(log2 B) + 1 runs. Two distinct labels differ in some bit,
so some run tags exactly one of them.
Cut form (causal alignment): tag operand positions j >= t (causal) or j < t (mirror, sliding
window). Output rows whose TRUE visible set misses the tagged set must be bitwise invariant; every
other row must change (if its tagged weight mass is provably visible in the output precision).

Prior art (annotation required by the Prior Art Directive; external items unverified -- lead to
check; search terms: "Goldberg 1991 What every computer scientist should know floating-point",
"Chen Cheung Yiu 1998 metamorphic testing", "Freivalds 1977 probabilistic machines",
"DeMillo Lipton Sayward 1978 hints on test data selection", "FlashAttention-2 bottom-right causal"):
  - OpenAI (2026) openai/math #107 Prop 3.1 (degeneration weights w = (g-h)^2, keep the eps^0 part)
    and #135 (IMM derivative/path coefficients), via the TX2 Opus M4a and Astra M04 dossiers
    (/mnt/ForgeRealm/reference/openai-math-notes/receipts/tx2_opus/MUTATIONS.md M4,
    .../tx2_astra/MUTATIONS.md M04). TAKEN: eps^w -> exact 2^(E*w); Fourier phase copies -> Walsh-bit
    runs; the cut predicate; Astra's sparse basis path probes and witness list.
  - Goldberg (1991): exactness of power-of-two scaling under IEEE 754 (absent over/underflow).
    TAKEN: the survival argument; the normal-range operand conditions are its hypotheses.
  - Chen et al. (1998), metamorphic testing: a relation between two runs replaces a reference
    oracle. TAKEN: the oracle shape (tagged run vs untagged run of the SAME kernel).
  - Freivalds (1977): verifying a product cheaply without recomputing it. Related, not used.
  - DeMillo, Lipton & Sayward (1978), mutation testing: seeded defects measure the detector.
  - Dao (2023) FlashAttention-2: the bottom-right causal convention the cut predicate checks.
  MINE (APA-LD1): this engine harness -- the per-dtype exponent/operand regime (positive V in
  [1,2)*2^vexp, |score| <= 3 by construction, E = 64 fp32/bf16, E = 20 fp16), the operand-validity
  pre/postcheck with a provable tagged-mass bound for must-change rows, the kernel-offset estimate
  k_hat = t - (first changed row) from an interior cut boundary, and the dV-leg mirror cut.
"""
from __future__ import annotations

import math
import time

import numpy as np

# ---- operand regime (registered; artifacts/apa_ld1/registration.json "detector") ----------------
DT = {
    # E: tag exponent; vexp: V magnitude 2^vexp*[1,2); mant: significand bits; maxv / minnorm: range
    "float32": dict(E=64, vexp=0, mant=24, maxv=2.0 ** 127, minnorm=2.0 ** -126),
    "bfloat16": dict(E=64, vexp=0, mant=8, maxv=2.0 ** 127, minnorm=2.0 ** -126),
    "float16": dict(E=20, vexp=-8, mant=11, maxv=65504.0, minnorm=2.0 ** -14),
}
SIGMA = 3.0             # |score| <= scale*|q|*|k| <= SIGMA (rows built with exact norms, then rounded)
QUANT_SLACK = 1.5       # bulk / quantized-key scores bounded by QUANT_SLACK*sigma in the precheck
WMIN_LOG2 = -100        # order: every nonzero softmax weight >= 2^-100
OUT_MARGIN = 16.0       # nonzero untagged outputs must be >= OUT_MARGIN * minnorm


def round_to(x, dtype):
    """Round fp32 numpy values to the target dtype's grid (returned as fp32)."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    if dtype == "float32":
        return x
    if dtype == "float16":
        return x.astype(np.float16).astype(np.float32)
    if dtype == "bfloat16":   # round-to-nearest-even on the top 16 bits
        u = x.view(np.uint32).astype(np.uint64)
        u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
        return u.astype(np.uint32).view(np.float32)
    raise ValueError(dtype)


def unit_rows(rng, shape):
    x = rng.standard_normal(shape)
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def make_qk(rng, qshape, kshape, scale, dtype, sigma=SIGMA):
    """q, k rows with |q|*|k|*scale ~= 0.98*sigma (exact norms before dtype rounding)."""
    r = math.sqrt(0.98 * sigma / scale)
    q = round_to(unit_rows(rng, qshape) * r, dtype)
    k = round_to(unit_rows(rng, kshape) * r, dtype)
    return q, k


def make_v(rng, shape, dtype):
    """Positive V on the exact grid 2^vexp * (1 + n/128), n in [0, 128): exact in fp32/bf16/fp16."""
    n = rng.integers(0, 128, size=shape)
    return ((1.0 + n / 128.0) * 2.0 ** DT[dtype]["vexp"]).astype(np.float32)


def score_bound(scale, *rows_sets):
    """sigma = scale * max|q| * max|k| over the given (q, k...) row sets (Cauchy-Schwarz)."""
    q = rows_sets[0]
    qn = float(np.max(np.linalg.norm(q.astype(np.float64), axis=-1)))
    kn = max(float(np.max(np.linalg.norm(k.astype(np.float64), axis=-1))) for k in rows_sets[1:])
    return scale * qn * kn


def precheck(dtype, sigma, n_den_max, slack=QUANT_SLACK):
    """Global operand validity: min weight >= 2^-100 and tagged V within range. Returns (ok, why)."""
    p = DT[dtype]
    log2_wmin = (-2.0 * slack * sigma) / math.log(2.0) - math.log2(max(1, n_den_max))
    vmax = 2.0 * 2.0 ** p["vexp"]
    tagged_max = vmax * 2.0 ** p["E"]
    if log2_wmin < WMIN_LOG2:
        return False, f"min weight 2^{log2_wmin:.1f} < 2^{WMIN_LOG2}"
    if tagged_max > p["maxv"] / 4:
        return False, f"tagged |V| {tagged_max:.3g} > max/4"
    return True, f"min weight >= 2^{log2_wmin:.1f}; tagged |V| <= {tagged_max:.3g}"


def postcheck(dtype, base):
    """Reference-free check on the kernel's OWN untagged output: finite; every nonzero |y| normal
    with margin; tagged |y| (<= |y| * 2^E) in range."""
    p = DT[dtype]
    a = np.abs(base.astype(np.float64))
    if not np.all(np.isfinite(a)):
        return False, "non-finite base output"
    nz = a[a > 0]
    if nz.size and nz.min() < OUT_MARGIN * p["minnorm"]:
        return False, f"base output {nz.min():.3g} below {OUT_MARGIN}*minnorm"
    if a.size and a.max() * 2.0 ** p["E"] > p["maxv"] / 2:
        return False, "tagged output would overflow"
    return True, "ok"


def bits(y):
    y = np.ascontiguousarray(y, dtype=np.float32)
    return y.view(np.uint32)


def walsh_schedule(nlab, E):
    """[zeros] + for each bit r: tag {bit r == 1}, then {bit r == 0}. len = 1 + 2*ceil(log2 nlab)."""
    R = int(math.ceil(math.log2(nlab))) if nlab > 1 else 0
    runs = [np.zeros(nlab, dtype=np.int64)]
    for r in range(R):
        for want in (1, 0):
            runs.append(np.array([E if ((g >> r) & 1) == want else 0 for g in range(nlab)], np.int64))
    return runs


def run_budget(nlab):
    return 2 * (int(math.ceil(math.log2(nlab))) if nlab > 1 else 0) + 1


def _first_index(mask):
    idx = np.argwhere(mask)
    return None if idx.size == 0 else [int(x) for x in idx[0]]


def label_check(run, X, lab_in, lab_out, dtype, *, repeat=True):
    """Walsh-bit head/batch/slab/bank tags.

    run(X) -> Y (float32 numpy). X: (A0, A1, ...) operand whose first two axes carry lab_in[a0, a1];
    Y: (O0, O1, ...) output whose first two axes carry the label they SHOULD have read, lab_out.
    """
    E = DT[dtype]["E"]
    nlab = int(max(lab_in.max(), lab_out.max())) + 1
    sched = walsh_schedule(nlab, E)
    t0 = time.perf_counter()
    base = run(X)
    ok, why = postcheck(dtype, base)
    res = dict(mode="label", nlabels=nlab, runs=len(sched), budget=run_budget(nlab))
    if not ok:
        res.update(verdict="REFUSED-operand", why=why)
        return res
    if repeat:
        rep = run(X)
        noise = int(np.count_nonzero(bits(rep) != bits(base)))
    else:
        noise = None
    nviol, first, run_hit = 0, None, None
    xs = (slice(None),) * 2 + (None,) * (X.ndim - 2)
    ys = (slice(None),) * 2 + (None,) * (base.ndim - 2)
    for ri, c in enumerate(sched[1:], start=1):
        Xt = X * (2.0 ** c[lab_in])[xs].astype(np.float32)
        Y = run(Xt)
        U = Y.astype(np.float64) * (2.0 ** -c[lab_out].astype(np.float64))[ys]
        diff = bits(U.astype(np.float32)) != bits(base)
        n = int(np.count_nonzero(diff))
        if n and first is None:
            first, run_hit = _first_index(diff), ri
        nviol += n
    res.update(verdict="ALARM" if nviol else "PASS", violations=nviol, first_violation=first,
               first_run=run_hit, repeat_noise=noise, deterministic_observed=(noise == 0) if repeat else None,
               seconds=round(time.perf_counter() - t0, 4))
    return res


def cut_check(run, X, axis, lo, hi, cuts, dtype, sigma, n_den, *, repeat=True, slack=QUANT_SLACK):
    """Cut form. X: operand; `axis` = its position axis. lo/hi: (O0, O1, Lout) int arrays, the TRUE
    inclusive interval of operand positions output row r reads (lo > hi: reads nothing).
    cuts: list of (t, side), side 'ge' tags positions >= t, 'lt' tags positions < t.
    n_den: (O0, O1, Lout) denominator count for the tagged-mass lower bound (visible keys, +1 per sink).
    Output Y: (O0, O1, Lout, C); a row 'changed' iff any of its C entries changed bitwise.
    """
    p = DT[dtype]
    E = p["E"]
    vmin, vmax = 2.0 ** p["vexp"], 2.0 * 2.0 ** p["vexp"]
    need = 2.0 ** -(p["mant"] - 2)
    t0 = time.perf_counter()
    base = run(X)
    ok, why = postcheck(dtype, base)
    res = dict(mode="cut", runs=1 + len(cuts), cuts=[list(c) for c in cuts])
    if not ok:
        res.update(verdict="REFUSED-operand", why=why)
        return res
    noise = int(np.count_nonzero(bits(run(X)) != bits(base))) if repeat else None
    bb = bits(base)
    total_leak = total_blind = 0
    per_cut = []
    for t, side in cuts:
        Xt = X.copy()
        sl = [slice(None)] * X.ndim
        sl[axis] = slice(t, None) if side == "ge" else slice(0, t)
        Xt[tuple(sl)] *= np.float32(2.0 ** E)
        Y = run(Xt)
        changed = np.any(bits(Y) != bb, axis=-1)
        if side == "ge":
            a, b_ = np.maximum(lo, t), hi
        else:
            a, b_ = lo, np.minimum(hi, t - 1)
        n_t = np.maximum(0, b_ - a + 1)
        touched = n_t > 0
        mass_lb = n_t * math.exp(-2.0 * slack * sigma) / np.maximum(1, n_den)
        mass_ok = mass_lb * (2.0 ** E - 1.0) * vmin / vmax >= need
        must_inv = ~touched
        must_chg = touched & mass_ok
        indet = touched & ~mass_ok
        leak = changed & must_inv
        blind = (~changed) & must_chg
        viol = leak | blind
        nl, nb = int(np.count_nonzero(leak)), int(np.count_nonzero(blind))
        total_leak += nl
        total_blind += nb
        # alignment: per (o0, o1) the first changed row vs the true first row reading a tagged position
        L_out = changed.shape[-1]
        ar = np.arange(L_out)
        first_true = np.where(touched.any(-1), np.argmax(touched, -1), L_out)
        first_kern = np.where(changed.any(-1), np.argmax(changed, -1), L_out)
        disc = first_kern - first_true
        interior = (first_kern > 0) & (first_kern < L_out)
        rec = dict(t=int(t), side=side, leak_rows=nl, blind_rows=nb, indeterminate_rows=int(np.count_nonzero(indet)),
                   must_change_rows=int(np.count_nonzero(must_chg)), must_invariant_rows=int(np.count_nonzero(must_inv)),
                   first_violation=_first_index(viol),
                   first_violating_row_min=(int(np.min(np.where(viol.any(-1), np.argmax(viol, -1), L_out))) if viol.any() else None),
                   first_changed_row=[int(np.min(first_kern)), int(np.max(first_kern))],
                   true_first_row=[int(np.min(first_true)), int(np.max(first_true))],
                   row_offset_discrepancy=[int(np.min(disc)), int(np.max(disc))],
                   boundary_interior=bool(interior.all()))
        if side == "ge" and interior.any():
            # kernel alignment offset pinned by an INTERIOR boundary: row i reads up to i + k, so the first
            # changed row i* satisfies i* + k_hat = t. True offset k = hi[..., 0] (row 0's last key).
            k_hat = (t - first_kern)[interior]
            k_true = hi[..., 0][interior]
            rec.update(kernel_offset_estimate=[int(k_hat.min()), int(k_hat.max())],
                       true_offset=[int(k_true.min()), int(k_true.max())],
                       offset_discrepancy=[int((k_true - k_hat).min()), int((k_true - k_hat).max())])
        del ar
        per_cut.append(rec)
    nv = total_leak + total_blind
    res.update(verdict="ALARM" if nv else "PASS", violations=nv, leak_rows=total_leak, blind_rows=total_blind,
               per_cut=per_cut, repeat_noise=noise, deterministic_observed=(noise == 0) if repeat else None,
               seconds=round(time.perf_counter() - t0, 4))
    return res


def causal_intervals(B, H, L, S, *, row0=0, Lq=None, window=0, pad=None):
    """TRUE bottom-right visibility: query row i (absolute position S - Lq + row0 + i) reads keys
    j in (pos - window, pos] (window 0 = full causal), minus left-pad keys j < pad[b].
    Returns lo, hi of shape (B, H, L)."""
    Lq = L if Lq is None else Lq
    pos = (S - Lq) + row0 + np.arange(L)
    hi = np.broadcast_to(pos, (B, H, L)).astype(np.int64).copy()
    lo = np.zeros((B, H, L), np.int64) if not window else np.maximum(0, hi - window + 1)
    if pad is not None:
        lo = np.maximum(lo, np.asarray(pad, np.int64)[:, None, None])
    return lo, hi


def full_intervals(B, H, L, S):
    return np.zeros((B, H, L), np.int64), np.full((B, H, L), S - 1, np.int64)


def dv_intervals(B, G, S, L, *, causal=True):
    """dV leg: key j of (b, g) is read by query rows i with S-L+i >= j, i.e. i in [max(0, j-(S-L)), L-1]."""
    j = np.arange(S)
    lo = np.maximum(0, j - (S - L)) if causal else np.zeros(S, np.int64)
    return (np.broadcast_to(lo, (B, G, S)).astype(np.int64).copy(),
            np.full((B, G, S), L - 1, np.int64))


def causal_cuts(L, S):
    """t_a = (S-L) + L//2 pins a bottom-right boundary inside the query block; t_b = max(1, min(L//2,
    S-L)) pins a top-left (offset 0) boundary inside the block. Deduplicated, each in [1, S-1]."""
    ta = min(max(1, (S - L) + L // 2), S - 1)
    tb = min(max(1, min(L // 2, S - L)), S - 1)
    return [(t, "ge") for t in sorted({ta, tb})]


def window_cuts(L, S, W):
    """mirror cut: tag positions < t with t = first visible key of row L//2 (if the window binds)."""
    t = (S - L) + L // 2 - W + 1
    return [(t, "lt")] if 1 <= t <= S - 1 else []


# ---- Astra M04 basis path probes (exact 0/1 expectations; GEMM only) ---------------------------
def basis_paths(B, M, K, N, *, ks=None):
    """Probe paths (b, i, k, j): first/last/tile-boundary k, distinct (i, j) and k per batch so the
    superposed probes have no cross terms. Expected C = sum_p e_(b_p, i_p, j_p) exactly."""
    ks = ks or sorted({0, K - 1, 31 % K, 32 % K, 63 % K, 64 % K, K // 2})
    paths = []
    for b in range(B):
        for n, k in enumerate(ks):
            i = (n * 7 + b) % M
            j = (n * 13 + 3 * b) % N
            paths.append((b, i, k, j))
    # drop collisions (same b with same i or j or k)
    seen, out = set(), []
    for b, i, k, j in paths:
        if (b, "i", i) in seen or (b, "j", j) in seen or (b, "k", k) in seen:
            continue
        seen.update({(b, "i", i), (b, "j", j), (b, "k", k)})
        out.append((b, i, k, j))
    return out


def basis_operands(B, M, K, N, paths, trans_b=False):
    A = np.zeros((B, M, K), np.float32)
    Bm = np.zeros((B, N, K) if trans_b else (B, K, N), np.float32)
    C = np.zeros((B, M, N), np.float32)
    for b, i, k, j in paths:
        A[b, i, k] = 1.0
        if trans_b:
            Bm[b, j, k] = 1.0
        else:
            Bm[b, k, j] = 1.0
        C[b, i, j] = 1.0
    return A, Bm, C


def basis_check(run, A, Bm, C):
    t0 = time.perf_counter()
    Y = run(A, Bm)
    if Y.shape != C.shape:
        return dict(mode="basis", verdict="ALARM", why=f"shape {Y.shape} != {C.shape}", runs=1,
                    seconds=round(time.perf_counter() - t0, 4))
    diff = bits(Y) != bits(C)
    n = int(np.count_nonzero(diff))
    return dict(mode="basis", verdict="ALARM" if n else "PASS", violations=n, first_violation=_first_index(diff),
                runs=1, seconds=round(time.perf_counter() - t0, 4))


# =================================================================================================
# Engine adapter (APA-REGT1 port). Ported from scripts/apa_ld1_entries.py @ cb2fac1 (APA-LD1) as a LINE SUBSET:
# every retained line is verbatim; lines were only removed. Kept: up / down / expand_heads / batch_stride_off1
# and the Attn cell class for the entry families the ported regression tests exercise (sdpa, apa_quant,
# apa_sel) with the Python leak mutants they use (tile, topleft, offby1). The pure-geometry methods
# (_kv_layout, _heads_for_kernel, labels, operand, intervals, cuts) are kept whole, so the TRUE visibility the
# detectors judge against is the fork's, byte for byte.
# Removed, with the reason: load_engine (env-var engine pinning; production tests import the installed
# tensor_cuda like every other test in tensor_cuda/tests); the GRM / GRAPA callers and their hardcoded external
# repo paths (GRM and GRAPA are out of this order's scope); the other entry families, the dV leg (_run_bwd),
# run_grad, _mask_t and _synthetic_margins (which wrote under artifacts/); RaggedBank. Nothing in this section
# touches the file system.
# The adapter lines say `P.<name>` because apa_ld1_entries imported this module as P; the alias below keeps them
# verbatim. The test module sets tc / F / Q to the imported engine before it builds an Attn.
# Prior art: the PDWL references in this module's header; the adapters themselves are APA-LD1's (no prior art
# known to me for them).
# =================================================================================================
import sys as _sys  # noqa: E402

P = _sys.modules[__name__]

tc = None
F = None
Q = None


def up(x, dtype):
    return tc.tensor(np.ascontiguousarray(x, dtype=np.float32), dtype=dtype)


def down(t):
    return np.ascontiguousarray(t.float().numpy(), dtype=np.float32)


def expand_heads(x, rep, how="repeat"):
    """(B, KVH, ...) -> (B, KVH*rep, ...): repeat (h -> h//rep, the correct GQA map) or tile
    (h -> h % KVH, the T4a mutant)."""
    if rep == 1:
        return x
    return np.repeat(x, rep, axis=1) if how == "repeat" else np.tile(x, (1, rep) + (1,) * (x.ndim - 2))


def batch_stride_off1(x):
    """T4a 'batch stride off-by-one': slab b is read from flat offset b*(stride-1) (stride = the
    per-batch element count): batch b >= 1 starts b elements early, reading batch b-1's tail."""
    B = x.shape[0]
    stride = int(np.prod(x.shape[1:]))
    flat = x.reshape(-1)
    out = np.empty_like(x)
    for b in range(B):
        off = b * (stride - 1)
        out[b] = flat[off:off + stride].reshape(x.shape[1:])
    return out


class Attn:
    """One (entry, dtype, shape) cell. Builds q/k/(kq)/v operands, fixed except the tagged one."""

    def __init__(self, family, dtype, B, H, KVH, L, S, D, VD=None, *, causal=True, window=0,
                 scale=None, seed=0, mutant=None, row0=0, Lq=None, pad=None, variant=None, zthr=None):
        self.family, self.dtype, self.B, self.H, self.KVH = family, dtype, B, H, KVH
        self.L, self.S, self.D, self.VD = L, S, D, (VD or D)
        self.causal, self.window, self.mutant, self.row0, self.Lq, self.pad = causal, window, mutant, row0, Lq, pad
        self.variant = variant
        self.rep = H // KVH
        self.scale = (1.0 / math.sqrt(D)) if scale is None else scale
        self.zthr = Q._norm_ppf(1.0 - 0.15) if zthr is None else zthr
        rng = np.random.default_rng(seed)
        self.q, self.k = P.make_qk(rng, (B, H, L, D), (B, KVH, S, D), self.scale, dtype)
        self.v = P.make_v(rng, (B, KVH, S, self.VD), dtype)
        self.sinks = None
        self.n_extra_den = 0
        self.dv_leg = family.startswith("bwd")
        self.sigma_rows = [self.k]
        self._prep()
        self.sigma = P.score_bound(self.scale, self.q, *self.sigma_rows)
        if self.dv_leg:
            self.dO = P.make_v(rng, (B, H, L, self.VD), dtype)

    # ---------------------------------------------------------------- operand prep (no tagging)
    def _kv_layout(self, x):
        """apply the KV-side mutant to a (B, KVH, S, *) operand; returns the array the kernel sees."""
        if self.mutant == "bstride":
            x = batch_stride_off1(x)
        if self.mutant == "topleft":
            x = x[:, :, :self.L]
        if self.mutant == "offby1":   # EXPLORATORY (unregistered): one extra future key per row (S' = S+1)
            x = np.concatenate([x, x[:, :, -1:]], axis=2)
        return x

    def _prep(self):
        f, dt = self.family, self.dtype
        self.k_np = self._kv_layout(self.k)
        if f in ("apa_sel", "apa_sink", "fwd_train", "fwd_train_h", "bwd", "grm_mcpm3_apa", "grm_blend",
                 "grapa_sel"):
            R, CB, BND = Q._tables(self.D, 2, self.KVH, True, "cuda")
            # row-independent quantizer: chunk along S (GRM gemma4_tc does the same, "proven bit-exact")
            parts = []
            for s0 in range(0, self.S, 8192):
                kq = Q._quantize_keys(up(self.k[:, :, s0:s0 + 8192], "float32"), R, CB, BND)
                parts.append(down(kq))
                del kq
            self.kq = P.round_to(np.concatenate(parts, axis=2), dt)
            self.sigma_rows = [self.k, self.kq]
            self.kq_np = self._kv_layout(self.kq)
        # upload the fixed operands once
        self.q_t = up(self.q, dt)
        gq = self._heads_for_kernel()
        self.k_t = up(expand_heads(self.k_np, gq[0], gq[1]), dt)
        if hasattr(self, "kq_np"):
            self.kq_t = up(expand_heads(self.kq_np, gq[0], gq[1]), dt)

    def _heads_for_kernel(self):
        """(rep, how) the adapter must expand KV heads by before the kernel."""
        gqa_native = self.family in ("apa_sel", "apa_int4", "apa_int4_ws", "apa_sp", "apa_sink",
                                     "apa_gemm", "apa_gemm_cached", "fwd_train", "fwd_train_h", "bwd",
                                     "grm_blend", "grm_g4_sliding", "grapa_sel", "grapa_sdpa")
        if self.mutant == "tile":
            return self.rep, "tile"          # expand with the WRONG map; kernel then sees KVH = H
        if gqa_native:
            return 1, "repeat"
        return self.rep, "repeat"

    # ---------------------------------------------------------------- the kernel call
    def run(self, X):
        dt, f = self.dtype, self.family
        rep, how = self._heads_for_kernel()
        v_t = up(expand_heads(self._kv_layout(X), rep, how), dt)
        q, k = self.q_t, self.k_t
        caus = self.causal
        if self.mutant == "topleft":
            caus = True       # square truncation: bottom-right on (L, L) == top-left on (L, S)
        with tc.no_grad():
            if f == "sdpa":
                y = F.scaled_dot_product_attention(q, k, v_t, is_causal=caus, scale=self.scale)
            elif f == "apa_quant":   # zthr < -10 = the baselines' all-refine regime (refine_percentile 1.0)
                y = Q.apa_quant_attention(q, k, v_t, is_causal=caus, scale=self.scale,
                                          refine_percentile=1.0 if self.zthr < -10 else 0.15)
            elif f == "apa_sel":
                y = tc.apa_selective_attention(q, k, self.kq_t, v_t, self.scale, self.zthr, caus)
            else:
                raise ValueError(f)
        return down(y)

    # ---------------------------------------------------------------- label & interval geometry
    def labels(self, mode):
        B, H, KVH = self.B, self.H, self.KVH
        g_of_h = np.arange(H) // self.rep
        if mode == "head":
            lab_kv = np.broadcast_to(np.arange(KVH)[None, :], (B, KVH))
        elif mode == "batch":
            lab_kv = np.broadcast_to(np.arange(B)[:, None], (B, KVH))
        elif mode == "slab":
            lab_kv = np.arange(B)[:, None] * KVH + np.arange(KVH)[None, :]
        else:
            raise ValueError(mode)
        lab_kv = np.ascontiguousarray(lab_kv)
        lab_q = lab_kv[:, g_of_h]
        if self.dv_leg:
            return lab_q, lab_kv          # operand dO carries query-head labels; dV the KV-head labels
        return lab_kv, lab_q

    def operand(self):
        return self.dO if self.dv_leg else self.v

    def intervals(self):
        B, H, L, S = self.B, self.H, self.L, self.S
        if self.dv_leg:
            lo, hi = P.dv_intervals(B, self.KVH, S, L, causal=self.causal)
            return lo, hi, np.full(lo.shape, S * self.rep * L)
        if self.family == "sdpa_mask" and self.window:
            lo, hi = P.causal_intervals(B, H, L, S, window=self.window)
        elif self.family == "sdpa_mask":
            lo, hi = P.causal_intervals(B, H, L, S, pad=self.pad)
        elif self.family == "grm_g4_sliding":
            lo, hi = P.causal_intervals(B, H, L, S, window=self.window if S > self.window else 0)
            if S <= self.window and L == 1:
                lo, hi = P.full_intervals(B, H, L, S)
        elif self.family == "apa_gemm" or self.family == "apa_gemm_cached":
            lo, hi = (P.causal_intervals(B, H, L, S, row0=self.row0, Lq=self.Lq, window=self.window)
                      if self.causal else P.full_intervals(B, H, L, S))
        elif self.causal:
            lo, hi = P.causal_intervals(B, H, L, S)
        else:
            lo, hi = P.full_intervals(B, H, L, S)
        n_den = np.maximum(0, hi - lo + 1) + self.n_extra_den
        return lo, hi, n_den

    def cuts(self):
        L, S = self.L, self.S
        if self.dv_leg:
            t = max(1, min(L // 2, L - 1)) if L > 1 else 1
            return [(t, "lt")] if L > 1 else []
        if self.family in ("sdpa_mask", "grm_g4_sliding", "apa_gemm", "apa_gemm_cached") and self.window \
                and S > self.window:
            return P.causal_cuts(L, S) + P.window_cuts(L, S, self.window)
        if self.family in ("apa_gemm", "apa_gemm_cached") and self.causal and self.Lq:
            pos0 = (S - self.Lq) + self.row0
            ta = min(max(1, pos0 + L // 2), S - 1)
            tb = min(max(1, min(L // 2, pos0)), S - 1)
            return [(t, "ge") for t in sorted({ta, tb})]
        if self.causal or self.family == "sdpa_mask":
            return P.causal_cuts(L, S)
        return [(max(1, S // 2), "ge")]
