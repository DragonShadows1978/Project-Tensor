"""APA-OM135 core: operands with exactness certificates, the IMM derivative self-reduction check, the integer
textbook-formula baseline, and gradcheck-style central finite differences, for tensor_cuda backward ops.

THE IMM CHECK (derivative self-reduction). For a matrix chain Y = alpha * M_1 M_2 ... M_k (M_m = X_m or X_m^T),
the gradient of the single output entry Y[i, j] is
    dM_1[i, :] = alpha * (M_2 ... M_k)[:, j]^T,   dM_k[:, j] = alpha * (M_1 ... M_{k-1})[i, :]^T,
    dM_m       = alpha * (M_1 ... M_{m-1})[i, :]^T (x) (M_{m+1} ... M_k)[:, j]^T   (outer product, 1 < m < k).
With M_1[i, :] := e_a^T and M_k[:, j] := e_b (basis row / column) every one of these is a slice of a SHORTER
chain of the middle factors M_2 ... M_{k-1} (the second-derivative statement
d^2 (X_1...X_k)_{11} / d(X_1)_{1a} d(X_k)_{b1} = (X_2...X_{k-1})_{ab}). So one first-order backward pass with the
one-hot seed dY = e_ij (one seed per batch element) must reproduce slices of forward products: the reference is
built from input slices plus engine-FORWARD matmuls of the middle chain (k = 4: M_2 M_3), never from a CPU
evaluation of the backward formula. Compared exactly (IEEE ==, NaN = mismatch) over the WHOLE gradient tensors
(zeros everywhere off the seeded rows/columns are part of the reference).
Attention: SDPA = P V with P = softmax(scale Q K^T + masks). V-leg: dV[:, c] = P[i, :]^T, P read from an engine
FORWARD with V := I (the basis-column substitution). Q/K-legs: the score chain Q K^T is bilinear, its
self-reduction needs the seeded row of dS, built from forward quantities with engine forward ops via the
softmax-Jacobian identity dS_i = P_i * (V[:, c] - O_ic) (textbook, Bridle 1990 -- the one external identity).
APA selective attention (fused kernel): one-hot P (score gaps >= 128), P rows read by an engine forward with a
bit-coded V; dQ = dK = 0 exactly on this operand family (dS = 0 for one-hot rows) -- only the dV leg is informative.

Prior art (Prior Art Directive; unverified -- lead to check):
  - openai/math #135 (2026, 'Homogeneous depth-five lower bounds for iterated matrix multiplication', eq. (1) and
    the section 1.3 derivative method) via the TX2 Opus dossier M4c (MUTATIONS.md:330-336): the derivative
    self-reduction of IMM. TAKEN: the identity and the basis-row/column substitution.
  - Nisan (1991, non-commutative ABP / partial-derivative rank) and Nisan & Wigderson (1997, partial-derivative
    measures): derivatives of IMM are IMM entries of shorter chains. TAKEN: background only.
  - Baur & Strassen (1983): all partials at a constant multiple of the cost of the function; Griewank & Walther
    (2008, 'Evaluating Derivatives'): reverse mode with a Cartesian basis seed returns one Jacobian row.
    TAKEN: one backward pass per one-hot seed = one row; the cost argument.
  - Freivalds (1977): randomized verification of a matrix product. Contrast only (not used).
  - PyTorch torch.autograd.gradcheck (element-wise central FD of the analytical Jacobian; fast_mode 2021):
    the finite-difference baseline is that technique on a scalar loss <dY, F(X)>. TAKEN: the method.
  - Bridle (1990) softmax; the softmax Jacobian dS = P*(dP - rowsum(P*dP)). TAKEN: the identity (SDPA Q/K legs).
  - DeMillo, Lipton & Sayward (1978): mutation testing. TAKEN: the evaluation design.
  - APA-LD1 (2026, this fork): engine pinning, scratch-copy source mutants, exact power-of-two operands.
  MINE: the operand family (signed partial permutations + sparse extras) with abs-product exactness certificates
  in every dtype, the two-hot SDPA / one-hot APA operands, the one-seed-per-batch-element generalization, and the
  harness.
"""
from __future__ import annotations

import hashlib
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/apa_om135"

# exactness: an integer (or a dyadic numerator in its finest unit) with |n| <= LIMIT is exact in the dtype;
# every GEMM accumulates in fp32, so every partial sum <= 2^24 is exact in any order.
LIMIT = {"float32": 1 << 24, "bfloat16": 1 << 8, "float16": 1 << 11}
ACC = 1 << 24
DTYPES = ("float32", "bfloat16", "float16")

# gradcheck (registered): PyTorch fast_mode-style directional central differences of L = <dY, F(X)> along GC_NDIR
# dense random ternary directions U per differentiable input, L computed in fp64 on the host from an fp32 engine
# forward (the engine has no fp64 matmul); analytical side <grad, U> from the cell-dtype backward (fp64 host dot).
# Step H_LIN for inputs F is linear in (exact: certified), H_NL otherwise; alarm iff |num - ana| > ATOL + RTOL*|ana|
# at any probe.
GC_NDIR = 2
GC_H_LIN = 1.0
GC_H_NL = 2.0 ** -4
GC_ATOL = 2e-3
GC_RTOL = 1e-2

tc = None
F = None
nn = None


def load_engine():
    """Import tensor_cuda from APA_OM135_ENGINE (dir with _tensor_cuda*.so) + APA_OM135_PKG (dir holding tensor_cuda/)."""
    global tc, F, nn
    eng, pkg = os.environ["APA_OM135_ENGINE"], os.environ["APA_OM135_PKG"]
    for p in (pkg, eng):
        if p not in sys.path:
            sys.path.insert(0, p)
    import tensor_cuda as _tc
    from tensor_cuda import functional as _F
    from tensor_cuda import nn as _nn
    so = Path(_tc._C.__file__).resolve()
    if not str(so).startswith(str(Path(eng).resolve())):
        raise RuntimeError(f"engine {so} not under {eng}")
    if not str(Path(_tc.__file__).resolve()).startswith(str(Path(pkg).resolve())):
        raise RuntimeError(f"package {_tc.__file__} not under {pkg}")
    tc, F, nn = _tc, _F, _nn
    return dict(engine_so=str(so), engine_sha256=hashlib.sha256(so.read_bytes()).hexdigest(),
                package=str(Path(_tc.__file__).resolve().parent))


def driver_check():
    """Native driver only: /proc/driver/nvidia/version must name 595.99.02 (open module form accepted)."""
    txt = Path("/proc/driver/nvidia/version").read_text()
    first = txt.splitlines()[0]
    ok = "595.99.02" in first and "595.84" not in first
    if any("595.84" in os.environ.get(v, "") for v in ("LD_LIBRARY_PATH", "LD_PRELOAD", "PATH")):
        ok = False
    return ok, first.strip()


# ------------------------------------------------------------------ dtype helpers
def round_to(x, dtype):
    x = np.asarray(x, dtype=np.float64)
    if dtype == "float32":
        return x.astype(np.float32).astype(np.float64)
    if dtype == "float16":
        return x.astype(np.float16).astype(np.float64)
    if dtype == "bfloat16":
        u = x.astype(np.float32).view(np.uint32).astype(np.uint64)
        u = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) << 16
        return u.astype(np.uint32).view(np.float32).astype(np.float64)
    raise ValueError(dtype)


def up(x, dtype, grad=False):
    """Host fp64 array -> engine tensor in dtype (values certified exact), optionally a grad leaf."""
    if dtype == "float16":
        t = tc.tensor(np.ascontiguousarray(x, dtype=np.float16), dtype="float16")
    else:
        t = tc.tensor(np.ascontiguousarray(x, dtype=np.float32), dtype=dtype)
    if grad:
        t.requires_grad = True
    return t


def down(t):
    return np.asarray(t.numpy(), dtype=np.float64)


def exact_equal(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return False, f"shape {a.shape} != {b.shape}"
    bad = ~(a == b)          # NaN != NaN counts as a mismatch; -0 == +0
    n = int(bad.sum())
    if n == 0:
        return True, "exact"
    idx = np.argwhere(bad)[0].tolist()
    return False, f"{n} mismatches; first {idx}: {a[tuple(idx)]!r} vs {b[tuple(idx)]!r}"


# ------------------------------------------------------------------ operand family
def ppx(rng, shape, extra):
    """Signed partial permutation-like integer matrix: every row has one +-1 at a random column, plus Bernoulli
    +-1 extras with `extra` expected nonzeros per row. Batched over leading dims. Values in {-1, 0, 1} (rarely +-2
    never: extras only land on zero entries)."""
    shape = tuple(shape)
    *lead, M, K = shape
    X = np.zeros(shape, dtype=np.float64)
    flat = X.reshape(-1, M, K)
    nb = flat.shape[0]
    cols = rng.integers(0, K, size=(nb, M))
    sgn = rng.choice(np.array([-1.0, 1.0]), size=(nb, M))
    flat[np.arange(nb)[:, None], np.arange(M)[None, :], cols] = sgn
    if extra > 0:
        mask = (rng.random(flat.shape) < extra / K) & (flat == 0)
        vals = rng.choice(np.array([-1.0, 1.0]), size=flat.shape)
        flat[mask] = vals[mask]
    return flat.reshape(shape)


def tern(rng, shape, density):
    """Bernoulli(density) +-1 entries (dense-ish ternary)."""
    m = rng.random(shape) < density
    return np.where(m, rng.choice(np.array([-1.0, 1.0]), size=shape), 0.0)


# ------------------------------------------------------------------ chain kinds (chain / linear / bbm / bmm)
def _T(x):
    return np.swapaxes(x, -1, -2)


def _reduce_to(g, shape):
    while g.ndim > len(shape):
        g = g.sum(0)
    return g


def chain_eff(Xs, trans):
    return [_T(X) if t else X for X, t in zip(Xs, trans)]


def chain_textbook(Xs, trans, alpha, dY, absval=False):
    """fp64 textbook gradients dX_m = alpha * prefix^T dY suffix^T (reduced over broadcast batch dims).
    absval=True evaluates the same products on |.| (bounds every partial sum in any summation order).
    Returns (grads, materialized) where materialized lists every GEMM output the engine forms."""
    f = np.abs if absval else (lambda z: z)
    M = [f(m) for m in chain_eff(Xs, trans)]
    a = abs(alpha) if absval else alpha
    G = f(dY) * a
    k = len(M)
    pre = [None] * (k + 1)
    pre[1] = M[0]
    mats = []
    for m in range(2, k + 1):
        pre[m] = np.matmul(pre[m - 1], M[m - 1])
        mats.append(pre[m])
    suf = [None] * (k + 2)          # suf[m] = M_m ... M_k
    suf[k] = M[k - 1]
    for m in range(k - 1, 1, -1):
        suf[m] = np.matmul(M[m - 1], suf[m + 1])
    grads = []
    for m in range(1, k + 1):
        g = G
        if m < k:
            g = np.matmul(g, _T(suf[m + 1]))        # backward intermediate dP_m (materialized)
            mats.append(g)
        if m > 1:
            g = np.matmul(_T(pre[m - 1]), g)
        g = _reduce_to(g, M[m - 1].shape)
        g = _T(g) if trans[m - 1] else g
        grads.append(g)
        mats.append(g)
    return grads, mats, pre, suf


def chain_cert(Xs, trans, alpha, dY, dtype, bias=False, mid_products=()):
    """Exactness certificate: every materialized GEMM output and every partial sum is an integer of magnitude
    <= LIMIT[dtype] (abs-product bound). FD exactness (fp32 forward, step 1): Ybound + max|prefix|*max|suffix|."""
    grads, mats, pre, suf = chain_textbook(Xs, trans, alpha, dY, absval=True)
    worst = max(float(np.max(m)) for m in mats)
    if bias:
        worst = max(worst, float(np.max(_reduce_to(np.abs(dY), dY.shape[-1:]))))
    for mp in mid_products:
        worst = max(worst, float(np.max(mp)))
    k = len(Xs)
    M = [np.abs(m) for m in chain_eff(Xs, trans)]
    fd = 0.0
    for m in range(k):      # forward with M_m + U (|U| <= 1 dense): abs chain with |M_m| + 1
        P = M[0] + 1.0 if m == 0 else M[0]
        for n in range(1, k):
            P = np.matmul(P, M[n] + 1.0 if n == m else M[n])
        fd = max(fd, float(np.max(P)) * abs(alpha) + (1.0 if bias else 0.0))
    if bias:
        fd = max(fd, float(np.max(pre[k])) * abs(alpha) + 2.0)
    ok = worst <= LIMIT[dtype] and fd <= ACC
    return ok, dict(max_abs_bound=worst, limit=LIMIT[dtype], fd_bound=fd, fd_limit=ACC)


class ChainCell:
    """kind in {chain, linear, bbm, bmm}. factors: list of (shape, trans, alpha_of_this_step). For 'linear' the
    engine forward is nn.Linear (weight (out,in) used as trans_b) + bias."""

    def __init__(self, cid, kind, shapes, trans, alphas, extra, dy_extra, seed, bias=False, shared=None):
        self.cid, self.kind, self.shapes, self.trans, self.alphas = cid, kind, shapes, trans, alphas
        self.alpha = float(np.prod(alphas))
        self.extra, self.dy_extra, self.seed, self.bias = extra, dy_extra, seed, bias
        self.shared = shared or [False] * len(shapes)   # factor broadcast across the batch (2D weight)
        self.k = len(shapes)

    # ---------------- operands
    def out_shape(self):
        eff = [tuple(s[:-2]) + ((s[-1], s[-2]) if t else (s[-2], s[-1])) for s, t in zip(self.shapes, self.trans)]
        lead = max((e[:-2] for e in eff), key=len)
        return tuple(lead) + (eff[0][-2], eff[-1][-1])

    def operands(self, rng):
        Xs = [ppx(rng, s, self.extra) for s in self.shapes]
        b = tern(rng, (self.shapes[1][0],), 0.25) if self.bias else None
        return Xs, b

    def dy(self, rng):
        return ppx(rng, self.out_shape(), self.dy_extra)

    def batch_index(self):
        lead = self.out_shape()[:-2]
        return list(np.ndindex(*lead)) if lead else [()]

    def imm_operands(self, rng):
        """Basis substitution: M_1[beta, i, :] = e_a, M_k[beta, :, j] = e_b; one-hot seed dY[beta, i, j] = 1 per batch
        element beta (distinct j when M_k is shared)."""
        Xs, b = self.operands(rng)
        Ms = chain_eff(Xs, self.trans)
        M1, Mk = Ms[0], Ms[-1]
        ys = self.out_shape()
        Mr, Nc = ys[-2], ys[-1]
        K1 = M1.shape[-1]
        Kk = Mk.shape[-2]
        bidx = self.batch_index()
        js = rng.choice(Nc, size=len(bidx), replace=False) if len(bidx) <= Nc else rng.integers(0, Nc, len(bidx))
        seeds = []
        dY = np.zeros(ys)
        for n, beta in enumerate(bidx):
            i = int(rng.integers(0, Mr))
            j = int(js[n])
            a = int(rng.integers(0, K1))
            bb = int(rng.integers(0, Kk))
            seeds.append((beta, i, j, a, bb))
            dY[beta + (i, j)] = 1.0
        for beta, i, j, a, bb in seeds:
            b1 = beta[len(beta) - (M1.ndim - 2):] if M1.ndim > 2 else ()
            row = np.zeros(K1)
            row[a] = 1.0
            M1[b1 + (i,)] = row
            bk = beta[len(beta) - (Mk.ndim - 2):] if Mk.ndim > 2 else ()
            col = np.zeros(Kk)
            col[bb] = 1.0
            Mk[bk + (slice(None), j)] = col
        Xs = [(_T(M) if t else M).copy() for M, t in zip(Ms, self.trans)]
        return Xs, b, dY, seeds

    def certify(self, Xs, dY, dtype, mid=()):
        return chain_cert(Xs, self.trans, self.alpha, dY, dtype, bias=self.bias, mid_products=mid)

    # ---------------- engine forward
    def forward(self, ts, bt, lin=None):
        if self.kind == "linear":
            lin.weight = ts[1]
            if bt is not None:
                lin.bias = bt
            return lin.forward(ts[0])
        Y = ts[0]
        for m in range(1, self.k):
            Y = tc.matmul(Y, ts[m], alpha=self.alphas[m], trans_b=self.trans[m])
        return Y

    def make_linear(self, dtype):
        out_f, in_f = self.shapes[1]
        lin = nn.Linear(in_f, out_f, bias=self.bias)
        return lin

    def run_backward(self, Xs, b, dY, dtype, lin):
        ts = [up(X, dtype, grad=True) for X in Xs]
        bt = up(b, dtype, grad=True) if b is not None else None
        Y = self.forward(ts, bt, lin)
        loss = (Y * up(dY, dtype)).sum()
        loss.backward()
        grads = [down(t.grad) for t in ts]
        if bt is not None:
            grads.append(down(bt.grad))
        return grads

    def textbook(self, Xs, dY):
        grads = chain_textbook(Xs, self.trans, self.alpha, dY)[0]
        if self.bias:
            grads.append(_reduce_to(dY, dY.shape[-1:]))
        return grads

    def imm_reference(self, Xs, seeds, dtype):
        """Expected gradients from slices of the middle chain M_2..M_{k-1} (+ one engine forward matmul for k=4)."""
        Ms = chain_eff(Xs, self.trans)
        k = self.k
        al = self.alpha
        exp = [np.zeros(M.shape) for M in Ms]
        mid = None
        if k == 4:   # M_2 M_3 by the ENGINE forward (no grad graph: detached leaves)
            mid = down(tc.matmul(up(Ms[1], dtype), up(Ms[2], dtype)))
        for beta, i, j, a, bb in seeds:
            def bsel(M):
                return beta[len(beta) - (M.ndim - 2):] if M.ndim > 2 else ()
            for m in range(1, k + 1):
                M = Ms[m - 1]
                bs = bsel(M)
                if m == 1:
                    if k == 2:
                        v = np.eye(M.shape[-1])[bb]
                    elif k == 3:
                        v = Ms[1][bsel(Ms[1]) + (slice(None), bb)]
                    else:
                        v = mid[bsel(Ms[1]) + (slice(None), bb)]
                    exp[0][bs + (i,)] += al * v
                elif m == k:
                    if k == 2:
                        v = np.eye(M.shape[-2])[a]
                    elif k == 3:
                        v = Ms[1][bsel(Ms[1]) + (a,)]
                    else:
                        v = mid[bsel(Ms[1]) + (a,)]
                    exp[k - 1][bs + (slice(None), j)] += al * v
                else:
                    # row a of M_2..M_{m-1} (empty -> e_a), column b of M_{m+1}..M_{k-1} (empty -> e_b)
                    r = np.eye(M.shape[-2])[a] if m == 2 else Ms[1][bsel(Ms[1]) + (a,)]
                    c = np.eye(M.shape[-1])[bb] if m == k - 1 else Ms[m][bsel(Ms[m]) + (slice(None), bb)]
                    exp[m - 1][bs] += al * np.outer(r, c)
        out = [_T(e) if t else e for e, t in zip(exp, self.trans)]
        if self.bias:
            db = np.zeros(self.out_shape()[-1])
            for beta, i, j, a, bb in seeds:
                db[j] += 1.0
            out.append(db)
        return out, mid

    def inputs_linear(self):
        return [True] * (self.k + (1 if self.bias else 0))


# ------------------------------------------------------------------ SDPA (composed, causal + two-hot mask)
class SdpaCell:
    """B,H,L=S,D; causal + additive mask leaving one key pair per row (one key for row 0). Keys come in pairs
    (2m, 2m+1) equal on the coordinates where every query is nonzero (first D/2) and independent on the rest
    (Z, where every query is 0), so each row's two visible scores tie exactly -> P in {0, 1/2, 1} exactly.
    scale = 1/8 passed explicitly (dyadic)."""
    kind = "sdpa"

    def __init__(self, cid, B, H, S, D, seed, q_density, v_density, dy_density, window):
        self.cid, self.B, self.H, self.L, self.S, self.D, self.seed = cid, B, H, S, S, D, seed
        self.qd, self.vd, self.dyd, self.window = q_density, v_density, dy_density, window
        self.scale = 0.125

    def operands(self, rng):
        B, H, L, S, D = self.B, self.H, self.L, self.S, self.D
        Dh = D // 2
        q = np.zeros((B, H, L, D))
        q[..., :Dh] = tern(rng, (B, H, L, Dh), self.qd)
        k = tern(rng, (B, H, S, D), self.qd)
        k[:, :, 1::2, :Dh] = k[:, :, 0:S - 1:2, :Dh] if S % 2 else k[:, :, 0::2, :Dh]
        v = tern(rng, (B, H, S, D), self.vd)
        mask = np.full((L, S), -np.inf)
        mask[0, 0] = 0.0
        pair = np.zeros(L, dtype=np.int64)
        for i in range(1, L):
            npairs = (i - 1) // 2 + 1                  # pairs (2m, 2m+1) with 2m+1 <= i (S == L)
            lo = max(0, npairs - self.window)
            m = int(rng.integers(lo, npairs))
            mask[i, 2 * m] = mask[i, 2 * m + 1] = 0.0
            pair[i] = m
        return dict(q=q, k=k, v=v, mask=mask)

    def dy(self, rng):
        return tern(rng, (self.B, self.H, self.L, self.D), self.dyd)

    def _np_fwd(self, q, k, v, mask):
        s = np.matmul(q, _T(k)) * self.scale
        causal = np.triu(np.ones((self.L, self.S), bool), 1 + self.S - self.L)
        s = np.where(causal, -np.inf, s) + mask
        mx = s.max(-1, keepdims=True)
        e = np.exp(s - mx)
        P = e / e.sum(-1, keepdims=True)
        return P, np.matmul(P, v)

    def textbook(self, ops, dY):
        q, k, v = ops["q"], ops["k"], ops["v"]
        P, O = self._np_fwd(q, k, v, ops["mask"])
        dV = np.matmul(_T(P), dY)
        dP = np.matmul(dY, _T(v))
        dS = P * (dP - (P * dP).sum(-1, keepdims=True))
        dQ = np.matmul(dS, k) * self.scale
        dK = np.matmul(_T(dS), q) * self.scale
        return [dQ, dK, dV]

    def certify(self, ops, dY, dtype):
        """scores: integer q.k <= LIMIT (scaled by 1/8, dyadic); P exact; the backward values are dyadic with
        denominators 2 (O, dV), 4 (dS), 32 (dQ, dK = dS K / 8): each numerator <= LIMIT, each GEMM's abs-sum <= 2^24."""
        q, k, v = ops["q"], ops["k"], ops["v"]
        P, O = self._np_fwd(q, k, v, ops["mask"])
        sc = np.matmul(np.abs(q), _T(np.abs(k)))
        dP = np.matmul(np.abs(dY), _T(np.abs(v)))
        dQ, dK, dV = self.textbook(ops, dY)
        vals = [(sc, 1), (P, 2), (O, 2), (dP, 1), (np.abs(dQ), 32), (np.abs(dK), 32), (np.abs(dV), 2), (dP, 4)]
        absg = [np.matmul(P, np.abs(v)) * 2,
                np.matmul(np.abs(dP) / 2, np.abs(k)) * 4,
                np.matmul(_T(np.abs(dP) / 2), np.abs(q)) * 4,
                np.matmul(_T(P), np.abs(dY)) * 2]
        worst = max(float(np.max(np.abs(x) * u)) for x, u in vals if np.isfinite(x).all())
        accw = max(float(np.max(x)) for x in absg)
        pset = set(np.unique(P).tolist())
        ok = worst <= LIMIT[dtype] and accw <= ACC and pset <= {0.0, 0.5, 1.0}
        for x in (dQ, dK, dV, O):
            ok = ok and np.array_equal(round_to(x, dtype), x)
        return ok, dict(max_numerator=worst, limit=LIMIT[dtype], acc_bound=accw, P_values=sorted(pset))

    def forward(self, t, dtype, v_override=None):
        v = v_override if v_override is not None else t["v"]
        return F.scaled_dot_product_attention(t["q"], t["k"], v, attn_mask=t["mask"], is_causal=True,
                                              scale=self.scale)

    def upload(self, ops, dtype, grad):
        t = {n: up(ops[n], dtype, grad=grad) for n in ("q", "k", "v")}
        t["mask"] = up(ops["mask"], dtype)
        return t

    def run_backward(self, ops, dY, dtype):
        t = self.upload(ops, dtype, True)
        O = self.forward(t, dtype)
        (O * up(dY, dtype)).sum().backward()
        return [down(t[n].grad) for n in ("q", "k", "v")]

    def imm_seeds(self, rng):
        seeds = []
        dY = np.zeros((self.B, self.H, self.L, self.D))
        for b in range(self.B):
            for h in range(self.H):
                i = int(rng.integers(1, self.L))     # rows >= 1 carry a two-hot (informative) P row
                c = int(rng.integers(0, self.D))
                seeds.append((b, h, i, c))
                dY[b, h, i, c] = 1.0
        return dY, seeds

    def eye_host(self, dtype):
        dt = np.float16 if dtype == "float16" else np.float32
        return np.ascontiguousarray(np.broadcast_to(np.eye(self.S, dtype=dt), (self.B, self.H, self.S, self.S)))

    def imm_reference(self, ops, seeds, dtype, eye_host=None):
        """Engine-forward references (grad-enabled composed path on non-grad leaves, i.e. the training kernels)."""
        B, H, L, S, D = self.B, self.H, self.L, self.S, self.D
        t = self.upload(ops, dtype, False)
        eye = up(eye_host if eye_host is not None else self.eye_host(dtype), dtype)
        Pt = self.forward(t, dtype, v_override=eye)                 # P (B,H,L,S) via V := I
        Ot = self.forward(t, dtype)                                 # O (B,H,L,D)
        E = np.zeros((B, H, 1, L))
        C = np.zeros((B, H, D, 1))
        for b, h, i, c in seeds:
            E[b, h, 0, i] = 1.0
            C[b, h, c, 0] = 1.0
        Et, Ct = up(E, dtype), up(C, dtype)
        Pi = tc.matmul(Et, Pt)                                      # (B,H,1,S)
        Vc = tc.matmul(t["v"], Ct)                                  # (B,H,S,1)
        Oic = tc.matmul(tc.matmul(Et, Ot), Ct)                      # (B,H,1,1)
        dS = Pi * (Vc.transpose_last() - Oic)                       # (B,H,1,S) softmax-Jacobian row
        Qi = tc.matmul(Et, t["q"])                                  # (B,H,1,D)
        dQ = tc.matmul(Et.transpose_last(), tc.matmul(dS, t["k"], alpha=self.scale))
        dK = tc.matmul(dS.transpose_last(), Qi, alpha=self.scale)
        dV = tc.matmul(Pi.transpose_last(), Ct.transpose_last())
        return [down(dQ), down(dK), down(dV)]

    def gc_inputs(self, ops):
        return [("q", False), ("k", False), ("v", True)]


# ------------------------------------------------------------------ APA selective attention (fused train kernel)
def bf16_ladder(n):
    """First n nonnegative values exactly representable in bf16 (and fp16 for n <= 1025)."""
    out = list(range(0, 257))
    e = 8
    while len(out) < n:
        step = 1 << (e - 7)
        out += list(range((1 << e) + step, (1 << (e + 1)) + 1, step))
        e += 1
    return np.array(out[:n], dtype=np.float64)


class ApaCell:
    """apa_selective_train(q, k, kq=k, v, scale=1, zthr=0, causal), L = S, GQA H/KVH = 2. q = 128 e_0 for every row;
    k[:, 0] = a block-reversed ladder of exact values (blocks of BLK keys) so row i's unique max is the first key of
    its block with a score gap >= 128 to every other visible key: P is exactly one-hot, dV = P^T dO exactly
    (atomicAdd of integers: order-independent), dQ = dK = 0 exactly."""
    kind = "apa"
    BLK = 8

    def __init__(self, cid, B, H, KVH, S, D, seed, v_density, dy_density):
        self.cid, self.B, self.H, self.KVH, self.L, self.S, self.D, self.seed = cid, B, H, KVH, S, S, D, seed
        self.vd, self.dyd = v_density, dy_density
        self.scale, self.zthr = 1.0, 0.0

    def operands(self, rng):
        B, H, KVH, L, S, D = self.B, self.H, self.KVH, self.L, self.S, self.D
        q = np.zeros((B, H, L, D))
        q[..., 0] = 128.0
        lad = bf16_ladder(S)
        perm = np.arange(S)
        for s0 in range(0, S, self.BLK):
            perm[s0:s0 + self.BLK] = perm[s0:s0 + self.BLK][::-1]
        k = tern(rng, (B, KVH, S, D), 0.5)
        k[..., 0] = lad[perm][None, None, :]
        v = tern(rng, (B, KVH, S, D), self.vd)
        return dict(q=q, k=k, v=v)

    def dy(self, rng):
        return tern(rng, (self.B, self.H, self.L, self.D), self.dyd)

    def sigma(self):
        return (np.arange(self.L) // self.BLK) * self.BLK

    def textbook(self, ops, dY):
        q, k, v = ops["q"], ops["k"], ops["v"]
        g = self.H // self.KVH
        kk, vv = np.repeat(k, g, 1), np.repeat(v, g, 1)
        s = np.matmul(q, _T(kk)) * self.scale
        causal = np.triu(np.ones((self.L, self.S), bool), 1 + self.S - self.L)
        s = np.where(causal, -np.inf, s)
        P = np.exp(s - s.max(-1, keepdims=True))
        P = P / P.sum(-1, keepdims=True)
        dP = np.matmul(dY, _T(vv))
        dS = P * (dP - (P * dP).sum(-1, keepdims=True))
        dQ = np.matmul(dS, kk) * self.scale
        dK = np.matmul(_T(dS), q) * self.scale
        dV = np.matmul(_T(P), dY)
        B, KVH = self.B, self.KVH
        dK = dK.reshape(B, KVH, g, self.S, self.D).sum(2)
        dV = dV.reshape(B, KVH, g, self.S, self.D).sum(2)
        return [dQ, dK, dV]

    def certify(self, ops, dY, dtype):
        g = self.H // self.KVH
        sig = self.sigma()
        lad_ok = np.array_equal(round_to(ops["k"][..., 0], dtype), ops["k"][..., 0])
        cnt = np.zeros((self.B, self.KVH, self.S, self.D))
        ad = np.abs(dY).reshape(self.B, self.KVH, g, self.L, self.D).sum(2)
        np.add.at(cnt, (slice(None), slice(None), sig), ad)
        # score gap: within a row the max beats every other visible key by >= 128 * (ladder gap >= 1)
        worst = float(cnt.max())
        ok = lad_ok and worst <= LIMIT[dtype] and float(ops["k"][..., 0].max()) * 128 <= ACC
        return ok, dict(max_dv_abs_bound=worst, limit=LIMIT[dtype], ladder_exact=bool(lad_ok))

    def upload(self, ops, dtype, grad):
        t = {n: up(ops[n], dtype, grad=grad) for n in ("q", "k", "v")}
        t["kq"] = up(ops["k"], dtype)
        return t

    def forward(self, t, v_override=None):
        v = v_override if v_override is not None else t["v"]
        return tc.apa_selective_train(t["q"], t["k"], t["kq"], v, self.scale, self.zthr, True)

    def run_backward(self, ops, dY, dtype):
        t = self.upload(ops, dtype, True)
        O = self.forward(t)
        (O * up(dY, dtype)).sum().backward()
        return [down(t[n].grad) for n in ("q", "k", "v")]

    def imm_seeds(self, rng):
        seeds = []
        dY = np.zeros((self.B, self.H, self.L, self.D))
        for b in range(self.B):
            for h in range(self.H):
                i = int(rng.integers(0, self.L))
                c = int(rng.integers(0, self.D))
                seeds.append((b, h, i, c))
                dY[b, h, i, c] = 1.0
        return dY, seeds

    def imm_reference(self, ops, seeds, dtype, eye_host=None):
        """P rows by an engine forward with bit-coded V (V[j, r] = bit r of j): O[i] = bits of the selected key."""
        B, KVH, S = self.B, self.KVH, self.S
        nb = int(S - 1).bit_length()
        bits = ((np.arange(S)[:, None] >> np.arange(nb)[None, :]) & 1).astype(np.float64)
        t = self.upload(ops, dtype, False)
        Ob = down(self.forward(t, v_override=up(np.broadcast_to(bits, (B, KVH, S, nb)), dtype)))
        g = self.H // self.KVH
        dQ = np.zeros((self.B, self.H, self.L, self.D))
        dK = np.zeros((B, KVH, S, self.D))
        dV = np.zeros((B, KVH, S, self.D))
        for b, h, i, c in seeds:
            j = int(round(float((Ob[b, h, i] * (1 << np.arange(nb))).sum())))
            dV[b, h // g, j, c] += 1.0
        return [dQ, dK, dV]

    def gc_inputs(self, ops):
        return [("q", False), ("k", False), ("v", True)]


# ------------------------------------------------------------------ the three methods
def method_imm(cell, rng_seed, dtype, lin=None, prepared=None):
    """IMM self-reduction check. Returns (alarm, detail, seconds). Operands/seeds are prepared outside the timer."""
    if isinstance(cell, ChainCell):
        Xs, b, dY, seeds = prepared
        t0 = time.perf_counter()
        grads = cell.run_backward(Xs, b, dY, dtype, lin)
        ref, _ = cell.imm_reference(Xs, seeds, dtype)
    else:
        ops, dY, seeds, eye_host = prepared
        t0 = time.perf_counter()
        grads = cell.run_backward(ops, dY, dtype)
        ref = cell.imm_reference(ops, seeds, dtype, eye_host)
    res = [exact_equal(g, r) for g, r in zip(grads, ref)]
    dt = time.perf_counter() - t0
    alarm = not all(ok for ok, _ in res)
    return alarm, [w for _, w in res], dt


def method_textbook(cell, dtype, lin=None, prepared=None):
    """Integer baseline: dense integer-valued operands and dY; engine gradients vs fp64 numpy textbook formulas
    (rounded once to the dtype: identity on certified-exact values), exact comparison."""
    t0 = time.perf_counter()
    if isinstance(cell, ChainCell):
        Xs, b, dY = prepared
        grads = cell.run_backward(Xs, b, dY, dtype, lin)
        ref = cell.textbook(Xs, dY)
    else:
        ops, dY = prepared
        grads = cell.run_backward(ops, dY, dtype)
        ref = cell.textbook(ops, dY)
    res = [exact_equal(g, round_to(r, dtype)) for g, r in zip(grads, ref)]
    dt = time.perf_counter() - t0
    return not all(ok for ok, _ in res), [w for _, w in res], dt


def _perturbed(t32, flat_idx, shape, h):
    """t32 + h * e_idx on the GPU through a rank-1 outer product (tiny uploads, exact)."""
    R = int(np.prod(shape[:-1])) if len(shape) > 1 else 1
    C = shape[-1]
    r, c = divmod(int(flat_idx), C)
    u = np.zeros((R, 1))
    u[r, 0] = 1.0
    w = np.zeros((1, C))
    w[0, c] = 1.0
    d = tc.matmul(up(u, "float32"), up(w, "float32"), alpha=float(h)).reshape(list(shape))
    return t32 + d


def method_gradcheck(cell, dtype, lin32=None, lin=None, prepared=None, probe_seed=0):
    """fast_mode-style gradcheck: central FD of L = <dY, F(X)> (fp32 engine forward, fp64 host loss) along GC_NDIR
    dense random ternary directions per input, against <grad, U> from the cell-dtype backward."""
    rng = np.random.default_rng(probe_seed)
    if isinstance(cell, ChainCell):
        _arrs = list(prepared[0]) + ([prepared[1]] if prepared[1] is not None else [])
    else:
        _arrs = [prepared[0][n] for n, _ in cell.gc_inputs(prepared[0])]
    dirs = [[tern(rng, a.shape, 0.5) for _ in range(GC_NDIR)] for a in _arrs]   # prepared outside the timer
    t0 = time.perf_counter()
    if isinstance(cell, ChainCell):
        Xs, b, dY = prepared
        grads = cell.run_backward(Xs, b, dY, dtype, lin)
        names = list(range(len(Xs))) + (["bias"] if b is not None else [])
        arrs = list(Xs) + ([b] if b is not None else [])
        lin_flags = cell.inputs_linear()
        base32 = [up(x, "float32") for x in arrs]

        def loss(ts):
            if b is not None:
                Y = cell.forward(ts[:-1], ts[-1], lin32)
            else:
                Y = cell.forward(ts, None, lin32)
            return float(np.dot(down(Y).ravel(), dY.ravel()))
    else:
        ops, dY = prepared
        grads = cell.run_backward(ops, dY, dtype)
        spec = cell.gc_inputs(ops)
        names = [n for n, _ in spec]
        lin_flags = [lf for _, lf in spec]
        arrs = [ops[n] for n in names]
        t32 = cell.upload(ops, "float32", False)
        base32 = [t32[n] for n in names]

        def loss(ts):
            tt = dict(t32)
            for n, x in zip(names, ts):
                tt[n] = x
            if cell.kind == "apa":
                Y = cell.forward(tt)
            else:
                Y = cell.forward(tt, "float32")
            return float(np.dot(down(Y).ravel(), dY.ravel()))
    worst = None
    alarm = False
    nprobe = 0
    tf0 = time.perf_counter()
    for n, (arr, g, lf) in enumerate(zip(arrs, grads, lin_flags)):
        h = GC_H_LIN if lf else GC_H_NL
        for U in dirs[n]:
            Ut = up(U * h, "float32")
            tsp = list(base32)
            tsp[n] = base32[n] + Ut
            lp = loss(tsp)
            tsp[n] = base32[n] - Ut
            lm = loss(tsp)
            num = (lp - lm) / (2 * h)
            ana = float(np.dot(g.ravel(), U.ravel()))
            err = abs(num - ana)
            nprobe += 1
            if not (err <= GC_ATOL + GC_RTOL * abs(ana)):
                alarm = True
                if worst is None:
                    worst = dict(input=str(names[n]), num=num, ana=ana)
    tfd = time.perf_counter() - tf0
    dt = time.perf_counter() - t0
    numel = int(sum(a.size for a in arrs))
    per_fwd = tfd / max(1, 2 * nprobe)
    return alarm, dict(first_violation=worst, probes=nprobe, forwards=2 * nprobe, sec_per_fd_forward=per_fwd,
                       elementwise_full_jacobian_extrapolated_s=2 * numel * per_fwd), dt


# ------------------------------------------------------------------ operand preparation (APA-REGT1 port)
# Ported verbatim from scripts/apa_om135_prep.py @ 2bbee82 (APA-OM135), whose docstring follows. That module did
# `from apa_om135_core import ChainCell, chain_eff`; both names are defined above, so the import line is dropped.
# It moved here because the port order authorizes only this script for OM135.
# "APA-OM135 operand preparation (shared by the CPU certificate dry run and the GPU runner). Deterministic per
# (cell, dtype): baseline/gradcheck operands from seed, IMM operands from seed+1, gradcheck probes from seed+2.
# Prior art: see apa_om135_core.py header. Mine: the preparation layout."
def prepare(cell, dtype):
    """Returns dict(textbook=..., imm=..., cert={...}, ok=bool). Host-only (no engine)."""
    rng = np.random.default_rng(cell.seed)
    rng_imm = np.random.default_rng(cell.seed + 1)
    out = {}
    if isinstance(cell, ChainCell):
        Xs, b = cell.operands(rng)
        dY = cell.dy(rng)
        ok_t, cert_t = cell.certify(Xs, dY, dtype)
        Xi, bi, dYi, seeds = cell.imm_operands(rng_imm)
        mid = ()
        if cell.k == 4:
            Ms = chain_eff(Xi, cell.trans)
            mid = (np.matmul(np.abs(Ms[1]), np.abs(Ms[2])),)
        ok_i, cert_i = cell.certify(Xi, dYi, dtype, mid=mid)
        out["textbook"] = (Xs, b, dY)
        out["imm"] = (Xi, bi, dYi, seeds)
    else:
        ops = cell.operands(rng)
        dY = cell.dy(rng)
        ok_t, cert_t = cell.certify(ops, dY, dtype)
        ops_i = cell.operands(rng_imm)
        dYi, seeds = cell.imm_seeds(rng_imm)
        ok_i, cert_i = cell.certify(ops_i, dYi, dtype)
        eye = cell.eye_host(dtype) if cell.kind == "sdpa" else None
        out["textbook"] = (ops, dY)
        out["imm"] = (ops_i, dYi, seeds, eye)
    out["cert"] = dict(textbook=dict(ok=bool(ok_t), **cert_t), imm=dict(ok=bool(ok_i), **cert_i))
    out["ok"] = bool(ok_t and ok_i)
    return out
