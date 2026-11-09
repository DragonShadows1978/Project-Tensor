#!/usr/bin/env python3
"""GN1-REV1 reviewer's own GroupNorm computations (CPU only, NumPy).

Independent of the GN1 test file's oracle except where it imports the test's
INPUT generators (_inputs, low_variance_inputs, represented) so that the
operands are the registered ones. Contents:

1. ref_loops: an independent float64 GroupNorm forward + VJP written with
   explicit per-(n, group) loops (no shared code with groupnorm_reference).
2. engine_graph: a step-by-step emulation of the engine's op graph for
   nn.GroupNorm (candidate and mutants), forward and autograd backward, with
   every op rounded to a chosen compute dtype (float32 / float16 / bfloat16)
   and reductions accumulated in float32 (the engine's ld<T> -> float path).
3. running_bound: a first-order running error bound (Wilkinson-style running
   error analysis) for the float32 candidate graph, forward and backward.

Prior art: Wu and He (2018), Group Normalization (the formula); Wilkinson
(1960s) and Higham, Accuracy and Stability of Numerical Algorithms (2002),
ch. 3 (standard model fl(a op b) = (a op b)(1+d), |d|<=u; gamma_n bound for
summation; running error analysis). IEEE 754 round-to-nearest-even for the
float16/bfloat16 emulation. CUDA powf ulp bound taken as 4 ulp (unverified -
lead to check: "CUDA Math API single precision powf maximum ulp error").
Ours: the specific graph transcription of this engine and the bound wiring.
"""
import math
import numpy as np

U32 = 2.0 ** -24


# ------------------------------------------------------------------ rounding
def rnd(a, dt):
    a = np.asarray(a, dtype=np.float64)
    if dt == "float64":
        return a
    if dt == "float32":
        return a.astype(np.float32).astype(np.float64)
    if dt == "float16":
        return a.astype(np.float16).astype(np.float64)
    if dt == "bfloat16":
        f = a.astype(np.float32)
        bits = f.view(np.uint32).copy()
        r = (bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))) & np.uint32(0xFFFF0000)
        return r.view(np.float32).astype(np.float64)
    raise ValueError(dt)


def rsum(a, axis, keepdims, dt, order="seq"):
    """Reduction with float32 accumulation (engine: ld<T> to float), result rounded to dt."""
    a = np.moveaxis(np.asarray(a, np.float64), axis, -1)
    if order == "seq":
        acc = np.zeros(a.shape[:-1])
        for k in range(a.shape[-1]):
            acc = rnd(acc + a[..., k], "float32")
    elif order == "rev":
        acc = np.zeros(a.shape[:-1])
        for k in reversed(range(a.shape[-1])):
            acc = rnd(acc + a[..., k], "float32")
    else:  # pairwise tree
        vals = [a[..., k] for k in range(a.shape[-1])]
        while len(vals) > 1:
            nxt = [rnd(vals[i] + vals[i + 1], "float32") for i in range(0, len(vals) - 1, 2)]
            if len(vals) % 2:
                nxt.append(vals[-1])
            vals = nxt
        acc = vals[0]
    out = rnd(acc, dt)
    if keepdims:
        out = np.expand_dims(out, -1)
        out = np.moveaxis(out, -1, axis)
    return out


def reduce_to(g, shape, dt, order):
    """Engine reduce_to: sum broadcast axes back to `shape` (aligned right)."""
    nd, nt = g.ndim, len(shape)
    out = g
    for d in range(nd):
        td = d - (nd - nt)
        tsz = 1 if td < 0 else shape[td]
        if tsz == 1 and g.shape[d] > 1:
            out = rsum(out, d, True, dt, order)
    return out.reshape(shape)


# ------------------------------------------------------------------ reference (independent)
def ref_loops(x, G, w, b, eps, p):
    x = np.asarray(x, np.float64)
    N, C = x.shape[:2]
    S = int(np.prod(x.shape[2:])) if x.ndim > 2 else 1
    xs = x.reshape(N, C, S)
    ps = np.asarray(p, np.float64).reshape(N, C, S)
    cg = C // G
    y = np.zeros_like(xs); dx = np.zeros_like(xs); dw = np.zeros(C); db = np.zeros(C)
    for n in range(N):
        for gi in range(G):
            chans = range(gi * cg, (gi + 1) * cg)
            vals = np.array([xs[n, c, s] for c in chans for s in range(S)])
            m = len(vals)
            mu = math.fsum(vals) / m
            var = math.fsum((v - mu) ** 2 for v in vals) / m
            inv = 1.0 / math.sqrt(var + eps)
            xh = {}
            for c in chans:
                for s in range(S):
                    xh[c, s] = (xs[n, c, s] - mu) * inv
                    y[n, c, s] = xh[c, s] * w[c] + b[c]
            # VJP derived by the reviewer: dL/dxhat = q = p*w;
            # dx = inv/m * (m q - sum q - xhat * sum(q xhat))
            q = {k: ps[n, k[0], k[1]] * w[k[0]] for k in xh}
            sq = math.fsum(q.values())
            sqx = math.fsum(q[k] * xh[k] for k in xh)
            for (c, s) in xh:
                dx[n, c, s] = inv / m * (m * q[c, s] - sq - xh[c, s] * sqx)
                dw[c] += ps[n, c, s] * xh[c, s]
                db[c] += ps[n, c, s]
    return y.reshape(x.shape), dx.reshape(x.shape), dw, db


# ------------------------------------------------------------------ engine graph emulation
def engine_graph(x, G, w, b, eps, p, ct="float32", order="seq", mutant=None, out_dt=None,
                 w_dt="float32", b_dt="float32"):
    """Forward + backward of the engine's op graph for nn.GroupNorm.

    ct: compute dtype of every op (candidate: float32 always).
    out_dt: destination rounding of the output and of x's gradient (x's dtype).
    w_dt/b_dt: leaf dtypes of weight/bias (their gradients are rounded to these).
    mutant: None or one of 'M01','M02','M03','M04','M05','M12','M13','nocenter'.
    Returns y, dx, dw, db and max |intermediate| for overflow inspection.
    """
    R = lambda a: rnd(a, ct)
    x = np.asarray(x, np.float64); w = np.asarray(w, np.float64); b = np.asarray(b, np.float64)
    p = np.asarray(p, np.float64)
    N, C = x.shape[:2]; spatial = list(x.shape[2:]); S = int(np.prod(spatial)) if spatial else 1
    M = (C // G) * S
    if mutant == "M05":
        M = C * S  # reshape refuses: element count mismatch unless G == 1
        if N * G * M != x.size:
            raise RuntimeError("reshape: element count mismatch")
    g0 = x.reshape(N, G, M)
    rax = 1 if mutant == "M04" else 2
    cnt = g0.shape[rax]
    inv_cnt = rnd(1.0 / cnt, "float32")
    big = 0.0
    mean = R(rsum(g0, rax, True, ct, order) * inv_cnt)
    m2 = R(rsum(g0, rax, True, ct, order) * inv_cnt)
    d = R(g0 - m2); dd = R(d * d)
    var = R(rsum(dd, rax, True, ct, order) * inv_cnt)
    if mutant == "M01":
        var = R(var * (g0.shape[2] / (g0.shape[2] - 1)))
    feps = rnd(eps, "float32")
    if mutant == "M02":
        sq = R(np.power(var, 0.5)); ve = R(sq + feps); pw = -1.0; base = ve
    elif mutant == "M03":
        ve = var; pw = -0.5; base = var
    elif mutant == "M13":
        ve = R(var + feps); pw = -1.0; base = ve
    else:
        ve = R(var + feps); pw = -0.5; base = ve
    with np.errstate(all="ignore"):
        s = R(np.power(base, pw))
    c = R(g0 - mean) if mutant != "nocenter" else g0
    h = R(c * s) if mutant != "M12" else g0
    h2 = h.reshape(x.shape)
    ws = [1, C] + [1] * len(spatial)
    wr = R(rnd(w, w_dt)).reshape(ws); br = R(rnd(b, b_dt)).reshape(ws)
    t = R(h2 * wr); y = R(t + br)
    y_out = rnd(y, out_dt or ct)
    # ---------------- backward (upstream gradient p, exact in ct)
    with np.errstate(all="ignore"):
        gy = R(p)
        gbr = reduce_to(gy, ws, ct, order)
        gt = gy
        gh2 = R(gt * wr); gwr = reduce_to(R(gt * h2), ws, ct, order)
        gh = gh2.reshape(g0.shape)
        if mutant == "M12":
            gg0_parts = [gh]
            gx = rnd(gh, ct)
        else:
            gc = R(gh * s); gs = rsum(R(gh * c), 2 if rax == 2 else 1, True, ct, order)
            if mutant == "M03":
                p15 = R(np.power(var, -1.5)); gve = R(gs * R(p15 * -0.5)); gvar = gve
            elif mutant == "M02":
                p2 = R(np.power(ve, -2.0)); gve = R(gs * R(p2 * -1.0))
                psq = R(np.power(var, -0.5)); gvar = R(gve * R(psq * 0.5))
            elif mutant == "M13":
                p2 = R(np.power(ve, -2.0)); gvar = R(gs * R(p2 * -1.0))
            else:
                p15 = R(np.power(ve, -1.5)); gve = R(gs * R(p15 * -0.5)); gvar = gve
            big = max(big, float(np.nanmax(np.abs(p15))) if mutant not in ("M02", "M13") else 0.0)
            if mutant == "M01":
                gvar = R(gvar * (g0.shape[2] / (g0.shape[2] - 1)))
            gsum = R(gvar * inv_cnt)
            gdd = np.broadcast_to(gsum, dd.shape)
            gd = R(R(gdd * d) + R(gdd * d))
            gm2 = rsum(R(-gd), rax, True, ct, order)
            gmean = rsum(R(-gc), rax, True, ct, order) if mutant != "nocenter" else None
            parts = [gd, np.broadcast_to(R(gm2 * inv_cnt), g0.shape), gc]
            if gmean is not None:
                parts.append(np.broadcast_to(R(gmean * inv_cnt), g0.shape))
            acc = parts[0]
            for part in parts[1:]:
                acc = R(acc + part)
            gx = acc
    dx = rnd(gx.reshape(x.shape), out_dt or ct)
    dw = rnd(gwr.reshape(C), w_dt); db = rnd(gbr.reshape(C), b_dt)
    return y_out, dx, dw, db, big


# ------------------------------------------------------------------ running error bound (float32 candidate)
def running_bound(x, G, w, b, eps, p, upow=8 * U32):
    """First-order bounds |computed - exact| for y and dx, dw, db of the float32
    candidate graph (any summation order). Each op: |fl(z)-z| <= u|z| on top of
    propagated input errors; sum of m terms adds gamma_{m-1} * sum|terms|.
    powf: upow relative. Inputs are exact (represented operands)."""
    u = U32
    gam = lambda m: (m - 1) * u / (1 - (m - 1) * u)
    x = np.asarray(x, np.float64); w = np.asarray(w, np.float64); b = np.asarray(b, np.float64)
    p = np.asarray(p, np.float64)
    N, C = x.shape[:2]; spatial = list(x.shape[2:]); S = int(np.prod(spatial)) if spatial else 1
    M = (C // G) * S
    g0 = x.reshape(N, G, M); A = np.abs
    ic = 1.0 / M; e_ic = u * ic  # fl(1/M)

    def ssum(v, e, axis=2):  # value, error bound of a float32 sum
        m = v.shape[axis]
        return v.sum(axis=axis, keepdims=True), e.sum(axis=axis, keepdims=True) + gam(m) * A(v).sum(axis=axis, keepdims=True)

    def smul(a, ea, bb, eb):
        z = a * bb
        exact = np.isscalar(bb) and bb == 1.0 and eb == 0.0
        return z, A(bb) * ea + A(a) * eb + (0.0 if exact else u) * A(z)

    def sadd(a, ea, bb, eb):
        z = a + bb
        return z, ea + eb + u * A(z)

    z0 = np.zeros_like(g0)
    sm, esm = ssum(g0, z0)
    if M == 1:  # singleton group: fl(x*1.0) = x and x - x = 0 are exact
        mean, emean = sm, esm
        d, ed = z0.copy(), z0.copy()
        e_ic = 0.0
    else:
        mean, emean = smul(sm, esm, ic, e_ic)
        d, ed = sadd(g0, z0, -mean, emean)
    dd, edd = smul(d, ed, d, ed)
    sv, esv = ssum(dd, edd)
    var, evar = smul(sv, esv, ic, e_ic)
    ve, eve = sadd(var, evar, eps, u * eps)
    s = ve ** -0.5
    es = 0.5 * ve ** -1.5 * eve + upow * s
    c, ec = d, ed
    h, eh = smul(c, ec, s, es)
    ws = [1, C] + [1] * len(spatial)
    h2 = h.reshape(x.shape); eh2 = eh.reshape(x.shape)
    wr = w.reshape(ws); br = b.reshape(ws)
    t, et = smul(h2, eh2, wr, 0 * wr)
    y, ey = sadd(t, et, br, 0 * br)
    # backward
    gy = p; egy = 0 * p
    red = (0,) + tuple(range(2, x.ndim))

    def rsum_axes(v, e, axes):
        m = int(np.prod([v.shape[a] for a in axes]))
        return v.sum(axis=axes), e.sum(axis=axes) + gam(m) * A(v).sum(axis=axes)

    db, edb = rsum_axes(gy, egy, red)
    q1, eq1 = smul(gy, egy, h2, eh2)
    dw, edw = rsum_axes(q1, eq1, red)
    gh2, egh2 = smul(gy, egy, wr, 0 * wr)
    gh = gh2.reshape(g0.shape); egh = egh2.reshape(g0.shape)
    gc, egc = smul(gh, egh, s, es)
    ghc, eghc = smul(gh, egh, c, ec)
    gs, egs = ssum(ghc, eghc)
    p15 = ve ** -1.5
    ep15 = 1.5 * ve ** -2.5 * eve + upow * p15
    k, ek = smul(p15, ep15, -0.5, 0.0)
    gve, egve = smul(gs, egs, k, ek)
    gsum, egsum = smul(gve, egve, ic, e_ic)
    gdd1, egdd1 = smul(np.broadcast_to(gsum, d.shape), np.broadcast_to(egsum, d.shape), d, ed)
    gd = 2 * gdd1; egd = 2 * egdd1 + u * A(gd)  # sum of two equal rounded products (exact doubling) - u kept
    gm2s, egm2s = ssum(-gd, egd)
    gm2, egm2 = smul(gm2s, egm2s, ic, e_ic)
    gms, egms = ssum(-gc, egc)
    gmean, egmean = smul(gms, egms, ic, e_ic)
    parts = [(gd, egd), (np.broadcast_to(gm2, g0.shape), np.broadcast_to(egm2, g0.shape)),
             (gc, egc), (np.broadcast_to(gmean, g0.shape), np.broadcast_to(egmean, g0.shape))]
    acc, eacc = parts[0]
    for v, e in parts[1:]:
        acc, eacc = sadd(acc, eacc, v, e)
    if M == 1:  # gx = gd(=0) + gm2(=0) + gc + (-gc) = 0 exactly (Sterbenz: x - x)
        acc, eacc = np.zeros_like(acc), np.zeros_like(eacc)
    return dict(y=y, ey=ey, dx=acc.reshape(x.shape), edx=eacc.reshape(x.shape), dw=dw, edw=edw, db=db, edb=edb)
