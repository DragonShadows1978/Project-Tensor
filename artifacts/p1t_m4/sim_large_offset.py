"""CPU simulation (reasoning receipt, not a gate): fp32 statistics on a large-offset row (mean 1e4, std 1e-2).
Compares (1) plain two-pass fp32 (mean rounded to fp32, then centred sum of squares), (2) shifted two-pass fp32
(shift K = x[0]), (3) one-pass E[x^2]-E[x]^2 fp32, each against a float64 two-pass oracle; normwise N."""
import numpy as np
rng = np.random.default_rng(20261012)
for L in (82, 1193, 3000, 65536):
    z = rng.standard_normal(L)
    x = (1e4 + 1e-2 * z).astype(np.float32)
    x64 = x.astype(np.float64)
    m = x64.mean(); xc = x64 - m; o = xc / np.sqrt((xc * xc).mean() + 1e-5)
    f = np.float32
    def N(y): return np.max(np.abs(y.astype(np.float64) - o)) / np.max(np.abs(o))
    # (1) plain two-pass, fp32 throughout (numpy's float32 pairwise sum: more accurate than any per-thread loop)
    mu = f(x.sum(dtype=np.float32) / f(L)); c = x - mu
    var = f((c * c).sum(dtype=np.float32) / f(L)); y1 = c * (f(1) / np.sqrt(var + f(1e-5)))
    # (2) shifted two-pass, K = x[0]
    K = x[0]; s = x - K; d = f(s.sum(dtype=np.float32) / f(L)); c2 = s - d
    var2 = f((c2 * c2).sum(dtype=np.float32) / f(L)); y2 = c2 * (f(1) / np.sqrt(var2 + f(1e-5)))
    # (3) one-pass E[x^2]-E[x]^2 on raw x, clamped at 0
    mu3 = f(x.sum(dtype=np.float32) / f(L)); ex2 = f((x * x).sum(dtype=np.float32) / f(L))
    var3 = max(f(ex2 - mu3 * mu3), f(0)); y3 = (x - mu3) * (f(1) / np.sqrt(var3 + f(1e-5)))
    print(f'L={L:6d}  plain two-pass N={N(y1):.3e}  shifted two-pass N={N(y2):.3e}  one-pass N={N(y3):.3e}  '
          f'(fp32 mean err {abs(float(mu) - m):.2e}, std {np.sqrt((xc*xc).mean()):.2e})')
