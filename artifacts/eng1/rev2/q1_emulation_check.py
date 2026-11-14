#!/usr/bin/env python3
"""ENG1-REV2 Q1: an independent, straight-line re-emulation of two E3 classes end to end (LayerNorm: the reviewed
class, all 8 rows; GroupNorm: the float32-upcast class, all 8 rows), compared bit for bit with test_eng1.py's
e3_emulation tape, plus the review's two forward numbers, plus an independent float64 reference.

Independence: rounding uses PyTorch's float32->float16/bfloat16 conversions (not the test's NumPy cast / bit-hack);
arithmetic is torch float32 elementwise ops; reductions are hand-written lane loops copied from kernels.cu
reduce_lastdim_kernel (256 lanes, serial per lane, then the halving tree); the reverse-mode accumulation ORDER is
written out by hand from autograd.cpp's post-order DFS over the nn.py graph (derivation in REPORT.md), not computed
by a tape. The float64 reference is my own closed form, not e3_reference.
Prior art: IEEE 754-2019 roundTiesToEven (binary16; bfloat16 as the top half of binary32 with RNE, as implemented
by PyTorch's c10::BFloat16 round_to_nearest_even); Wengert (1964) / Linnainmaa (1970) reverse accumulation;
layer norm (Ba, Kiros and Hinton, 2016) and group norm (Wu and He, 2018) VJPs; Higham (2002) for the error
vocabulary. Taken: the standard definitions. Ours: the hand-ordered transcription for this engine. Citations
unverified -- lead to check ("IEEE 754-2019 roundTiesToEven", "c10 BFloat16 round_to_nearest_even").
"""
import importlib.util, json, math, sys
from pathlib import Path
import numpy as np
import torch

torch.set_num_threads(1)
REV2 = Path(__file__).resolve().parent
ROOT = REV2.parents[2]
ARM = ROOT / "artifacts/eng1/tool/arms/candidate/tensor_cuda"
sys.path.insert(0, str(ARM))
spec = importlib.util.spec_from_file_location("eng1_test", ROOT / "tensor_cuda/tests/test_eng1.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

F32 = torch.float32
TD = {"float16": torch.float16, "bfloat16": torch.bfloat16}


def R(t, dt):
    """Store a float32 tensor in dt (RNE via PyTorch) and return the stored value as float32."""
    return t.to(TD[dt]).to(F32) if dt in TD else t.to(F32)


def f32(v):
    return torch.tensor(float(np.float32(v)), dtype=F32)


def tree_last(t):
    """reduce_lastdim_kernel: lane i accumulates elements i, i+256, ... in order (fp32), then the halving tree."""
    lead, n = t.shape[:-1], t.shape[-1]
    t2 = t.reshape(-1, n)
    lanes = torch.zeros(t2.shape[0], 256, dtype=F32)
    for i in range(n):
        lanes[:, i % 256] = lanes[:, i % 256] + t2[:, i]
    off = 128
    while off:
        lanes[:, :off] = lanes[:, :off] + lanes[:, off:2 * off]
        off //= 2
    return lanes[:, 0].reshape(*lead, 1)


def powf_cr(t, p):
    """Correctly rounded float32 power (float64 pow, one rounding to float32)."""
    return torch.pow(t.to(torch.float64), p).to(F32)


def layernorm(x, w, b, g, dt):
    """nn.py LayerNorm: mean=x.mean; xc=x-mean; var=(xc*xc).mean; inv=(var+eps)^-0.5; y=xc*inv*w+b."""
    s16 = f32(1.0 / 16)
    s1 = R(tree_last(x), dt); mean = R(s1 * s16, dt)
    xc = R(x - mean, dt); sq = R(xc * xc, dt)
    s2 = R(tree_last(sq), dt); var = R(s2 * s16, dt)
    ve = R(var + f32(1e-5), dt); inv = R(powf_cr(ve, -0.5), dt)
    a = R(xc * inv, dt); yw = R(a * w, dt); y = R(yw + b, dt)
    # Reverse order (post-order DFS reversed): T, Y, y(add), yw(mul a,w), a(mul xc,inv), inv(pow), ve, var, s2,
    # sq(mul xc,xc), xc(sub x,mean), mean, s1, x.
    gy = R(torch.ones_like(y) * g, dt)          # Y: mul(y, g) -> y.grad = g * 1
    gyw = gy                                     # add(yw, b)
    ga = R(gyw * w, dt)                          # mul(a, w) -> a.grad
    c1 = R(ga * inv, dt)                         # mul(xc, inv) -> xc.grad (first contribution)
    ginv = R(tree_last(R(ga * xc, dt)), dt)      # mul(xc, inv) -> inv.grad, reduced over the broadcast axis
    sc = powf_cr(ve, -1.5) * f32(-0.5)           # E2 widened: nmuls(ew_pow(x32, p-1), p) in fp32
    gve = R(ginv * sc, dt)                       # nmul(g32, .) in fp32, ONE cast to dt
    gvar = gve                                   # add_scalar
    gs2 = R(gvar * s16, dt)                      # mul_scalar
    gsq = R(torch.zeros_like(sq) + gs2, dt)      # sum -> broadcast
    d = R(gsq * xc, dt)                          # mul(xc, xc): a then b, both equal
    gxc = R(R(c1 + d, dt) + d, dt)
    gmean = R(tree_last(R(-gxc, dt)), dt)        # sub(x, mean) -> mean.grad = reduce(-g)
    gs1 = R(gmean * s16, dt)                     # mul_scalar
    gx = R(gxc + (torch.zeros_like(x) + gs1), dt)   # x.grad = gxc (sub) then + broadcast(s1.grad)
    return y, gx


def groupnorm(x, w, b, g, dt):
    """nn.py GroupNorm (G=2, half input): float32 work, cast back once; var() has its own mean node."""
    N, C = x.shape[:2]
    X = x.reshape(N, 2, -1)                      # cast (exact) then reshape
    n = X.shape[-1]
    sn = f32(1.0 / n)
    m1 = tree_last(X) * sn                       # mean = sum (fp32) * (float)(1/n), fp32
    m2 = tree_last(X) * sn                       # var's own mean
    D = X - m2; Q = D * D; var = tree_last(Q) * sn
    ve = var + f32(1e-5); P = powf_cr(ve, -0.5)
    Xc = X - m1; Nn = Xc * P
    N4 = Nn.reshape(x.shape)
    W = N4 * w.reshape(1, C, 1, 1); B = W + b.reshape(1, C, 1, 1)
    y = R(B, dt)
    # Reverse order: T, Y, C1(cast), B, W, R1, N(mul Xc,P), P, E, V, S3, Q, D, M2, S2, X(sub), M1, S1, R0, C0, x
    gY = R(torch.ones_like(y) * g, dt)           # y.grad (dt)
    gB = gY                                      # cast back to fp32: exact
    gN4 = gB * w.reshape(1, C, 1, 1)             # fp32
    gN = gN4.reshape(Nn.shape)
    gXc = gN * P                                 # X(sub).grad (fp32)
    gP = tree_last(gN * Xc)
    gE = gP * (powf_cr(ve, -1.5) * f32(-0.5))    # float32 path of pow backward (unchanged line)
    gS3 = gE * sn
    gQ = torch.zeros_like(Q) + gS3
    gD = gQ * D
    gD = gD + gQ * D
    gR0 = gD                                     # D: sub(R0, M2) -> R0 first contribution
    gM2 = tree_last(-gD)
    gS2 = gM2 * sn
    gR0 = gR0 + (torch.zeros_like(X) + gS2)      # S2 broadcast
    gR0 = gR0 + gXc                              # X: sub(R0, M1) -> R0
    gM1 = tree_last(-gXc)
    gS1 = gM1 * sn
    gR0 = gR0 + (torch.zeros_like(X) + gS1)      # S1 broadcast
    gx = R(gR0.reshape(x.shape), dt)             # cast's VJP: one rounding to dt
    return y, gx


def ref64(cls, x, w, b, g):
    """My own float64 closed forms (Ba et al. 2016; Wu and He 2018): y and dL/dx for L = sum(y * g)."""
    if cls == "LayerNorm":
        xs, gs, ax, wb, bb = x, g * w, -1, w, b
    else:
        N, C = x.shape[:2]
        wb, bb = w.reshape(1, C, 1, 1), b.reshape(1, C, 1, 1)
        xs, gs, ax = x.reshape(N, 2, -1), (g * wb).reshape(N, 2, -1), -1
    mu = xs.mean(axis=ax, keepdims=True)
    sig = np.sqrt(((xs - mu) ** 2).mean(axis=ax, keepdims=True) + 1e-5)
    xh = (xs - mu) / sig
    n = xs.shape[-1]
    dx = (n * gs - gs.sum(axis=ax, keepdims=True) - xh * (gs * xh).sum(axis=ax, keepdims=True)) / (n * sig)
    return xh.reshape(x.shape) * wb + bb, dx.reshape(x.shape)


def stored(cls, case, dt):
    x = R(torch.tensor(M.e3_inputs(cls, case), dtype=torch.float64).to(F32), dt)
    g = R(torch.tensor(M._e3_pattern(M.E3_SHAPE[cls], 1.1), dtype=torch.float64).to(F32), dt)
    C, _ = M._affine_shape(cls, M.E3_SHAPE[cls])
    w = R(torch.tensor(1.0 + 0.1 * np.arange(C), dtype=torch.float64).to(F32), dt)
    b = R(torch.tensor(0.05 * np.arange(C) - 0.1, dtype=torch.float64).to(F32), dt)
    return x, w, b, g


rows, checks = [], {}
a1 = {(r["cls"], r["dtype"], r["case"], r["direction"]): r for r in
      json.loads((ROOT / "artifacts/eng1/a1/e3_emulation.json").read_text())}
with np.errstate(over="ignore", invalid="ignore", under="ignore", divide="ignore"):
    for cls, fn in (("LayerNorm", layernorm), ("GroupNorm", groupnorm)):
        for dt in ("float16", "bfloat16"):
            for case in ("regular", "near_eps"):
                x, w, b, g = stored(cls, case, dt)
                xn, wn, bn, gn = (t.to(torch.float64).numpy() for t in (x, w, b, g))
                # The same stored operands under the test's own rounding instrument (independence check).
                same_store = all(np.array_equal(M.e3_round(v, dt), u) for v, u in
                                 ((M.e3_inputs(cls, case), xn), (M._e3_pattern(M.E3_SHAPE[cls], 1.1), gn)))
                mine = [t.to(torch.float64).numpy() for t in fn(x, w, b, g, dt)]
                tape = M.e3_emulation(cls, dt, xn, wn, bn, gn)
                mref = ref64(cls, xn, wn, bn, gn)
                tref = M.e3_reference(cls, xn, wn, bn, gn)
                for k, direction in enumerate(("forward", "backward")):
                    bit_equal = bool(np.array_equal(mine[k], tape[k], equal_nan=True))
                    err = M.e3_distance(mine[k], mref[k])
                    row = dict(cls=cls, dtype=dt, case=case, direction=direction, stored_operands_agree=same_store,
                               mine_vs_tape_bit_equal=bit_equal,
                               mine_nonfinite=int(np.count_nonzero(~np.isfinite(mine[k]))),
                               tape_nonfinite=int(np.count_nonzero(~np.isfinite(tape[k]))),
                               max_abs_mine_minus_tape=(None if not bit_equal and (np.any(~np.isfinite(mine[k])) or np.any(~np.isfinite(tape[k]))) else float(np.max(np.abs(mine[k] - tape[k])))),
                               my_ref_vs_test_ref_max_abs=float(np.max(np.abs(mref[k] - tref[k]))),
                               my_emulation_error_vs_my_float64=err,
                               a1_emulation_error_vs_float64=a1[(cls, dt, case, direction)]["emulation_error_vs_float64"],
                               a1_target=a1[(cls, dt, case, direction)]["accuracy_target"])
                    row["error_agrees_with_a1_within_1e-12"] = (
                        (err is None and row["a1_emulation_error_vs_float64"] is None) or
                        (err is not None and row["a1_emulation_error_vs_float64"] is not None and
                         abs(err - row["a1_emulation_error_vs_float64"]) <= 1e-12))
                    rows.append(row)
    # The review's two forward numbers (REV1 numerics.py), reproduced with my own instruments.
    x, w, b, g = stored("LayerNorm", "near_eps", "bfloat16")
    y, _ = layernorm(x, w, b, g, "bfloat16")
    ln = float(np.max(np.abs(y.double().numpy() - ref64("LayerNorm", *(t.double().numpy() for t in (x, w, b, g)))[0])))
    xr, wr, _, gr = stored("RMSNorm", "near_eps", "float16")
    s16 = f32(1.0 / 16)
    ms = R(R(tree_last(R(xr * xr, "float16")), "float16") * s16, "float16")
    yr = R(R(xr * R(powf_cr(R(ms + f32(1e-6), "float16"), -0.5), "float16"), "float16") * wr, "float16")
    xd, wd = xr.double().numpy(), wr.double().numpy()
    rref = xd / np.sqrt((xd * xd).mean(axis=-1, keepdims=True) + 1e-6) * wd
    rms = float(np.max(np.abs(yr.double().numpy() - rref)))
    checks["review_numbers"] = dict(
        RMSNorm_float16_near_eps_forward=dict(mine=rms, rev1=0.0156582687232443, abs_diff=abs(rms - 0.0156582687232443)),
        LayerNorm_bfloat16_near_eps_forward=dict(mine=ln, rev1=0.204845083463340, abs_diff=abs(ln - 0.204845083463340)))
summary = dict(rows=len(rows), all_bit_equal=all(r["mine_vs_tape_bit_equal"] for r in rows),
               all_stored_agree=all(r["stored_operands_agree"] for r in rows),
               all_errors_agree_with_a1=all(r["error_agrees_with_a1_within_1e-12"] for r in rows),
               max_ref_disagreement=max(r["my_ref_vs_test_ref_max_abs"] for r in rows))
out = dict(summary=summary, checks=checks, rows=rows)
(REV2 / "q1_emulation_check.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(summary)); print(json.dumps(checks))
for r in rows:
    print(r["cls"], r["dtype"], r["case"], r["direction"], "bit_equal=", r["mine_vs_tape_bit_equal"],
          "nonfinite mine/tape=", r["mine_nonfinite"], r["tape_nonfinite"], "err=", r["my_emulation_error_vs_my_float64"],
          "a1=", r["a1_emulation_error_vs_float64"], "target=", r["a1_target"])
