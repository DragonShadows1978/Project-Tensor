#!/usr/bin/env python3
"""ENG1-REV2 Q3: re-derive, in exact rational arithmetic, whether the new non-power-of-two E2 witnesses separate the
final-cast decision, and whether M15 (bfloat16-only early cast) is killed by the bfloat16 witness.

Instrument: Python fractions.Fraction (exact), and my own IEEE 754 roundTiesToEven to a binary format with p
significand bits and minimum normal exponent emin (subnormals included). No NumPy/PyTorch rounding is used, so this is
independent of the test file's and A1's instruments. Engine operation sequences are transcribed from ops.cpp
pow_scalar (candidate), mutants/M15/mutation.diff (M15) and the baseline's unchanged float-path line applied in the
half dtype (baseline arm: nmul(g, nmuls(ew_pow(x, p-1), p)) with every op stored in the dtype).
The tolerance is test_eng1.py's registered E2 form: atol + rtol*|ref|, rtol = u_d + 2^-20, atol = 2^-25 (float16) or
2^-134 (bfloat16) -- read from the module constants, not chosen here.
Prior art: IEEE 754-2019 roundTiesToEven; Higham (2002) unit roundoff; DeMillo, Lipton and Sayward (1978) mutation
analysis; CUDA powf <= 4 ulp (CUDA C Programming Guide; unverified -- lead to check). Ours: the transcription.
"""
import json
from fractions import Fraction as Fr
from pathlib import Path

REV2 = Path(__file__).resolve().parent
FMT = {"float32": (24, -126), "float16": (11, -14), "bfloat16": (8, -126)}


def rne(v, fmt):
    """Round the exact rational v to the nearest representable value of fmt, ties to even (finite range only)."""
    p, emin = FMT[fmt]
    if v == 0:
        return Fr(0)
    s, a = (-1 if v < 0 else 1), abs(v)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    while Fr(2) ** e > a:
        e -= 1
    while Fr(2) ** (e + 1) <= a:
        e += 1
    q = Fr(2) ** (max(e, emin) - (p - 1))          # spacing at this binade (subnormal spacing below emin)
    n = a / q
    fl = n.numerator // n.denominator
    rem = n - fl
    if rem > Fr(1, 2) or (rem == Fr(1, 2) and fl % 2 == 1):
        fl += 1
    return s * fl * q


def ulp32(v):
    p, emin = FMT["float32"]
    a = abs(v)
    e = 0
    while Fr(2) ** e > a:
        e -= 1
    while Fr(2) ** (e + 1) <= a:
        e += 1
    return Fr(2) ** (max(e, emin) - (p - 1))


def paths(x, g, p, dt, powf_ulps=0):
    pw = rne(x ** (p - 1), "float32")                # correctly rounded powf(x32, p-1) (p-1 = 2: exact here)
    pw = pw + powf_ulps * ulp32(pw)                  # CUDA powf error model: +-k ulp
    sc = rne(pw * p, "float32")                      # nmuls(., p) in fp32
    late = rne(rne(g * sc, "float32"), dt)           # candidate: nmul(g32, .) fp32, ONE cast to dt
    early = rne(rne(g * rne(sc, dt), "float32"), dt) # M15 (and M07's float16 analogue): cast p*x^(p-1) first
    bpw = rne(rne(x ** (p - 1) + powf_ulps * ulp32(x ** (p - 1)), "float32"), dt)
    base = rne(rne(g * rne(rne(bpw * p, "float32"), dt), "float32"), dt)  # baseline arm: every op stored in dt
    return late, early, base


U = {"float16": Fr(1, 2 ** 11), "bfloat16": Fr(1, 2 ** 8)}
ATOL = {"float16": Fr(1, 2 ** 25), "bfloat16": Fr(1, 2 ** 134)}
W = {"float16": (Fr(0.1492919921875), Fr(0.30078125), 3), "bfloat16": (Fr(0.0002994537353515625), Fr(0.30078125), 3)}
out = {}
for dt, (x, g, p) in W.items():
    assert rne(x, dt) == x and rne(g, dt) == g, "witness operands are not exactly stored in " + dt
    ref = g * p * x ** (p - 1)                       # exact derivative of g * x^p at the stored operands
    tol = ATOL[dt] + (U[dt] + Fr(1, 2 ** 20)) * abs(ref)
    rows = {}
    for k in range(-4, 5):
        late, early, base = paths(x, g, p, dt, k)
        rows[k] = dict(late_err=float(abs(late - ref)), early_err=float(abs(early - ref)), base_err=float(abs(base - ref)),
                       late_passes=abs(late - ref) <= tol, early_fails=abs(early - ref) > tol,
                       baseline_passes=abs(base - ref) <= tol)
    l0, e0, b0 = paths(x, g, p, dt, 0)
    out[dt] = dict(x=float(x), g=float(g), p=p, g_is_power_of_two=(g.numerator & (g.numerator - 1) == 0 and
                                                                  g.denominator & (g.denominator - 1) == 0),
                   reference=float(ref), tolerance=float(tol), late=float(l0), early=float(e0), baseline=float(b0),
                   late_err=float(abs(l0 - ref)), early_err=float(abs(e0 - ref)), baseline_err=float(abs(b0 - ref)),
                   early_over_tol=float(abs(e0 - ref) / tol), late_over_tol=float(abs(l0 - ref) / tol),
                   separates_at_all_powf_ulps=all(r["late_passes"] and r["early_fails"] for r in rows.values()),
                   baseline_passes_at_all_powf_ulps=all(r["baseline_passes"] for r in rows.values()),
                   baseline_fails_at_all_powf_ulps=all(not r["baseline_passes"] for r in rows.values()),
                   powf_ulp_sweep=rows)
(REV2 / "q3_m15_arith.json").write_text(json.dumps(out, indent=1) + "\n")
for dt, r in out.items():
    print(dt, {k: v for k, v in r.items() if k != "powf_ulp_sweep"})
