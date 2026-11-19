"""G1 fixtures from P1-D/P1-F3 and PF2 (Project-Tensor, 2026)."""
import numpy as np

GROUPS = (1, 2, 3, 4, 5, 6, 7, 8)

def l1_cases():
    out = []
    for D in (64, 128):
        for B in (1, 2):
            for G in GROUPS:
                for S in (255, 256, 257, 2048):
                    for causal in (False, True):
                        for sinks in (False, True):
                            out.append(dict(D=D, B=B, KV=2, H=2 * G, S=S, causal=causal, sinks=sinks))
    return out


def case_inputs(c, i):
    rng = np.random.default_rng(np.random.SeedSequence([20261115, 1, i]))
    q = (3 * rng.standard_normal((c['B'], c['H'], 1, c['D']))).astype(np.float32)
    k = rng.standard_normal((c['B'], c['KV'], c['S'], c['D']), dtype=np.float32)
    v = rng.standard_normal((c['B'], c['KV'], c['S'], c['D']), dtype=np.float32)
    s = rng.standard_normal((c['H'],), dtype=np.float32) if c['sinks'] else None
    return q, k, v, s


def g1_cases():
    """The design's PF-I1 grid (DESIGN.md 5.1) with VD = D, plus two VD sub-grids that reach the D64 lane order
    (VD 128 with G in {1, 2}) and the CAP 256/512 instantiations. fp16 is not a packed K dtype (refused, G4)."""
    out = []
    for D in (64, 128):
        for G in (1, 2, 4, 8):
            for dt in ('float32', 'bfloat16'):
                for B in (1, 2):
                    for S in (255, 256, 257, 2047, 2049, 8192):
                        for L in (2, 33, S):
                            for causal in (False, True):
                                for sinks in (False, True):
                                    for capx in (0, 37):
                                        out.append(dict(D=D, VD=D, G=G, dt=dt, B=B, S=S, L=L, causal=causal,
                                                        sinks=sinks, cap=S + capx, sub='grid'))
    for G in (1, 2, 4, 8):
        for dt in ('float32', 'bfloat16'):
            for S in (257, 2049):
                for L in (2, 33, S):
                    for causal in (False, True):
                        for capx in (0, 37):
                            out.append(dict(D=64, VD=128, G=G, dt=dt, B=1, S=S, L=L, causal=causal, sinks=True,
                                            cap=S + capx, sub='d64_vd128'))
    for VD in (256, 512):
        for G in (1, 4):
            for dt in ('float32', 'bfloat16'):
                for L in (2, 33, 257):
                    for causal in (False, True):
                        out.append(dict(D=128, VD=VD, G=G, dt=dt, B=1, S=257, L=L, causal=causal, sinks=True,
                                        cap=257, sub='vd%d' % VD))
    return out


def case_id(c):
    return (f"{c['sub']}_D{c['D']}_VD{c['VD']}_G{c['G']}_{c['dt']}_B{c['B']}_S{c['S']}_L{c['L']}_c{int(c['causal'])}"
            f"_s{int(c['sinks'])}_cap{c['cap']}")


def case_tensors(torch, c, idx):
    rng = np.random.default_rng(np.random.SeedSequence([20261116, 1, idx]))
    KV = 2
    H = c['G'] * KV
    q = (3 * rng.standard_normal((c['B'], H, c['L'], c['D']))).astype(np.float32)
    k = rng.standard_normal((c['B'], KV, c['S'], c['D']), dtype=np.float32)
    v = rng.standard_normal((c['B'], KV, c['S'], c['VD']), dtype=np.float32)
    s = rng.standard_normal((H,), dtype=np.float32) if c['sinks'] else None
    tdt = torch.float32 if c['dt'] == 'float32' else torch.bfloat16
    T = lambda a: None if a is None else torch.from_numpy(a).cuda().to(tdt).contiguous()  # noqa: E731
    return T(q), T(k), T(v), T(s)


class Env:
    def __init__(self, tc, torch):
        self.tc, self.torch = tc, torch
        self.spo2 = tc._spo2

    def to_tc(self, x):
        return None if x is None else self.tc.from_dlpack(x.contiguous())

    def tt(self, t):
        return self.torch.from_dlpack(t)

    def store(self, k, cap):
        ws = self.tc.ApaSpq1Workspace(self.to_tc(k), cap, fmt='tq4')
        return ws, ws.packed

    def prefill(self, q, k, v, packed, scale, delta, causal, sinks, dump=False):
        r = self.spo2.packed_prefill(self.to_tc(q), self.to_tc(k), self.to_tc(v), packed, scale, delta, causal,
                                     self.to_tc(sinks), True, fmt='tq4', dump=dump)
        return tuple(self.tt(x) for x in r)

    def decode(self, q1, k1, v1, packed, scale, delta, sinks):
        r = self.spo2.packed_run(self.to_tc(q1), self.to_tc(k1), self.to_tc(v1), packed, scale, delta, False,
                                 self.to_tc(sinks), True, fmt='tq4', carry='row', dump=True, gqa_path='r2')
        return tuple(self.tt(x) for x in r)   # out [B,H,1,VD], mask, bulk, prefix [B,H,1,S]

    def dense(self, q, k, v, scale, delta, causal, sinks):
        r = self.tc._C.apa_selective_attention_sp(self.to_tc(q), self.to_tc(k), self.to_tc(k), self.to_tc(v), scale,
                                                  delta, causal, self.to_tc(sinks), True)
        return tuple(self.tt(x) for x in r)
