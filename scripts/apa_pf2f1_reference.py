#!/usr/bin/env python3
"""APA-P1-PF2 GPU lines (run only inside forge-gpu-run) and their CPU steps.

Modes:
  stage <PF|PFH|mutant:<name>>   CPU: stage an importable arm (copies, never symlinks) under artifacts/apa_pf2f1/arms
  precheck <arm>                 CPU, before the lease: module sha256 == its build receipt (L103); for PF/PFH also the
                                 working tree's engine source map == the receipt's
  g1  --arm PF                   G1 on the product build: PF-I1 masks (grid + exact-tie rows), PF-I1n, PF-I2, PF-I3
  g1h --arm PFH                  G1 on the hooks build: PF-I1 bulk scores (uint32 views) and masks, same grid
  (more modes are added by the other line scripts of this round)

PF-I1's oracle: the packed L = 1 decode (spo2.packed_run, carry="row", dump=True, the same delta) run with query row
i on the store's first smax_i keys (k and v sliced to smax_i rows, the same store tuple; the packed quantizer works
row by row, packed.cuh prepare/pack_tq, so the store prefix is the store a fill of only those keys gives; checked on
a sample by PREFIX_STORE rows). PF-I2's reference: the dense prefill kernel (_C.apa_selective_attention_sp, raw delta)
on the same q, k, v, sinks with kq = k, at a delta that refines every key (checked). PF-I3: per row
||out_prefill - out_decode|| <= tau ||out_decode||, tau = min(4 x the largest value of the same comparison in the
all-refined regime, 1e-5) (rule registered in artifacts/apa_pf2f1/REGISTRATION.json before the line).

Prior art: P1-D/F1/F2/F3 paired unsigned-view bit equality and line plumbing (Project-Tensor 2026); the exact-tie
idea of tests/test_apa_p1_exact_tie.py (bit-identical scores straddling a cutoff), restated for the SP threshold;
IEEE 754 round-to-nearest subtraction for the tie construction. Mutation testing: DeMillo, Lipton and Sayward (1978).
No new algorithm here.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/apa_pf2f1'
WT = ROOT.parent
F3_ARM = WT / 'tensor_fork_apa_wt_p1f2/artifacts/apa_p1f3/arms/F3'
F3_SHA = '2e557a23eaf228512a2f1a2f0629401d91d02d05705087661be5c48849c0f3e9'
P1D_ART = ROOT / 'artifacts/apa_p1d'
PACKED_TABLE = P1D_ART / 'packed_e_q_table.json'
DENSE_TABLE = Path('/mnt/ForgeRealm/Project-Tensor/artifacts/apa_sp2/e_q_table.json')
DENSE_TABLE_SHA = '5b6f8bfb405b40c03da21ad8d50eb40c11b85ed9559ee717d2a105c90c0e618d'
sys.path.insert(0, str(ROOT / 'scripts'))
import apa_pf2f1_build as pfb  # noqa: E402

EPSILON = 0.01                  # P1-D L1's registered epsilon, the design's primary
DELTA_ALL = 1e30                # every finite bulk score is >= running max - 1e30 (all-refined regime)
TAU_FACTOR, TAU_CAP = 4.0, 1e-5


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(data, f, indent=1, default=str)
        f.write('\n')


def step(log, label, **fields):
    rec = dict(t=time.time(), label=label, **fields)
    log.append(rec)
    print('STEP', json.dumps(rec, default=str), flush=True)


# ------------------------------------------------------------------ arms
def arm_dir(name):
    if name == 'F3':
        return F3_ARM
    if name in ('PF', 'PFH'):
        return ART / 'arms' / name
    return ART / 'mutants' / name.split(':', 1)[1] / 'arm'


def mutant_spec(m):
    return {x['name']: x for x in json.loads((ART / 'mutants/MUTANTS.json').read_text())['mutants']}[m]


def receipt(name):
    if name == 'F3':
        return dict(status='PASS', label='F3', module_sha256=F3_SHA)
    if name in ('PF', 'PFH'):
        return pfb.receipt(name)
    m = name.split(':', 1)[1]
    if mutant_spec(m)['kind'] == 'python':
        return pfb.receipt('PF')
    recs = [json.loads(p.read_text()) for p in sorted((ART / 'mutants' / m).glob('build*/BUILD.json'))]
    recs = [r for r in recs if r.get('status') == 'PASS']
    assert recs, f'no passing build receipt for mutant {m}'
    return max(recs, key=lambda r: r.get('attempt', 1))


def precheck(name):
    rec = receipt(name)
    arm = json.loads((arm_dir(name) / 'ARM.json').read_text())
    module = (F3_ARM.parent.parent.parent.parent if name == 'F3' else ROOT) / arm['module']
    got = sha(module)
    assert got == rec['module_sha256'] == arm['module_sha256'], f'{name}: module sha {got} != receipt'
    if name == 'F3':
        assert got == F3_SHA
    if name in ('PF', 'PFH'):
        now = pfb.f2b.f1b.engine_sources(ROOT)
        want = rec.get('source_sha256_after', rec['source_sha256'])
        drift = sorted(k for k in set(now) | set(want) if now.get(k) != want.get(k))
        assert not drift, f'{name}: working tree differs from the build receipt: {drift}'
        for rel, h in arm['python_files'].items():
            assert sha(ROOT / rel) == h, f'{name}: staged python file differs from the tree: {rel}'
    return dict(arm=name, module_sha256=got)


def stage(name):
    if name in ('PF', 'PFH'):
        rec = pfb.receipt(name)
        module, src = ROOT / rec['module'], ROOT / 'tensor_cuda/tensor_cuda'
    else:
        m = name.split(':', 1)[1]
        spec = mutant_spec(m)
        rec = receipt(name)
        module = ROOT / rec['module']
        src = (ART / 'mutants' / m / 'tree/tensor_cuda/tensor_cuda') if spec['kind'] == 'python' else \
            ROOT / 'tensor_cuda/tensor_cuda'
    assert rec['status'] == 'PASS' and sha(module) == rec['module_sha256']
    dest = arm_dir(name) / 'tensor_cuda'
    dest.mkdir(parents=True)
    files = {}
    for path in sorted(src.rglob('*.py')):
        if '__pycache__' in path.parts:
            continue
        out = dest / path.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, out)
        files['tensor_cuda/tensor_cuda/' + str(path.relative_to(src))] = sha(out)
    shutil.copyfile(module, dest / module.name)
    record = dict(arm=name, package=str(dest.relative_to(ROOT)), python_files=files, python_source=str(src),
                  module=str((dest / module.name).relative_to(ROOT)), module_sha256=sha(dest / module.name),
                  build_record=str(rec.get('label')))
    assert record['module_sha256'] == rec['module_sha256']
    (dest.parent / 'ARM.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))
    return 0


def load_arm(name):
    """Import the staged package of an arm (one arm per process) and check the loaded module's sha."""
    pkg_parent = arm_dir(name)
    sys.path.insert(0, str(pkg_parent))
    import tensor_cuda as tc
    assert Path(tc.__file__).resolve().parent == (pkg_parent / 'tensor_cuda').resolve(), tc.__file__
    got = sha(tc._C.__file__)
    assert got == receipt(name)['module_sha256'], (name, got)
    return tc, dict(arm=name, module_sha256=got, package=str(pkg_parent / 'tensor_cuda'))


# ------------------------------------------------------------------ G1 grid
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


def oracle_rows(c, idx):
    L, S = c['L'], c['S']
    if S <= 2049:
        return list(range(L))
    fixed = {0, 1, 255, 256, 257, 2047, 2048, 2049, L - 1}
    rng = np.random.default_rng(np.random.SeedSequence([20261116, 7, idx]))
    seeded = set(int(x) for x in rng.choice(L, size=min(64, L), replace=False))
    return sorted(x for x in fixed | seeded if 0 <= x < L)


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


def u32(torch, x):
    return x.contiguous().view(torch.int32)


def bits(torch, x):
    """Same-width integer view (L09): int32 for fp32, int16 for bf16."""
    return x.contiguous().view(torch.int32 if x.element_size() == 4 else torch.int16)


def informative_t(torch, x):
    u = x.contiguous().reshape(-1).view(torch.uint8 if x.element_size() == 1 else
                                        (torch.int16 if x.element_size() == 2 else torch.int32))
    return bool(u.numel()) and bool((u != 0).any()) and bool((u != u[0]).any())


def rel_rows(torch, a, b):
    """Per (b, h) row ||a - b|| / ||b|| in fp64; a, b [B, H, VD]."""
    a64, b64 = a.double(), b.double()
    num = torch.linalg.vector_norm(a64 - b64, dim=-1)
    den = torch.linalg.vector_norm(b64, dim=-1)
    return num, den


def run_case(E, c, idx, hooks, stats, tau_rows):
    torch = E.torch
    q, k, v, s = case_tensors(torch, c, idx)
    scale = float(np.float32(1 / math.sqrt(c['D'])))
    delta = E.tc.apa_packed.derive_delta(EPSILON, E.tc.apa_packed.PACKED_PINNED_ENTRIES['4:%d' % c['D']])
    ws, packed = E.store(k, c['cap'])
    S, L = c['S'], c['L']
    rec = dict(id=case_id(c), rows=0, decisions=0, mask_equal_rows=0, mask_unequal_rows=0, tail_zero_rows=0,
               bulk_compared=0, bulk_equal_rows=0, bulk_unequal_rows=0)
    big = c['B'] * c['G'] * 2 * L * S * 4 > (1 << 31)
    rows = oracle_rows(c, idx)
    if not hooks:
        out_a, mask_a = E.prefill(q, k, v, packed, scale, DELTA_ALL, c['causal'], s)[:2]
        ar = torch.arange(L, device='cuda', dtype=torch.int32)
        smax_vec = (S - L + ar + 1) if c['causal'] else torch.full((L,), S, device='cuda', dtype=torch.int32)
        rs_p = mask_a.sum(-1, dtype=torch.int32)
        all_ref_p = bool((rs_p == smax_vec).all()) and bool((mask_a.amax() <= 1))
        del mask_a
        out_d, mask_d = E.dense(q, k, v, scale, DELTA_ALL, c['causal'], s)
        rs_d = mask_d.sum(-1, dtype=torch.int32)
        all_ref_d = bool((rs_d == smax_vec).all()) and bool((mask_d.amax() <= 1))
        del mask_d
        # mask values are 0/1; a row sum equal to smax_i with keys [0, smax_i) visited in order and a zero tail
        # (prefill writes the tail explicitly; checked per row below for the prefill) means every visible key refined.
        bp, bd = bits(torch, out_a), bits(torch, out_d)
        uneq = int((bp != bd).sum())
        rec['pf_i2'] = dict(all_refined_prefill=all_ref_p, all_refined_dense=all_ref_d, elements=int(out_a.numel()),
                            equal=uneq == 0 and out_a.dtype == out_d.dtype and out_a.shape == out_d.shape,
                            unequal_elements=uneq, informative=[informative_t(torch, out_a), informative_t(torch, out_d)])
        if uneq:
            rec['pf_i2']['max_abs_diff'] = float((out_a.double() - out_d.double()).abs().max())
            num, den = rel_rows(torch, out_a.reshape(-1, out_a.shape[-1]), out_d.reshape(-1, out_d.shape[-1]))
            rec['pf_i2']['max_row_rel'] = float((num / den.clamp_min(1e-300)).max())
        del out_d, rs_p, rs_d, bp, bd
        if big:
            torch.cuda.empty_cache()
            E.tc.empty_cache()
    if not (hooks and big):
        out_p, mask_p, *rest = E.prefill(q, k, v, packed, scale, delta, c['causal'], s, dump=hooks)
        bulk_p = rest[0] if hooks else None
        rec['informative_prefill_mask'] = informative_t(torch, mask_p)
        rec['informative_prefill_out'] = informative_t(torch, out_p)
    i3_rel, i3_all_rel = [], []
    for i in rows:
        smax = S - L + i + 1 if c['causal'] else S
        q1 = q[:, :, i:i + 1].contiguous()
        k1, v1 = k[:, :, :smax].contiguous(), v[:, :, :smax].contiguous()
        if hooks and big:
            # Hooks build only, where the full [B,H,L,S] fp32 bulk dump exceeds 2 GiB: run row i alone as an L = 1
            # causal prefill on keys [0, smax_i) (same smax_i, same kernel arithmetic per row).
            o_r, m_r, b_r = E.prefill(q1, k1, v1, packed, scale, delta, True, s, dump=True)
            pm, pb = m_r[:, :, 0, :smax], b_r[:, :, 0, :smax]
            tail_ok = True
        else:
            pm = mask_p[:, :, i, :smax]
            tail_ok = bool((mask_p[:, :, i, smax:] == 0).all())
            pb = bulk_p[:, :, i, :smax] if hooks else None
        o_dec, m_dec, b_dec, _ = E.decode(q1, k1, v1, packed, scale, delta, s)
        eq = bool(torch.equal(pm, m_dec[:, :, 0, :smax]))
        rec['rows'] += 1
        rec['decisions'] += int(pm.numel())
        rec['mask_equal_rows' if eq else 'mask_unequal_rows'] += 1
        rec['tail_zero_rows'] += int(tail_ok)
        if hooks:
            beq = bool(torch.equal(u32(torch, pb), u32(torch, b_dec[:, :, 0, :smax])))
            rec['bulk_compared'] += int(pb.numel())
            rec['bulk_equal_rows' if beq else 'bulk_unequal_rows'] += 1
        else:
            num, den = rel_rows(torch, out_p[:, :, i].float(), o_dec[:, :, 0].float())
            i3_rel.append((num / den.clamp_min(1e-300)).max().item())
            o_all = E.decode(q1, k1, v1, packed, scale, DELTA_ALL, s)[0]
            num2, den2 = rel_rows(torch, out_a[:, :, i].float(), o_all[:, :, 0].float())
            i3_all_rel.append((num2 / den2.clamp_min(1e-300)).max().item())
    if not hooks:
        rec['pf_i3_max_rel'] = max(i3_rel) if i3_rel else None
        rec['pf_i3_all_refined_max_rel'] = max(i3_all_rel) if i3_all_rel else None
        tau_rows.append((rec['id'], i3_rel, i3_all_rel))
    for key in ('rows', 'decisions', 'mask_unequal_rows', 'bulk_unequal_rows'):
        stats[key] = stats.get(key, 0) + rec[key]
    del ws
    if big:
        torch.cuda.empty_cache()
        E.tc.empty_cache()
    return rec


def tie_cases():
    out = []
    for D in (64, 128):
        for G in (1, 2, 4, 8):
            for dt in ('float32', 'bfloat16'):
                for causal in (False, True):
                    out.append(dict(D=D, VD=D, G=G, dt=dt, B=1, S=257, L=33, causal=causal, sinks=False, cap=257,
                                    sub='tie'))
    return out


def tie_deltas(p, b):
    """fp32 deltas d >= 0 with fl(p - d) == b (eq), == nextafter(b, +inf) (above), == nextafter(b, -inf) (below)."""
    p32, b32 = np.float32(p), np.float32(b)
    targets = dict(eq=b32, above=np.nextafter(b32, np.float32(np.inf)), below=np.nextafter(b32, np.float32(-np.inf)))
    found = {}
    for name, tgt in targets.items():
        d = np.float32(np.float64(p32) - np.float64(tgt))
        for _ in range(64):
            got = np.float32(p32 - d)
            if got == tgt:
                found[name] = float(d)
                break
            d = np.nextafter(d, np.float32(np.inf) if got > tgt else np.float32(-np.inf))
    return found if len(found) == 3 and all(x >= 0 for x in found.values()) else None


def run_tie(E, c, idx):
    torch = E.torch
    q, k, v, s = case_tensors(torch, c, 10_000 + idx)
    scale = float(np.float32(1 / math.sqrt(c['D'])))
    delta0 = E.tc.apa_packed.derive_delta(EPSILON, E.tc.apa_packed.PACKED_PINNED_ENTRIES['4:%d' % c['D']])
    ws, packed = E.store(k, c['cap'])
    S, L = c['S'], c['L']
    i = L - 1
    smax = S - L + i + 1 if c['causal'] else S
    q1, k1, v1 = q[:, :, i:i + 1].contiguous(), k[:, :, :smax].contiguous(), v[:, :, :smax].contiguous()
    _, _, bulk, prefix = E.decode(q1, k1, v1, packed, scale, delta0, s)
    b = bulk[0, 0, 0, :smax].cpu().numpy()
    p = prefix[0, 0, 0, :smax].cpu().numpy()
    # A key with a gap to its running maximum between 1 and 10 (a threshold that is neither trivial nor all-refine).
    rec = dict(id=case_id(c), row=i, head=0)
    for j in range(1, smax):
        if 1.0 < float(p[j]) - float(b[j]) < 10.0 and b[j] < p[j]:
            dl = tie_deltas(p[j], b[j])
            if dl:
                rec.update(key=int(j), bulk_bits='%08x' % int(np.float32(b[j]).view(np.uint32)),
                           prefix_bits='%08x' % int(np.float32(p[j]).view(np.uint32)), deltas=dl)
                break
    if 'key' not in rec:
        rec['error'] = 'no tie key found'
        return rec, False
    ok = True
    rec['runs'] = {}
    expect = dict(eq=1, below=1, above=0)
    for name, d in rec['deltas'].items():
        out_p, mask_p = E.prefill(q, k, v, packed, scale, d, c['causal'], s)[:2]
        o_dec, m_dec, b_dec, pre_dec = E.decode(q1, k1, v1, packed, scale, d, s)
        thr = np.float32(np.float32(pre_dec[0, 0, 0, rec['key']].item()) - np.float32(d))
        eq = bool(torch.equal(mask_p[:, :, i, :smax], m_dec[:, :, 0, :smax]))
        dec_bit = int(m_dec[0, 0, 0, rec['key']].item())
        pf_bit = int(mask_p[0, 0, i, rec['key']].item())
        run_ok = eq and dec_bit == expect[name] and pf_bit == expect[name]
        rec['runs'][name] = dict(delta=d, threshold_bits='%08x' % int(thr.view(np.uint32)), mask_equal=eq,
                                 decode_bit=dec_bit, prefill_bit=pf_bit, expected_bit=expect[name], ok=run_ok)
        ok = ok and run_ok
    rec['ok'] = ok
    del ws
    return rec, ok


def prefix_store_check(E):
    """The oracle's store-prefix premise: a store filled with only smax keys has the same codes and norms as the
    first smax rows of a larger store (packed quantizer is per row)."""
    torch = E.torch
    rows = []
    for D in (64, 128):
        for dt in ('float32', 'bfloat16'):
            c = dict(D=D, VD=D, G=2, dt=dt, B=2, S=2049, L=2, causal=True, sinks=False, cap=2049, sub='prefix')
            _, k, _, _ = case_tensors(torch, c, 20_000 + D)
            _, full = E.store(k, 2049 + 37)
            for smax in (1, 255, 1000, 2048):
                _, part = E.store(k[:, :, :smax].contiguous(), smax)
                cf, sf = E.tt(full[0])[:, :, :smax], E.tt(full[1])[:, :, :smax]
                cp, sp = E.tt(part[0]), E.tt(part[1])
                rows.append(dict(D=D, dt=dt, smax=smax, codes_equal=bool(torch.equal(cf, cp)),
                                 norms_equal=bool(torch.equal(u32(torch, sf), u32(torch, sp)))))
    return rows, all(r['codes_equal'] and r['norms_equal'] for r in rows)


# ------------------------------------------------------------------ PF-I1n
def nonfinite_cases():
    out = []
    for D in (64, 128):
        for G in (1, 4):
            for dt in ('float32', 'bfloat16'):
                for causal in (False, True):
                    for L in (2, 33):
                        for kind in ('nan_k', 'inf_k', 'ninf_k', 'ovf_k', 'nan_q', 'nan_v'):
                            out.append(dict(D=D, VD=D, G=G, dt=dt, B=1, S=257, L=L, causal=causal, sinks=False,
                                            cap=257, sub='nf_' + kind, kind=kind))
    return out


def run_nonfinite(E, c, idx, dm):
    torch = E.torch
    q, k, v, s = case_tensors(torch, c, 30_000 + idx)
    j_bad = 200                      # seen by every row when non-causal; by rows with smax > 200 when causal
    kind = c['kind']
    if kind == 'nan_k':
        k[0, 0, j_bad, 3] = float('nan')
    elif kind == 'inf_k':
        k[0, 0, j_bad, 3] = float('inf')
    elif kind == 'ninf_k':
        k[0, 1, j_bad, 5] = float('-inf')
    elif kind == 'ovf_k':
        k[0, 0, j_bad, 7] = 1e20      # finite element, sum of squares overflows fp32 (F3 R2')
    elif kind == 'nan_q':
        q[0, 1, c['L'] - 1, 2] = float('nan')
    elif kind == 'nan_v':
        v[0, 1, j_bad, 1] = float('nan')
    scale = float(np.float32(1 / math.sqrt(c['D'])))
    delta = E.tc.apa_packed.derive_delta(EPSILON, E.tc.apa_packed.PACKED_PINNED_ENTRIES['4:%d' % c['D']])
    ws, packed = E.store(k, c['cap'])
    out_p, mask_p = E.prefill(q, k, v, packed, scale, delta, c['causal'], s)[:2]
    S, L = c['S'], c['L']
    mask_eq = 0
    for i in range(L):
        smax = S - L + i + 1 if c['causal'] else S
        q1, k1, v1 = q[:, :, i:i + 1].contiguous(), k[:, :, :smax].contiguous(), v[:, :, :smax].contiguous()
        m_dec = E.decode(q1, k1, v1, packed, scale, delta, s)[1]
        mask_eq += int(torch.equal(mask_p[:, :, i, :smax], m_dec[:, :, 0, :smax]))
    # Reference: dense SP prefill on the dense quantizer's keys (public dense route; fp32 only on the public route,
    # so bf16 cases use the raw dense entry with the dense quantizer's kq cast to bf16, as the dense route stores it).
    norms = E.tt(packed[1])
    stored_norm_nan = None
    if kind in ('nan_k', 'inf_k', 'ninf_k', 'ovf_k'):
        kv_bad = 1 if kind == 'ninf_k' else 0
        stored_norm_nan = bool(torch.isnan(norms[0, kv_bad, j_bad]))
    k32 = E.to_tc(k.float())
    kd = E.tc.quant._quantize_keys(k32, *E.tc.quant._tables(c['D'], 4, k.shape[1], True, 'cuda'))
    kd_t = E.tt(kd).to(k.dtype).contiguous()
    e_q = dm.lookup(4, c['D'], 'float32', scale)
    delta_d = E.tc._C.apa_sp2_delta(EPSILON, e_q)
    out_d = E.tt(E.tc._C.apa_selective_attention_sp(E.to_tc(q), E.to_tc(k), E.to_tc(kd_t), E.to_tc(v), scale, delta_d,
                                                     c['causal'], E.to_tc(s), False))
    bad_p = ~torch.isfinite(out_p.float()).all(-1)
    bad_d = ~torch.isfinite(out_d.float()).all(-1)
    finite_where_ref_nonfinite = int((bad_d & ~bad_p).sum())
    rec = dict(id=case_id(c), kind=kind, rows=int(bad_p.numel()), nonfinite_rows_packed=int(bad_p.sum()),
               nonfinite_rows_dense=int(bad_d.sum()), row_sets_equal=bool(torch.equal(bad_p, bad_d)),
               finite_where_dense_nonfinite=finite_where_ref_nonfinite, mask_equal_rows=mask_eq, rows_checked=L,
               stored_norm_nan=stored_norm_nan)
    ok = finite_where_ref_nonfinite == 0 and mask_eq == L and rec['row_sets_equal'] and \
        (stored_norm_nan is None or stored_norm_nan) and rec['nonfinite_rows_packed'] > 0
    rec['ok'] = ok
    del ws
    return rec, ok


def g1(args):
    """One G1 line part: grid cases from --start until done or --max-seconds is reached (resumable; each part
    writes part_<start>.json), then, in the part that reaches the end of the grid, the extras (prefix store, tie
    rows, PF-I1n on Build PF)."""
    import torch
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    hooks = args.arm == 'PFH'
    tc, arm = load_arm(args.arm)
    os.environ['TC_APA_SP'] = '1'
    os.environ.pop('TC_APA_SPQ1_REFERENCE', None)
    E = Env(tc, torch)
    log = []
    step(log, 'start', arm=arm, device=torch.cuda.get_device_name(0), mem=torch.cuda.mem_get_info())
    cases = g1_cases()
    start = int(args.start or 0)
    stats, tau_rows, recs = {}, [], []
    t0 = time.time()
    nxt = start
    with tc.no_grad():
        for idx in range(start, len(cases)):
            if time.time() - t0 > args.max_seconds:
                break
            try:
                recs.append(run_case(E, cases[idx], idx, hooks, stats, tau_rows))
            except BaseException as exc:
                save(out / f'part_{start:05d}_stopped.json', dict(start=start, next=idx, failed_case=case_id(cases[idx]),
                     error=repr(exc), stats=stats, cases=recs, tau_rows=tau_rows, steps=log))
                raise
            nxt = idx + 1
            if (idx - start) % 100 == 0:
                torch.cuda.synchronize()
                step(log, 'progress', idx=idx, of=len(cases), seconds=round(time.time() - t0, 1), stats=stats)
        extras = {}
        if nxt == len(cases) and time.time() - t0 < args.max_seconds:
            extras['prefix_store'], extras['prefix_store_ok'] = prefix_store_check(E)
            extras['ties'] = [run_tie(E, c, i)[0] for i, c in enumerate(tie_cases())]
            step(log, 'ties', ok=all(r.get('ok') for r in extras['ties']))
            if not hooks:
                dm = tc.RegisteredMargins.load(DENSE_TABLE, expected_sha256=DENSE_TABLE_SHA)
                extras['nonfinite'] = [run_nonfinite(E, c, i, dm)[0] for i, c in enumerate(nonfinite_cases())]
                step(log, 'nonfinite', ok=all(r['ok'] for r in extras['nonfinite']))
    torch.cuda.synchronize()
    part = dict(line='G1H' if hooks else 'G1', arm=arm, start=start, next=nxt, total=len(cases),
                seconds=time.time() - t0, registration_sha256=sha(ART / 'REGISTRATION.json'), stats=stats,
                cases=recs, tau_rows=tau_rows, extras=extras, steps=log)
    save(out / f'part_{start:05d}.json', part)
    print('G1 PART', start, '->', nxt, 'of', len(cases), json.dumps(stats), 'extras', sorted(extras), flush=True)
    return 0


def g1_merge(args):
    """CPU: merge the parts of a G1/G1H line into RESULT.json (every case exactly once, extras present)."""
    out = Path(args.out)
    parts = [json.loads(p.read_text()) for p in sorted(out.glob('part_*.json'))]
    hooks = parts[0]['line'] == 'G1H'
    recs = [r for pt in parts for r in pt['cases']]
    ids = [r['id'] for r in recs]
    want = [case_id(c) for c in g1_cases()]
    extras = {}
    for pt in parts:
        extras.update(pt['extras'])
    reg = json.loads((ART / 'REGISTRATION.json').read_text())
    res = dict(line='G1H' if hooks else 'G1', arms=sorted({json.dumps(pt['arm'], sort_keys=True) for pt in parts}),
               parts=[(pt['start'], pt['next'], round(pt['seconds'], 1)) for pt in parts],
               registration_sha256=sha(ART / 'REGISTRATION.json'),
               complete=sorted(ids) == sorted(want) and len(ids) == len(set(ids)), cases=len(recs))
    rows = sum(r['rows'] for r in recs)
    res['pf_i1'] = dict(rows=rows, decisions=sum(r['decisions'] for r in recs),
                        mask_unequal_rows=sum(r['mask_unequal_rows'] for r in recs),
                        tail_nonzero_rows=sum(r['rows'] - r['tail_zero_rows'] for r in recs),
                        uninformative_prefill_masks=[r['id'] for r in recs if r.get('informative_prefill_mask') is False],
                        cases_with_unequal_rows=[r['id'] for r in recs if r['mask_unequal_rows']][:40],
                        ties=len(extras.get('ties', [])), ties_ok=sum(bool(t.get('ok')) for t in extras.get('ties', [])),
                        prefix_store_ok=extras.get('prefix_store_ok'))
    by_sub = {}
    for r in recs:
        key = r['id'].split('_D')[0] + '_D' + r['id'].split('_D')[1].split('_')[0] + '_' + \
            [x for x in r['id'].split('_') if x in ('float32', 'bfloat16')][0]
        b = by_sub.setdefault(key, dict(cases=0, rows=0, decisions=0, mask_unequal_rows=0, bulk_values=0,
                                        bulk_unequal_rows=0, pf_i2_equal=0, pf_i2_elements=0))
        b['cases'] += 1
        b['rows'] += r['rows']
        b['decisions'] += r['decisions']
        b['mask_unequal_rows'] += r['mask_unequal_rows']
        b['bulk_values'] += r['bulk_compared']
        b['bulk_unequal_rows'] += r['bulk_unequal_rows']
        if 'pf_i2' in r:
            b['pf_i2_equal'] += int(r['pf_i2']['equal'])
            b['pf_i2_elements'] += r['pf_i2']['elements']
    res['per_class'] = by_sub
    ok = res['complete'] and res['pf_i1']['mask_unequal_rows'] == 0 and res['pf_i1']['tail_nonzero_rows'] == 0 and \
        not res['pf_i1']['uninformative_prefill_masks'] and res['pf_i1']['ties'] == 32 and \
        res['pf_i1']['ties_ok'] == 32 and res['pf_i1']['prefix_store_ok'] is True
    if hooks:
        res['pf_i1_bulk'] = dict(values=sum(r['bulk_compared'] for r in recs),
                                 unequal_rows=sum(r['bulk_unequal_rows'] for r in recs))
        ok = ok and res['pf_i1_bulk']['unequal_rows'] == 0 and res['pf_i1_bulk']['values'] > 0
        res['status'] = dict(pf_i1_masks_and_bulk='PASS' if ok else 'FAIL')
    else:
        i2 = [r['pf_i2'] for r in recs]
        res['pf_i2'] = dict(cases=len(i2), equal=sum(x['equal'] for x in i2), elements=sum(x['elements'] for x in i2),
                            unequal_elements=sum(x['unequal_elements'] for x in i2),
                            not_all_refined=[r['id'] for r in recs if not (r['pf_i2']['all_refined_prefill'] and
                                                                             r['pf_i2']['all_refined_dense'])],
                            uninformative=[r['id'] for r in recs if r['pf_i2']['informative'] != [True, True]],
                            unequal_cases=[dict(id=r['id'], **{k: r['pf_i2'].get(k) for k in
                                                ('unequal_elements', 'max_abs_diff', 'max_row_rel')})
                                           for r in recs if not r['pf_i2']['equal']][:60])
        tau_rows = [t for pt in parts for t in pt['tau_rows']]
        all_ref = [x for _, _, xs in tau_rows for x in xs]
        at_delta = [x for _, xs, _ in tau_rows for x in xs]
        tau_max = max(all_ref)
        tau = min(TAU_FACTOR * tau_max, TAU_CAP)
        res['pf_i3'] = dict(rule=reg['pf_i3_tau_rule'], all_refined_max=tau_max, tau=tau, rows=len(at_delta),
                            max_at_delta=max(at_delta), rows_above_tau=sum(x > tau for x in at_delta),
                            cases_above_tau=[i for i, xs, _ in tau_rows if any(x > tau for x in xs)][:40])
        nf = extras.get('nonfinite', [])
        res['pf_i1n'] = dict(cases=len(nf), ok=sum(r['ok'] for r in nf),
                             finite_where_dense_nonfinite=sum(r['finite_where_dense_nonfinite'] for r in nf),
                             row_sets_equal=sum(r['row_sets_equal'] for r in nf),
                             failing=[r for r in nf if not r['ok']][:20])
        i2ok = res['pf_i2']['equal'] == len(i2) and not res['pf_i2']['not_all_refined'] and \
            not res['pf_i2']['uninformative']
        res['status'] = dict(pf_i1='PASS' if ok else 'FAIL', pf_i2='PASS' if i2ok else 'FAIL',
                             pf_i3='PASS' if res['pf_i3']['rows_above_tau'] == 0 else 'FAIL',
                             pf_i1n='PASS' if nf and res['pf_i1n']['ok'] == len(nf) == 192 else 'FAIL')
    res['extras'] = extras
    save(out / 'RESULT.json', res)
    print(json.dumps({k: v for k, v in res.items() if k != 'extras'}, default=str)[:6000])
    return 0


MODES = {'g1': g1, 'g1h': g1}
CPU_MODES = {'g1-merge': g1_merge}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode')
    ap.add_argument('target', nargs='?')
    ap.add_argument('--out')
    ap.add_argument('--arm')
    ap.add_argument('--start')
    ap.add_argument('--max-seconds', type=float, default=1500)
    args, rest = ap.parse_known_args()
    args.rest = rest
    if args.mode == 'stage':
        return stage(args.target)
    if args.mode == 'precheck':
        print(json.dumps(precheck(args.target)))
        return 0
    if args.mode == 'cases':
        print(len(g1_cases()), len(tie_cases()), len(nonfinite_cases()))
        return 0
    if args.mode in CPU_MODES:
        return CPU_MODES[args.mode](args)
    assert os.environ.get('CUDA_VISIBLE_DEVICES') != '', 'GPU modes run inside forge-gpu-run'
    try:
        code = MODES[args.mode](args)
    except BaseException:
        if args.out:
            Path(args.out).mkdir(parents=True, exist_ok=True)
            (Path(args.out) / f'exception_{args.mode}_{time.time_ns()}.txt').write_text(traceback.format_exc())
        raise
    return code


if __name__ == '__main__':
    raise SystemExit(main())
