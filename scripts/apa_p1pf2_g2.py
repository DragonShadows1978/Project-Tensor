#!/usr/bin/env python3
"""APA-P1-PF2 G2 (PF-A1, PF-A2, PF-I4) on the four SPQ3 layers, Build PF, inside forge-gpu-run.

  python3 -B -P scripts/apa_p1pf2_g2.py --out <dir> [--units <model:layer,...>]

Per unit (REGISTRATION.json g2): q, k, v from the SPQ3 captures (bf16 bit patterns widened exactly to fp32), S = L =
8192, causal, B 1, no sinks, epsilon 0.01. Arms, all through the public entry except the exact reference:
  dense  : quantize_sp2_keys(k, 4, storage='dense'), dense table, diagnostics (mask)
  packed : quantize_sp2_keys(k, 4) (shipped default, packed), pinned packed table, diagnostics (mask); route record
           must be the prefill backend with the pinned packed delta
  exact  : _C.apa_selective_attention_sp(q, k, k, v, scale, 1e30, True, None, False) (every key scored from k)
Flips are counted on the GPU over causal pairs (the masks' tails are 0 in both arms). Row norms in fp64.
Prior art: SPQ3 captures and the P1-PF design's M2/M3 measurement definitions (Project-Tensor 2026), restated on the
device; pooled norms as sqrt of summed squares (the design's merge). No new technique.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import apa_p1pf2_gpu as G  # noqa: E402

CAP = ROOT.parent / 'tensor_fork_apa_wt_spq3/artifacts/apa_spq3/captures'
UNITS = [('Qwen3-1.7B', 11), ('Qwen3-1.7B', 22), ('Qwen3-4B', 14), ('Qwen3-4B', 28)]
EMULATED_FLIPS = {'Qwen3-1.7B:11': 63, 'Qwen3-1.7B:22': 22, 'Qwen3-4B:14': 91, 'Qwen3-4B:28': 165}
FLIP_BAR, RATIO_BAR, ERR_BAR, ROW_FRAC_BAR, FLOOR = 1e-6, 0.02, 1.01, 1e-4, 1e-4


def load_bf16(path):
    bits = np.load(path)
    assert bits.dtype == np.uint16
    return (bits.astype(np.uint32) << np.uint32(16)).view(np.float32)


def unit(E, model, layer, dm, pm, log):
    torch, tc = E.torch, E.tc
    d = CAP / model
    q = load_bf16(d / f'L{layer:02d}_q.npy')
    k = load_bf16(d / f'L{layer:02d}_k.npy')
    v = load_bf16(d / f'L{layer:02d}_v.npy')
    H, S, D = q.shape
    KV = k.shape[0]
    files = {n: G.sha(d / f'L{layer:02d}_{n}.npy') for n in 'qkv'}
    qt = torch.from_numpy(q[None]).cuda().contiguous()
    kt = torch.from_numpy(k[None]).cuda().contiguous()
    vt = torch.from_numpy(v[None]).cuda().contiguous()
    Q, K, V = E.to_tc(qt), E.to_tc(kt), E.to_tc(vt)
    scale = float(np.float32(1 / math.sqrt(D)))
    t0 = time.time()
    kd = tc.quantize_sp2_keys(K, 4, storage='dense')
    out_d, mask_d = tc.apa_selective_attention_sp(Q, K, kd, V, scale, G.EPSILON, bulk_bits=4, margins=dm,
                                                  is_causal=True, diagnostics=True)
    route_d = tc.apa_packed.last_route('attention')
    out_d, mask_d = E.tt(out_d), E.tt(mask_d)
    del kd
    t1 = time.time()
    kp = tc.quantize_sp2_keys(K, 4)
    route_q = tc.apa_packed.last_route('quantize')
    assert isinstance(kp, tc.PackedSpKeys)
    out_p, mask_p = tc.apa_selective_attention_sp(Q, K, kp, V, scale, G.EPSILON, bulk_bits=4, margins=pm,
                                                  is_causal=True, diagnostics=True)
    route_p = tc.apa_packed.last_route('attention')
    out_p, mask_p = E.tt(out_p), E.tt(mask_p)
    t2 = time.time()
    out_x = E.tt(tc._C.apa_selective_attention_sp(Q, K, K, V, scale, G.DELTA_ALL, True, None, False))
    torch.cuda.synchronize()
    t3 = time.time()
    flips = d_only = p_only = 0
    refined_d = refined_p = 0
    for h in range(H):
        a, b = mask_d[0, h], mask_p[0, h]
        flips += int((a != b).sum())
        d_only += int(((a == 1) & (b == 0)).sum())
        p_only += int(((a == 0) & (b == 1)).sum())
        refined_d += int(a.sum(dtype=torch.int64))
        refined_p += int(b.sum(dtype=torch.int64))
    # Amendment 2: upper-triangle (j > i) mask entries counted head by head (peak memory one head's mask).
    tails = sum(int(torch.triu(m[0, h], 1).sum(dtype=torch.int64)) for m in (mask_d, mask_p) for h in range(H))
    N = H * S * (S + 1) // 2
    xd, pd_, xx = out_d[0].double(), out_p[0].double(), out_x[0].double()
    e_sp = torch.linalg.vector_norm(xd - xx, dim=-1).reshape(-1)
    dd = torch.linalg.vector_norm(pd_ - xd, dim=-1).reshape(-1)
    e_a = torch.linalg.vector_norm(pd_ - xx, dim=-1).reshape(-1)
    nx = torch.linalg.vector_norm(xx, dim=-1).reshape(-1)
    pool = lambda t: float(torch.sqrt((t * t).sum()))  # noqa: E731
    rows = int(e_sp.numel())
    big = e_sp >= FLOOR * nx
    rows_d_gt_e = int(((dd > e_sp) & big).sum())
    floor_rows = int((~big).sum())
    floor_viol = int(((dd > FLOOR * nx) & ~big).sum())
    ratio = (dd / e_sp.clamp_min(1e-300))[big]
    rec = dict(unit=f'{model}:{layer}', files_sha256=files, H=H, KV=KV, S=S, D=D, scale=scale,
               routes=dict(dense=route_d, packed=route_p, packed_quantize=route_q),
               seconds=dict(dense=t1 - t0, packed=t2 - t1, exact=t3 - t2),
               pf_a1=dict(decisions=N, flips=flips, dense_only=d_only, packed_only=p_only, rate=flips / N,
                          bar=FLIP_BAR, max_flips=int(FLIP_BAR * N), refined_dense=refined_d, refined_packed=refined_p,
                          refined_fraction_dense=refined_d / N, refined_fraction_packed=refined_p / N,
                          upper_triangle_mask_entries=tails, pass_=flips <= FLIP_BAR * N),
               pf_a2=dict(rows=rows, pooled_e_sp=pool(e_sp), pooled_d=pool(dd), pooled_e_packed=pool(e_a),
                          pooled_exact_norm=pool(nx), pooled_rel_e_sp=pool(e_sp) / pool(nx),
                          ratio_d_over_e_sp=pool(dd) / pool(e_sp), ratio_e_packed_over_e_sp=pool(e_a) / pool(e_sp),
                          rows_above_floor=int(big.sum()), rows_d_gt_e_sp=rows_d_gt_e,
                          max_rows_d_gt_e_sp=int(ROW_FRAC_BAR * rows), floor_rows=floor_rows,
                          floor_rows_d_above_floor=floor_viol,
                          row_ratio_median=float(ratio.median()) if ratio.numel() else None,
                          row_ratio_p99=float(torch.quantile(ratio.float(), 0.99)) if ratio.numel() else None,
                          row_ratio_max=float(ratio.max()) if ratio.numel() else None,
                          rows_d_zero=int((dd == 0).sum())),
               pf_i4=dict(device_flips=flips, emulated_flips=EMULATED_FLIPS[f'{model}:{layer}']))
    a2 = rec['pf_a2']
    a2['pass_'] = (a2['ratio_d_over_e_sp'] <= RATIO_BAR and a2['ratio_e_packed_over_e_sp'] <= ERR_BAR and
                   rows_d_gt_e <= ROW_FRAC_BAR * rows and floor_viol == 0)
    rec['route_ok'] = (route_p['backend'] == 'spq_packed_prefill_tq4_row' and route_p['storage'] == 'packed' and
                       route_p['delta'] == tc.apa_packed.derive_delta(G.EPSILON, tc.apa_packed.PACKED_PINNED_ENTRIES['4:%d' % D])
                       and route_d['storage'] == 'dense' and route_q['storage'] == 'packed')
    rec['informative'] = [G.informative_t(torch, x) for x in (out_d, out_p, out_x, mask_d, mask_p)]
    G.step(log, 'unit', unit=rec['unit'], flips=flips, ratio=a2['ratio_d_over_e_sp'], rows_d_gt_e=rows_d_gt_e,
           seconds=rec['seconds'])
    del out_d, out_p, out_x, mask_d, mask_p, kp
    torch.cuda.empty_cache()
    return rec


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--units')
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tc, arm = G.load_arm('PF')
    os.environ['TC_APA_SP'] = '1'
    os.environ.pop('TC_APA_SP_STORAGE', None)
    E = G.Env(tc, torch)
    dm = tc.RegisteredMargins.load(G.DENSE_TABLE, expected_sha256=G.DENSE_TABLE_SHA)
    pm = tc.PackedRegisteredMargins.load(G.PACKED_TABLE)
    log = []
    G.step(log, 'start', arm=arm, device=torch.cuda.get_device_name(0), mem=torch.cuda.mem_get_info())
    units = UNITS if not args.units else [(u.split(':')[0], int(u.split(':')[1])) for u in args.units.split(',')]
    recs = []
    with tc.no_grad():
        for model, layer in units:
            path = out / f'{model}_L{layer:02d}.json'
            if path.exists():
                recs.append(json.loads(path.read_text()))
                continue
            r = unit(E, model, layer, dm, pm, log)
            G.save(path, r)
            recs.append(r)
    done = len(recs) == 4
    status = dict(pf_a1='PASS' if done and all(r['pf_a1']['pass_'] for r in recs) else 'FAIL',
                  pf_a2='PASS' if done and all(r['pf_a2']['pass_'] for r in recs) else 'FAIL',
                  routes='PASS' if all(r['route_ok'] for r in recs) else 'FAIL',
                  informative='PASS' if all(all(r['informative']) for r in recs) else 'FAIL')
    G.save(out / 'RESULT.json', dict(line='G2', arm=arm, complete=done, status=status, units=recs, steps=log,
                                     registration_sha256=G.sha(G.ART / 'REGISTRATION.json'), terminal='G2_DONE'))
    print('G2', status, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
