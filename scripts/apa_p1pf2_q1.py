#!/usr/bin/env python3
"""APA-P1-PF2 G3 (PF-Q1) through P1-Q's harness (imported unchanged), Build PF, inside forge-gpu-run.

  job --model M --out D [--windows 0,4,...] [--max-seconds N]   (GPU) windows not yet done, one process
  report --out D                                                (CPU) the per-model, per-population table + gates

What is reused unchanged from tensor_fork_apa_wt_p1q/scripts (read-only): apa_p1q_harness (Runner, Bridge, Ctx,
SPLayer and its L = 1 decode path, FLASH path, chunked prefill), apa_p1q_common (windows, metrics, bootstrap_ci,
bitwise views), apa_p1q_tuning (constants). What this round adds (registered, REGISTRATION.json g3):
- PFLayer(SPLayer): on layers >= 1 of the SP arms, an L > 1 chunk goes through the public SP route on the cache
  prefix (q, k, v exact fp32 upcasts of the bf16 cache, quantize_sp2_keys(k, 4, storage=<'dense' | default>),
  apa_selective_attention_sp(..., epsilon 0.01, is_causal=True)), i.e. dense SP prefill or packed prefill; L = 1 is
  SPLayer's own decode path. The harness's make_layer is pointed at a function that returns PFLayer for SP layers.
- pf_pass: decode_pass (harness) restated with an on_chunk hook that also keeps the logits of the last 1,024
  prefill positions (population i); population ii is the 256 teacher-forced decode steps.
Reference arm: FLASH (P1-Q's exact-attention reference). Arms: SP-DENSE (storage 'dense'), SP-PACKED (default).
Prior art: P1-Q / AB-1 harness, windows and metrics (Project-Tensor 2026); KL (Kullback and Leibler, 1951); paired
percentile bootstrap over windows (Efron, 1979); BLASST (arXiv 2512.12087) and TurboQuant (arXiv 2504.19874) inside
the engine. Ours: the prefill wiring and the prefill-logit population. No new statistic.
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
P1Q = ROOT.parent / 'tensor_fork_apa_wt_p1q/scripts'
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.append(str(P1Q))
import apa_p1pf2_gpu as G  # noqa: E402

ARMS = ('FLASH', 'SP-DENSE', 'SP-PACKED')
CTX, S0, N, POP1 = 8192, 7936, 256, 1024
PREFILL_BACKEND = 'spq_packed_prefill_tq4_row'
KL_FACTOR, KL_ABS, TOP1_SLACK, DNLL_MAX, DNLL_CI_MAX = 1.10, 2e-5, 0.003, 0.002, 0.005


def setup():
    import torch
    import apa_p1q_tuning as PT
    import apa_p1q_common as C
    import apa_p1q_harness as H
    tc, arm = G.load_arm('PF')
    H.tc = tc
    H.register()   # Amendment 2: P1-Q's 'p1q' attention hook, as P1-Q's main() registers it
    os.environ['TC_APA_SP'] = '1'
    os.environ.pop('TC_APA_SP_STORAGE', None)
    orig_make = H.make_layer
    TALLY = {}

    class PFLayer(H.SPLayer):
        def forward(self, q, k, v, p0):
            L = q.shape[2]
            if L == 1:
                return super().forward(q, k, v, p0)
            n = p0 + L
            self.write(k, v, p0, n)
            c = self.c
            k32 = self.K[:, :, :n].float().contiguous()
            v32 = self.V[:, :, :n].float().contiguous()
            q32 = q.float().contiguous()
            kt, vt, qt = H.to_tc(k32), H.to_tc(v32), H.to_tc(q32)
            H.clear_routes()
            kq = tc.quantize_sp2_keys(kt, PT.BULK_BITS, storage=c.storage)
            rq = tc.apa_packed.last_route('quantize')
            out = tc.apa_selective_attention_sp(qt, kt, kq, vt, c.scale, c.eps, bulk_bits=PT.BULK_BITS,
                                                margins=c.table, is_causal=True)
            ra = tc.apa_packed.last_route('attention')
            if c.storage == 'dense':
                ok = (ra['storage'] == 'dense' and ra['backend'] == tc.apa_packed.DENSE_BACKEND and ra['l'] == L and
                      rq['storage'] == 'dense')
            else:
                ok = (ra['storage'] == 'packed' and ra['backend'] == PREFILL_BACKEND and ra['l'] == L and
                      ra['delta'] == c.expected_delta and rq['storage'] == 'packed' and rq['source'] == 'default')
            t = TALLY.setdefault(c.arm, dict(calls=0, ok=0, bad=[]))
            t['calls'] += 1
            t['ok'] += int(ok)
            if not ok and len(t['bad']) < 4:
                t['bad'].append(dict(layer=self.li, attention=ra, quantize=rq))
            o = H.to_torch(out)
            assert o.dtype == torch.float32 and tuple(o.shape) == (1, c.H, L, c.D)
            return o.to(torch.bfloat16)

    def make_layer(c, li):
        if c.arm in PT.SP_ARMS and not (li == 0 and PT.LAYER0_EXACT):
            return PFLayer(c, li, 'SP')
        return orig_make(c, li)
    H.make_layer = make_layer
    return torch, tc, arm, PT, C, H, TALLY


def pf_pass(torch, H, R, arm, ids):
    """decode_pass (P1-Q harness) restated; also keeps the logits of prefill positions S0-1024 .. S0-1."""
    g = H.gpu_ids(ids)
    t0 = time.time()
    logits = torch.empty((N, R.V), dtype=torch.bfloat16, device='cuda')
    pre = torch.empty((POP1, R.V), dtype=torch.bfloat16, device='cuda')
    a0 = S0 - POP1

    def on_chunk(p0, L, h):
        a, b = max(p0, a0), p0 + L
        for x in range(a, b, 256):          # 256-row blocks of the lm_head (memory only)
            y = min(b, x + 256)
            pre[x - a0:y - a0] = R.logits(h[:, x - p0:y - p0])[0]
    with R.session(arm, S0 + N, N) as br:
        c = br.c
        pos = R.prefill(br, g, S0, on_chunk=on_chunk)
        t1 = time.time()
        for j in range(N):
            c.j = j
            h = R.forward(br, g, S0 + j, 1, pos)
            logits[j] = R.logits(h[:, -1])[0]
        c.j = -1
        torch.cuda.synchronize()
        rec = dict(arm=arm, wall_s=time.time() - t0, prefill_s=t1 - t0, decode_s=time.time() - t1,
                   eps=c.eps if arm != 'FLASH' else None, storage_argument=c.storage if arm != 'FLASH' else None)
        if arm != 'FLASH':
            rec['decode_routes'] = c.summary(N)
    return rec, pre, logits


def job(args):
    torch, tc, arm, PT, C, H, TALLY = setup()
    out = Path(args.out) / args.model
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    R = H.Runner(args.model)
    log = []
    G.step(log, 'loaded', model=args.model, load_s=R.load_s, arm=arm, mem=torch.cuda.mem_get_info())
    wins = [int(x) for x in args.windows.split(',')] if args.windows else list(PT.AB1_WIN_INDICES)
    per_window = []
    for wid in wins:
        jp = out / f'w{wid}.json'
        if jp.exists():
            continue
        if per_window and time.time() - t_start + max(per_window) * 1.15 > args.max_seconds:
            G.step(log, 'budget_stop', before_window=wid)
            break
        tw = time.time()
        ids = C.window_tokens(args.model, wid, CTX)
        assert len(ids) == CTX + 1
        tg = H.gpu_ids(ids)[0]
        tgt_pre = tg[S0 - POP1 + 1:S0 + 1]
        tgt_dec = tg[S0 + 1:S0 + N + 1]
        rec = dict(model=args.model, window=wid, ctx=CTX, S0=S0, steps=N, pop1=POP1, tokens_sha256=C.tokens_sha(ids),
                   window_parent_sha256=C.AB1C.window_hash(8192, wid), arms={})
        arrays = {}
        ref = None
        TALLY.clear()
        for a in ARMS:
            r, pre, dec = pf_pass(torch, H, R, a, ids)
            if a == 'FLASH':
                ref = (pre, dec)
            for pop, z, zr, tg_ in (('prefill', pre, ref[0], tgt_pre), ('decode', dec, ref[1], tgt_dec)):
                kl, na, nr, t1, am, ar = C.metrics(z, zr, tg_)
                for k_, v_ in dict(kl=kl, nll_arm=na, nll_ref=nr, top1=t1).items():
                    arrays[f'{a}__{pop}__{k_}'] = v_.cpu().numpy()
                assert np.isfinite(arrays[f'{a}__{pop}__kl']).all() and np.isfinite(arrays[f'{a}__{pop}__nll_arm']).all()
                r[f'{pop}_sums'] = dict(kl=float(kl.double().sum()), nll=float(na.double().sum()),
                                        top1=int(t1.sum()), n=int(kl.numel()))
            r['prefill_routes'] = TALLY.get(a)
            rec['arms'][a] = r
            del pre, dec
            print('ARM', args.model, wid, a, json.dumps({k: v for k, v in r.items() if k.endswith('_sums')}),
                  'wall %.1f' % r['wall_s'], flush=True)
        ref = None
        torch.cuda.empty_cache()
        rec['peak'] = R.peak.result()
        npz = jp.with_suffix('.npz')
        C.save_npz(npz, **arrays)
        rec['npz_sha256'] = C.sha(npz)
        rec['engine'] = arm
        rec['window_seconds'] = time.time() - tw
        C.save_json(jp, rec)
        per_window.append(rec['window_seconds'])
        G.step(log, 'window', wid=wid, seconds=rec['window_seconds'])
    G.save(out / f'job_{int(time.time())}.json', dict(model=args.model, windows=wins, steps=log,
                                                       seconds=time.time() - t_start))
    return 0


def report(args):
    import apa_p1q_tuning as PT
    import apa_p1q_common as C
    base = Path(args.out)
    table, ok_all = [], True
    for model in ('Qwen3-1.7B', 'Qwen3-4B'):
        recs = [json.loads((base / model / f'w{w}.json').read_text()) for w in PT.AB1_WIN_INDICES
                if (base / model / f'w{w}.json').exists()]
        if len(recs) != 10:
            table.append(dict(model=model, complete=False, windows=len(recs)))
            ok_all = False
            continue
        Z = [np.load(base / model / f'w{r["window"]}.npz') for r in recs]
        routes_ok = all(r['arms'][a]['prefill_routes']['calls'] == r['arms'][a]['prefill_routes']['ok'] > 0
                        for r in recs for a in ('SP-DENSE', 'SP-PACKED'))
        for pop, slack_events in (('prefill', 30), ('decode', 7)):
            def cat(a, k):
                return [z[f'{a}__{pop}__{k}'].astype(np.float64) for z in Z]
            kl_d, kl_p = np.concatenate(cat('SP-DENSE', 'kl')), np.concatenate(cat('SP-PACKED', 'kl'))
            t_d, t_p = int(np.concatenate(cat('SP-DENSE', 'top1')).sum()), int(np.concatenate(cat('SP-PACKED', 'top1')).sum())
            nd, np_ = cat('SP-DENSE', 'nll_arm'), cat('SP-PACKED', 'nll_arm')
            dn = [p - d for p, d in zip(np_, nd)]
            Nn = int(kl_d.size)
            dnll = float(np.concatenate(dn).mean())
            sums = np.array([x.sum() for x in dn])
            cnts = np.array([x.size for x in dn])
            lo, hi = C.bootstrap_ci(lambda idx: float(sums[idx].sum() / cnts[idx].sum()), len(dn))
            kl_ok = kl_p.mean() <= KL_FACTOR * kl_d.mean() + KL_ABS
            top_ok = t_p >= t_d - slack_events
            dn_ok = dnll <= DNLL_MAX and hi <= DNLL_CI_MAX
            row = dict(model=model, population=pop, N=Nn, mean_kl_dense=float(kl_d.mean()), mean_kl_packed=float(kl_p.mean()),
                       kl_bound=float(KL_FACTOR * kl_d.mean() + KL_ABS), kl_ok=bool(kl_ok),
                       top1_dense=t_d, top1_packed=t_p, top1_slack_events=slack_events,
                       top1_frac_dense=t_d / Nn, top1_frac_packed=t_p / Nn, top1_ok=bool(top_ok),
                       dnll_mean=dnll, dnll_ci95=[lo, hi], dnll_ok=bool(dn_ok),
                       pass_=bool(kl_ok and top_ok and dn_ok), prefill_routes_ok=routes_ok)
            ok_all = ok_all and row['pass_'] and routes_ok
            table.append(row)
    res = dict(line='G3', status='PASS' if ok_all and len(table) == 4 else 'FAIL', table=table,
               registration_sha256=G.sha(G.ART / 'REGISTRATION.json'))
    G.save(base / 'RESULT.json', res)
    print(json.dumps(res, indent=1))
    return 0


def top1_gate_selftest():
    """L104 boundary form (QR3): an arm exactly at the slack passes, one event past it fails."""
    rows = []
    for N_, slack in ((10240, 30), (2560, 7)):
        d = 9000 if N_ == 10240 else 2000
        rows.append((N_, d - slack >= d - slack, not (d - slack - 1 >= d - slack),
                     (d - slack) / N_ >= d / N_ - TOP1_SLACK, not ((d - slack - 1) / N_ >= d / N_ - TOP1_SLACK)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode')
    ap.add_argument('--out', required=False)
    ap.add_argument('--model')
    ap.add_argument('--windows')
    ap.add_argument('--max-seconds', type=float, default=1500)
    args = ap.parse_args()
    if args.mode == 'report':
        return report(args)
    if args.mode == 'selftest':
        print(top1_gate_selftest())
        return 0
    assert os.environ.get('CUDA_VISIBLE_DEVICES') != '', 'GPU modes run inside forge-gpu-run'
    return job(args)


if __name__ == '__main__':
    raise SystemExit(main())
