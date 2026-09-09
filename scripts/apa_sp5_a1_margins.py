#!/usr/bin/env python3
"""APA-SP5 amendment 1 item 4 -- BITWISE REPLAY margins on the full-attention
layers, arms B and C, for one window.

Construction (SP4G a2): capture, during the ACTUAL scoring forward, every
full-attention call's inputs AND the kernel's own selection mask; then replay
each call and require the replayed output and mask to be BITWISE identical to
the captured ones before any metric is computed. A metric taken from a
recomputed selection is not a receipt; a metric taken from a bitwise-replayed
one is. If replay is not bitwise the cell is RED and reports it.

Metrics per full-attention layer, over all executed causal (query,key) pairs:
  |bulk - exact| mean / p99 / max      (bulk = quantized-key logit)
  unrefined softmax mass               (mass NOT promoted to full precision)
  max relative weight of a skipped key (p_skipped_max / p_max)
  realised refine fraction
  sink softmax mass per query          (new: no earlier SP model had sinks)

The question this cell exists to answer (lead, amendment 1): on window 1,
what does the running-max tail skip that the z-score tail refines, and vice
versa -- is the 2.9x spread a MASS story or a SINK story?

Prior art: SP4G a2 bitwise replay (this repo, 2026); SP3 native masks/replay;
BLASST running-max criterion (Yuan et al., arXiv 2512.12087); ThriftAttention
weight-proportional error (Sharratt, arXiv 2605.23081); GPT-OSS attention
sinks (OpenAI model card, 2025). Population statistics only; no new selector,
no tolerance introduced.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, '/mnt/ForgeRealm/Project-Tensor-wt-apa-sp5/scripts')
from apa_sp5_model import load_runtime, port, gpu_mib, tokens, A, SNAP

REG = '3cc3b3e112479ad71077a98a221fd5af521013d16ce423d52afe308e9ae58159'
AMD = '06a5a1e742ae2474469f8d5177bc5b9d35b4d89e9553d928c9cac1d7a6753e33'


def tile_metrics(q, k, kq, sinks, scale, sel, kvh, qstep):
    """fp64 metrics on a subsample of query rows. `sel` is the KERNEL's own
    bitwise-replayed mask, never recomputed here."""
    H, nq_all, D = q.shape[1], q.shape[2], q.shape[3]
    S = k.shape[2]
    g = H // kvh
    err = []
    unref_mass = []
    max_skipped_rel = []
    frac = []
    sinkmass = []
    rows = list(range(0, nq_all, qstep))
    for h in range(H):
        kh = k[0, h // g].astype(np.float64)
        kqh = kq[0, h // g].astype(np.float64)
        sk = float(sinks[h])
        for i in rows:
            lim = i + 1
            qv = q[0, h, i].astype(np.float64)
            ex = (qv @ kh[:lim].T) * scale
            bu = (qv @ kqh[:lim].T) * scale
            err.append(np.abs(ex - bu))
            s = sel[0, h, i, :lim].astype(bool)
            frac.append(s.mean())
            m = max(ex.max(), sk)
            w = np.exp(ex - m)
            den = w.sum() + np.exp(sk - m)
            p = w / den
            sinkmass.append(float(np.exp(sk - m) / den))
            unref_mass.append(float(p[~s].sum()))
            max_skipped_rel.append(float(p[~s].max() / p.max()) if (~s).any() else 0.0)
    e = np.concatenate(err)
    return dict(
        pairs=int(e.size), queries=len(rows) * H,
        e_mean=float(e.mean()), e_p99=float(np.percentile(e, 99)),
        e_max=float(e.max()),
        unrefined_mass_mean=float(np.mean(unref_mass)),
        unrefined_mass_p99=float(np.percentile(unref_mass, 99)),
        unrefined_mass_max=float(np.max(unref_mass)),
        max_skipped_relative_weight=float(np.max(max_skipped_rel)),
        max_skipped_relative_weight_mean=float(np.mean(max_skipped_rel)),
        fraction=float(np.mean(frac)),
        sink_mass_mean=float(np.mean(sinkmass)),
        sink_mass_median=float(np.median(sinkmass)),
        sink_mass_max=float(np.max(sinkmass)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=('B', 'C'))
    ap.add_argument('--window', type=int, required=True)
    ap.add_argument('--W', type=int, default=1024)
    ap.add_argument('--bulk-bits', type=int, default=4)
    ap.add_argument('--delta', type=float, default=3.16)
    ap.add_argument('--refine-percentile', type=float, default=0.15)
    ap.add_argument('--qstep', type=int, default=16)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    cell = f'margins_{args.arm}_W{args.W}_w{args.window:02d}'
    out = Path(args.out) if args.out else (A / 'margins_a1' / f'{cell}.json')
    if out.exists():
        print(json.dumps(dict(cell=cell, status='DONE_ALREADY', artifact=str(out))))
        return 0

    tc = load_runtime(); G = port(tc)
    from core.mistral7b_tc import QuantLinearTC
    QuantLinearTC.USE_FUSED = True
    from tensor_cuda.quant import _quantize_keys, _tables, _norm_ppf
    ids, meta = tokens()
    model, info = G.GptOss20B_TC.from_pretrained(SNAP)
    cfg = model.config
    z = _norm_ppf(1.0 - args.refine_percentile)

    captured = {}
    original = G.GptOssAttentionTC.__call__

    def patched(self, x, cos, sin, position_offset=0, kv_cache=None):
        if not cfg.is_full_attention(self.layer_idx):
            return original(self, x, cos, sin, position_offset, kv_cache)
        B, L, _ = x.shape
        q = self.q_proj(x).reshape([B, L, self.num_heads, self.head_dim]).transpose(1, 2)
        k = self.k_proj(x).reshape([B, L, self.num_kv_heads, self.head_dim]).transpose(1, 2)
        v = self.v_proj(x).reshape([B, L, self.num_kv_heads, self.head_dim]).transpose(1, 2)
        cs = cos.slice(0, position_offset, L); ss = sin.slice(0, position_offset, L)
        q = G.F.apply_rotary(q, cs, ss); k = G.F.apply_rotary(k, cs, ss)
        if kv_cache is not None:
            k = tc.cat([kv_cache[0], k], dim=2); v = tc.cat([kv_cache[1], v], dim=2)
        Rt, CB, BND = _tables(self.head_dim, args.bulk_bits, self.num_kv_heads,
                              True, q.device.split(':')[0])
        kq = _quantize_keys(k, Rt, CB, BND)
        if args.arm == 'C':
            attn, sel = tc._C.apa_selective_attention_sp(
                q, k, kq, v, self.scaling, float(args.delta), True, self.sinks, True)
            selnp = sel.numpy()
            self.last_attention_backend = 'apa_sp_sink_fused'
        else:
            attn = tc.apa_selective_attention_sink(
                q, k, kq, v, self.sinks, self.scaling, float(z), True)
            # B's rule is a GLOBAL statistic on |bulk| -- the two-pass entry
            # exposes no mask, so the mask is derived from the SAME quantized
            # keys the kernel used, by the SAME published rule. This is
            # labelled below as a DERIVED mask, not a kernel mask.
            selnp = None
            self.last_attention_backend = 'apa_selective_sink_fused'
        captured[self.layer_idx] = dict(
            q=q.float().numpy().copy(), k=k.float().numpy().copy(),
            kq=kq.float().numpy().copy(), v=v.float().numpy().copy(),
            sinks=self.sinks.float().numpy().copy(), scale=self.scaling,
            out=attn.float().numpy().copy(), sel=selnp,
            kvh=self.num_kv_heads, L=L, S=k.shape[2])
        o = attn.transpose(1, 2).reshape([B, L, self.num_heads * self.head_dim])
        return self.o_proj(o), (k, v)

    G.GptOssAttentionTC.__call__ = patched
    for i, l in enumerate(model.layers):
        l.self_attn.attention_mode = 'standard'
        l.self_attn.bulk_bits = args.bulk_bits
    lo = args.window * args.W
    win = ids[lo:lo + args.W].reshape(1, args.W)
    t0 = time.perf_counter()
    with tc.no_grad():
        model.extend_rope(args.W + 8)
        h = model.embed_tokens(win)
        for l in model.layers:
            h, kv, _ = l(h, model.rope_cos, model.rope_sin, 0, None)
            del kv
        tc.synchronize()
    G.GptOssAttentionTC.__call__ = original
    fwd_s = time.perf_counter() - t0
    del h; tc.empty_cache()

    rows = []
    replay_all_bitwise = True
    for li in sorted(captured):
        c = captured[li]
        qt = tc.tensor(c['q'], dtype='bfloat16'); kt = tc.tensor(c['k'], dtype='bfloat16')
        kqt = tc.tensor(c['kq'], dtype='bfloat16'); vt = tc.tensor(c['v'], dtype='bfloat16')
        st = tc.tensor(c['sinks'], dtype='bfloat16')
        with tc.no_grad():
            if args.arm == 'C':
                o2, s2 = tc._C.apa_selective_attention_sp(
                    qt, kt, kqt, vt, c['scale'], float(args.delta), True, st, True)
                mask_bitwise = bool(np.array_equal(s2.numpy(), c['sel']))
                sel = c['sel']
                mask_kind = 'kernel diagnostic mask, bitwise-replayed'
            else:
                o2 = tc.apa_selective_attention_sink(
                    qt, kt, kqt, vt, st, c['scale'], float(z), True)
                mask_bitwise = None
                # derive B's published rule on the captured quantized keys
                H = c['q'].shape[1]; S = c['S']; g = H // c['kvh']
                sel = np.zeros((1, H, c['L'], S), dtype=np.uint8)
                for hh in range(H):
                    kqh = c['kq'][0, hh // g].astype(np.float64)
                    for i in range(c['L']):
                        lim = i + 1
                        bu = (c['q'][0, hh, i].astype(np.float64) @ kqh[:lim].T) * c['scale']
                        ab = np.abs(bu)
                        thr = ab.mean() + z * ab.std() if lim > 1 else -np.inf
                        sel[0, hh, i, :lim] = (ab >= thr).astype(np.uint8)
                mask_kind = ('DERIVED from the published z-score rule on the '
                             'captured quantized keys; the two-pass entry '
                             'exposes no kernel mask')
        out_bitwise = bool(np.array_equal(o2.float().numpy(), c['out']))
        if not out_bitwise or (mask_bitwise is False):
            replay_all_bitwise = False
        m = tile_metrics(c['q'], c['k'], c['kq'], c['sinks'], c['scale'],
                         sel, c['kvh'], args.qstep)
        m.update(layer=li, L=c['L'], S=c['S'],
                 output_replay_bitwise=out_bitwise,
                 mask_replay_bitwise=mask_bitwise, mask_kind=mask_kind)
        rows.append(m)
        del qt, kt, kqt, vt, st, o2
        tc.empty_cache()
        captured[li] = None

    rec = dict(cell=cell, arm=args.arm, window=args.window, W=args.W,
               delta=args.delta if args.arm == 'C' else None,
               refine_percentile=args.refine_percentile if args.arm == 'B' else None,
               z=float(z) if args.arm == 'B' else None,
               bulk_bits=args.bulk_bits, qstep=args.qstep,
               registration_sha256=REG, amendment='APA_SP5_AMENDMENT_1',
               amendment_sha256=AMD, token_sha256=meta['token_sha256'],
               forward_wall_s=fwd_s, per_layer=rows,
               replay_all_bitwise=replay_all_bitwise,
               status='PASS' if replay_all_bitwise else 'RED_REPLAY_NOT_BITWISE',
               summary=dict(
                 e_mean_median=float(np.median([r['e_mean'] for r in rows])),
                 unrefined_mass_mean_median=float(np.median([r['unrefined_mass_mean'] for r in rows])),
                 unrefined_mass_mean_max=float(np.max([r['unrefined_mass_mean'] for r in rows])),
                 fraction_median=float(np.median([r['fraction'] for r in rows])),
                 sink_mass_median_of_medians=float(np.median([r['sink_mass_median'] for r in rows])),
                 max_skipped_relative_weight=float(np.max([r['max_skipped_relative_weight'] for r in rows]))),
               coverage=f'every {args.qstep}th query position, all 64 heads, all '
                        f'12 full-attention layers, all executed causal pairs '
                        f'for those queries; fp64 exact dots')
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as f:
        json.dump(rec, f, indent=2, allow_nan=False); f.write('\n')
    rec['artifact'] = str(out)
    print(json.dumps({k: rec[k] for k in ('cell', 'arm', 'window', 'status',
                                          'replay_all_bitwise', 'summary',
                                          'forward_wall_s', 'artifact')}, indent=2))
    return 0 if replay_all_bitwise else 3


if __name__ == '__main__':
    raise SystemExit(main())
