#!/usr/bin/env python3
"""APA-SP5 calibration on REAL GPT-OSS-20B keys (pre-registration inputs):

  (a) bulk quantization error e_q on this model's D=64 GQA keys at 4 (and 8)
      bits -> arm E's provable delta = ln(1/eps) + 2*e_q.
  (b) arm B's realised refine fraction on the FULL-attention layers at the
      port's registered refine_percentile -> the target arm C's single global
      delta must match to +/-0.01.
  (c) the delta that hits that fraction, by bisection on the running-max rule.
  (d) sink softmax mass per query (item 4's new metric) on the same keys.

Prior art: SP2/SP3 e_q rule (this repo, 2026) -- delta = ln(1/eps) + 2*e_q,
whose 2*e_q term exists because APA's comparator runs on quantized scores on
BOTH sides; BLASST (Yuan et al. 2512.12087) for the running-max criterion
itself; ThriftAttention (Sharratt 2605.23081) for weight-proportional error;
TurboQuant-style rotation+codebook quantizer (tensor_cuda.quant);
GPT-OSS attention sinks (OpenAI model card, 2025). The sink-mass metric is
new here only in that no previous SP model test had sinks to measure.
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, '/mnt/ForgeRealm/Project-Tensor-wt-apa-sp5/scripts')
from apa_sp5_model import load_runtime, port, gpu_mib, tokens, A, SNAP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--W', type=int, default=1024)
    ap.add_argument('--bulk-bits', type=int, default=4)
    ap.add_argument('--refine-percentile', type=float, default=0.15)
    ap.add_argument('--eps', type=float, default=0.01)
    ap.add_argument('--layers', default='')   # '' = all full layers
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    tc = load_runtime()
    G = port(tc)
    from core.mistral7b_tc import QuantLinearTC
    QuantLinearTC.USE_FUSED = True
    from tensor_cuda.quant import _quantize_keys, _tables, _norm_ppf
    ids, meta = tokens()
    W = args.W
    model, info = G.GptOss20B_TC.from_pretrained(SNAP)
    cfg = model.config
    fulls = cfg.full_attention_indices()
    want = ([int(x) for x in args.layers.split(',')] if args.layers else fulls)

    caps = {}
    original = G.GptOssAttentionTC.__call__

    def patched(self, x, cos, sin, position_offset=0, kv_cache=None):
        out = original(self, x, cos, sin, position_offset, kv_cache)
        if self.layer_idx in want:
            B, L, _ = x.shape
            q = self.q_proj(x).reshape([B, L, self.num_heads, self.head_dim]).transpose(1, 2)
            k = self.k_proj(x).reshape([B, L, self.num_kv_heads, self.head_dim]).transpose(1, 2)
            cseg = cos.slice(0, position_offset, L)
            sseg = sin.slice(0, position_offset, L)
            q = G.F.apply_rotary(q, cseg, sseg)
            k = G.F.apply_rotary(k, cseg, sseg)
            # subsample QUERY rows to keep host memory bounded; keys are whole
            step = max(1, L // 64)
            qi = q.slice(2, 0, L)
            caps[self.layer_idx] = dict(
                q=qi.float().numpy()[0, :, ::step, :].copy(),
                k=k.float().numpy()[0].copy(),
                sinks=self.sinks.float().numpy().copy(),
                scale=self.scaling, L=L, step=step,
                qrows=list(range(0, L, step)))
            del q, k, qi
            tc.empty_cache()
        return out

    G.GptOssAttentionTC.__call__ = patched
    for lyr in model.layers:
        lyr.self_attn.attention_mode = 'standard'
    win = ids[:W].reshape(1, W)
    with tc.no_grad():
        model.extend_rope(W + 8)
        h = model.embed_tokens(win)
        for lyr in model.layers:
            h, kv, _ = lyr(h, model.rope_cos, model.rope_sin, 0, None)
            del kv
            tc.empty_cache()
        tc.synchronize()
    G.GptOssAttentionTC.__call__ = original
    del h, model
    tc.empty_cache()

    z = _norm_ppf(1.0 - max(0.0, min(1.0, args.refine_percentile)))
    rows = []
    for li in sorted(caps):
        c = caps[li]
        qn, kn = c['q'], c['k']            # q [H,nq,D], k [KVH,S,D]
        H, nq, D = qn.shape
        KVH, S, _ = kn.shape
        g = H // KVH
        kt = tc.tensor(kn.reshape(1, KVH, S, D).astype(np.float32), dtype='bfloat16')
        Rt, CB, BND = _tables(D, args.bulk_bits, KVH, True, kt.device.split(':')[0])
        kqt = _quantize_keys(kt, Rt, CB, BND)
        kq = kqt.float().numpy()[0]
        scale = c['scale']
        errs = []; fracB = []; massB = []; relB = []
        sinkmass = []
        allbulk = []; allexact = []
        for h_ in range(H):
            kh = kn[h_ // g].astype(np.float64)
            kqh = kq[h_ // g].astype(np.float64)
            for j, qpos in enumerate(c['qrows']):
                lim = qpos + 1
                qv = qn[h_, j].astype(np.float64)
                ex = (qv @ kh[:lim].T) * scale
                bu = (qv @ kqh[:lim].T) * scale
                errs.append(np.abs(ex - bu))
                allbulk.append(bu); allexact.append(ex)
                # arm B: z-score threshold on |bulk| (port's rule)
                thr = np.abs(bu).mean() + z * np.abs(bu).std() if lim > 1 else -np.inf
                selB = np.abs(bu) >= thr
                fracB.append(selB.mean())
                # softmax mass left unrefined, under EXACT scores + the sink
                sk = float(c['sinks'][h_])
                m = max(ex.max(), sk)
                w = np.exp(ex - m); den = w.sum() + np.exp(sk - m)
                p = w / den
                sinkmass.append(float(np.exp(sk - m) / den))
                massB.append(float(p[~selB].sum()))
                relB.append(float(p[~selB].max() / p.max()) if (~selB).any() else 0.0)
        err = np.concatenate(errs)
        row = dict(layer=li, S=S, H=H, KVH=KVH, D=D, queries=nq,
                   bulk_bits=args.bulk_bits,
                   e_mean=float(err.mean()), e_p99=float(np.percentile(err, 99)),
                   e_max=float(err.max()),
                   B_frac=float(np.mean(fracB)),
                   B_unrefined_mass_mean=float(np.mean(massB)),
                   B_unrefined_mass_max=float(np.max(massB)),
                   B_max_skipped_rel=float(np.max(relB)),
                   sink_mass_mean=float(np.mean(sinkmass)),
                   sink_mass_median=float(np.median(sinkmass)),
                   sink_mass_max=float(np.max(sinkmass)))
        # (c) delta that reproduces B_frac under the running-max rule
        target = row['B_frac']
        def frac_at(delta):
            tot = 0; sel = 0
            for bu in allbulk:
                run = np.maximum.accumulate(bu)
                s = bu >= run - delta
                sel += int(s.sum()); tot += s.size
            return sel / tot
        lo_d, hi_d = 0.0, 60.0
        for _ in range(40):
            mid = (lo_d + hi_d) / 2
            if frac_at(mid) < target:
                lo_d = mid
            else:
                hi_d = mid
        row['C_delta_matched'] = float((lo_d + hi_d) / 2)
        row['C_frac_at_delta'] = float(frac_at(row['C_delta_matched']))
        rows.append(row)
        del kt, kqt
        tc.empty_cache()

    e_max_all = max(r['e_max'] for r in rows)
    e_p99_all = max(r['e_p99'] for r in rows)
    delta_E = float(np.log(1.0 / args.eps) + 2.0 * e_max_all)
    res = dict(W=W, bulk_bits=args.bulk_bits,
               refine_percentile=args.refine_percentile, z=float(z),
               eps=args.eps, full_layers=fulls, measured_layers=sorted(caps),
               per_layer=rows,
               e_q_rule='max over all measured (layer, head, query, key) '
                        'absolute bulk-vs-exact logit errors -- the SP2/SP3 '
                        'registered rule (the max, not a percentile)',
               e_q_max=e_max_all, e_q_p99=e_p99_all,
               E_delta_provable=delta_E,
               B_frac_median=float(np.median([r['B_frac'] for r in rows])),
               C_delta_median=float(np.median([r['C_delta_matched'] for r in rows])),
               sink_mass_median_of_layer_medians=float(
                   np.median([r['sink_mass_median'] for r in rows])),
               token_sha256=meta['token_sha256'])
    out = args.out or (A / f'calib_W{W}_b{args.bulk_bits}.json')
    Path(out).write_text(json.dumps(res, indent=2) + '\n')
    slim = {k: v for k, v in res.items() if k != 'per_layer'}
    slim['artifact'] = str(out)
    slim['per_layer_summary'] = [
        {k: round(r[k], 4) if isinstance(r[k], float) else r[k]
         for k in ('layer', 'e_mean', 'e_p99', 'e_max', 'B_frac',
                   'B_unrefined_mass_mean', 'B_max_skipped_rel',
                   'sink_mass_median', 'C_delta_matched', 'C_frac_at_delta')}
        for r in rows]
    print(json.dumps(slim, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
