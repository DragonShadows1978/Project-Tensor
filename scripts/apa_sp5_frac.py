#!/usr/bin/env python3
"""APA-SP5 realised refine fraction per arm on the real forward, measured from
the SP kernel's own diagnostic selection mask (not a reimplementation).

Prior art: SP3/SP4G diagnostic-mask fraction cells (this repo, 2026);
BLASST running-max criterion (Yuan et al., arXiv 2512.12087). Measurement only.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, '/mnt/ForgeRealm/Project-Tensor-wt-apa-sp5/scripts')
from apa_sp5_model import load_runtime, port, tokens, A, SNAP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--delta', type=float, required=True)
    ap.add_argument('--label', default='')
    ap.add_argument('--W', type=int, default=1024)
    ap.add_argument('--bulk-bits', type=int, default=4)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    tc = load_runtime(); G = port(tc)
    from core.mistral7b_tc import QuantLinearTC
    QuantLinearTC.USE_FUSED = True
    from tensor_cuda.quant import _quantize_keys, _tables
    ids, meta = tokens()
    model, info = G.GptOss20B_TC.from_pretrained(SNAP)
    cfg = model.config
    rows = []
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
        S = k.shape[2]
        Rt, CB, BND = _tables(self.head_dim, self.bulk_bits, self.num_kv_heads,
                              True, q.device.split(':')[0])
        kq = _quantize_keys(k, Rt, CB, BND)
        attn, sel = tc._C.apa_selective_attention_sp(
            q, k, kq, v, self.scaling, float(args.delta), True, self.sinks, True)
        m = sel.numpy().astype(bool)
        e2 = np.arange(S)[None, :] <= (S - L + np.arange(L))[:, None]
        e = np.broadcast_to(e2[None, None], m.shape)
        rows.append(dict(layer=self.layer_idx, L=L, S=S,
                         eligible=int(e.sum()), selected=int(m[e].sum()),
                         fraction=float(m[e].mean()),
                         leak=int((m & ~e).sum())))
        del sel, m
        self.last_attention_backend = 'apa_sp_sink_fused'
        out = attn.transpose(1, 2).reshape([B, L, self.num_heads * self.head_dim])
        return self.o_proj(out), (k, v)

    G.GptOssAttentionTC.__call__ = patched
    for lyr in model.layers:
        lyr.self_attn.attention_mode = 'standard'
        lyr.self_attn.bulk_bits = args.bulk_bits
    win = ids[:args.W].reshape(1, args.W)
    with tc.no_grad():
        model.extend_rope(args.W + 8)
        h = model.embed_tokens(win)
        for lyr in model.layers:
            h, kv, _ = lyr(h, model.rope_cos, model.rope_sin, 0, None)
            del kv; tc.empty_cache()
    G.GptOssAttentionTC.__call__ = original
    tot_e = sum(r['eligible'] for r in rows)
    tot_s = sum(r['selected'] for r in rows)
    res = dict(delta=args.delta, label=args.label, W=args.W,
               bulk_bits=args.bulk_bits, per_layer=rows,
               overall_fraction=tot_s / tot_e,
               fraction_min=min(r['fraction'] for r in rows),
               fraction_max=max(r['fraction'] for r in rows),
               fraction_median=float(np.median([r['fraction'] for r in rows])),
               total_leak=sum(r['leak'] for r in rows),
               token_sha256=meta['token_sha256'])
    out = args.out or (A / f'frac_d{args.delta:g}{("_"+args.label) if args.label else ""}.json')
    Path(out).write_text(json.dumps(res, indent=2) + '\n')
    res['artifact'] = str(out)
    print(json.dumps({k: v for k, v in res.items() if k != 'per_layer'}, indent=2))
    print('per_layer:', json.dumps([{'l': r['layer'], 'f': round(r['fraction'], 4)}
                                    for r in rows]))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
