#!/usr/bin/env python3
"""APA-SP5 ITEM 3 -- per-call exactness on REAL activations (the registered
gate G_exactness_per_call, threshold <= 1e-4 max-abs / <= 1e-5 relF in fp32).

Captures the actual q/k/v/sinks a real GPT-OSS-20B forward presents to one
FULL-attention layer, for two call shapes:
  (i)  a prefill chunk   (L = S, kv_cache = None)
  (ii) a cached short block (L < S, kv_cache present) -- the decode-adjacent
       geometry the smoke never exercised.
Then compares, on those identical bytes, in BOTH fp32 and bf16:
  single-pass refine-all  vs  the port's standard sink path  vs  a dense
  fp64/fp32 NumPy reference that folds the sink into the denominator only.

Prior art: SP4G a3/a5 same-input capture and replay (this repo, 2026);
GPT-OSS sink convention (OpenAI model card 2025) as implemented in the port's
sink_attention_tc. No new algorithm; this is a measurement.
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, '/mnt/ForgeRealm/Project-Tensor-wt-apa-sp5/scripts')
from apa_sp5_model import load_runtime, port, gpu_mib, tokens, A, SNAP


def dense_ref(q, k, v, sinks, scale, kvh, causal_offset):
    """fp64 dense reference; sink in the denominator only (port convention)."""
    B, H, L, D = q.shape
    S = k.shape[2]
    g = H // kvh
    out = np.zeros((B, H, L, v.shape[3]), dtype=np.float64)
    for b in range(B):
        for h in range(H):
            kh = k[b, h // g].astype(np.float64)
            vh = v[b, h // g].astype(np.float64)
            sk = float(sinks[h])
            for i in range(L):
                lim = causal_offset + i + 1
                s = (q[b, h, i].astype(np.float64) @ kh[:lim].T) * scale
                m = max(s.max(), sk)
                w = np.exp(s - m)
                out[b, h, i] = (w @ vh[:lim]) / (w.sum() + np.exp(sk - m))
    return out


def cmp(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    d = a - b
    nb = np.linalg.norm(b)
    return dict(max_abs=float(np.abs(d).max()),
                rel_frobenius=float(np.linalg.norm(d) / nb) if nb else 0.0,
                bitwise=bool(np.array_equal(a, b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--layer', type=int, default=11)
    ap.add_argument('--W', type=int, default=1024)
    ap.add_argument('--prefill-chunk', type=int, default=256)
    ap.add_argument('--cached-block', type=int, default=64)
    ap.add_argument('--bulk-bits', type=int, default=4)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    tc = load_runtime()
    G = port(tc)
    from core.mistral7b_tc import QuantLinearTC
    QuantLinearTC.USE_FUSED = True
    from tensor_cuda.quant import _quantize_keys, _tables
    ids, meta = tokens()
    model, info = G.GptOss20B_TC.from_pretrained(SNAP)
    cfg = model.config
    if not cfg.is_full_attention(args.layer):
        raise SystemExit(f'layer {args.layer} is not a full-attention layer')

    cap = {}
    original = G.GptOssAttentionTC.__call__

    def patched(self, x, cos, sin, position_offset=0, kv_cache=None):
        out = original(self, x, cos, sin, position_offset, kv_cache)
        if self.layer_idx == args.layer and 'q' not in cap:
            B, L, _ = x.shape
            q = self.q_proj(x).reshape([B, L, self.num_heads, self.head_dim]).transpose(1, 2)
            k = self.k_proj(x).reshape([B, L, self.num_kv_heads, self.head_dim]).transpose(1, 2)
            v = self.v_proj(x).reshape([B, L, self.num_kv_heads, self.head_dim]).transpose(1, 2)
            cseg = cos.slice(0, position_offset, L)
            sseg = sin.slice(0, position_offset, L)
            q = G.F.apply_rotary(q, cseg, sseg)
            k = G.F.apply_rotary(k, cseg, sseg)
            cap['q'] = q.float().numpy().copy()
            cap['k'] = k.float().numpy().copy()
            cap['v'] = v.float().numpy().copy()
            cap['sinks'] = self.sinks.float().numpy().copy()
            cap['scale'] = self.scaling
            cap['nhpk'] = self.num_heads_per_kv
            del q, k, v
            tc.empty_cache()
        return out

    G.GptOssAttentionTC.__call__ = patched
    for lyr in model.layers:
        lyr.self_attn.attention_mode = 'standard'
    win = ids[:args.W].reshape(1, args.W)
    with tc.no_grad():
        model.extend_rope(args.W + 8)
        h = model.embed_tokens(win)
        for i, lyr in enumerate(model.layers):
            h, kv, _ = lyr(h, model.rope_cos, model.rope_sin, 0, None)
            del kv
            tc.empty_cache()
            if i >= args.layer:
                break
    G.GptOssAttentionTC.__call__ = original
    del h, model
    tc.empty_cache()

    qn, kn, vn, skn = cap['q'], cap['k'], cap['v'], cap['sinks']
    scale, nhpk = cap['scale'], cap['nhpk']
    KVH = kn.shape[1]
    rec = dict(layer=args.layer, W=args.W, bulk_bits=args.bulk_bits,
               shape=dict(H=qn.shape[1], KVH=KVH, D=qn.shape[3],
                          S_captured=kn.shape[2]),
               scale=scale, token_sha256=meta['token_sha256'], cases=[])

    for case, nq, S in (('prefill_chunk', args.prefill_chunk, args.prefill_chunk),
                        ('cached_block', args.cached_block, args.W)):
        q_s = qn[:, :, S - nq:S, :].copy()
        k_s = kn[:, :, :S, :].copy()
        v_s = vn[:, :, :S, :].copy()
        offset = S - nq
        ref = dense_ref(q_s, k_s, v_s, skn, scale, KVH, offset)
        for dtype in ('float32', 'bfloat16'):
            qt = tc.tensor(q_s, dtype=dtype)
            kt = tc.tensor(k_s, dtype=dtype)
            vt = tc.tensor(v_s, dtype=dtype)
            st = tc.tensor(skn, dtype=dtype)
            Rt, CB, BND = _tables(qn.shape[3], args.bulk_bits, KVH, True,
                                  qt.device.split(':')[0])
            kqt = _quantize_keys(kt, Rt, CB, BND)
            # round-trip reference on the exact bytes the kernels see
            ref_rt = dense_ref(qt.float().numpy(), kt.float().numpy(),
                               vt.float().numpy(), st.float().numpy(),
                               scale, KVH, offset)
            with tc.no_grad():
                t0 = time.perf_counter()
                sp, sel = tc._C.apa_selective_attention_sp(
                    qt, kt, kqt, vt, scale, 1e9, True, st, True)
                tc.synchronize()
                sp_wall = time.perf_counter() - t0
                mask = G._gpt_oss_attention_mask(nq, S, sliding_window=None,
                                                 dtype=dtype)
                t0 = time.perf_counter()
                std = G.sink_attention_tc(qt, kt, vt, st, scale=scale,
                                          attention_mask=mask,
                                          num_heads_per_kv=nhpk)
                tc.synchronize()
                std_wall = time.perf_counter() - t0
            spn = sp.float().numpy()
            stdn = std.float().numpy()
            selm = sel.numpy().astype(bool)
            elig2 = np.arange(S)[None, :] <= (offset + np.arange(nq))[:, None]
            elig = np.broadcast_to(elig2[None, None], selm.shape)
            rec['cases'].append(dict(
                case=case, dtype=dtype, L=nq, S=S, causal_offset=offset,
                SP_vs_standard=cmp(spn, stdn),
                SP_vs_dense=cmp(spn, ref_rt),
                standard_vs_dense=cmp(stdn, ref_rt),
                SP_vs_dense_capture_fp32ref=cmp(spn, ref),
                all_refined=bool(selm[elig].all()),
                refined_fraction=float(selm[elig].mean()),
                causal_leak=int((selm & ~elig).sum()),
                sp_wall_s=sp_wall, std_wall_s=std_wall))
            del qt, kt, vt, st, kqt, sp, std, sel, mask
            tc.empty_cache()

    # registered gate evaluation, fp32 only
    fp32 = [c for c in rec['cases'] if c['dtype'] == 'float32']
    worst_ma = max(c['SP_vs_standard']['max_abs'] for c in fp32)
    worst_rf = max(c['SP_vs_standard']['rel_frobenius'] for c in fp32)
    worst_ma_d = max(c['SP_vs_dense']['max_abs'] for c in fp32)
    worst_rf_d = max(c['SP_vs_dense']['rel_frobenius'] for c in fp32)
    all_ref = all(c['all_refined'] for c in rec['cases'])
    leak = sum(c['causal_leak'] for c in rec['cases'])
    passed = (worst_ma <= 1e-4 and worst_rf <= 1e-5 and
              worst_ma_d <= 1e-4 and worst_rf_d <= 1e-5 and
              all_ref and leak == 0)
    rec['gate'] = dict(name='G_exactness_per_call',
                       threshold='fp32 <= 1e-4 max-abs and <= 1e-5 relF',
                       worst_fp32_SP_vs_standard_max_abs=worst_ma,
                       worst_fp32_SP_vs_standard_relF=worst_rf,
                       worst_fp32_SP_vs_dense_max_abs=worst_ma_d,
                       worst_fp32_SP_vs_dense_relF=worst_rf_d,
                       refine_all_confirmed=all_ref, causal_leak_total=leak,
                       status='PASS' if passed else 'RED')
    out = args.out or (A / f'item3_exactness_L{args.layer}.json')
    Path(out).write_text(json.dumps(rec, indent=2) + '\n')
    rec['artifact'] = str(out)
    print(json.dumps(rec, indent=2))
    return 0 if passed else 3


if __name__ == '__main__':
    raise SystemExit(main())
