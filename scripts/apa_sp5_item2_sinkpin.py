#!/usr/bin/env python3
"""APA-SP5 ITEM 2 sink pin -- per-call comparison on ONE full layer with the
learned sinks ON vs ZEROED, for every arm, on identical real activations.

Purpose (order item 2): pin exactly how the sink logit enters each arm and
prove the claim that the sink is NOT a selection candidate under the
single-pass running-max rule -- i.e. zeroing the sink must change the OUTPUT
(it changes the denominator) but must NOT change the SELECTION MASK.

Prior art: GPT-OSS attention sinks (OpenAI model card, 2025); StreamingLLM
(Xiao et al., arXiv 2309.17453, 2023) for the sink phenomenon; the port's
sink_attention_tc / apa_selective_attention_sink. The selection-invariance
claim is read off kernels.cu (the sink is folded AFTER the key loop); this
script is the empirical pin of that reading.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, '/mnt/ForgeRealm/Project-Tensor-wt-apa-sp5/scripts')
from apa_sp5_model import load_runtime, port, tokens, A, SNAP


def cmp(a, b):
    a = np.asarray(a, np.float64).ravel(); b = np.asarray(b, np.float64).ravel()
    d = a - b; nb = np.linalg.norm(b)
    return dict(max_abs=float(np.abs(d).max()),
                rel_frobenius=float(np.linalg.norm(d) / nb) if nb else 0.0,
                bitwise=bool(np.array_equal(a, b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--layer', type=int, default=19)
    ap.add_argument('--W', type=int, default=1024)
    ap.add_argument('--L', type=int, default=256)
    ap.add_argument('--bulk-bits', type=int, default=4)
    ap.add_argument('--delta', type=float, default=3.16)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    tc = load_runtime(); G = port(tc)
    from core.mistral7b_tc import QuantLinearTC
    QuantLinearTC.USE_FUSED = True
    from tensor_cuda.quant import _quantize_keys, _tables, _norm_ppf
    ids, meta = tokens()
    model, info = G.GptOss20B_TC.from_pretrained(SNAP)
    cfg = model.config
    cap = {}
    original = G.GptOssAttentionTC.__call__

    def patched(self, x, cos, sin, position_offset=0, kv_cache=None):
        out = original(self, x, cos, sin, position_offset, kv_cache)
        if self.layer_idx == args.layer and 'q' not in cap:
            B, L, _ = x.shape
            q = self.q_proj(x).reshape([B, L, self.num_heads, self.head_dim]).transpose(1, 2)
            k = self.k_proj(x).reshape([B, L, self.num_kv_heads, self.head_dim]).transpose(1, 2)
            v = self.v_proj(x).reshape([B, L, self.num_kv_heads, self.head_dim]).transpose(1, 2)
            cs = cos.slice(0, position_offset, L); ss = sin.slice(0, position_offset, L)
            q = G.F.apply_rotary(q, cs, ss); k = G.F.apply_rotary(k, cs, ss)
            cap.update(q=q.float().numpy().copy(), k=k.float().numpy().copy(),
                       v=v.float().numpy().copy(),
                       sinks=self.sinks.float().numpy().copy(),
                       scale=self.scaling, nhpk=self.num_heads_per_kv)
            del q, k, v; tc.empty_cache()
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
            del kv; tc.empty_cache()
            if i >= args.layer:
                break
    G.GptOssAttentionTC.__call__ = original
    del h, model; tc.empty_cache()

    nq = args.L; S = args.L
    q_s = cap['q'][:, :, :nq, :].copy(); k_s = cap['k'][:, :, :S, :].copy()
    v_s = cap['v'][:, :, :S, :].copy(); skn = cap['sinks']
    scale, nhpk = cap['scale'], cap['nhpk']
    KVH, D = k_s.shape[1], q_s.shape[3]
    dtype = 'bfloat16'
    qt = tc.tensor(q_s, dtype=dtype); kt = tc.tensor(k_s, dtype=dtype)
    vt = tc.tensor(v_s, dtype=dtype)
    st = tc.tensor(skn, dtype=dtype)
    z0 = tc.tensor(np.zeros_like(skn), dtype=dtype)
    Rt, CB, BND = _tables(D, args.bulk_bits, KVH, True, qt.device.split(':')[0])
    kqt = _quantize_keys(kt, Rt, CB, BND)
    z = _norm_ppf(1.0 - 0.15)
    rec = dict(layer=args.layer, L=nq, S=S, dtype=dtype, delta=args.delta,
               sinks_abs_mean=float(np.abs(skn).mean()),
               sinks_min=float(skn.min()), sinks_max=float(skn.max()),
               token_sha256=meta['token_sha256'], arms={})
    with tc.no_grad():
        # A standard
        mask = G._gpt_oss_attention_mask(nq, S, sliding_window=None, dtype=dtype)
        a_on = G.sink_attention_tc(qt, kt, vt, st, scale=scale,
                                   attention_mask=mask, num_heads_per_kv=nhpk)
        a_off = G.sink_attention_tc(qt, kt, vt, z0, scale=scale,
                                    attention_mask=mask, num_heads_per_kv=nhpk)
        rec['arms']['A'] = dict(
            sink_site='concatenated as an extra score column, softmax over S+1, '
                      'sink column sliced off before the value matmul '
                      '(denominator only)',
            on_vs_off=cmp(a_on.float().numpy(), a_off.float().numpy()),
            selection_mask='n/a — standard refines every key')
        # B two-pass
        b_on = tc.apa_selective_attention_sink(qt, kt, kqt, vt, st, scale, float(z), True)
        b_off = tc.apa_selective_attention_sink(qt, kt, kqt, vt, z0, scale, float(z), True)
        rec['arms']['B'] = dict(
            sink_site='folded inside the fused two-pass sink kernel at the end '
                      'of the key loop; the z-score threshold is computed on '
                      '|bulk| over real keys only, so the sink is not a '
                      'selection candidate',
            on_vs_off=cmp(b_on.float().numpy(), b_off.float().numpy()),
            selection_mask='not exposed by this entry')
        # C / D single pass
        for name, dl in (('C', args.delta), ('D', 1e9)):
            on, sel_on = tc._C.apa_selective_attention_sp(
                qt, kt, kqt, vt, scale, float(dl), True, st, True)
            off, sel_off = tc._C.apa_selective_attention_sp(
                qt, kt, kqt, vt, scale, float(dl), True, z0, True)
            non, noff = sel_on.numpy(), sel_off.numpy()
            elig2 = np.arange(S)[None, :] <= np.arange(nq)[:, None]
            elig = np.broadcast_to(elig2[None, None], non.shape)
            rec['arms'][name] = dict(
                delta=dl,
                sink_site='folded into the running max AFTER the key loop '
                          '(kernels.cu apa_selective_sp_kernel: "if (sinks) '
                          '{...}" runs once, after every key has been decided) '
                          '— denominator + accumulator rescale only',
                on_vs_off=cmp(on.float().numpy(), off.float().numpy()),
                selection_mask_identical=bool(np.array_equal(non, noff)),
                selection_differing_entries=int((non != noff).sum()),
                refined_fraction_on=float(non.astype(bool)[elig].mean()),
                refined_fraction_off=float(noff.astype(bool)[elig].mean()))
    ok = (rec['arms']['C']['selection_mask_identical'] and
          rec['arms']['D']['selection_mask_identical'] and
          not rec['arms']['A']['on_vs_off']['bitwise'] and
          not rec['arms']['C']['on_vs_off']['bitwise'])
    rec['pin'] = dict(
        claim='The learned sink changes every arm\'s OUTPUT (it is in the '
              'denominator) but does NOT change the single-pass SELECTION '
              '(it is folded after the key loop and is not a key).',
        verdict='CONFIRMED' if ok else 'NOT CONFIRMED',
        status='PASS' if ok else 'RED')
    out = args.out or (A / f'item2_sinkpin_L{args.layer}.json')
    Path(out).write_text(json.dumps(rec, indent=2) + '\n')
    rec['artifact'] = str(out)
    print(json.dumps(rec, indent=2))
    return 0 if ok else 3


if __name__ == '__main__':
    raise SystemExit(main())
