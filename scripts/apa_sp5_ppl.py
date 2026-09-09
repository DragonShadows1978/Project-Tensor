#!/usr/bin/env python3
"""APA-SP5 PROTOCOL-O perplexity cell: one arm over N consecutive windows of
W tokens, scoring the last W/2 targets of each window in fp64.

Registered protocol (artifacts/apa_sp5/registration.json,
sha 3cc3b3e112479ad71077a98a221fd5af521013d16ce423d52afe308e9ae58159):
wikitext-2-raw-v1 test, the model's own harmony/o200k tokenizer, no chat
template, W=1024, N=4, 2048 scored targets.

Prior art: SP3 PROTOCOL-2 / SP4G PROTOCOL-G windowed scoring (this repo,
2026); wikitext-2 (Merity et al., 2016). Arms and their sink handling are as
registered; see apa_sp5_model.ARMS. The single-pass rule itself is BLASST's
running-max criterion (Yuan et al., arXiv 2512.12087) applied to precision
rather than sparsity -- APA is David's design; the impossibility result for
the z-score rule is APA-SP1's.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, '/mnt/ForgeRealm/Project-Tensor-wt-apa-sp5/scripts')
from apa_sp5_model import load_runtime, port, gpu_mib, tokens, A, SNAP, install_sp


def score(tc, model, cfg, ids, lo, W, half):
    win = ids[lo:lo + W].reshape(1, W)
    with tc.no_grad():
        model.extend_rope(W + 8)
        h = model.embed_tokens(win)
        for lyr in model.layers:
            h, kv, _ = lyr(h, model.rope_cos, model.rope_sin, 0, None)
            del kv
        tc.synchronize()
        hn = model.norm(h).astype('bfloat16')
        del h; tc.empty_cache()
        parts = []
        for c0 in range(0, W, 256):
            c1 = min(c0 + 256, W)
            hc = hn.slice(1, c0, c1 - c0)
            lc = model.lm_head(hc)
            parts.append(lc.float().numpy().astype(np.float32))
            del lc, hc; tc.empty_cache()
        lg = np.concatenate(parts, axis=1).astype(np.float64).reshape(-1, cfg.vocab_size)
        del parts, hn; tc.empty_cache()
    tgt = ids[lo + 1:lo + W].astype(np.int64)
    r = lg[:W - 1][W - 1 - half:W - 1]
    t = tgt[W - 1 - half:W - 1]
    mx = r.max(axis=1, keepdims=True)
    lse = mx[:, 0] + np.log(np.exp(r - mx).sum(axis=1))
    nll = lse - r[np.arange(r.shape[0]), t]
    return nll


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=('A', 'B', 'C', 'D', 'E'))
    ap.add_argument('--W', type=int, default=1024)
    ap.add_argument('--N', type=int, default=4)
    ap.add_argument('--bulk-bits', type=int, default=4)
    ap.add_argument('--delta', type=float, default=None)
    ap.add_argument('--refine-percentile', type=float, default=0.15)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    DELTAS = {'C': 3.16, 'D': 1e9, 'E': 44.315}
    delta = args.delta if args.delta is not None else DELTAS.get(args.arm)
    tc = load_runtime(); G = port(tc)
    from core.mistral7b_tc import QuantLinearTC
    QuantLinearTC.USE_FUSED = True
    ids, meta = tokens()
    t0 = time.perf_counter()
    model, info = G.GptOss20B_TC.from_pretrained(SNAP)
    cfg = model.config
    load_s = time.perf_counter() - t0

    if args.arm == 'A':
        for lyr in model.layers:
            lyr.self_attn.attention_mode = 'standard'
    elif args.arm == 'B':
        for i, lyr in enumerate(model.layers):
            r = G.resolve_gpt_oss_attention_mode(cfg, i, 'apa_selective',
                                                 apa_layer_scope='full')
            lyr.self_attn.attention_mode = r['effective_attention_mode']
            lyr.self_attn.refine_percentile = args.refine_percentile
            lyr.self_attn.bulk_bits = args.bulk_bits
    else:
        install_sp(G, tc, delta)
        for i, lyr in enumerate(model.layers):
            lyr.self_attn.attention_mode = 'standard'
            lyr.self_attn.bulk_bits = args.bulk_bits
            lyr.self_attn._sp_delta = delta if cfg.is_full_attention(i) else None

    W, N, half = args.W, args.N, args.W // 2
    rec = dict(arm=args.arm, W=W, N=N, half=half, delta=delta,
               bulk_bits=args.bulk_bits,
               refine_percentile=args.refine_percentile,
               load_wall_s=load_s, token_sha256=meta['token_sha256'],
               registration_sha256='3cc3b3e112479ad71077a98a221fd5af521013d16ce423d52afe308e9ae58159',
               windows=[])
    allnll = []
    for w in range(N):
        t0 = time.perf_counter()
        nll = score(tc, model, cfg, ids, w * W, W, half)
        allnll.append(nll)
        rec['windows'].append(dict(window=w, lo=w * W, targets=int(nll.size),
                                   mean_nll=float(nll.mean()),
                                   ppl=float(np.exp(nll.mean())),
                                   wall_s=time.perf_counter() - t0,
                                   gpu_mib=gpu_mib()))
        tc.empty_cache()
    cat = np.concatenate(allnll)
    rec['total_targets'] = int(cat.size)
    rec['mean_nll'] = float(cat.mean())
    rec['ppl'] = float(np.exp(cat.mean()))
    rec['backend_full'] = next(l.self_attn.last_attention_backend
                               for i, l in enumerate(model.layers)
                               if cfg.is_full_attention(i))
    rec['backend_sliding'] = next(l.self_attn.last_attention_backend
                                  for i, l in enumerate(model.layers)
                                  if cfg.is_sliding_attention(i))
    rec['peak_gpu_mib'] = gpu_mib()
    out = args.out or (A / f'ppl_{args.arm}_W{W}_N{N}_b{args.bulk_bits}.json')
    Path(out).write_text(json.dumps(rec, indent=2) + '\n')
    slim = {k: rec[k] for k in ('arm', 'W', 'N', 'delta', 'bulk_bits',
                                'total_targets', 'mean_nll', 'ppl',
                                'backend_full', 'backend_sliding',
                                'peak_gpu_mib', 'load_wall_s')}
    slim['per_window_ppl'] = [round(w['ppl'], 3) for w in rec['windows']]
    slim['artifact'] = str(out)
    print(json.dumps(slim, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
