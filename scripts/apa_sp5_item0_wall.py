#!/usr/bin/env python3
"""APA-SP5 ITEM 0 -- cost of one window on the card, BEFORE registering
PROTOCOL-O. Everything downstream is sized from this number.

Measures, for a given W (window tokens) and a given arm:
  - model load wall + resident VRAM (resident GptOss20B_TC path)
  - one prefill forward of W tokens: wall, peak VRAM, per-layer breakdown

Prior art: SP4G item-0 equivalent sizing (this repo, 2026); the port's own
streamed smoke (GraftRepository, 2026). No new algorithm.
"""
import argparse, gc, json, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, '/mnt/ForgeRealm/Project-Tensor-wt-apa-sp5/scripts')
from apa_sp5_model import load_runtime, port, gpu_mib, tokens, A, SNAP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--W', type=int, default=1024)
    ap.add_argument('--arm', default='A', choices=('A', 'B', 'C'))
    ap.add_argument('--delta', type=float, default=3.0)
    ap.add_argument('--refine-percentile', type=float, default=0.15)
    ap.add_argument('--bulk-bits', type=int, default=4)
    ap.add_argument('--out', default=None)
    ap.add_argument('--layers', type=int, default=None)
    args = ap.parse_args()

    t_start = time.perf_counter()
    rec = dict(W=args.W, arm=args.arm, delta=args.delta,
               bulk_bits=args.bulk_bits,
               refine_percentile=args.refine_percentile,
               gpu_mib_before=gpu_mib())
    tc = load_runtime()
    G = port(tc)
    from core.mistral7b_tc import QuantLinearTC
    # FINDING (item 0, receipted): the resident GPT-OSS-20B body leaves only
    # ~927 MiB free on this 12 GB card, and int4_linear's DEFAULT two-stage
    # path dequantizes the whole lm_head weight to a (2880 x 201088) fp16
    # buffer = 1,105 MiB -- it OOMs at EVERY chunk size, including M=1.
    # int4_linear_fused never materializes that buffer. Same arithmetic
    # (kernels.cu: "Correctness-identical to int4_linear"); opt-in flag only.
    # This is a harness/VRAM choice, not a model or kernel change.
    QuantLinearTC.USE_FUSED = True

    ids, meta = tokens()
    rec['token_meta'] = {k: meta[k] for k in ('token_count', 'token_sha256')}

    t0 = time.perf_counter()
    model, info = G.GptOss20B_TC.from_pretrained(SNAP)
    tc.synchronize()
    rec['load_wall_s'] = time.perf_counter() - t0
    rec['gpu_mib_after_load'] = gpu_mib()
    rec['model_info'] = {k: info[k] for k in
                         ('layers', 'full_attention_layers',
                          'sliding_attention_layers', 'expert_mode')}
    cfg = model.config

    # arm wiring
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
        from apa_sp5_model import install_sp
        install_sp(G, tc, args.delta)
        for i, lyr in enumerate(model.layers):
            lyr.self_attn.attention_mode = 'standard'
            lyr.self_attn.bulk_bits = args.bulk_bits
            if cfg.is_full_attention(i):
                lyr.self_attn._sp_delta = args.delta
            else:
                lyr.self_attn._sp_delta = None

    W = args.W
    win = ids[:W].reshape(1, W)
    nl = args.layers if args.layers else cfg.num_layers

    per_layer = []
    with tc.no_grad():
        model.extend_rope(W + 8)
        h = model.embed_tokens(win)
        t1 = time.perf_counter()
        peak = rec['gpu_mib_after_load']
        for i, lyr in enumerate(model.layers[:nl]):
            ts = time.perf_counter()
            h, kv, _ = lyr(h, model.rope_cos, model.rope_sin, 0, None)
            tc.synchronize()
            used = gpu_mib()
            peak = max(peak, used)
            per_layer.append(dict(layer=i, type=cfg.layer_types[i],
                                  backend=lyr.self_attn.last_attention_backend,
                                  wall_s=time.perf_counter() - ts,
                                  gpu_mib=used))
            del kv
            tc.empty_cache()
        tc.synchronize()
        rec['layers_wall_s'] = time.perf_counter() - t1
        if nl == cfg.num_layers:
            t2 = time.perf_counter()
            hn = model.norm(h)
            hn = hn.astype('bfloat16')
            # The lm_head over W positions x 201,088 vocab does not fit next
            # to the resident MXFP4 body on 12 GB (measured: OOM at W=1024,
            # ~824 MiB of fp32 logits alone against ~1.35 GB headroom).
            # Chunk the head over query positions; identical arithmetic.
            rec['gpu_mib_before_head'] = gpu_mib()
            tc.empty_cache()
            rec['gpu_mib_before_head_after_empty'] = gpu_mib()
            chunk = int(os.environ.get('SP5_HEAD_CHUNK', '256'))
            parts = []
            for c0 in range(0, W, chunk):
                c1 = min(c0 + chunk, W)
                hc = hn.slice(1, c0, c1 - c0)
                lc = model.lm_head(hc)
                parts.append(lc.float().numpy().astype(np.float32))
                del lc, hc
                tc.empty_cache()
            lg = np.concatenate(parts, axis=1).astype(np.float64).reshape(-1, cfg.vocab_size)
            del parts
            tc.synchronize()
            rec['head_wall_s'] = time.perf_counter() - t2
            rec['head_chunk'] = chunk
            peak = max(peak, gpu_mib())
            # score last W/2 in fp64
            half = W // 2
            tgt = ids[1:W].astype(np.int64)
            rows = lg[:W - 1]
            sl = slice(W - 1 - half, W - 1)
            r = rows[sl]
            t = tgt[sl]
            mx = r.max(axis=1, keepdims=True)
            lse = (mx[:, 0] + np.log(np.exp(r - mx).sum(axis=1)))
            nll = lse - r[np.arange(r.shape[0]), t]
            rec['scored_targets'] = int(r.shape[0])
            rec['mean_nll'] = float(nll.mean())
            rec['ppl'] = float(np.exp(nll.mean()))
    rec['peak_gpu_mib'] = peak
    rec['per_layer'] = per_layer
    rec['forward_wall_s'] = rec['layers_wall_s'] + rec.get('head_wall_s', 0.0)
    rec['total_wall_s'] = time.perf_counter() - t_start
    rec['full_layer_wall_s'] = sum(x['wall_s'] for x in per_layer
                                   if x['type'] == 'full_attention')
    rec['sliding_layer_wall_s'] = sum(x['wall_s'] for x in per_layer
                                      if x['type'] == 'sliding_attention')
    out = args.out or (A / f'item0_W{args.W}_{args.arm}.json')
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(rec, indent=2) + '\n')
    slim = {k: rec[k] for k in ('W', 'arm', 'load_wall_s', 'layers_wall_s',
                                'forward_wall_s', 'total_wall_s',
                                'full_layer_wall_s', 'sliding_layer_wall_s',
                                'gpu_mib_after_load', 'peak_gpu_mib')
            if k in rec}
    slim.update({k: rec[k] for k in ('ppl', 'mean_nll', 'scored_targets') if k in rec})
    slim['artifact'] = str(out)
    print(json.dumps(slim, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
