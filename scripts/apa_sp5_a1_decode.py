#!/usr/bin/env python3
"""APA-SP5 amendment 1 item 5 -- clean decode, 32 synced steps, ms/token.

Cell id: decode_{A,B,C}_2048.

REPORTED AS AN ADAPTER COST, NOT A KERNEL COST. The cited line:

    core/gpt_oss20b_tc.py:741      kq = _quantize_keys(k, R, CB, BND)

sits inside GptOssAttentionTC.__call__ AFTER `k` has been concatenated with
the whole kv_cache (line ~717: `k = tc.cat([kv_cache[0], k], dim=2)`), so on
every decode step the adapter re-quantizes the ENTIRE key history rather than
just the one new row. Nothing in the port caches quantized keys (verified:
`kq` appears only at :741 and its two use sites :747/:763). Standard does no
quantization at all, so it pays none of this. Any APA-vs-standard decode gap
measured here is therefore dominated by an adapter design choice, exactly as
SP3 found on MiniCPM3 (7-14x, fixed on Gemma by caching quantized keys).
No fix and no patch is applied here; the amendment says measure and cite.

Prior art: SP3 a6 clean-decode bisect and SP4G a6 decode cells (this repo,
2026) -- same construction (June flags, incremental kv_cache, synced steps).
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, '/mnt/ForgeRealm/Project-Tensor-wt-apa-sp5/scripts')
from apa_sp5_model import load_runtime, port, gpu_mib, tokens, A, SNAP, install_sp

REG = '3cc3b3e112479ad71077a98a221fd5af521013d16ce423d52afe308e9ae58159'
AMD = '06a5a1e742ae2474469f8d5177bc5b9d35b4d89e9553d928c9cac1d7a6753e33'
DELTAS = {'C': 3.16, 'D': 1e9, 'E': 44.315}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=('A', 'B', 'C'))
    ap.add_argument('--S', type=int, default=2048)
    ap.add_argument('--steps', type=int, default=32)
    ap.add_argument('--bulk-bits', type=int, default=4)
    ap.add_argument('--delta', type=float, default=None)
    ap.add_argument('--refine-percentile', type=float, default=0.15)
    ap.add_argument('--prefill-chunk', type=int, default=1024)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    cell = f'decode_{args.arm}_{args.S}'
    out = Path(args.out) if args.out else (A / 'decode_a1' / f'{cell}.json')
    if out.exists():
        print(json.dumps(dict(cell=cell, status='DONE_ALREADY', artifact=str(out))))
        return 0
    delta = args.delta if args.delta is not None else DELTAS.get(args.arm)

    tc = load_runtime(); G = port(tc)
    from core.mistral7b_tc import QuantLinearTC
    QuantLinearTC.USE_FUSED = True
    ids, meta = tokens()
    model, info = G.GptOss20B_TC.from_pretrained(SNAP)
    cfg = model.config

    if args.arm == 'A':
        for l in model.layers:
            l.self_attn.attention_mode = 'standard'
    elif args.arm == 'B':
        for i, l in enumerate(model.layers):
            r = G.resolve_gpt_oss_attention_mode(cfg, i, 'apa_selective',
                                                 apa_layer_scope='full')
            l.self_attn.attention_mode = r['effective_attention_mode']
            l.self_attn.refine_percentile = args.refine_percentile
            l.self_attn.bulk_bits = args.bulk_bits
    else:
        install_sp(G, tc, delta)
        for i, l in enumerate(model.layers):
            l.self_attn.attention_mode = 'standard'
            l.self_attn.bulk_bits = args.bulk_bits
            l.self_attn._sp_delta = delta if cfg.is_full_attention(i) else None

    S = args.S
    rec = dict(cell=cell, arm=args.arm, S=S, steps=args.steps, delta=delta,
               bulk_bits=args.bulk_bits, registration_sha256=REG,
               amendment='APA_SP5_AMENDMENT_1', amendment_sha256=AMD,
               token_sha256=meta['token_sha256'],
               adapter_cost_citation=dict(
                 file='/mnt/ForgeRealm/GraftRepository/core/gpt_oss20b_tc.py',
                 line=741, code='kq = _quantize_keys(k, R, CB, BND)',
                 why=('k at this point is the WHOLE concatenated key history '
                      '(kv_cache + the one new row), so every decode step '
                      're-quantizes the entire cache; nothing in the port '
                      'caches quantized keys. Standard does no quantization '
                      'and pays none of this. The gap below is therefore an '
                      'ADAPTER cost, not a kernel cost.')))
    try:
        with tc.no_grad():
            model.extend_rope(S + args.steps + 8)
            # chunked prefill to S, keeping the kv cache
            caches = None
            pos = 0
            t0 = time.perf_counter()
            while pos < S:
                n = min(args.prefill_chunk, S - pos)
                chunk = ids[pos:pos + n].reshape(1, n)
                _, caches, _ = model(chunk, kv_caches=caches,
                                     position_offset=pos, max_layers=cfg.num_layers)
                pos += n
            tc.synchronize()
            rec['prefill_wall_s'] = time.perf_counter() - t0
            rec['gpu_mib_after_prefill'] = gpu_mib()
            # warm one step
            nxt = ids[S:S + 1].reshape(1, 1)
            _, caches, _ = model(nxt, kv_caches=caches, position_offset=pos,
                                 max_layers=cfg.num_layers)
            tc.synchronize()
            pos += 1
            per = []
            for st in range(args.steps):
                tok = ids[S + 1 + st:S + 2 + st].reshape(1, 1)
                t1 = time.perf_counter()
                _, caches, _ = model(tok, kv_caches=caches, position_offset=pos,
                                     max_layers=cfg.num_layers)
                tc.synchronize()
                per.append((time.perf_counter() - t1) * 1000.0)
                pos += 1
            rec.update(status='PASS', steps_ms=per,
                       ms_per_token=float(np.mean(per)),
                       ms_per_token_median=float(np.median(per)),
                       ms_per_token_min=float(np.min(per)),
                       backend_full=next(l.self_attn.last_attention_backend
                                         for i, l in enumerate(model.layers)
                                         if cfg.is_full_attention(i)),
                       gpu_mib=gpu_mib())
    except Exception as e:
        rec.update(status='RED', error_type=type(e).__name__, error=str(e)[:400],
                   gpu_mib=gpu_mib())
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as f:
        json.dump(rec, f, indent=2, allow_nan=False); f.write('\n')
    rec['artifact'] = str(out)
    print(json.dumps({k: rec[k] for k in
                      ('cell', 'arm', 'S', 'status', 'ms_per_token',
                       'ms_per_token_median', 'prefill_wall_s',
                       'backend_full', 'gpu_mib', 'error', 'artifact')
                      if k in rec}, indent=2))
    return 0 if rec['status'] == 'PASS' else 3


if __name__ == '__main__':
    raise SystemExit(main())
