#!/usr/bin/env python3
"""APA-SP5 amendment 1 item 6 -- prefill ceiling cells.

Cell id: ceiling_{B,C}_{4096,8192,16384}. Arm A is a REGISTERED NON-FIT
already at 2048 (r1 receipt: item0_W2048_A OOM, because the port's
sink_attention_tc materializes a dense BxHxLxS score tensor -- 512 MiB at
S=2048 and 2,048 MiB at S=4096 against ~927 MiB free); that receipt is kept
and A is not re-run here.

Reports peak VRAM per rung from cudaMemGetInfo (free/total), which is the
honest number: nvidia-smi reports process usage and misses the driver
reserve, and this card has only ~927 MiB of headroom after the resident
MXFP4 body loads.

Distinguishes OOM (memory wall) from RAIL (time wall) -- both are results.

Prior art: SP4G a7 ceiling cells (this repo, 2026), ascending-to-first-failure
construction. No new algorithm.
"""
import argparse, ctypes, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, '/mnt/ForgeRealm/Project-Tensor-wt-apa-sp5/scripts')
from apa_sp5_model import load_runtime, port, gpu_mib, tokens, A, SNAP, install_sp

REG = '3cc3b3e112479ad71077a98a221fd5af521013d16ce423d52afe308e9ae58159'
AMD = '06a5a1e742ae2474469f8d5177bc5b9d35b4d89e9553d928c9cac1d7a6753e33'
DELTAS = {'C': 3.16, 'D': 1e9, 'E': 44.315}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=('A', 'B', 'C', 'D', 'E'))
    ap.add_argument('--S', type=int, required=True)
    ap.add_argument('--bulk-bits', type=int, default=4)
    ap.add_argument('--delta', type=float, default=None)
    ap.add_argument('--refine-percentile', type=float, default=0.15)
    ap.add_argument('--chunk', type=int, default=1024)
    ap.add_argument('--budget-s', type=float, default=250.0)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    cell = f'ceiling_{args.arm}_{args.S}'
    out = Path(args.out) if args.out else (A / 'ceiling_a1' / f'{cell}.json')
    if out.exists():
        print(json.dumps(dict(cell=cell, status='DONE_ALREADY', artifact=str(out))))
        return 0
    delta = args.delta if args.delta is not None else DELTAS.get(args.arm)
    rt = ctypes.CDLL('libcudart.so')

    def freemib():
        f = ctypes.c_size_t(); t = ctypes.c_size_t()
        rt.cudaMemGetInfo(ctypes.byref(f), ctypes.byref(t))
        return f.value // (1 << 20), t.value // (1 << 20)

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
    free0, total = freemib()
    rec = dict(cell=cell, arm=args.arm, S=S, chunk=args.chunk, delta=delta,
               bulk_bits=args.bulk_bits, registration_sha256=REG,
               amendment='APA_SP5_AMENDMENT_1', amendment_sha256=AMD,
               token_sha256=meta['token_sha256'], budget_s=args.budget_s,
               free_mib_after_load=free0, total_mib=total,
               gpu_mib_after_load=gpu_mib())
    min_free = free0
    t0 = time.perf_counter()
    pos = 0
    trace = []
    try:
        with tc.no_grad():
            model.extend_rope(S + 8)
            caches = None
            while pos < S:
                n = min(args.chunk, S - pos)
                chunk = ids[pos:pos + n].reshape(1, n)
                _, caches, _ = model(chunk, kv_caches=caches,
                                     position_offset=pos, max_layers=cfg.num_layers)
                tc.synchronize()
                pos += n
                fnow = freemib()[0]
                min_free = min(min_free, fnow)
                # amendment 2: the completed-token trace IS the measurement --
                # an OOM must report exactly how far the arm got.
                trace.append(dict(tokens=pos, free_mib=fnow,
                                  elapsed_s=round(time.perf_counter() - t0, 2)))
                el = time.perf_counter() - t0
                if el > args.budget_s and pos < S:
                    rec.update(status='RAIL', outcome='RAIL',
                               reached_tokens=pos, prefill_wall_s=el,
                               note=('exceeded the in-cell time budget before '
                                     'reaching S; TIME wall, not memory'))
                    break
            else:
                el = time.perf_counter() - t0
                rec.update(status='PASS', outcome='FITS', reached_tokens=S,
                           prefill_wall_s=el)
    except Exception as e:
        msg = str(e)
        oom = 'out of memory' in msg.lower()
        rec.update(status='RED', outcome='OOM' if oom else 'ERROR',
                   error_type=type(e).__name__, error=msg[:400],
                   completed_tokens_at_failure=pos,
                   failed_on_chunk_starting_at=pos,
                   free_mib_before_failing_chunk=(trace[-1]['free_mib'] if trace else free0),
                   prefill_wall_s=time.perf_counter() - t0)
    rec['token_trace'] = trace
    rec['min_free_mib'] = min_free
    rec['peak_used_mib'] = total - min_free
    rec['gpu_mib_end'] = gpu_mib()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as f:
        json.dump(rec, f, indent=2, allow_nan=False); f.write('\n')
    rec['artifact'] = str(out)
    print(json.dumps({k: rec[k] for k in
                      ('cell', 'arm', 'S', 'status', 'outcome',
                       'reached_tokens', 'prefill_wall_s', 'min_free_mib',
                       'peak_used_mib', 'error', 'artifact') if k in rec},
                     indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
