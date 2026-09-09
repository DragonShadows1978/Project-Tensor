#!/usr/bin/env python3
"""APA-SP5 amendment 1 -- ONE PROTOCOL-O window as ONE cell.

Cell id: ppl_{arm}_W1024_w{NN}   (arm in A,B,C,D,E,A32)
A32 = standard with the 12 full-attention layers in fp32 -- the per-window
sensitivity floor partner for that window.

Lead ruling (amendment 1): a window is a cell. Each cell is one model load
(~12.5 s warm) plus one ~34 s window, so it sits far under the 285 s worker
rail even when the load is cold (37 s).

Prior art: SP3 PROTOCOL-2 / SP4G PROTOCOL-G windowed scoring (this repo,
2026); wikitext-2 (Merity et al., 2016); SP4G amendment 6 for the bf16-vs-fp32
floor construction. The single-pass rule is BLASST's running-max criterion
(Yuan et al., arXiv 2512.12087) applied to precision rather than sparsity --
APA is David's design, the single pass is APA-SP1's. Nothing new here.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, '/mnt/ForgeRealm/Project-Tensor-wt-apa-sp5/scripts')
from apa_sp5_model import load_runtime, port, gpu_mib, tokens, A, SNAP, install_sp

REG_SHA = '3cc3b3e112479ad71077a98a221fd5af521013d16ce423d52afe308e9ae58159'
DELTAS = {'C': 3.16, 'D': 1e9, 'E': 44.315}


def free_mib(rt):
    import ctypes
    f = ctypes.c_size_t(); t = ctypes.c_size_t()
    rt.cudaMemGetInfo(ctypes.byref(f), ctypes.byref(t))
    return f.value // (1 << 20), t.value // (1 << 20)


def install_fp32_full(G, tc):
    """A32: full-attention layers in fp32, chunked over query positions.
    Arithmetically identical to one dense fp32 call (per-row softmax is
    independent); required because a dense fp32 64x1024x1024 score tensor is
    256 MiB and sink_attention_tc holds several at once."""
    import os
    original = G.GptOssAttentionTC.__call__

    def patched(self, x, cos, sin, position_offset=0, kv_cache=None):
        if not getattr(self, '_fp32_full', False):
            return original(self, x, cos, sin, position_offset, kv_cache)
        B, L, _ = x.shape
        q = self.q_proj(x).reshape([B, L, self.num_heads, self.head_dim]).transpose(1, 2)
        k = self.k_proj(x).reshape([B, L, self.num_kv_heads, self.head_dim]).transpose(1, 2)
        v = self.v_proj(x).reshape([B, L, self.num_kv_heads, self.head_dim]).transpose(1, 2)
        cs = cos.slice(0, position_offset, L); ss = sin.slice(0, position_offset, L)
        q = G.F.apply_rotary(q, cs, ss); k = G.F.apply_rotary(k, cs, ss)
        if kv_cache is not None:
            k = tc.cat([kv_cache[0], k], dim=2); v = tc.cat([kv_cache[1], v], dim=2)
        S = k.shape[2]
        q32, k32, v32 = q.astype('float32'), k.astype('float32'), v.astype('float32')
        s32 = self.sinks.astype('float32')
        blk = 128
        outs = []
        for i0 in range(0, L, blk):
            n = min(blk, L - i0)
            lim = S - L + i0 + n
            oi = G.sink_attention_tc(
                q32.slice(2, i0, n), k32.slice(2, 0, lim), v32.slice(2, 0, lim),
                s32, scale=self.scaling,
                attention_mask=G._gpt_oss_attention_mask(n, lim, sliding_window=None,
                                                         dtype='float32'),
                num_heads_per_kv=self.num_heads_per_kv)
            outs.append(oi.astype(x.dtype)); del oi
            tc.empty_cache()
        attn = tc.cat(outs, dim=2)
        del outs, q32, k32, v32
        tc.empty_cache()
        self.last_attention_backend = 'standard_sink_fp32'
        out = attn.transpose(1, 2).reshape([B, L, self.num_heads * self.head_dim])
        return self.o_proj(out), (k, v)

    G.GptOssAttentionTC.__call__ = patched
    return original


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=('A', 'B', 'C', 'D', 'E', 'A32'))
    ap.add_argument('--window', type=int, required=True)
    ap.add_argument('--W', type=int, default=1024)
    ap.add_argument('--bulk-bits', type=int, default=4)
    ap.add_argument('--delta', type=float, default=None)
    ap.add_argument('--refine-percentile', type=float, default=0.15)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    import ctypes
    rt = ctypes.CDLL('libcudart.so')
    cell = f'ppl_{args.arm}_W{args.W}_w{args.window:02d}'
    out = Path(args.out) if args.out else (A / 'windows_a1' / f'{cell}.json')
    if out.exists():
        print(json.dumps(dict(cell=cell, status='DONE_ALREADY', artifact=str(out))))
        return 0
    delta = args.delta if args.delta is not None else DELTAS.get(args.arm)

    t_all = time.perf_counter()
    tc = load_runtime(); G = port(tc)
    from core.mistral7b_tc import QuantLinearTC
    QuantLinearTC.USE_FUSED = True
    ids, meta = tokens()
    W = args.W; half = W // 2; lo = args.window * W
    if lo + W > ids.size:
        raise SystemExit(f'window {args.window} exceeds the token stream')

    t0 = time.perf_counter()
    model, info = G.GptOss20B_TC.from_pretrained(SNAP)
    cfg = model.config
    load_s = time.perf_counter() - t0

    if args.arm == 'A':
        for l in model.layers:
            l.self_attn.attention_mode = 'standard'
    elif args.arm == 'A32':
        install_fp32_full(G, tc)
        for i, l in enumerate(model.layers):
            l.self_attn.attention_mode = 'standard'
            l.self_attn._fp32_full = bool(cfg.is_full_attention(i))
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

    t0 = time.perf_counter()
    win = ids[lo:lo + W].reshape(1, W)
    with tc.no_grad():
        model.extend_rope(W + 8)
        h = model.embed_tokens(win)
        for l in model.layers:
            h, kv, _ = l(h, model.rope_cos, model.rope_sin, 0, None)
            del kv
        tc.synchronize()
        peak_free = free_mib(rt)[0]
        hn = model.norm(h).astype('bfloat16'); del h; tc.empty_cache()
        parts = []
        for c0 in range(0, W, 256):
            c1 = min(c0 + 256, W)
            hc = hn.slice(1, c0, c1 - c0)
            lc = model.lm_head(hc)
            parts.append(lc.float().numpy().astype(np.float32))
            del lc, hc; tc.empty_cache()
        lg = np.concatenate(parts, axis=1).astype(np.float64).reshape(-1, cfg.vocab_size)
        del parts, hn; tc.empty_cache()
    fwd_s = time.perf_counter() - t0

    tgt = ids[lo + 1:lo + W].astype(np.int64)
    r = lg[:W - 1][W - 1 - half:W - 1]
    t = tgt[W - 1 - half:W - 1]
    mx = r.max(axis=1, keepdims=True)
    lse = mx[:, 0] + np.log(np.exp(r - mx).sum(axis=1))
    nll = lse - r[np.arange(r.shape[0]), t]

    rec = dict(cell=cell, arm=args.arm, window=args.window, lo=lo, W=W,
               half=half, delta=delta, bulk_bits=args.bulk_bits,
               refine_percentile=args.refine_percentile,
               registration_sha256=REG_SHA,
               amendment='APA_SP5_AMENDMENT_1',
               amendment_sha256='06a5a1e742ae2474469f8d5177bc5b9d35b4d89e9553d928c9cac1d7a6753e33',
               token_sha256=meta['token_sha256'],
               targets=int(nll.size),
               mean_nll=float(nll.mean()), ppl=float(np.exp(nll.mean())),
               nll_sum=float(nll.sum()),
               backend_full=next(l.self_attn.last_attention_backend
                                 for i, l in enumerate(model.layers)
                                 if cfg.is_full_attention(i)),
               backend_sliding=next(l.self_attn.last_attention_backend
                                    for i, l in enumerate(model.layers)
                                    if cfg.is_sliding_attention(i)),
               load_wall_s=load_s, forward_wall_s=fwd_s,
               total_wall_s=time.perf_counter() - t_all,
               gpu_mib=gpu_mib(), free_mib_after_layers=peak_free,
               status='PASS')
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as f:
        json.dump(rec, f, indent=2, allow_nan=False); f.write('\n')
    rec['artifact'] = str(out)
    print(json.dumps({k: rec[k] for k in
                      ('cell', 'arm', 'window', 'ppl', 'mean_nll', 'targets',
                       'backend_full', 'load_wall_s', 'forward_wall_s',
                       'total_wall_s', 'free_mib_after_layers', 'artifact')},
                     indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
