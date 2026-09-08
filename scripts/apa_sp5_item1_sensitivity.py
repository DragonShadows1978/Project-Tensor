#!/usr/bin/env python3
"""APA-SP5 ITEM 1 -- the sensitivity floor (SP4G principle 3).

Standard in bf16 vs standard with the FULL-ATTENTION layers computed in
fp32, on window 0. The resulting ppl difference IS the noise floor for
every model-level comparison in this order; differences inside it are
"not resolvable on this model".

Prior art: SP4G amendment 6 (this repo, 2026) -- which established the
principle and the A-bf16-vs-A-fp32 construction after Gemma 4 amplified
rounding-level perturbations ~30x; Higham & Mary (2022) for mixed-precision
error context. Nothing new algorithmically; the fp32 promotion site (q/k/v
in, bf16 back out before o_proj) is transcribed from SP4G's PrecisionModel.
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, '/mnt/ForgeRealm/Project-Tensor-wt-apa-sp5/scripts')
from apa_sp5_model import load_runtime, port, gpu_mib, tokens, A, SNAP


def score_window(tc, G, model, cfg, ids, lo, W, half):
    win = ids[lo:lo + W].reshape(1, W)
    with tc.no_grad():
        model.extend_rope(W + 8)
        h = model.embed_tokens(win)
        for lyr in model.layers:
            h, kv, _ = lyr(h, model.rope_cos, model.rope_sin, 0, None)
            del kv
        tc.synchronize()
        hn = model.norm(h).astype('bfloat16')
        del h
        tc.empty_cache()
        parts = []
        for c0 in range(0, W, 256):
            c1 = min(c0 + 256, W)
            hc = hn.slice(1, c0, c1 - c0)
            lc = model.lm_head(hc)
            parts.append(lc.float().numpy().astype(np.float32))
            del lc, hc
            tc.empty_cache()
        lg = np.concatenate(parts, axis=1).astype(np.float64).reshape(-1, cfg.vocab_size)
        del parts, hn
        tc.empty_cache()
    tgt = ids[lo + 1:lo + W].astype(np.int64)
    rows = lg[:W - 1]
    sl = slice(W - 1 - half, W - 1)
    r, t = rows[sl], tgt[sl]
    mx = r.max(axis=1, keepdims=True)
    lse = mx[:, 0] + np.log(np.exp(r - mx).sum(axis=1))
    nll = lse - r[np.arange(r.shape[0]), t]
    return float(nll.mean()), float(np.exp(nll.mean())), int(r.shape[0])


def install_fp32_full(G, tc, cfg):
    """Promote FULL-attention layers' attention math to fp32; return to
    bf16 before o_proj (SP4G PrecisionModel convention)."""
    original = G.GptOssAttentionTC.__call__

    def patched(self, x, cos, sin, position_offset=0, kv_cache=None):
        if not getattr(self, '_fp32_full', False):
            return original(self, x, cos, sin, position_offset, kv_cache)
        B, L, _ = x.shape
        q = self.q_proj(x).reshape([B, L, self.num_heads, self.head_dim]).transpose(1, 2)
        k = self.k_proj(x).reshape([B, L, self.num_kv_heads, self.head_dim]).transpose(1, 2)
        v = self.v_proj(x).reshape([B, L, self.num_kv_heads, self.head_dim]).transpose(1, 2)
        cseg = cos.slice(0, position_offset, L)
        sseg = sin.slice(0, position_offset, L)
        q = G.F.apply_rotary(q, cseg, sseg)
        k = G.F.apply_rotary(k, cseg, sseg)
        if kv_cache is not None:
            k = tc.cat([kv_cache[0], k], dim=2)
            v = tc.cat([kv_cache[1], v], dim=2)
        S = k.shape[2]
        q32, k32, v32 = q.astype('float32'), k.astype('float32'), v.astype('float32')
        s32 = self.sinks.astype('float32')
        # Chunk over QUERY positions. Each query row's softmax is independent,
        # so this is arithmetically identical to one dense call; it is required
        # because a full fp32 BxHxLxS score tensor is 256 MiB at L=S=1024 and
        # sink_attention_tc holds several at once (measured OOM at 927 MiB
        # free). Same convention as the port's own chunked sliding path.
        blk = int(os.environ.get('SP5_FP32_BLOCK', '128'))
        outs = []
        for i0 in range(0, L, blk):
            n = min(blk, L - i0)
            qi = q32.slice(2, i0, n)
            lim = S - L + i0 + n
            ki = k32.slice(2, 0, lim)
            vi = v32.slice(2, 0, lim)
            mi = G._gpt_oss_attention_mask(n, lim, sliding_window=None,
                                           dtype='float32')
            oi = G.sink_attention_tc(qi, ki, vi, s32, scale=self.scaling,
                                     attention_mask=mi,
                                     num_heads_per_kv=self.num_heads_per_kv)
            outs.append(oi.astype(x.dtype))
            del qi, ki, vi, mi, oi
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
    ap.add_argument('--W', type=int, default=1024)
    ap.add_argument('--window', type=int, default=0)
    ap.add_argument('--N', type=int, default=1)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    tc = load_runtime()
    G = port(tc)
    from core.mistral7b_tc import QuantLinearTC
    QuantLinearTC.USE_FUSED = True
    ids, meta = tokens()
    W = args.W
    half = W // 2
    lo = args.window * W
    t0 = time.perf_counter()
    model, info = G.GptOss20B_TC.from_pretrained(SNAP)
    cfg = model.config
    load_s = time.perf_counter() - t0

    rec = dict(W=W, window=args.window, lo=lo, half=half, load_wall_s=load_s,
               token_sha256=meta['token_sha256'],
               full_layers=cfg.full_attention_indices())

    # A: engine standard, bf16 everywhere
    for lyr in model.layers:
        lyr.self_attn.attention_mode = 'standard'
        lyr.self_attn._fp32_full = False
    t0 = time.perf_counter()
    accA = []
    for w in range(args.N):
        nl, _, n = score_window(tc, G, model, cfg, ids, (args.window + w) * W, W, half)
        accA.append((nl, n)); tc.empty_cache()
    nll_bf16 = float(np.average([a for a, _ in accA], weights=[b for _, b in accA]))
    ppl_bf16 = float(np.exp(nll_bf16))
    rec['A_bf16'] = dict(mean_nll=nll_bf16, ppl=ppl_bf16,
                         targets=sum(b for _, b in accA),
                         per_window_ppl=[float(np.exp(a)) for a, _ in accA],
                         wall_s=time.perf_counter() - t0)
    tc.empty_cache()

    # A32: full-attention layers in fp32
    install_fp32_full(G, tc, cfg)
    for i, lyr in enumerate(model.layers):
        lyr.self_attn._fp32_full = bool(cfg.is_full_attention(i))
    t0 = time.perf_counter()
    accB = []
    for w in range(args.N):
        nl, _, n = score_window(tc, G, model, cfg, ids, (args.window + w) * W, W, half)
        accB.append((nl, n)); tc.empty_cache()
    nll_fp32 = float(np.average([a for a, _ in accB], weights=[b for _, b in accB]))
    ppl_fp32 = float(np.exp(nll_fp32))
    rec['A32_fp32_full'] = dict(mean_nll=nll_fp32, ppl=ppl_fp32,
                                targets=sum(b for _, b in accB),
                                per_window_ppl=[float(np.exp(a)) for a, _ in accB],
                                wall_s=time.perf_counter() - t0)
    rec['peak_gpu_mib'] = gpu_mib()
    rec['floor_ppl'] = abs(ppl_bf16 - ppl_fp32)
    rec['floor_ppl_relative'] = abs(ppl_bf16 - ppl_fp32) / ppl_bf16
    rec['floor_nll'] = abs(nll_bf16 - nll_fp32)
    out = args.out or (A / f'item1_sensitivity_W{W}_win{args.window}_N{args.N}.json')
    Path(out).write_text(json.dumps(rec, indent=2) + '\n')
    rec['artifact'] = str(out)
    print(json.dumps({k: v for k, v in rec.items() if k != 'full_layers'}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
