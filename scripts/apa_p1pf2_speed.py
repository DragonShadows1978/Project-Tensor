#!/usr/bin/env python3
"""APA-P1-PF2 speed line (informative, not gated), Build PF, inside forge-gpu-run.

Cells (REGISTRATION.json speed): geometries H16/KV8 and H32/KV8, D 128, causal; (S, L) in {(8192, 8192), (32768,
2048)}. Arms, attention call only (the key quantization is done once before timing):
  packed : apa_selective_attention_sp on quantize_sp2_keys(k, 4) (default packed: the new prefill route), fp32
  dense  : apa_selective_attention_sp on quantize_sp2_keys(k, 4, storage='dense') (dense SP prefill), fp32
  flash  : torch SDPA, FLASH_ATTENTION backend, bf16 copies of the same q/k/v, enable_gqa, bottom-right causal via
           the flash op when L < S (as P1-Q's FLASH arm)
Median of 5 timed calls (CUDA events) after 2 warm-ups; reported as multiples of flash.
Prior art: FlashAttention-2 (Dao, 2023) as the reference; CUDA event timing. No new technique.
"""
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import apa_p1pf2_gpu as G  # noqa: E402


def main():
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import sdpa_kernel, SDPBackend
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    tc, arm = G.load_arm('PF')
    os.environ['TC_APA_SP'] = '1'
    os.environ.pop('TC_APA_SP_STORAGE', None)
    dm = tc.RegisteredMargins.load(G.DENSE_TABLE, expected_sha256=G.DENSE_TABLE_SHA)
    pm = tc.PackedRegisteredMargins.load(G.PACKED_TABLE)
    flash_op = torch.ops.aten._scaled_dot_product_flash_attention
    rows = []

    def timed(fn):
        for _ in range(2):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(5):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            fn()
            b.record()
            torch.cuda.synchronize()
            ts.append(a.elapsed_time(b))
        return float(np.median(ts)), ts

    with tc.no_grad(), torch.inference_mode():
        for H in (16, 32):
            for S, L in ((8192, 8192), (32768, 2048)):
                g = torch.Generator(device='cuda').manual_seed(20261116 + H + S)
                q = (3 * torch.randn((1, H, L, 128), device='cuda', generator=g)).contiguous()
                k = torch.randn((1, 8, S, 128), device='cuda', generator=g).contiguous()
                v = torch.randn((1, 8, S, 128), device='cuda', generator=g).contiguous()
                Q, K, V = tc.from_dlpack(q), tc.from_dlpack(k), tc.from_dlpack(v)
                scale = float(np.float32(1 / math.sqrt(128)))
                kp = tc.quantize_sp2_keys(K, 4)
                kd = tc.quantize_sp2_keys(K, 4, storage='dense')
                t_p, ts_p = timed(lambda: tc.apa_selective_attention_sp(Q, K, kp, V, scale, .01, bulk_bits=4,
                                                                         margins=pm, is_causal=True))
                route = tc.apa_packed.last_route('attention')
                t_d, ts_d = timed(lambda: tc.apa_selective_attention_sp(Q, K, kd, V, scale, .01, bulk_bits=4,
                                                                         margins=dm, is_causal=True))
                qb, kb, vb = q.bfloat16(), k.bfloat16(), v.bfloat16()
                kb2, vb2 = kb.repeat_interleave(H // 8, 1), vb.repeat_interleave(H // 8, 1)
                with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
                    if L == S:
                        fn = lambda: F.scaled_dot_product_attention(qb, kb, vb, is_causal=True, scale=scale,  # noqa
                                                                    enable_gqa=True)
                    else:
                        fn = lambda: flash_op(qb, kb2, vb2, 0.0, is_causal=True, return_debug_mask=False,  # noqa
                                              scale=scale)
                    t_f, ts_f = timed(fn)
                rows.append(dict(H=H, KV=8, D=128, S=S, L=L, ms=dict(packed=t_p, dense=t_d, flash=t_f),
                                 samples=dict(packed=ts_p, dense=ts_d, flash=ts_f),
                                 x_flash=dict(packed=t_p / t_f, dense=t_d / t_f), packed_over_dense=t_p / t_d,
                                 packed_route=route['backend']))
                print('SPEED', json.dumps(rows[-1]['x_flash']), H, S, L, flush=True)
                del kp, kd, Q, K, V, q, k, v, qb, kb, vb, kb2, vb2
                torch.cuda.empty_cache()
                tc.empty_cache()
    G.save(out / 'RESULT.json', dict(line='SPEED', informative=True, arm=arm, device=torch.cuda.get_device_name(0),
                                     rows=rows, terminal='SPEED_DONE'))
    return 0


if __name__ == '__main__':
    assert os.environ.get('CUDA_VISIBLE_DEVICES') != '', 'GPU modes run inside forge-gpu-run'
    raise SystemExit(main())
