#!/usr/bin/env python3
"""MiniCPM3 experiment seam, preserving the read-only adapter and weight path.

Prior art: MiniCPM Team (2024) MLA adapter and Perry APA draft (2026) are reused.
Arm A is ordinary scaled dot-product attention (Vaswani et al., 2017);
it does not call an external FlashAttention-2 package.
TurboQuant (Zandieh et al., 2025) is the existing rotation/scalar-codebook key
quantizer, with reconstructed BF16 Kq, not packed FP4 arithmetic or QJL.
BLASST (Yuan et al., 2025/2026) supplies the running-max log comparator used by
existing SP; ThriftAttention (Sharratt 2026) motivates precision selection by
attention weight. The existing online softmax/split merge follows standard
online normalization (Milakov/Gimelshein 2018; Dao FlashAttention-2 2023).
SP3 contributes the model seam, captures, controls and reporting, not kernels.
"""
from __future__ import annotations
import ctypes
import gc
import math
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np
from apa_sp3_common import ART, BUILD, ROOT, FEEDING, Red, load_runtime, publish, registration, sha, verify_weight_stat


def last512(logits, ids):
    # Prior art: standard teacher-forced NLL/perplexity (Shannon, 1948).
    # Reference: GraftRepository/tests/minicpm3_bulkbits_floor.py::window_nll
    # (2026). Lead amendment 2 explicitly corrects its 511-target loop to 512
    # and requires FP64. Use the prefix's final logit at index S-513 too.
    ids = np.asarray(ids)
    if logits.shape[0] != len(ids) or len(ids) < 513:
        raise Red('last-512 scoring shape')
    x = np.asarray(logits[-513:-1], np.float64)
    targets = ids[-512:]
    if not np.isfinite(x).all():
        raise Red('nonfinite logits')
    mx = x.max(-1)
    nll = mx + np.log(np.exp(x - mx[:, None]).sum(-1)) - x[np.arange(512), targets]
    return {'ppl': math.exp(float(nll.mean())), 'mean_nll': float(nll.mean()),
            'total_nll': float(nll.sum()), 'targets': 512,
            'target_sha256': __import__('hashlib').sha256(targets.astype('<i8').tobytes()).hexdigest()}


def score_windows(model, ids, S=1024):
    # Prior art: floor.py (2026) independent windows + pooled teacher-forced
    # NLL. Amendment 2 chooses full prefills and 512 targets at float64.
    count = 6 if S == 1024 else 1
    if S not in (1024,8192,32768) or len(ids) < count*S:
        raise Red('PROTOCOL-2 incomplete/invalid scoring windows')
    rows = [dict(model.forward(ids[w*S:(w+1)*S]), window=w, start=w*S) for w in range(count)]
    if any(r['targets'] != 512 or not math.isfinite(r['total_nll']) for r in rows):
        raise Red('PROTOCOL-2 missing or nonfinite window targets')
    nll = math.fsum(r['total_nll'] for r in rows)
    targets = sum(r['targets'] for r in rows)
    all_targets = np.concatenate([ids[w*S+S-512:(w+1)*S] for w in range(count)])
    return dict(ppl=math.exp(nll/targets),total_nll=nll,mean_nll=nll/targets,targets=targets,
                windows=rows,feeding=FEEDING,
                target_sha256=__import__('hashlib').sha256(all_targets.astype('<i8').tobytes()).hexdigest(),
                wall_ms=sum(r['wall_ms'] for r in rows),
                timing='sum of independent full-prefill forward walls; sync each; NLL excluded',
                peak_resident_mib=max(r['peak_resident_mib'] for r in rows),
                peak_status='max of per-window estimated peaks; see window receipts')


def fraction_summary(rows):
    total = sum(r['pairs'] for r in rows)
    selected = sum(r['selected'] for r in rows)
    if not total or not rows:
        raise Red('vacuous fraction measurement')
    fs = [r['selected'] / r['pairs'] for r in rows]
    return {'fraction': selected / total, 'selected': selected, 'pairs': total,
            'per_layer_min': min(fs), 'per_layer_max': max(fs),
            'per_layer_std': float(np.std(fs)), 'layers': rows}


class Peak:
    """Exact observed cudaMalloc high-water plus sampled context offset estimate.

    Prior art: allocation interposition/high-water profiling, standard systems
    instrumentation. It does not observe every internal driver allocation.
    """
    def __init__(self):
        verify_weight_stat()
        self.lib=ctypes.CDLL(None)
        for name in ('apa_sp3_live_bytes','apa_sp3_peak_bytes'):
            getattr(self.lib,name).restype=ctypes.c_uint64
        self.cuda=ctypes.CDLL('/usr/local/cuda-12.6/lib64/libcudart.so')

    def reset(self):
        free,total=ctypes.c_size_t(),ctypes.c_size_t()
        rc=self.cuda.cudaMemGetInfo(ctypes.byref(free),ctypes.byref(total))
        if rc:
            raise Red(f'cudaMemGetInfo failed: {rc}')
        live=self.lib.apa_sp3_live_bytes()
        if live<=0:
            raise Red('allocation observer did not intercept resident model allocations')
        self.offset=max(0,total.value-free.value-live)
        self.lib.apa_sp3_peak_reset()

    def result(self):
        tracked=self.lib.apa_sp3_peak_bytes()
        return {'peak_tracked_cuda_malloc_mib':tracked/(1<<20),
                'peak_resident_mib':(tracked+self.offset)/(1<<20),
                'peak_status':'ESTIMATE: exact intercepted cudaMalloc high-water + pre-call device/context offset; internal driver transient allocations may be missed; raw pool OFF'}


class Model:
    def __init__(self):
        self.tc = tc = load_runtime()
        for k in list(os.environ):
            if k.startswith('TC_'):
                del os.environ[k]
        os.environ.update(TC_APA_SP='0', TC_APA_SELECTIVE_PATH='0', TC_ATTN_QTILE='0')
        tc.set_alloc_pooling(False)
        # Import engine FIRST: adapter's historical sys.path insertion cannot
        # replace the pinned module. Suppress bytecode writes to read-only trees.
        sys.dont_write_bytecode = True
        sys.path.insert(0, '/mnt/ForgeRealm/GraftRepository')
        from core import minicpm3_tc as mini, mistral7b_tc as base
        import _apa_sp3_diag as diag
        self.mini, self.base, self.diag = mini, base, diag
        assert Path(tc._C.__file__).resolve().parent == BUILD
        assert Path(mini.__file__).resolve() == Path('/mnt/ForgeRealm/GraftRepository/core/minicpm3_tc.py')
        base.QuantLinearTC.WEIGHT_BITS = 4
        base.QuantLinearTC.FUSED_DECODE = False
        base.RMSNormTC.USE_FUSED = False
        base.F.USE_FUSED_SOFTMAX = False
        with tc.no_grad():
            self.model, self.info = mini.MiniCPM3_TC.from_pretrained(registration()['model']['snapshot'])
        if self.info['weight_bits'] != 4:
            raise Red('wrong INT4 weight path')
        if base.BlockTC.COMPUTE_DTYPE != 'bfloat16' or base.LinearTC.DTYPE != 'bfloat16' or base.GROUP_SIZE != 128:
            raise Red('adapter default compute dtype / weight group differs from registered path')
        self.original_blend = base._cublas_blend_attention
        self.original_selective = tc.apa_selective_attention
        self.original_attn = mini.MLAAttentionTC.__call__
        owner = self

        def call(attention, *args, **kwargs):
            owner.layer = owner.layer_ids[id(attention)]
            return owner.original_attn(attention, *args, **kwargs)

        self.layer_ids = {id(l.self_attn): i for i, l in enumerate(self.model.layers)}
        mini.MLAAttentionTC.__call__ = call
        base._cublas_blend_attention = self.blend
        tc.apa_selective_attention = self.selective
        self.arm, self.bits, self.delta = 'A', 4, None
        self.observe, self.capture_dir, self.rows = False, None, []
        self.peak = Peak()

    def set(self, arm, bits=4, delta=None, observe=False, capture=None):
        if arm not in ('A', 'B', 'C', 'D', 'E'):
            raise Red('unknown arm')
        if arm in ('C', 'D', 'E') and (delta is None or not math.isfinite(delta) or delta < 0):
            raise Red('missing or invalid SP delta')
        self.arm, self.bits, self.delta = arm, bits, delta
        self.observe, self.capture_dir, self.rows = observe, capture, []
        os.environ['TC_APA_SP'] = '1' if arm in ('C', 'D', 'E') else '0'
        for layer in self.model.layers:
            a = layer.self_attn
            a.attention_mode = 'standard' if arm == 'A' else 'apa_selective'
            a.bulk_bits, a.refine_percentile = bits, .10
            a.absorbed_decode = False
            a.fast_max_seq = 4096 if arm == 'B' else 0
        if capture:
            Path(capture).mkdir(parents=True, exist_ok=False)

    def capture(self, q, k, kq, mask, causal, path_kind, native_bulk_chunks=None):
        B, H, L, D = q.shape
        S = k.shape[2]
        if (B, H, D) != (1, 40, 96) or k.shape[1] != H:
            raise Red('model composite-key contract changed')
        lengths = S - L + np.arange(L) + 1 if causal else np.full(L, S)
        pairs = int(B * H * lengths.sum())
        mn = mask.numpy().astype(np.uint8)
        if mn.shape != (B, H, L, S):
            raise Red('diagnostic mask geometry')
        valid = np.arange(S)[None, None, None, :] < lengths[None, None, :, None]
        if np.any(mn & ~valid):
            raise Red('selected a causally invalid pair')
        selected = int(mn.sum(dtype=np.int64))
        row = {'layer': self.layer, 'selected': selected, 'pairs': pairs,
               'fraction': selected / pairs, 'path': path_kind}
        self.rows.append(row)
        if self.capture_dir:
            p = Path(self.capture_dir) / f'layer{self.layer:02d}'
            p.mkdir(exist_ok=False)
            files = {}
            for name, tensor in [('q', q), ('k', k), ('kq', kq)]:
                f = p / (name + '.npy')
                np.save(f, tensor.float().numpy(), allow_pickle=False)
                files[f.name] = sha(f)
            f = p / 'selected.pack.npy'
            np.save(f, np.packbits(mn, axis=-1, bitorder='little'), allow_pickle=False)
            files[f.name] = sha(f)
            publish(p / 'capture.json', dict(row, shapes={'q': list(q.shape), 'k': list(k.shape)},
                    files=files, causal=causal, arm=self.arm, bits=self.bits, delta=self.delta,
                    scale=96 ** -.5, dtype=q.dtype,
                    native_bulk_chunks=native_bulk_chunks,
                    mask_evidence='native SP diagnostic or instrumented literal B kernel; bit parity checked'))

    def blend(self, q, k, kq, v, group, scale, z, causal, blk):
        out = self.original_blend(q, k, kq, v, group, scale, z, causal, blk)
        if not self.observe:
            return out
        if group != 1:
            raise Red('SP3 blend diagnostic is MHA only')
        tc = self.tc
        chunks, masks, bulk_pins = [], [], []
        for i in range(0, q.shape[2], blk):
            qi = q.slice(2, i, min(blk, q.shape[2] - i))
            bulk = tc.matmul(qi, kq.transpose(-2, -1)) * scale
            if self.capture_dir:
                import hashlib
                bulk_pins.append({'row0':i,'length':qi.shape[2],
                                  'sha256':hashlib.sha256(bulk.float().numpy().tobytes()).hexdigest()})
            rank = tc.matmul(qi, k.transpose(-2, -1)) * scale
            w, mask = self.diag.blend(bulk, rank, z, q.shape[2] if causal else 0, i)
            chunks.append(tc.matmul(w, v))
            # tc.cat is float-only; 0/1 masks round-trip exactly through FP32
            # and capture's existing uint8 cast. Prior art: routine dtype
            # adaptation; no prior art known to me for this specific repair.
            masks.append(mask.float())
        diagnostic_out = tc.cat(chunks, dim=2)
        if not np.array_equal(out.numpy(), diagnostic_out.numpy()):
            raise Red('B diagnostic changes native blend output; not claimed fixed')
        self.capture(q, k, kq, tc.cat(masks, dim=2), causal, 'B_native_blend', bulk_pins)
        return out

    def selective(self, q, k, kq, v, scale, z, causal=False):
        if q.shape[-1] != 96 or v.shape[-1] != 96:
            raise Red('SP3 requires padded D=VD=96')
        if self.arm == 'B':
            out = self.original_selective(q, k, kq, v, scale, z, causal)
            if self.observe:
                if q.shape[2] == 1:
                    raise Red('B diagnostic capture only supports prefill')
                other, mask = self.diag.selective(q, k, kq, v, scale, z, causal)
                if not np.array_equal(out.numpy(), other.numpy()):
                    raise Red('B diagnostic changes native fused output; not claimed fixed')
                self.capture(q, k, kq, mask, causal, 'B_native_fused')
            return out
        if self.arm not in ('C', 'D', 'E'):
            raise Red('SP dispatch reached under wrong arm')
        # Binding order is scale, delta, causal, sinks, diagnostics (verified).
        result = self.tc._C.apa_selective_attention_sp(q, k, kq, v, scale, self.delta,
                                                       causal, None, self.observe)
        if self.observe:
            out, mask = result
            self.capture(q, k, kq, mask, causal, 'SP_splitK' if q.shape[2] == 1 else 'SP_prefill')
            return out
        return result

    def forward(self, ids, score=True):
        self.rows = []
        self.tc.synchronize()
        self.peak.reset()
        start = time.perf_counter()
        with self.tc.no_grad():
            logits, caches = self.model(np.asarray(ids, np.int64)[None], last_token_only=not score)
        self.tc.synchronize()
        elapsed = time.perf_counter() - start
        result = {'wall_ms': elapsed * 1000, 'timing': 'wall; sync before/after; one call; prior-call history recorded in worker',
                  'instrumented': self.observe, **self.peak.result()}
        if score:
            result.update(last512(logits.float().numpy()[0], np.asarray(ids)))
        if self.observe:
            if sorted(r['layer'] for r in self.rows) != list(range(62)):
                raise Red('capture missed/duplicated model layers')
            result['refinement'] = fraction_summary(self.rows)
        del logits, caches
        gc.collect()
        self.tc.empty_cache()
        return result

    def decode(self, ids, S, delta=None):
        # Prior art: standard teacher-forced cached decode benchmark; identical
        # continuation controls differing greedy generations. New SP3 registry.
        self.model.extend_rope(S + 32)
        self.tc.synchronize()
        self.peak.reset()
        pre = time.perf_counter()
        with self.tc.no_grad():
            lg, cache = self.model(ids[:S][None], last_token_only=True)
        self.tc.synchronize()
        pre = time.perf_counter() - pre
        del lg
        times = []
        with self.tc.no_grad():
            for pos in range(S, S + 31):
                self.tc.synchronize()
                start = time.perf_counter()
                lg, cache = self.model(ids[pos:pos+1][None], kv_caches=cache,
                                       position_offset=pos, last_token_only=True)
                self.tc.synchronize()
                times.append(time.perf_counter() - start)
                # Validate output outside measured forward; no greedy sampling.
                if not np.isfinite(lg.float().numpy()).all():
                    raise Red('nonfinite decode logits')
                del lg
        return {'tokens_s': 32 / sum(times), 'steps': 32, 'seconds_per_step': times,
                'prefill_s': pre, 'timing': 'per-token wall with CUDA sync',
                'attention_contexts': [S+1, S+32], 'trained_window_exceeded': S+32 > 32768,
                'decode_path': 'expanded MLA; SP uses native split-K; no absorbed baseline',
                **self.peak.result()}


def capture_space(S):
    # Conservative disk budget for both full FP32 activation captures and
    # packed native masks. Standard size accounting, not an optimization.
    required = int(62 * (3 * 40 * S * 96 * 4 + 40 * S * ((S+7)//8)) * 1.15)
    free = shutil.disk_usage(ART).free
    if free < required:
        raise Red(f'CAPTURE_DISK_OOM: need {required} bytes, available {free}')
    return required
