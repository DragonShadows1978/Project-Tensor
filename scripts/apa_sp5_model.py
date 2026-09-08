"""APA-SP5 GPT-OSS-20B harness.

Prior art at this code site:
  - The port itself (GraftRepository/core/gpt_oss20b_tc.py, READ-ONLY) --
    GptOss20B_TC / GptOssDiagnosticBlockTC / gpt_oss_yarn_rope_tables.
  - Streamed per-layer forward: GraftRepository scripts/
    gpt_oss20b_stream_forward_smoke.py (2026), whose per-layer loop this
    harness's PARITY comment in the port names as the bit-for-bit contract.
  - Windowed ppl protocol shape: SP3 PROTOCOL-2 / SP4G PROTOCOL-G (this
    repo, 2026); wikitext-2 (Merity et al., 2016).
  - Attention sinks in the denominator: GPT-OSS model card (OpenAI, 2025).
  - Running-max single-pass selection: BLASST (Yuan et al., arXiv
    2512.12087, MLSys 2026), applied to precision not sparsity (APA,
    David's design); FlashAttention-2 (Dao 2023) / online softmax
    (Milakov & Gimelshein 2018) for the fused loop; ThriftAttention
    (Sharratt, arXiv 2605.23081, 2026) for weight-proportional error;
    TurboQuant-style bulk key quantizer (tensor_cuda.quant).
Mine here: the arm plumbing, the sink-handling ablation and the wall
measurement -- no new algorithm.
"""
import gc, json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path('/mnt/ForgeRealm/Project-Tensor-wt-apa-sp5')
A = ROOT / 'artifacts/apa_sp5'
BUILD = A / 'build'
GRAFT = Path('/mnt/ForgeRealm/GraftRepository')
SNAP = ('/home/vader/.cache/huggingface/hub/models--openai--gpt-oss-20b/'
        'snapshots/6cee5e81ee83917806bbde320786a8fb61efebee')

ENV = dict(TC_APA_SP='1', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
           TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1')


def load_runtime():
    os.environ.update(ENV)
    sys.dont_write_bytecode = True
    for p in (str(BUILD), str(ROOT / 'tensor_cuda'), str(GRAFT)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import tensor_cuda as tc
    if Path(tc._C.__file__).resolve().parent != BUILD:
        raise RuntimeError('WRONG_ENGINE_CHECKOUT: ' + tc._C.__file__)
    if Path(tc.__file__).resolve().parent != ROOT / 'tensor_cuda/tensor_cuda':
        raise RuntimeError('WRONG_TC_PACKAGE: ' + tc.__file__)
    return tc


def port(tc):
    """Import the READ-ONLY GPT-OSS port against our engine build."""
    from core import gpt_oss20b_tc as G
    return G


def gpu_mib():
    import subprocess
    out = subprocess.run(['nvidia-smi', '--query-gpu=memory.used',
                          '--format=csv,noheader,nounits'],
                         capture_output=True, text=True, timeout=30)
    return int(out.stdout.strip().splitlines()[0])


def tokens():
    meta = json.loads((A / 'tokens_meta.json').read_text())
    ids = np.load(A / 'tokens.npy', allow_pickle=False)
    import hashlib
    if hashlib.sha256(ids.tobytes()).hexdigest() != meta['token_sha256']:
        raise RuntimeError('TOKEN_STREAM_CHANGED')
    return ids, meta


# ---------------------------------------------------------------- arms
# Arm registry (item 2). Sink handling is stated per arm and is the SAME
# learned sink tensor in all of them; only the fold site differs.
ARMS = {
    # A: engine standard. sink_attention_tc concatenates the [H] sink logit
    #    as an extra score column, softmaxes S+1 columns, then DROPS the
    #    sink column before the value matmul -> sink is in the denominator
    #    only. Full layers: dense B,H,L,S scores. Sliding layers: chunked.
    'A': dict(mode='standard', sp=False),
    # B: two-pass. tc.apa_selective_attention_sink -- z-score threshold on
    #    |bulk|, sink folded inside the fused kernel at the end.
    'B': dict(mode='apa_selective', sp=False),
    # C/D/E: single pass. tc._C.apa_selective_attention_sp(..., sinks=...)
    #    folds the sink logit into the running max AFTER the key loop
    #    (kernels.cu apa_selective_sp_kernel: `if (sinks) { ... }`), so the
    #    sink participates in the denominator and rescales the accumulator
    #    but is never a selection candidate -- it cannot be "refined".
    'C': dict(mode='apa_sp', sp=True),
    'D': dict(mode='apa_sp', sp=True, refine_all=True),
    'E': dict(mode='apa_sp', sp=True),
}


def install_sp(G, tc, delta, scope='full', capture=None):
    """Route full-attention layers to the single-pass entry.

    Monkey-patches only OUR harness's view of the port's attention call;
    the port file on disk is never edited (it is READ-ONLY).
    """
    from tensor_cuda.quant import _quantize_keys, _tables
    original = G.GptOssAttentionTC.__call__
    state = dict(calls=[])

    def patched(self, x, cos, sin, position_offset=0, kv_cache=None):
        if getattr(self, '_sp_delta', None) is None:
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
        dev = q.device.split(':')[0]
        Rt, CB, BND = _tables(self.head_dim, self.bulk_bits, self.num_kv_heads, True, dev)
        kq = _quantize_keys(k, Rt, CB, BND)
        want_sel = capture is not None
        res = tc._C.apa_selective_attention_sp(
            q, k, kq, v, self.scaling, float(self._sp_delta), True,
            self.sinks, want_sel)
        if want_sel:
            attn, sel = res
            capture(self.layer_idx, L, S, sel, q, k, kq, v, self.sinks, self.scaling)
        else:
            attn = res
        self.last_attention_backend = 'apa_sp_sink_fused'
        state['calls'].append((self.layer_idx, L, S))
        out = attn.transpose(1, 2).reshape([B, L, self.num_heads * self.head_dim])
        return self.o_proj(out), (k, v)

    G.GptOssAttentionTC.__call__ = patched
    return original, state
