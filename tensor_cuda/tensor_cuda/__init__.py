"""Project Tensor — standalone CUDA tensor library (no PyTorch, no CuPy).

Phase 1 (the core engine) exposes a NumPy-friendly Tensor with reverse-mode
autograd backed entirely by hand-written CUDA kernels + cuBLAS. The C++ does the
work; this module is the thin Python ergonomics layer (factory helpers, dtype
plumbing, no_grad). See ROADMAP.md for the layers still being ported.
"""

from __future__ import annotations

import contextlib

import numpy as np

try:
    from . import _tensor_cuda as _C
except ImportError:  # pragma: no cover
    import _tensor_cuda as _C  # fallback when the .so sits on PYTHONPATH

Tensor = _C.Tensor

_NP_DTYPE = {
    "float32": np.float32,
    "float16": np.float16,
    "int64": np.int64,
    "bool": np.bool_,
    "uint8": np.uint8,
}

# bfloat16 has no NumPy equivalent: build the host array as fp32 and cast the
# device tensor to bf16 afterward. Listing it here so callers passing
# dtype="bfloat16" no longer fall through _NP_DTYPE.get(..., np.float32) and get
# a SILENT fp32 downcast (which corrupts bf16 constants for OLMoE/Qwen-class
# models). dtype_from_string in the C++ already supports "bfloat16".
_DEVICE_CAST_DTYPE = {"bfloat16"}


def tensor(data, *, device="cuda", dtype="float32", requires_grad=False):
    """Create a Tensor from array-like data."""
    if dtype in _DEVICE_CAST_DTYPE:
        arr = np.ascontiguousarray(np.asarray(data, dtype=np.float32))
        return _C.tensor(arr, device, requires_grad).astype(dtype)
    arr = np.asarray(data, dtype=_NP_DTYPE.get(dtype, np.float32))
    arr = np.ascontiguousarray(arr)
    return _C.tensor(arr, device, requires_grad)


def _factory(np_fn):
    def make(*shape, device="cuda", dtype="float32", requires_grad=False):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        if dtype in _DEVICE_CAST_DTYPE:
            arr = np.ascontiguousarray(np_fn(shape).astype(np.float32))
            return _C.tensor(arr, device, requires_grad).astype(dtype)
        arr = np_fn(shape).astype(_NP_DTYPE.get(dtype, np.float32))
        return _C.tensor(np.ascontiguousarray(arr), device, requires_grad)
    return make


zeros = _factory(np.zeros)
ones = _factory(np.ones)
randn = _factory(lambda s: np.random.randn(*s))
rand = _factory(lambda s: np.random.rand(*s))


def from_numpy(arr, *, device="cuda", requires_grad=False):
    return _C.tensor(np.ascontiguousarray(arr), device, requires_grad)


def matmul(a, b, alpha=1.0, trans_b=False):
    # trans_b reads b as (..., N, K) row-major via cuBLAS OP_T — no transpose
    # copy. alpha is applied in the fp32 accumulator before the 16-bit store.
    return _C.matmul(a, b, alpha, trans_b)


def causal_softmax(scores):
    """Fused bottom-right causal softmax over (..., L, S) scores, S >= L.
    Inference-only (backward raises). Equivalent to adding the -1e4 causal
    bias then softmax, with masked entries exactly zero."""
    return _C.causal_softmax(scores)


def rms_norm(x, w, eps=1e-6):
    """Fused RMSNorm over the last dim (single kernel, fp32 accumulate,
    output in x's dtype). Inference-only: backward raises — training code
    must use an unfused op chain. w must be fp32."""
    return _C.rms_norm(x, w, eps)


def rope_apply(x, cos, sin, pos0=0, inverse=False, pair_swap=False):
    """Fused RoPE: out = x*cos[pos0+l] + rotate_half(x)*sin[pos0+l] in
    ONE launch (the composed chain is ~8). x (..., L, D); tables (T, D)
    in x's dtype. Inference-only: backward raises."""
    return _C.rope_apply(x, cos, sin, pos0, inverse, pair_swap)


def write_rows(buf, src, start=0):
    """IN-PLACE ring write: src rows land at (start+l) %% CAP along
    buf's dim -2. The decode-cache primitive (zero-copy appends).
    Inference-only (raises under grad). MUTATES buf — callers own the
    sharing contract: never alias a written buffer from a held cache."""
    _C.write_rows(buf, src, start)


def export_rows(cache, dim, start, length):
    return _C.export_rows(cache, dim, start, length)


def export_rope_rows(cache, cos, sin, dim, start, length, pos0,
                     inverse=False, pair_swap=False):
    return _C.export_rope_rows(cache, cos, sin, dim, start, length, pos0,
                               inverse, pair_swap)


def export_row_pair(raw_cache, rope_cache, cos, sin, raw_dim, rope_dim,
                    raw_start, rope_start, length, pos0, inverse=False,
                    pair_swap=False):
    return _C.export_row_pair(raw_cache, rope_cache, cos, sin, raw_dim,
                              rope_dim, raw_start, rope_start, length, pos0,
                              inverse, pair_swap)


def export_row_pairs(raw_caches, rope_caches, cos, sin, raw_dim, rope_dim,
                     raw_starts, rope_starts, length, pos0, inverse=False,
                     pair_swap=False):
    return _C.export_row_pairs(list(raw_caches), list(rope_caches), cos, sin,
                               raw_dim, rope_dim, list(raw_starts),
                               list(rope_starts), length, pos0, inverse,
                               pair_swap)


def swap_row_pairs_with_rope(raw_caches, rope_caches, raw_inserts,
                             rope_inserts, cos, sin, raw_dim, rope_dim,
                             head_tokens, tail_start, pos0, pair_swap=False):
    return _C.swap_row_pairs_with_rope(
        list(raw_caches), list(rope_caches), list(raw_inserts),
        list(rope_inserts), cos, sin, raw_dim, rope_dim, head_tokens,
        tail_start, pos0, pair_swap)


def evict_row_pairs(raw_caches, rope_caches, raw_dim, rope_dim, head_tokens,
                    drop_tokens):
    return _C.evict_row_pairs(list(raw_caches), list(rope_caches), raw_dim,
                              rope_dim, head_tokens, drop_tokens)


def arena_row_pair_transaction(raw_caches, rope_caches, raw_inserts,
                               rope_inserts, cos, sin, raw_dim, rope_dim,
                               sink_tokens, current_mount_tokens,
                               arena_width, pair_swap=False):
    return _C.arena_row_pair_transaction(
        list(raw_caches), list(rope_caches), list(raw_inserts),
        list(rope_inserts), cos, sin, raw_dim, rope_dim, sink_tokens,
        current_mount_tokens, arena_width, pair_swap)


def int4_linear(x, packed, scales, zeros, group_size=128):
    """INT4 group-quantized linear: y = x @ dequant(W)^T. Inference only.
    Two-stage: dequant W to a full (K,N) fp16 buffer then cuBLAS matmul."""
    return _C.int4_linear(x, packed, scales, zeros, group_size)


def int4_linear_fused(x, packed, scales, zeros, group_size=128):
    """INT4 linear via a fused dequant-GEMM — same result as int4_linear but
    dequantizes the weight in shared-memory tiles inside the GEMM, avoiding the
    full (K,N) fp16 weight transient. Opt-in: a hand GEMM can lose to cuBLAS at
    large N, so benchmark before defaulting to it."""
    return _C.int4_linear_fused(x, packed, scales, zeros, group_size)


def int4_dequant(packed, scales, zeros, group_size=128, out_dtype="float16"):
    """Dequantize packed INT4 weight to a (K, N) transposed fp16/fp32 matrix."""
    return _C.int4_dequant(packed, scales, zeros, group_size, out_dtype)


def intn_linear(x, packed, scales, zeros, bits, in_features, group_size=128):
    """INT2/INT3 group-quantized linear: y = x @ dequant(W)^T."""
    return _C.intn_linear(x, packed, scales, zeros, bits, in_features, group_size)


def intn_linear_fused(x, packed, scales, zeros, bits, in_features, group_size=128):
    """INT2/INT3 fused dequant-GEMM/GEMV path without a full weight transient."""
    return _C.intn_linear_fused(
        x, packed, scales, zeros, bits, in_features, group_size
    )


def intn_dequant(
    packed, scales, zeros, bits, in_features, group_size=128, out_dtype="float16"
):
    """Dequantize packed INT2/INT3 weight to a (K, N) transposed matrix."""
    return _C.intn_dequant(
        packed, scales, zeros, bits, in_features, group_size, out_dtype
    )


def mxfp4_linear(x, blocks, scales):
    """GPT-OSS MXFP4 expert linear: y = x @ dequant(blocks, scales).

    `blocks` is shaped `(out_features, groups, 16)` uint8 and `scales` is
    `(out_features, groups)` uint8. Each 16-byte group expands to 32 FP4
    weights with E8M0 exponent scales. Inference-only frozen-weight path.
    """
    return _C.mxfp4_linear(x, blocks, scales)


def mxfp4_linear_expert(x, blocks, scales, expert_idx):
    """GPT-OSS resident MXFP4 expert linear from `[experts, N, G, 16]`.

    `expert_idx` selects the packed expert inside the CUDA op, avoiding uint8
    slicing in Python.
    """
    return _C.mxfp4_linear_expert(x, blocks, scales, expert_idx)


def kv_int4_pack(x, group=32):
    """Quantize+pack a KV tensor (B,KV,S,D) to D-grouped 4-bit. Returns
    (packed_uint8 (B,KV,S,D/2), scales (B,KV,S,D/group) in x.dtype).
    Symmetric-8, group-32. Distinct from int4_dequant (weight, K-grouped)."""
    return _C.kv_int4_pack(x, group)


def kv_int4_unpack(packed, scales, group=32, lo=0, n=0, out_dtype="bfloat16"):
    """Dequantize rows [lo:lo+n) of a packed KV buffer -> (B,KV,n,D)."""
    return _C.kv_int4_unpack(packed, scales, group, lo, n, out_dtype)


def gated_delta_step(q, k, v, a, b, A_neg, dt_bias, state):
    """Fused Gated DeltaNet decode step (one token, one layer, ONE launch):
    l2norm(q,k) + gate math (sigmoid/softplus/exp) + decay-first delta-rule
    state update + readout. All fp32. FUNCTIONAL: returns (out, new_state)
    and leaves the input state untouched, so callers may branch or hold
    references freely (the GRM restore-once-decode-many contract).
    q,k (B,Hk,Dk) raw heads; v (B,H,Dv); a,b (B,H); A_neg = -exp(A_log),
    dt_bias: H elements; state (B,H,Dk,Dv). Inference only (no autograd)."""
    return _C.gated_delta_step(q, k, v, a, b, A_neg, dt_bias, state)


def apa_selective_attention(q, k, kq, v, scale, zthr, is_causal=False):
    """Fused sparse selective APA attention: full-precision dot only on the keys
    the bulk/quantized pass selects (|bulk| >= mean+zthr*std), rest stay quantized.
    q,k,kq,v: (B,H,L,D)/(B,H,S,D). Inference only (no autograd)."""
    return _C.apa_selective_attention(q, k, kq, v, scale, zthr, is_causal)


def apa_blend_softmax(bulk, rank, zthr):
    """Fused APA blend+softmax over precomputed bulk/rank score matrices (..., S):
    per row thr = mean(|rank|)+zthr*std(|rank|); score = |rank|>=thr ? rank : bulk;
    returns softmax(score). Causal masking must be baked into bulk/rank as large
    negative scores by the caller. Pairs with cuBLAS bulk/rank matmuls."""
    return _C.apa_blend_softmax(bulk, rank, zthr)


def mse_loss(pred, target):
    return _C.mse_loss(pred, target)


def where(cond, x, y):
    return _C.where(cond, x, y)


def einsum(equation, *operands):
    from . import functional
    return functional.einsum(equation, *operands)


def embedding(weight, idx):
    if not isinstance(idx, Tensor):
        idx = _C.tensor(np.ascontiguousarray(np.asarray(idx, dtype=np.int64)), "cuda", False)
    return _C.embedding(weight, idx)


def cat(tensors, dim=0):
    return _C.cat(list(tensors), dim)


def splice_rows(old_cache, insert, dim, head_tokens, tail_start):
    return _C.splice_rows(old_cache, insert, dim, head_tokens, tail_start)


def evict_rows(old_cache, dim, head_tokens, drop_tokens):
    return _C.evict_rows(old_cache, dim, head_tokens, drop_tokens)


def stack(tensors, dim=0):
    return _C.stack(list(tensors), dim)


def cross_entropy(logits, target, *, device="cuda"):
    """Cross-entropy from logits. `target` may be int class labels (1D) or a
    float one-hot Tensor matching `logits`."""
    if isinstance(target, Tensor):
        return _C.cross_entropy(logits, target)
    labels = np.asarray(target).astype(np.int64).ravel()
    num_classes = logits.shape[-1]
    onehot = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    onehot[np.arange(labels.shape[0]), labels] = 1.0
    onehot = onehot.reshape(logits.shape)
    return _C.cross_entropy(logits, _C.tensor(onehot, device, False))


def save_checkpoint(path, model, **extra):
    """Save model parameters (+ optional extra scalars) to a .npz file."""
    sd = model.state_dict()
    payload = {f"model.{k}": v for k, v in sd.items()}
    for k, v in extra.items():
        payload[f"extra.{k}"] = np.array(v)
    np.savez(path, **payload)


def load_checkpoint(path, model):
    """Load parameters saved by save_checkpoint into `model`. Returns extras."""
    data = np.load(path, allow_pickle=True)
    sd = {k[len("model."):]: data[k] for k in data.files if k.startswith("model.")}
    model.load_state_dict(sd)
    return {k[len("extra."):]: data[k] for k in data.files if k.startswith("extra.")}


def weight_tie(src_module, src_attr, dst_module, dst_attr):
    """Tie two parameters to share one Tensor (e.g. embedding <-> LM head).

    Both modules then reference the same parameter object; gradients accumulate
    once and optimizers (which dedup by identity) update it once.
    """
    shared = getattr(src_module, src_attr)
    setattr(dst_module, dst_attr, shared)
    return shared


def checkpoint(fn, *inputs):
    """Gradient checkpointing.

    Runs `fn(*inputs)` under no_grad during the forward pass and replays it
    during backward, saving only the checkpoint inputs instead of the full
    interior activation graph. The function must return one Tensor.
    """
    return _C.checkpoint(fn, list(inputs))


def synchronize():
    _C.synchronize()


def empty_cache():
    """Release device blocks held idle by the caching allocator back to the
    driver. Live tensors are unaffected."""
    _C.empty_cache()


def set_alloc_pooling(enabled):
    """Enable the stream-ordered transients pool. Call AFTER model/weight
    loading: allocations made while disabled use raw cudaMalloc (persistents
    must stay raw — live pooled blocks pin pool chunks and cost context
    ceiling at OOM walls). Forward-pass transients allocated while enabled
    are pooled, removing the cudaMalloc/cudaFree serialization tax."""
    _C.set_alloc_pooling(bool(enabled))


def is_grad_enabled():
    return _C.is_grad_enabled()


@contextlib.contextmanager
def no_grad():
    prev = _C.is_grad_enabled()
    _C.set_grad_enabled(False)
    try:
        yield
    finally:
        _C.set_grad_enabled(prev)


from . import functional  # noqa: E402
from . import nn  # noqa: E402  (after _C and helpers are defined)
from . import optim  # noqa: E402
from . import quant  # noqa: E402
from . import quantization  # noqa: E402

apa_quant_attention = quant.apa_quant_attention

__all__ = [
    "Tensor", "tensor", "from_numpy", "zeros", "ones", "randn", "rand",
    "matmul", "rms_norm", "rope_apply", "write_rows", "export_rows",
    "export_rope_rows", "export_row_pair", "export_row_pairs",
    "swap_row_pairs_with_rope", "evict_row_pairs",
    "arena_row_pair_transaction", "causal_softmax", "mse_loss", "cross_entropy", "where", "cat", "stack", "embedding",
    "synchronize", "empty_cache", "set_alloc_pooling", "no_grad", "is_grad_enabled", "nn", "optim", "functional",
    "quant", "quantization", "apa_quant_attention", "save_checkpoint", "load_checkpoint",
    "weight_tie", "checkpoint", "einsum", "int4_linear", "int4_linear_fused",
    "intn_linear", "intn_linear_fused",
    "mxfp4_linear", "mxfp4_linear_expert",
    "gated_delta_step",
    "int4_dequant", "intn_dequant", "apa_selective_attention",
    "kv_int4_pack", "kv_int4_unpack",
    "apa_selective_fwd_train", "apa_selective_bwd", "apa_selective_train",
]


def apa_selective_fwd_train(q, k, kq, v, scale, zthr, is_causal=False):
    """O(L)-memory selective-attention training forward. Returns
    (out, lse, thr) — lse/thr are the saved per-row state the backward needs."""
    return _C.apa_selective_fwd_train(q, k, kq, v, scale, zthr, is_causal)


def apa_selective_bwd(q, k, kq, v, dO, lse, thr, scale, is_causal=False):
    """Selective-attention backward. Returns (dq, dk, dv)."""
    return _C.apa_selective_bwd(q, k, kq, v, dO, lse, thr, scale, is_causal)


def apa_selective_train(q, k, kq, v, scale, zthr, is_causal=False):
    """Differentiable O(L)-memory selective attention (graft-native training).
    Selection is a stop-gradient (kq detached); q,k,v receive gradients.
    Returns a single (B,H,L,D) tensor with autograd wired."""
    return _C.apa_selective_train(q, k, kq, v, scale, zthr, is_causal)
__version__ = "0.1.0-phase1"
