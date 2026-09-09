"""Independent dense reference and descriptive comparisons.
Prior art: Vaswani et al. (2017), scaled dot-product attention; ordinary
max-shift softmax, Frobenius norm; IEEE754 (2019) ties-to-even bit rounding.
June Gemma (2026) bottom-right causal MQA. No new attention algorithm.
"""
import itertools
import numpy as np
from apa_sp4g_common import Red

MAX_ABS = 1e-4
REL_FROB = 1e-5

def compare(candidate, reference):
    a, b = np.asarray(candidate), np.asarray(reference)
    if (a.shape != b.shape or a.size == 0 or not np.isfinite(a).all()
            or not np.isfinite(b).all()):
        raise Red('A3_INVALID_COMPARISON')
    diff = a.astype(np.float64) - b.astype(np.float64)
    denom = max(float(np.linalg.norm(b.astype(np.float64).ravel())), 1e-30)
    absmax = float(np.max(np.abs(diff)))
    relative = float(np.linalg.norm(diff.ravel()) / denom)
    bitwise = a.dtype == b.dtype and a.tobytes() == b.tobytes()
    return dict(max_abs=absmax, relative_frobenius=relative, bitwise=bitwise,
                different_values=int(np.count_nonzero(diff)),
                denominator='max(||reference||F,1e-30); float64 reductions',
                fp32_agreement=absmax <= MAX_ABS and relative <= REL_FROB)

def bf16_round(x):
    # IEEE754 round-to-nearest/ties-to-even, reused BF16 bit representation.
    x = np.ascontiguousarray(x, dtype=np.float32)
    if not np.isfinite(x).all():
        raise Red('A3_NONFINITE_BF16_INPUT')
    bits = x.view(np.uint32)
    rounded = (bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) & np.uint32(0xffff0000)
    return rounded.view(np.float32)

def eligible(L, S, causal, offset, position_offset):
    if L < 1 or S < L or position_offset < 0:
        raise Red('A3_BAD_GEOMETRY')
    if offset not in ('native_bottom_right', 'absolute_rowwise', 'zero_rowwise'):
        raise Red('A3_UNKNOWN_OFFSET')
    start = S - L if offset == 'native_bottom_right' else (position_offset if offset == 'absolute_rowwise' else 0)
    lengths = start + np.arange(L) + 1 if causal else np.full(L, S)
    if (lengths < 1).any() or (lengths > S).any():
        raise Red('A3_OFFSET_OUT_OF_RANGE')
    return np.arange(S) < lengths[:, None]

def dense(q, k, v, scale=1., sink='none', causal=True,
          offset='native_bottom_right', position_offset=0, staged_bf16=False):
    # Dense NumPy FP32 reference independent of TensorCUDA and SP kernels.
    # MQA broadcast is explicit by head indexing, also supporting grouped KV.
    q, k, v = [np.asarray(t, dtype=np.float32) for t in (q, k, v)]
    B, H, L, D = q.shape
    S = k.shape[2]
    if (k.ndim != 4 or v.shape != k.shape or k.shape[0] != B
            or k.shape[3] != D or H % k.shape[1] or sink not in ('none', 'zero')
            or not np.isfinite(scale) or scale <= 0
            or any(not np.isfinite(t).all() for t in (q, k, v))):
        raise Red('A3_DENSE_INPUT')
    mask = eligible(L, S, causal, offset, position_offset)
    out = np.empty_like(q)
    for b in range(B):
        for h in range(H):
            kvh = h // (H // k.shape[1])
            scores = np.float32(scale) * (q[b, h] @ k[b, kvh].T)
            if staged_bf16:
                scores = bf16_round(scores)
            scores = np.where(mask, scores, np.float32(-np.inf))
            maximum = scores.max(axis=-1, keepdims=True)
            if sink == 'zero':
                maximum = np.maximum(maximum, np.float32(0))
            exps = np.exp(scores - maximum)
            denom = exps.sum(axis=-1, keepdims=True, dtype=np.float32)
            if sink == 'zero':
                denom += 0  # extra zero-logit, zero-value sink
            probs = exps / denom
            if staged_bf16:
                probs = bf16_round(probs)
            value = probs @ v[b, kvh]
            out[b, h] = bf16_round(value) if staged_bf16 else value
    if not np.isfinite(out).all():
        raise Red('A3_DENSE_NONFINITE')
    return out

def variants():
    return [dict(scale=scale, sink=sink, causal=causal, offset=offset)
            for scale, sink, causal, offset in itertools.product(
                (1., 512 ** -.5), ('none', 'zero'), (True, False),
                ('native_bottom_right', 'absolute_rowwise', 'zero_rowwise'))]

NOMINAL = dict(scale=1., sink='none', causal=True, offset='native_bottom_right')

def classify(call_results):
    # Standard single-factor treatment comparison, not a new selection rule.
    # Must rescue BOTH calls; an already-matching baseline cannot be a rescue.
    if len(call_results) != 2:
        raise Red('A3_TWO_CALLS_REQUIRED')
    nominal = [next(r for r in rows if r['variant'] == NOMINAL) for rows in call_results]
    if all(r['vs_standard']['fp32_agreement'] and r['vs_own_dense']['fp32_agreement']
           and r['standard_vs_nominal_dense']['fp32_agreement'] for r in nominal):
        return dict(outcome='NOMINAL_AGREES_NO_ARGUMENT_FIX', single_change_matches=[],
                    C_unblocked=False, explanation='Both nominal calls agree locally. Keep PPL RED; inspect merge/propagation.')
    candidates = []
    for variant in variants():
        if sum(variant[k] != NOMINAL[k] for k in NOMINAL) != 1:
            continue
        rows = [next(r for r in rs if r['variant'] == variant) for rs in call_results]
        if all(r['vs_standard']['fp32_agreement'] and r['vs_own_dense']['fp32_agreement'] for r in rows):
            # A candidate is a measurement lead, not proof of its semantics.
            candidates.append(variant)
    return dict(outcome='SINGLE_CHANGE_CANDIDATE' if len(candidates) == 1 else 'UNRESOLVED_OR_AMBIGUOUS',
                single_change_matches=candidates, C_unblocked=False,
                explanation='Requires named source cause and immutable treatment or removed-op registration; .005 PPL unchanged.')
