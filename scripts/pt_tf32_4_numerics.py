"""PT-TF32-4 numerical gates; CPU only, no native imports.

Prior art: Gumbel (1958), David & Nagaraja (2003), Gaussian maxima;
Higham (2002), gamma_n and absolute product sums; NVIDIA TF32 (2020),
CUDA 12.6 (2024) __expf accuracy. Taken formulas; ours: registration-driven
dK tail envelope and all-selected dV midpoint intervals. No novelty claim.
https://doi.org/10.7312/gumb92958
https://doi.org/10.1002/0471722162.ch4
https://docs.nvidia.com/cuda/archive/12.6.2/cuda-c-programming-guide/index.html#intrinsic-functions
"""
import math
import numpy as np
import pt_tf32_2_numerics as previous
from pt_tf32_2_numerics import forward_reference, backward_reference, backward_model
import pt_tf32_1 as old


def dk_tail(reg, s):
    """Peak-normalized scale envelope, NOT an RMS-to-maximum identity.

    Prior art: Gaussian tail + union bound, Gumbel (1958)/David & Nagaraja
    (2003). epsilon*RMS(ref) is the homoscedastic sigma; we explicitly assume
    marginal sigma_i <= epsilon*peak(ref) for heteroscedastic causal sums.
    L2 alone cannot imply this. Alpha is a model tail budget, not measured
    hardware reliability. No independence is needed by the union bound.
    """
    rule = reg['dk_tail']
    count = math.prod(s[k] for k in ('B', 'KVH', 'S', 'D'))
    if count < 1 or not isinstance(count, int):
        raise ValueError('invalid dK element count')
    alpha, family = rule['family_alpha'], rule['family_comparisons']
    if not 0 < alpha < 1 or family < 1:
        raise ValueError('invalid tail risk')
    eps = previous.bounds(reg, s, 'dK')['expected']
    leading = eps*math.sqrt(2*math.log(count))
    bound = eps*math.sqrt(2*math.log(2*count*family/alpha))
    return dict(elements=count, expected_rel_L2=eps, relative_L2_bound=3*eps,
                normalized_max_leading=leading, normalized_max_margin=bound-leading,
                normalized_max_bound=bound, sigma_normalized_peak=eps,
                model='assumed sub-Gaussian peak-scale envelope; not a worst-case theorem')


def precision_gate(candidate, reference, reg, s, name):
    """Only dK receives a new maximum bar; all inherited L2 bars remain."""
    b = previous.bounds(reg, s, name)
    if name != 'dK':
        return dict(previous.metric_gate(candidate, reference, b['bound']), expected=b['expected'])
    tail = dk_tail(reg, s)
    a, r = np.asarray(candidate, np.float64), np.asarray(reference, np.float64)
    if a.shape != r.shape or a.size != tail['elements']:
        raise ValueError('dK shape/registered element count mismatch')
    result = previous.metric_gate(a, r, b['bound'])
    result.update(expected=b['expected'], tail=tail, relative_L2_bound=b['bound'],
                  normalized_max_bound=tail['normalized_max_bound'])
    if not result['finite'] or result.get('zero_reference'):
        return result
    good = (result['relative_L2'] <= b['bound'] and
            result['normalized_max_abs'] <= tail['normalized_max_bound'])
    result['verdict'] = 'GREEN' if good else 'RED'
    peak = float(np.max(abs(r)))
    rms = float(np.linalg.norm(r)/math.sqrt(r.size))
    result.update(reference_peak=peak, reference_rms=rms,
                  sigma_envelope_abs=b['expected']*peak,
                  homoscedastic_sigma_abs_diagnostic=b['expected']*rms)
    # On a real failure, preserve the coordinate and state-scale evidence.
    # Full arrays remain ephemeral; the lead can run a focused reproduction.
    at = np.unravel_index(int(np.argmax(abs(a-r))), a.shape)
    result['max_witness'] = dict(index=list(map(int,at)), candidate=float(a[at]), reference=float(r[at]))
    return result


def gamma(n, u=2**-24):
    if n < 0 or n*u >= 1:
        raise ValueError('gamma_n outside model domain')
    return n*u/(1-n*u)


def probability_interval(q, k, saved_lse, scale, visible, u, exp_slope):
    """Input-only interval for the native selected scalar dot and __expf.

    Higham (2002): gamma_(2D+1) covers multiply/add or FMA plus scale.
    NVIDIA CUDA 12.6 (2024): __expf(z) has <=2+floor(abs(1.173*z)) ulp
    error. For normal results ulp(p)/p <= 2^-23. No GPU result enters.
    """
    score = (q @ k.T)*scale
    abs_product = abs(q) @ abs(k).T
    # Also enclose the CPU FP64 BLAS/exp evaluation. This term is tiny but
    # avoids assuming that our independently evaluated dot is an exact real.
    e64 = abs(scale)*gamma(2*q.shape[1]+1, 2**-53)*abs_product
    es = abs(scale)*gamma(2*q.shape[1]+1, u)*abs_product + e64
    z = score-saved_lse[:, None]
    ez = es + u*(abs(z)+es) + 2**-52*(abs(score)+abs(saved_lse[:,None]))
    ulps = 2 + np.floor(exp_slope*(abs(z)+ez))
    rel = ulps*(2*u)
    p = np.exp(z)
    lo = np.exp(z-ez)*(1-rel-2**-51)
    hi = np.exp(z+ez)*(1+rel+2**-51)
    single = visible.sum(axis=1) == 1
    for a in (p, lo, hi):
        a[~visible] = 0.
        a[single] = visible[single].astype(np.float64)
    normal = np.finfo(np.float32).tiny
    if (not all(np.isfinite(a).all() for a in (p, lo, hi)) or
            np.any(lo[visible] < normal) or np.any(hi > np.finfo(np.float32).max) or
            np.any(rel >= 1)):
        raise ValueError('probability interval outside normal finite model')
    # Outward representable endpoints before RNA_TF32 preserve monotonicity
    # through the FP32 storage conversion. The center is the legacy CPU model.
    low32 = np.nextafter(lo.astype(np.float32), np.float32(-np.inf))
    high32 = np.nextafter(hi.astype(np.float32), np.float32(np.inf))
    low32[~visible] = high32[~visible] = 0
    low32[single] = high32[single] = visible[single].astype(np.float32)
    center, lower, upper = [old.tf32_round(a).astype(np.float64) for a in (p, low32, high32)]
    delta = np.maximum(abs(center-lower), abs(upper-center))
    return center, lower, upper, delta, es


def edge_dv_budget(x, state, s, reg):
    """Per-element absolute dV tolerance for the all-selected padding suite.

    Taken: interval propagation/absolute sums (Higham 2002), TF32 adjacent
    bins and NVIDIA __expf bound. Ours: grouped-head placement. No case IDs,
    observed errors, or near-zero rtol floors enter this test-only budget.
    """
    rule = reg['edge_gate']['dV_model']
    B,H,KH,L,S,D,VD = [s[n] for n in ('B','H','KVH','L','S','D','VD')]
    if H % KH or not np.array_equal(x['kq'], x['k']):
        raise ValueError('edge model requires grouped all-selected kq==k')
    for name in ('q','k','kq','v','dO'):
        if x[name].dtype != np.float32 or not np.isfinite(x[name]).all():
            raise ValueError('edge model requires finite FP32 inputs')
    if not all(np.isfinite(state[n]).all() for n in ('out','lse','thr')):
        raise ValueError('nonfinite edge saved state')
    if state['lse'].dtype != np.float32:
        raise ValueError('edge model requires exact FP32 saved lse')
    scale = float(np.float32(s['scale']))
    visible = (np.arange(S)[None,:] <= S-L+np.arange(L)[:,None]
               if s['causal'] else np.ones((L,S),bool))
    budget = np.zeros((B,KH,S,VD),np.float64)
    center = np.zeros_like(budget)
    jumps = np.zeros_like(budget)
    sums = np.zeros_like(budget)
    crossings = 0
    for b in range(B):
        for h in range(H):
            kh = h//(H//KH)
            q, k = x['q'][b,h].astype(np.float64), x['k'][b,kh].astype(np.float64)
            p, lower, upper, delta, es = probability_interval(
                q,k,state['lse'][b,h].astype(np.float64),scale,visible,rule['u32'],rule['exp_ulp_slope'])
            # A negative threshold guarantees all finite scores are selected;
            # singleton rows are selected by definition. Fail closed otherwise.
            nonsingle = visible.sum(1)>1
            if np.any(state['thr'][b,h,nonsingle] >= 0):
                raise ValueError('edge interval requires negative all-selected thresholds')
            do = old.tf32_round(x['dO'][b,h]).astype(np.float64)
            if np.any((do != 0) & (abs(do) < np.finfo(np.float32).tiny)):
                raise ValueError('subnormal edge operand outside model')
            # TF32 products have <=22 significand bits, so FP32 holds each
            # product exactly when normal. Prior art: NVIDIA PTX ISA 8.5
            # (2024) leaves WMMA rounding unspecified. AMENDMENT_001 uses
            # Higham's (2002) one-ulp, rather than RN-only, accumulation
            # envelope. This is an explicit model, not a hardware theorem.
            nonzero_do = abs(do[do != 0])
            nonzero_p = lower[lower != 0]
            if (nonzero_do.size and nonzero_p.size and
                    float(nonzero_do.min())*float(nonzero_p.min()) < np.finfo(np.float32).tiny):
                raise ValueError('subnormal edge product outside model')
            crossings += int(np.count_nonzero(delta))
            center[b,kh] += p.T @ do
            jumps[b,kh] += delta.T @ abs(do)
            sums[b,kh] += np.maximum(abs(lower),abs(upper)).T @ abs(do)
    m = (H//KH)*((L+15)//16)*16
    acc=reg['edge_accumulator']
    accumulation = (gamma(m,acc['unit_roundoff']) + gamma(2*m+H//KH,2**-53))*sums
    accumulation += m*acc['absolute_underflow_per_addition']/(1-m*acc['unit_roundoff'])
    budget[:] = jumps+accumulation
    if not all(np.isfinite(a).all() for a in (center,budget)):
        raise ValueError('edge interval overflow')
    return dict(reference=center, absolute_tolerance=budget, midpoint_allowance=jumps,
                accumulation_allowance=accumulation, crossing_pairs=crossings,
                padded_additions=m, evidence_class='input-derived CPU interval, not native certification')


def edge_dv_gate(candidate, model):
    a = np.asarray(candidate,np.float64)
    r, bound = model['reference'], model['absolute_tolerance']
    if a.shape != r.shape or a.size == 0:
        raise ValueError('edge shape mismatch')
    finite = bool(np.isfinite(a).all())
    error = abs(a-r)
    good = finite and bool(np.all(error <= bound))
    return dict(verdict='GREEN' if good else 'RED', finite=finite,
                violating_elements=int(np.count_nonzero(~np.isfinite(error) | (error > bound))),
                max_abs=float(error.max()) if finite else None,
                max_derived_atol=float(bound.max()), crossing_pairs=model['crossing_pairs'])
