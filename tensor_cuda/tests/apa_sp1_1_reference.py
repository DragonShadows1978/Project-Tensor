"""SP1.1 CPU specifications. Original SP1 mathematical reference stays pinned."""
import numpy as np
import apa_sp1_reference as ref

F = np.float32
PART_KEYS = 128 * 16  # inherited TC_APA_SPLITK_PART_KEYS, checked by G1


def partition_mask(bulk, delta, part_keys=PART_KEYS, lengths=None):
    bulk = np.asarray(bulk, dtype=F)
    result = np.zeros(bulk.shape, bool)
    for first in range(0, bulk.shape[-1], part_keys):
        result[..., first:first+part_keys] = ref.prefix_mask(bulk[..., first:first+part_keys], delta)
    if lengths is not None:
        result &= np.arange(bulk.shape[-1]) < np.asarray(lengths)[..., None]
    return result


def partition_online(bulk, exact, values, delta, partitions, sink=None):
    """Literal per-partition online recurrence and merge; arbitrary ordered subsets.

    Empty partitions are the identity. A partition must preserve key order;
    all nonempty partitions together must cover each key exactly once.
    """
    bulk, exact, values = (np.asarray(x, dtype=F) for x in (bulk, exact, values))
    mask = np.zeros(len(bulk), bool)
    partials = []
    for indices in partitions:
        mx, m, den = F(-np.inf), F(-np.inf), F(0)
        acc = np.zeros(values.shape[-1], F)
        for j in indices:
            mx = max(mx, bulk[j])
            mask[j] = bulk[j] >= F(mx - F(delta))
            score = exact[j] if mask[j] else bulk[j]
            nxt = max(m, score)
            corr, w = np.exp(F(m-nxt)), np.exp(F(score-nxt))
            den = F(den*corr+w)
            acc = acc*corr+w*values[j]
            m = nxt
        if den > 0:
            partials.append((m, den, acc))
    m = max(p[0] for p in partials)
    den, acc = F(0), np.zeros(values.shape[-1], F)
    for pm, pl, pa in partials:
        corr = np.exp(F(pm-m))
        den = F(den+pl*corr)
        acc += pa*corr
    if sink is not None:
        nxt = max(m, F(sink))
        corr = np.exp(F(m-nxt))
        den = F(den*corr+np.exp(F(F(sink)-nxt)))
        acc *= corr
    return acc/den, mask


def fma32(a, b, c):
    """FP32 fused arithmetic using FP64 intermediates (no input upcasting loss).

    For these bounded normal inputs, this matches fmaf; a scalar libm check
    pins the boundary witnesses. Not an emulator of CUDA approximate expf.
    """
    return (np.asarray(a, np.float64)*np.asarray(b, np.float64)+np.asarray(c, np.float64)).astype(F)


def cuda_bulk(q, keys, scale, wcoop=True):
    """Dots in the kernel's lane/FMA/shuffle order, for a query chunk."""
    n, D = q.shape
    if not wcoop:
        dot = np.zeros((n, len(keys)), F)
        for d in range(D):
            dot = fma32(q[:, d, None], keys[None, :, d], dot)
    else:
        lanes = np.zeros((n, len(keys), 32), F)
        for first in range(0, D, 32):
            width = min(32, D-first)
            lanes[..., :width] = fma32(q[:, None, first:first+width],
                                       keys[None, :, first:first+width], lanes[..., :width])
        # shfl_down uses old values for every lane at each stage.
        for offset in (16, 8, 4, 2, 1):
            lanes[..., :32-offset] = lanes[..., :32-offset]+lanes[..., offset:]
        dot = lanes[..., 0]
    return dot*F(scale)


def cuda_zmask(bulk, z, lengths, wcoop=True):
    """128-thread baseline stats, four warp streams for WCOOP; FMA enabled."""
    n, S = bulk.shape
    streams = 4 if wcoop else 128
    sums, sqs = np.zeros((n, streams), F), np.zeros((n, streams), F)
    for first in range(0, S, streams):
        width = min(streams, S-first)
        a = np.where(np.arange(first, first+width)[None, :] < lengths[:, None],
                     np.abs(bulk[:, first:first+width]), F(0))
        sums[:, :width] += a
        sqs[:, :width] = fma32(a, a, sqs[:, :width])
    # Shared red[128]: only lane 0 of each warp contributes in WCOOP.
    red = np.zeros((2, n, 128), F)
    red[0, :, ::32 if wcoop else 1] = sums
    red[1, :, ::32 if wcoop else 1] = sqs
    for off in (64, 32, 16, 8, 4, 2, 1):
        red[..., :off] += red[..., off:2*off]
    mean = red[0, :, 0]/lengths.astype(F)
    var = fma32(-mean, mean, red[1, :, 0]/lengths.astype(F))
    thr = fma32(F(z), np.sqrt(np.maximum(var, F(0))), mean)
    valid = np.arange(S)[None, :] < lengths[:, None]
    return valid & (np.abs(bulk) >= thr[:, None]), thr
