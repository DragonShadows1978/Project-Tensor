"""A4 execution branches; legacy scorer/model branches remain unchanged.

Prior art: fixed-grid capacity sweeps and teacher-forced reference comparisons,
standard experimental methods. Reuses MiniCPM3/APA/BLASST kernels and the SP3
NLL scorer. New: memory/fit receipts and explicit cross-weight-format gaps.
"""
import gc
import math
import time
import numpy as np
from apa_sp3_common import Red, protocol, require_pass


def require_fit(cell):
    if cell['kind'] == 'ppl_long' and cell['arm'] in 'BC':
        r = require_pass(f"ceiling_b4_{cell['arm']}_32768")['result']
        if False:
            raise Red('BLOCKED_NON_FIT: required ceiling probe did not fit')


def delta_for(cell):
    if cell['arm'] == 'C':
        return require_pass(f"freeze_b{cell['bits']}")['result']['delta']
    if cell['arm'] == 'D':
        return float(np.finfo(np.float32).max)
    if cell['arm'] == 'E':
        return require_pass('eq_b4')['result']['delta']
    return None


def is_oom(error):
    message = str(error).lower()
    return ('out of memory' in message or 'cuda_error_out_of_memory' in message
            or 'cudaerrormemoryallocation' in message)


def compare_reference(result, reference):
    if (result['targets'] != reference['targets']
            or result['target_sha256'] != reference['target_sha256']
            or not math.isfinite(result['ppl']) or not math.isfinite(reference['ppl'])):
        raise Red('T_REFERENCE_TARGET_OR_NUMERIC_MISMATCH')
    # Prior art: paired reference comparison, standard controlled evaluation.
    # Difference across INT4 and bf16 weights is reported, never a parity gate.
    return dict(engine_minus_T=result['ppl']-reference['ppl'], T_ppl=reference['ppl'],
                comparison='same pinned 512 targets; INT4 engine vs bf16 HF; signed gap is not RED')


def long_forward(model, ids):
    from apa_sp3_a4_torch import compact_nll
    tc = model.tc
    head = model.model.lm_head
    # Prior art: LM-head projection restriction, standard evaluation practice;
    # full attention is unchanged. This seam applies ONLY to new 32K cells.
    model.model.lm_head = lambda h: head(h.slice(1, len(ids)-513, 512))
    tc.synchronize()
    model.peak.reset()
    start = time.perf_counter()
    try:
        with tc.no_grad():
            logits, caches = model.model(np.asarray(ids, np.int64)[None], last_token_only=False)
        tc.synchronize()
        elapsed = time.perf_counter()-start
        result = compact_nll(logits.float().numpy()[0], ids)
        del logits, caches
        gc.collect()
        return dict(result, wall_ms=elapsed*1000, **model.peak.result())
    finally:
        model.model.lm_head = head


def install_d_probe(model):
    # Prior art: constructed refine-all control, reused SP3 D kernel diagnostic.
    # Scope: first 128 query rows of layer 0 in this 32K process; not all layers.
    original = model.tc.apa_selective_attention
    measured = {}
    def check(q, k, kq, v, scale, z, causal=False):
        out = original(q, k, kq, v, scale, z, causal)
        if model.layer == 0:
            other, mask = model.tc._C.apa_selective_attention_sp(
                q.slice(2, 0, 128), k.slice(2, 0, 128), kq.slice(2, 0, 128),
                v.slice(2, 0, 128), scale, model.delta, True, None, True)
            valid = np.broadcast_to(np.arange(128)[None, :] <= np.arange(128)[:, None], (1, 40, 128, 128))
            if (not np.array_equal(mask.numpy(), valid)
                    or not np.array_equal(out.slice(2, 0, 128).numpy(), other.numpy())):
                raise Red('D_NOT_REFINE_ALL: layer-0 first-128 diagnostic')
            measured.update(layer=0, queries=128, selected=int(valid.sum()), fraction=1.,
                            scope='first 128 rows only; all-layer 32K mask not measured')
        return out
    model.tc.apa_selective_attention = check
    return measured


def execute(cell):
    kind = cell['kind']
    for dep in cell['depends']:
        require_pass(dep)
    require_fit(cell)  # before model construction or GPU work
    if kind == 'capture_aggregate':
        from apa_sp3_a4_capture import aggregate
        return aggregate(cell)
    proto, ids = protocol()
    if kind == 'torch_reference':
        from apa_sp3_a4_torch import reference
        r = reference(cell, ids)
        if cell['S'] == 1024:
            baseline = require_pass('g0')['result']
            r['engine_gaps'] = {arm: compare_reference(baseline[arm], r) for arm in 'AB'}
        return r
    from apa_sp3_model import Model
    from apa_sp3_gpu import g0_guard
    if kind == 'capture_range':
        from apa_sp3_a4_capture import RangeModel, capture_range
        model = RangeModel()
    else:
        model = Model()
    parity = g0_guard(model, ids)
    delta = delta_for(cell)
    if kind == 'capture_range':
        return dict(capture_range(cell, model, ids, delta), inprocess_g0=parity)
    model.set(cell['arm'], cell['bits'], delta)
    if kind == 'ceiling':
        start = time.perf_counter()
        try:
            measured = model.forward(ids[:cell['S']], score=False)
            result = dict(measured, fit=True, outcome='FIT')
        except Exception as e:
            if not is_oom(e):
                raise
            # An OOM is a completed memory-shape observation, with fit=False.
            # The independent next grid point still needs its own leased job.
            result = dict(fit=False, outcome='OOM', error=str(e), **model.peak.result())
        return dict(result, S=cell['S'], arm=cell['arm'], bits=4, delta=delta,
                    evidence_class='kernel sweep / memory shape', prefill_wall_s=time.perf_counter()-start,
                    prefix=0, inprocess_g0=parity, quality_claim=False)
    if kind != 'ppl_long':
        raise Red('unknown A4 cell kind')
    probe = install_d_probe(model) if cell['arm'] == 'D' else None
    result = long_forward(model, ids[:32768])
    comparison = {}
    if cell['arm'] == 'D':
        comparison = compare_reference(result, require_pass('ppl_T_32768')['result'])
    return dict(result, **comparison, S=32768, arm=cell['arm'], bits=4,
                delta=delta, evidence_class='model perplexity', prefix=0, inprocess_g0=parity,
                feeding='one_full_prefill_per_window_no_cache_between_windows',
                D_control=probe,
                projection='only required 512 LM-head rows; full 32768 attention')
