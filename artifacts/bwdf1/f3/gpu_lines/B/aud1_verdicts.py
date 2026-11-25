"""BWD-AUD1 post-device logic of the staged lines (readback comparison and verdicts), pure NumPy, no tensor_cuda.

Driven on constructed arrays by post_selftest.py (L48) before staging. Expectations come from the forward's definition
(tensor_cuda/src/matmul.cu:25-81) and the gradient of a sum: zeros for an empty contraction, A^T g and g op(B)^T
otherwise. Prior art: none beyond the textbook matmul adjoint (Griewank and Walther 2008) known to me.
"""
import numpy as np


def bits_nonzero(arr):
    """Count elements whose bit pattern is not +0.0 (exact: a -0.0, a NaN or any value counts)."""
    a = np.ascontiguousarray(arr)
    view = {2: np.uint16, 4: np.uint32, 8: np.uint64}[a.dtype.itemsize]
    return int(np.count_nonzero(a.view(view)))


def f01_case(label, dtype, grad, expected_shape, sentinel):
    """grad: the host readback (numpy) of the gradient that must be all +0.0; None when undefined."""
    if grad is None:
        return {'case': label, 'dtype': dtype, 'result': 'GRAD-UNDEFINED'}
    out = {'case': label, 'dtype': dtype, 'shape': list(grad.shape), 'expected_shape': list(expected_shape)}
    if tuple(grad.shape) != tuple(expected_shape):
        out['result'] = 'SHAPE-MISMATCH'
        return out
    nz = bits_nonzero(grad)
    out['nonzero_elements'] = nz
    out['sentinel_elements'] = int(np.count_nonzero(grad.astype(np.float64) == sentinel))
    out['sample'] = [float(v) for v in grad.reshape(-1)[:6]]
    out['result'] = 'ZEROS' if nz == 0 else 'NOT-ZERO'
    return out


def f01_verdict(cases):
    if any(c.get('result') == 'ERROR' for c in cases):
        return 'ERROR: a case raised outside the expected refusal; see cases'
    low = [c for c in cases if c['dtype'] in ('float16', 'bfloat16')]
    ctrl = [c for c in cases if c['dtype'] == 'float32']
    if any(c['result'] != 'ZEROS' for c in ctrl):
        return 'CONTROL-FAIL: the fp32 path did not return exact zeros'
    if any(c['result'] in ('NOT-ZERO', 'SHAPE-MISMATCH', 'GRAD-UNDEFINED') for c in low):
        return 'DEFECT-CONFIRMED: a float16/bfloat16 matmul gradient with an empty contraction is not exact zeros'
    return ('NOT-DETECTED: every float16/bfloat16 gradient read back as zeros (the pool may have handed back '
            'zeroed memory; reading still shows lda/ldb = 0 reaching cuBLAS with the status discarded)')


def f02_case(label, dtype, y, y_ref, backward_error, grads):
    """y: forward readback; y_ref: host product of the FIRST stored B matrices (the forward's stride rule)."""
    out = {'case': label, 'dtype': dtype}
    tol = 1e-5 if dtype == 'float32' else 5e-2
    out['forward_matches_first_matrices'] = bool(np.allclose(y.astype(np.float64), y_ref, rtol=tol, atol=tol))
    if backward_error is not None:
        out['backward'] = 'RAISED'
        out['message'] = backward_error
        out['grads_after_refusal'] = {k: (None if v is None else list(v.shape)) for k, v in grads.items()}
        out['result'] = 'REFUSED-AFTER-FORWARD' if out['forward_matches_first_matrices'] else 'FORWARD-UNEXPECTED'
    else:
        out['backward'] = 'RETURNED'
        out['result'] = 'COMPUTED'
    return out


def f02_verdict(cases):
    if any(c.get('result') == 'ERROR' for c in cases):
        return 'ERROR: a case raised outside the expected refusal; see cases'
    if all(c['result'] == 'REFUSED-AFTER-FORWARD' for c in cases):
        return ('DEFECT-CONFIRMED: the forward accepted each layout and produced the first-matrices product; the '
                'backward refused it after the forward had run')
    if any(c['result'] == 'FORWARD-UNEXPECTED' for c in cases):
        return 'INCONCLUSIVE: the forward did not match the stride rule read at matmul.cu:36-50'
    return 'NOT-CONFIRMED: at least one layout was differentiated'


def f03_case(label, dtype, backward_error, ga, gb, a_shape, b_shape):
    out = {'case': label, 'dtype': dtype}
    if backward_error is not None:
        out['backward'] = 'RAISED'
        out['message'] = backward_error
        out['result'] = 'REFUSED'
        return out
    ok_a = ga is not None and tuple(ga.shape) == tuple(a_shape)
    ok_b = gb is not None and tuple(gb.shape) == tuple(b_shape) and bits_nonzero(gb) == 0
    out['backward'] = 'RETURNED'
    out['result'] = 'ZEROS' if ok_a and ok_b else 'WRONG'
    return out


def f03_verdict(cases):
    if any(c.get('result') == 'ERROR' for c in cases):
        return 'ERROR: a case raised outside the expected refusal; see cases'
    if any(c['result'] == 'REFUSED' and 'element count mismatch' in c.get('message', '') for c in cases):
        return ('DEFECT-CONFIRMED: a zero-extent batch forward was accepted and its backward raised the reshape '
                'element-count error instead of returning dB = 0')
    if all(c['result'] == 'ZEROS' for c in cases):
        return 'NOT-CONFIRMED: every case returned the zero gradients'
    return 'OTHER: see cases'


def error_case(label, dtype, exc):
    return {'case': label, 'dtype': dtype, 'result': 'ERROR', 'message': f'{type(exc).__name__}: {exc}'}
