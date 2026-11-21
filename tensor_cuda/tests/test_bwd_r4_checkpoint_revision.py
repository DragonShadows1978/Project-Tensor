"""BWD-R4a r5b item 1: checkpoint refuses to recompute over an input changed after its forward.

Host cases build the checkpointed function from attach_vjp with NumPy values, so they run on the CPU lane. CUDA cases
change the input with a TensorCUDA in-place op (fill_) and with an optimizer step (SGD); the lead runs them.
Prior art: PyTorch's saved-tensor version counter (Paszke et al., 2019) and gradient checkpointing (Chen et al., 2016).
Ours: these cases, written against TensorCUDA's Storage revision.
"""

import ctypes
import os

import numpy as np
import pytest

import tensor_cuda as tc


gpu = pytest.mark.skipif(os.environ.get("CUDA_VISIBLE_DEVICES") == "", reason="CUDA disabled for CPU lane")
REFUSAL = "checkpoint: saved input slot {} was modified after the checkpointed forward"


def host(values, active=False):
    return tc.tensor(np.asarray(values, dtype=np.float32), device="cpu", requires_grad=active)


def checked(array):
    # Compared arrays are non-empty and nonconstant before any comparison.
    assert array.size and np.ptp(array) > 0
    return array


def product(calls):
    """y = a * b with an exact NumPy VJP; records each call's operand values."""

    def fn(a, b):
        av, bv = a.numpy(), b.numpy()
        calls.append((av.tolist(), bv.tolist(), tc.is_grad_enabled()))

        def vjp(dy, needs):
            g = dy.numpy()
            return (host(g * bv) if needs[0] else None, host(g * av) if needs[1] else None)

        return tc.attach_vjp(host(av * bv), [a, b], vjp, name="host product")

    return fn


def write_first(t, value):
    # An external write through the raw pointer, then the documented mark_modified() contract.
    ctypes.c_float.from_address(t.data_ptr()).value = value
    t.mark_modified()


@pytest.mark.parametrize("slot", (0, 1))
def test_host_checkpoint_refuses_changed_input(slot):
    a = host([1.0, 2.0, 3.0], True)
    b = host([4.0, -5.0, 6.0], slot == 0)  # slot 1 changes a frozen saved input; slot 0 an active one
    calls = []
    y = tc.checkpoint(product(calls), a, b)
    checked(y.numpy())
    target = (a, b)[slot]
    revision = target.revision
    write_first(target, 7.0)
    assert target.revision != revision
    with pytest.raises(RuntimeError, match=REFUSAL.format(slot)):
        y.backward()
    # Refused before any recomputation: one forward call, no replay, no gradient accumulated.
    assert [grad_mode for *_, grad_mode in calls] == [False]
    assert a.grad is None and b.grad is None


def test_host_checkpoint_refuses_mark_modified_without_value_change():
    a = host([1.0, 2.0, 3.0], True)
    b = host([4.0, -5.0, 6.0])
    calls = []
    y = tc.checkpoint(product(calls), a, b)
    a.mark_modified()
    with pytest.raises(RuntimeError, match=REFUSAL.format(0)):
        y.backward()
    assert len(calls) == 1 and a.grad is None


def test_host_checkpoint_unchanged_inputs_give_exact_gradients():
    av, bv = [1.0, 2.0, 3.0], [4.0, -5.0, 6.0]
    a, b = host(av, True), host(bv, True)
    revisions = (a.revision, b.revision)
    calls = []
    y = tc.checkpoint(product(calls), a, b)
    checked(y.numpy())
    y.backward()
    assert (a.revision, b.revision) == revisions
    # Forward under no_grad, then exactly one replay with grad enabled on the same values.
    assert calls == [(av, bv, False), (av, bv, True)]
    np.testing.assert_array_equal(checked(a.grad.numpy()), np.asarray(bv, dtype=np.float32))
    np.testing.assert_array_equal(checked(b.grad.numpy()), np.asarray(av, dtype=np.float32))


@gpu
def test_gpu_checkpoint_refuses_inplace_fill():
    x = tc.tensor([2.0, -3.0, 5.0], requires_grad=True)
    seed = tc.tensor([3.0, 2.0, -7.0])
    replays = []

    def square(t):
        replays.append(tc.is_grad_enabled())
        return t * t

    loss = (tc.checkpoint(square, x) * seed).sum()
    x.fill_(1.0)
    with pytest.raises(RuntimeError, match=REFUSAL.format(0)):
        loss.backward()
    assert replays == [False] and x.grad is None


@gpu
def test_gpu_checkpoint_refuses_optimizer_step():
    w = tc.tensor([0.5, -1.5, 2.0], requires_grad=True)
    x = tc.tensor([2.0, -3.0, 5.0])
    seed = tc.tensor([3.0, 2.0, -7.0])
    optimizer = tc.optim.SGD([w], lr=0.25)
    (w * x * seed).sum().backward()  # a gradient for the step, outside any checkpoint
    loss = (tc.checkpoint(lambda a, b: a * b, x, w) * seed).sum()
    revision = w.revision
    optimizer.step()
    assert w.revision != revision
    with pytest.raises(RuntimeError, match=REFUSAL.format(1)):
        loss.backward()


@gpu
def test_gpu_checkpoint_unchanged_matches_plain_bitwise():
    values = np.array([[0.75, -1.25, 2.5], [3.0, -0.5, 1.75]], dtype=np.float32)
    seed = tc.tensor(np.array([[3.0, 2.0, -7.0], [1.0, -4.0, 0.5]], dtype=np.float32))
    grads = []
    for use_checkpoint in (False, True):
        x = tc.tensor(values, requires_grad=True)

        def body(t):
            return (t * t + t) * t

        y = tc.checkpoint(body, x) if use_checkpoint else body(x)
        (y * seed).sum().backward()
        grads.append(checked(x.grad.numpy()))
    assert grads[0].view(np.uint32).tolist() == grads[1].view(np.uint32).tolist()


# BWD-R4a r5d (BWD-REV4 re-check BWDR5R-F01, F02). The revision snapshot is taken before fn runs.
# Case B and case C follow the reviewer's probe (rev4 artifacts/bwd_rev4/recheck_r5/checks/r5_checkpoint_guard_probe.py).
FORWARD_REFUSAL = "checkpoint: fn modified input slot {} in place; a checkpointed function must not write its inputs"


def writes_own_input(calls):
    """Case B: fn reads a, then writes a[0] += 1 with mark_modified() before returning."""

    def fn(a, b):
        av, bv = a.numpy().copy(), b.numpy().copy()
        calls.append((av.tolist(), tc.is_grad_enabled()))
        write_first(a, float(av[0]) + 1.0)

        def vjp(dy, needs):
            g = dy.numpy()
            return (host(g * bv) if needs[0] else None, host(g * av) if needs[1] else None)

        return tc.attach_vjp(host(av * bv), [a, b], vjp, name="writes own input")

    return fn


def test_host_checkpoint_refuses_fn_writing_its_own_input():
    a = host([1.0, 2.0, 3.0], True)
    b = host([4.0, -5.0, 6.0], True)
    calls = []
    with pytest.raises(RuntimeError, match=FORWARD_REFUSAL.format(0)):
        tc.checkpoint(writes_own_input(calls), a, b)
    # Refused in the forward: fn ran once, so the input carries exactly one write and no replay can repeat it.
    assert calls == [([1.0, 2.0, 3.0], False)]
    np.testing.assert_array_equal(checked(a.numpy()), np.array([2.0, 2.0, 3.0], dtype=np.float32))
    assert a.grad is None and b.grad is None


def test_host_checkpoint_accepts_fn_writing_a_fresh_tensor():
    av, bv = [1.0, 2.0, 3.0], [4.0, -5.0, 6.0]
    a, b = host(av, True), host(bv, True)
    revisions = (a.revision, b.revision)
    calls = []

    def fn(p, q):
        pv, qv = p.numpy(), q.numpy()
        calls.append(tc.is_grad_enabled())
        fresh = host(np.zeros(3))  # created inside fn, written in place, then marked: not an input
        ctypes.memmove(fresh.data_ptr(), (pv * qv).astype(np.float32).ctypes.data, 12)
        fresh.mark_modified()

        def vjp(dy, needs):
            g = dy.numpy()
            return (host(g * qv) if needs[0] else None, host(g * pv) if needs[1] else None)

        return tc.attach_vjp(fresh, [p, q], vjp, name="fresh output")

    y = tc.checkpoint(fn, a, b)
    np.testing.assert_array_equal(checked(y.numpy()), np.asarray(av, np.float32) * np.asarray(bv, np.float32))
    y.backward()
    assert calls == [False, True] and (a.revision, b.revision) == revisions
    np.testing.assert_array_equal(checked(a.grad.numpy()), np.asarray(bv, dtype=np.float32))
    np.testing.assert_array_equal(checked(b.grad.numpy()), np.asarray(av, dtype=np.float32))


def test_host_checkpoint_closure_captured_tensor_is_outside_guard_documented_limit():
    # BWDR5R-F02, recorded as the documented contract: every tensor fn reads must be passed in inputs. A tensor fn
    # captures in a closure is neither saved nor guarded, so changing it after the forward is NOT refused and the
    # recomputation uses its new value. This test pins today's behaviour; it does not endorse it.
    a = host([1.0, 2.0, 3.0], True)
    w = host([4.0, -5.0, 6.0])
    calls = []
    y = tc.checkpoint(lambda p: product(calls)(p, w), a)
    write_first(w, 40.0)
    y.backward()  # no refusal
    np.testing.assert_array_equal(checked(a.grad.numpy()), np.array([40.0, -5.0, 6.0], dtype=np.float32))
    assert [mode for *_, mode in calls] == [False, True]


def test_host_checkpoint_restores_grad_mode_after_invalid_replay_return():
    x = host([1.0, 2.0, 3.0], True)
    modes = []

    # BWD-REV4AX F02 (2026): use the reviewed invalid-return probe through the shipped replay binding.
    def fn(t):
        modes.append(tc.is_grad_enabled())
        return {"invalid": t} if modes[-1] else t

    y = tc.checkpoint(fn, x)
    with tc.no_grad():
        assert not tc.is_grad_enabled()
        with pytest.raises(RuntimeError, match=r"Unable to cast.*dict"):
            y.backward()
        assert modes == [False, True]
        assert not tc.is_grad_enabled(), "checkpoint replay cast leaked grad mode"
