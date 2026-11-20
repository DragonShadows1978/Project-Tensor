"""First-order attached VJPs, separated into host and CUDA cases."""

import os
import weakref

import numpy as np
import pytest

import tensor_cuda as tc


gpu = pytest.mark.skipif(os.environ.get("CUDA_VISIBLE_DEVICES") == "", reason="CUDA disabled for CPU lane")


def host(values, active=False, dtype="float32"):
    return tc.tensor(values, device="cpu", dtype=dtype, requires_grad=active)


def test_host_single_inverse_reshape():
    x = host([[2, -3, 8], [7, 1, -2]], True)
    reference = x.reshape([3, 2])
    value = tc._C._core_clone(reference)

    def vjp(dy, needs):
        print("inverse reshape callback executed", flush=True)
        assert needs == (True,) and not dy.requires_grad and not tc.is_grad_enabled()
        return (dy.reshape(x.shape),)

    y = tc.attach_vjp(value, [x], vjp, name="inverse reshape")
    assert y.data_ptr() == value.data_ptr() and y.requires_grad and not value.requires_grad
    got = y.numpy()
    assert got.size and np.ptp(got)
    np.testing.assert_array_equal(got, reference.numpy())
    reference.backward()
    expected = x.grad.numpy()
    x.zero_grad()
    y.backward()
    assert x.grad is not None, "attached gradient was dropped"
    np.testing.assert_array_equal(x.grad.numpy(), expected)


def test_host_ordered_active_slots():
    a, b = host([1, 2], True), host([3, 4], True)

    def vjp(dy, needs):
        print("ordered active-slot callback executed", flush=True)
        assert needs == (True, True)
        return host([2, -3]), host([-11, 7])

    tc.attach_vjp(host([1, 3]), (a, b), vjp).backward()
    np.testing.assert_array_equal(a.grad.numpy(), [2, -3])
    np.testing.assert_array_equal(b.grad.numpy(), [-11, 7])


def test_host_frozen_saved_slot_and_structural_zero():
    x = host([1, 2], True)
    saved, zero = host([2, -3]), host([3, 4], True)

    def vjp(dy, needs):
        assert needs == (True, False, True)
        return host(dy.numpy() * saved.numpy()), None, None

    value = host([2, -6])
    tc.attach_vjp(value, [x, saved, zero], vjp).backward()
    np.testing.assert_array_equal(x.grad.numpy(), [2, -3])
    assert saved.grad is None and zero.grad is None


@pytest.mark.parametrize("bypass", ["no_grad", "frozen"])
def test_host_bypass_identity_and_no_retain(bypass):
    value = host([1, 2])
    x = host([3, 4], bypass == "no_grad")

    def vjp(*_):
        pytest.fail("bypassed callback ran")

    ref = weakref.ref(vjp)
    if bypass == "no_grad":
        with tc.no_grad():
            y = tc.attach_vjp(value, [x], vjp)
    else:
        y = tc.attach_vjp(value, [x], vjp)
    del vjp
    assert y is value and ref() is None and not y.requires_grad


@pytest.mark.parametrize(("fault", "error", "message"), [
    ("shape", ValueError, "gradient slot 1: shape mismatch"),
    ("dtype", ValueError, "gradient slot 1: dtype mismatch"),
    ("count", ValueError, "wrong gradient count"),
    ("frozen", ValueError, "gradient slot 1: frozen input needs None"),
    ("element", TypeError, "Tensor or None"),
    ("container", TypeError, "return a tuple"),
])
def test_host_validation_is_before_accumulation(fault, error, message):
    a = host([1, 2], True)
    b = host([3, 4], fault != "frozen")

    def vjp(dy, needs):
        return {
            "shape": (dy, host([[3, 4]])),
            "dtype": (dy, host([3, 4], dtype="float16")),
            "count": (dy,),
            "frozen": (dy, dy),
            "element": (dy, 1),
            "container": [dy, dy],
        }[fault]

    name = "".join(["temporary ", "name"])
    y = tc.attach_vjp(host([1, 3]), [a, b], vjp, name=name)
    del name
    with pytest.raises(error, match=message) as raised:
        y.backward()
    if error is ValueError:
        assert "temporary name" in str(raised.value)
    assert a.grad is None and b.grad is None


def test_host_value_must_be_detached_and_separate():
    x = host([1, 2], True)
    with pytest.raises(ValueError, match="fresh detached output"):
        tc.attach_vjp(x.reshape([2, 1]), [x], lambda dy, need: (dy,))
    with pytest.raises(ValueError, match="storage separate"):
        tc.attach_vjp(x.detach(), [x], lambda dy, need: (dy,))


def test_host_callback_exception_restores_mode():
    class CallbackError(Exception):
        pass

    def vjp(*_):
        assert not tc.is_grad_enabled()
        raise CallbackError("VJP exception sentinel")

    x = host([1, 2], True)
    y = tc.attach_vjp(host([1, 3]), [x], vjp)
    with pytest.raises(CallbackError, match="sentinel"):
        y.backward()
    assert tc.is_grad_enabled() and x.grad is None


@pytest.mark.parametrize("target", ["frozen_save", "value"])
def test_host_revision_detects_alias_mutation(target):
    x, saved, value = host([1, 2], True), host([2, 3]), host([2, 6])
    calls = []

    def vjp(dy, needs):
        calls.append(1)
        print("stale callback executed", flush=True)
        return (dy, None)

    y = tc.attach_vjp(value, [x, saved], vjp)
    alias = (saved if target == "frozen_save" else value).reshape([2, 1])
    revision = alias.revision
    alias.mark_modified()
    assert alias.revision == revision + 1
    print("registered storage revision changed", alias.revision, flush=True)
    with pytest.raises(RuntimeError, match="was modified after attachment"):
        y.backward()
    assert not calls and x.grad is None


@pytest.mark.parametrize("exit_path", ["backward", "abandoned", "exception"])
def test_host_callable_lifetime(exit_path):
    def vjp(dy, needs):
        if exit_path == "exception":
            raise RuntimeError("lifetime sentinel")
        return (dy,)

    ref = weakref.ref(vjp)
    x = host([1, 2], True)
    y = tc.attach_vjp(host([3, 4]), [x], vjp)
    del vjp
    assert ref() is not None
    if exit_path == "abandoned":
        del y
    elif exit_path == "exception":
        # Keep only the exception message, avoiding a Python traceback that
        # legitimately retains the callback frame and possibly the callable.
        with pytest.raises(RuntimeError, match="lifetime sentinel"):
            y.backward()
    else:
        y.backward()
        with pytest.raises(RuntimeError, match="callback already consumed"):
            y.backward()
    assert ref() is None


def test_host_callback_reentrancy_refused():
    x, other = host([1, 2], True), host([3, 4], True)

    def vjp(dy, needs):
        other.backward()
        return (dy,)

    y = tc.attach_vjp(host([1, 3]), [x], vjp)
    with pytest.raises(RuntimeError, match="re-entrant backward"):
        y.backward()
    assert tc.is_grad_enabled() and x.grad is None and other.grad is None
    other.backward()
    assert other.grad is not None


@gpu
def test_gpu_seed_fanin_and_repeated_slots():
    stored = np.array([2, -3, 4, 5], dtype=np.float32)
    seed = tc.tensor([2, -5, 7, 1])
    x = tc.tensor(stored, requires_grad=True)
    with tc.no_grad():
        value = x * x
    saved = x.detach()

    def vjp(dy, needs):
        assert needs == (True, True) and not tc.is_grad_enabled()
        return (dy * saved, dy * saved)

    y = tc.attach_vjp(value, (x, x), vjp, name="square")
    ((y + y + x) * seed).sum().backward()
    expected = (4 * stored + 1) * seed.numpy()
    reference = tc.tensor(stored, requires_grad=True)
    ((reference * reference * 2 + reference) * seed).sum().backward()
    assert expected.size and np.ptp(expected)
    np.testing.assert_array_equal(reference.grad.numpy(), expected)
    np.testing.assert_array_equal(x.grad.numpy(), expected)


@gpu
@pytest.mark.parametrize("slot", range(3))
def test_gpu_wrong_device_is_atomic(slot):
    inputs = [tc.tensor([1, 2], requires_grad=True) for _ in range(3)]

    def vjp(dy, needs):
        grads = [dy, dy, dy]
        grads[slot] = host([2, 3])
        return tuple(grads)

    with pytest.raises(ValueError, match=f"gradient slot {slot}: device mismatch"):
        tc.attach_vjp(tc.tensor([2, 4]), inputs, vjp).backward()
    assert all(t.grad is None for t in inputs)
    with pytest.raises(ValueError, match="value's device"):
        tc.attach_vjp(tc.tensor([2, 4]), [host([1, 2], True)], lambda dy, needs: (dy,))


@gpu
@pytest.mark.parametrize("write", ["engine", "dlpack"])
def test_gpu_actual_write_rejected(write):
    x = tc.tensor([1, 2], requires_grad=True)
    y = tc.attach_vjp(tc.tensor([1, 4]), [x], lambda *_: pytest.fail("stale callback called"))
    if write == "engine":
        x.fill_(7)
    else:
        alias = tc.from_dlpack(x)
        rev = x.revision
        assert alias.data_ptr() == x.data_ptr()
        alias.fill_(7)
        np.testing.assert_array_equal(x.numpy(), [7, 7])
        assert x.revision == rev
        x.mark_modified()
    with pytest.raises(RuntimeError, match="saved input slot 0 was modified"):
        y.backward()
    assert x.grad is None


@gpu
@pytest.mark.parametrize("run_backward", [False, True])
def test_gpu_external_value_storage_lifetime(run_backward):
    from test_dlpack import _Producer

    x = tc.tensor([1, 2], requires_grad=True)
    owner = tc.tensor([5, -4])
    producer = _Producer(owner, owner.shape, (1,), device_id=0)
    detached = tc.from_dlpack(producer.capsule())
    y = tc.attach_vjp(detached, [x], lambda dy, _: (dy * 3,))
    assert y.data_ptr() == owner.data_ptr()
    del detached, owner
    assert producer.deleter_calls == 0
    np.testing.assert_array_equal(y.numpy(), [5, -4])
    if run_backward:
        (y * tc.tensor([2, -7])).sum().backward()
        np.testing.assert_array_equal(x.grad.numpy(), [6, -21])
    assert producer.deleter_calls == 0
    del y
    assert producer.deleter_calls == 1


@gpu
def test_gpu_checkpoint_replay_through_hook():
    x = tc.tensor([2, -3, 5], requires_grad=True)
    seed = tc.tensor([3, 2, -7])
    modes = []

    def forward(a):
        modes.append(tc.is_grad_enabled())
        saved = a.detach()
        with tc.no_grad():
            value = a * a
        return tc.attach_vjp(value, [a], lambda dy, _: (dy * (saved * 2),))

    (tc.checkpoint(forward, x) * seed).sum().backward()
    assert modes == [False, True]
    np.testing.assert_array_equal(x.grad.numpy(), [12, -12, -70])
