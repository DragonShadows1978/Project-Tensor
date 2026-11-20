"""Regression tests for attach_vjp failure, concurrency and ownership."""

import gc
import threading
import weakref

import numpy as np
import pytest

import tensor_cuda as tc
from test_bwd_r1_hook import host


def failing_graph():
    x, w = host([1, 2], True), host([3, 4], True)
    h = x.reshape([2])

    def fail(dy, needs):
        raise RuntimeError("k fails")

    k = tc.attach_vjp(host([5, 6]), [w], fail, name="k")
    root = tc.attach_vjp(host([7, 8]), [h, k], lambda dy, needs: (dy, dy))
    with pytest.raises(RuntimeError, match="k fails"):
        root.backward()
    return x, w, h, k, root


def test_host_failed_walk_discards_nonleaf_gradients():
    x, w, h, k, root = failing_graph()
    assert all(t.grad is None for t in (x, w, h, k, root))


@pytest.mark.parametrize("reuse", [True, False], ids=["C1b-reuse", "C1c-fresh"])
def test_host_second_walk_through_intermediate(reuse):
    if reuse:
        x, w, h, _, _ = failing_graph()
        x.zero_grad()
        w.zero_grad()
        # Deliberately reach the next accumulation into h even on r1.
    else:
        x = host([1, 2], True)
        h = x.reshape([2])

    def seed(dy, needs):
        return (host([2., -3.]),)

    tc.attach_vjp(host([9, 10]), [h], seed).backward()
    actual = x.grad.numpy()
    assert actual.size and np.ptp(actual) > 0
    np.testing.assert_array_equal(actual, [2., -3.])


def test_host_consumed_ancestor_refuses_before_seeding():
    x, w = host([1, 2], True), host([3, 4], True)
    p = tc.attach_vjp(host([5, 6]), [x], lambda dy, need: (dy,), name="P")
    p.backward()
    x.zero_grad()
    calls = []

    def second(dy, needs):
        calls.append("ran")
        return (dy, dy)

    root = tc.attach_vjp(host([7, 8]), [p, w], second)
    with pytest.raises(RuntimeError, match=r"attach_vjp\(P\): callback already consumed"):
        root.backward()
    assert calls == []
    assert all(t.grad is None for t in (root, p, w, x))


@pytest.mark.parametrize("slot", range(3))
@pytest.mark.parametrize(("fault", "message"), [
    ("shape", "shape mismatch"), ("dtype", "dtype mismatch"),
    ("frozen", "frozen input needs None"), ("count", "wrong gradient count"),
])
def test_host_validation_all_slots(slot, fault, message):
    inputs = [host([1, 2], not (fault == "frozen" and i == slot)) for i in range(3)]
    calls = []

    def vjp(dy, needs):
        calls.append(slot)
        print(f"validation callback invoked for {fault} slot {slot}", flush=True)
        grads = [dy, dy, dy]
        if fault == "count":
            del grads[slot]
        elif fault == "shape":
            grads[slot] = host([[1, 2]])
        elif fault == "dtype":
            grads[slot] = host([1, 2], dtype="float16")
        return tuple(grads)

    if fault != "count":
        message = f"gradient slot {slot}: {message}"
    root = tc.attach_vjp(host([2, 3]), inputs, vjp)
    with pytest.raises(ValueError, match=message):
        root.backward()
    assert calls == [slot]
    assert all(t.grad is None for t in inputs)


def test_host_post_callback_revision_check():
    x, saved = host([1, 2], True), host([3, 4])
    before = saved.revision

    def vjp(dy, needs):
        assert saved.revision == before
        saved.mark_modified()
        print("post-callback mutation executed", flush=True)
        return (dy, None)

    root = tc.attach_vjp(host([7, 8]), [x, saved], vjp)
    with pytest.raises(RuntimeError, match="saved input slot 1 was modified after attachment"):
        root.backward()
    assert saved.revision == before + 1 and x.grad is None and saved.grad is None


def test_host_active_input_revision_check():
    x = host([1, 2], True)
    calls = []
    root = tc.attach_vjp(host([7, 8]), [x], lambda dy, needs: (calls.append(dy),))
    x.mark_modified()
    with pytest.raises(RuntimeError, match="saved input slot 0 was modified after attachment"):
        root.backward()
    assert calls == [] and x.grad is None


def test_host_external_write_contract():
    doc = tc.attach_vjp.__doc__
    for text in ("only then call attach_vjp", "DLPack re-import", "packed_append", "Clear leaf gradients"):
        assert text in doc
    x = host([1, 2], True)
    value = host([5, 6])
    value.mark_modified()
    result = tc.attach_vjp(value, [x], lambda dy, needs: (host([-2., 3.]),))
    result.backward()
    actual = x.grad.numpy()
    assert actual.size and np.ptp(actual) > 0
    np.testing.assert_array_equal(actual, [-2., 3.])


def test_host_callback_grad_mode_is_local_and_backward_has_one_lane():
    x, other = host([1, 2], True), host([3, 4], True)
    seen = {}

    def worker():
        seen["grad_enabled"] = tc.is_grad_enabled()
        seen["recorded"] = other.reshape([1, 2])
        try:
            other.backward()
        except RuntimeError as error:
            seen["error"] = str(error)

    def vjp(dy, needs):
        assert not tc.is_grad_enabled()
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(timeout=10)
        assert not thread.is_alive()
        assert not tc.is_grad_enabled()
        return (dy,)

    tc.attach_vjp(host([5, 6]), [x], vjp).backward()
    assert seen["grad_enabled"] and seen["recorded"].requires_grad
    assert "only one host dispatch lane" in seen["error"]
    assert "re-entrant" not in seen["error"]
    assert tc.is_grad_enabled() and other.grad is None
    seen["recorded"].backward()
    assert other.grad is not None


def result_cycle():
    x = host([1, 2], True)
    holder = {}

    def vjp(dy, needs):
        assert holder["result"] is not None
        return (host([-2., 3.]),)

    result = tc.attach_vjp(host([3, 4]), [x], vjp)
    holder["result"] = result
    return result, weakref.ref(vjp), x


def test_host_collect_result_capturing_callback():
    result, callback, _ = result_cycle()
    result_ref = weakref.ref(result)
    del result
    gc.collect()
    assert callback() is None and result_ref() is None


def test_host_result_cycle_keeps_callback_for_live_native_descendant():
    result, callback, x = result_cycle()
    child = result.reshape([1, 2])
    del result
    gc.collect()
    assert callback() is not None
    child.backward()
    actual = x.grad.numpy()
    assert actual.size and np.ptp(actual) > 0
    np.testing.assert_array_equal(actual, [-2., 3.])


# Prior art: PyTorch Autograd mechanics saved-output packing and PR #4437
# (2018) sole-owner traversal. This checks the detached value pattern with
# a TensorCUDA native descendant owned by an otherwise unreachable cycle.
@pytest.mark.parametrize("keep_descendant", [False, True], ids=["collect", "backward"])
def test_host_saved_value_with_native_descendant(keep_descendant):
    x = host([1, 2], True)
    value = host([-2., 3.])

    def vjp(dy, needs, saved=value):
        assert needs == (True,) and dy.shape == (2,)
        return (saved,)

    result = tc.attach_vjp(value, [x], vjp)
    descendant = result.reshape([1, 2])
    owner = [descendant]
    owner.append(owner)
    refs = tuple(weakref.ref(obj) for obj in (descendant, vjp, value))
    live = descendant if keep_descendant else None
    del descendant, owner, result, vjp, value
    gc.collect()
    if keep_descendant:
        assert all(ref() is not None for ref in refs)
        live.backward()
        expected = np.array([-2., 3.], dtype=np.float32)
        actual = x.grad.numpy()
        assert actual.size and expected.size and np.ptp(actual) and np.ptp(expected)
        np.testing.assert_array_equal(actual, expected)
        del live
        gc.collect()
    assert all(ref() is None for ref in refs)
