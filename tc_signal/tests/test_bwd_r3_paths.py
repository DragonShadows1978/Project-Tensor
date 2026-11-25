"""The real planners and BWD-R1 hook at both convolution entry points."""
from itertools import product
import hashlib

import numpy as np
import pytest
import tc_signal as ts
import tensor_cuda as tc

from bwd_r3_support.cases import exact, gradients, require_gpu, synthetic, tracked

OPS = ("conv1d", "conv_transpose1d")


class Producer:
    def __init__(self, tensor):
        self.tensor = tensor
        self.events = []

    def __dlpack_device__(self):
        self.events.append(("device",))
        return self.tensor.__dlpack_device__()

    def __dlpack__(self, **kwargs):
        self.events.append(("export", kwargs))
        return self.tensor.__dlpack__(**kwargs)


def operands(active=None, device="cpu"):
    return tuple(tc.tensor(value, device=device, requires_grad=i == active) for i, value in enumerate(
        (np.arange(16, dtype=np.float32).reshape(1, 2, 8),
         np.arange(12, dtype=np.float32).reshape(2, 2, 3), [2.0, -4.0])))


@pytest.mark.parametrize("op,slot", product(OPS, range(3)))
def test_host_tracked_refuses_out(op, slot):
    args = operands(slot)
    with pytest.raises(ValueError, match="out= is unsupported when recording gradients"):
        getattr(ts, op)(*args, out=args[0])


@pytest.mark.parametrize("op,slot", product(OPS, range(3)))
def test_host_tracked_refuses_foreign(op, slot):
    args = list(operands((slot + 1) % 3))
    foreign = Producer(args[slot])
    args[slot] = foreign
    with pytest.raises(TypeError, match="requires TensorCUDA operands"):
        getattr(ts, op)(*args)
    assert not foreign.events


@pytest.mark.parametrize("op", OPS)
def test_host_real_planner_and_no_grad(op):
    fn = getattr(ts, op)
    args = operands(0)
    with pytest.raises(BufferError, match="CUDA"):
        fn(*args)
    with tc.no_grad(), pytest.raises(BufferError, match="CUDA"):
        fn(*args, out=args[0])
    # BWD-F1 H1 (lead ruling (a), 2026-11-23): frozen inputs with an out= that requires grad are refused before the
    # planner, as on the tracked path.
    with pytest.raises(RuntimeError, match=f"tc_signal: {op}: out cannot require grad"):
        fn(*operands(), out=args[0])


F10 = [
    ("layer_norm", lambda t, slot: ts.layer_norm(*[t if i == slot else None for i in range(3)]), 3),
    ("group_norm", lambda t, slot: ts.group_norm(t if slot == 0 else None, 1,
                                               t if slot == 1 else None, t if slot == 2 else None), 3),
    ("lstm", lambda t, slot: ts.lstm(*[t if i == slot else None for i in range(4)], hidden_size=1,
                                   **{name: t if i + 4 == slot else None for i, name in
                                      enumerate(("initial_h", "initial_c", "sequence_lens"))}), 7),
    ("rint", lambda t, slot: ts.rint(t), 1),
]


@pytest.mark.parametrize("name,call,slot", [
    pytest.param(name, call, slot, id=f"{name}-slot{slot}") for name, call, count in F10 for slot in range(count)
])
def test_host_f10_no_backward(name, call, slot):
    t = operands(0)[0]
    with pytest.raises(RuntimeError, match=f"{name}: no backward yet"):
        call(t, slot)
    # An untracked call reaches the real planner, which sees missing or CPU-only operands.
    with tc.no_grad(), pytest.raises((BufferError, TypeError), match="CUDA|required|None"):
        call(t, slot)
    t.requires_grad = False
    with pytest.raises((BufferError, TypeError), match="CUDA|required|None"):
        call(t, slot)


@pytest.mark.parametrize("name,container,slot", [
    ("layer_norm", None, 0), ("group_norm", None, 0), ("rint", None, 0),
    *(("lstm", container, slot) for container, slot in product((tuple, list), range(3))),
])
def test_host_f06_destination(name, container, slot):
    frozen = operands()[0]
    dest = operands(0)[0]
    out = dest if container is None else container(dest if i == slot else frozen for i in range(3))
    calls = {
        "layer_norm": lambda: ts.layer_norm(frozen, out=out),
        "group_norm": lambda: ts.group_norm(frozen, 1, out=out),
        "rint": lambda: ts.rint(frozen, out=out),
        "lstm": lambda: ts.lstm(frozen, frozen, frozen, hidden_size=1, out=out),
    }
    call = calls[name]
    with pytest.raises(RuntimeError, match=f"^tc_signal: {name}: out cannot require grad$"):
        call()
    with tc.no_grad(), pytest.raises(BufferError, match="CUDA"):
        call()
    dest.requires_grad = False
    with pytest.raises(BufferError, match="CUDA"):
        call()


@pytest.mark.parametrize("op,slot", product(OPS, range(3)))
def test_gpu_active_slots(op, slot):
    require_gpu()
    case = synthetic(op, "active-slots")
    all_gradients = gradients(case)
    selective = gradients(case, tuple(i == slot for i in range(3)))
    exact(all_gradients[slot], selective[slot])


@pytest.mark.parametrize("op,slot", product(OPS, range(4)))
def test_gpu_saved_revisions(op, slot):
    require_gpu()
    case = synthetic(op, "saved-revisions")
    leaves, y, seed = tracked(case, (True, False, False))
    (*leaves, y)[slot].mark_modified()
    message = "saved value was modified" if slot == 3 else f"saved input slot {slot} was modified"
    with pytest.raises(RuntimeError, match=message):
        (y * seed).sum().backward()
    assert leaves[0].grad is None


def forward_words(case):
    sources = tuple(Producer(tc.tensor(a)) for a in case.inputs)
    fn = getattr(ts, case.op)
    destinations = []

    def alloc(shape):
        p = Producer(tc.empty(*shape))
        destinations.append(p)
        return p

    previous = ts.set_allocator(alloc)
    try:
        default = fn(*sources, **case.cuda_attrs)
        explicit = Producer(tc.empty(*case.seed.shape))
        returned = fn(*sources, out=explicit, **case.cuda_attrs)
        assert returned is explicit and default is destinations[0]
        assert explicit.events and all(p.events for p in (*destinations, *sources))
        exact(default.tensor.numpy(), explicit.tensor.numpy())
        words = default.tensor.numpy().view(np.uint32)
        assert words.size > 0 and np.ptp(words) > 0
        return {"shape": words.shape, "uint32_sha256": hashlib.sha256(words).hexdigest(),
                "source_calls": [p.events for p in sources],
                "alloc_calls": [(p.tensor.shape, p.events) for p in destinations],
                "out_calls": explicit.events}
    finally:
        ts.set_allocator(previous)


@pytest.mark.parametrize("op", OPS)
def test_gpu_no_grad_and_dlpack(op):
    require_gpu()
    case = synthetic(op, "dlpack", stride=2, pads=(2, 1))
    leaves, y, seed = tracked(case)
    producer = Producer(tc.tensor(case.x))
    imported = tc.from_dlpack(producer)
    assert imported.data_ptr() == producer.tensor.data_ptr()
    imported.requires_grad = True
    fn = getattr(ts, op)
    prev = ts.set_allocator(lambda shape: Producer(tc.empty(*shape)))
    try:
        replay = fn(imported, leaves[1], leaves[2], **case.cuda_attrs)
        with tc.no_grad():
            dest = Producer(tc.empty(*y.shape))
            assert fn(*leaves, out=dest, **case.cuda_attrs) is dest
            assert not dest.tensor.requires_grad and dest.events
            allocated = fn(*leaves, **case.cuda_attrs)
            assert isinstance(allocated, Producer)
        exact(y.numpy(), dest.tensor.numpy())
        exact(y.numpy(), allocated.tensor.numpy())
        exact(y.numpy(), replay.numpy())
        (y * seed).sum().backward()
        (replay * seed).sum().backward()
        exact(leaves[0].grad.numpy(), imported.grad.numpy())
    finally:
        ts.set_allocator(prev)
