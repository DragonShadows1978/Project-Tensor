"""Tracked routing, the hook's saved records, and DLPack interoperability."""

import numpy as np
import pytest

import tc_signal as ts
import tensor_cuda as tc
from bwd_r2_support import cases
from bwd_r2_support.cases import OPS, exact, nonconstant, require_gpu

SIGNAL_OPS = OPS[:3]
X = np.array([[[-2, 7, 3, -4, 1], [4, -6, 2, 8, -7]]], dtype=np.float32)


def call(op, x, **kwargs):
    if op == "upsample_nearest1d":
        return ts.upsample_nearest1d(x, 2, **kwargs)
    if op == "reflection_pad1d":
        return ts.reflection_pad1d(x, (2, 3), **kwargs)
    return ts.instance_norm1d(x, **kwargs)


class Producer:
    def __init__(self, tensor):
        self.tensor = tensor
        self.exports = 0

    def __dlpack__(self, **kwargs):
        self.exports += 1
        return self.tensor.__dlpack__(**kwargs)

    def __dlpack_device__(self):
        return self.tensor.__dlpack_device__()


@pytest.mark.parametrize("op", SIGNAL_OPS)
def test_host_out_refusal(op):
    x = tc.tensor(X, device="cpu", requires_grad=True)
    with pytest.raises(ValueError, match="out= is unsupported when recording gradients"):
        call(op, x, out=x)


@pytest.mark.parametrize("op", SIGNAL_OPS)
def test_host_cpu_planner_boundary(op):
    # Real TensorCUDA inputs reach the real CUDA-only planner if the
    # Python argument gate allows them. Both grad modes are exercised.
    x = tc.tensor(X, device="cpu", requires_grad=True)
    with pytest.raises(BufferError, match="CUDA"):
        call(op, x)
    with tc.no_grad(), pytest.raises(BufferError, match="CUDA"):
        call(op, x, out=x)
    x.requires_grad = False
    with pytest.raises(BufferError, match="CUDA"):
        call(op, x, out=x)


@pytest.mark.parametrize("role", ("x", "scale", "bias"))
def test_host_foreign_operand_refusal(role):
    x = tc.tensor(X, device="cpu", requires_grad=True)
    scale = tc.tensor([1, 2], device="cpu", requires_grad=role == "x")
    bias = tc.tensor([0, 0], device="cpu")
    args = {"x": x, "scale": scale, "bias": bias}
    producer = Producer(args[role])
    args[role] = producer
    with pytest.raises(TypeError, match="recording gradients requires TensorCUDA operands"):
        ts.instance_norm1d(**args)
    assert producer.exports == 0


@pytest.mark.parametrize("role", ("scale", "bias"))
def test_host_trainable_affine_reaches_cuda_planner(role):
    # BWD-R4a r5b item 9: a trainable scale or bias is no longer refused; the tracked call reaches the native CUDA
    # planner, which rejects the host tensor. The out= witness: the tracked route refuses out= before the planner,
    # and the untracked route (no_grad) takes out= to the same planner.
    x = tc.tensor(X, device="cpu")
    with pytest.raises(BufferError, match="CUDA"):
        ts.instance_norm1d(x, **{role: tc.tensor([1, 2], device="cpu", requires_grad=True)})
    with pytest.raises(ValueError, match="out= is unsupported when recording gradients"):
        ts.instance_norm1d(x, out=x, **{role: tc.tensor([1, 2], device="cpu", requires_grad=True)})
    with tc.no_grad(), pytest.raises(BufferError, match="CUDA"):
        ts.instance_norm1d(x, out=x, **{role: tc.tensor([1, 2], device="cpu", requires_grad=True)})


@pytest.mark.parametrize("op", SIGNAL_OPS)
def test_host_missing_attach_vjp_before_forward(op, monkeypatch):
    x = tc.tensor(X, device="cpu", requires_grad=True)
    monkeypatch.delattr(tc, "attach_vjp")
    # The real planner would instead reject this host tensor with BufferError.
    with pytest.raises(RuntimeError, match="requires TensorCUDA with attach_vjp"):
        call(op, x)
    with tc.no_grad(), pytest.raises(BufferError, match="CUDA"):
        call(op, x)
    x.requires_grad = False
    with pytest.raises(BufferError, match="CUDA"):
        call(op, x)


@pytest.mark.parametrize(
    ("visible", "status", "devices", "reason"),
    [
        pytest.param("", None, 0, "CUDA disabled for CPU lane", id="hidden"),
        pytest.param(None, 0, 0, "no usable CUDA device", id="unset"),
        pytest.param("-1", 0, 0, "no usable CUDA device", id="masked"),
        pytest.param("0", 35, 1, "no usable CUDA device", id="runtime-status"),
        pytest.param("0", None, 0, "no usable CUDA device", id="runtime-unavailable"),
        pytest.param("0", 0, 1, None, id="usable"),
    ],
)
def test_host_cuda_device_probe(visible, status, devices, reason, monkeypatch):
    if visible is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    loads = []

    def count(ptr):
        cases.ctypes.cast(ptr, cases.ctypes.POINTER(cases.ctypes.c_int))[0] = devices
        return status

    def load(name):
        from types import SimpleNamespace

        loads.append(name)
        if status is None:
            raise OSError("CUDA runtime unavailable")
        return SimpleNamespace(cudaGetDeviceCount=count)

    monkeypatch.setattr(cases.ctypes, "CDLL", load)
    if reason is None:
        require_gpu()
    else:
        with pytest.raises(pytest.skip.Exception, match=reason):
            require_gpu()
    assert loads == (["libcudart.so.12"] if visible != "" else [])


@pytest.mark.parametrize("op", SIGNAL_OPS)
def test_gpu_paths_and_interop(op):
    require_gpu()
    x = tc.tensor(X, requires_grad=True)
    prev = ts.set_allocator(lambda shape: Producer(tc.empty(*shape)))
    try:
        y = call(op, x)
        assert y.requires_grad and y.revision == 1
        with pytest.raises(ValueError, match="out= is unsupported"):
            call(op, x, out=tc.empty(*y.shape))
        with tc.no_grad():
            destination = Producer(tc.empty(*y.shape))
            assert call(op, x, out=destination) is destination
            assert destination.exports == 1
            assert not destination.tensor.requires_grad
            exact(y.numpy(), destination.tensor.numpy())
            wrapped = call(op, x)
            assert type(wrapped) is Producer and not wrapped.tensor.requires_grad
            exact(y.numpy(), wrapped.tensor.numpy())
        source = Producer(tc.tensor(X))
        untracked = call(op, source)
        assert type(untracked) is Producer and source.exports == 1
        exact(y.numpy(), untracked.tensor.numpy())

        seed = tc.tensor(np.linspace(-0.8, 1.2, int(np.prod(y.shape)), dtype=np.float32).reshape(y.shape))
        (y * seed).sum().backward()
        native_grad = x.grad.numpy()
        nonconstant(native_grad)
        imported = tc.from_dlpack(source)
        assert imported.data_ptr() == source.tensor.data_ptr()
        imported.requires_grad = True
        imported_y = call(op, imported)
        assert type(imported_y) is tc.Tensor and imported_y.requires_grad
        (imported_y * seed).sum().backward()
        exact(imported.grad.numpy(), native_grad)
    finally:
        ts.set_allocator(prev)


@pytest.mark.parametrize("op", SIGNAL_OPS)
def test_gpu_saved_input_revision(op):
    require_gpu()
    x = tc.tensor(X, requires_grad=True)
    y = call(op, x)
    x.mark_modified()
    with pytest.raises(RuntimeError, match="saved input slot 0 was modified"):
        y.sum().backward()
    assert x.grad is None


def test_gpu_saved_affine_and_output_revisions():
    require_gpu()
    x = tc.tensor(X, requires_grad=True)
    scale = tc.tensor([1, 2])
    y = ts.instance_norm1d(x, scale)
    scale.fill_(3.)
    with pytest.raises(RuntimeError, match="saved input slot 1 was modified"):
        y.sum().backward()
    y = ts.instance_norm1d(x)
    y.mark_modified()
    with pytest.raises(RuntimeError, match="saved value was modified"):
        y.sum().backward()
    assert x.grad is None


def test_gpu_foreign_operand_refusal():
    require_gpu()
    x = tc.tensor(X, requires_grad=True)
    foreign = Producer(tc.tensor([1, 2]))
    with pytest.raises(TypeError, match="requires TensorCUDA operands"):
        ts.instance_norm1d(x, scale=foreign)
    assert foreign.exports == 0


@pytest.mark.parametrize("op", SIGNAL_OPS[:2])
def test_gpu_forward_special_bit_patterns(op):
    require_gpu()
    bits = np.array([0x80000000, 0, 0x7F800000, 0xFF800000, 0x7FC01234], dtype=np.uint32)
    x = tc.tensor(bits.view(np.float32), requires_grad=True)
    tracked = call(op, x)
    with tc.no_grad():
        plain = call(op, x)
    expected = np.repeat(bits, 2) if op == SIGNAL_OPS[0] else np.pad(bits, (2, 3), mode="reflect")
    exact(tracked.numpy(), expected.view(np.float32))
    exact(plain.numpy(), expected.view(np.float32))
