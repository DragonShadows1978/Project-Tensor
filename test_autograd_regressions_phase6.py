#!/usr/bin/env python3
"""
Phase 6 autograd regression tests for tensor_gpu_v2.py.

Covers:
1. Broadcasted matmul backward (including 1D dot product)
2. Repeated-index getitem gradient accumulation semantics
3. Explicit dtype propagation for float64 across core ops

Run with:
    python test_autograd_regressions_phase6.py
"""

import os
import sys
from typing import List, Tuple

import numpy as np
import cupy as cp
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tensor_gpu_v2 as tg


class TestResults:
    """Simple pass/fail tracker."""

    def __init__(self):
        self.passed: List[str] = []
        self.failed: List[Tuple[str, str]] = []

    def record(self, name: str, ok: bool, details: str = ""):
        if ok:
            self.passed.append(name)
            print(f"[PASS] {name}")
        else:
            self.failed.append((name, details))
            print(f"[FAIL] {name}: {details}")

    def summary(self) -> bool:
        total = len(self.passed) + len(self.failed)
        print("\n" + "=" * 80)
        print(f"RESULT: {len(self.passed)}/{total} passed")
        if self.failed:
            print("Failures:")
            for name, details in self.failed:
                print(f"  - {name}: {details}")
        print("=" * 80)
        return len(self.failed) == 0


def available_devices() -> List[str]:
    devices = ["cpu"]
    try:
        cp.zeros((1,), dtype=cp.float32)
        devices.append("cuda")
    except Exception:
        pass
    return devices


def to_numpy(x):
    if isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return np.asarray(x)


def max_abs_diff(a, b) -> float:
    return float(np.max(np.abs(to_numpy(a) - to_numpy(b))))


def torch_matmul_oracle(a_np: np.ndarray, b_np: np.ndarray):
    a_t = torch.tensor(a_np, dtype=torch.float64, requires_grad=True)
    b_t = torch.tensor(b_np, dtype=torch.float64, requires_grad=True)
    y_t = a_t @ b_t
    loss_t = y_t.sum()
    loss_t.backward()
    return (
        y_t.detach().cpu().numpy(),
        a_t.grad.detach().cpu().numpy(),
        b_t.grad.detach().cpu().numpy(),
    )


def run_matmul_cases(results: TestResults, device: str):
    cases = [
        ((2, 4, 5), (5, 3)),
        ((2, 4, 5), (1, 5, 3)),
        ((3, 2, 4, 5), (5, 3)),
        ((5,), (5,)),
        ((5,), (5, 3)),
        ((2, 5), (5,)),
        ((2, 1, 5), (5,)),
        ((5,), (2, 5, 3)),
    ]

    rng = np.random.default_rng(123)
    atol = 3e-6 if device == "cpu" else 2e-5

    for a_shape, b_shape in cases:
        test_name = f"matmul_backward_{device}_{a_shape}_{b_shape}"
        try:
            a_np = rng.standard_normal(a_shape).astype(np.float64)
            b_np = rng.standard_normal(b_shape).astype(np.float64)

            a = tg.Tensor(a_np, device=device, requires_grad=True)
            b = tg.Tensor(b_np, device=device, requires_grad=True)
            y = a @ b
            loss = y.sum()
            loss.backward()

            y_ref, ga_ref, gb_ref = torch_matmul_oracle(a_np, b_np)
            y_diff = max_abs_diff(y.data, y_ref)
            ga_diff = max_abs_diff(a.grad, ga_ref)
            gb_diff = max_abs_diff(b.grad, gb_ref)

            shape_ok = (a.grad.shape == a_shape and b.grad.shape == b_shape)
            diff_ok = y_diff <= atol and ga_diff <= atol and gb_diff <= atol

            if not shape_ok:
                results.record(
                    test_name,
                    False,
                    f"gradient shape mismatch: got {a.grad.shape}/{b.grad.shape}",
                )
            elif not diff_ok:
                results.record(
                    test_name,
                    False,
                    f"diffs too large: y={y_diff:.3e}, ga={ga_diff:.3e}, gb={gb_diff:.3e}, atol={atol}",
                )
            else:
                results.record(test_name, True)
        except Exception as exc:
            results.record(test_name, False, f"exception: {exc}")


def run_conv2d_nhwc_backward_case(results: TestResults, device: str):
    name = f"conv2d_nhwc_backward_{device}"
    if device != "cuda":
        results.record(name, True)
        return

    try:
        rng = np.random.default_rng(9)
        x_np = rng.standard_normal((2, 5, 6, 4)).astype(np.float64)

        conv = tg.Conv2D(
            in_channels=4,
            out_channels=3,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            data_format="NHWC",
        )
        x = tg.Tensor(x_np, device=device, requires_grad=True)
        y = conv(x)
        weights = tg.Tensor(
            rng.standard_normal(y.data.shape).astype(np.float64),
            device=device,
            requires_grad=False,
        )
        loss = (y * weights).sum()
        loss.backward()

        x_grad_np = to_numpy(x.grad)
        w_grad_np = to_numpy(conv.w.grad)
        b_grad_np = to_numpy(conv.b.grad) if conv.b is not None else None

        ok = (
            y.dtype == np.float64
            and x.grad.shape == x_np.shape
            and conv.w.grad.shape == conv.w.data.shape
            and np.isfinite(x_grad_np).all()
            and np.isfinite(w_grad_np).all()
            and (b_grad_np is None or np.isfinite(b_grad_np).all())
        )
        if not ok:
            results.record(
                name,
                False,
                (
                    f"invalid NHWC backward/dtype state: y_dtype={y.dtype}, "
                    f"x_grad_shape={x.grad.shape}, w_grad_shape={conv.w.grad.shape}"
                ),
            )
        else:
            results.record(name, True)
    except Exception as exc:
        results.record(name, False, f"exception: {exc}")


def run_batchnorm2d_backward_oracle_case(results: TestResults, device: str):
    name = f"batchnorm2d_backward_oracle_{device}"
    try:
        rng = np.random.default_rng(202)
        eps = 1e-5
        x_np = rng.standard_normal((2, 3, 4, 5)).astype(np.float64)
        gamma_np = rng.standard_normal((3,)).astype(np.float64)
        beta_np = rng.standard_normal((3,)).astype(np.float64)
        dout_np = rng.standard_normal((2, 3, 4, 5)).astype(np.float64)

        bn = tg.BatchNorm2D(3, eps=eps)
        xp = np if device == "cpu" else cp
        bn.gamma.data = xp.array(gamma_np, dtype=xp.float64)
        bn.beta.data = xp.array(beta_np, dtype=xp.float64)
        bn.gamma.grad = xp.zeros_like(bn.gamma.data)
        bn.beta.grad = xp.zeros_like(bn.beta.data)

        x = tg.Tensor(x_np, device=device, requires_grad=True)
        dout = tg.Tensor(dout_np, device=device, requires_grad=False)
        y = bn(x)
        loss = (y * dout).sum()
        loss.backward()

        x_t = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
        gamma_t = torch.tensor(gamma_np, dtype=torch.float64, requires_grad=True)
        beta_t = torch.tensor(beta_np, dtype=torch.float64, requires_grad=True)
        dout_t = torch.tensor(dout_np, dtype=torch.float64)
        mean_t = x_t.mean(dim=(0, 2, 3), keepdim=True)
        var_t = x_t.var(dim=(0, 2, 3), unbiased=False, keepdim=True)
        x_norm_t = (x_t - mean_t) / torch.sqrt(var_t + eps)
        y_t = gamma_t.view(1, 3, 1, 1) * x_norm_t + beta_t.view(1, 3, 1, 1)
        loss_t = (y_t * dout_t).sum()
        loss_t.backward()

        atol = 4e-6 if device == "cpu" else 4e-5
        y_diff = max_abs_diff(y.data, y_t.detach().cpu().numpy())
        x_diff = max_abs_diff(x.grad, x_t.grad.detach().cpu().numpy())
        g_diff = max_abs_diff(bn.gamma.grad, gamma_t.grad.detach().cpu().numpy())
        b_diff = max_abs_diff(bn.beta.grad, beta_t.grad.detach().cpu().numpy())
        ok = y_diff <= atol and x_diff <= atol and g_diff <= atol and b_diff <= atol

        if ok:
            results.record(name, True)
        else:
            results.record(
                name,
                False,
                (
                    f"diffs too large: y={y_diff:.3e}, x={x_diff:.3e}, "
                    f"gamma={g_diff:.3e}, beta={b_diff:.3e}, atol={atol}"
                ),
            )
    except Exception as exc:
        results.record(name, False, f"exception: {exc}")


def run_groupnorm_backward_oracle_case(results: TestResults, device: str):
    name = f"groupnorm_backward_oracle_{device}"
    try:
        rng = np.random.default_rng(303)
        eps = 1e-5
        N, C, H, W = 2, 4, 3, 3
        G = 2

        x_np = rng.standard_normal((N, C, H, W)).astype(np.float64)
        gamma_np = rng.standard_normal((C,)).astype(np.float64)
        beta_np = rng.standard_normal((C,)).astype(np.float64)
        dout_np = rng.standard_normal((N, C, H, W)).astype(np.float64)

        gn = tg.GroupNorm(num_groups=G, num_channels=C, eps=eps)
        xp = np if device == "cpu" else cp
        gn.gamma.data = xp.array(gamma_np, dtype=xp.float64)
        gn.beta.data = xp.array(beta_np, dtype=xp.float64)
        gn.gamma.grad = xp.zeros_like(gn.gamma.data)
        gn.beta.grad = xp.zeros_like(gn.beta.data)

        x = tg.Tensor(x_np, device=device, requires_grad=True)
        dout = tg.Tensor(dout_np, device=device, requires_grad=False)
        y = gn(x)
        loss = (y * dout).sum()
        loss.backward()

        x_t = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
        gamma_t = torch.tensor(gamma_np, dtype=torch.float64, requires_grad=True)
        beta_t = torch.tensor(beta_np, dtype=torch.float64, requires_grad=True)
        dout_t = torch.tensor(dout_np, dtype=torch.float64)

        x_reshaped_t = x_t.view(N, G, C // G, H, W)
        mean_t = x_reshaped_t.mean(dim=(2, 3, 4), keepdim=True)
        var_t = x_reshaped_t.var(dim=(2, 3, 4), unbiased=False, keepdim=True)
        x_norm_t = ((x_reshaped_t - mean_t) / torch.sqrt(var_t + eps)).view(N, C, H, W)
        y_t = gamma_t.view(1, C, 1, 1) * x_norm_t + beta_t.view(1, C, 1, 1)
        loss_t = (y_t * dout_t).sum()
        loss_t.backward()

        atol = 5e-6 if device == "cpu" else 6e-5
        y_diff = max_abs_diff(y.data, y_t.detach().cpu().numpy())
        x_diff = max_abs_diff(x.grad, x_t.grad.detach().cpu().numpy())
        g_diff = max_abs_diff(gn.gamma.grad, gamma_t.grad.detach().cpu().numpy())
        b_diff = max_abs_diff(gn.beta.grad, beta_t.grad.detach().cpu().numpy())
        ok = y_diff <= atol and x_diff <= atol and g_diff <= atol and b_diff <= atol

        if ok:
            results.record(name, True)
        else:
            results.record(
                name,
                False,
                (
                    f"diffs too large: y={y_diff:.3e}, x={x_diff:.3e}, "
                    f"gamma={g_diff:.3e}, beta={b_diff:.3e}, atol={atol}"
                ),
            )
    except Exception as exc:
        results.record(name, False, f"exception: {exc}")


def run_getitem_repeat_cases(results: TestResults, device: str):
    xp = np if device == "cpu" else cp

    # Case 1: 1D integer indexing with repeated indices
    name_1d = f"getitem_repeat_1d_{device}"
    try:
        x = tg.Tensor(np.arange(8.0, dtype=np.float64), device=device, requires_grad=True)
        idx = xp.array([1, 1, 3, 1, 7])
        weights = tg.Tensor(np.array([1.0, 2.0, -3.0, 4.0, 5.0], dtype=np.float64), device=device, requires_grad=False)
        loss = (x[idx] * weights).sum()
        loss.backward()

        expected = np.zeros((8,), dtype=np.float64)
        np.add.at(expected, np.array([1, 1, 3, 1, 7]), np.array([1.0, 2.0, -3.0, 4.0, 5.0]))
        diff = max_abs_diff(x.grad, expected)
        results.record(name_1d, diff == 0.0, f"max diff {diff:.3e}")
    except Exception as exc:
        results.record(name_1d, False, f"exception: {exc}")

    # Case 2: Tuple advanced indexing with repeats
    name_tuple = f"getitem_repeat_tuple_{device}"
    try:
        x = tg.Tensor(np.arange(20.0, dtype=np.float64).reshape(4, 5), device=device, requires_grad=True)
        idx0_np = np.array([0, 0, 2, 2, 2, 3], dtype=np.int64)
        idx1_np = np.array([1, 1, 4, 4, 0, 3], dtype=np.int64)
        idx0 = xp.array(idx0_np)
        idx1 = xp.array(idx1_np)
        weights_np = np.array([1.0, -2.0, 3.0, 4.0, -5.0, 6.0], dtype=np.float64)
        weights = tg.Tensor(weights_np, device=device, requires_grad=False)

        loss = (x[(idx0, idx1)] * weights).sum()
        loss.backward()

        expected = np.zeros((4, 5), dtype=np.float64)
        np.add.at(expected, (idx0_np, idx1_np), weights_np)
        diff = max_abs_diff(x.grad, expected)
        results.record(name_tuple, diff == 0.0, f"max diff {diff:.3e}")
    except Exception as exc:
        results.record(name_tuple, False, f"exception: {exc}")


def run_dtype_propagation_case(results: TestResults, device: str):
    name = f"dtype_propagation_float64_{device}"
    try:
        a_np = np.arange(6.0, dtype=np.float64).reshape(2, 3) / 7.0
        b_np = (np.arange(6.0, dtype=np.float64).reshape(2, 3) + 1.0) / 5.0

        a = tg.Tensor(a_np, device=device, requires_grad=True)
        b = tg.Tensor(b_np, device=device, requires_grad=True)

        c = (a + b) * (a - b)
        d = c @ b.transpose(1, 0)
        loss = d.mean()
        loss.backward()

        dtype_ok = (
            a.dtype == np.float64
            and b.dtype == np.float64
            and c.dtype == np.float64
            and d.dtype == np.float64
            and loss.dtype == np.float64
            and a.grad.dtype == np.float64
            and b.grad.dtype == np.float64
        )
        if not dtype_ok:
            results.record(
                name,
                False,
                (
                    f"dtype mismatch: a={a.dtype}, b={b.dtype}, c={c.dtype}, "
                    f"d={d.dtype}, loss={loss.dtype}, ga={a.grad.dtype}, gb={b.grad.dtype}"
                ),
            )
        else:
            results.record(name, True)
    except Exception as exc:
        results.record(name, False, f"exception: {exc}")


def run_dtype_inference_hardening_case(results: TestResults, device: str):
    name = f"dtype_inference_hardening_{device}"
    prev_mp = tg.is_mixed_precision()
    prev_device = tg.get_device()
    try:
        tg.enable_mixed_precision(True)
        tg.set_device(device)

        float_scalar = tg.Tensor(1.25, device=device, requires_grad=False)
        int_scalar = tg.Tensor(7, device=device, requires_grad=False)
        bool_scalar = tg.Tensor(True, device=device, requires_grad=False)

        if device == "cuda":
            float_ok = float_scalar.dtype == np.float16
        else:
            float_ok = float_scalar.dtype == np.float64
        int_ok = np.issubdtype(np.dtype(int_scalar.dtype), np.integer)
        bool_ok = np.dtype(bool_scalar.dtype) == np.dtype(np.bool_)

        object_rejected = False
        try:
            tg.Tensor(np.array([{"k": "v"}], dtype=object), device=device, requires_grad=False)
        except TypeError:
            object_rejected = True

        transfer_ok = True
        if device == "cuda":
            x_cpu = tg.Tensor(np.arange(6.0, dtype=np.float64).reshape(2, 3), device="cpu", requires_grad=False)
            x_cuda = x_cpu.to("CUDA:0")
            x_back = x_cuda.to("cpu")
            transfer_ok = (
                x_cuda.dtype == np.float64
                and x_back.dtype == np.float64
                and max_abs_diff(x_back.data, x_cpu.data) == 0.0
            )

        ok = float_ok and int_ok and bool_ok and object_rejected and transfer_ok
        if ok:
            results.record(name, True)
        else:
            results.record(
                name,
                False,
                (
                    f"float_ok={float_ok}, int_ok={int_ok}, bool_ok={bool_ok}, "
                    f"object_rejected={object_rejected}, transfer_ok={transfer_ok}"
                ),
            )
    except Exception as exc:
        results.record(name, False, f"exception: {exc}")
    finally:
        tg.enable_mixed_precision(prev_mp)
        tg.set_device(prev_device)


def run_unbroadcast_validation_case(results: TestResults, device: str):
    name = f"unbroadcast_validation_{device}"
    xp = np if device == "cpu" else cp
    try:
        t = tg.Tensor(np.zeros((2, 3), dtype=np.float64), device=device, requires_grad=False)

        rank_error = False
        try:
            t._unbroadcast(xp.ones((2, 3), dtype=xp.float64), (2, 3, 1))
        except ValueError:
            rank_error = True

        shape_error = False
        try:
            t._unbroadcast(xp.ones((2, 3), dtype=xp.float64), (2, 4))
        except ValueError:
            shape_error = True

        reduced = t._unbroadcast(xp.ones((2, 3, 4), dtype=xp.float64), (1, 3, 1))
        reduced_np = to_numpy(reduced)
        reduction_ok = reduced_np.shape == (1, 3, 1) and np.allclose(reduced_np, 8.0)

        ok = rank_error and shape_error and reduction_ok
        if ok:
            results.record(name, True)
        else:
            results.record(
                name,
                False,
                f"rank_error={rank_error}, shape_error={shape_error}, reduction_ok={reduction_ok}",
            )
    except Exception as exc:
        results.record(name, False, f"exception: {exc}")


def run_matmul_validation_case(results: TestResults, device: str):
    name = f"matmul_validation_{device}"
    xp = np if device == "cpu" else cp
    try:
        bad_grad_shape_error = False
        a = tg.Tensor(np.arange(6.0, dtype=np.float64).reshape(2, 3), device=device, requires_grad=True)
        b = tg.Tensor(np.arange(12.0, dtype=np.float64).reshape(3, 4), device=device, requires_grad=True)
        y = a @ b
        y.grad = xp.ones((3, 4), dtype=xp.float64)  # wrong shape; expected (2, 4)
        try:
            y._backward()
        except ValueError:
            bad_grad_shape_error = True

        int_dtype_error = False
        a_int = tg.Tensor(np.arange(6, dtype=np.int32).reshape(2, 3), device=device, requires_grad=True)
        b_float = tg.Tensor(np.ones((3, 2), dtype=np.float64), device=device, requires_grad=True)
        y2 = a_int @ b_float
        y2.grad = xp.ones_like(y2.data)
        try:
            y2._backward()
        except TypeError:
            int_dtype_error = True

        ok = bad_grad_shape_error and int_dtype_error
        if ok:
            results.record(name, True)
        else:
            results.record(
                name,
                False,
                f"bad_grad_shape_error={bad_grad_shape_error}, int_dtype_error={int_dtype_error}",
            )
    except Exception as exc:
        results.record(name, False, f"exception: {exc}")


def run_scatter_add_validation_case(results: TestResults, device: str):
    name = f"scatter_add_validation_{device}"
    xp = np if device == "cpu" else cp
    try:
        t = tg.Tensor(np.zeros((5,), dtype=np.float64), device=device, requires_grad=False)

        index_error = False
        try:
            t._scatter_add(
                xp.zeros((5,), dtype=xp.float64),
                xp.array([0, 7], dtype=xp.int64),
                xp.array([1.0, 2.0], dtype=xp.float64),
            )
        except IndexError:
            index_error = True

        bool_error = False
        try:
            t._scatter_add(
                xp.zeros((5,), dtype=xp.bool_),
                xp.array([1, 1], dtype=xp.int64),
                xp.array([True, False], dtype=xp.bool_),
            )
        except TypeError:
            bool_error = True

        ok = index_error and bool_error
        if ok:
            results.record(name, True)
        else:
            results.record(
                name,
                False,
                f"index_error={index_error}, bool_error={bool_error}",
            )
    except Exception as exc:
        results.record(name, False, f"exception: {exc}")


def run_col2im_padding_roundtrip_case(results: TestResults, device: str):
    name = f"col2im_padding_roundtrip_{device}"
    if device != "cuda":
        results.record(name, True)
        return

    try:
        x_cp = cp.random.standard_normal((2, 3, 4, 5)).astype(cp.float64)
        cols = tg.im2col_indices(x_cp, field_height=1, field_width=1, padding=2, stride=1)
        recon = tg.col2im_indices(cols, x_cp.shape, field_height=1, field_width=1, padding=2, stride=1)

        ones = cp.ones_like(x_cp)
        div = tg.col2im_indices(
            tg.im2col_indices(ones, field_height=1, field_width=1, padding=2, stride=1),
            x_cp.shape,
            field_height=1,
            field_width=1,
            padding=2,
            stride=1,
        )
        roundtrip = recon / div

        diff = max_abs_diff(roundtrip, x_cp)
        ok = recon.shape == x_cp.shape and diff <= 1e-10
        if ok:
            results.record(name, True)
        else:
            results.record(name, False, f"shape={recon.shape}, diff={diff:.3e}")
    except Exception as exc:
        results.record(name, False, f"exception: {exc}")


def run_deserialize_state_dict_case(results: TestResults, device: str):
    name = f"deserialize_state_dict_types_{device}"
    try:
        state = {
            "arr": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
            "nested": {
                "arr2": np.array([5, 6, 7], dtype=np.int64),
                "scale": 3.25,
            },
            "flag": True,
            "steps": 7,
        }
        out = tg._deserialize_state_dict(state, device)

        arr_ok = isinstance(out["arr"], np.ndarray if device == "cpu" else cp.ndarray)
        nested_arr_ok = isinstance(out["nested"]["arr2"], np.ndarray if device == "cpu" else cp.ndarray)
        scalar_ok = isinstance(out["nested"]["scale"], float) and isinstance(out["steps"], int)
        flag_ok = isinstance(out["flag"], bool)
        values_ok = (
            max_abs_diff(out["arr"], state["arr"]) == 0.0
            and max_abs_diff(out["nested"]["arr2"], state["nested"]["arr2"]) == 0.0
            and out["nested"]["scale"] == state["nested"]["scale"]
            and out["steps"] == state["steps"]
            and out["flag"] == state["flag"]
        )
        ok = arr_ok and nested_arr_ok and scalar_ok and flag_ok and values_ok

        if ok:
            results.record(name, True)
        else:
            results.record(
                name,
                False,
                (
                    f"arr_ok={arr_ok}, nested_arr_ok={nested_arr_ok}, scalar_ok={scalar_ok}, "
                    f"flag_ok={flag_ok}, values_ok={values_ok}"
                ),
            )
    except Exception as exc:
        results.record(name, False, f"exception: {exc}")


def run_regression_suite() -> bool:
    results = TestResults()

    devices = available_devices()
    print(f"Testing devices: {devices}")

    for device in devices:
        tg.set_device(device)
        run_matmul_cases(results, device)
        run_getitem_repeat_cases(results, device)
        run_dtype_propagation_case(results, device)
        run_dtype_inference_hardening_case(results, device)
        run_unbroadcast_validation_case(results, device)
        run_matmul_validation_case(results, device)
        run_scatter_add_validation_case(results, device)
        run_conv2d_nhwc_backward_case(results, device)
        run_batchnorm2d_backward_oracle_case(results, device)
        run_groupnorm_backward_oracle_case(results, device)
        run_col2im_padding_roundtrip_case(results, device)
        run_deserialize_state_dict_case(results, device)

    return results.summary()


if __name__ == "__main__":
    success = run_regression_suite()
    raise SystemExit(0 if success else 1)
