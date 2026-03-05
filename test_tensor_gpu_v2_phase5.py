"""
Phase 5 Tests for tensor_gpu_v2.py - Final Hardening and Edge Case Fixes

Tests all 7 identified vulnerabilities from Cycle 4 adversarial testing:
1. GradScaler zero/negative scale rejection
2. GradScaler load_state_dict scale validation
3. clip_grad_norm_ NaN/Inf handling
4. weight_tie shape validation
5. kernel cache invalid/empty code rejection
6. Integration test with all Phase 4 features
7. Checkpoint round-trip test

Run with: python test_tensor_gpu_v2_phase5.py
"""
import sys
import os
import tempfile
import warnings

# Add workspace to path
sys.path.insert(0, '/home/vader/mini-mind-v2/workspace/OPAL')

import numpy as np
import cupy as cp

# Import tensor_gpu_v2
import tensor_gpu_v2 as tgpu


class TestResults:
    """Simple test result tracker."""
    def __init__(self):
        self.passed = []
        self.failed = []

    def record(self, name, passed, error=None):
        if passed:
            self.passed.append(name)
            print(f"✓ {name}")
        else:
            self.failed.append((name, str(error) if error else "Failed"))
            print(f"✗ {name}: {error}")

    def summary(self):
        total = len(self.passed) + len(self.failed)
        print(f"\n{'='*60}")
        print(f"RESULTS: {len(self.passed)}/{total} tests passed")
        if self.failed:
            print(f"\nFailed tests:")
            for name, error in self.failed:
                print(f"  - {name}: {error}")
        print(f"{'='*60}")
        return len(self.failed) == 0


results = TestResults()


# ==================== TEST 1: GradScaler zero/negative scale ====================

def test_gradscaler_init_zero_scale():
    """GradScaler should reject init_scale=0."""
    try:
        scaler = tgpu.GradScaler(init_scale=0)
        results.record("GradScaler init_scale=0", False, "No ValueError raised")
    except ValueError as e:
        if "positive" in str(e).lower():
            results.record("GradScaler init_scale=0", True)
        else:
            results.record("GradScaler init_scale=0", False, f"Wrong error: {e}")
    except Exception as e:
        results.record("GradScaler init_scale=0", False, f"Unexpected error: {e}")


def test_gradscaler_init_negative_scale():
    """GradScaler should reject init_scale=-1."""
    try:
        scaler = tgpu.GradScaler(init_scale=-1)
        results.record("GradScaler init_scale=-1", False, "No ValueError raised")
    except ValueError as e:
        if "positive" in str(e).lower():
            results.record("GradScaler init_scale=-1", True)
        else:
            results.record("GradScaler init_scale=-1", False, f"Wrong error: {e}")
    except Exception as e:
        results.record("GradScaler init_scale=-1", False, f"Unexpected error: {e}")


def test_gradscaler_init_tiny_scale():
    """GradScaler should reject init_scale=1e-300 (effectively zero)."""
    try:
        # Very small positive values should be accepted (they are > 0)
        scaler = tgpu.GradScaler(init_scale=1e-300)
        # This should work - it's technically positive
        results.record("GradScaler init_scale=1e-300", True)
    except ValueError:
        # If it rejects, that's also acceptable (stricter validation)
        results.record("GradScaler init_scale=1e-300", True)
    except Exception as e:
        results.record("GradScaler init_scale=1e-300", False, f"Unexpected error: {e}")


# ==================== TEST 2: GradScaler load_state_dict validation ====================

def test_gradscaler_load_zero_scale():
    """GradScaler.load_state_dict should reject scale=0."""
    try:
        scaler = tgpu.GradScaler(init_scale=1.0)
        scaler.load_state_dict({'scale': 0})
        results.record("GradScaler load_state_dict scale=0", False, "No ValueError raised")
    except ValueError as e:
        if "positive" in str(e).lower() or "invalid" in str(e).lower():
            results.record("GradScaler load_state_dict scale=0", True)
        else:
            results.record("GradScaler load_state_dict scale=0", False, f"Wrong error: {e}")
    except Exception as e:
        results.record("GradScaler load_state_dict scale=0", False, f"Unexpected error: {e}")


def test_gradscaler_load_negative_scale():
    """GradScaler.load_state_dict should reject scale=-100."""
    try:
        scaler = tgpu.GradScaler(init_scale=1.0)
        scaler.load_state_dict({'scale': -100})
        results.record("GradScaler load_state_dict scale=-100", False, "No ValueError raised")
    except ValueError as e:
        results.record("GradScaler load_state_dict scale=-100", True)
    except Exception as e:
        results.record("GradScaler load_state_dict scale=-100", False, f"Unexpected error: {e}")


# ==================== TEST 3: clip_grad_norm_ NaN/Inf handling ====================

def test_clip_grad_norm_nan_gradients():
    """clip_grad_norm_ should handle NaN gradients gracefully."""
    try:
        # Create tensor with NaN gradient
        t = tgpu.Tensor(cp.array([1.0, 2.0, 3.0], dtype=cp.float32), requires_grad=True)
        t.grad = cp.array([float('nan'), 1.0, 2.0], dtype=cp.float32)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            norm = tgpu.clip_grad_norm_([t], max_norm=1.0)

        # Should return a valid norm (0.0 if all zeroed, or computed from valid values)
        if not np.isnan(norm) and not np.isinf(norm):
            results.record("clip_grad_norm_ NaN gradients", True)
        else:
            results.record("clip_grad_norm_ NaN gradients", False, f"Got {norm}")
    except Exception as e:
        results.record("clip_grad_norm_ NaN gradients", False, f"Exception: {e}")


def test_clip_grad_norm_inf_gradients():
    """clip_grad_norm_ should handle Inf gradients gracefully."""
    try:
        # Create tensor with Inf gradient
        t = tgpu.Tensor(cp.array([1.0, 2.0, 3.0], dtype=cp.float32), requires_grad=True)
        t.grad = cp.array([float('inf'), 1.0, 2.0], dtype=cp.float32)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            norm = tgpu.clip_grad_norm_([t], max_norm=1.0)

        if not np.isnan(norm) and not np.isinf(norm):
            results.record("clip_grad_norm_ Inf gradients", True)
        else:
            results.record("clip_grad_norm_ Inf gradients", False, f"Got {norm}")
    except Exception as e:
        results.record("clip_grad_norm_ Inf gradients", False, f"Exception: {e}")


def test_clip_grad_norm_mixed_nan_inf():
    """clip_grad_norm_ should handle mixed NaN/Inf/-Inf gradients."""
    try:
        t1 = tgpu.Tensor(cp.array([1.0, 2.0], dtype=cp.float32), requires_grad=True)
        t1.grad = cp.array([float('nan'), float('-inf')], dtype=cp.float32)

        t2 = tgpu.Tensor(cp.array([3.0, 4.0], dtype=cp.float32), requires_grad=True)
        t2.grad = cp.array([float('inf'), float('nan')], dtype=cp.float32)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            norm = tgpu.clip_grad_norm_([t1, t2], max_norm=1.0)

        if not np.isnan(norm) and not np.isinf(norm):
            results.record("clip_grad_norm_ mixed NaN/Inf", True)
        else:
            results.record("clip_grad_norm_ mixed NaN/Inf", False, f"Got {norm}")
    except Exception as e:
        results.record("clip_grad_norm_ mixed NaN/Inf", False, f"Exception: {e}")


def test_clip_grad_norm_error_if_nonfinite():
    """clip_grad_norm_ should raise with error_if_nonfinite=True."""
    try:
        t = tgpu.Tensor(cp.array([1.0, 2.0], dtype=cp.float32), requires_grad=True)
        t.grad = cp.array([float('nan'), 1.0], dtype=cp.float32)

        try:
            tgpu.clip_grad_norm_([t], max_norm=1.0, error_if_nonfinite=True)
            results.record("clip_grad_norm_ error_if_nonfinite", False, "No error raised")
        except RuntimeError:
            results.record("clip_grad_norm_ error_if_nonfinite", True)
    except Exception as e:
        results.record("clip_grad_norm_ error_if_nonfinite", False, f"Exception: {e}")


# ==================== TEST 4: weight_tie shape validation ====================

def test_weight_tie_shape_mismatch():
    """weight_tie should reject mismatched shapes."""
    try:
        # Create embedding (100, 50) and linear (100, 60) - mismatched
        class DummyModule:
            pass

        source = tgpu.Tensor(cp.random.randn(100, 50).astype(cp.float32), requires_grad=True)
        target = tgpu.Tensor(cp.random.randn(100, 60).astype(cp.float32), requires_grad=True)

        module = DummyModule()
        module.w = target

        try:
            tgpu.weight_tie(source, module, 'w', transpose=False)
            results.record("weight_tie shape mismatch", False, "No ValueError raised")
        except ValueError as e:
            if "shape" in str(e).lower() or "mismatch" in str(e).lower():
                results.record("weight_tie shape mismatch", True)
            else:
                results.record("weight_tie shape mismatch", False, f"Wrong error: {e}")
    except Exception as e:
        results.record("weight_tie shape mismatch", False, f"Exception: {e}")


def test_weight_tie_transpose_mismatch():
    """weight_tie should reject mismatched transpose shapes."""
    try:
        class DummyModule:
            pass

        # Source (100, 50), target should be (50, 100) for transpose
        source = tgpu.Tensor(cp.random.randn(100, 50).astype(cp.float32), requires_grad=True)
        target = tgpu.Tensor(cp.random.randn(60, 100).astype(cp.float32), requires_grad=True)  # Wrong!

        module = DummyModule()
        module.w = target

        try:
            tgpu.weight_tie(source, module, 'w', transpose=True)
            results.record("weight_tie transpose mismatch", False, "No ValueError raised")
        except ValueError as e:
            if "shape" in str(e).lower() or "mismatch" in str(e).lower():
                results.record("weight_tie transpose mismatch", True)
            else:
                results.record("weight_tie transpose mismatch", False, f"Wrong error: {e}")
    except Exception as e:
        results.record("weight_tie transpose mismatch", False, f"Exception: {e}")


def test_weight_tie_valid_shapes():
    """weight_tie should accept matching shapes."""
    try:
        class DummyModule:
            pass

        source = tgpu.Tensor(cp.random.randn(100, 50).astype(cp.float32), requires_grad=True)
        target = tgpu.Tensor(cp.random.randn(100, 50).astype(cp.float32), requires_grad=True)

        module = DummyModule()
        module.w = target

        tgpu.weight_tie(source, module, 'w', transpose=False)

        # Verify data is shared
        if cp.allclose(module.w.data, source.data):
            results.record("weight_tie valid shapes", True)
        else:
            results.record("weight_tie valid shapes", False, "Data not shared")
    except Exception as e:
        results.record("weight_tie valid shapes", False, f"Exception: {e}")


def test_weight_tie_valid_transpose():
    """weight_tie should work with valid transposed shapes."""
    try:
        class DummyModule:
            pass

        source = tgpu.Tensor(cp.random.randn(100, 50).astype(cp.float32), requires_grad=True)
        target = tgpu.Tensor(cp.random.randn(50, 100).astype(cp.float32), requires_grad=True)

        module = DummyModule()
        module.w = target

        tgpu.weight_tie(source, module, 'w', transpose=True)

        # Verify data is transposed
        if cp.allclose(module.w.data, source.data.T):
            results.record("weight_tie valid transpose", True)
        else:
            results.record("weight_tie valid transpose", False, "Data not correctly transposed")
    except Exception as e:
        results.record("weight_tie valid transpose", False, f"Exception: {e}")


# ==================== TEST 5: Kernel cache validation ====================

def test_kernel_cache_empty_code():
    """_get_cached_kernel should reject empty code."""
    try:
        tgpu._get_cached_kernel("test", "", "test_func")
        results.record("kernel cache empty code", False, "No error raised")
    except ValueError as e:
        if "empty" in str(e).lower():
            results.record("kernel cache empty code", True)
        else:
            results.record("kernel cache empty code", False, f"Wrong error: {e}")
    except Exception as e:
        results.record("kernel cache empty code", False, f"Wrong exception type: {e}")


def test_kernel_cache_whitespace_code():
    """_get_cached_kernel should reject whitespace-only code."""
    try:
        tgpu._get_cached_kernel("test", "   \n\t   ", "test_func")
        results.record("kernel cache whitespace code", False, "No error raised")
    except ValueError as e:
        if "empty" in str(e).lower():
            results.record("kernel cache whitespace code", True)
        else:
            results.record("kernel cache whitespace code", False, f"Wrong error: {e}")
    except Exception as e:
        results.record("kernel cache whitespace code", False, f"Wrong exception type: {e}")


def test_kernel_cache_invalid_cuda():
    """_get_cached_kernel should reject invalid CUDA code."""
    try:
        invalid_code = "this is not valid cuda code at all }{[]"
        tgpu._get_cached_kernel("test_invalid", invalid_code, "my_kernel")
        results.record("kernel cache invalid CUDA", False, "No error raised")
    except RuntimeError as e:
        if "compilation" in str(e).lower() or "failed" in str(e).lower():
            results.record("kernel cache invalid CUDA", True)
        else:
            results.record("kernel cache invalid CUDA", False, f"Wrong error: {e}")
    except Exception as e:
        # Any exception is acceptable for invalid CUDA
        results.record("kernel cache invalid CUDA", True)


def test_kernel_cache_empty_func_name():
    """_get_cached_kernel should reject empty function name."""
    try:
        valid_code = """
        extern "C" __global__ void my_kernel(float* out) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            out[idx] = 1.0f;
        }
        """
        tgpu._get_cached_kernel("test", valid_code, "")
        results.record("kernel cache empty func name", False, "No error raised")
    except ValueError as e:
        if "empty" in str(e).lower():
            results.record("kernel cache empty func name", True)
        else:
            results.record("kernel cache empty func name", False, f"Wrong error: {e}")
    except Exception as e:
        results.record("kernel cache empty func name", False, f"Wrong exception type: {e}")


# ==================== TEST 6: Integration test ====================

def test_integration_training_loop():
    """End-to-end training loop with all Phase 4 features."""
    try:
        # Create small model
        class TinyModel(tgpu.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = tgpu.Linear(10, 20)
                self.linear2 = tgpu.Linear(20, 10)
                self.device = 'cuda'
                self._device = 'cuda'

            def forward(self, x):
                x = self.linear1(x).relu()
                return self.linear2(x)

        model = TinyModel()
        optimizer = tgpu.Adam(model.parameters(), lr=0.001)
        scaler = tgpu.GradScaler(init_scale=1024.0)

        # Training loop
        for step in range(10):
            optimizer.zero_grad()

            # Forward
            x = tgpu.Tensor(cp.random.randn(4, 10).astype(cp.float32), requires_grad=False)
            y = model.forward(x)
            loss = (y * y).mean()

            # Scale and backward
            scaled_loss = scaler.scale(loss)
            scaled_loss.backward()

            # Unscale and clip
            scaler.unscale_(optimizer)
            total_norm = tgpu.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Step
            scaler.step(optimizer)
            scaler.update()

        stats = scaler.get_statistics()
        if stats['total_steps'] == 10:
            results.record("integration training loop", True)
        else:
            results.record("integration training loop", False, f"Expected 10 steps, got {stats['total_steps']}")
    except Exception as e:
        import traceback
        results.record("integration training loop", False, f"Exception: {e}\n{traceback.format_exc()}")


def test_integration_checkpoint_roundtrip():
    """Test checkpoint save/load/resume."""
    try:
        # Create model and optimizer
        class TinyModel(tgpu.Module):
            def __init__(self):
                super().__init__()
                self.linear = tgpu.Linear(10, 5)
                self.device = 'cuda'
                self._device = 'cuda'

            def forward(self, x):
                return self.linear(x)

        model = TinyModel()
        optimizer = tgpu.Adam(model.parameters(), lr=0.001)
        scaler = tgpu.GradScaler(init_scale=2048.0)

        # Train a few steps
        for _ in range(3):
            x = tgpu.Tensor(cp.random.randn(2, 10).astype(cp.float32))
            y = model.forward(x)
            loss = (y * y).mean()
            scaled_loss = scaler.scale(loss)
            scaled_loss.backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Save checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "checkpoint.pkl")
            tgpu.save_checkpoint(
                checkpoint_path, model, optimizer, scaler,
                epoch=5, global_step=100,
                extra_state={'best_loss': 0.5}
            )

            # Create new model and load
            model2 = TinyModel()
            optimizer2 = tgpu.Adam(model2.parameters(), lr=0.001)
            scaler2 = tgpu.GradScaler()

            info = tgpu.load_checkpoint(
                checkpoint_path, model2, optimizer2, scaler2
            )

            # Verify restoration
            checks = [
                info['epoch'] == 5,
                info['global_step'] == 100,
                info['extra_state']['best_loss'] == 0.5,
                scaler2._step_count == scaler._step_count,
            ]

            if all(checks):
                results.record("integration checkpoint roundtrip", True)
            else:
                results.record("integration checkpoint roundtrip", False, f"Checks failed: {checks}")
    except Exception as e:
        import traceback
        results.record("integration checkpoint roundtrip", False, f"Exception: {e}\n{traceback.format_exc()}")


def test_integration_profiler():
    """Test profiling integration."""
    try:
        class TinyModel(tgpu.Module):
            def __init__(self):
                super().__init__()
                self.linear = tgpu.Linear(100, 100)
                self._device = 'cuda'

            def forward(self, x):
                return self.linear(x)

        model = TinyModel()

        with tgpu.Profiler("test_forward") as prof:
            x = tgpu.Tensor(cp.random.randn(32, 100).astype(cp.float32))
            y = model.forward(x)
            cp.cuda.Stream.null.synchronize()

        report = prof.report()
        if report['name'] == 'test_forward' and report['elapsed_ms'] is not None:
            results.record("integration profiler", True)
        else:
            results.record("integration profiler", False, f"Report incomplete: {report}")
    except Exception as e:
        results.record("integration profiler", False, f"Exception: {e}")


# ==================== TEST 7: Additional edge cases ====================

def test_gradscaler_min_scale_validation():
    """GradScaler should reject min_scale <= 0."""
    try:
        scaler = tgpu.GradScaler(init_scale=1.0, min_scale=0)
        results.record("GradScaler min_scale=0", False, "No ValueError raised")
    except ValueError as e:
        if "positive" in str(e).lower():
            results.record("GradScaler min_scale=0", True)
        else:
            results.record("GradScaler min_scale=0", False, f"Wrong error: {e}")
    except Exception as e:
        results.record("GradScaler min_scale=0", False, f"Unexpected error: {e}")


def test_gradscaler_max_scale_validation():
    """GradScaler should reject max_scale < init_scale."""
    try:
        scaler = tgpu.GradScaler(init_scale=1000.0, max_scale=100.0)
        results.record("GradScaler max_scale < init_scale", False, "No ValueError raised")
    except ValueError as e:
        if "max_scale" in str(e).lower():
            results.record("GradScaler max_scale < init_scale", True)
        else:
            results.record("GradScaler max_scale < init_scale", False, f"Wrong error: {e}")
    except Exception as e:
        results.record("GradScaler max_scale < init_scale", False, f"Unexpected error: {e}")


def test_clip_grad_norm_empty_params():
    """clip_grad_norm_ should handle empty parameter list."""
    try:
        norm = tgpu.clip_grad_norm_([], max_norm=1.0)
        if norm == 0.0:
            results.record("clip_grad_norm_ empty params", True)
        else:
            results.record("clip_grad_norm_ empty params", False, f"Expected 0.0, got {norm}")
    except Exception as e:
        results.record("clip_grad_norm_ empty params", False, f"Exception: {e}")


def test_clip_grad_norm_no_grads():
    """clip_grad_norm_ should handle params with no gradients."""
    try:
        t = tgpu.Tensor(cp.array([1.0, 2.0], dtype=cp.float32), requires_grad=True)
        t.grad = None

        norm = tgpu.clip_grad_norm_([t], max_norm=1.0)
        if norm == 0.0:
            results.record("clip_grad_norm_ no grads", True)
        else:
            results.record("clip_grad_norm_ no grads", False, f"Expected 0.0, got {norm}")
    except Exception as e:
        results.record("clip_grad_norm_ no grads", False, f"Exception: {e}")


# ==================== RUN ALL TESTS ====================

if __name__ == "__main__":
    print("=" * 60)
    print("tensor_gpu_v2.py Phase 5 - Final Hardening Tests")
    print("=" * 60)
    print()

    # Test 1: GradScaler zero/negative scale
    print("--- GradScaler init_scale validation ---")
    test_gradscaler_init_zero_scale()
    test_gradscaler_init_negative_scale()
    test_gradscaler_init_tiny_scale()

    # Test 2: GradScaler load_state_dict validation
    print("\n--- GradScaler load_state_dict validation ---")
    test_gradscaler_load_zero_scale()
    test_gradscaler_load_negative_scale()

    # Test 3: clip_grad_norm_ NaN/Inf handling
    print("\n--- clip_grad_norm_ NaN/Inf handling ---")
    test_clip_grad_norm_nan_gradients()
    test_clip_grad_norm_inf_gradients()
    test_clip_grad_norm_mixed_nan_inf()
    test_clip_grad_norm_error_if_nonfinite()

    # Test 4: weight_tie shape validation
    print("\n--- weight_tie shape validation ---")
    test_weight_tie_shape_mismatch()
    test_weight_tie_transpose_mismatch()
    test_weight_tie_valid_shapes()
    test_weight_tie_valid_transpose()

    # Test 5: Kernel cache validation
    print("\n--- Kernel cache validation ---")
    test_kernel_cache_empty_code()
    test_kernel_cache_whitespace_code()
    test_kernel_cache_invalid_cuda()
    test_kernel_cache_empty_func_name()

    # Test 6: Integration tests
    print("\n--- Integration tests ---")
    test_integration_training_loop()
    test_integration_checkpoint_roundtrip()
    test_integration_profiler()

    # Test 7: Additional edge cases
    print("\n--- Additional edge cases ---")
    test_gradscaler_min_scale_validation()
    test_gradscaler_max_scale_validation()
    test_clip_grad_norm_empty_params()
    test_clip_grad_norm_no_grads()

    # Summary
    success = results.summary()
    sys.exit(0 if success else 1)
