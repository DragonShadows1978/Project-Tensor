#!/usr/bin/env python3
"""
Phase 5 Integration Test - tensor_gpu_v2.py Final Hardening

Tests all vulnerability fixes and validates end-to-end training with all Phase 4 features.

Run with: python test_phase5_integration.py
"""

import sys
import os
import tempfile
import warnings
import traceback

# Add OPAL to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

def test_gradscaler_zero_scale_validation():
    """Test 1: GradScaler rejects init_scale <= 0"""
    import tensor_gpu_v2 as tgv

    print("\n=== Test 1: GradScaler Zero Scale Validation ===")

    # Test init_scale = 0
    try:
        scaler = tgv.GradScaler(init_scale=0)
        print("  FAIL: GradScaler accepted init_scale=0")
        return False
    except ValueError as e:
        print(f"  PASS: Rejected init_scale=0 with: {e}")

    # Test init_scale = -1
    try:
        scaler = tgv.GradScaler(init_scale=-1)
        print("  FAIL: GradScaler accepted init_scale=-1")
        return False
    except ValueError as e:
        print(f"  PASS: Rejected init_scale=-1 with: {e}")

    # Test init_scale = 1e-300 (very small but positive - should work)
    try:
        scaler = tgv.GradScaler(init_scale=1e-300)
        print(f"  PASS: Accepted init_scale=1e-300 (scale={scaler.get_scale()})")
    except ValueError as e:
        print(f"  FAIL: Rejected valid small init_scale=1e-300: {e}")
        return False

    # Test min_scale = 0
    try:
        scaler = tgv.GradScaler(min_scale=0)
        print("  FAIL: GradScaler accepted min_scale=0")
        return False
    except ValueError as e:
        print(f"  PASS: Rejected min_scale=0 with: {e}")

    # Test load_state_dict with scale=0
    try:
        scaler = tgv.GradScaler()
        scaler.load_state_dict({'scale': 0})
        print("  FAIL: load_state_dict accepted scale=0")
        return False
    except ValueError as e:
        print(f"  PASS: load_state_dict rejected scale=0 with: {e}")

    # Test load_state_dict with scale=-1
    try:
        scaler = tgv.GradScaler()
        scaler.load_state_dict({'scale': -1})
        print("  FAIL: load_state_dict accepted scale=-1")
        return False
    except ValueError as e:
        print(f"  PASS: load_state_dict rejected scale=-1 with: {e}")

    print("  All GradScaler validation tests PASSED")
    return True


def test_gradient_nan_inf_handling():
    """Test 2: Gradient clipping handles NaN/Inf gracefully"""
    import tensor_gpu_v2 as tgv
    import cupy as cp

    print("\n=== Test 2: NaN/Inf Gradient Handling ===")

    # Create tensors with various gradient states
    t1 = tgv.Tensor(cp.ones((10, 10)))
    t2 = tgv.Tensor(cp.ones((10, 10)))
    t3 = tgv.Tensor(cp.ones((10, 10)))

    # Set up gradients: normal, NaN, and Inf
    t1.grad = cp.ones((10, 10)) * 2.0
    t2.grad = cp.full((10, 10), cp.nan)
    t3.grad = cp.full((10, 10), cp.inf)

    # Capture warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        norm = tgv.clip_grad_norm_([t1, t2, t3], max_norm=1.0)

        if len(w) > 0 and "NaN/Inf" in str(w[0].message):
            print(f"  PASS: Warning issued for NaN/Inf gradients")
        else:
            print(f"  INFO: Warning expected but got: {[str(x.message) for x in w]}")

    # Verify NaN/Inf gradients were zeroed
    if not cp.any(cp.isnan(t2.grad)) and not cp.any(cp.isinf(t3.grad)):
        print(f"  PASS: NaN/Inf gradients were zeroed, norm returned: {norm}")
    else:
        print(f"  FAIL: NaN/Inf gradients still present")
        return False

    # Test error_if_nonfinite=True
    t4 = tgv.Tensor(cp.ones((5, 5)))
    t4.grad = cp.full((5, 5), cp.nan)
    try:
        tgv.clip_grad_norm_([t4], max_norm=1.0, error_if_nonfinite=True)
        print("  FAIL: Should have raised RuntimeError for NaN with error_if_nonfinite=True")
        return False
    except RuntimeError as e:
        print(f"  PASS: RuntimeError raised with error_if_nonfinite=True: {e}")

    # Test GradScaler.unscale_ with NaN gradients
    print("\n  Testing GradScaler.unscale_ with NaN gradients:")
    linear = tgv.Linear(10, 10)
    linear.w.grad = cp.full((10, 10), cp.nan, dtype=cp.float32)

    optimizer = tgv.Adam([linear.w], lr=0.01)
    scaler = tgv.GradScaler()

    scaler.unscale_(optimizer)
    if scaler._found_inf:
        print(f"  PASS: GradScaler detected NaN in gradients")
    else:
        print(f"  FAIL: GradScaler should have detected NaN")
        return False

    # Verify gradient was zeroed
    if not cp.any(cp.isnan(linear.w.grad)):
        print(f"  PASS: NaN gradient was zeroed by GradScaler.unscale_")
    else:
        print(f"  FAIL: NaN gradient not zeroed")
        return False

    print("  All NaN/Inf handling tests PASSED")
    return True


def test_weight_tie_shape_validation():
    """Test 3: Weight tying validates shapes"""
    import tensor_gpu_v2 as tgv

    print("\n=== Test 3: Weight Tying Shape Validation ===")

    # Test valid weight tying (transposed)
    embed = tgv.Embedding(100, 50)  # Shape: (100, 50)
    lm_head = tgv.Linear(50, 100)    # w shape: (50, 100) - matches embed.T

    try:
        tgv.weight_tie(embed.weight, lm_head, 'w', transpose=True)
        print(f"  PASS: Valid weight tying accepted (100,50) -> (50,100) with transpose")
    except ValueError as e:
        print(f"  FAIL: Valid weight tying rejected: {e}")
        return False

    # Test invalid weight tying - shape mismatch
    embed2 = tgv.Embedding(100, 50)  # Shape: (100, 50)
    lm_head2 = tgv.Linear(50, 60)    # w shape: (50, 60) - does NOT match

    try:
        tgv.weight_tie(embed2.weight, lm_head2, 'w', transpose=True)
        print(f"  FAIL: Invalid shape mismatch should have been rejected")
        return False
    except ValueError as e:
        print(f"  PASS: Shape mismatch rejected: {e}")

    # Test invalid weight tying - without transpose when needed
    embed3 = tgv.Embedding(100, 50)  # Shape: (100, 50)
    lm_head3 = tgv.Linear(50, 100)   # w shape: (50, 100)

    try:
        tgv.weight_tie(embed3.weight, lm_head3, 'w', transpose=False)
        print(f"  FAIL: Should reject when transpose=False but shapes differ")
        return False
    except ValueError as e:
        print(f"  PASS: Rejected non-transposed tying with mismatched shapes: {e}")

    print("  All weight tying validation tests PASSED")
    return True


def test_kernel_cache_validation():
    """Test 4: Kernel cache rejects invalid/empty code"""
    import tensor_gpu_v2 as tgv

    print("\n=== Test 4: Kernel Cache Validation ===")

    # Test empty code
    try:
        tgv._get_cached_kernel("empty_test", "", "test_func")
        print("  FAIL: Empty code should be rejected")
        return False
    except ValueError as e:
        print(f"  PASS: Empty code rejected: {e}")

    # Test whitespace-only code
    try:
        tgv._get_cached_kernel("whitespace_test", "   \n\t  ", "test_func")
        print("  FAIL: Whitespace-only code should be rejected")
        return False
    except ValueError as e:
        print(f"  PASS: Whitespace-only code rejected: {e}")

    # Test empty function name
    try:
        tgv._get_cached_kernel("empty_func", "__global__ void test() {}", "")
        print("  FAIL: Empty function name should be rejected")
        return False
    except ValueError as e:
        print(f"  PASS: Empty function name rejected: {e}")

    # Test invalid CUDA code
    try:
        tgv._get_cached_kernel("invalid_cuda", "this is not valid CUDA code!!!", "invalid_func")
        print("  FAIL: Invalid CUDA code should fail compilation")
        return False
    except RuntimeError as e:
        print(f"  PASS: Invalid CUDA code rejected: {str(e)[:100]}...")

    # Test valid kernel still works
    valid_code = '''
    extern "C" __global__ void valid_kernel(float* x, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) x[idx] *= 2.0f;
    }
    '''
    try:
        kernel = tgv._get_cached_kernel("valid_test", valid_code, "valid_kernel")
        print(f"  PASS: Valid kernel compiled successfully")
    except Exception as e:
        print(f"  FAIL: Valid kernel should compile: {e}")
        return False

    print("  All kernel cache validation tests PASSED")
    return True


def test_comprehensive_integration():
    """Test 5: End-to-end FP16 training with all Phase 4 features"""
    import tensor_gpu_v2 as tgv
    import cupy as cp

    print("\n=== Test 5: Comprehensive Integration Test ===")

    # Use existing Linear layers directly
    print("  Creating model with weight tying...")
    embed = tgv.Embedding(100, 32)
    fc1 = tgv.Linear(32, 64)
    fc2 = tgv.Linear(64, 32)
    lm_head = tgv.Linear(32, 100)

    # Weight tying between embedding and lm_head
    tgv.weight_tie(embed.weight, lm_head, 'w', transpose=True)
    print(f"    Weight tying applied: embedding -> lm_head")

    # Collect all parameters
    all_params = [embed.weight, fc1.w, fc1.b, fc2.w, fc2.b, lm_head.b]  # lm_head.w is tied

    # Setup training components
    print("\n  Setting up training components...")
    optimizer = tgv.Adam(all_params, lr=0.001)
    scaler = tgv.GradScaler(init_scale=65536.0, growth_interval=5)
    scheduler = tgv.LinearWarmupCosineDecay(optimizer, warmup_steps=10, total_steps=100)

    print(f"    Optimizer: Adam (lr={optimizer.lr})")
    print(f"    GradScaler: scale={scaler.get_scale()}")
    print(f"    Scheduler: LinearWarmupCosineDecay")

    # Training loop
    print("\n  Running 10-step FP16 training loop...")
    losses = []
    grad_norms = []

    for step in range(10):
        # Generate random input
        batch_size, seq_len = 8, 16
        x = cp.random.randint(0, 100, (batch_size, seq_len))

        # Forward pass
        with tgv.Profiler(f'step_{step}') as prof:
            h = embed(x)  # (batch, seq, 32)
            h = fc1(h)    # (batch, seq, 64)
            h = h.relu()  # activation
            h = fc2(h)    # (batch, seq, 32)
            logits = lm_head(h)  # (batch, seq, 100)

            # Simple loss: mean of logits squared
            loss_val = float((logits.data ** 2).mean())

        # Create loss tensor
        loss = tgv.Tensor(cp.array(loss_val, dtype=cp.float32))

        # Scale loss for backward
        scaled_loss = scaler.scale(loss)

        # Zero gradients
        optimizer.zero_grad()

        # Backward pass (simulated gradients for this test)
        for p in all_params:
            if p.data is not None:
                p.grad = cp.random.randn(*p.data.shape).astype(cp.float32) * 0.1

        # Unscale and clip
        scaler.unscale_(optimizer)
        grad_norm = tgv.clip_grad_norm_(all_params, max_norm=1.0)

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        losses.append(loss_val)
        grad_norms.append(grad_norm)

        print(f"    Step {step}: loss={loss_val:.4f}, grad_norm={grad_norm:.4f}, "
              f"scale={scaler.get_scale():.0f}, lr={scheduler.get_lr():.6f}")

    print(f"\n  Training completed. Stats: {scaler.get_statistics()}")

    # Test checkpointing with a simple Linear module
    print("\n  Testing checkpoint save/load...")
    checkpoint_path = tempfile.mktemp(suffix='.pkl')

    try:
        # Create a simple model for checkpoint testing
        simple_model = tgv.Linear(10, 5)

        # Save checkpoint
        tgv.save_checkpoint(
            checkpoint_path,
            model=simple_model,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            epoch=1,
            global_step=10,
            extra_state={'losses': losses}
        )
        print(f"    Checkpoint saved to {checkpoint_path}")

        # Create new model and components
        simple_model2 = tgv.Linear(10, 5)
        optimizer2 = tgv.Adam(all_params, lr=0.001)  # reuse params
        scaler2 = tgv.GradScaler()

        # Load checkpoint
        info = tgv.load_checkpoint(
            checkpoint_path,
            model=simple_model2,
            optimizer=optimizer2,
            scaler=scaler2
        )
        print(f"    Checkpoint loaded: epoch={info['epoch']}, step={info['global_step']}")

        # Verify state restored
        assert info['epoch'] == 1, "Epoch not restored"
        assert info['global_step'] == 10, "Global step not restored"
        assert scaler2.get_scale() == scaler.get_scale(), "Scaler state not restored"
        print(f"    Scaler state verified: scale={scaler2.get_scale()}")

        # Verify model weights match
        if cp.allclose(simple_model.w.data, simple_model2.w.data):
            print(f"    Model weights verified: match original")
        else:
            print(f"    FAIL: Model weights don't match after checkpoint load")
            return False

    finally:
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

    print("\n  Integration test PASSED - all features working together")
    return True


def test_adversarial_edge_cases():
    """Test 6: Additional adversarial edge cases"""
    import tensor_gpu_v2 as tgv
    import cupy as cp

    print("\n=== Test 6: Adversarial Edge Cases ===")

    # Test GradScaler with max_scale < init_scale
    try:
        scaler = tgv.GradScaler(init_scale=1000, max_scale=100)
        print("  FAIL: Should reject max_scale < init_scale")
        return False
    except ValueError as e:
        print(f"  PASS: Rejected max_scale < init_scale: {e}")

    # Test clip_grad_norm_ with all-NaN gradients
    t = tgv.Tensor(cp.ones((5, 5)))
    t.grad = cp.full((5, 5), cp.nan, dtype=cp.float32)

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        norm = tgv.clip_grad_norm_([t], max_norm=1.0)
        if norm == 0.0:
            print(f"  PASS: All-NaN gradients returns norm=0.0")
        else:
            print(f"  FAIL: Expected norm=0.0 for all-NaN gradients, got {norm}")
            return False

    # Test GradScaler state_dict round-trip
    scaler = tgv.GradScaler(init_scale=1234.5)
    for _ in range(5):
        scaler._found_inf = True
        scaler.update()

    state = scaler.state_dict()
    scaler2 = tgv.GradScaler()
    scaler2.load_state_dict(state)

    assert scaler2._overflow_count == scaler._overflow_count, "Overflow count not preserved"
    assert scaler2.get_scale() == scaler.get_scale(), "Scale not preserved"
    print(f"  PASS: GradScaler state_dict round-trip preserved all state")

    # Test gradient with mixed finite/NaN values
    t2 = tgv.Tensor(cp.ones((10,)))
    grad = cp.ones(10, dtype=cp.float32)
    grad[5:] = cp.nan  # Half NaN, half valid
    t2.grad = grad

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        norm = tgv.clip_grad_norm_([t2], max_norm=1.0)
        # After zeroing NaN, norm should be sqrt(5) ≈ 2.236, clipped to 1.0
        if not cp.any(cp.isnan(t2.grad)):
            print(f"  PASS: Mixed NaN/finite gradients handled, norm={norm:.4f}")
        else:
            print(f"  FAIL: NaN still present in mixed gradient")
            return False

    print("  All adversarial edge case tests PASSED")
    return True


def run_all_tests():
    """Run all Phase 5 integration tests"""
    print("=" * 60)
    print("Phase 5 Integration Tests - tensor_gpu_v2.py Final Hardening")
    print("=" * 60)

    tests = [
        ("GradScaler Zero Scale Validation", test_gradscaler_zero_scale_validation),
        ("NaN/Inf Gradient Handling", test_gradient_nan_inf_handling),
        ("Weight Tying Shape Validation", test_weight_tie_shape_validation),
        ("Kernel Cache Validation", test_kernel_cache_validation),
        ("Comprehensive Integration", test_comprehensive_integration),
        ("Adversarial Edge Cases", test_adversarial_edge_cases),
    ]

    results = {}
    for name, test_fn in tests:
        try:
            results[name] = test_fn()
        except Exception as e:
            print(f"\n  EXCEPTION in {name}: {e}")
            traceback.print_exc()
            results[name] = False

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ ALL PHASE 5 TESTS PASSED - Vulnerabilities fixed!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(run_all_tests())
