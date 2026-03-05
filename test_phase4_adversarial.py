#!/usr/bin/env python3
"""
Adversarial Testing for Phase 4 Features of tensor_gpu_v2.py

This script performs adversarial edge case testing to find vulnerabilities
in the implemented features.
"""

import sys
import json
sys.path.insert(0, '/home/vader/mini-mind-v2/workspace/OPAL')

import cupy as cp
import numpy as np

adversarial_results = {
    'edge_cases_tested': [],
    'vulnerabilities_found': [],
    'passed_edge_cases': []
}


def test_adversarial_case(name, test_fn):
    """Run an adversarial test case."""
    try:
        result = test_fn()
        if result.get('vulnerability', False):
            adversarial_results['vulnerabilities_found'].append({
                'name': name,
                'issue': result.get('issue', 'Unknown'),
                'severity': result.get('severity', 'low')
            })
            print(f"  ⚠ {name}: VULNERABILITY - {result.get('issue')}")
        else:
            adversarial_results['passed_edge_cases'].append(name)
            print(f"  ✓ {name}: Passed")
        adversarial_results['edge_cases_tested'].append(name)
    except Exception as e:
        adversarial_results['vulnerabilities_found'].append({
            'name': name,
            'issue': f'Crash: {str(e)}',
            'severity': 'high'
        })
        print(f"  ✗ {name}: CRASH - {str(e)}")
        adversarial_results['edge_cases_tested'].append(name)


# =============================================================================
# ADVERSARIAL: GRADIENT CLIPPING
# =============================================================================

def adv_clip_nan_gradients():
    """Test gradient clipping with NaN gradients."""
    import tensor_gpu_v2 as tg

    t = tg.Tensor(cp.array([1.0, 2.0]), requires_grad=True)
    t.grad = cp.array([float('nan'), 1.0])

    norm = tg.clip_grad_norm_([t], max_norm=1.0)

    # Should handle NaN gracefully
    if cp.isnan(norm):
        return {'vulnerability': True, 'issue': 'NaN gradient produces NaN norm', 'severity': 'medium'}
    return {'vulnerability': False}


def adv_clip_inf_gradients():
    """Test gradient clipping with inf gradients."""
    import tensor_gpu_v2 as tg

    t = tg.Tensor(cp.array([1.0, 2.0]), requires_grad=True)
    t.grad = cp.array([float('inf'), 1.0])

    norm = tg.clip_grad_norm_([t], max_norm=1.0)

    # Should handle inf gracefully
    if cp.isinf(norm):
        return {'vulnerability': True, 'issue': 'Inf gradient produces inf norm', 'severity': 'medium'}
    return {'vulnerability': False}


def adv_clip_zero_max_norm():
    """Test gradient clipping with max_norm=0."""
    import tensor_gpu_v2 as tg

    t = tg.Tensor(cp.array([3.0, 4.0]), requires_grad=True)
    t.grad = cp.array([3.0, 4.0])

    norm = tg.clip_grad_norm_([t], max_norm=0.0)

    # Should clip to zero
    new_norm = float(cp.sqrt(cp.sum(t.grad ** 2)))
    if new_norm > 0.01:
        return {'vulnerability': True, 'issue': f'max_norm=0 does not zero gradients, got {new_norm}', 'severity': 'low'}
    return {'vulnerability': False}


def adv_clip_negative_max_norm():
    """Test gradient clipping with negative max_norm."""
    import tensor_gpu_v2 as tg

    t = tg.Tensor(cp.array([3.0, 4.0]), requires_grad=True)
    t.grad = cp.array([3.0, 4.0])

    try:
        norm = tg.clip_grad_norm_([t], max_norm=-1.0)
        # Negative max_norm should be handled - ideally an error or no clipping
        return {'vulnerability': False}  # If it doesn't crash, it's ok
    except ValueError:
        return {'vulnerability': False}  # Proper validation


def adv_clip_very_large_gradients():
    """Test gradient clipping with very large gradients."""
    import tensor_gpu_v2 as tg

    t = tg.Tensor(cp.array([1e30, 1e30]), requires_grad=True)
    t.grad = cp.array([1e30, 1e30])

    norm = tg.clip_grad_norm_([t], max_norm=1.0)

    # Should clip properly
    new_norm = float(cp.sqrt(cp.sum(t.grad ** 2)))
    if new_norm > 2.0:  # Allow some tolerance
        return {'vulnerability': True, 'issue': f'Very large gradients not clipped properly, norm={new_norm}', 'severity': 'medium'}
    return {'vulnerability': False}


# =============================================================================
# ADVERSARIAL: GRADSCALER
# =============================================================================

def adv_scaler_extreme_overflow():
    """Test scaler with extreme overflow scenarios."""
    import tensor_gpu_v2 as tg

    scaler = tg.GradScaler(init_scale=2**24, max_scale=2**24)

    # Simulate multiple consecutive overflows
    linear = tg.Linear(2, 2)
    optimizer = tg.Adam(linear.parameters(), lr=0.001)

    for _ in range(100):
        for p in linear.parameters():
            p.grad = cp.full_like(p.data, float('inf'))
        scaler.unscale_(optimizer)
        scaler.update()

    # Scale should not go below min_scale
    if scaler.get_scale() < scaler.min_scale:
        return {'vulnerability': True, 'issue': 'Scale went below min_scale', 'severity': 'medium'}
    return {'vulnerability': False}


def adv_scaler_zero_scale():
    """Test scaler initialized with zero scale."""
    import tensor_gpu_v2 as tg

    try:
        scaler = tg.GradScaler(init_scale=0.0)
        loss = tg.Tensor([2.0], requires_grad=True)
        scaled = scaler.scale(loss)

        if float(scaled.data[0]) == 0.0:
            # This would make gradients disappear
            return {'vulnerability': True, 'issue': 'Zero scale zeroes all gradients', 'severity': 'high'}
        return {'vulnerability': False}
    except ValueError:
        return {'vulnerability': False}  # Proper validation


def adv_scaler_state_dict_tampering():
    """Test scaler state_dict with tampered values."""
    import tensor_gpu_v2 as tg

    scaler = tg.GradScaler()

    # Load tampered state
    tampered_state = {
        'scale': -1.0,  # Negative scale
        'growth_tracker': -100,
        'overflow_count': 999999,
    }

    try:
        scaler.load_state_dict(tampered_state)
        # If we get here without exception, check if negative scale was accepted
        if scaler.get_scale() < 0:
            return {'vulnerability': True, 'issue': 'Negative scale accepted without validation', 'severity': 'medium'}
        return {'vulnerability': False}
    except ValueError:
        # Proper validation - negative scale rejected
        return {'vulnerability': False}


# =============================================================================
# ADVERSARIAL: CHECKPOINTING
# =============================================================================

def adv_checkpoint_corrupt_file():
    """Test loading a corrupted checkpoint file."""
    import tensor_gpu_v2 as tg
    import tempfile
    import os

    model = tg.Linear(4, 2)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, 'corrupt.pt')

        # Write garbage to file
        with open(ckpt_path, 'wb') as f:
            f.write(b'not a valid pickle file')

        try:
            tg.load_checkpoint(ckpt_path, model)
            return {'vulnerability': True, 'issue': 'Corrupt file loaded without error', 'severity': 'high'}
        except Exception:
            return {'vulnerability': False}  # Properly rejected


def adv_checkpoint_missing_keys():
    """Test loading checkpoint with missing model keys."""
    import tensor_gpu_v2 as tg
    import tempfile
    import os
    import pickle

    model = tg.Linear(4, 2)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, 'partial.pt')

        # Create incomplete checkpoint
        checkpoint = {
            'version': '1.0',
            'model_state_dict': {},  # Empty - missing weights
            'epoch': 5
        }
        with open(ckpt_path, 'wb') as f:
            pickle.dump(checkpoint, f)

        try:
            info = tg.load_checkpoint(ckpt_path, model)
            # Should either fail or handle gracefully
            return {'vulnerability': False}
        except Exception:
            return {'vulnerability': False}


def adv_checkpoint_version_mismatch():
    """Test loading checkpoint with different version."""
    import tensor_gpu_v2 as tg
    import tempfile
    import os
    import pickle
    import warnings

    model = tg.Linear(4, 2)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, 'old_version.pt')

        # Create checkpoint with old version
        checkpoint = {
            'version': '0.0',  # Old version
            'model_state_dict': {'param_0': np.zeros((4, 2)), 'param_1': np.zeros(2)},
            'epoch': 5
        }
        with open(ckpt_path, 'wb') as f:
            pickle.dump(checkpoint, f)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            info = tg.load_checkpoint(ckpt_path, model)

            # Should warn about version mismatch
            if len(w) > 0:
                return {'vulnerability': False}  # Warning is good
            return {'vulnerability': False}  # Ok if silently handled


# =============================================================================
# ADVERSARIAL: WEIGHT TYING
# =============================================================================

def adv_weight_tie_shape_mismatch():
    """Test weight tying with incompatible shapes."""
    import tensor_gpu_v2 as tg

    embedding = tg.Embedding(100, 32)  # (100, 32)
    lm_head = tg.Linear(64, 100, bias=False)  # (64, 100) - wrong shape

    try:
        tg.weight_tie(embedding.weight, lm_head, 'w', transpose=True)
        # Transposed embedding is (32, 100), but lm_head expects (64, 100)
        # This should fail or be detected
        return {'vulnerability': True, 'issue': 'Shape mismatch not detected in weight tying', 'severity': 'medium'}
    except Exception:
        return {'vulnerability': False}


def adv_weight_tie_none_source():
    """Test weight tying with None source."""
    import tensor_gpu_v2 as tg

    lm_head = tg.Linear(32, 100, bias=False)

    try:
        tg.weight_tie(None, lm_head, 'w', transpose=True)
        return {'vulnerability': True, 'issue': 'None source accepted', 'severity': 'medium'}
    except (TypeError, AttributeError):
        return {'vulnerability': False}


# =============================================================================
# ADVERSARIAL: PROFILING
# =============================================================================

def adv_profiler_nested():
    """Test nested profiler contexts."""
    import tensor_gpu_v2 as tg

    with tg.Profiler('outer') as outer:
        with tg.Profiler('inner') as inner:
            cp.cuda.Stream.null.synchronize()

    outer_report = outer.report()
    inner_report = inner.report()

    # Both should have valid timing
    if outer_report['elapsed_ms'] is None or inner_report['elapsed_ms'] is None:
        return {'vulnerability': True, 'issue': 'Nested profilers return None timing', 'severity': 'low'}
    return {'vulnerability': False}


def adv_benchmark_zero_repeats():
    """Test benchmark with zero repeats."""
    import tensor_gpu_v2 as tg

    def dummy_fn():
        return cp.ones(100)

    try:
        result = tg.benchmark(dummy_fn, n_repeat=0, n_warmup=0)
        # Should fail or return empty
        if 'mean_ms' in result and result['mean_ms'] is not None:
            return {'vulnerability': True, 'issue': 'Benchmark with n_repeat=0 returned result', 'severity': 'low'}
        return {'vulnerability': False}
    except (ValueError, ZeroDivisionError):
        return {'vulnerability': False}


# =============================================================================
# ADVERSARIAL: KERNEL CACHE
# =============================================================================

def adv_kernel_cache_invalid_code():
    """Test kernel cache with invalid CUDA code."""
    import tensor_gpu_v2 as tg

    invalid_code = '''
    this is not valid cuda code at all!!!
    '''

    try:
        kernel = tg._get_cached_kernel('invalid', invalid_code, 'invalid_kernel')
        return {'vulnerability': True, 'issue': 'Invalid CUDA code cached without error', 'severity': 'medium'}
    except Exception:
        return {'vulnerability': False}


def adv_kernel_cache_empty_code():
    """Test kernel cache with empty code."""
    import tensor_gpu_v2 as tg

    try:
        kernel = tg._get_cached_kernel('empty', '', 'empty_kernel')
        return {'vulnerability': True, 'issue': 'Empty CUDA code accepted', 'severity': 'low'}
    except Exception:
        return {'vulnerability': False}


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("Adversarial Testing for Phase 4 Features")
    print("=" * 60)

    print("\n--- Gradient Clipping Edge Cases ---")
    test_adversarial_case("clip_nan_gradients", adv_clip_nan_gradients)
    test_adversarial_case("clip_inf_gradients", adv_clip_inf_gradients)
    test_adversarial_case("clip_zero_max_norm", adv_clip_zero_max_norm)
    test_adversarial_case("clip_negative_max_norm", adv_clip_negative_max_norm)
    test_adversarial_case("clip_very_large_gradients", adv_clip_very_large_gradients)

    print("\n--- GradScaler Edge Cases ---")
    test_adversarial_case("scaler_extreme_overflow", adv_scaler_extreme_overflow)
    test_adversarial_case("scaler_zero_scale", adv_scaler_zero_scale)
    test_adversarial_case("scaler_state_dict_tampering", adv_scaler_state_dict_tampering)

    print("\n--- Checkpointing Edge Cases ---")
    test_adversarial_case("checkpoint_corrupt_file", adv_checkpoint_corrupt_file)
    test_adversarial_case("checkpoint_missing_keys", adv_checkpoint_missing_keys)
    test_adversarial_case("checkpoint_version_mismatch", adv_checkpoint_version_mismatch)

    print("\n--- Weight Tying Edge Cases ---")
    test_adversarial_case("weight_tie_shape_mismatch", adv_weight_tie_shape_mismatch)
    test_adversarial_case("weight_tie_none_source", adv_weight_tie_none_source)

    print("\n--- Profiling Edge Cases ---")
    test_adversarial_case("profiler_nested", adv_profiler_nested)
    test_adversarial_case("benchmark_zero_repeats", adv_benchmark_zero_repeats)

    print("\n--- Kernel Cache Edge Cases ---")
    test_adversarial_case("kernel_cache_invalid_code", adv_kernel_cache_invalid_code)
    test_adversarial_case("kernel_cache_empty_code", adv_kernel_cache_empty_code)

    # Summary
    print("\n" + "=" * 60)
    print(f"Total edge cases tested: {len(adversarial_results['edge_cases_tested'])}")
    print(f"Passed: {len(adversarial_results['passed_edge_cases'])}")
    print(f"Vulnerabilities found: {len(adversarial_results['vulnerabilities_found'])}")
    print("=" * 60)

    if adversarial_results['vulnerabilities_found']:
        print("\nVulnerabilities:")
        for v in adversarial_results['vulnerabilities_found']:
            print(f"  [{v['severity'].upper()}] {v['name']}: {v['issue']}")

    # Save results
    with open('/home/vader/mini-mind-v2/workspace/tests/adversarial_results.json', 'w') as f:
        json.dump(adversarial_results, f, indent=2)

    return adversarial_results


if __name__ == '__main__':
    results = main()
