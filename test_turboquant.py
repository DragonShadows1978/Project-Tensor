import numpy as np

import tensor_gpu_v2 as tg
import tensor_gpu_v2._quant as tq_internal


def test_turboquant_mse_roundtrip_shape_and_scale_cpu():
    tg.set_device("cpu")
    rng = np.random.default_rng(7)
    x = rng.standard_normal((5, 16)).astype(np.float32) * 3.0

    quant = tg.TurboQuantMSE(16, 3, seed=123, grid_size=4097, max_iter=64)
    enc = quant.quantize(x)
    recon = quant.dequantize(enc)

    assert enc.indices.shape == x.shape
    assert enc.norms.shape == (5,)
    assert recon.shape == x.shape

    original_norms = np.linalg.norm(x, axis=1)
    recon_norms = np.linalg.norm(recon, axis=1)
    np.testing.assert_allclose(recon_norms, original_norms, rtol=0.15, atol=1e-3)


def test_turboquant_mse_accepts_tensor_input():
    tg.set_device("cpu")
    rng = np.random.default_rng(11)
    x = tg.Tensor(rng.standard_normal((3, 8)).astype(np.float32), requires_grad=False)

    quant = tg.TurboQuantMSE(8, 2, seed=99, grid_size=2049, max_iter=48)
    enc = quant.quantize(x)
    recon = quant.dequantize(enc)

    assert enc.indices.shape == (3, 8)
    assert recon.shape == (3, 8)


def test_turboquant_prod_attention_score_matches_explicit_dequant():
    tg.set_device("cpu")
    rng = np.random.default_rng(21)
    q = rng.standard_normal((4, 12)).astype(np.float32)
    k = rng.standard_normal((4, 12)).astype(np.float32)

    quant = tg.TurboQuantProd(12, 3, seed=314, grid_size=4097, max_iter=64)
    enc = quant.quantize(k)
    recon = quant.dequantize(enc)

    approx_scores = quant.attention_score(q, enc)
    explicit_scores = np.sum(q * recon, axis=1)

    np.testing.assert_allclose(approx_scores, explicit_scores, rtol=1e-5, atol=1e-5)


def test_turboquant_prod_single_query_broadcasts_over_many_keys():
    tg.set_device("cpu")
    rng = np.random.default_rng(29)
    q = rng.standard_normal((10,)).astype(np.float32)
    k = rng.standard_normal((6, 10)).astype(np.float32)

    quant = tg.TurboQuantProd(10, 3, seed=2026, grid_size=2049, max_iter=48)
    enc = quant.quantize(k)
    recon = quant.dequantize(enc)

    scores = quant.attention_score(q, enc)
    explicit = recon @ q

    np.testing.assert_allclose(scores, explicit, rtol=1e-5, atol=1e-5)


def test_turboquant_mse_rotation_is_orthogonal():
    tg.set_device("cpu")
    quant = tg.TurboQuantMSE(16, 3, seed=123)

    identity = quant.rotation.T @ quant.rotation
    np.testing.assert_allclose(identity, np.eye(16, dtype=np.float32), rtol=1e-5, atol=1e-5)


def test_lloyd_max_codebook_is_sorted_and_bounded():
    codebook = tq_internal._build_lloyd_max_codebook(32, 3, grid_size=2049, max_iter=64)

    assert np.all(np.diff(codebook) >= 0.0)
    assert codebook.min() >= -1.0 - 1e-6
    assert codebook.max() <= 1.0 + 1e-6


def test_turboquant_mse_reconstruction_improves_with_more_bits():
    tg.set_device("cpu")
    rng = np.random.default_rng(101)
    x = rng.standard_normal((128, 24)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-12

    low = tg.TurboQuantMSE(24, 2, seed=77, grid_size=2049, max_iter=64)
    high = tg.TurboQuantMSE(24, 5, seed=77, grid_size=2049, max_iter=64)

    low_recon = low.dequantize(low.quantize(x))
    high_recon = high.dequantize(high.quantize(x))

    low_mse = float(np.mean((x - low_recon) ** 2))
    high_mse = float(np.mean((x - high_recon) ** 2))

    assert high_mse < low_mse


def test_turboquant_prod_dot_error_improves_with_more_bits():
    tg.set_device("cpu")
    rng = np.random.default_rng(303)
    q = rng.standard_normal((16, 20)).astype(np.float32)
    k = rng.standard_normal((32, 20)).astype(np.float32)

    low = tg.TurboQuantProd(20, 3, seed=88, grid_size=2049, max_iter=64)
    high = tg.TurboQuantProd(20, 6, seed=88, grid_size=2049, max_iter=64)

    low_enc = low.quantize(k)
    high_enc = high.quantize(k)

    exact = q @ k.T
    low_scores = np.stack([low.attention_score(query, low_enc) for query in q], axis=0)
    high_scores = np.stack([high.attention_score(query, high_enc) for query in q], axis=0)

    low_mae = float(np.mean(np.abs(exact - low_scores)))
    high_mae = float(np.mean(np.abs(exact - high_scores)))

    assert high_mae < low_mae
