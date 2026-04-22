#!/usr/bin/env python3
"""Numerical validation harness for TurboQuant.

This script checks three things:
1. structural invariants of the implementation
2. distortion-rate behavior for reconstruction
3. inner-product and retrieval quality for the product variant
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

import tensor_gpu_v2 as tg


def random_unit_vectors(count: int, dimension: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((count, dimension)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    return vectors


def random_vectors(count: int, dimension: int, seed: int, scale: float = 3.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((count, dimension)) * scale).astype(np.float32)


def uniform_scalar_baseline(vectors: np.ndarray, bits: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dimension = vectors.shape[-1]
    gaussian = rng.standard_normal((dimension, dimension))
    rotation, r = np.linalg.qr(gaussian)
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    rotation = (rotation * signs).astype(np.float32)

    norms = np.linalg.norm(vectors, axis=1)
    safe_norms = np.where(norms > 0.0, norms, 1.0)
    unit = vectors / safe_norms[:, None]
    unit = np.where(norms[:, None] > 0.0, unit, 0.0)

    rotated = unit @ rotation.T
    levels = 1 << bits
    centroids = np.linspace(-1.0, 1.0, levels, dtype=np.float32)
    boundaries = 0.5 * (centroids[:-1] + centroids[1:])
    indices = np.searchsorted(boundaries, rotated, side="left")
    recon = centroids[indices] @ rotation
    return recon * norms[:, None]


def mean_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dots = np.sum(a * b, axis=1)
    norms = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    safe_norms = np.where(norms > 0.0, norms, 1.0)
    cosine = dots / safe_norms
    return float(np.mean(cosine))


def topk_recall(exact: np.ndarray, approx: np.ndarray, k: int) -> float:
    exact_idx = np.argpartition(-exact, kth=k - 1, axis=1)[:, :k]
    approx_idx = np.argpartition(-approx, kth=k - 1, axis=1)[:, :k]
    matches = []
    for exact_row, approx_row in zip(exact_idx, approx_idx):
        matches.append(len(set(exact_row.tolist()) & set(approx_row.tolist())) / k)
    return float(np.mean(matches))


def collect_metrics(dimension: int, bits: int, seed: int, vector_count: int, query_count: int) -> Dict[str, float]:
    unit_vectors = random_unit_vectors(vector_count, dimension, seed)
    arbitrary_vectors = random_vectors(vector_count, dimension, seed + 1)
    queries = random_vectors(query_count, dimension, seed + 2, scale=1.5)

    mse_quant = tg.TurboQuantMSE(dimension, bits, seed=seed, grid_size=4097, max_iter=96)
    prod_quant = tg.TurboQuantProd(dimension, bits, seed=seed, grid_size=4097, max_iter=96)

    mse_enc = mse_quant.quantize(arbitrary_vectors)
    mse_recon = mse_quant.dequantize(mse_enc)
    unit_enc = mse_quant.quantize(unit_vectors)
    unit_recon = mse_quant.dequantize(unit_enc)
    uniform_recon = uniform_scalar_baseline(arbitrary_vectors, bits, seed)

    prod_enc = prod_quant.quantize(arbitrary_vectors)
    prod_recon = prod_quant.dequantize(prod_enc)

    exact_scores = queries @ arbitrary_vectors.T
    approx_scores = np.stack(
        [prod_quant.attention_score(query, prod_enc) for query in queries],
        axis=0,
    )
    explicit_scores = queries @ prod_recon.T

    ortho_err = float(
        np.linalg.norm(mse_quant.rotation.T @ mse_quant.rotation - np.eye(dimension, dtype=np.float32))
    )

    metrics = {
        "dimension": dimension,
        "bits": bits,
        "rotation_orthogonality_error": ortho_err,
        "codebook_min": float(mse_quant.codebook.min()),
        "codebook_max": float(mse_quant.codebook.max()),
        "codebook_sorted": bool(np.all(np.diff(mse_quant.codebook) >= 0.0)),
        "mse_mse": float(np.mean((arbitrary_vectors - mse_recon) ** 2)),
        "mse_uniform_baseline": float(np.mean((arbitrary_vectors - uniform_recon) ** 2)),
        "mse_vs_uniform_ratio": float(
            np.mean((arbitrary_vectors - mse_recon) ** 2)
            / np.mean((arbitrary_vectors - uniform_recon) ** 2)
        ),
        "unit_mse": float(np.mean((unit_vectors - unit_recon) ** 2)),
        "mean_cosine_similarity": mean_cosine_similarity(arbitrary_vectors, mse_recon),
        "unit_mean_cosine_similarity": mean_cosine_similarity(unit_vectors, unit_recon),
        "prod_dot_mae": float(np.mean(np.abs(exact_scores - approx_scores))),
        "prod_dot_rmse": float(np.sqrt(np.mean((exact_scores - approx_scores) ** 2))),
        "prod_vs_explicit_gap": float(np.mean(np.abs(approx_scores - explicit_scores))),
        "prod_recall_at_1": topk_recall(exact_scores, approx_scores, 1),
        "prod_recall_at_5": topk_recall(exact_scores, approx_scores, 5),
        "prod_recall_at_10": topk_recall(exact_scores, approx_scores, 10),
    }
    return metrics


def monotonic_summary(results: List[Dict[str, float]]) -> Dict[str, bool]:
    by_dim: Dict[int, List[Dict[str, float]]] = {}
    for row in results:
        by_dim.setdefault(int(row["dimension"]), []).append(row)

    summary = {}
    for dimension, rows in by_dim.items():
        rows = sorted(rows, key=lambda item: int(item["bits"]))
        mse_ok = all(rows[i + 1]["mse_mse"] < rows[i]["mse_mse"] for i in range(len(rows) - 1))
        dot_ok = all(rows[i + 1]["prod_dot_mae"] < rows[i]["prod_dot_mae"] for i in range(len(rows) - 1))
        summary[f"dim_{dimension}_mse_improves_with_bits"] = mse_ok
        summary[f"dim_{dimension}_prod_improves_with_bits"] = dot_ok
    return summary


def print_report(results: List[Dict[str, float]], checks: Dict[str, bool]) -> None:
    print("TurboQuant validation")
    print("====================")
    for row in results:
        print(
            "dim={dimension:>3} bits={bits:>2} "
            "mse={mse_mse:.6f} uniform={mse_uniform_baseline:.6f} "
            "ratio={mse_vs_uniform_ratio:.3f} cos={mean_cosine_similarity:.4f} "
            "dot_mae={prod_dot_mae:.5f} r@10={prod_recall_at_10:.3f} ppl_check={codebook_sorted}".format(
                **row
            )
        )
    print()
    print("Checks")
    print("------")
    for key, value in checks.items():
        print(f"{key}: {'PASS' if value else 'FAIL'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate TurboQuant numerically.")
    parser.add_argument("--dimensions", nargs="+", type=int, default=[64, 128, 256])
    parser.add_argument("--bits", nargs="+", type=int, default=[2, 4, 6])
    parser.add_argument("--vector-count", type=int, default=256)
    parser.add_argument("--query-count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    tg.set_device("cpu")

    results = []
    for dimension in args.dimensions:
        for bits in args.bits:
            results.append(
                collect_metrics(
                    dimension=dimension,
                    bits=bits,
                    seed=args.seed + dimension * 10 + bits,
                    vector_count=args.vector_count,
                    query_count=args.query_count,
                )
            )

    checks = monotonic_summary(results)
    print_report(results, checks)

    if args.json_out is not None:
        payload = {"results": results, "checks": checks}
        args.json_out.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
