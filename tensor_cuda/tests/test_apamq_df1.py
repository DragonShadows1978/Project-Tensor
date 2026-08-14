"""APAMQ-DF1/DF2 default-off launch-plan and variant-equivalence gates.

Registered V-on versus V-off tolerance: fp32 rtol=3e-3, atol=3e-3.  V1 changes
packed-storage lifetime, V2 reuses the stats launch's exact fp32 bulk scores,
V3 reassociates online-softmax partitions, and V4 reassociates stats partials.
The rail matches F-A1's existing split-K reassociation tolerance.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

import tensor_cuda as tc
from tensor_cuda.quant import _norm_ppf


TOL = dict(rtol=3e-3, atol=3e-3)
DF_ENV = (
    "TC_APAMQ_DF_V2", "TC_APAMQ_DF_V3", "TC_APAMQ_DF_V4",
    "TC_APA_SELECTIVE_PATH",
)


@pytest.fixture(autouse=True)
def _clean_env():
    for name in DF_ENV:
        os.environ.pop(name, None)
    yield
    for name in DF_ENV:
        os.environ.pop(name, None)


def _require_cuda():
    try:
        tc.tensor(np.zeros(1, dtype=np.float32))
        tc.synchronize()
    except Exception as exc:  # pragma: no cover - expected in CPU-only CI
        pytest.skip(f"CUDA unavailable: {exc}")


def _arrays(sequence=4096, dim=128, seed=991):
    rng = np.random.default_rng(seed)
    q = (rng.standard_normal((1, 16, 1, dim)) * 0.1).astype(np.float32)
    k = (rng.standard_normal((1, 1, sequence, dim)) * 0.1).astype(np.float32)
    v = (rng.standard_normal((1, 1, sequence, dim)) * 0.1).astype(np.float32)
    return q, k, v


def _run(q, k, v, *, workspace=None, v2=False, v3=False, v4=False):
    os.environ["TC_APA_SELECTIVE_PATH"] = "2"
    if v2:
        os.environ["TC_APAMQ_DF_V2"] = "1"
    else:
        os.environ.pop("TC_APAMQ_DF_V2", None)
    if v3:
        os.environ["TC_APAMQ_DF_V3"] = "1"
    else:
        os.environ.pop("TC_APAMQ_DF_V3", None)
    if v4:
        os.environ["TC_APAMQ_DF_V4"] = "1"
    else:
        os.environ.pop("TC_APAMQ_DF_V4", None)
    d = q.shape[-1]
    out = tc.apa_selective_attention_int4(
        tc.tensor(q), tc.tensor(k), tc.tensor(v), 1.0 / math.sqrt(d),
        float(_norm_ppf(0.90)), True, workspace=workspace,
    )
    tc.synchronize()
    return out.numpy().astype(np.float32)


def test_df2_per_stage_partition_policy_cpu():
    baseline = tc._apa_int4_decode_plan(16, 65536, grid_fill=False)
    v3 = tc._apa_int4_decode_plan(16, 65536, grid_fill=True)
    v2_v3 = tc._apa_int4_decode_plan(
        16, 65536, grid_fill=True, cache_bulk=True
    )
    v4 = tc._apa_int4_decode_plan(16, 65536, split_stats=True)
    short_v4 = tc._apa_int4_decode_plan(16, 4096, split_stats=True)
    short_v2_v3 = tc._apa_int4_decode_plan(
        16, 4096, grid_fill=True, cache_bulk=True
    )
    assert baseline == {
        "stats_partitions": 1, "stats_partition_keys": 65536,
        "split_partitions": 32, "split_partition_keys": 2048,
        "stats_partial_blocks": 16, "stats_reduce_blocks": 0,
        "split_blocks": 512, "merge_blocks": 16,
    }
    assert v3 == {
        "stats_partitions": 1, "stats_partition_keys": 65536,
        "split_partitions": 4, "split_partition_keys": 16384,
        "stats_partial_blocks": 16, "stats_reduce_blocks": 0,
        "split_blocks": 64, "merge_blocks": 16,
    }
    assert v2_v3["split_partitions"] == 32
    assert v2_v3["split_blocks"] == 512
    assert v4 == {
        "stats_partitions": 4, "stats_partition_keys": 16384,
        "split_partitions": 32, "split_partition_keys": 2048,
        "stats_partial_blocks": 64, "stats_reduce_blocks": 16,
        "split_blocks": 512, "merge_blocks": 16,
    }
    assert short_v4["stats_partitions"] == 4
    assert short_v4["stats_partial_blocks"] == 64
    assert short_v2_v3["split_partitions"] == 4
    assert short_v2_v3["split_blocks"] == 64


@pytest.mark.parametrize("v2,v3", [(True, False), (False, True), (True, True)])
def test_v2_v3_decode_equivalence_gpu(v2, v3):
    _require_cuda()
    q, k, v = _arrays()
    baseline = _run(q, k, v)
    variant = _run(q, k, v, v2=v2, v3=v3)
    np.testing.assert_allclose(variant, baseline, **TOL)


@pytest.mark.parametrize("cache_bulk", [False, True])
def test_v4_stats_splitk_matches_monolithic_gpu(cache_bulk):
    """DF2 gate: stats split-K and its V2 cache emission preserve decode."""
    _require_cuda()
    q, k, v = _arrays()
    monolithic = _run(q, k, v, v2=cache_bulk)
    split_stats = _run(q, k, v, v2=cache_bulk, v4=True)
    np.testing.assert_allclose(split_stats, monolithic, **TOL)


def test_v1_incremental_workspace_decode_equivalence_gpu():
    _require_cuda()
    q, k, v = _arrays()
    prefix = 257
    k_prefix = tc.tensor(k[:, :, :prefix])
    workspace = tc.apa_int4_workspace(k_prefix, capacity=k.shape[2])
    assert workspace.valid_rows == prefix
    baseline = _run(q, k, v)
    variant = _run(q, k, v, workspace=workspace)
    assert workspace.valid_rows == k.shape[2]
    np.testing.assert_allclose(variant, baseline, **TOL)


def test_v1_reset_repacks_after_nonappend_mutation_gpu():
    _require_cuda()
    q, k, v = _arrays(sequence=2048)
    k_t = tc.tensor(k)
    workspace = tc.apa_int4_workspace(k_t, capacity=k.shape[2])
    first = _run(q, k, v, workspace=workspace)
    workspace.reset()
    assert workspace.valid_rows == 0
    second = _run(q, k, v, workspace=workspace)
    assert workspace.valid_rows == k.shape[2]
    np.testing.assert_allclose(second, first, **TOL)
