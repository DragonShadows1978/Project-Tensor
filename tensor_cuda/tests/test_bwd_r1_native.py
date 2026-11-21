"""Run the native backward invariants in the imported engine."""

import json

import pytest

import tensor_cuda as tc
from tensor_cuda import _C

OVERLAP_CHECKS = {"adjacent", "distinct-storage", "right-overlap", "left-overlap"}


def check_case(case, expected_checks, *, gpu=False):
    result = _C._bwd_native_selftest(case)
    assert result["case"] == case, result
    if gpu and result["status"] == "skipped":
        assert result["reason"] == "no CUDA device present" and not result["checks"], result
        pytest.skip(result["reason"])
    assert result["checks"].keys() == expected_checks, result
    assert result["status"] == "passed" and all(row["passed"] for row in result["checks"].values()), (
        json.dumps(result, sort_keys=True)
    )
    print(f"native backward self-test passed: {case}", flush=True)


@pytest.mark.parametrize("case,checks", [
    pytest.param("overlap", OVERLAP_CHECKS, id="overlap"),
    pytest.param("batch-mismatch", {"batch-mismatch"}, id="batch-mismatch"),
])
def test_host_native_backward_checks(case, checks):
    check_case(case, checks)


def test_host_native_overlap_under_no_grad():
    previous = tc.is_grad_enabled()
    with tc.no_grad():
        assert not tc.is_grad_enabled()
        check_case("overlap", OVERLAP_CHECKS)
        assert not tc.is_grad_enabled()
    assert tc.is_grad_enabled() == previous


@pytest.mark.parametrize("case", ["current-device", "zero-k"])
def test_gpu_native_backward_checks(case):
    check_case(case, {case}, gpu=True)
