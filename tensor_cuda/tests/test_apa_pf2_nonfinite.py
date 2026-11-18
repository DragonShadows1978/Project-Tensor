"""PF2: REV1's sink/norm witnesses and partition-carry decode coverage.
Prior art: IEEE 754-2008; Project-Tensor REV1 reproductions (2026), preserved
operands and extended decode coverage by kernel and dtype. ISO C++17/CERT INT32-C boundary widening
(unverified - lead to check). No new numerical algorithm.
"""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import apa_pf2f3_nonfinite as N


# PF2-F1/F2 (Project-Tensor, 2026): compile the actual index helper under
# UBSan and exercise its C++17 int64_t additions across the INT_MAX boundary.
HARNESS = r'''
#include <climits>
#include <iostream>
#include <type_traits>
int main() {
  static_assert(std::is_same<decltype(diagnostic_tail_index(0,0)), int64_t>::value);
  int tests=0;
  for (const int S : {INT_MAX, INT_MAX-31}) {
    for (int lane=0; lane<32; ++lane) {
      const int64_t expected=int64_t(S)+int64_t(lane);
      const int64_t got=diagnostic_tail_index(S,lane);
      if(got!=expected || got<0) return 10;
      const int64_t next=diagnostic_tail_index(got,32);
      if(next!=expected+int64_t(32) || next<=got) return 11;
      ++tests;
    }
  }
  for(const int S : {1,17,31,32,33,257}) {
    for(int smax=0;smax<=S;++smax) {
      int counts[258]={};
      for(int lane=0;lane<32;++lane)
        for(int64_t j=diagnostic_tail_index(smax,lane);j<S;j=diagnostic_tail_index(j,32)) {
          if(j<smax || j<0 || j>=S) return 12;
          ++counts[j];
        }
      for(int j=0;j<S;++j) if(counts[j]!=(j>=smax ? 1 : 0)) return 13;
      ++tests;
    }
  }
  std::cout << "INDEX_PASS checks=" << tests << " boundary_lanes=64 additions=128\n";
}
'''


def test_actual_diagnostic_tail_helper(tmp_path):
    compiler = next(
        (path for name in ("c++", "g++", "clang++") if (path := shutil.which(name))),
        None,
    )
    if compiler is None:
        pytest.skip("no host C++ compiler")

    header = (ROOT / "tensor_cuda/src/apa_spo/packed_prefill.cuh").read_text()
    assert "int64_t j = diagnostic_tail_index(smax, lane)" in header
    assert "j = diagnostic_tail_index(j, 32)" in header
    helper = header.split("// APA_PF2F1_INDEX_BEGIN\n", 1)[1].split("// APA_PF2F1_INDEX_END", 1)[0]
    source = tmp_path / "index_check.cpp"
    binary = tmp_path / "index_check"
    source.write_text("#include <cstdint>\n#define __host__\n#define __device__\n" + helper + HARNESS)
    compile_result = subprocess.run(
        [compiler, "-std=c++17", "-O1", "-fsanitize=undefined",
         "-fno-sanitize-recover=all", str(source), "-o", str(binary)],
        capture_output=True, text=True, timeout=60, cwd=tmp_path,
    )
    assert compile_result.returncode == 0, compile_result.stdout + compile_result.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=30, cwd=tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "INDEX_PASS checks=441 boundary_lanes=64 additions=128" in result.stdout


@pytest.fixture(scope="module")
def runtime():
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        pytest.skip("no GPU visible")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    import tensor_cuda

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("TC_APA_SP", "1")
        yield tensor_cuda


@pytest.mark.parametrize('case',N.sink_cases())
def test_nonfinite_sink(runtime,case):
    r=N.run_sink(runtime,case)
    assert not r['failed'],r


@pytest.mark.parametrize('case',N.norm_cases())
def test_nonfinite_stored_norm(runtime,case):
    r=N.run_norm(runtime,case)
    assert not r['failed'],r


@pytest.mark.parametrize('case', N.partition_norm_cases())
def test_decode_partition_raw_nonfinite_norm(runtime, case):
    result = N.run_partition_norm(runtime, case)
    assert not result['failed'], result
