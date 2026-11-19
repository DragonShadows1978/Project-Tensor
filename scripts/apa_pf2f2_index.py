"""PF2-F1 index regression (2026), rebound to PF2-F2 scratch.
Prior art: ISO C++17 integer promotion and CERT INT32-C (unverified).
"""
from pathlib import Path
import os
import subprocess
import apa_pf2f2_build as B

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


def check(header, tag):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert os.environ['TMPDIR'].startswith('/tmp/pf2f2_')
    s = Path(header).read_text()
    assert 'int64_t j = diagnostic_tail_index(smax, lane)' in s
    assert 'j = diagnostic_tail_index(j, 32)' in s
    fn = s.split('// APA_PF2F1_INDEX_BEGIN\n', 1)[1].split('// APA_PF2F1_INDEX_END', 1)[0]
    scratch = Path(os.environ['TMPDIR']) / tag
    scratch.mkdir(parents=True, exist_ok=True)
    src, binary = scratch / 'index_check.cpp', scratch / 'index_check'
    src.write_text('#include <cstdint>\n#define __host__\n#define __device__\n' + fn + HARNESS)
    cmd = ['timeout', '60s', 'nice', '-n', '19', 'c++', '-std=c++17', '-O1', '-fsanitize=undefined',
           '-fno-sanitize-recover=all', str(src), '-o', str(binary)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    r = subprocess.run(['timeout', '30s', str(binary)], capture_output=True, text=True)
    return dict(tag=tag, header=str(header), header_sha256=B.sha(header), extracted_helper=fn,
                compiler_argv=cmd, compile_exit=p.returncode, exit=r.returncode,
                stdout=r.stdout, stderr=r.stderr, status='PASS' if r.returncode==0 and 'INDEX_PASS' in r.stdout else 'FAIL')

