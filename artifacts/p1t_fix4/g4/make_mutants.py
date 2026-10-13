"""P1T-FIX4 G4: build each registered mutant tree OUTSIDE the worktree (/tmp/p1t_fix4_mut_<m>/), one asserted edit each.

Each tree is a copy of the worktree's tensor_cuda/ and tc_signal/ (sources and tests; no build/ dir, no .so). The edit's
unified diff is written to artifacts/p1t_fix4/g4/mut_<m>.diff.
  (a) V4-01: the guard removed. dlpack_fence_consumer never switches to the tensor's device: the event is created and
      recorded on the CURRENT device, as in 62c8d9f (the label proof stays).
  (b) V4-03: the check removed. The import's cudaGetDevice result is ignored again and `cur` starts at 0, as in 62c8d9f.
  (c) V4-04: the capture removed. synchronize() no longer takes the calling thread's pre-existing error.
Usage: make_mutants.py <m>... (a b c).
Prior art: DeMillo, Lipton & Sayward (1978) mutation testing; the registered mutants are the order's (a), (b), (c).
"""
import difflib
import shutil
import sys
from pathlib import Path

W = Path('/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix4')
G4 = W / 'artifacts/p1t_fix4/g4'

MUTANTS = {
    'a': ('tensor_cuda/src/bindings.cpp',
          '  const bool switched = (prev != device);\n',
          '  const bool switched = false;  // MUTANT (a): V4-01 guard removed (fence on the CURRENT device, as 62c8d9f)\n'
          '  (void)device;\n'),
    'b': ('tensor_cuda/src/bindings.cpp',
          '''  int cur = -1;
  const cudaError_t qe = cudaGetDevice(&cur);
  if (qe != cudaSuccess) {
    cudaGetLastError();
    throw std::runtime_error(std::string("from_dlpack: the current-device query cudaGetDevice failed (") +
                             cudaGetErrorName(qe) + ": " + cudaGetErrorString(qe) +
                             "); refused before taking ownership (the capsule is not consumed)");
  }
''',
          '''  int cur = 0;
  cudaGetDevice(&cur);  // MUTANT (b): V4-03 check removed (result ignored, cur stays 0 on failure, as 62c8d9f)
'''),
    'c': ('tc_signal/src/dlpack_operand.cpp',
          '  const cudaError_t prior = cudaGetLastError();\n',
          '  const cudaError_t prior = cudaSuccess;  // MUTANT (c): V4-04 capture removed (the slot is never read)\n'),
}


def main():
    for m in sys.argv[1:]:
        f, old, new = MUTANTS[m]
        root = Path(f'/tmp/p1t_fix4_mut_{m}')
        for sub in ('tensor_cuda', 'tc_signal'):
            shutil.copytree(W / sub, root / sub, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns('build', '*.so', '__pycache__', '.pytest_cache'))
        p = root / f
        src = (W / f).read_text()
        assert src.count(old) == 1, (m, src.count(old))
        mut = src.replace(old, new)
        p.write_text(mut)
        diff = ''.join(difflib.unified_diff(src.splitlines(True), mut.splitlines(True), f'a/{f}', f'b/{f} (mutant {m})'))
        (G4 / f'mut_{m}.diff').write_text(diff)
        print(f'mutant {m}: {p} edited ({len(diff.splitlines())} diff lines)')


if __name__ == '__main__':
    main()
