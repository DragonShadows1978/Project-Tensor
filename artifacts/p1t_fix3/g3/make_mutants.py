"""P1T-FIX3 G3: build each registered mutant tree OUTSIDE the worktree (/tmp/p1t_fix3_mut_<m>/), one asserted edit each.

Each tree is a copy of the worktree's tensor_cuda/ and tc_signal/ (sources and tests; no build/ dir, no .so). The edit's
unified diff is written to artifacts/p1t_fix3/g3/mut_<m>.diff.
  (a) V3-01: FIX2's bounded return restored. The exit wait returns right after its first diagnostic, i.e. after
      _DLPACK_SHUTDOWN_WAIT_S (default 5 s), with admitted releases possibly still running: FIX2's `wait_for(bound)`
      semantics in the FIX3 loop.
  (b) V3-02: the aggregate pre-launch check removed. aggregate_operand_bytes no longer refuses: an overflowing sum is
      silently saturated (so the destructor's accounting stays defined) and the call goes on to its launch.
Usage: make_mutants.py <m>... (a b).
Prior art: DeMillo, Lipton & Sayward (1978) mutation testing; the registered mutants are the order's (a) and (b).
"""
import difflib
import shutil
import sys
from pathlib import Path

W = Path('/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix3')
G3 = W / 'artifacts/p1t_fix3/g3'

MUTANTS = {
    'a': ('tensor_cuda/src/bindings.cpp',
          '''      next = now + (waited > tick ? waited : tick);
      lk.lock();
    }
''',
          '''      next = now + (waited > tick ? waited : tick);
      lk.lock();
      break;  // MUTANT (a): V3-01, FIX2's bounded return restored (return after the first-diagnostic delay)
    }
'''),
    'b': ('tc_signal/src/dlpack_operand.cpp',
          '''    total = ck_add(total, b, "the aggregate fp32 byte count of this call's operands (their sum)");
''',
          '''    if (__builtin_add_overflow(total, b, &total)) total = std::numeric_limits<int64_t>::max();  // MUTANT (b): V3-02,
    // the aggregate pre-launch check removed (no refusal; the call goes on to its launch)
'''),
}


def main():
    for m in sys.argv[1:]:
        f, old, new = MUTANTS[m]
        root = Path(f'/tmp/p1t_fix3_mut_{m}')
        for sub in ('tensor_cuda', 'tc_signal'):
            shutil.copytree(W / sub, root / sub, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns('build', '*.so', '__pycache__', '.pytest_cache'))
        p = root / f
        src = (W / f).read_text()
        assert src.count(old) == 1, (m, src.count(old))
        mut = src.replace(old, new)
        p.write_text(mut)
        diff = ''.join(difflib.unified_diff(src.splitlines(True), mut.splitlines(True), f'a/{f}', f'b/{f} (mutant {m})'))
        (G3 / f'mut_{m}.diff').write_text(diff)
        print(f'mutant {m}: {p} edited ({len(diff.splitlines())} diff lines)')


if __name__ == '__main__':
    main()
