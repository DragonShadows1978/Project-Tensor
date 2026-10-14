"""P1T-FIX2 G3: build each registered mutant tree OUTSIDE the worktree (/tmp/p1t_fix2_mut_<m>/), one asserted edit each.

Each tree is a copy of the worktree's tensor_cuda/ and tc_signal/ (sources and tests; no build/ dir, no .so). The
edit's unified diff is written to artifacts/p1t_fix2/g3/mut_<m>.diff. Usage: make_mutants.py <m>... (a b c d1 d2 e).
Prior art: DeMillo, Lipton & Sayward (1978) mutation testing; the registered mutants are the order's (a)-(e).
"""
import difflib
import shutil
import sys
from pathlib import Path

W = Path('/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix2')
G3 = W / 'artifacts/p1t_fix2/g3'

MUTANTS = {
    'a': ('tc_signal/src/dlpack_operand.cpp',
          '    rep.query_failed = true;  // N1: was silently skipped (synchronize() then "succeeded" with no device)\n',
          '    rep.query_failed = false;  // MUTANT (a): N1, the failed current-device query swallowed again\n'),
    'b': ('tensor_cuda/src/bindings.cpp',
          'int64_t checked_extent_bound(const int64_t* dims, int n, const char* who) {\n  int64_t p = 1;\n',
          'int64_t checked_extent_bound(const int64_t* dims, int n, const char* who) {\n'
          '  for (int i = 0; i < n; ++i)\n    if (dims[i] == 0) return 0;  // MUTANT (b): N2, zero extents exempt the others again\n'
          '  int64_t p = 1;\n'),
    'c': ('tc_signal/src/dlpack_operand.cpp',
          '''    try {
      if (test_hooks::take(test_hooks::kAllocPendingPush)) throw std::bad_alloc();
      st.q.push_back(std::move(e));
      note_push(st, st.q.back());
    } catch (...) {
      ++st.push_alloc_failures;
      r = cudaErrorMemoryAllocation;  // e was not moved (push_back's strong guarantee): use the fallback below
      e.event = nullptr;
    }
''',
          '''    // MUTANT (c): N3, the catch around the pending-list push removed (a bad_alloc leaves the noexcept destructor)
    if (test_hooks::take(test_hooks::kAllocPendingPush)) throw std::bad_alloc();
    st.q.push_back(std::move(e));
    note_push(st, st.q.back());
'''),
    'd1': ('tc_signal/src/dlpack_operand.cpp',
           '''  try {
    m->deleter(m);
  } catch (...) {
    g_deleter_exceptions.fetch_add(1, std::memory_order_relaxed);
  }
''',
           '''  m->deleter(m);  // MUTANT (d1): N4, tc_signal's catch around the producer deleter removed
'''),
    'd2': ('tensor_cuda/src/bindings.cpp',
           '''  try {
    m->deleter(m);
    g_import_released.fetch_add(1);
  } catch (...) {
    g_import_deleter_exceptions.fetch_add(1);        // N4: contained (GIL still held while the exception dies)
  }
''',
           '''  m->deleter(m);  // MUTANT (d2): N4, core's catch around the producer deleter removed
  g_import_released.fetch_add(1);
'''),
    'e': ('tensor_cuda/src/bindings.cpp',
          '    if (at_close > 0) timed_out = !lc.cv.wait_for(lk, bound, [&lc] { return lc.in_flight == 0; });\n',
          '    (void)bound;  // MUTANT (e): N5, the in-flight wait removed (admission still closes)\n'),
}


def copy_tree(dst):
    ign = shutil.ignore_patterns('build', '*.so', '__pycache__', '.pytest_cache')
    for sub in ('tensor_cuda', 'tc_signal'):
        if (dst / sub).exists():
            raise SystemExit(f'{dst / sub} exists: refusing to overwrite (no deletes); use a fresh name')
        shutil.copytree(W / sub, dst / sub, ignore=ign)


def main():
    for m in sys.argv[1:]:
        rel, old, new = MUTANTS[m]
        dst = Path(f'/tmp/p1t_fix2_mut_{m}')
        dst.mkdir(parents=True, exist_ok=True)
        copy_tree(dst)
        f = dst / rel
        s = f.read_text()
        assert s.count(old) == 1, (m, 'edit anchor not unique/present')
        t = s.replace(old, new)
        f.write_text(t)
        d = ''.join(difflib.unified_diff(s.splitlines(True), t.splitlines(True), f'worktree/{rel}', f'mut_{m}/{rel}'))
        (G3 / f'mut_{m}.diff').write_text(d)
        print(f'mutant {m}: {dst} ({rel}), diff {len(d.splitlines())} lines')


if __name__ == '__main__':
    main()
