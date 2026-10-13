"""P1T-FIX6 G4: the three registered mutants, each a copy of this worktree's tensor_cuda/ and tc_signal/ trees under
/tmp/p1t_fix6_mut_<m>/ (build directories, .so files and caches excluded; the copies are rebuilt) with ONE asserted
edit; the unified diff of each edit is written to artifacts/p1t_fix6/g4/mut_<m>.diff.
  (a) V6-01: FIX5's wait for other threads restored in ordinary synchronize() (before its drain, skipped inside the
      caller's own admitted section, exactly where FIX5 had it; it now counts both section kinds).
  (b) V6-02: tensor_dlpack's C-level argument reading replaced by acd42fa's pybind11 casts (copy.cast<bool>(),
      cast<py::sequence>() + py::len() + item cast<int>(), stream.cast<long long>()).
  (c) V6-03: the persistence of synchronize()'s own drain failures moved AFTER the first message allocation point (the
      test hook), so a message-allocation failure loses them again (acd42fa's behavior).
Usage: make_mutants.py <a|b|c>. No prior art claimed beyond DeMillo, Lipton & Sayward (1978) mutation testing.
"""
import difflib
import shutil
import sys
from pathlib import Path

W = Path('/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix6')
OUT = W / 'artifacts/p1t_fix6/g4'

EDITS = {
    'a': ('tc_signal/src/dlpack_operand.cpp', '''  // P1T-FIX6 (V6-01): no wait for other threads here (FIX5's wait deadlocked against caller-held locks; see the release
  // lifecycle block). This call releases what is on the pending list, on this thread.
  PendingState& st = pending();''', '''  // MUTANT (a): FIX5's wait for releases running on other threads restored.
  if (t_depth[kRelease] == 0 && t_depth[kCallback] == 0) (void)wait_for_other_threads(5.0);
  PendingState& st = pending();'''),
    'b': ('tensor_cuda/src/bindings.cpp', None, None),   # built below (a block replacement)
    'c': ('tc_signal/src/dlpack_operand.cpp', None, None),
}


def mutate(m):
    dst = Path(f'/tmp/p1t_fix6_mut_{m}')
    assert not dst.exists(), f'{dst} exists: use a fresh mutant directory'
    ign = shutil.ignore_patterns('build', '*.so', '__pycache__', '.pytest_cache')
    shutil.copytree(W / 'tensor_cuda', dst / 'tensor_cuda', ignore=ign)
    shutil.copytree(W / 'tc_signal', dst / 'tc_signal', ignore=ign)
    rel = EDITS[m][0]
    src = (W / rel).read_text()
    if m == 'a':
        a, b = EDITS[m][1], EDITS[m][2]
        assert src.count(a) == 1
        new = src.replace(a, b)
    elif m == 'b':
        start = src.index('  // P1T-FIX6 (V6-02): read every argument first, with C-level calls only (see above).')
        end = src.index('  if (!t.defined() || !t.data().defined())', start)
        old_block = '''  // MUTANT (b): acd42fa's pybind11 casts (they call the argument's own Python methods).
  bool has_stream = !stream.is_none();
  long long stream_value = 0;
  if (has_stream) {
    if (!py::isinstance<py::int_>(stream)) throw py::type_error("__dlpack__: stream must be an int or None");
    stream_value = stream.cast<long long>();
  }
  bool versioned = false;
  if (!max_version.is_none()) {
    py::sequence mv = max_version.cast<py::sequence>();
    if (py::len(mv) < 1) throw py::value_error("__dlpack__: max_version must be a (major, minor) pair");
    versioned = mv[0].cast<int>() >= 1;
  }
  bool has_dl_device = false;
  long long dl_type = 0, dl_id = 0;
  if (!dl_device.is_none()) {
    py::sequence d = dl_device.cast<py::sequence>();
    if (py::len(d) != 2) throw py::value_error("__dlpack__: dl_device must be a (device_type, device_id) pair");
    dl_type = d[0].cast<int>();
    dl_id = d[1].cast<int>();
    has_dl_device = true;
  }
  const bool want_copy = !copy.is_none() && copy.cast<bool>();
'''
        new = src[:start] + old_block + src[end:]
    else:
        a = '''  if (rep.failed()) {  // V6-03: persisted before any allocation
    st.drain_unreported.merge(rep);
    ++st.drain_unreported_calls;
  }
  if (st.drain_unreported_calls == 0 && st.unreported_count == 0 && st.unraisable_unreported == 0 &&
      st.prior_unreported == 0)
    return;'''
        b = '''  // MUTANT (c): no persistence before the first message allocation.
  if (!rep.failed() && st.drain_unreported_calls == 0 && st.unreported_count == 0 && st.unraisable_unreported == 0 &&
      st.prior_unreported == 0)
    return;'''
        assert src.count(a) == 1
        new = src.replace(a, b)
        a2 = '''  if (test_hooks::take(test_hooks::kAllocErrorMessage)) throw std::bad_alloc();
  const SyncReport& d = st.drain_unreported;'''
        b2 = '''  if (test_hooks::take(test_hooks::kAllocErrorMessage)) throw std::bad_alloc();
  if (rep.failed()) {  // MUTANT (c): persisted only after the first allocation point
    st.drain_unreported.merge(rep);
    ++st.drain_unreported_calls;
  }
  const SyncReport& d = st.drain_unreported;'''
        assert new.count(a2) == 1
        new = new.replace(a2, b2)
    (dst / rel).write_text(new)
    (OUT / f'mut_{m}.diff').write_text(''.join(difflib.unified_diff(
        src.splitlines(True), new.splitlines(True), f'a/{rel}', f'b/{rel} (mutant {m})')))
    print(f'mutant {m}: {dst} ({rel})')


if __name__ == '__main__':
    mutate(sys.argv[1])
