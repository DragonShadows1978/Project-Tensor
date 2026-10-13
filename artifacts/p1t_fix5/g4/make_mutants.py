"""P1T-FIX5 G4: build the four registered mutant trees outside the worktree, /tmp/p1t_fix5_mut_<m>/, each a copy of this
worktree's tc_signal/ (and tensor_cuda/ for (d)) and artifacts/p1t_rev5/probes/, with ONE asserted edit; the unified
diff of each edit is written to artifacts/p1t_fix5/g4/mut_<m>.diff. Test files and shims are copied unchanged (cmp
checked by run_mutant.sh).
  (a) V5-01 admission wait removed: wait_for_other_threads returns at once (no exit wait, no synchronize() wait).
  (b) V5-02 record removed: drain_completed consumes a non-NotReady query failure without recording it (1475fae).
  (c) V5-03 ownership removed: the NewHandleOwner destructor of BOTH caches never destroys (1475fae's insertion leak).
  (d) V5-04 refusal removed: fill_grid_blocks no longer refuses an extent beyond the grid limit.
Usage: make_mutants.py
Prior art: DeMillo, Lipton & Sayward (1978) mutation testing; registered single-site edits as in FIX2-FIX4's g4.
"""
import difflib
import shutil
from pathlib import Path

W = Path('/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix5')
OUT = W / 'artifacts/p1t_fix5/g4'
IGN = shutil.ignore_patterns('build', '*.so', '__pycache__', '.pytest_cache')

EDITS = {
    'a': [('tc_signal/src/dlpack_operand.cpp',
           '  if (w.running_at_start <= 0) return w;  // nothing to wait for: the GIL is not touched\n',
           '  if (true) return w;  // MUTANT (a): the V5-01 admission wait removed\n')],
    'b': [('tc_signal/src/dlpack_operand.cpp',
           '        ++st.event_query_failures;\n'
           '        note_unraisable_failure(st, "cudaEventQuery of a pending entry\'s completion event in the next-call drain (the "\n'
           '                                    "entry stayed pending; synchronize() re-queried it)",\n'
           '                                st.q.front().device, r);\n'
           '        cudaGetLastError();\n',
           '        cudaGetLastError();  // MUTANT (b): the V5-02 record removed\n')],
    'c': [(f, '      if (h == nullptr) return;  // committed to the cache, or already destroyed by settle\n',
           '      return;  // MUTANT (c): the V5-03 ownership removed (never destroys)\n')
          for f in ('tc_signal/src/conv.cu', 'tc_signal/src/lstm.cu')],
    'd': [('tensor_cuda/src/kernels.cu',
           '  if (blocks > kFillMaxGridX)\n    throw std::invalid_argument(\n',
           '  if (false)  // MUTANT (d): the V5-04 refusal removed\n    throw std::invalid_argument(\n')],
}


def main():
    for m, edits in EDITS.items():
        dst = Path(f'/tmp/p1t_fix5_mut_{m}')
        dst.mkdir(parents=True, exist_ok=True)
        dirs = ['tc_signal'] + (['tensor_cuda'] if m == 'd' else [])
        for d in dirs:
            shutil.copytree(W / d, dst / d, ignore=IGN, dirs_exist_ok=True)
        shutil.copytree(W / 'artifacts/p1t_rev5/probes', dst / 'artifacts/p1t_rev5/probes', dirs_exist_ok=True)
        diff = ''
        for rel, old, new in edits:
            p = dst / rel
            src = (W / rel).read_text()
            assert src.count(old) == 1, (m, rel)
            out = src.replace(old, new)
            p.write_text(out)
            diff += ''.join(difflib.unified_diff(src.splitlines(True), out.splitlines(True), f'a/{rel}', f'b/{rel}'))
        (OUT / f'mut_{m}.diff').write_text(diff)
        print(f'mutant ({m}) -> {dst}: {len(edits)} edit(s)')


if __name__ == '__main__':
    main()
