"""P1T-FIX8 G4: the three registered mutants, each a copy of this worktree's product tree under /tmp/p1t_fix8_mut_<m>/
with ONE asserted edit (diff written to artifacts/p1t_fix8/g4/mut_<m>.diff). Copied: tensor_cuda/{src,include,
tensor_cuda,tests,CMakeLists.txt,build.sh}, tc_signal/{src,include,tc_signal,tests,CMakeLists.txt,build.sh} (the built
.so files included) and every artifacts/p1t_rev*/probes directory (byte-identical, so the round-8 probes, which find
their package through parents[3], test the mutant). Each mutant changes C++: run_mutant.sh rebuilds that tree's
tc_signal (e, f) or core (g).

  (e) the record of a failed marker call removed: dlpack_operand.cpp note_exit_marker_failure no longer calls
      note_unraisable_failure (the failure is counted, then consumed silently, as in 6991939's L1 code).
  (f) the add-on exit logger's allocation-free fallback removed: exit_log_unreported returns after a failed full report
      (nothing written, records kept: 6991939's behaviour).
  (g) the core runner's allocation-free fallback removed: core_exit_last_reader no longer writes the summary when the
      full message cannot be built (nothing written for it).
Usage: make_mutants.py <e|f|g>. Refuses an existing target (no deletes). No prior art claimed beyond DeMillo, Lipton &
Sayward (1978) mutation testing.
"""
import difflib
import shutil
import sys
from pathlib import Path

W = Path('/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix8')
G4 = W / 'artifacts/p1t_fix8/g4'


def copy_tree(dst):
    if dst.exists():
        raise SystemExit(f'refusing: {dst} exists (no deletes; use a fresh path)')
    for pkg in ('tensor_cuda', 'tc_signal'):
        for it in ['src', 'include', pkg, 'tests', 'CMakeLists.txt', 'build.sh']:
            s, d = W / pkg / it, dst / pkg / it
            d.parent.mkdir(parents=True, exist_ok=True)
            (shutil.copytree if s.is_dir() else shutil.copy2)(s, d, **({'ignore': shutil.ignore_patterns('__pycache__')} if s.is_dir() else {}))
    for r in sorted((W / 'artifacts').glob('p1t_rev*/probes')):
        shutil.copytree(r, dst / 'artifacts' / r.parent.name / 'probes')


def edit(dst, rel, old, new):
    p = dst / rel
    src = p.read_text()
    assert src.count(old) == 1, (rel, src.count(old))
    out = src.replace(old, new, 1)
    p.write_text(out)
    return ''.join(difflib.unified_diff(src.splitlines(True), out.splitlines(True), f'a/{rel}', f'b/{rel}'))


def main():
    m = sys.argv[1]
    dst = Path(f'/tmp/p1t_fix8_mut_{m}')
    copy_tree(dst)
    if m == 'e':
        d = edit(dst, 'tc_signal/src/dlpack_operand.cpp',
                 '''  sat_inc(st.exit_marker_failures);
  note_unraisable_failure(st, {call, Purpose::kExitWaitMarker, device, err});''',
                 '''  sat_inc(st.exit_marker_failures);
  (void)call; (void)device; (void)err;  // G4 MUTANT (e): the record removed''')
    elif m == 'f':
        d = edit(dst, 'tc_signal/src/dlpack_operand.cpp',
                 '''    // bad_alloc building the report: the allocation-free fallback below (V8-02).
  }''',
                 '''    // bad_alloc building the report: the allocation-free fallback below (V8-02).
  }
  return;  // G4 MUTANT (f): the allocation-free fallback removed''')
    elif m == 'g':
        d = edit(dst, 'tensor_cuda/src/bindings.cpp',
                 '''    if (!written)
      write_core_summary(''',
                 '''    if (false && !written)  // G4 MUTANT (g): the allocation-free fallback removed
      write_core_summary(''')
    else:
        raise SystemExit('mutant e, f or g')
    (G4 / f'mut_{m}.diff').write_text(d)
    print(f'mutant {m}: {dst}; diff {G4}/mut_{m}.diff ({len(d.splitlines())} lines)')


if __name__ == '__main__':
    main()
