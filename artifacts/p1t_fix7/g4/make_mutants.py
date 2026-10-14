"""P1T-FIX7 G4: the three registered mutants, each a copy of this worktree's product tree under /tmp/p1t_fix7_mut_<m>/
with ONE asserted edit (diff written to artifacts/p1t_fix7/g4/mut_<m>.diff). Copied: tensor_cuda/{src,include,
tensor_cuda,tests,CMakeLists.txt,build.sh}, tc_signal/{src,include,tc_signal,tests,CMakeLists.txt,build.sh} (the built
.so files included) and every artifacts/p1t_rev*/probes directory (byte-identical, so the round-7 probes, which find
their package through parents[3], test the mutant). Mutant (c) changes C++: run_mutants.sh rebuilds that tree's
tc_signal; (a) and (b) change Python only (the copied, unchanged .so files are the mutant's build).

  (a) the add-on hook's knob fallback narrowed back to (TypeError, ValueError), the close skipped on any other
      exception: tc_signal/__init__.py `_release_pending_at_exit` body replaced by FIX6's (429446d) body, verbatim
      (float(knob) with `except (TypeError, ValueError)`, the close inside the same outer try, `except Exception`
      logs and returns). The native runner registration is kept (it acts only when the hook raises).
  (b) the same in the core hook: tensor_cuda/__init__.py `_dlpack_shutdown_hook` body replaced by FIX6's, verbatim.
  (c) the failure-kind tag dropped: dlpack_operand.cpp SyncReport::add stores every record as a stream-sync failure
      (FailedCall::kStreamSync / Purpose::kSyncStream), so a query failure (and a guard failure) is recorded as a
      stream failure, as in 429446d.
  (d) (lead send-back L1) the diagnostic removed from the core's lock-retry loop: tensor_cuda/src/bindings.cpp
      lock_retrying's `if (now >= next) {` (the line, its counters and the doubling) disabled, so a lifecycle lock
      that keeps failing at the exit close is retried silently, as in this round's first i19 fix. Changes C++: the
      mutant's core is rebuilt (run_mutant.sh).
Usage: make_mutants.py <a|b|c|d> [dst] [tag]. dst defaults to /tmp/p1t_fix7_mut_<m>; the diff goes to
g4/mut_<m><tag>.diff (L1: the refreshed trees /tmp/p1t_fix7_mutL1_<m>, tag _L1). Refuses an existing target (no
deletes). No prior art claimed beyond DeMillo, Lipton & Sayward (1978) mutation testing.
"""
import difflib
import shutil
import sys
from pathlib import Path

W = Path('/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix7')
W6 = Path('/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix6')
G4 = W / 'artifacts/p1t_fix7/g4'


def body(src, name):
    """The text of top-level function `name`, from its def line to the line before the next top-level statement."""
    lines = src.splitlines(True)
    start = next(i for i, ln in enumerate(lines) if ln.startswith(f'def {name}('))
    end = next(j for j in range(start + 1, len(lines)) if lines[j].strip() and not lines[j][0].isspace())
    while not lines[end - 1].strip():
        end -= 1
    return ''.join(lines[start:end])


def copy_tree(dst):
    if dst.exists():
        raise SystemExit(f'refusing: {dst} exists (no deletes; use a fresh path)')
    for pkg, items in (('tensor_cuda', ['src', 'include', 'tensor_cuda', 'tests', 'CMakeLists.txt', 'build.sh']),
                       ('tc_signal', ['src', 'include', 'tc_signal', 'tests', 'CMakeLists.txt', 'build.sh'])):
        for it in items:
            s, d = W / pkg / it, dst / pkg / it
            d.parent.mkdir(parents=True, exist_ok=True)
            (shutil.copytree if s.is_dir() else shutil.copy2)(s, d)
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
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(f'/tmp/p1t_fix7_mut_{m}')
    tag = sys.argv[3] if len(sys.argv) > 3 else ''
    copy_tree(dst)
    if m == 'a':
        rel = 'tc_signal/tc_signal/__init__.py'
        d = edit(dst, rel, body((W / rel).read_text(), '_release_pending_at_exit'),
                 body((W6 / rel).read_text(), '_release_pending_at_exit'))
    elif m == 'b':
        rel = 'tensor_cuda/tensor_cuda/__init__.py'
        d = edit(dst, rel, body((W / rel).read_text(), '_dlpack_shutdown_hook'),
                 body((W6 / rel).read_text(), '_dlpack_shutdown_hook'))
    elif m == 'c':
        d = edit(dst, 'tc_signal/src/dlpack_operand.cpp', '''    if (n < tuning::kSyncReportMax)
      rec[n++] = r;''', '''    if (n < tuning::kSyncReportMax)
      rec[n++] = FailureRecord{FailedCall::kStreamSync, Purpose::kSyncStream, r.device, r.err};  // G4 MUTANT (c)''')
    elif m == 'd':
        d = edit(dst, 'tensor_cuda/src/bindings.cpp', '''      if (now >= next) {
        std::fprintf(stderr,
                     "tensor_cuda: interpreter exit cannot take''', '''      if (false && now >= next) {  // G4 MUTANT (d): the lock-retry diagnostic removed
        std::fprintf(stderr,
                     "tensor_cuda: interpreter exit cannot take''')
    else:
        raise SystemExit('mutant a, b, c or d')
    (G4 / f'mut_{m}{tag}.diff').write_text(d)
    print(f'mutant {m}: {dst}; diff {G4}/mut_{m}{tag}.diff ({len(d.splitlines())} lines)')


if __name__ == '__main__':
    main()
