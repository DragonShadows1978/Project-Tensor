"""P1T-FIX6 (a copy of P1T-FIX5's make_probe_copies.py, retargeted: probe sources read from this worktree, which are
byte-identical to wt_fix5's; outputs under /tmp/p1t_fix6_probes/<arm>). P1T-FIX5: path-only copies of round-5 reviewer probes whose ORIGINALS cannot run on a chosen build without writing
outside this seat's area (/tmp/p1t_fix5_*) or testing the wrong tree. The originals stay byte-identical and are never
edited. Same method as FIX3/FIX4's make_probe_copies.py.

Each substitution is a literal asserted to occur exactly once; case bodies and PASS logic are untouched; the unified diff
of every copy is written next to it:
  * device_error_sweep_single_gpu.py runs pytest on root = parents[3] (the probe file's own tree) with
    --basetemp=/tmp/p1t_rev5_fix4_pytest (outside /tmp/p1t_fix5_*; pytest empties it). The copy sets root = the arm's
    tree (before: wt_fix4, its own FIX4 tests on its own prebuilt binaries; after: this tree) and the basetemp under
    <out_dir>.
  * release_failures.py runs pytest on parents[3]. BEFORE arm only: root = wt_fix4 (its own FIX1/FIX2 tests on its own
    build; FIX4 D9 precedent). The AFTER arm runs the original.
  * build_cpu.sh builds into /tmp/p1t_rev5_build (the reviewer's scratch). The copy builds into <out_dir>/rev5_build;
    every occurrence of that path is replaced (it is also the rewrite target of the copied round-4 drivers).
Usage: make_probe_copies.py <arm> <arm_root> <out_dir>
No prior art claimed (plain text substitution).
"""
import difflib
import sys
from pathlib import Path

W = '/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix6'
P = Path(W) / 'artifacts/p1t_rev5/probes'


def copy(name, subs, out_dir, counts=None):
    src = (P / name).read_text()
    new = src
    for i, (a, b) in enumerate(subs):
        want = (counts or {}).get(i, 1)
        assert new.count(a) == want, (name, a, new.count(a), want)
        new = new.replace(a, b)
    out = Path(out_dir) / (Path(name).stem + '_paths' + Path(name).suffix)
    out.write_text(new)
    diff = ''.join(difflib.unified_diff(src.splitlines(True), new.splitlines(True), str(P / name), str(out)))
    (Path(out_dir) / (out.stem + '.diff')).write_text(diff)
    print(f'{name} -> {out} ({len(subs)} substitution kinds)')


def main():
    arm, root, out_dir = sys.argv[1:4]
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    copy('device_error_sweep_single_gpu.py', [
        ("root=Path(__file__).resolve().parents[3]", f"root=Path({root!r})"),
        ("'--basetemp=/tmp/p1t_rev5_fix4_pytest'", repr('--basetemp=' + out_dir + '/rev5_fix4_pytest'))], out_dir)
    if arm == 'before':
        copy('release_failures.py', [("root=Path(__file__).resolve().parents[3]", f"root=Path({root!r})")], out_dir)
    src = (P / 'build_cpu.sh').read_text()
    copy('build_cpu.sh', [('/tmp/p1t_rev5_build', out_dir + '/rev5_build')], out_dir,
         counts={0: src.count('/tmp/p1t_rev5_build')})


if __name__ == '__main__':
    main()
