"""P1T-FIX3: path-only copies of the round-3 reviewer's probes whose ORIGINALS cannot run on a chosen build without
writing outside this seat's area (the originals stay byte-identical and are never edited).

Why each copy exists (each substitution is a literal asserted to occur exactly once; case lists, case bodies and
PASS logic are untouched; the unified diff of every copy is written next to it):
  * host_checks.py hard-codes ROOT = the reviewer's worktree (wt_p1trev3) and writes host_checks.json THERE, and runs
    the reviewer's scratch builds /tmp/p1t_rev3_build/{tc_signal,core_cpu,cpu_native} (5f9f955 code). The copy points
    ROOT at the arm's tree (for tc_signal/tests/tcs_producer.py), OUT and the two native helpers at
    /tmp/p1t_fix3_probes/<arm>/, and PYTHONPATH at the arm's built tc_signal package.
  * operand_census.py and real_metadata_plans.py write their JSON into artifacts/p1t_rev3/ (committed receipts). The
    copies write to /tmp/p1t_fix3_probes/<arm>/ instead; ROOT stays this worktree (read-only: tcs_producer.py and the
    reviewer's real_data_census.json, which only this tree has).
  * release_failures.py runs pytest on parents[3] = the probe file's own tree and PREPENDS that tree's tc_signal and
    tensor_cuda to PYTHONPATH, so run from this worktree it always tests this worktree's build. For the BEFORE arm the
    copy sets root = the 5f9f955 tree (wt_fix2: its own FIX1/FIX2 tests on its own prebuilt binaries, read-only; pytest
    with -p no:cacheprovider and PYTHONDONTWRITEBYTECODE writes nothing there). The AFTER arm runs the original.
Usage: make_probe_copies.py <arm> <arm_root> <out_dir>
No prior art claimed (plain text substitution; same method as FIX2's make_host_checks_copy.py).
"""
import difflib
import sys
from pathlib import Path

W = '/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix3'
P = Path(W) / 'artifacts/p1t_rev3/probes'


def copy(name, subs, out_dir, out_name=None):
    src = (P / name).read_text()
    new = src
    for a, b in subs:
        assert new.count(a) == 1, (name, a, new.count(a))
        new = new.replace(a, b)
    out = Path(out_dir) / (out_name or name.replace('.py', '_paths.py'))
    out.write_text(new)
    diff = ''.join(difflib.unified_diff(src.splitlines(True), new.splitlines(True), str(P / name), str(out)))
    (Path(out_dir) / (out.stem + '.diff')).write_text(diff)
    print(f'{name} -> {out} ({len(subs)} substitutions)')


def main():
    arm, root, out_dir = sys.argv[1:4]
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    copy('host_checks.py', [
        ("ROOT=Path('/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_p1trev3')",
         f"ROOT=Path({root!r})"),
        ("OUT=ROOT/'artifacts/p1t_rev3'", f"OUT=Path({out_dir!r})"),
        ("env['PYTHONPATH']='/tmp/p1t_rev3_build/tc_signal'", f"env['PYTHONPATH']={root + '/tc_signal'!r}"),
        ("'/tmp/p1t_rev3_build/core_cpu'", repr(out_dir + '/core_cpu')),
        ("'/tmp/p1t_rev3_build/cpu_native'", repr(out_dir + '/cpu_native'))], out_dir)
    copy('operand_census.py', [
        ("ROOT=Path(__file__).resolve().parents[3]", f"ROOT=Path({W!r})"),
        ("(ROOT/'artifacts/p1t_rev3/operand_census.json')", f"(Path({out_dir!r})/'operand_census.json')")], out_dir)
    copy('real_metadata_plans.py', [
        ("ROOT=Path(__file__).resolve().parents[3]", f"ROOT=Path({W!r})"),
        ("(ROOT/'artifacts/p1t_rev3/real_metadata_plans.json')", f"(Path({out_dir!r})/'real_metadata_plans.json')")],
        out_dir)
    if arm == 'before':
        copy('release_failures.py', [
            ("root=Path(__file__).resolve().parents[3]", f"root=Path({root!r})")], out_dir)


if __name__ == '__main__':
    main()
