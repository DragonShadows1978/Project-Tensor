"""P1T-FIX6: path-only copies of the round-6 reviewer scripts that hard-code the reviewer's paths
(/tmp/p1t_rev6_build, the reviewer's worktree via parents[3], and outputs written into artifacts/p1t_rev6/ and
logs/p1t_rev6_*). The originals stay byte-identical. Every substitution is a literal asserted to occur exactly once;
probe logic, modes, bounds, environment and PASS rules are untouched; a unified diff of every copy is written next to it.

  * build_host_probes.sh: bin dir -> <bin>; the trace probe source -> its absolute path in this worktree (the before
    arm's tree, wt_fix5 at acd42fa, has no artifacts/p1t_rev6/); cwd stays the arm's root, so the trace module and the
    native harnesses are compiled from THAT arm's tc_signal/src and tests (regenerated per arm).
  * build_core_host.py: root -> <arm root> (the slices are taken from that arm's sources: regenerated per arm);
    out dir -> <bin>; the slice-hash receipt -> <out_dir>/source_slices.json (instead of artifacts/p1t_rev6/).
  * run_cpu_checks.py: root (cwd of every job, and the operand-census source) -> <arm root>; the probe directory ->
    this worktree's artifacts/p1t_rev6 (byte-identical probes; the before arm has none); bin -> <bin>; the per-case logs
    and cpu_results.json -> <out_dir>; the operand-census copy's result file -> <out_dir>/operand_census.json.
Usage: make_rev6_copies.py <arm_root> <bin> <out_dir>
No prior art claimed (plain text substitution; same method as FIX3-FIX5's make_probe_copies.py).
"""
import difflib
import sys
from pathlib import Path

W = Path('/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix6')
P = W / 'artifacts/p1t_rev6/probes'


def copy(name, subs, out_dir):
    src = (P / name).read_text()
    new = src
    for a, b in subs:
        assert new.count(a) == 1, (name, a, new.count(a))
        new = new.replace(a, b)
    out = Path(out_dir) / (Path(name).stem + '_paths' + Path(name).suffix)
    out.write_text(new)
    (Path(out_dir) / (out.stem + '.diff')).write_text(
        ''.join(difflib.unified_diff(src.splitlines(True), new.splitlines(True), str(P / name), str(out))))
    print(f'{name} -> {out} ({len(subs)} substitutions)')


def main():
    root, bin_dir, out_dir = sys.argv[1:4]
    Path(bin_dir).mkdir(parents=True, exist_ok=True)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    copy('build_host_probes.sh', [
        ('bin=/tmp/p1t_rev6_build', f'bin={bin_dir}'),
        ('artifacts/p1t_rev6/probes/release_trace.cpp', str(P / 'release_trace.cpp')),
    ], bin_dir)
    copy('build_core_host.py', [
        ('root=Path(__file__).resolve().parents[3]', f"root=Path('{root}')"),
        ("out=Path('/tmp/p1t_rev6_build')", f"out=Path('{bin_dir}')"),
        ("(root/'artifacts/p1t_rev6/source_slices.json')", f"(Path('{out_dir}')/'source_slices.json')"),
    ], bin_dir)
    copy('run_cpu_checks.py', [
        ('root=Path(__file__).resolve().parents[3]', f"root=Path('{root}')"),
        ("out=root/'artifacts/p1t_rev6'", f"out=Path('{W}/artifacts/p1t_rev6')"),
        ("bin=Path('/tmp/p1t_rev6_build')", f"bin=Path('{bin_dir}')"),
        ("(root/'logs'/('p1t_rev6_'+name+'.log'))", f"(Path('{out_dir}')/('rev6_'+name+'.log'))"),
        ("(out/'cpu_results.json')", f"(Path('{out_dir}')/'cpu_results.json')"),
        # the generated operand-census copy would write <arm>/artifacts/p1t_rev6/operand_census.json (a reviewer
        # artifact on the after arm; a missing directory on the before arm): its output goes to <out_dir> instead.
        (".replace('artifacts/p1t_rev4','artifacts/p1t_rev6')",
         f""".replace("(ROOT/'artifacts/p1t_rev4/operand_census.json')","(Path('{out_dir}/operand_census.json'))")"""),
    ], bin_dir)


if __name__ == '__main__':
    main()
