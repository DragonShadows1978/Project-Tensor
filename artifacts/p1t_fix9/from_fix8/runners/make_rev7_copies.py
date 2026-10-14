"""P1T-FIX7: path-only copies of the round-7 reviewer scripts that hard-code the reviewer's paths
(/tmp/p1t_rev7_build, the reviewer's scratch, and outputs written into artifacts/p1t_rev7/ and logs/p1t_rev7_*). The
originals stay byte-identical. Every substitution is a literal asserted to occur exactly once; probe logic, modes, bounds,
environment and PASS rules are untouched; a unified diff of every copy is written next to it.

  * build_cpu.sh: build dir -> <bin> (cwd stays the arm's root, so the trace module and the native harnesses are
    compiled from THAT arm's tc_signal/src and tests: regenerated per arm).
  * build_core_host.py: root -> <arm root> (slices taken from that arm's sources: regenerated per arm); out dir -> <bin>;
    the slice-hash receipt -> <out_dir>/core_source_slices.json (not artifacts/p1t_rev7/).
  * run_cpu_checks.py: root (cwd of every job, and the round-6 probe paths) -> <arm root>; build dir -> <bin>; the
    per-case logs and cpu_results.json -> <out_dir>.
  * argument_census_cpu.py: root (its source and test reads) -> <arm root>; its JSON -> <out_dir>.
  * gpu_regressions.py: root (cwd, test paths, PYTHONPATH) -> <arm root>; its junit receipt -> <out_dir>.
Usage: make_rev7_copies.py <arm_root> <bin> <out_dir>
No prior art claimed (plain text substitution; FIX3-FIX6's make_probe_copies.py / make_rev6_copies.py method).
"""
import difflib
import sys
from pathlib import Path

W = Path('/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix8')
P = W / 'artifacts/p1t_rev7/probes'
ROOT_LINE = 'root=Path(__file__).resolve().parents[3]'


def copy(name, subs, bin_dir):
    src = (P / name).read_text()
    new = src
    for a, b in subs:
        assert new.count(a) == 1, (name, a, new.count(a))
        new = new.replace(a, b)
    out = Path(bin_dir) / (Path(name).stem + '_paths' + Path(name).suffix)
    out.write_text(new)
    (Path(bin_dir) / (out.stem + '.diff')).write_text(
        ''.join(difflib.unified_diff(src.splitlines(True), new.splitlines(True), str(P / name), str(out))))
    print(f'{name} -> {out} ({len(subs)} substitutions)')


def main():
    root, bin_dir, out_dir = sys.argv[1:4]
    Path(bin_dir).mkdir(parents=True, exist_ok=True)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    copy('build_cpu.sh', [('b=/tmp/p1t_rev7_build', f'b={bin_dir}')], bin_dir)
    copy('build_core_host.py', [
        (ROOT_LINE, f"root=Path('{root}')"),
        ("out=Path('/tmp/p1t_rev7_build')", f"out=Path('{bin_dir}')"),
        ("(root/'artifacts/p1t_rev7/core_source_slices.json')", f"(Path('{out_dir}')/'core_source_slices.json')"),
    ], bin_dir)
    copy('run_cpu_checks.py', [
        (ROOT_LINE, f"root=Path('{root}')"),
        ("b=Path('/tmp/p1t_rev7_build')", f"b=Path('{bin_dir}')"),
        ("out=root/'artifacts/p1t_rev7'", f"out=Path('{out_dir}')"),
        ("(root/'logs'/('p1t_rev7_'+name+'.log'))", f"(Path('{out_dir}')/('rev7_'+name+'.log'))"),
    ], bin_dir)
    copy('argument_census_cpu.py', [
        (ROOT_LINE, f"root=Path('{root}')"),
        ("(root/'artifacts/p1t_rev7/argument_census.json')", f"(Path('{out_dir}')/'argument_census.json')"),
    ], bin_dir)
    copy('gpu_regressions.py', [
        (ROOT_LINE, f"root=Path('{root}')"),
        ("out=root/'artifacts/p1t_rev7'", f"out=Path('{out_dir}')"),
    ], bin_dir)


if __name__ == '__main__':
    main()
