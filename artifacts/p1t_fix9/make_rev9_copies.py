"""P1T-FIX9: path-only copies of the round-9 reviewer scripts that hard-code the reviewer's scratch
(/tmp/p1t_rev9_build), write into artifacts/p1t_rev9/ or logs/p1t_rev9_*, or find the arm through
Path(__file__).parents[3] / Path.cwd(). The originals stay byte-identical (G5). Every substitution is a literal that
must occur in the file (asserted); probe logic, modes, bounds, environment and PASS rules are untouched; a unified diff
of every copy is written next to it (FIX8's make_rev8_copies.py method, retargeted).

  * Path(__file__).resolve().parents[3] and Path.cwd()  -> Path('<arm root>')   (the copy lives elsewhere)
  * /tmp/p1t_rev9_build                                  -> <bin>               (the arm's builds)
  * root/'artifacts/p1t_rev9/<file>' and root/'artifacts/p1t_rev9' -> Path('<out_dir>')/'<file>' / Path('<out_dir>')
  * root/'logs'/('p1t_rev9_'                             -> Path('<out_dir>')/('p1t_rev9_'
Usage: make_rev9_copies.py <arm_root> <bin> <out_dir>
No prior art claimed (plain text substitution).
"""
import difflib
import sys
from pathlib import Path

W = Path('/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix9')
root, b, out = sys.argv[1], sys.argv[2], sys.argv[3]
P = Path(root) / 'artifacts/p1t_rev9/probes'
Path(b).mkdir(parents=True, exist_ok=True)
Path(out).mkdir(parents=True, exist_ok=True)
FILES = ['build_cpu.sh', 'build_core_host.py', 'build_core_host_oom.py', 'build_core_fence_cpu.py',
         'build_core_race_cpu.py', 'core_fence_cpu.py', 'core_snapshot_cpu.py', 'acknowledgement_cpu.py',
         'watchdog_cpu.py', 'compound_cleanup_cpu.py', 'run_cpu_checks.py', 'argument_census_cpu.py',
         'attribution_cpu.py', 'fix7_cpu_census.py', 'gpu_regressions.py', 'lock_checks_cpu.py',
         'operand_census_cpu.py']
SUBS = [
    ("Path(__file__).resolve().parents[3]", f"Path({root!r})"),
    ("Path.cwd()", f"Path({root!r})"),
    ("root/'artifacts/p1t_rev9/", f"Path({out!r})/'"),
    ("root/'artifacts/p1t_rev9'", f"Path({out!r})"),
    ("root/'logs'/('p1t_rev9_'", f"Path({out!r})/('p1t_rev9_'"),
    ("/tmp/p1t_rev9_build", b),
]
for name in FILES:
    src = (P / name).read_text()
    new = src
    used = []
    for a, c in SUBS:
        if a in new:
            new = new.replace(a, c)
            used.append(a)
    assert used, name
    stem, dot, ext = name.rpartition('.')
    dst = Path(b) / f'{stem}_paths.{ext}'
    dst.write_text(new)
    (Path(out) / f'{stem}_paths.diff').write_text(''.join(difflib.unified_diff(
        src.splitlines(True), new.splitlines(True), f'a/{name}', f'b/{dst.name}')))
    print(f'{name} -> {dst} (substituted: {used})')
