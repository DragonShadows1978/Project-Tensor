"""P1T-FIX8: path-only copies of the round-8 reviewer scripts that hard-code the reviewer's paths
(/tmp/p1t_rev8_build, outputs written into artifacts/p1t_rev8/ and logs/p1t_rev8_*) or find the arm through
Path(__file__).parents[3] / Path.cwd(). The originals stay byte-identical (G5). Every substitution is a literal that
must occur in the file (asserted); probe logic, modes, bounds, environment and PASS rules are untouched; a unified diff
of every copy is written next to it.

  * Path(__file__).resolve().parents[3] and Path.cwd()  -> Path('<arm root>')   (the copy lives elsewhere)
  * /tmp/p1t_rev8_build                                  -> <bin>               (the arm's builds)
  * root/'artifacts/p1t_rev8' (and root/'artifacts/p1t_rev8/<file>')  -> Path('<out_dir>') (/'<file>')
  * root/'logs'/('p1t_rev8_'                             -> Path('<out_dir>')/('p1t_rev8_'
  * build_cpu.sh: b=/tmp/p1t_rev8_build                  -> b=<bin>
Usage: make_rev8_copies.py <arm_root> <bin> <out_dir>
No prior art claimed (plain text substitution; FIX3-FIX7's make_rev*_copies.py method).
"""
import difflib
import sys
from pathlib import Path

W = Path('/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix8')
P = W / 'artifacts/p1t_rev8/probes'
root, b, out = sys.argv[1], sys.argv[2], sys.argv[3]
Path(b).mkdir(parents=True, exist_ok=True)
Path(out).mkdir(parents=True, exist_ok=True)
FILES = ['build_cpu.sh', 'build_core_host.py', 'build_core_host_oom.py', 'run_cpu_checks.py', 'argument_census_cpu.py',
         'attribution_cpu.py', 'fix7_cpu_census.py', 'gpu_regressions.py', 'lock_checks_cpu.py', 'operand_census_cpu.py']
SUBS = [
    ("Path(__file__).resolve().parents[3]", f"Path({root!r})"),
    ("Path.cwd()", f"Path({root!r})"),
    ("root/'artifacts/p1t_rev8/", f"Path({out!r})/'"),
    ("root/'artifacts/p1t_rev8'", f"Path({out!r})"),
    ("root/'logs'/('p1t_rev8_'", f"Path({out!r})/('p1t_rev8_'"),
    ("/tmp/p1t_rev8_build", b),
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
