"""P1T-FIX2: a path-only copy of the round-2 reviewer's host_checks.py (artifacts/p1t_rev2/probes/host_checks.py, which
stays byte-identical and is never edited) so it can run on a chosen build without writing outside /tmp/p1t_fix2_*.

The original hard-codes the reviewer's scratch paths (/tmp/p1t_rev2_build_review/tc_signal, /tmp/p1t_rev2_core_cpu,
/tmp/p1t_rev2_cpu_native) and writes its JSON into artifacts/p1t_rev2/ (a committed receipt). Exactly five literal
substitutions are made, each asserted to occur exactly once; the case list, case bodies and PASS logic are untouched.
The unified diff is printed so the change is auditable.
Usage: make_host_checks_copy.py <original> <tree_root_for_tcs_producer> <tc_signal_pkg_dir> <probe_dir> <out.py>
No prior art claimed (plain text substitution).
"""
import difflib
import sys
from pathlib import Path

orig, tree, pkg, pdir, out = sys.argv[1:6]
src = Path(orig).read_text()
subs = [("ROOT=Path(__file__).resolve().parents[3]", f"ROOT=Path({tree!r})"),
        ("OUT=ROOT/'artifacts/p1t_rev2'", f"OUT=Path({pdir!r})"),
        ("env['PYTHONPATH']='/tmp/p1t_rev2_build_review/tc_signal'", f"env['PYTHONPATH']={pkg!r}"),
        ("'/tmp/p1t_rev2_core_cpu'", repr(pdir + '/core_cpu')),
        ("'/tmp/p1t_rev2_cpu_native'", repr(pdir + '/cpu_native'))]
new = src
for a, b in subs:
    assert new.count(a) == 1, (a, new.count(a))
    new = new.replace(a, b)
Path(out).write_text(new)
sys.stdout.writelines(difflib.unified_diff(src.splitlines(True), new.splitlines(True), orig, out))
