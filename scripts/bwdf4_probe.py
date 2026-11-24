"""BWD-F4 probe (L81): run ad-hoc rows on one build and print each result (no verdicts). Reuses BWD-F2's row runner
(tests/test_bwdf2_fixes.py run_rows: one child per row, ENG3's LD_PRELOAD counter). CPU only.
  python3 -B -P scripts/bwdf4_probe.py <pkg_parent>
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tensor_cuda", "tests"))
import test_bwdf2_fixes as T2  # noqa: E402

R9 = "[2,2,1,1,1,1,1,1,1]"
ROWS = [
    ("rev-case", "H1", "refuse", "a=C((0,)).reshape([2**32,2**32,0,0])", "a.mean([0,1,2])", [], {}, ""),
    ("rev-case-neg", "H1", "refuse", "a=C((0,)).reshape([2**32,2**32,0,0])", "a.mean([-4,-3,-2])", [], {}, ""),
    ("ovf-nonempty-axes", "H1", "refuse", "a=C((0,)).reshape([2**32,2**32,1,0])", "a.mean([0,1])", [], {}, ""),
    ("ovf-nonempty-axes-sum", "H1", "refuse", "a=C((0,)).reshape([2**32,2**32,1,0])", "a.sum([0,1])", [], {}, ""),
    ("ovf-3axes", "H1", "refuse", "a=C((0,)).reshape([2**21,2**21,2**21,0])", "a.mean([0,1,2])", [], {}, ""),
    ("fits-nonempty-axes", "H1", "refuse", "a=C((0,)).reshape([2**31,2**31,1,0])", "a.mean([0,1])", [], {}, ""),
    ("empty-axes-list", "H1", "refuse", "a=C((0,)).reshape([2**32,2**32,0,0])", "a.mean()", [], {}, ""),
    ("host-r1", "ctl", "refuse", "a=H((3,))", "a.mean([0])", [], {}, ""),
    ("host-r2", "ctl", "refuse", "a=H((2,3))", "a.mean([0,1])", [], {}, ""),
    ("cuda-empty-selected", "ctl", "refuse", "a=C((0,3))", "a.mean([0])", [], {}, ""),
    ("cuda-empty-all", "ctl", "refuse", "a=C((0,3))", "a.mean([0,1])", [], {}, ""),
    ("rank9-table-sin", "H2", "refuse", "x=C((0,2,2)); cs=H((2,2)); sn=H((2,2)).reshape(" + R9 + ")",
     "tc.rope_apply(x, cs, sn, 0)", [], {}, ""),
    ("rank9-table-cos", "H2", "refuse", "x=C((0,2,2)); cs=H((2,2)).reshape(" + R9 + "); sn=H((2,2))",
     "tc.rope_apply(x, cs, sn, 0)", [], {}, ""),
    ("rank8-table", "H2", "refuse", "x=C((0,2,2)); cs=H((2,2)); sn=H((2,2)).reshape([2,2,1,1,1,1,1,1])",
     "tc.rope_apply(x, cs, sn, 0)", [], {}, ""),
    ("export-rope-rank9", "H2", "refuse", "c=C((0,4,2)); cs=H((8,2)); sn=H((8,2)).reshape([8,2]+[1]*7)",
     "tc.export_rope_rows(c, cs, sn, 1, 0, 1, 0)", [], {"no_grad": 1}, ""),
    ("f03-export-rows-view", "H3", "refuse", "c=C((0,)).reshape([0,2**62,2]) if False else C((0,),'uint8').reshape([0,2**62,2])",
     "tc.export_rows(c, 1, 0, 2**62)", [], {"no_grad": 1}, ""),
]
res = T2.run_rows(sys.argv[1], "/tmp/bwdf4_probe", ROWS)
print(res["tensor_cuda_file"], "control", res["control_calls"])
for r in ROWS:
    print(r[0], json.dumps(res["results"].get(r[0])))
