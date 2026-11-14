#!/usr/bin/env python3
"""ENG2-A3 R2 control (L16, no vacuous census): constructed sources, one per argument class and resolution route, fed to
consumer_census.census_file; each expected (entry, class, newly_refused) must come out exactly, and the count of sites
must match (a census that calls everything "supported" fails here). Writes a3/CONSUMER_CENSUS_SELFTEST.json.
Prior art: unit tests of a static analyser on constructed inputs; no other prior art known to me."""
import json
from pathlib import Path
import sys

A3 = Path(__file__).resolve().parent
sys.path.insert(0, str(A3))
import consumer_census as cc  # noqa: E402

SRC = '''
import numpy as np
import pytest
import tensor_cuda as tc
import tensor_cuda as T
import tensor_cuda.nn as tnn
from tensor_cuda import functional as F

tc.tensor(x, dtype="float64")                       # L9 unsupported-literal, refused
tc.zeros(3, dtype=np.float32)                       # L10 non-string-literal, refused
tc.ones(3, dtype=None)                              # L11 non-string-literal, refused
dt = "half"
tc.randn(2, dtype=dt)                               # L13 traced-unsupported, refused

@pytest.mark.parametrize("d", ["float16", np.float16])
def f(d):
    return tc.tensor(x, dtype=d)                    # L17 traced-non-string, refused

def g(a, d):
    return tc.rand(a, dtype=d)                      # L20 traced-supported (call site below)
g(1, "bfloat16")

def h(q):
    t = tc.zeros(3)
    tc.tensor(x, dtype=t.dtype)                     # L25 traced-supported (Tensor.dtype)
    return tc.tensor(x, dtype=q.dtype)              # L26 unresolved
F._causal_mask(1, 1, "cpu", "float")                # L27 unsupported-literal, refused (positional)
tnn.parameter(a, dtype=float)                       # L28 non-string-literal (type), refused
class K:
    def m(self):
        return self.tc.tensor(x, dtype="int32")     # L31 unsupported-literal via attribute alias, refused
arr.astype(np.float32)                              # L32 Tensor.astype non-string, unchanged entry
tc.tensor(x, **kw)                                  # L33 unresolved (**kwargs)
T.empty(2, dtype="double")                          # L34 unchanged entry: not newly refused
tc.tensor(x, dtype="float32")                       # L35 supported-literal
tc.tensor(x)                                        # L36 no dtype: not a site
CODE = """import tensor_cuda as tc
tc.tensor(1, dtype="float64")
"""                                                 # L37-39 string code: refused
def z(dtype=None):
    dt = dtype or "float16"
    return tc.zeros(2, dtype=dt)                    # L42 traced-supported: `None or "float16"` is never None
z()
w = None or None
tc.ones(2, dtype=w)                                 # L45 traced-non-string: the last operand is kept
'''
EXPECT = {
    9: ("tc.tensor", "unsupported-literal", True), 10: ("tc.zeros", "non-string-literal", True),
    11: ("tc.ones", "non-string-literal", True), 13: ("tc.randn", "traced-unsupported", True),
    17: ("tc.tensor", "traced-non-string", True), 20: ("tc.rand", "traced-supported", False),
    25: ("tc.tensor", "traced-supported", False), 26: ("tc.tensor", "unresolved", False),
    27: ("functional._causal_mask", "unsupported-literal", True), 28: ("tc.nn.parameter", "non-string-literal", True),
    31: ("tc.tensor", "unsupported-literal", True), 32: ("Tensor.astype", "non-string-literal", False),
    33: ("tc.tensor", "unresolved", False), 34: ("tc.empty", "unsupported-literal", False),
    35: ("tc.tensor", "supported-literal", False), 38: ("tc.tensor", "unsupported-literal", True),
    42: ("tc.zeros", "traced-supported", False), 45: ("tc.ones", "traced-non-string", True),
}
_, rows = cc.census_file("/selftest", "pkg/selftest.py", SRC)
got = {r["line"]: (r["entry"], r["cls"], r["newly_refused"]) for r in rows}
bad = {k: dict(expected=EXPECT.get(k), got=got.get(k)) for k in sorted(set(EXPECT) | set(got)) if EXPECT.get(k) != got.get(k)}
roles = {p: cc.role_of(p) for p in ("artifacts/x/a.py", "tests/a.py", "pkg/test_a.py", "scripts/a.py", "grapa/a.py")}
roles_ok = roles == {"artifacts/x/a.py": "receipt", "tests/a.py": "test", "pkg/test_a.py": "test",
                     "scripts/a.py": "script", "grapa/a.py": "product"}
# Cross-file aliases: core/a.py binds tc and F; core/b.py imports them from core.a; pkg/c.py imports tc from core.b
# (second order) and calls tc.zeros(dtype=np.float16) -> must resolve and be newly refused. Scratch files under /tmp only.
X = Path("/tmp/eng2_a3_selftest_repo")
for rel, text in {"core/a.py": "import tensor_cuda as tc\nfrom tensor_cuda import functional as F\n",
                  "core/b.py": "from core.a import tc, F\n",
                  "pkg/c.py": "import numpy as np\nfrom core.b import tc\nfrom .d import F\ntc.zeros(2, dtype=np.float16)\n",
                  "pkg/d.py": "from core.a import F\n"}.items():
    (X / rel).parent.mkdir(parents=True, exist_ok=True)
    (X / rel).write_text(text)
aliases, failed = cc.cross_file_aliases(str(X), ["core/a.py", "core/b.py", "pkg/c.py", "pkg/d.py"])
_, xrows = cc.census_file(str(X), "pkg/c.py", (X / "pkg/c.py").read_text(), aliases.get("pkg/c.py"))
cross = dict(aliases=aliases, rows=[(r["line"], r["entry"], r["cls"], r["newly_refused"]) for r in xrows],
             ok=aliases == {"core/b.py": {"tc": "tensor_cuda", "F": "tensor_cuda.functional"},
                            "pkg/c.py": {"tc": "tensor_cuda", "F": "tensor_cuda.functional"},
                            "pkg/d.py": {"F": "tensor_cuda.functional"}}
             and [(r["line"], r["entry"], r["cls"], r["newly_refused"]) for r in xrows] == [(4, "tc.zeros", "non-string-literal", True)])
out = dict(sites=len(rows), expected_sites=len(EXPECT), mismatches=bad, roles=roles, roles_ok=roles_ok, cross_file=cross,
           ok=not bad and len(rows) == len(EXPECT) and roles_ok and cross["ok"])
(A3 / "CONSUMER_CENSUS_SELFTEST.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out, indent=1))
sys.exit(0 if out["ok"] else 1)
