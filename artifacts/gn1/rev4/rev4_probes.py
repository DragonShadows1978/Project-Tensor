#!/usr/bin/env python3
"""GN1-REV4 probes: (1) REV3's seal_runtime_imports against A4's seal; (2) REV3's
npbool_deletion_probe against A4's REGISTERED equivalence arms (E_npbool_g/c,
regenerated in the /tmp mirror) under A4's cpu_shim; (3) census-instrument
neutrality (initial/precorrection traces vs final); (4) the hostile operand
runtime_sentinel.py vs the seal and the commit.
Prior art: Python sys.modules origin census (PSF docs); DeMillo, Lipton, Sayward
(1978) equivalent mutants; PEP 357 (2006) operator.index. Ours: the selection.
"""
import hashlib, importlib.util, json, operator, os, subprocess, sys
from pathlib import Path
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
OUT = Path(__file__).resolve().parent
REAL = OUT.parents[2]
A4 = OUT.parent / "a4"
MROOT = Path("/tmp/gn1_rev4_m/W"); MA4 = MROOT / "artifacts/gn1/a4"
which = sys.argv[1]
home = str(Path.home())
seal = json.loads((A4 / "staging_seal.json").read_text())
res = {}
if which == "seal":
    pinned = {p.replace("$HOME/", home + "/", 1) for p in seal["external_modules"]}
    import numpy as np, pytest  # noqa
    files = lambda: {str(Path(m.__file__).resolve()) for m in list(sys.modules.values()) if getattr(m, "__file__", None) and Path(m.__file__).is_file()}
    before = files()
    np.random.seed(571); _ = np.testing.assert_allclose
    new = sorted(files() - before)
    unp = [p for p in new if p not in pinned]
    res = dict(numpy=np.__version__, newly_loaded_at_runtime=len(new), unpinned=len(unp),
               unpinned_sample=[p.replace(home, "$HOME", 1) for p in unp[:15]], seal_external_count=len(pinned),
               rev3_baseline="A3: 23 loaded, 19 unpinned of 513")
    out = "seal_runtime_imports_a4.json"
elif which == "npbool":
    import numpy as np, pytest
    spec = importlib.util.spec_from_file_location("a4_cpu_shim", MA4 / "cpu_shim.py")
    S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
    res["numpy"] = np.__version__
    for v in (np.True_, np.False_):
        try:
            operator.index(v); res["operator.index(%r)" % v] = "accepted"
        except TypeError as e:
            res["operator.index(%r)" % v] = "TypeError: " + str(e)
    class Count:
        def __init__(self): self.failed = []; self.n = 0
        def pytest_runtest_logreport(self, report):
            if report.when == "call": self.n += 1
            if report.failed: self.failed.append(report.nodeid)
    for arm in ("candidate", "E_npbool_g", "E_npbool_c", "D5_g_npbool", "D5_c_npbool"):
        path = MA4 / "arms" / arm / "tensor_cuda/nn.py"
        if arm != "candidate":
            res[arm + "::diff"] = (MA4 / "mutants" / arm / "nn.diff").read_text()[-700:]
        S.install(path)
        c = Count()
        rc = int(pytest.main(["-q", "-p", "no:cacheprovider", "--import-mode=importlib", "--rootdir=" + str(MROOT),
                              "--basetemp=/tmp/gn1_rev4_tmp/npb_" + arm,
                              str(MROOT / "tensor_cuda/tests/test_groupnorm_gn1.py")], plugins=[c]))
        res[arm] = dict(exit=rc, calls=c.n, failed=len(c.failed), failed_sample=c.failed[:3])
    out = "npbool_deletion_probe_a4.json"
elif which == "census":
    def mods(p):
        d = json.loads(Path(p).read_text())
        return d, {r["path"]: r.get("sha256") for r in d["modules"]}
    for suite, old in (("regression", ".initial"), ("focus", ".precorrection"), ("mutation", ".precorrection")):
        new_p, old_p = A4 / ("runtime_trace_%s.json" % suite), A4 / ("runtime_trace_%s.json%s" % (suite, old))
        dn, mn = mods(new_p); do, mo = mods(old_p)
        res[suite] = dict(files_identical=new_p.read_bytes() == old_p.read_bytes(), modules_new=len(mn), modules_old=len(mo),
                          module_maps_equal=mn == mo, only_new=sorted(set(mn) - set(mo))[:10], only_old=sorted(set(mo) - set(mn))[:10],
                          exit_new=dn.get("exit"), exit_old=do.get("exit"), ids=len(dn["collected_ids"]),
                          setup_reports=len({r["id"] for r in dn["reports"] if r["when"] == "setup"}),
                          call_reports=len({r["id"] for r in dn["reports"] if r["when"] == "call"}),
                          call_outcomes={o: sum(1 for r in dn["reports"] if r["when"] == "call" and r["outcome"] == o) for o in ("passed", "failed", "skipped")},
                          setup_failed=sum(1 for r in dn["reports"] if r["when"] == "setup" and r["outcome"] == "failed"))
    sealed_paths = set(seal["external_modules"]) | set(seal["import_modules"])
    all_trace = set()
    for suite in ("focus", "mutation", "regression"):
        all_trace |= {r["path"] for r in json.loads((A4 / ("runtime_trace_%s.json" % suite)).read_text())["modules"]
                      if not r["path"].startswith("artifacts/gn1/a4/arms/")}
    res["trace_paths_not_in_seal"] = sorted(all_trace - sealed_paths)[:10]
    res["numpy_random_sealed"] = sum("numpy/random/" in p for p in seal["external_modules"])
    res["numpy_testing_sealed"] = sum("numpy/testing/" in p for p in seal["external_modules"])
    res["runtime_sentinel_in_tool_files"] = "runtime_sentinel.py" in seal["tool_files"]
    ls = subprocess.run(["git", "-C", str(REAL), "ls-tree", "--name-only", "a14c0aad", "artifacts/gn1/a4/runtime_sentinel.py"],
                        capture_output=True, text=True)
    res["runtime_sentinel_in_commit_a14c0aad"] = bool(ls.stdout.strip())
    out = "census_neutrality_a4.json"
(OUT / out).write_text(json.dumps(res, indent=2) + "\n")
print(json.dumps(res)[:3000])
