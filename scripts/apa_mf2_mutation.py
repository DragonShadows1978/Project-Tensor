"""APA-MF2 registered mutation run (registration.json "mutants" M1-M9).

Sources are never edited: each mutant is ONE textual substitution (asserted to match
exactly once) in a scratch copy of the tensor_cuda package under /tmp, placed first
on PYTHONPATH; the import path is verified per mutant before its tests count. M5
mutates a copy of apa_cuda/csrc/apa.cpp, rebuilt through the stub harness
(scripts/apa_mf2_ext_stub_build.py) and run via APA_MF2_EXT_DIR.
Gate tests: tensor_cuda/tests/test_apa_mf2_rule.py, test_apa_mf2_mask.py,
test_apa_phase6.py (M1-M4, M6-M9); apa_cuda/tests/test_apa_mf2_extension.py and
test_parity.py (M5). The unmutated baseline is NOT all-green (7 registered exactness
ids fail -- the G2a residual), so a mutant is KILLED iff >= 1 test that PASSES on the
baseline FAILS on the mutant with an AssertionError; a failure only through other
exceptions classifies it ERROR (not a kill, excluded from the denominator).
Prior art: DeMillo, Lipton & Sayward 1978 ("Hints on Test Data Selection", mutation
adequacy) -- unverified, lead to check; mutant list and harness: mine.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCR = "/tmp/claude-1000/apa_mf2_mut"
PKG = os.path.join(ROOT, "tensor_cuda/tensor_cuda")
TC_TESTS = ["tensor_cuda/tests/test_apa_mf2_rule.py", "tensor_cuda/tests/test_apa_mf2_mask.py",
            "tensor_cuda/tests/test_apa_phase6.py"]
EXT_TESTS = ["apa_cuda/tests/test_apa_mf2_extension.py", "apa_cuda/tests/test_parity.py"]

MUTANTS = {
    "M1_bf16_bulk_for_stats": ("quant.py",
        """            mask = _bk4_selection(query.detach().float(), key_quant.float(), scale, z,
                                  is_causal)""",
        """            mask = _bk4_selection(query.detach().float(), key_quant.float(), scale, z,
                                  is_causal, bulk32=bulk_raw.detach().float())"""),
    "M2_visible_predicate_dropped": ("quant.py",
        "        sel = sel * vis                      # visible && (0/1 product, never NaN)",
        "        pass"),
    "M3_sample_variance": ("quant.py",
        "    var = ((absb * absb).sum([-1], True) / cnt - mean * mean).relu()",
        "    var = ((absb * absb).sum([-1], True) / cnt - mean * mean).relu() * cnt / (cnt - 1.0)"),
    "M4_bias_back_to_minus_1e4": ("functional.py",
        "bias = np.triu(np.full((L, S), -np.inf, dtype=np.float32), k=1 + (S - L))",
        "bias = np.triu(np.full((L, S), -1e4, dtype=np.float32), k=1 + (S - L))"),
    "M6_count_off_by_one": ("quant.py",
        "cnt = np.maximum(np.arange(S - L + 1, S + 1, dtype=np.float32), 1.0)",
        "cnt = np.maximum(np.arange(S - L, S, dtype=np.float32), 1.0)"),
    "M7_gt_instead_of_ge": ("quant.py",
        "    sel = absb.ge(thr)",
        "    sel = absb.gt(thr)"),
    "M8_clamp_dropped": ("quant.py",
        "    var = ((absb * absb).sum([-1], True) / cnt - mean * mean).relu()",
        "    var = ((absb * absb).sum([-1], True) / cnt - mean * mean)"),
    "M9_noncausal_centered": ("quant.py",
        "    thr = mean + var.sqrt() * z\n",
        "    thr = mean + var.sqrt() * z\n    if not is_causal:\n        thr = absb.mean([-1], True) + absb.std([-1], True) * z\n"),
}
M5 = ("ri = at::arange(row_base + (S - L), row_base + (S - L) + rows,",
      "ri = at::arange(row_base, row_base + rows,")


def env(extra_pp=None, ext_dir=None):
    e = dict(os.environ)
    pp = e.get("PYTHONPATH", "")
    if extra_pp:
        pp = extra_pp + ":" + pp
    e["PYTHONPATH"] = pp
    if ext_dir:
        e["APA_MF2_EXT_DIR"] = ext_dir
    return e


def run_tests(tests, e, xml):
    subprocess.run([sys.executable, "-m", "pytest", *tests, "-q", "-p", "no:cacheprovider",
                    f"--junitxml={xml}"], cwd=ROOT, env=e, capture_output=True, text=True)
    st = {}
    for t in ET.parse(xml).iter("testcase"):
        k = t.get("classname") + "::" + t.get("name")
        s = "passed"
        for ch in t:
            if ch.tag in ("failure", "error"):
                msg = (ch.get("message") or "") + (ch.text or "")[:400]
                s = "assert" if ("AssertionError" in msg or msg.startswith("assert")) else "error"
            elif ch.tag == "skipped":
                s = "skipped"
        st[k] = s
    return st


def classify(base, mut):
    passing = [k for k, v in base.items() if v == "passed"]
    a = [k for k in passing if mut.get(k) == "assert"]
    e = [k for k in passing if mut.get(k) == "error"]
    return ("KILLED" if a else ("ERROR" if e else "SURVIVED")), a, e


def mutant_pkg(name, fname, old, new):
    d = os.path.join(SCR, name)
    shutil.copytree(PKG, os.path.join(d, "tensor_cuda"), ignore=shutil.ignore_patterns("__pycache__"),
                    dirs_exist_ok=True)
    if fname:
        p = os.path.join(d, "tensor_cuda", fname)
        s = open(p).read()
        assert s.count(old) == 1, (name, s.count(old))
        open(p, "w").write(s.replace(old, new))
    chk = subprocess.run([sys.executable, "-c", "import tensor_cuda.quant as q, tensor_cuda.functional as f, tensor_cuda as tc;"
                          "print(q.__file__, f.__file__, tc._C.__file__)"], cwd="/tmp", env=env(d),
                         capture_output=True, text=True)
    paths = chk.stdout.split()
    assert len(paths) == 3 and paths[0].startswith(d) and paths[1].startswith(d) and "artifacts/apa_mf2/build" in paths[2], chk
    return d


def main():
    out, mode = sys.argv[1], sys.argv[2]
    os.makedirs(SCR, exist_ok=True)
    res = {"mode": mode, "baseline": {}, "mutants": {}}
    if mode == "ext":
        return ext_part(out, res)
    d0 = mutant_pkg("baseline", None, None, None)
    base_tc = run_tests(TC_TESTS, env(d0), os.path.join(SCR, "baseline_tc.xml"))
    res["baseline"]["tensor_cuda"] = {"n": len(base_tc), "passed": sum(v == "passed" for v in base_tc.values()),
                                      "failing_ids": sorted(k for k, v in base_tc.items() if v != "passed")}
    print("BASELINE tc", res["baseline"]["tensor_cuda"]["n"], res["baseline"]["tensor_cuda"]["passed"], flush=True)
    for name, (fname, old, new) in MUTANTS.items():
        d = mutant_pkg(name, fname, old, new)
        st = run_tests(TC_TESTS, env(d), os.path.join(SCR, f"{name}.xml"))
        verdict, a, e = classify(base_tc, st)
        res["mutants"][name] = {"verdict": verdict, "killing_tests": len(a), "error_tests": len(e),
                                "examples": a[:4] or e[:4], "file": fname}
        print(name, verdict, len(a), len(e), a[:2], flush=True)
    finish(out, res)


def ext_part(out, res):
    # M5: apa.cpp top-left, through the stub harness
    stub_base = os.path.join(ROOT, "artifacts/apa_mf2/ext_stub/after")
    base_ext = run_tests(EXT_TESTS, env(ext_dir=stub_base), os.path.join(SCR, "baseline_ext.xml"))
    res["baseline"]["apa_cuda_stub"] = {"n": len(base_ext), "passed": sum(v == "passed" for v in base_ext.values())}
    src = open(os.path.join(ROOT, "apa_cuda/csrc/apa.cpp")).read()
    assert src.count(M5[0]) == 1
    m5 = os.path.join(SCR, "M5_apa_cpp_top_left")
    os.makedirs(m5, exist_ok=True)
    open(os.path.join(m5, "apa.cpp"), "w").write(src.replace(M5[0], M5[1]))
    b = subprocess.run([sys.executable, os.path.join(ROOT, "scripts/apa_mf2_ext_stub_build.py"),
                        os.path.join(m5, "apa.cpp"), os.path.join(m5, "ext")], cwd=ROOT, capture_output=True, text=True)
    assert b.returncode == 0, b.stderr[-2000:]
    st = run_tests(EXT_TESTS, env(ext_dir=os.path.join(m5, "ext")), os.path.join(SCR, "M5.xml"))
    verdict, a, e = classify(base_ext, st)
    res["mutants"]["M5_apa_cpp_top_left"] = {"verdict": verdict, "killing_tests": len(a), "error_tests": len(e),
                                             "examples": a[:4] or e[:4], "file": "apa.cpp (stub harness)"}
    print("M5_apa_cpp_top_left", verdict, len(a), len(e), a[:2], flush=True)
    finish(out, res)


def finish(out, res):
    v = [m["verdict"] for m in res["mutants"].values()]
    nonerr = [x for x in v if x != "ERROR"]
    res["killed"] = v.count("KILLED"); res["survived"] = v.count("SURVIVED"); res["error"] = v.count("ERROR")
    res["kill_rate_non_error"] = res["killed"] / len(nonerr) if nonerr else 0.0
    res["gate_ge_0_80"] = res["kill_rate_non_error"] >= 0.80
    with open(out, "x") as f:
        json.dump(res, f, indent=1)
    print("KILLED", res["killed"], "SURVIVED", res["survived"], "ERROR", res["error"],
          "rate", res["kill_rate_non_error"], "WROTE", out, flush=True)


if __name__ == "__main__":
    main()
