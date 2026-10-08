"""APA-MF2.1 G5: registered mutation run (registration.json "mutants" M1-M18).

Sources of the tree are never edited: each mutant is a textual substitution (asserted to
match exactly once) in a SCRATCH copy under /tmp/claude-1000/apa_mf2_1_mut:
  py      copy of the tensor_cuda package, first on PYTHONPATH (import path verified)
  kernel  copy of tensor_cuda/{src,include}, rebuilt with scripts/apa_mf2_1_build.sh
          (APA_MF2_1_SRC_ROOT / APA_MF2_1_BUILD_DIR), tests run against that engine
  ext     copy of apa_cuda/csrc/apa.cpp, rebuilt through the MF2 stand-in harness
          (scripts/apa_mf2_ext_stub_build.py), apa_cuda tests via APA_MF2_EXT_DIR
  test    copy of the two test files (F9 reference mutant), run from the copy
Gate tests (tc): test_apa_mf2_1_{short,sentinel,definition,gaps}.py, test_apa_mf2_rule.py,
test_apa_mf2_mask.py, test_apa_phase6.py, test_apa_mf1_quant.py. (ext):
apa_cuda/tests/test_apa_mf2_1_extension.py, test_apa_mf2_extension.py.
Kill rule (registered): a test passing on the unmutated baseline fails with an
AssertionError or pytest "DID NOT RAISE"; failure only through other exceptions -> ERROR
(excluded from the denominator, named).
usage: apa_mf2_1_mutation.py <group: py|kernel|ext|test> <out.json>
Prior art: DeMillo, Lipton & Sayward (1978) "Hints on Test Data Selection" (single-site
mutants, mutation adequacy) -- unverified, lead to check. Mutant list and harness: mine
(MF2's apa_mf2_mutation.py pattern).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCR = "/tmp/claude-1000/apa_mf2_1_mut"
PKG = os.path.join(ROOT, "tensor_cuda/tensor_cuda")
ENG = os.path.join(ROOT, "artifacts/apa_mf2_1/build")
TC_TESTS = [f"tensor_cuda/tests/{t}" for t in (
    "test_apa_mf2_1_short.py", "test_apa_mf2_1_sentinel.py", "test_apa_mf2_1_definition.py",
    "test_apa_mf2_1_gaps.py", "test_apa_mf2_rule.py", "test_apa_mf2_mask.py", "test_apa_phase6.py",
    "test_apa_mf1_quant.py")]
EXT_TESTS = ["apa_cuda/tests/test_apa_mf2_1_extension.py", "apa_cuda/tests/test_apa_mf2_extension.py"]

PY = {
    "M1_F4_sdpa_additive_reversion": ("functional.py",
        '''        scores = scores.masked_fill(
            _causal_mask(L, S, query.device.split(":")[0], query.dtype), float("-inf"))''',
        '''        scores = scores + _causal_mask(L, S, query.device.split(":")[0], query.dtype)'''),
    "M2_F4_quant_ranking_additive": ("quant.py",
        '''        ranking = ranking.masked_fill(causal_add, float("-inf"))''',
        '''        ranking = ranking + causal_add'''),
    "M3_F4_quant_bulk_additive": ("quant.py",
        '''        bulk = bulk.masked_fill(causal_add, float("-inf"))''',
        '''        bulk = bulk + causal_add'''),
    "M5_F4_count_divisor_S": ("quant.py",
        '''cnt = np.maximum(np.arange(S - L + 1, S + 1, dtype=np.float32), 1.0)''',
        '''cnt = np.full((L,), S, dtype=np.float32)'''),
    "M8_one_pass_variance": ("quant.py",
        '''    var = (dev_ * dev_).sum([-1], True) / cnt''',
        '''    var = ((absb * absb).sum([-1], True) / cnt - mean * mean).relu()'''),
    "M9_gt_instead_of_ge": ("quant.py", '''    sel = absb.ge(thr)''', '''    sel = absb.gt(thr)'''),
    "M10_visible_predicate_dropped": ("quant.py",
        '''        sel = sel * vis                      # visible && (0/1 product, never NaN)''', '''        pass'''),
    "M11_sample_variance": ("quant.py",
        '''    var = (dev_ * dev_).sum([-1], True) / cnt''',
        '''    var = (dev_ * dev_).sum([-1], True) / (cnt - 1.0)'''),
    "M12_deviation_unmasked": ("quant.py",
        '''        dev_ = dev_ * vis                    # invisible deviations -> 0 (0/1 product)''', '''        pass'''),
    "M13_bf16_bulk_for_stats": ("quant.py",
        '''            mask = _definition_selection(query.detach().float(), key_quant.float(), scale, z,
                                         is_causal)''',
        '''            mask = _definition_selection(query.detach().float(), key_quant.float(), scale, z,
                                         is_causal, bulk32=bulk_raw.detach().float())'''),
    "M14_sdpa_s_lt_l_check_removed": ("functional.py",
        '''    if is_causal:
        _check_causal_lengths(L, S)      # APA-MF2.1: S < L raises before any matmul''', '''    pass'''),
    "M15_quant_s_lt_l_check_removed": ("quant.py",
        '''    if is_causal:
        F._check_causal_lengths(L, S)    # APA-MF2.1: S < L raises before any matmul''', '''    pass'''),
    "M16_causal_mask_s_lt_l_check_removed": ("functional.py",
        '''def _causal_mask(L, S, device, dtype):
    _check_causal_lengths(L, S)
''', '''def _causal_mask(L, S, device, dtype):
'''),
}
KERNEL = {
    "M6_sentinel_reduce_max": [
        ('''            : (MODE == 1 ? -INFINITY : (MODE == 2 ? INFINITY : 1.f));''',
         '''            : (MODE == 1 ? -3.4e38f : (MODE == 2 ? 3.4e38f : 1.f));'''),
        ('''  float acc = MODE == 0 ? 0.f : (MODE == 1 ? -INFINITY : (MODE == 2 ? INFINITY : 1.f));''',
         '''  float acc = MODE == 0 ? 0.f : (MODE == 1 ? -3.4e38f : (MODE == 2 ? 3.4e38f : 1.f));''')],
    "M7_sentinel_causal_softmax": [('''  float mx = -INFINITY;''', '''  float mx = -3.0e38f;''')],
}
EXT = {
    "M4_F4_apacpp_kNegInf_only": ('''constexpr double kNegInf = -INFINITY;''', '''constexpr double kNegInf = -1e9;'''),
    "M17_apacpp_s_lt_l_check_removed": ('''  TORCH_CHECK(!is_causal || S >= L, "APA forward: causal attention requires S >= L "
              "(bottom-right causal); got L=", L, " queries, S=", S, " keys");''', ''''''),
}
TEST = {"M18_phase6_ref_top_left": ("test_apa_phase6.py",
        '''        cm = np.triu(np.full((q.shape[2], S), -1e9, np.float32), 1 + (S - q.shape[2]))''',
        '''        cm = np.triu(np.full((q.shape[2], S), -1e9, np.float32), 1)''')}


def env(pkg_dir=None, eng=ENG, ext_dir=None):
    e = dict(os.environ)
    e["PYTHONPATH"] = ":".join([eng] + ([pkg_dir] if pkg_dir else []) + [os.path.join(ROOT, "tensor_cuda"), ROOT])
    e["PYTHONDONTWRITEBYTECODE"] = "1"
    if ext_dir:
        e["APA_MF2_EXT_DIR"] = ext_dir
    return e


def run_tests(tests, e, xml, cwd=ROOT):
    subprocess.run([sys.executable, "-m", "pytest", *tests, "-q", "-p", "no:cacheprovider", f"--junitxml={xml}"],
                   cwd=cwd, env=e, capture_output=True, text=True)
    st = {}
    for t in ET.parse(xml).iter("testcase"):
        k = t.get("classname") + "::" + t.get("name")
        s = "passed"
        for ch in t:
            if ch.tag in ("failure", "error"):
                msg = (ch.get("message") or "") + (ch.text or "")[:600]
                s = "assert" if ("AssertionError" in msg or msg.lstrip().startswith("assert")
                                 or "DID NOT RAISE" in msg) else "error"
                if ch.tag == "error":
                    s = "error"
            elif ch.tag == "skipped":
                s = "skipped"
        st[k] = s
    return st


def classify(base, mut):
    passing = [k for k, v in base.items() if v == "passed"]
    a = [k for k in passing if mut.get(k) == "assert"]
    e = [k for k in passing if mut.get(k) == "error"]
    return ("KILLED" if a else ("ERROR" if e else "SURVIVED")), a, e


def sub(path, old, new):
    s = open(path).read()
    assert s.count(old) == 1, (path, s.count(old), old[:60])
    open(path, "w").write(s.replace(old, new))


def pkg_copy(name, fname=None, old=None, new=None):
    d = os.path.join(SCR, "py", name)
    shutil.copytree(PKG, os.path.join(d, "tensor_cuda"), ignore=shutil.ignore_patterns("__pycache__"), dirs_exist_ok=True)
    if fname:
        sub(os.path.join(d, "tensor_cuda", fname), old, new)
    chk = subprocess.run([sys.executable, "-c", "import tensor_cuda.quant as q, tensor_cuda.functional as f, tensor_cuda as tc;"
                          "print(q.__file__, f.__file__, tc._C.__file__)"], cwd="/tmp", env=env(d), capture_output=True, text=True)
    p = chk.stdout.split()
    assert len(p) == 3 and p[0].startswith(d) and p[1].startswith(d) and p[2].startswith(ENG), chk
    return d


def record(res, name, base, st, where):
    v, a, e = classify(base, st)
    res["mutants"][name] = {"verdict": v, "killing_tests": len(a), "error_tests": len(e),
                            "examples": a[:4] or e[:4], "where": where}
    print(name, v, len(a), len(e), (a or e)[:2], flush=True)


def group_py(res):
    d0 = pkg_copy("baseline")
    base = run_tests(TC_TESTS, env(d0), os.path.join(SCR, "py_baseline.xml"))
    res["baseline"] = {"n": len(base), "passed": sum(v == "passed" for v in base.values()),
                       "not_passed": sorted(k for k, v in base.items() if v not in ("passed", "skipped"))}
    print("BASELINE", res["baseline"]["n"], res["baseline"]["passed"], res["baseline"]["not_passed"][:3], flush=True)
    for name, (fname, old, new) in PY.items():
        d = pkg_copy(name, fname, old, new)
        record(res, name, base, run_tests(TC_TESTS, env(d), os.path.join(SCR, f"{name}.xml")), fname)


def group_kernel(res, which):
    base_xml = os.path.join(SCR, "kernel_baseline.xml")
    base = run_tests(TC_TESTS, env(), base_xml)
    res["baseline"] = {"n": len(base), "passed": sum(v == "passed" for v in base.values())}
    print("BASELINE", res["baseline"], flush=True)
    for name in which:
        root = os.path.join(SCR, "kernel", name)
        for sd in ("src", "include"):
            shutil.copytree(os.path.join(ROOT, "tensor_cuda", sd), os.path.join(root, "tensor_cuda", sd), dirs_exist_ok=True)
        for old, new in KERNEL[name]:
            sub(os.path.join(root, "tensor_cuda/src/kernels.cu"), old, new)
        b = subprocess.run(["bash", os.path.join(ROOT, "scripts/apa_mf2_1_build.sh")], cwd=ROOT, capture_output=True, text=True,
                           env={**os.environ, "APA_MF2_1_SRC_ROOT": root, "APA_MF2_1_BUILD_DIR": os.path.join(root, "build")})
        if b.returncode != 0:
            res["mutants"][name] = {"verdict": "ERROR", "build": b.stdout[-800:] + b.stderr[-800:]}
            continue
        record(res, name, base, run_tests(TC_TESTS, env(eng=os.path.join(root, "build")), os.path.join(SCR, f"{name}.xml")),
               "kernels.cu (scratch build)")


def group_ext(res):
    stub = os.path.join(ROOT, "artifacts/apa_mf2_1/ext_stub/after")
    base = run_tests(EXT_TESTS, env(ext_dir=stub), os.path.join(SCR, "ext_baseline.xml"))
    res["baseline"] = {"n": len(base), "passed": sum(v == "passed" for v in base.values())}
    print("BASELINE", res["baseline"], flush=True)
    for name, (old, new) in EXT.items():
        m = os.path.join(SCR, "ext", name)
        os.makedirs(m, exist_ok=True)
        shutil.copyfile(os.path.join(ROOT, "apa_cuda/csrc/apa.cpp"), os.path.join(m, "apa.cpp"))
        sub(os.path.join(m, "apa.cpp"), old, new)
        b = subprocess.run([sys.executable, os.path.join(ROOT, "scripts/apa_mf2_ext_stub_build.py"),
                            os.path.join(m, "apa.cpp"), os.path.join(m, "ext")], cwd=ROOT, capture_output=True, text=True)
        if b.returncode != 0:
            res["mutants"][name] = {"verdict": "ERROR", "build": b.stderr[-800:]}
            continue
        record(res, name, base, run_tests(EXT_TESTS, env(ext_dir=os.path.join(m, "ext")), os.path.join(SCR, f"{name}.xml")),
               "apa.cpp (stand-in harness)")


def group_test(res):
    for name, (fname, old, new) in TEST.items():
        out = {}
        for arm in ("baseline", name):
            t = os.path.join(SCR, "test", arm, "tests")
            os.makedirs(t, exist_ok=True)
            for f in ("test_apa_mf2_1_gaps.py", "test_apa_phase6.py"):
                shutil.copyfile(os.path.join(ROOT, "tensor_cuda/tests", f), os.path.join(t, f))
            if arm != "baseline":
                sub(os.path.join(t, fname), old, new)
            out[arm] = run_tests([os.path.join(t, "test_apa_mf2_1_gaps.py"), os.path.join(t, "test_apa_phase6.py")],
                                 env(), os.path.join(SCR, f"test_{arm}.xml"), cwd=t)
        res["baseline"] = {"n": len(out["baseline"]), "passed": sum(v == "passed" for v in out["baseline"].values())}
        record(res, name, out["baseline"], out[name], "tests/test_apa_phase6.py (copy)")


def main():
    group, out = sys.argv[1], sys.argv[2]
    os.makedirs(SCR, exist_ok=True)
    res = {"group": group, "mutants": {}}
    if group == "py":
        group_py(res)
    elif group.startswith("kernel"):
        group_kernel(res, [k for k in KERNEL if group == "kernel" or k.startswith(group.split(":")[1])])
    elif group == "ext":
        group_ext(res)
    else:
        group_test(res)
    v = [m["verdict"] for m in res["mutants"].values()]
    res["killed"], res["survived"], res["error"] = v.count("KILLED"), v.count("SURVIVED"), v.count("ERROR")
    with open(out, "x") as f:
        json.dump(res, f, indent=1)
    print("KILLED", res["killed"], "SURVIVED", res["survived"], "ERROR", res["error"], "WROTE", out, flush=True)


if __name__ == "__main__":
    main()
