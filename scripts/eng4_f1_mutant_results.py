"""ENG4-F1 H4 mutant runs -> artifacts/eng4/f1/MUTANT_RESULTS_F1.json.

  CUDA_VISIBLE_DEVICES= python3 -B -P scripts/eng4_f1_mutant_results.py cpu   -> every CPU killing test, per mutant
  python3 -B -P scripts/eng4_f1_mutant_results.py gpu                         -> adds the GPU lines mut_m02_f1, mut_m08

CPU: each killing test (artifacts/eng4/f1/MUTANTS_F1.json, the non-GPU ones) runs in its own pytest process against
the mutant's package (/tmp/eng4_src_<id>/tensor_cuda; module checked against f1/build/BUILD_<id>.json), CUDA hidden.
Killed = the test failed on its NAMED assertion (the registered failure text: pytest.raises 'Regex pattern did not
match' / 'DID NOT RAISE', or m08's 'zero check') AND its captured output shows 'ENG4-MUTANT <id> executed' (L41).

Prior art: mutation testing (DeMillo, Lipton and Sayward, 1978); ENG4's eng4_mutant_results.py (this repo).
No other prior art known to me.
"""
import datetime
import hashlib
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SO = "_tensor_cuda.cpython-312-x86_64-linux-gnu.so"
F1 = os.path.join(ROOT, "artifacts/eng4/f1")
OUT = os.path.join(F1, "MUTANT_RESULTS_F1.json")
NAMED = {"m08_k0_gemm": ["zero check"]}
DEFAULT_NAMED = ["Regex pattern did not match", "DID NOT RAISE"]


def sha(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def junit_case(path):
    if not os.path.exists(path):
        return {"outcome": "missing", "message": "", "captured": ""}
    for tc in ET.parse(path).getroot().iter("testcase"):
        f, e, s = tc.find("failure"), tc.find("error"), tc.find("skipped")
        out = "".join((x.text or "") for x in tc.findall("system-err")) + "".join(
            (x.text or "") for x in tc.findall("system-out"))
        node = f if f is not None else e
        return {"outcome": "failed" if f is not None else "error" if e is not None else
                "skipped" if s is not None else "passed",
                "message": (node.get("message") or "") if node is not None else "", "captured": out}
    return {"outcome": "missing", "message": "", "captured": ""}


def cpu():
    reg = json.load(open(os.path.join(F1, "MUTANTS_F1.json")))
    res = {"written_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "MUTANTS_F1.json_sha256": sha(os.path.join(F1, "MUTANTS_F1.json")), "mutants": {}}
    d = os.path.join(F1, "mutants", "runs")
    os.makedirs(d, exist_ok=True)
    for mid, m in reg["mutants"].items():
        pkg = f"/tmp/eng4_src_{mid}/tensor_cuda"
        want = json.load(open(os.path.join(F1, "build", f"BUILD_{mid}.json")))["module_sha256"]
        got = sha(os.path.join(pkg, "tensor_cuda", SO))
        assert got == want, (mid, got, want)
        named = NAMED.get(mid, DEFAULT_NAMED)
        rows = []
        for i, tid in enumerate(t for t in m["killing_tests"] if "test_gpu_" not in t):
            junit = os.path.join(d, f"{mid}_{i}.xml")
            env = dict(os.environ, PYTHONPATH=pkg, CUDA_VISIBLE_DEVICES="", TMPDIR="/tmp/eng4_tmp")
            p = subprocess.run(["python3", "-B", "-P", "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rA",
                                f"--basetemp=/tmp/eng4_tmp/pt/f1mut_{mid}_{i}", f"--junitxml={junit}",
                                "-o", "junit_logging=all", os.path.join(ROOT, tid)],
                               capture_output=True, text=True, timeout=600, cwd=ROOT, env=env)
            with open(os.path.join(d, f"{mid}_{i}.log"), "w") as f:
                f.write(p.stdout + p.stderr)
            c = junit_case(junit)
            is_named = c["outcome"] == "failed" and any(n in c["message"] for n in named)
            marker = f"ENG4-MUTANT {mid} executed" in c["captured"]
            rows.append({"test": tid, "pytest_exit": p.returncode, "outcome": c["outcome"],
                         "failure_message": c["message"][:600], "named_assertion_failed": is_named,
                         "mutated_line_executed_in_test": marker, "killed": is_named and marker,
                         "junit": os.path.relpath(junit, ROOT)})
        res["mutants"][mid] = {"module_sha256": got, "named_assertion": m["named_assertion"], "cpu_rows": rows,
                               "cpu_killing_tests": len(rows),
                               "killed_on_cpu": bool(rows) and all(r["killed"] for r in rows)}
        print(mid, len(rows), "killed" if res["mutants"][mid]["killed_on_cpu"] else
              [(r["test"].split("::")[1], r["outcome"], r["named_assertion_failed"], r["mutated_line_executed_in_test"])
               for r in rows if not r["killed"]][:4])
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)


def gpu():
    res = json.load(open(OUT))
    for line, mid in (("mut_m02_f1", "m02_f1"), ("mut_m08", "m08_k0_gemm")):
        p = os.path.join(F1, "gpu", line, f"MUT_{mid}.json")
        rows = json.load(open(p)) if os.path.exists(p) else None
        res["mutants"][mid]["gpu_line"] = {"line": line, "rows": rows,
                                           "killed": bool(rows) and all(r["killed"] for r in rows)}
        print(mid, res["mutants"][mid]["gpu_line"]["killed"])
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    {"cpu": cpu, "gpu": gpu}[sys.argv[1]]()
