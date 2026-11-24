"""ENG4 G5 mutant runs and MUTANT_RESULTS.json.

  CUDA_VISIBLE_DEVICES= python3 -B -P scripts/eng4_mutant_results.py cpu   -> runs each mutant's CPU killing tests
  python3 -B -P scripts/eng4_mutant_results.py gpu                         -> adds mut_m02's GPU line result

CPU: for each mutant in artifacts/eng4/MUTANTS.json, each killing test with a CPU witness runs in its own pytest
process against the mutant's package (/tmp/eng4_src_<id>/tensor_cuda, its module checked against BUILD_<id>.json),
with CUDA hidden. A test kills the mutant when it fails on its NAMED assertion: the junit failure message must come
from the pytest.raises in that test ('Regex pattern did not match' or 'DID NOT RAISE'). L41: the test's captured
stderr must contain 'ENG4-MUTANT <id> executed' (the mutated line ran inside that test).

Prior art: mutation testing (DeMillo, Lipton and Sayward, 1978); ENG3's eng3_mutant_results.py (this repo). No other
prior art known to me.
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
OUT = os.path.join(ROOT, "artifacts/eng4/MUTANT_RESULTS.json")


def sha(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def junit_case(path):
    root = ET.parse(path).getroot()
    for tc in root.iter("testcase"):
        fail = tc.find("failure")
        err = tc.find("error")
        skip = tc.find("skipped")
        out = "".join((x.text or "") for x in tc.findall("system-err")) + "".join(
            (x.text or "") for x in tc.findall("system-out"))
        node = fail if fail is not None else err
        return {"outcome": "failed" if fail is not None else "error" if err is not None else
                "skipped" if skip is not None else "passed",
                "message": (node.get("message") if node is not None else None), "captured": out}
    return {"outcome": "missing"}


def cpu():
    reg = json.load(open(os.path.join(ROOT, "artifacts/eng4/MUTANTS.json")))
    res = {"written_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "MUTANTS.json_sha256": sha(os.path.join(ROOT, "artifacts/eng4/MUTANTS.json")), "mutants": {}}
    d = os.path.join(ROOT, "artifacts/eng4/mutants/runs")
    os.makedirs(d, exist_ok=True)
    for mid, m in reg["mutants"].items():
        pkg = f"/tmp/eng4_src_{mid}/tensor_cuda"
        want = json.load(open(os.path.join(ROOT, f"artifacts/eng4/build/BUILD_{mid}.json")))["module_sha256"]
        got = sha(os.path.join(pkg, "tensor_cuda", SO))
        assert got == want, (mid, got, want)
        rows = []
        for i, tid in enumerate(t for t in m["killing_tests"] if "test_gpu_" not in t):
            junit = os.path.join(d, f"{mid}_{i}.xml")
            env = dict(os.environ, PYTHONPATH=pkg, CUDA_VISIBLE_DEVICES="", TMPDIR="/tmp/eng4_tmp")
            p = subprocess.run(["python3", "-B", "-P", "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rA",
                                f"--basetemp=/tmp/eng4_tmp/pt/mut_{mid}_{i}", f"--junitxml={junit}",
                                "-o", "junit_logging=all", os.path.join(ROOT, tid)],
                               capture_output=True, text=True, timeout=600, cwd=ROOT, env=env)
            with open(os.path.join(d, f"{mid}_{i}.log"), "w") as f:
                f.write(p.stdout + p.stderr)
            c = junit_case(junit)
            msg = c.get("message") or ""
            named = c["outcome"] == "failed" and ("Regex pattern did not match" in msg or "DID NOT RAISE" in msg)
            marker = f"ENG4-MUTANT {mid} executed" in (c.get("captured") or "")
            rows.append({"test": tid, "pytest_exit": p.returncode, "outcome": c["outcome"],
                         "failure_message": msg[:600], "named_assertion_failed": named,
                         "mutated_line_executed_in_test": marker, "killed": named and marker,
                         "junit": os.path.relpath(junit, ROOT)})
        res["mutants"][mid] = {"module_sha256": got, "named_assertion": m["named_assertion"], "rows": rows,
                               "killed_on_cpu": bool(rows) and all(r["killed"] for r in rows)}
        print(mid, [(r["outcome"], r["named_assertion_failed"], r["mutated_line_executed_in_test"]) for r in rows])
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)


def gpu():
    res = json.load(open(OUT))
    p = os.path.join(ROOT, "artifacts/eng4/gpu/mut_m02/MUT_m02.json")
    rows = json.load(open(p)) if os.path.exists(p) else None
    res["mutants"]["m02_no_device_check"]["gpu_line"] = {"rows": rows,
                                                        "killed": bool(rows) and all(r["killed"] for r in rows)}
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res["mutants"]["m02_no_device_check"]["gpu_line"], indent=1))


if __name__ == "__main__":
    {"cpu": cpu, "gpu": gpu}[sys.argv[1]]()
