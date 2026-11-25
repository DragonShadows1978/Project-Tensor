"""BWD-F3 P4 (CPU half): a copy of scripts/bwdf2_mutant_results.py running tensor_cuda/tests/test_bwdf3_fixes.py
on each mutant's frozen package (CUDA_VISIBLE_DEVICES empty), scored against artifacts/bwdf1/f3/MUTANTS_F3.json:
KILLED when every listed CPU killing test fails, with its failure text (the mutated line's own behaviour, L41).
Killing entries that name a staged GPU line are listed as STAGED and are not part of the CPU verdict.

  python3 -B -P scripts/bwdf3_mutant_results.py     -> artifacts/bwdf1/f3/MUTANT_RESULTS_F3.json
Prior art: mutation testing (DeMillo, Lipton and Sayward, 1978); scripts/bwdf2_mutant_results.py (this repo).
"""
import hashlib
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC = json.load(open(os.path.join(ROOT, "artifacts/bwdf1/f3/MUTANTS_F3.json")))
RUNS = os.path.join(ROOT, "artifacts/bwdf1/f3/mutants/runs")
SO = "_tensor_cuda.cpython-312-x86_64-linux-gnu.so"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    os.makedirs(RUNS, exist_ok=True)
    out = {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "mutants": {}}
    for name, m in SPEC["mutants"].items():
        pkg = os.path.join(ROOT, "tensor_cuda/build-bwdf1", name, "pkg")
        rec = json.load(open(os.path.join(ROOT, f"artifacts/bwdf1/f3/build/BUILD_tc_{name}.json")))
        got = sha(os.path.join(pkg, "tensor_cuda", SO))
        assert got == rec["module_sha256"], (name, got)
        row = {"ruling": m["ruling"], "where": m["where"], "module_sha256": got, "assertion": m["assertion"]}
        if not m["where"].startswith("CPU"):
            row["result"] = "STAGED (GPU line " + m["gpu_line"] + ")"
            out["mutants"][name] = row
            continue
        xml = os.path.join(RUNS, f"{name}.xml")
        if os.path.exists(xml):
            raise SystemExit(f"{xml} exists; move it aside before a retry")
        if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
            raise SystemExit("CPU lane only: CUDA_VISIBLE_DEVICES must be empty")
        env = dict(os.environ, BWDF3_PKG_PARENT=pkg)
        p = subprocess.run([sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
                            f"--basetemp=/tmp/bwdf3_tmp/pt/mut_{name}", "--junitxml=" + xml,
                            os.path.join(ROOT, "tensor_cuda/tests/test_bwdf3_fixes.py")], cwd=ROOT, env=env,
                           capture_output=True, text=True, timeout=900)
        root = ET.parse(xml).getroot()
        cases = {}
        for tcse in root.iter("testcase"):
            nid = tcse.get("classname").replace(".", "/") + ".py::" + tcse.get("name")
            nid = "tensor_cuda/tests/test_bwdf3_fixes.py::" + tcse.get("name")
            fail = tcse.find("failure")
            err = tcse.find("error")
            cases[nid] = {"outcome": "failed" if fail is not None else ("error" if err is not None else "passed"),
                          "message": (fail.get("message") if fail is not None else
                                      (err.get("message") if err is not None else None))}
        kills = {k: cases.get(k, {"outcome": "missing"}) for k in m["kills"] if "::" in k}
        row["staged_gpu_kills"] = [k for k in m["kills"] if "::" not in k]
        row["pytest_exit"] = p.returncode
        row["killing_tests"] = {k: {"outcome": v["outcome"], "message": (v.get("message") or "")[:600]}
                                for k, v in kills.items()}
        row["all_failing_tests"] = sorted(k for k, v in cases.items() if v["outcome"] != "passed")
        row["result"] = "KILLED" if all(v["outcome"] == "failed" for v in kills.values()) else "SURVIVED"
        out["mutants"][name] = row
        print(name, row["result"], "failing", len(row["all_failing_tests"]), flush=True)
    json.dump(out, open(os.path.join(ROOT, "artifacts/bwdf1/f3/MUTANT_RESULTS_F3.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
