#!/usr/bin/env python3
"""ENG1 CPU mutant kills (no device; CUDA hidden): run the CPU-capable IDs of test_eng1.py (-k cpu, zero-byte CPU
tensors only) in each mutant arm and in the controls, and record which registered killer IDs failed. A mutant whose
killers are all GPU IDs (M05-M07) is left to the lead's GPU line. The receipt shows the killing test executed the
mutated line through the failure it reports (a value only the mutated line produces). Prior art: DeMillo, Lipton and
Sayward (1978). Usage: cpu_mutant_kills.py ARM [ARM ...]
"""
import fnmatch, json, os, subprocess, sys
import xml.etree.ElementTree as ET
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
E = ROOT / "artifacts/eng1"
OUT = E / "cpu_mutant_kills"
MUT = {m["id"]: m for m in json.loads((E / "mutants/mutants.json").read_text())}
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
rows = {}
for arm in sys.argv[1:]:
    test = E / "tool/arms" / arm / "tensor_cuda/tests/test_eng1.py"
    xml = OUT / f"{arm}.xml"
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    p = subprocess.run([sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", "--import-mode=importlib",
                        "--rootdir=" + str(ROOT), "--basetemp=/tmp/eng1_cpu/bt_kill_" + arm, "-k", "cpu",
                        "--junitxml=" + str(xml), str(test)], capture_output=True, text=True, timeout=900,
                       cwd="/tmp/eng1_cpu", env=env)
    (OUT / f"{arm}.log").write_text(p.stdout[-20000:] + p.stderr[-5000:])
    failed, details = [], {}
    for case in ET.parse(xml).getroot().iter("testcase"):
        bad = [e for e in case if e.tag in ("failure", "error")]
        if bad:
            name = case.get("name")
            failed.append(name)
            details[name] = (bad[0].get("message") or "")[:300]
    killers = MUT[arm]["killers"] if arm in MUT else []
    lit = lambda k: k.replace("[", "\x00").replace("]", "[]]").replace("\x00", "[[]")  # '[' ']' literal, '*' wildcard
    hit = [f for f in failed if any(fnmatch.fnmatchcase(f, lit(k)) for k in killers)]
    rows[arm] = dict(pytest_exit=p.returncode, failed=failed, registered_killers=killers, killers_failed=hit,
                     killed=bool(hit), evidence={k: details[k] for k in hit[:4]})
    print(arm, "exit", p.returncode, "failed", len(failed), "killers failed", hit[:6], flush=True)
prev = json.loads((OUT / "summary.json").read_text()) if (OUT / "summary.json").exists() else {}
prev.update(rows)
(OUT / "summary.json").write_text(json.dumps(prev, indent=1) + "\n")
