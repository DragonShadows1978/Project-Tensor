"""G1 / named mutant assertion runner, pinned copied engine package.
Prior art: pytest hooks (pytest, 2004 onward), SHA-256 (NIST, 2001).
Taken: phase reports and artifact hashes; ours: this receipt layout.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts/eng3f1"


def main():
    arm = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else "G1"
    receipt = json.loads((ART / "build" / f"BUILD_{arm}.json").read_text())
    module = ROOT / receipt["module"]
    assert hashlib.sha256(module.read_bytes()).hexdigest() == receipt["module_sha256"]
    pkg = str(module.parent.parent)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", TMPDIR="/tmp/eng3f1_tmp", PYTHONPATH=pkg,
               ENG3F1_PACKAGE_PARENT=pkg, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", ENG3_RECORD_DIR=str(ART/"cpu"/label))
    out = ART/"cpu"/label
    out.mkdir(parents=True, exist_ok=False)
    selection = "g1_cpu or (g1b and cpu) or f1_cpu"
    files = ["tensor_cuda/tests/test_eng3_host.py", "artifacts/eng3f1/tools/eng3f1_checks.py"]
    if arm != "F":
        mutants = json.loads((ART/"MUTANTS.json").read_text())["mutants"]
        selection = mutants[arm]["test"]
        files = files[1:]
    cmd = [sys.executable, "-B", "-P", "-m", "pytest", "-q", "-s", "-p", "no:cacheprovider",
           "--basetemp=/tmp/eng3f1_tmp/pt_"+label, "--junitxml="+str(out/"junit.xml"),
           "-k", selection] + files
    with (out/"run.log").open("w") as f:
        p = subprocess.run(["timeout", "1500s", "nice", "-n", "19"] + cmd, cwd=ROOT, env=env,
                           stdout=f, stderr=subprocess.STDOUT)
    log=(out/"run.log").read_text()
    obj={"arm":arm,"module_sha256":receipt["module_sha256"],"argv":cmd,"exit":p.returncode,
         "log_sha256":hashlib.sha256(log.encode()).hexdigest(),"mutated_line_executed": "ENG3F1_EXEC "+arm in log}
    (out/"RESULT.json").write_text(json.dumps(obj,indent=2))
    print(json.dumps(obj),flush=True)
    print(log[-4000:],flush=True)
    return p.returncode

if __name__ == "__main__":
    sys.exit(main())
