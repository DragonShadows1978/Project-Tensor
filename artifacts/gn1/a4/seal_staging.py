"""Seal A2 inputs before any supervised line.
Prior art: SHA-256 NIST (2001), A1 content seal; A2 exact immutable input maps.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
TOOL=Path(__file__).resolve().parent
ROOT=TOOL.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
files=["evidence.py","run_gpu_line.py","run_gpu_tests.py","score_gpu.py","start_gpu_run.py",
       "forge_gpu_run_standin.py","STAGED_GPU_COMMANDS.sh","build_mutants.py","regenerate_arms.py",
       "prepare_a1.py","inspect_cpu.py","cpu_checks.py","cpu_shim.py","derive_tolerances.py",
       "hostile_forge.py","hostile_scorer.py","seal_staging.py","run_cpu_gate.py","pytorch_probe.py","native_validation.py","verify_scope.py","verify_staging.py","gpu_lines.json","mutants.json",
       "engine_pin.json","nn_import_test_files.json","baseline_expected_failures.json","baseline_new_failures.json","baseline_legacy_failures.json",
       "d5_edits.json","order_commit.txt","EQUIVALENCES.md","TOLERANCES.md"]
files = sorted(set(files + [p.name for p in TOOL.glob("*.py")] + [p.name for p in TOOL.glob("ids_*.txt")]))
trace=json.loads((TOOL/"import_trace.json").read_text())
assert trace["exit"]==0
imports={r["path"]:r["sha256"] for r in trace["modules"] if r["local"] and not r["path"].startswith("artifacts/gn1/a4/arms/")}
external={r["path"]:r["sha256"] for r in trace["modules"] if not r["local"]}
# D5 union of collection and all CPU suite runtime origins. Prior art:
# Python sys.modules census (PSF), SHA-256/FIPS 180-4 (2015).
for suite in ("focus", "mutation", "regression"):
    runtime=json.loads((TOOL/("runtime_trace_"+suite+".json")).read_text())
    assert not runtime["collection_errors"]
    files.append("runtime_trace_"+suite+".json")
    for row in runtime["modules"]:
        assert "error" not in row, row
        path=row["path"]
        if path.startswith("artifacts/gn1/a4/arms/"):
            continue
        target=external if path.startswith(("/", "$HOME/")) else imports
        assert path not in target or target[path]==row["sha256"], "census byte drift: "+path
        target[path]=row["sha256"]
tests=json.loads((TOOL/"nn_import_test_files.json").read_text())
files += ["trace_imports.py", "import_trace.json", "REGISTRATION.md", "REGISTRATION.json", "SEAL_IMPORT_TRACE.md", "AMENDMENTS.md"]
seal=dict(order_commit=(TOOL/"order_commit.txt").read_text().strip(),
          order_sha256=sha(ROOT/"orders/GN1_A4_REV3_FIXES.md"),
          engine_sha256=json.loads((TOOL/"engine_pin.json").read_text())["sha256"],
          tool_files={f:sha(TOOL/f) for f in files},test_files={f:sha(ROOT/f) for f in tests},
          import_modules=imports, external_modules=external, import_trace_sha256=sha(TOOL/"import_trace.json"),
          arms=json.loads((TOOL/"arm_maps.json").read_text()), numpy_version=np.__version__,
          equivalences={"V3":"EQUIVALENCES.md#v3", "E_npbool_g":"EQUIVALENCES.md#numpy-bool-g", "E_npbool_c":"EQUIVALENCES.md#numpy-bool-c"})
path=TOOL/"staging_seal.json"
if path.exists():
    old=sha(path);archive=TOOL/("staging_seal.previous."+old+".json")
    if not archive.exists():archive.write_bytes(path.read_bytes())
path.write_text(json.dumps(seal,indent=2)+"\n")
print(json.dumps(dict(seal_sha256=sha(path),tool_files=len(files),arms=len(seal["arms"]))))
