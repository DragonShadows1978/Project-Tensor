#!/usr/bin/env python3
"""ENG1-REV2 Q2: rerun REV1's d4_child_probe.py scenario on CPU for one arm, unbound (REV1's exact scenario, as a
control) and bound (A1's arm_origin.bound_environment + preflight, as run_gpu_tests.py applies them), plus two
refusal probes of the origin preflight: a wrong pin, and a cwd that shadows tensor_cuda with a decoy package.
Usage (via run.py, fresh parent per arm): python3 -B q2_probe.py ARM
Every child's exit is recorded after subprocess.run returns, i.e. after the child exited (L45).
Prior art: REV1 d4_child_probe.py (credited: the scenario and its import line); Python sys.path/PYTHONPATH rules
(PSF docs: sys.path[0] is the cwd for -c); SHA-256 (FIPS 180-4). No new algorithm.
"""
import datetime, hashlib, json, os, subprocess, sys
from pathlib import Path

REV2 = Path(__file__).resolve().parent
ROOT = REV2.parents[2]
TOOL = ROOT / "artifacts/eng1/tool"
arm = sys.argv[1]
SCR = Path("/tmp/eng1_rev2_work/probe_" + arm)
SCR.mkdir(parents=True, exist_ok=True)
package = TOOL / "arms" / arm / "tensor_cuda"
module = next((package / "tensor_cuda").glob("_tensor_cuda*.so"))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
seal = json.loads((TOOL / "staging_seal_a1.json").read_text())
pin = seal["engine_sha256"][arm]
utc = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
assert "PYTHONPATH" not in os.environ and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
sys.path.insert(0, str(TOOL))
from arm_origin import bound_environment, preflight  # noqa: E402

# REV1 d4_child_probe.py's child code, verbatim.
CODE = ("import json, tensor_cuda as tc; print(json.dumps({'package':getattr(tc,'__file__',None),"
        "'engine':getattr(getattr(tc,'_C',None),'__file__',None)}))")


def child(env, cwd):
    start = utc()
    p = subprocess.run([sys.executable, "-B", "-c", CODE], cwd=str(cwd), env=env, capture_output=True, text=True,
                       timeout=60)
    rec = dict(start=start, child_exited=utc(), exit=p.returncode, stdout=p.stdout.strip(),
               stderr_tail=p.stderr.strip().splitlines()[-1:] if p.stderr.strip() else [], cwd=str(cwd),
               PYTHONPATH=env.get("PYTHONPATH"))
    if p.returncode == 0:
        out = json.loads(p.stdout)
        rec["imported_package"], rec["imported_engine"] = out["package"], out["engine"]
        rec["engine_sha256"] = sha(out["engine"]) if out["engine"] else None
        rec["engine_is_arm_pin"] = rec["engine_sha256"] == pin
        rec["package_in_arm"] = Path(out["package"]).resolve() == (package / "tensor_cuda/__init__.py").resolve()
    return rec


result = dict(arm=arm, pin=pin, module_sha256=sha(module), module_matches_seal=sha(module) == pin)
# 1. Control: REV1's exact scenario (process-local preload, inherited environment without PYTHONPATH, cwd=tool).
sys.path.insert(0, str(package))
import tensor_cuda as tc  # noqa: E402
result["parent_engine"] = tc._C.__file__
result["parent_engine_sha256"] = sha(tc._C.__file__)
result["unbound_rev1_scenario"] = child(dict(os.environ), TOOL)
# 2. A1's fix, exactly as run_gpu_tests.py applies it: bind os.environ, then preflight under that env and cwd.
os.environ.update(bound_environment(package))
result["bound_preflight"] = preflight(package, seal["module_name"], pin, str(TOOL), SCR / "origin_ok.json")
result["bound_rev1_scenario"] = child(dict(os.environ), TOOL)
# 3. Refusal: a wrong expected pin must raise before anything else runs.
try:
    preflight(package, seal["module_name"], "0" * 64, str(TOOL), SCR / "origin_wrong_pin.json")
    result["wrong_pin"] = dict(refused=False)
except AssertionError as e:
    result["wrong_pin"] = dict(refused=True, message=str(e), pytest_imported="pytest" in sys.modules,
                               receipt=json.loads((SCR / "origin_wrong_pin.json").read_text()))
# 4. Refusal: a cwd holding a decoy tensor_cuda package (sys.path[0] = cwd for -c precedes PYTHONPATH).
decoy = SCR / "decoy_cwd"
(decoy / "tensor_cuda").mkdir(parents=True, exist_ok=True)
(decoy / "tensor_cuda/__init__.py").write_text("# decoy package for ENG1-REV2 probe\n")
try:
    preflight(package, seal["module_name"], pin, str(decoy), SCR / "origin_decoy.json")
    result["decoy_cwd"] = dict(refused=False)
except AssertionError as e:
    result["decoy_cwd"] = dict(refused=True, message=str(e),
                               receipt_exit=json.loads((SCR / "origin_decoy.json").read_text())["exit"])
result["decoy_cwd_scenario_child"] = child(dict(os.environ), decoy)
(REV2 / f"q2_probe_{arm}.json").write_text(json.dumps(result, indent=1) + "\n")
print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk in (
    "exit", "refused", "message", "engine_is_arm_pin", "package_in_arm", "stderr_tail")}) for k, v in result.items()},
    indent=1))
