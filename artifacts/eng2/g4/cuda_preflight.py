#!/usr/bin/env python3
"""ENG2 G4 precondition (adapted from ENG1-A3's a3/cuda_preflight.py; change: no e6_record import, ENG2 paths): import
every module the pre-governor path imports (start_gpu_run.py's and run_gpu_line.py's), run the seal check and every
arm check start_gpu_run.py runs, then list any CUDA/engine library mapped into this process (/proc/self/maps) and any
tensor_cuda module in sys.modules. A mapping check, not a device check. Writes only its stdout.
Code reading (the other half): start_gpu_run.py hashes files and writes START.json; evidence.py imports
device_evidence (ctypes is imported, no library is loaded: CDLL runs only in capture(), called by run_gpu_tests.py)
and arm_origin (subprocess origin check, run only by run_gpu_tests.py); run_gpu_line.py up to its subprocess.run of
the runner only reads receipts, checks dependencies and disk space, and makes the attempt directory.
Prior art: Linux procfs /proc/<pid>/maps (proc(5)); Python sys.modules; ENG1-A3's script. No new algorithm.
"""
import json
from pathlib import Path
import sys

TOOL = Path(__file__).resolve().parents[1] / "tool"
sys.path.insert(0, str(TOOL))
import evidence  # noqa: E402
import device_evidence, arm_origin  # noqa: E402,F401  (evidence's own imports)
import importlib.util  # noqa: E402
for name in ("start_gpu_run", "run_gpu_line"):
    if name == "start_gpu_run":
        continue  # executes at import (writes START.json); its imports are evidence's, imported above
    spec = importlib.util.spec_from_file_location(name + "_module", TOOL / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # top level only defines functions; main() is guarded by __name__
seal = evidence.verify_seal()
arms = []
for arm in seal["arms"]:
    evidence.verify_arm(arm, seal)
    arms.append(arm)
maps = Path("/proc/self/maps").read_text().splitlines()
cuda_maps = sorted({line.split()[-1] for line in maps
                    if any(k in line for k in ("libcuda", "libcudart", "libnvidia", "_tensor_cuda", "libcublas"))})
mods = sorted(m for m in sys.modules if "tensor_cuda" in m or m in ("torch", "cupy", "numba"))
out = dict(seal=evidence.SEAL_NAME, order=evidence.ORDER, seal_sha256=evidence.sha(TOOL / evidence.SEAL_NAME),
           arms_verified=arms, n_arms=len(arms), cuda_libraries_mapped=cuda_maps, cuda_or_engine_modules=mods,
           ctypes_loaded="ctypes" in sys.modules)
print(json.dumps(out, indent=1))
assert not cuda_maps and not mods, "CUDA or engine loaded before the governor"
