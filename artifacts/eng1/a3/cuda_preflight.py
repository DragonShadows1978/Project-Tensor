#!/usr/bin/env python3
"""A3 G1 precondition, mechanical complement to the code reading: import every module the pre-governor path imports
(start_gpu_run.py's and run_gpu_line.py's imports), run the same seal and arm checks start_gpu_run.py runs, then
list any CUDA/engine library mapped into this process (/proc/self/maps) and any tensor_cuda module in sys.modules.
Run on CPU (CUDA_VISIBLE_DEVICES=''); a mapping check, not a device check. Writes nothing under the tool tree.
Prior art: Linux procfs /proc/<pid>/maps (proc(5)); Python sys.modules. No prior art for the combination known to me.
"""
import json
from pathlib import Path
import sys

TOOL = Path(__file__).resolve().parents[1] / 'tool'
sys.path.insert(0, str(TOOL))
import evidence  # noqa: E402
import device_evidence, e6_record, arm_origin  # noqa: E402,F401  (evidence's own imports)
import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location('run_gpu_line_module', TOOL / 'run_gpu_line.py')
rgl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rgl)  # top level only defines functions; main() is guarded by __name__
seal = evidence.verify_seal()
arms = []
for arm in seal['arms']:
    evidence.verify_arm(arm, seal)
    arms.append(arm)
maps = Path('/proc/self/maps').read_text().splitlines()
cuda_maps = sorted({line.split()[-1] for line in maps
                    if any(k in line for k in ('libcuda', 'libcudart', 'libnvidia', '_tensor_cuda', 'libcublas'))})
mods = sorted(m for m in sys.modules if 'tensor_cuda' in m or m in ('torch', 'cupy', 'numba'))
out = dict(seal=evidence.SEAL_NAME, order=evidence.ORDER, seal_sha256=evidence.sha(TOOL / evidence.SEAL_NAME),
           arms_verified=arms, n_arms=len(arms), cuda_libraries_mapped=cuda_maps, cuda_or_engine_modules=mods,
           ctypes_loaded='ctypes' in sys.modules)
print(json.dumps(out, indent=1))
assert not cuda_maps and not mods, 'CUDA or engine loaded before the governor'
