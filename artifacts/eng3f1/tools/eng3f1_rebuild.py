"""Rebuild final hook rank guards in the existing four CMake build trees.
Prior art: CMake incremental dependency rebuilds (Kitware, 2000).
"""
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eng3f1_build import run, sources, ROOT, OUT, BUILD, sha

for arm in sys.argv[1:]:
    src = BUILD / ("source_"+arm)
    cmd = run("build_"+arm+"_final", ["cmake", "--build", str(BUILD/arm), "-j4"])
    module=src/"tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so"
    obj={"arm":arm,"module":str(module.relative_to(ROOT)),"module_sha256":sha(module),
         "sources":sources(src),"commands":[cmd],"cmake_sha256":sha(src/"CMakeLists.txt"),
         "initial_build_receipt":"artifacts/eng3f1/build/BUILD_"+arm+"_r1.json"}
    (OUT/"build"/("BUILD_"+arm+".json")).write_text(json.dumps(obj,indent=2))
    print(arm, obj["module_sha256"], flush=True)
