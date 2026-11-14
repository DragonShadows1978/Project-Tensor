#!/usr/bin/env python3
"""ENG1 inventory of every file this seat wrote inside the worktree (sha256), plus scratch roots. Read-only git."""
import hashlib, json, subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
E = ROOT / "artifacts/eng1"
out = {"tracked_edits": {}, "new_tracked_path": {}, "artifacts_eng1": {}, "build_dirs": {}, "scratch_outside_worktree": [
    "/tmp/eng1_tmp", "/tmp/eng1_cpu", "/tmp/eng1_probe", "/tmp/eng1_mut_src", "/tmp/eng1_ubsan_src",
    "/tmp/eng1_audit_prelim", "/tmp/eng1_audit"]}
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
for f in ("tensor_cuda/src/kernels.cu", "tensor_cuda/src/ops.cpp", "tensor_cuda/src/conv.cu", "tensor_cuda/tests/test_terrain_render.py"):
    out["tracked_edits"][f] = sha(ROOT / f)
out["new_tracked_path"]["tensor_cuda/tests/test_eng1.py"] = sha(ROOT / "tensor_cuda/tests/test_eng1.py")
out["package_module_installed"] = {"tensor_cuda/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so":
                                   sha(ROOT / "tensor_cuda/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so")}
for p in sorted(E.rglob("*")):
    if p.is_file():
        out["artifacts_eng1"][str(p.relative_to(ROOT))] = sha(p)
for d in ("tensor_cuda/build-eng1", "tensor_cuda/build-eng1-base"):
    files = [p for p in (ROOT / d).rglob("*") if p.is_file()]
    out["build_dirs"][d] = dict(files=len(files), bytes=sum(p.stat().st_size for p in files))
(E / "files_written.json").write_text(json.dumps(out, indent=1) + "\n")
print(len(out["artifacts_eng1"]), "files under artifacts/eng1;", out["build_dirs"])
