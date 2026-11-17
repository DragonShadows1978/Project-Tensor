"""Foreground CMake build monitor. Prior art: CMake out-of-tree builds (2000+),
/proc/meminfo and time(1); ours: receipt assembly and order stop rails.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path.cwd()
OUT = ROOT / "artifacts/eng3f1"
BUILD = ROOT / "tensor_cuda/build-eng3f1"
SO = "_tensor_cuda.cpython-312-x86_64-linux-gnu.so"

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def sources(src):
    return {str(p.relative_to(src)): sha(p) for d in ("src", "include", "tensor_cuda")
            for p in sorted((src/d).rglob("*")) if p.is_file() and p.suffix not in (".so", ".pyc")}

def rails():
    avail = int(next(l.split()[1] for l in Path("/proc/meminfo").read_text().splitlines() if l.startswith("MemAvailable:")))
    free = shutil.disk_usage(ROOT).free
    if avail < 12*1024**2 or free < 40*1024**3:
        raise RuntimeError(f"STOP resource rail: available={avail} kB disk={free} B")
    return [avail, free]

def run(name, cmd):
    rails()
    log = OUT / "build" / (name+".log")
    log.parent.mkdir(exist_ok=True)
    low = rails()
    with log.open("w") as f:
        p = subprocess.Popen(["timeout", "3500s", "/usr/bin/time", "-v", "nice", "-n", "19"] + cmd,
                             stdout=f, stderr=subprocess.STDOUT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
                             start_new_session=True)
        try:
            while p.poll() is None:
                low = [min(a,b) for a,b in zip(low, rails())]
                try:
                    p.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
        except Exception:
            os.killpg(p.pid, signal.SIGTERM)  # only our own process group
            p.wait()
            raise
    rc = p.returncode
    with (OUT/"LEDGER.md").open("a") as f:
        f.write(f"- Build command {name}: {cmd!r}; exit {rc}; log {log.relative_to(ROOT)}; minima {low}.\n")
    if rc:
        raise RuntimeError(f"build command {name} failed: {rc}; see {log}")
    return {"argv": cmd, "exit": rc, "log": str(log.relative_to(ROOT)), "min_mem_kB_disk_bytes": low}

def main():
    arm = sys.argv[1]
    src = BUILD / ("source_"+arm)
    src.mkdir(parents=True, exist_ok=False)
    # Only engine source/config/package: no recursive build trees or tests.
    for d in ("src", "include", "tensor_cuda"):
        shutil.copytree(ROOT/"tensor_cuda"/d, src/d, ignore=shutil.ignore_patterns("*.so", "__pycache__", "*.pyc"))
    shutil.copy2(ROOT/"tensor_cuda/CMakeLists.txt", src/"CMakeLists.txt")
    if arm != "F":
        k = src/"src/kernels.cu"; b = src/"src/bindings.cpp"
        s=k.read_text()
        if arm == "f1m01_bytes_unchecked":
            begin=s.index("  // ENG3-F1 R1: check the byte bound")
            end=s.index("\n}", begin)
            s=s[:begin]+f"  std::fprintf(stderr, \"ENG3F1_EXEC {arm}\\n\");\n  storage = std::make_shared<Storage>(numel_of(shape_) * dtype_size(dtype_), device_);"+s[end:]
        elif arm == "f1m02_malloc_unchecked":
            s=s.replace("    if (ptr == nullptr) throw std::bad_alloc();  // pybind11 -> MemoryError",
                        f"    if (ptr == nullptr) std::fprintf(stderr, \"ENG3F1_EXEC {arm}\\n\");")
        k.write_text(s)
        if arm == "f1m03_device_lenient":
            s=b.read_text();begin=s.index("    const auto refuse = [&]()", s.index("const auto core_device"));end=s.index("\n  };",begin)
            s=s[:begin]+f"    std::fprintf(stderr, \"ENG3F1_EXEC {arm}\\n\");\n    return Device::from_string(text);"+s[end:];b.write_text(s)
    out = BUILD/arm
    receipts=[]
    if arm == "F":
        receipts.append(run("configure_"+arm, ["cmake", "-S", str(src), "-B", str(out), "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CUDA_ARCHITECTURES=89", "-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc", "-DFETCHCONTENT_SOURCE_DIR_PYBIND11=/mnt/ForgeRealm/Project-Tensor/tensor_cuda/build/_deps/pybind11-src", "-DFETCHCONTENT_FULLY_DISCONNECTED=ON"]))
    else:
        # Reuse the baseline objects, changing only copied-source path strings
        # in generated CMake files. Recompile changed TUs only, then relink.
        shutil.copytree(BUILD/"F", out)
        oldsrc=str(BUILD/"source_F");oldout=str(BUILD/"F")
        for p in out.rglob("*"):
            if p.is_file() and p.suffix not in (".o", ".a", ".so", ".bin"):
                raw=p.read_bytes()
                if b"\0" not in raw and (oldsrc.encode() in raw or oldout.encode() in raw):
                    p.write_bytes(raw.replace(oldsrc.encode(),str(src).encode()).replace(oldout.encode(),str(out).encode()))
        # Keep unchanged source mtimes identical to F; changed files stay new.
        for p in src.rglob("*"):
            orig=BUILD/"source_F"/p.relative_to(src)
            if p.is_file() and orig.is_file() and p.read_bytes()==orig.read_bytes():
                shutil.copystat(orig,p)
    receipts.append(run("build_"+arm, ["cmake", "--build", str(out), "-j4"]))
    module=src/"tensor_cuda"/SO
    obj={"arm":arm,"module":str(module.relative_to(ROOT)),"module_sha256":sha(module),"sources":sources(src),"commands":receipts,"cmake_sha256":sha(src/"CMakeLists.txt")}
    (OUT/"build"/("BUILD_"+arm+".json")).write_text(json.dumps(obj,indent=2))
    print(json.dumps({k:v for k,v in obj.items() if k not in ("sources","commands")}),flush=True)

if __name__ == "__main__":
    main()
