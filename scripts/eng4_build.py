"""ENG4 build helper: configure + build one arm of the engine out of tree, each step in its own systemd scope
(MemoryMax=12G, OOMPolicy=continue, CPUWeight=50, nice 19), then copy the module beside the build and write a build
receipt with the module sha256 and a sha256 map of the engine sources the arm was built from.

Usage (from the worktree root):
  python3 -B -P scripts/eng4_build.py <arm> <source tree: the directory holding CMakeLists.txt> [--reconfigure]
      [--receipts <dir under the worktree>]   (ENG4-F1: default artifacts/eng4/build; F1 uses artifacts/eng4/f1/build)

- The build directory is tensor_cuda/build-eng4/<arm>; the module is copied to tensor_cuda/build-eng4/<arm>/module/.
- Refuses to start a step with MemAvailable < 12 GB or /mnt/ForgeRealm free < 40 GB (order: Machine rules, L67).
- Receipts: artifacts/eng4/build/build_receipts.json (one row per step: argv, bounds, UTC start/end, exit, timeout,
  log path) and artifacts/eng4/build/BUILD_<arm>.json (module sha256, source map).
- The configure flags are ENG3's (artifacts/eng3/LEDGER.md in the eng3 worktree): Release, CUDA arch 89, production's
  fetched pybind11 source, FETCHCONTENT_FULLY_DISCONNECTED, CUDACXX cuda-12.6.

Prior art: CMake out-of-tree builds; content-hash manifests (reproducible-builds practice); ENG3's eng3_run.py,
eng3_shamap.py and eng3_build_arms.sh (this repo), whose receipt fields this copies. No other prior art known to me.
"""
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYB = "/mnt/ForgeRealm/Project-Tensor/tensor_cuda/build/_deps/pybind11-src"
SO = "_tensor_cuda.cpython-312-x86_64-linux-gnu.so"
SCOPE = ["systemd-run", "--user", "--scope", "-q", "-p", "MemoryMax=12G", "-p", "OOMPolicy=continue",
         "-p", "CPUWeight=50", "--setenv=TMPDIR=/tmp/eng4_tmp", "--setenv=CUDACXX=/usr/local/cuda-12.6/bin/nvcc",
         "nice", "-n", "19"]
RDIR = os.path.join(ROOT, sys.argv[sys.argv.index("--receipts") + 1] if "--receipts" in sys.argv
                    else "artifacts/eng4/build")
RECEIPTS = os.path.join(RDIR, "build_receipts.json")
SRC_ROOTS = ("src", "include", "tensor_cuda")


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def guard(step):
    with open("/proc/meminfo") as f:
        avail = next(int(l.split()[1]) for l in f if l.startswith("MemAvailable"))
    free = shutil.disk_usage("/mnt/ForgeRealm").free
    if avail < 12 * 1024 * 1024:
        raise SystemExit(f"STOP before {step}: MemAvailable {avail} kB < 12 GB")
    if free < 40 * 10 ** 9:
        raise SystemExit(f"STOP before {step}: /mnt/ForgeRealm free {free} B < 40 GB")
    return avail, free


def run(name, argv, bound):
    avail, free = guard(name)
    os.makedirs(os.path.dirname(RECEIPTS), exist_ok=True)
    log_path = os.path.join(RDIR, f"build_receipts_{name}.log")
    start = now()
    with open(log_path, "ab") as log:
        log.write(f"\n=== {name} {start}\n".encode())
        log.flush()
        p = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        try:
            rc, to = p.wait(timeout=bound), False
        except subprocess.TimeoutExpired:
            p.terminate()  # our own child only
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
            rc, to = 124, True
    rows = json.load(open(RECEIPTS)) if os.path.exists(RECEIPTS) else []
    rows.append({"name": name, "command": argv, "bound_seconds": bound, "start": start, "end": now(), "exit": rc,
                 "timeout": to, "log": os.path.relpath(log_path, ROOT), "mem_available_kB_at_start": avail,
                 "disk_free_B_at_start": free, "cwd": ROOT,
                 "env": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "TMPDIR")}})
    with open(RECEIPTS, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"{name}: exit {rc} timeout {to}", flush=True)
    if rc != 0:
        raise SystemExit(rc)


def source_map(tree):
    files = {}
    for root in SRC_ROOTS:
        for d, dirs, names in os.walk(os.path.join(tree, root)):
            dirs[:] = sorted(x for x in dirs if x != "__pycache__")
            for n in sorted(names):
                p = os.path.join(d, n)
                if n.endswith((".so", ".pyc")) or os.path.islink(p):
                    continue
                files["tensor_cuda/" + os.path.relpath(p, tree)] = sha(p)
    combined = hashlib.sha256("".join(f"{k}\t{v}\n" for k, v in sorted(files.items())).encode()).hexdigest()
    return files, combined


def main():
    arm, tree = sys.argv[1], os.path.abspath(sys.argv[2])
    build = os.path.join(ROOT, "tensor_cuda/build-eng4", arm)
    if "--reconfigure" in sys.argv or not os.path.exists(os.path.join(build, "CMakeCache.txt")):
        run(f"configure_{arm}", SCOPE + ["cmake", "-S", tree, "-B", build, "-DCMAKE_BUILD_TYPE=Release",
                                         "-DCMAKE_CUDA_ARCHITECTURES=89",
                                         f"-DFETCHCONTENT_SOURCE_DIR_PYBIND11={PYB}",
                                         "-DFETCHCONTENT_FULLY_DISCONNECTED=ON"], 600)
    run(f"build_{arm}", SCOPE + ["cmake", "--build", build, "-j6"], 3000)
    built = os.path.join(tree, "tensor_cuda", SO)
    mod_dir = os.path.join(build, "module")
    os.makedirs(mod_dir, exist_ok=True)
    dst = os.path.join(mod_dir, SO)
    if os.path.exists(dst):  # never overwrite a receipted module: keep the old one beside it
        os.replace(dst, dst + "." + datetime.datetime.now().strftime("%Y%m%dT%H%M%S"))
    shutil.copy2(built, dst)
    files, combined = source_map(tree)
    rec = {"arm": arm, "time": now(), "source_tree": tree, "built_module": built, "module_copy": dst,
           "module_sha256": sha(dst), "module_bytes": os.path.getsize(dst), "source_files": len(files),
           "source_map_sha256": combined, "sources": files}
    out = os.path.join(RDIR, f"BUILD_{arm}.json")
    if os.path.exists(out):
        os.replace(out, out[:-5] + "_" + datetime.datetime.now().strftime("%Y%m%dT%H%M%S") + ".json")
    with open(out, "w") as f:
        json.dump(rec, f, indent=2)
    print("BUILT", arm, rec["module_sha256"], "sources", len(files), combined, flush=True)


if __name__ == "__main__":
    main()
